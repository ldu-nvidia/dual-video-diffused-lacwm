#!/usr/bin/env python3
"""Register and evaluate the matched video residual-anchor quick screen.

The sampler receives only five observed RGB frames, actions, morphology, and
explicit Gaussian noise.  All clean targets are evaluator-owned and are
constructed only after every endpoint for the current batch has completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import vpm_phaselock_probe as phase  # noqa: E402


SCHEMA_VERSION = 1
KIND_REGISTRATION = "video_residual_anchor_registration"
KIND_ROW = "video_residual_anchor_validation_clip"
KIND_RANK = "video_residual_anchor_validation_rank"
KIND_INVENTORY = "video_residual_anchor_validation_inventory"
TRAINING_COMMIT = phase.TRAINING_COMMIT
EXPECTED_WORLD_SIZE = 8
EXPECTED_BATCH_SIZE_PER_RANK = 2
EXPECTED_VALIDATION_CLIPS = 64
VALIDATION_SAMPLE_ID_OFFSET = 4_000_000
NFE_GRID = (1, 2, 4)
PARENT_SNAPSHOT_SHA256 = (
    "f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21"
)
TRAIN_MANIFEST_SHA256 = (
    "eeace7b0c9f5b6598f32e802d6d3678b6deccccf5b62c6cea2bb3d5478ab8b74"
)
TRAIN_METADATA_SHA256 = (
    "fa22a213f352ffb8cc0b4dc0d35138b35aac349c03f362c597c621fa3473da43"
)
TRAIN_RGB_SHA256 = (
    "b5bdde4461c75bc88653c38b737021fcbd69b0b22f4c87bc8e8097c3494b64ee"
)
TRAIN_ACTIONS_SHA256 = (
    "f2cde809c1d864d4a00422aca8fcac0116229a0b0ac83a93850d1421d16c5b89"
)
PROTOCOL_PATH = REPO_ROOT / "docs" / "experiments" / "video_residual_anchor_protocol.md"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class VideoResidualAnchorEvaluationError(RuntimeError):
    """A frozen input, causal boundary, or matched-pair invariant failed."""


@dataclass(frozen=True)
class Arm:
    code: str
    config_name: str
    run_name: str
    representation_mode: str


ARMS = (
    Arm(
        "VPM-ABS",
        "ravenhuang/wan-dit/video_residual_anchor_absolute",
        "video-residual-anchor-vpm-abs-seed1234-u000200",
        "absolute",
    ),
    Arm(
        "VPM-RESIDUAL",
        "ravenhuang/wan-dit/video_residual_anchor_residual",
        "video-residual-anchor-vpm-residual-seed1234-u000200",
        "cumulative_residual",
    ),
)
ARM_BY_CODE = {arm.code: arm for arm in ARMS}


@dataclass(frozen=True)
class Endpoint:
    code: str
    nfe: int
    action_source: str
    primary_gate: bool


ENDPOINTS = tuple(
    Endpoint(f"autonomous_nfe_{nfe}", nfe, "matched", True)
    for nfe in NFE_GRID
) + (Endpoint("actions_shuffled_nfe_1", 1, "shuffled", False),)
ENDPOINT_BY_CODE = {endpoint.code: endpoint for endpoint in ENDPOINTS}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {**unsigned, "identity_sha256": _sha256_json(unsigned)}


def identity_valid(payload: Mapping[str, Any]) -> bool:
    identity = payload.get("identity_sha256")
    if not isinstance(identity, str) or SHA256_RE.fullmatch(identity) is None:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return _sha256_json(unsigned) == identity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, rehash: bool = True) -> dict[str, Any]:
    path = path.expanduser()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise VideoResidualAnchorEvaluationError(
            f"input must be a regular absolute file: {path}"
        )
    path = path.resolve(strict=True)
    record = {"path": str(path), "bytes": path.stat().st_size}
    if rehash:
        record["sha256"] = _sha256(path)
    return record


def _distributed_file_record(path: Path) -> dict[str, Any]:
    """Hash one shared large file on rank zero and broadcast its receipt."""

    try:
        import torch.distributed as dist
    except ImportError:
        return _file_record(path)
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return _file_record(path)
    payload: list[Any] = [None, None]
    if dist.get_rank() == 0:
        try:
            payload[0] = _file_record(path)
        except BaseException as exc:
            payload[1] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(payload, src=0)
    record, error = payload
    if error is not None or not isinstance(record, dict):
        raise VideoResidualAnchorEvaluationError(
            f"unable to bind shared snapshot: {error or 'invalid receipt'}"
        )
    return record


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VideoResidualAnchorEvaluationError(
            f"{label} is invalid JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise VideoResidualAnchorEvaluationError(f"{label} must contain one object")
    return value


def _exclusive_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise VideoResidualAnchorEvaluationError(
            f"refusing to overwrite output: {path}"
        ) from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    _exclusive_bytes(path, _canonical_json(payload) + b"\n")


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise VideoResidualAnchorEvaluationError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _clean_source(repo: Path, expected_commit: str, label: str) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise VideoResidualAnchorEvaluationError(f"{label} commit must be full SHA")
    if _git(repo, "rev-parse", "--show-toplevel") != str(repo):
        raise VideoResidualAnchorEvaluationError(f"{label} is not a worktree root")
    actual = _git(repo, "rev-parse", "HEAD")
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if actual != expected_commit or dirty:
        raise VideoResidualAnchorEvaluationError(f"{label} source is not clean/pinned")
    return {
        "path": str(repo),
        "git_commit": actual,
        "git_tree_sha": _git(repo, "rev-parse", "HEAD^{tree}"),
        "clean": True,
    }


def _fresh_lustre_root(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute() or not path.name or path.name in {".", ".."}:
        raise VideoResidualAnchorEvaluationError(
            "output root must be an absolute named path"
        )
    parent = path.parent.resolve(strict=True)
    canonical = parent / path.name
    if canonical != path or Path("/lustre") not in canonical.parents:
        raise VideoResidualAnchorEvaluationError(
            "output root must be canonical under /lustre"
        )
    if canonical.exists() or canonical.is_symlink():
        raise VideoResidualAnchorEvaluationError("output root must be fresh")
    return canonical


def _resolve_array(metadata_path: Path, metadata: Mapping[str, Any], key: str) -> Path:
    value = metadata.get(f"{key}_file")
    if not isinstance(value, str) or not value:
        raise VideoResidualAnchorEvaluationError(f"metadata lacks {key}_file")
    path = Path(value)
    if not path.is_absolute():
        path = metadata_path.parent / path
    return Path(_file_record(path, rehash=False)["path"])


def _manifest_descriptors(
    manifest: Path,
    *,
    expected_split: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Parse and validate the immutable clip identity/temporal geometry."""

    descriptors: list[dict[str, Any]] = []
    try:
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise TypeError("manifest row is not an object")
                    descriptors.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise VideoResidualAnchorEvaluationError(
            f"{expected_split} manifest is invalid"
        ) from exc
    if len(descriptors) != expected_count:
        raise VideoResidualAnchorEvaluationError(
            f"{expected_split} manifest count differs"
        )
    for index, row in enumerate(descriptors):
        clip_id = row.get("clip_id")
        episode_dir = row.get("episode_dir")
        start = row.get("start")
        frame_indices = row.get("frame_indices")
        if (
            row.get("split") != expected_split
            or row.get("auxiliary_index") != index
            or not isinstance(clip_id, str)
            or SHA256_RE.fullmatch(clip_id) is None
            or not isinstance(episode_dir, str)
            or not Path(episode_dir).is_absolute()
            or isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or row.get("sample_size") != 13
            or row.get("chunk_size") != 5
            or row.get("action_span") != 65
            or frame_indices != [start + 5 * offset for offset in range(13)]
        ):
            raise VideoResidualAnchorEvaluationError(
                f"{expected_split} manifest row {index} differs"
            )
    if (
        len({row["clip_id"] for row in descriptors}) != expected_count
        or len({row["episode_dir"] for row in descriptors}) != expected_count
    ):
        raise VideoResidualAnchorEvaluationError(
            f"{expected_split} manifest clips/episodes are not unique"
        )
    return descriptors


def _split_disjointness(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return canonical evidence and fail if train and validation overlap."""

    train_clip_ids = sorted(str(row["clip_id"]) for row in train)
    validation_clip_ids = sorted(str(row["clip_id"]) for row in validation)
    train_episodes = sorted(str(row["episode_dir"]) for row in train)
    validation_episodes = sorted(str(row["episode_dir"]) for row in validation)
    clip_overlap = sorted(set(train_clip_ids) & set(validation_clip_ids))
    episode_overlap = sorted(set(train_episodes) & set(validation_episodes))
    if clip_overlap or episode_overlap:
        raise VideoResidualAnchorEvaluationError(
            "training and validation manifests overlap by clip or episode"
        )
    return {
        "train_clip_ids_sha256": _sha256_json(train_clip_ids),
        "validation_clip_ids_sha256": _sha256_json(validation_clip_ids),
        "train_episode_dirs_sha256": _sha256_json(train_episodes),
        "validation_episode_dirs_sha256": _sha256_json(validation_episodes),
        "clip_id_overlap_count": 0,
        "episode_dir_overlap_count": 0,
        "episode_disjoint": True,
    }


def _validate_train_inputs(manifest: Path, metadata_path: Path) -> dict[str, Any]:
    manifest_record = _file_record(manifest)
    metadata_record = _file_record(metadata_path)
    if manifest_record["sha256"] != TRAIN_MANIFEST_SHA256:
        raise VideoResidualAnchorEvaluationError("train manifest changed")
    if metadata_record["sha256"] != TRAIN_METADATA_SHA256:
        raise VideoResidualAnchorEvaluationError("train metadata changed")
    metadata = _read_json(metadata_path, "train metadata")
    if (
        metadata.get("complete") is not True
        or metadata.get("split") != "train"
        or metadata.get("clip_count") != 512
        or metadata.get("clip_manifest_sha256") != TRAIN_MANIFEST_SHA256
        or metadata.get("rgb_sha256") != TRAIN_RGB_SHA256
        or metadata.get("actions_sha256") != TRAIN_ACTIONS_SHA256
        or metadata.get("rgb_shape") != [512, 13, 3, 180, 960]
        or metadata.get("actions_shape") != [512, 13, 5, 23]
    ):
        raise VideoResidualAnchorEvaluationError("train RGB/action metadata differs")
    rgb = _resolve_array(metadata_path, metadata, "rgb")
    actions = _resolve_array(metadata_path, metadata, "actions")
    records = {"rgb": _file_record(rgb), "actions": _file_record(actions)}
    if (
        records["rgb"]["sha256"] != TRAIN_RGB_SHA256
        or records["actions"]["sha256"] != TRAIN_ACTIONS_SHA256
    ):
        raise VideoResidualAnchorEvaluationError("train arrays changed")
    descriptors = _manifest_descriptors(
        Path(manifest_record["path"]), expected_split="train", expected_count=512
    )
    return {
        "manifest": manifest_record,
        "cache_metadata": metadata_record,
        **records,
        "clip_count": 512,
        "split": "train",
        "auxiliary_target_array_opened": False,
        "descriptors": descriptors,
    }


def _arm_run_identity(output_root: Path, tool_commit: str, arm: Arm) -> str:
    return _sha256_json(
        {
            "schema": "video-residual-anchor-run-identity-v1",
            "output_root": str(output_root),
            "tool_commit": tool_commit,
            "arm": asdict(arm),
            "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
            "updates": 200,
            "seed": 1234,
        }
    )


def command_register(args: argparse.Namespace) -> int:
    tool_repo = args.tool_repo.expanduser().resolve(strict=True)
    historical_repo = args.historical_repo.expanduser().resolve(strict=True)
    tool = _clean_source(tool_repo, args.expected_commit, "tool")
    historical = _clean_source(historical_repo, TRAINING_COMMIT, "historical model")
    if tool_repo != REPO_ROOT or tool_repo == historical_repo:
        raise VideoResidualAnchorEvaluationError("tool/model repository routing differs")
    # The implementation may add the residual coordinate, but it must retain
    # the exact historical training code in its ancestry rather than merely
    # presenting a compatible-looking snapshot schema.
    _git(
        tool_repo,
        "merge-base",
        "--is-ancestor",
        TRAINING_COMMIT,
        args.expected_commit,
    )
    if not PROTOCOL_PATH.is_file() or PROTOCOL_PATH.is_symlink():
        raise VideoResidualAnchorEvaluationError("prospective protocol is absent")
    output_root = _fresh_lustre_root(args.output_root)
    validated = phase._validate_study_metadata(
        args.study_root.expanduser().resolve(strict=True),
        historical_repo,
        rehash_snapshot=True,
    )
    if validated["snapshot_sha256"] != PARENT_SNAPSHOT_SHA256:
        raise VideoResidualAnchorEvaluationError("parent VPM snapshot digest changed")
    train = _validate_train_inputs(args.train_manifest, args.train_cache_metadata)
    train_descriptors = train.pop("descriptors")
    validation_descriptors = _manifest_descriptors(
        Path(validated["validation"]["manifest"]["path"]),
        expected_split="val",
        expected_count=EXPECTED_VALIDATION_CLIPS,
    )
    if validation_descriptors != validated["descriptors"]:
        raise VideoResidualAnchorEvaluationError(
            "validation descriptors differ from the frozen study"
        )
    disjointness = _split_disjointness(
        train_descriptors, validation_descriptors
    )
    # Preserve the venv launcher path. Resolving its symlink can bypass the
    # venv's pyvenv.cfg and silently select the system site-packages.
    python = args.python.expanduser()
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise VideoResidualAnchorEvaluationError("runtime Python is not executable")
    protocol_files = {}
    for relative in (
        "docs/experiments/video_residual_anchor_protocol.md",
        "tools/video_residual_anchor_evaluate.py",
        "tools/video_residual_anchor_analyze.py",
        "tools/video_residual_anchor_workflow.py",
        "tools/slurm/video_residual_anchor_screen.sbatch",
    ):
        protocol_files[relative] = _file_record(tool_repo / relative)
    arm_run_identities = {
        arm.code: _arm_run_identity(output_root, args.expected_commit, arm)
        for arm in ARMS
    }
    payload = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_REGISTRATION,
            "created_at_utc": _now(),
            "status": "registered_before_candidate_training",
            "output_root": str(output_root),
            "tool_repository": tool,
            "historical_model_repository": historical,
            "protocol_files": protocol_files,
            "controlled_study": {
                "study_root": str(args.study_root.resolve(strict=True)),
                "study_identity_sha256": validated["study"]["identity_sha256"],
                "arm_identity_sha256": validated["arm"]["identity_sha256"],
                "stage_identity_sha256": validated["stage"]["identity_sha256"],
                "parent_snapshot": _file_record(validated["paths"]["snapshot"]),
                "parent_completed_updates": 1000,
                "training_git_commit": TRAINING_COMMIT,
            },
            "training": train,
            "validation": validated["validation"],
            "validation_descriptors": validation_descriptors,
            "train_validation_disjointness": disjointness,
            "runtime": {**validated["runtime"], "python": str(python)},
            "arm_run_identity_sha256": arm_run_identities,
            "fixed_protocol": {
                "arms": [asdict(arm) for arm in ARMS],
                "continuation_updates": 200,
                "seed": 1234,
                "optimizer_state_policy": "fresh_identical_adamw",
                "ema_policy": "none_in_parent_and_none_in_both_arms",
                "normalization": "none",
                "history_policy": "exact_observed_history_tokens",
                "nfe_grid": list(NFE_GRID),
                "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
                "world_size": EXPECTED_WORLD_SIZE,
                "batch_size_per_rank": EXPECTED_BATCH_SIZE_PER_RANK,
                "validation_clips": EXPECTED_VALIDATION_CLIPS,
                "same_explicit_noise_across_arms_and_endpoints": True,
                "protected_test_access_allowed": False,
                "clean_future_or_teacher_allowed_in_sampler": False,
                "auxiliary_target_array_access_allowed": False,
                "claim_scope": "adjacent_structural_baseline_not_dual_diffusion",
            },
            "wandb": {
                "entity": "zijiandu",
                "project": "dual-video-diffusion-private",
                "group": None,
                "mode": "online",
            },
            "protected_test_accessed": False,
            "auxiliary_target_array_opened": False,
        }
    )
    output_root.mkdir(mode=0o700)
    _exclusive_json(output_root / "registration.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _validate_registration(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    registration = _read_json(path, "registration")
    fixed_protocol = registration.get("fixed_protocol", {})
    if (
        not identity_valid(registration)
        or registration.get("kind") != KIND_REGISTRATION
        or registration.get("status") != "registered_before_candidate_training"
        or fixed_protocol.get("arms") != [asdict(arm) for arm in ARMS]
        or fixed_protocol.get("continuation_updates") != 200
        or fixed_protocol.get("seed") != 1234
        or fixed_protocol.get("optimizer_state_policy")
        != "fresh_identical_adamw"
        or fixed_protocol.get("ema_policy")
        != "none_in_parent_and_none_in_both_arms"
        or fixed_protocol.get("normalization") != "none"
        or fixed_protocol.get("history_policy") != "exact_observed_history_tokens"
        or fixed_protocol.get("nfe_grid") != list(NFE_GRID)
        or fixed_protocol.get("endpoints") != [asdict(endpoint) for endpoint in ENDPOINTS]
        or fixed_protocol.get("world_size") != EXPECTED_WORLD_SIZE
        or fixed_protocol.get("batch_size_per_rank")
        != EXPECTED_BATCH_SIZE_PER_RANK
        or fixed_protocol.get("validation_clips") != EXPECTED_VALIDATION_CLIPS
        or fixed_protocol.get("same_explicit_noise_across_arms_and_endpoints")
        is not True
        or fixed_protocol.get("protected_test_access_allowed") is not False
        or fixed_protocol.get(
            "clean_future_or_teacher_allowed_in_sampler"
        )
        is not False
        or fixed_protocol.get("auxiliary_target_array_access_allowed") is not False
        or fixed_protocol.get("claim_scope")
        != "adjacent_structural_baseline_not_dual_diffusion"
        or registration.get("wandb")
        != {
            "entity": "zijiandu",
            "project": "dual-video-diffusion-private",
            "group": None,
            "mode": "online",
        }
        or registration.get("protected_test_accessed") is not False
        or registration.get("auxiliary_target_array_opened") is not False
    ):
        raise VideoResidualAnchorEvaluationError("registration identity/protocol differs")
    canonical = Path(registration["output_root"]) / "registration.json"
    if path != canonical.resolve(strict=True):
        raise VideoResidualAnchorEvaluationError("registration path is not canonical")
    tool = registration["tool_repository"]
    _clean_source(Path(tool["path"]), str(tool["git_commit"]), "tool")
    train_manifest = Path(registration["training"]["manifest"]["path"])
    validation_manifest = Path(registration["validation"]["manifest"]["path"])
    for manifest, record, split, count in (
        (train_manifest, registration["training"]["manifest"], "train", 512),
        (
            validation_manifest,
            registration["validation"]["manifest"],
            "val",
            EXPECTED_VALIDATION_CLIPS,
        ),
    ):
        observed = _file_record(manifest)
        if observed != record:
            raise VideoResidualAnchorEvaluationError(
                f"registered {split} manifest changed"
            )
    train_descriptors = _manifest_descriptors(
        train_manifest, expected_split="train", expected_count=512
    )
    validation_descriptors = _manifest_descriptors(
        validation_manifest,
        expected_split="val",
        expected_count=EXPECTED_VALIDATION_CLIPS,
    )
    if (
        validation_descriptors != registration.get("validation_descriptors")
        or _split_disjointness(train_descriptors, validation_descriptors)
        != registration.get("train_validation_disjointness")
    ):
        raise VideoResidualAnchorEvaluationError(
            "registered split identities/disjointness changed"
        )
    return registration


def _validate_training_trace(
    run_dir: Path, arm: Arm, registration: Mapping[str, Any]
) -> dict[str, Any]:
    trace = run_dir / "video_residual_anchor_training_trace.jsonl"
    complete_path = run_dir / "video_residual_anchor_training_trace_complete.json"
    training_complete_path = run_dir / "training_complete.json"
    trace_record = _file_record(trace)
    complete = _read_json(complete_path, "training trace completion")
    training_complete = _read_json(training_complete_path, "training completion")
    expected_run_identity = registration["arm_run_identity_sha256"][arm.code]
    expected_snapshot = (run_dir / "snapshot.pt").resolve(strict=True)
    if (
        complete.get("kind") != "video_residual_anchor_training_trace_complete"
        or complete.get("arm") != arm.code
        or complete.get("completed_updates") != 200
        or complete.get("trace_sha256") != trace_record["sha256"]
        or complete.get("protected_test_accessed") is not False
        or training_complete.get("schema_version") != 1
        or training_complete.get("status") != "completed"
        or training_complete.get("completed_updates") != 200
        or training_complete.get("max_iter") != 200
        or training_complete.get("run_identity_sha256") != expected_run_identity
        or Path(str(training_complete.get("snapshot", ""))).resolve(strict=True)
        != expected_snapshot
    ):
        raise VideoResidualAnchorEvaluationError("training completion receipt differs")
    try:
        with trace.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VideoResidualAnchorEvaluationError("training trace is invalid") from exc
    if not rows or not isinstance(rows[0], dict):
        raise VideoResidualAnchorEvaluationError("training trace is empty")
    header = rows[0]
    if (
        header.get("kind") != "video_residual_anchor_training_trace_header"
        or header.get("arm") != arm.code
        or header.get("representation_mode") != arm.representation_mode
        or header.get("normalization") != "none"
        or header.get("parent_snapshot_sha256") != PARENT_SNAPSHOT_SHA256
        or header.get("parent_completed_updates") != 1000
        or header.get("continuation_updates") != 200
        or header.get("optimizer_state_policy") != "fresh_identical_adamw"
        or header.get("ema_policy") != "none_in_parent_and_none_in_both_arms"
        or header.get("auxiliary_feature_access") is not False
        or Path(str(header.get("parent_snapshot", ""))).resolve(strict=True)
        != Path(
            registration["controlled_study"]["parent_snapshot"]["path"]
        ).resolve(strict=True)
        or complete.get("rows") != len(rows)
    ):
        raise VideoResidualAnchorEvaluationError("training trace header differs")
    train_events: dict[int, Mapping[str, Any]] = {}
    validation_iterations: list[int] = []
    audit_keys = (
        "train_loss/paired_audit/clip_index_mean",
        "train_loss/paired_audit/clip_index_square_mean",
        "train_loss/paired_audit/timestep_mean",
        "train_loss/paired_audit/timestep_square_mean",
        "train_loss/paired_audit/noise_probe",
    )
    for row in rows[1:]:
        metrics = row.get("metrics") if isinstance(row, Mapping) else None
        if (
            row.get("kind") != "video_residual_anchor_training_trace_event"
            or row.get("arm") != arm.code
            or not isinstance(metrics, Mapping)
        ):
            raise VideoResidualAnchorEvaluationError("training trace event differs")
        iteration = metrics.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise VideoResidualAnchorEvaluationError("training trace iteration differs")
        if row.get("total_observations") != metrics.get("total_observations"):
            raise VideoResidualAnchorEvaluationError(
                "training trace observation count differs"
            )
        if "train_loss/loss" in metrics:
            if iteration in train_events or any(
                key not in metrics
                or isinstance(metrics[key], bool)
                or not isinstance(metrics[key], (int, float))
                or not math.isfinite(float(metrics[key]))
                for key in audit_keys
            ):
                raise VideoResidualAnchorEvaluationError(
                    "training trace paired-audit event differs"
                )
            if metrics.get("total_observations") != (iteration + 1) * 8:
                raise VideoResidualAnchorEvaluationError(
                    "training trace global batch accounting differs"
                )
            train_events[iteration] = row
        elif any(str(key).startswith("val_loss/") for key in metrics):
            validation_iterations.append(iteration)
        else:
            raise VideoResidualAnchorEvaluationError(
                "training trace contains an unknown event"
            )
    if set(train_events) != set(range(200)) or validation_iterations != [0, 100, 199]:
        raise VideoResidualAnchorEvaluationError(
            "training trace update/validation inventory differs"
        )
    return {
        "trace": trace_record,
        "completion": _file_record(complete_path),
        "training_completion": _file_record(training_complete_path),
        "header": header,
        "train_update_events": 200,
        "validation_events": validation_iterations,
        "registration_identity_sha256": registration["identity_sha256"],
    }


def _load_model(
    registration: Mapping[str, Any], arm: Arm, run_dir: Path, device: Any
) -> tuple[Any, dict[str, Any]]:
    import torch

    tool_repo = Path(registration["tool_repository"]["path"])
    project_root = tool_repo / "projects" / "latent_action_models"
    shim = tool_repo / "tools" / "env" / "videox_shim"
    videox = Path(registration["runtime"]["videox_home"])
    for root in reversed((str(tool_repo), str(project_root), str(shim), str(videox))):
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    os.environ["WAN_DIR"] = registration["runtime"]["wan_dir"]
    os.environ["VIDEOX_HOME"] = registration["runtime"]["videox_home"]
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_path = run_dir / ".hydra" / "config.yaml"
    config_record = _file_record(config_path)
    config = OmegaConf.load(config_path)
    if (
        OmegaConf.to_container(config.model, resolve=True)
        != OmegaConf.to_container(config.trainer.model, resolve=True)
        or OmegaConf.to_container(config.optimizer_factory, resolve=True)
        != OmegaConf.to_container(config.trainer.optimizer_factory, resolve=True)
        or OmegaConf.to_container(config.lr_scheduler_factory, resolve=True)
        != OmegaConf.to_container(
            config.trainer.lr_scheduler_factory, resolve=True
        )
        or OmegaConf.to_container(config.dataset, resolve=True)
        != OmegaConf.to_container(config.trainer.data_loader.dataset, resolve=True)
        or OmegaConf.to_container(config.val_dataset, resolve=True)
        != OmegaConf.to_container(
            config.trainer.val_data_loader[0].dataset, resolve=True
        )
    ):
        raise VideoResidualAnchorEvaluationError(
            "resolved top-level and instantiated trainer inputs differ"
        )
    if (
        str(config.name) != arm.run_name
        or not str(config.model.get("_target_", "")).endswith(
            ".VideoResidualAnchorVPM"
        )
        or str(config.model.video_residual_anchor.mode) != arm.representation_mode
        or str(config.model.video_residual_anchor.normalization) != "none"
        or config.wandb.entity != "zijiandu"
        or config.wandb.project != "dual-video-diffusion-private"
        or config.wandb.group is not None
        or str(config.wandb.mode) != "online"
        or float(config.wandb.settings.finish_timeout) != 120.0
        or bool(config.wandb.settings.finish_timeout_raises)
        or int(config.seed) != 1234
        or int(config.data_loader.batch_size) != 1
        or not bool(config.dataset.infinite)
        or int(config.val_data_loader[0].batch_size) != 2
        or not bool(config.val_dataset.infinite)
        or int(config.trainer.config.max_iter) != 200
        or int(config.trainer.config.gradient_accumulation_steps) != 1
        or int(config.trainer.config.validation.val_every) != 100
        or int(config.trainer.config.validation.n_val_samples) != 8
        or bool(config.trainer.config.validation.save_best)
        or list(config.trainer.config.exclude_keys) != []
        or config.trainer.config.transition_handoff_path is not None
        or Path(str(config.trainer.config.load_path)).resolve(strict=True)
        != Path(
            registration["controlled_study"]["parent_snapshot"]["path"]
        ).resolve(strict=True)
        or float(config.optimizer_factory.lr) != 1.0e-4
        or tuple(float(value) for value in config.optimizer_factory.betas)
        != (0.9, 0.95)
        or int(config.lr_scheduler_factory.lr_lambda.warmup_steps) != 20
        or int(config.lr_scheduler_factory.lr_lambda.total_steps) != 200
        or float(config.lr_scheduler_factory.lr_lambda.final_learning_rate) != 1.0e-6
    ):
        raise VideoResidualAnchorEvaluationError("resolved arm configuration differs")
    model = instantiate(config.model)
    snapshot_path = run_dir / "snapshot.pt"
    snapshot_record = _distributed_file_record(snapshot_path)
    snapshot = torch.load(
        snapshot_path, map_location="cpu", weights_only=True, mmap=True
    )
    expected_run_identity = registration["arm_run_identity_sha256"][arm.code]
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("world_size") != EXPECTED_WORLD_SIZE
        or snapshot.get("gradient_accumulation_steps") != 1
        or snapshot.get("_start_iter") != 200
        or snapshot.get("run_identity_sha256") != expected_run_identity
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        or not isinstance(snapshot.get("model"), Mapping)
    ):
        raise VideoResidualAnchorEvaluationError("trained arm snapshot differs")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise VideoResidualAnchorEvaluationError("trained arm strict load failed")
    del snapshot
    model = model.to(device=device).eval()
    model._ensure_video_only_runtime_contract()
    if (
        model.video_representation_mode != arm.representation_mode
        or model.tf_condition_mode != "off"
        or model.condition_on_tf
        or bool(model.condition_on_tf_clock)
        or model.tf_loss_weight != 0.0
        or not model.parameter_matched_control
        or tuple(model.evaluation_nfe_steps) != NFE_GRID
        or tuple(model.evaluation_condition_sources) != ("autonomous",)
        or getattr(model, "time_frequency_transform", None) is not None
        or model.auxiliary_history_mode != "diffuse_all"
        or model.tf_schedule_mode != "aligned"
        or model.tf_lead_logit != 0.0
        or not bool(model.forward_model.tf_head_condition_on_clock)
    ):
        raise VideoResidualAnchorEvaluationError("trained model is not deployable VPM")
    suspicious = []
    for name, module in model.named_modules():
        identity = f"{name}:{type(module).__module__}.{type(module).__name__}".lower()
        if any(term in identity for term in ("vjepa", "teacher", "dino")):
            suspicious.append(identity)
    if suspicious:
        raise VideoResidualAnchorEvaluationError("online feature/teacher module is present")
    return model, {
        "snapshot": snapshot_record,
        "resolved_config": config_record,
        "training_trace": _validate_training_trace(run_dir, arm, registration),
    }


def _dataset(registration: Mapping[str, Any]) -> Any:
    validation = registration["validation"]
    return phase._RegisteredValidationInputs(
        rgb_path=Path(validation["arrays"]["rgb"]["path"]),
        actions_path=Path(validation["arrays"]["actions"]["path"]),
        descriptors=registration["validation_descriptors"],
        padding_dim=157,
    )


def _sampling_noises(model: Any, history_rgb: Any, sampling_ids: Any) -> tuple[Any, Any]:
    import torch.distributed as dist

    history_latents = model._encode_clip(history_rgb).to(history_rgb.dtype)
    latent_tokens = int(
        model.rgb_tokenizer.latent_temporal_len(
            model.num_history_frames + model.num_future_frames
        )
    )
    video_shape = (
        int(history_rgb.shape[0]),
        int(history_latents.shape[1]),
        latent_tokens,
        int(history_latents.shape[3]),
        int(history_latents.shape[4]),
    )
    rank = dist.get_rank()
    video_noise = model._evaluation_noise(
        video_shape,
        device=history_rgb.device,
        dtype=history_rgb.dtype,
        base_seed=model.evaluation_noise_seed,
        sample_ids=sampling_ids,
        stream=0,
        rank=rank,
    )
    auxiliary_shape = (
        video_shape[0],
        int(model.forward_model.tf_token_adapter.tf_channels),
        *video_shape[2:],
    )
    auxiliary_noise = model._evaluation_noise(
        auxiliary_shape,
        device=history_rgb.device,
        dtype=history_rgb.dtype,
        base_seed=model.evaluation_noise_seed,
        sample_ids=sampling_ids,
        stream=1,
        rank=rank,
    )
    return video_noise, auxiliary_noise


def _uint8(decoded: Any) -> Any:
    import torch

    return (
        ((decoded.float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(dtype=torch.uint8)
        .cpu()
    )


def _rows_for_endpoint(
    *,
    arm: Arm,
    endpoint: Endpoint,
    sample: Any,
    decoded: Any,
    scoring: Mapping[str, Any],
    clip_indexes: Sequence[int],
    clip_ids: Sequence[str],
    sampling_ids: Any,
    action_input: Any,
    action_donor_ids: Sequence[int | None],
    video_noise: Any,
    auxiliary_noise: Any,
    observed_calls: int,
    registration: Mapping[str, Any],
    arm_artifacts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    absolute = sample.absolute_video_latent.detach().cpu().to(torch.float16)
    representation = sample.representation.detach().cpu().to(torch.float16)
    clean = scoring["video_clean"]
    history_tokens = int(sample.history_tokens)
    if (
        observed_calls != endpoint.nfe
        or sample.model_calls != endpoint.nfe
        or absolute.shape != clean.shape
    ):
        raise VideoResidualAnchorEvaluationError("endpoint call/shape contract differs")
    latent_nmse = phase._per_sample_nmse(absolute, clean, history_tokens)
    latent_delta_nmse = phase._per_sample_future_delta_nmse(
        absolute, clean, history_tokens
    )
    decoded_metrics = phase._per_sample_decoded(
        decoded, scoring["ground_truth"], scoring["history_last"]
    )
    hashes = {
        "cached_rgb_input_sha256": scoring["rgb_hashes"],
        "cached_actions_input_sha256": scoring["actions_hashes"],
        "sampler_actions_sha256": phase._slice_hashes(action_input),
        "video_clean_scoring_sha256": phase._slice_hashes(clean),
        "raw_ground_truth_sha256": phase._slice_hashes(scoring["ground_truth"]),
        "raw_history_last_sha256": phase._slice_hashes(scoring["history_last"]),
        "video_initial_noise_sha256": phase._slice_hashes(video_noise),
        "auxiliary_initial_noise_sha256": phase._slice_hashes(auxiliary_noise),
        "generated_representation_sha256": phase._slice_hashes(representation),
        "absolute_video_final_sha256": phase._slice_hashes(absolute),
        "decoded_final_sha256": phase._slice_hashes(decoded),
    }
    sampling_values = [int(value) for value in sampling_ids.detach().cpu().tolist()]
    rows = []
    for offset, (clip_index, clip_id) in enumerate(zip(clip_indexes, clip_ids)):
        rows.append(
            identity_payload(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": KIND_ROW,
                    "registration_identity_sha256": registration["identity_sha256"],
                    "tool_git_commit": registration["tool_repository"]["git_commit"],
                    "training_git_commit": TRAINING_COMMIT,
                    "arm": asdict(arm),
                    "arm_snapshot": arm_artifacts["snapshot"],
                    "evaluation_split": "validation",
                    "protected_test_accessed": False,
                    "clip_index": int(clip_index),
                    "clip_id": str(clip_id),
                    "sampling_id": sampling_values[offset],
                    "endpoint": asdict(endpoint),
                    "action_donor_sampling_id": action_donor_ids[offset],
                    "clean_future_rgb_passed_to_sampler": False,
                    "clean_video_latent_passed_to_sampler": False,
                    "clean_auxiliary_passed_to_sampler": False,
                    "target_cache_array_opened": False,
                    "online_feature_or_teacher_call_count": 0,
                    "scoring_constructed_after_all_sampling": True,
                    "representation_inverted_before_decode_and_metrics": True,
                    "known_history_latents_exact": True,
                    "history_rgb_frames": 5,
                    "future_rgb_frames": 8,
                    "history_video_latent_tokens": history_tokens,
                    "future_video_latent_tokens": int(absolute.shape[2] - history_tokens),
                    "actual_transformer_call_count": observed_calls,
                    "declared_nfe": endpoint.nfe,
                    "metrics": {
                        "video_future_nmse": latent_nmse[offset],
                        "video_future_temporal_delta_nmse": latent_delta_nmse[offset],
                        "decoded_mse_unit_range": decoded_metrics[
                            "decoded_mse_unit_range"
                        ][offset],
                        "decoded_psnr_db": decoded_metrics["decoded_psnr_db"][offset],
                        "decoded_temporal_difference_mse_unit_range": decoded_metrics[
                            "decoded_temporal_difference_mse_unit_range"
                        ][offset],
                    },
                    "tensor_sha256": {
                        key: values[offset] for key, values in hashes.items()
                    },
                }
            )
        )
    return rows


def _expected_rank_indexes(rank: int) -> list[int]:
    return list(range(rank, EXPECTED_VALIDATION_CLIPS, EXPECTED_WORLD_SIZE))


def _validate_rank_receipt_payload(
    receipt: Mapping[str, Any],
    *,
    rank: int,
    arm: Arm,
    rows_record: Mapping[str, Any],
) -> None:
    assigned = _expected_rank_indexes(rank)
    expected_batches = math.ceil(len(assigned) / EXPECTED_BATCH_SIZE_PER_RANK)
    expected_calls = expected_batches * sum(endpoint.nfe for endpoint in ENDPOINTS)
    if (
        not identity_valid(receipt)
        or receipt.get("kind") != KIND_RANK
        or receipt.get("arm") != asdict(arm)
        or receipt.get("rank") != rank
        or receipt.get("world_size") != EXPECTED_WORLD_SIZE
        or receipt.get("indexes") != assigned
        or receipt.get("rows") != len(assigned) * len(ENDPOINTS)
        or receipt.get("rows_file") != rows_record
        or receipt.get("transformer_calls") != expected_calls
        or receipt.get("protected_test_accessed") is not False
        or receipt.get("target_cache_array_opened") is not False
    ):
        raise VideoResidualAnchorEvaluationError(
            f"rank {rank} evaluation receipt differs"
        )


def _validate_rows(
    rows: Sequence[Mapping[str, Any]], arm: Arm, registration: Mapping[str, Any]
) -> None:
    expected = {
        (index, endpoint.code)
        for index in range(EXPECTED_VALIDATION_CLIPS)
        for endpoint in ENDPOINTS
    }
    observed: dict[tuple[int, str], Mapping[str, Any]] = {}
    expected_snapshot_path = str(
        Path(registration["output_root"])
        / "training"
        / arm.run_name
        / "snapshot.pt"
    )
    observed_snapshot_records: set[bytes] = set()
    metric_names = (
        "video_future_nmse",
        "video_future_temporal_delta_nmse",
        "decoded_mse_unit_range",
        "decoded_psnr_db",
        "decoded_temporal_difference_mse_unit_range",
    )
    tensor_hash_names = (
        "cached_rgb_input_sha256",
        "cached_actions_input_sha256",
        "sampler_actions_sha256",
        "video_clean_scoring_sha256",
        "raw_ground_truth_sha256",
        "raw_history_last_sha256",
        "video_initial_noise_sha256",
        "auxiliary_initial_noise_sha256",
        "generated_representation_sha256",
        "absolute_video_final_sha256",
        "decoded_final_sha256",
    )
    for row in rows:
        endpoint_value = row.get("endpoint")
        code = endpoint_value.get("code") if isinstance(endpoint_value, Mapping) else None
        endpoint = ENDPOINT_BY_CODE.get(str(code))
        clip_index_value = row.get("clip_index")
        valid_clip_index = (
            not isinstance(clip_index_value, bool)
            and isinstance(clip_index_value, int)
            and 0 <= clip_index_value < EXPECTED_VALIDATION_CLIPS
        )
        arm_snapshot = row.get("arm_snapshot")
        valid_arm_snapshot = (
            isinstance(arm_snapshot, Mapping)
            and arm_snapshot.get("path") == expected_snapshot_path
            and isinstance(arm_snapshot.get("bytes"), int)
            and not isinstance(arm_snapshot.get("bytes"), bool)
            and arm_snapshot.get("bytes", 0) > 0
            and isinstance(arm_snapshot.get("sha256"), str)
            and SHA256_RE.fullmatch(str(arm_snapshot.get("sha256"))) is not None
        )
        key = (clip_index_value, str(code))
        if (
            not identity_valid(row)
            or row.get("schema_version") != SCHEMA_VERSION
            or row.get("kind") != KIND_ROW
            or row.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or row.get("tool_git_commit")
            != registration["tool_repository"]["git_commit"]
            or row.get("training_git_commit") != TRAINING_COMMIT
            or row.get("arm") != asdict(arm)
            or not valid_arm_snapshot
            or endpoint is None
            or dict(endpoint_value) != asdict(endpoint)
            or row.get("evaluation_split") != "validation"
            or row.get("protected_test_accessed") is not False
            or not valid_clip_index
            or row.get("sampling_id")
            != VALIDATION_SAMPLE_ID_OFFSET + clip_index_value
            or row.get("clip_id")
            != registration["validation_descriptors"][
                clip_index_value
            ].get("clip_id")
            or row.get("clean_future_rgb_passed_to_sampler") is not False
            or row.get("clean_video_latent_passed_to_sampler") is not False
            or row.get("clean_auxiliary_passed_to_sampler") is not False
            or row.get("target_cache_array_opened") is not False
            or row.get("online_feature_or_teacher_call_count") != 0
            or row.get("scoring_constructed_after_all_sampling") is not True
            or row.get("representation_inverted_before_decode_and_metrics") is not True
            or row.get("known_history_latents_exact") is not True
            or row.get("history_rgb_frames") != 5
            or row.get("future_rgb_frames") != 8
            or row.get("actual_transformer_call_count") != endpoint.nfe
            or row.get("declared_nfe") != endpoint.nfe
            or row.get("history_video_latent_tokens") != 2
            or row.get("future_video_latent_tokens") != 2
            or key in observed
        ):
            raise VideoResidualAnchorEvaluationError("validation row violates protocol")
        observed_snapshot_records.add(_canonical_json(dict(arm_snapshot)))
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or any(
            isinstance(metrics.get(name), bool)
            or not isinstance(metrics.get(name), (int, float))
            or not math.isfinite(float(metrics[name]))
            for name in metric_names
        ):
            raise VideoResidualAnchorEvaluationError("validation metrics are invalid")
        tensor_hashes = row.get("tensor_sha256")
        if not isinstance(tensor_hashes, Mapping) or any(
            not isinstance(tensor_hashes.get(name), str)
            or SHA256_RE.fullmatch(str(tensor_hashes[name])) is None
            for name in tensor_hash_names
        ):
            raise VideoResidualAnchorEvaluationError("validation tensor hashes are invalid")
        if endpoint.action_source == "matched":
            if row.get("action_donor_sampling_id") is not None:
                raise VideoResidualAnchorEvaluationError(
                    "matched endpoint unexpectedly has an action donor"
                )
        elif (
            isinstance(row.get("action_donor_sampling_id"), bool)
            or not isinstance(row.get("action_donor_sampling_id"), int)
        ):
            raise VideoResidualAnchorEvaluationError(
                "shuffled endpoint lacks a valid action donor"
            )
        observed[key] = row
    if set(observed) != expected:
        raise VideoResidualAnchorEvaluationError("validation row inventory is incomplete")
    if len(observed_snapshot_records) != 1:
        raise VideoResidualAnchorEvaluationError(
            "validation rows do not share one trained-arm snapshot"
        )
    invariant_hashes = (
        "cached_rgb_input_sha256",
        "cached_actions_input_sha256",
        "video_clean_scoring_sha256",
        "raw_ground_truth_sha256",
        "raw_history_last_sha256",
        "video_initial_noise_sha256",
        "auxiliary_initial_noise_sha256",
    )
    for clip_index in range(EXPECTED_VALIDATION_CLIPS):
        clip_rows = [observed[(clip_index, endpoint.code)] for endpoint in ENDPOINTS]
        reference = clip_rows[0]["tensor_sha256"]
        if any(
            row["tensor_sha256"].get(field) != reference.get(field)
            for row in clip_rows[1:]
            for field in invariant_hashes
        ):
            raise VideoResidualAnchorEvaluationError(
                f"paired target/noise changed for clip {clip_index}"
            )
        diagnostic = observed[(clip_index, "actions_shuffled_nfe_1")]
        own_action_hash = reference["cached_actions_input_sha256"]
        matched_rows = clip_rows[: len(NFE_GRID)]
        if any(
            row["tensor_sha256"]["sampler_actions_sha256"] != own_action_hash
            for row in matched_rows
        ):
            raise VideoResidualAnchorEvaluationError(
                "matched sampler actions differ from the registered clip"
            )
        donor_sampling_id = diagnostic.get("action_donor_sampling_id")
        donor_index = int(donor_sampling_id) - VALIDATION_SAMPLE_ID_OFFSET
        expected_donor_index = (
            clip_index + EXPECTED_WORLD_SIZE
            if (clip_index // EXPECTED_WORLD_SIZE) % 2 == 0
            else clip_index - EXPECTED_WORLD_SIZE
        )
        if (
            donor_sampling_id == diagnostic.get("sampling_id")
            or not 0 <= donor_index < EXPECTED_VALIDATION_CLIPS
            or donor_index != expected_donor_index
        ):
            raise VideoResidualAnchorEvaluationError("shuffled-action donor is not distinct")
        donor = observed[(donor_index, "autonomous_nfe_1")]
        donor_action_hash = donor["tensor_sha256"]["cached_actions_input_sha256"]
        if (
            diagnostic["tensor_sha256"]["sampler_actions_sha256"]
            != donor_action_hash
            or donor_action_hash == own_action_hash
            or registration["validation_descriptors"][clip_index]["episode_dir"]
            == registration["validation_descriptors"][donor_index]["episode_dir"]
        ):
            raise VideoResidualAnchorEvaluationError(
                "shuffled actions are not a changed, episode-disjoint donor input"
            )


def command_evaluate(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != EXPECTED_WORLD_SIZE:
        raise VideoResidualAnchorEvaluationError("evaluation requires eight ranks")
    if args.batch_size_per_rank != EXPECTED_BATCH_SIZE_PER_RANK:
        raise VideoResidualAnchorEvaluationError("batch size per rank is fixed at two")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if "B200" not in torch.cuda.get_device_properties(device).name.upper():
        raise VideoResidualAnchorEvaluationError("evaluation requires B200 GPUs")
    registration = _validate_registration(args.registration)
    arm = ARM_BY_CODE.get(args.arm)
    if arm is None:
        raise VideoResidualAnchorEvaluationError("unknown arm")
    output_root = Path(registration["output_root"])
    expected_output = output_root / "evaluation" / arm.code.lower()
    if args.output_dir.expanduser().absolute() != expected_output:
        raise VideoResidualAnchorEvaluationError("evaluation output path differs")
    run_dir = (output_root / "training" / arm.run_name).resolve(strict=True)
    if rank == 0:
        if expected_output.exists() or expected_output.is_symlink():
            raise VideoResidualAnchorEvaluationError("fresh evaluation output exists")
        expected_output.mkdir(parents=True, mode=0o700)
    dist.barrier()
    model, arm_artifacts = _load_model(registration, arm, run_dir, device)
    dataset = _dataset(registration)
    assigned = _expected_rank_indexes(rank)
    rows: list[dict[str, Any]] = []
    hook_calls = 0

    def count_calls(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal hook_calls
        hook_calls += 1

    handle = model.forward_model.register_forward_hook(count_calls)
    try:
        for start in range(0, len(assigned), EXPECTED_BATCH_SIZE_PER_RANK):
            indexes = assigned[start : start + EXPECTED_BATCH_SIZE_PER_RANK]
            samples = [dataset[index] for index in indexes]
            batch = phase._move_batch(samples, device)
            history = batch["rgb"][:, :5]
            sampling_ids = batch["clip_index"] + VALIDATION_SAMPLE_ID_OFFSET
            video_noise, auxiliary_noise = _sampling_noises(
                model, history, sampling_ids
            )
            endpoint_outputs: list[tuple[Endpoint, Any, Any, Any, list[int | None], int]] = []
            for endpoint in ENDPOINTS:
                if endpoint.action_source == "matched":
                    action_input = batch["actions"]
                    donors: list[int | None] = [None] * len(indexes)
                else:
                    if len(indexes) != 2:
                        raise VideoResidualAnchorEvaluationError(
                            "shuffled action control requires paired local batch"
                        )
                    action_input = batch["actions"].roll(shifts=1, dims=0)
                    donors = [
                        int(value)
                        for value in sampling_ids.roll(shifts=1, dims=0)
                        .detach()
                        .cpu()
                        .tolist()
                    ]
                before = hook_calls
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    generated = model.sample_video_residual_anchor(
                        history,
                        action_input,
                        batch["morphology_index"],
                        video_noise=video_noise,
                        auxiliary_noise=auxiliary_noise,
                        steps=endpoint.nfe,
                    )
                observed_calls = hook_calls - before
                endpoint_outputs.append(
                    (
                        endpoint,
                        generated,
                        _uint8(generated.decoded_future),
                        action_input.detach().cpu(),
                        donors,
                        observed_calls,
                    )
                )

            # The sampler has completed every endpoint.  Only now may the
            # evaluator encode and retain the hidden future scoring target.
            scoring = phase._scoring_targets(model, batch)
            clip_ids = [
                str(registration["validation_descriptors"][index]["clip_id"])
                for index in indexes
            ]
            for endpoint, generated, decoded, action_input, donors, calls in endpoint_outputs:
                rows.extend(
                    _rows_for_endpoint(
                        arm=arm,
                        endpoint=endpoint,
                        sample=generated,
                        decoded=decoded,
                        scoring=scoring,
                        clip_indexes=indexes,
                        clip_ids=clip_ids,
                        sampling_ids=sampling_ids,
                        action_input=action_input,
                        action_donor_ids=donors,
                        video_noise=video_noise.detach().cpu().to(torch.float16),
                        auxiliary_noise=auxiliary_noise.detach().cpu().to(torch.float16),
                        observed_calls=calls,
                        registration=registration,
                        arm_artifacts=arm_artifacts,
                    )
                )
    finally:
        handle.remove()

    rank_path = expected_output / f"rank_{rank:02d}.jsonl"
    content = b"".join(_canonical_json(row) + b"\n" for row in rows)
    _exclusive_bytes(rank_path, content)
    rank_receipt = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_RANK,
            "arm": asdict(arm),
            "rank": rank,
            "world_size": world_size,
            "indexes": assigned,
            "rows": len(rows),
            "rows_file": _file_record(rank_path),
            "transformer_calls": hook_calls,
            "protected_test_accessed": False,
            "target_cache_array_opened": False,
        }
    )
    _exclusive_json(expected_output / f"rank_{rank:02d}.json", rank_receipt)
    dist.barrier()
    if rank == 0:
        all_rows: list[dict[str, Any]] = []
        rank_records = []
        rank_receipt_records = []
        for source_rank in range(world_size):
            path = expected_output / f"rank_{source_rank:02d}.jsonl"
            rows_record = _file_record(path)
            rank_records.append(rows_record)
            with path.open(encoding="utf-8") as handle_rows:
                all_rows.extend(json.loads(line) for line in handle_rows if line.strip())
            receipt_path = expected_output / f"rank_{source_rank:02d}.json"
            receipt = _read_json(receipt_path, f"rank {source_rank} receipt")
            _validate_rank_receipt_payload(
                receipt,
                rank=source_rank,
                arm=arm,
                rows_record=rows_record,
            )
            rank_receipt_records.append(_file_record(receipt_path))
        _validate_rows(all_rows, arm, registration)
        inventory = identity_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": KIND_INVENTORY,
                "arm": asdict(arm),
                "registration_identity_sha256": registration["identity_sha256"],
                "rows": len(all_rows),
                "clips": EXPECTED_VALIDATION_CLIPS,
                "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
                "rank_files": rank_records,
                "rank_receipts": rank_receipt_records,
                "status": "complete",
                "protected_test_accessed": False,
                "target_cache_array_opened": False,
            }
        )
        _exclusive_json(expected_output / "inventory.json", inventory)
    dist.barrier()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--tool-repo", type=Path, required=True)
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--historical-repo", type=Path, required=True)
    register.add_argument("--study-root", type=Path, required=True)
    register.add_argument("--output-root", type=Path, required=True)
    register.add_argument("--train-manifest", type=Path, required=True)
    register.add_argument("--train-cache-metadata", type=Path, required=True)
    register.add_argument("--python", type=Path, required=True)
    register.set_defaults(handler=command_register)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--registration", type=Path, required=True)
    evaluate.add_argument("--arm", choices=tuple(ARM_BY_CODE), required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument(
        "--batch-size-per-rank", type=int, default=EXPECTED_BATCH_SIZE_PER_RANK
    )
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (VideoResidualAnchorEvaluationError, phase.PhaseLockProbeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
