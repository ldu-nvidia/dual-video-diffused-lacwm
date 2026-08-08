#!/usr/bin/env python3
"""Register and evaluate the paired VPM two-clock self-consistency screen.

Sampling accepts exactly five observed RGB frames, actions, morphology, and
deterministic Gaussian noise.  The evaluator retains the remaining eight RGB
frames privately and constructs clean Wan scoring latents only after every
autonomous endpoint in the batch has finished.
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

from tools import dual_abc_pilot  # noqa: E402
from tools import vpm_phaselock_probe as phase  # noqa: E402


SCHEMA_VERSION = 1
KIND_REGISTRATION = "two_clock_consistency_registration"
KIND_ROW = "two_clock_consistency_validation_clip"
KIND_RANK = "two_clock_consistency_validation_rank"
KIND_INVENTORY = "two_clock_consistency_validation_inventory"
TRAINING_COMMIT = phase.TRAINING_COMMIT
EXPECTED_WORLD_SIZE = 8
EXPECTED_BATCH_SIZE_PER_RANK = 2
EXPECTED_VALIDATION_CLIPS = 64
VALIDATION_SAMPLE_ID_OFFSET = 2_000_000
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
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs"
    / "experiments"
    / "vpm_two_clock_consistency_protocol.md"
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TwoClockConsistencyEvaluationError(RuntimeError):
    """A frozen input, deployment boundary, or paired row changed."""


@dataclass(frozen=True)
class Arm:
    code: str
    config_name: str
    run_name: str
    consistency_weight: float


ARMS = (
    Arm(
        "TC-CONT",
        "ravenhuang/wan-dit/two_clock_consistency_baseline",
        "two-clock-consistency-tc-cont-seed1234-u000200",
        0.0,
    ),
    Arm(
        "TC-CONS",
        "ravenhuang/wan-dit/two_clock_consistency_candidate",
        "two-clock-consistency-tc-cons-seed1234-u000200",
        0.2,
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
) + (
    # Preregistered diagnostic only; it never enters candidate selection.
    Endpoint("actions_shuffled_nfe_1", 1, "shuffled", False),
)
ENDPOINT_BY_CODE = {endpoint.code: endpoint for endpoint in ENDPOINTS}


def fixed_protocol() -> dict[str, Any]:
    """Return the complete preregistered contract for identity binding."""

    return {
        "arms": [asdict(arm) for arm in ARMS],
        "continuation_updates": 200,
        "optimizer_state_policy": "fresh_identical_adamw",
        "ema_policy": "none_in_historical_lacwm_and_none_in_both_arms",
        "train_clips": 512,
        "validation_clips": EXPECTED_VALIDATION_CLIPS,
        "world_size": EXPECTED_WORLD_SIZE,
        "local_training_batch_size": 1,
        "global_training_batch_size": 8,
        "training_validation": {
            "immutable_clips": 64,
            "batch_size_per_rank": 2,
            "batches_per_rank_per_event": 4,
            "iterations": [0, 100, 199],
            "iterator_cycles": True,
        },
        "two_clock_consistency": {
            "model_calls_per_update": 2,
            "shared_epsilon_trajectory": True,
            "clock_bands": {"high": [0.8, 1.0], "low": [0.0, 0.4]},
            "flow_supervision": "equal_mean_standard_rf_loss_on_both_calls",
            "consistency_direction": "high_x0_to_stopgrad_low_x0",
            "normalization": "masked_clean_future_latent_energy_plus_1e-6",
            "control_weight": 0.0,
            "candidate_weight": 0.2,
            "predicted_clean_conversion": "x0_hat=x_sigma-sigma*v_theta",
            "inference_override": False,
        },
        "nfe_grid": list(NFE_GRID),
        "evaluation_noise_seed": 20_260_726,
        "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
        "batch_size_per_evaluation_rank": EXPECTED_BATCH_SIZE_PER_RANK,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 20_260_807,
        "claim_contrasts": 9,
        "wandb_run_id_policy": "exact_arm_run_identity_sha256_resume_never",
        "protected_test_access_allowed": False,
        "target_cache_array_access_allowed_at_training": False,
        "target_cache_array_access_allowed_at_evaluation": False,
        "future_or_clean_feature_allowed_at_sampling": False,
        "train_validation_disjoint_by": [
            "clip_id",
            "episode_dir",
            "episode_dir_and_start",
        ],
    }


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


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: Mapping[str, Any]) -> bool:
    identity = payload.get("identity_sha256")
    if not isinstance(identity, str) or SHA256_RE.fullmatch(identity) is None:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest() == identity


def arm_run_identity(registration: Mapping[str, Any], arm: Arm) -> str:
    """Content identity written to every arm checkpoint and execution plan."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "two_clock_consistency_arm_identity",
        "registration_identity_sha256": registration["identity_sha256"],
        "tool_git_commit": registration["tool_repository"]["git_commit"],
        "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
        "arm": asdict(arm),
        "updates": 200,
        "seed": 1234,
        "world_size": 8,
        "local_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer_state_policy": "fresh_identical_adamw",
        "ema_policy": "none_in_historical_lacwm_and_none_in_both_arms",
    }
    return identity_payload(payload)["identity_sha256"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, rehash: bool = True) -> dict[str, Any]:
    path = path.expanduser()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise TwoClockConsistencyEvaluationError(f"input must be a regular absolute file: {path}")
    path = path.resolve(strict=True)
    record = {"path": str(path), "bytes": path.stat().st_size}
    if rehash:
        record["sha256"] = _sha256(path)
    return record


def _distributed_file_record(path: Path) -> dict[str, Any]:
    """Hash a large shared file once and broadcast its record to all ranks."""

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return _file_record(path)
    record = _file_record(path) if dist.get_rank() == 0 else None
    values = [record]
    dist.broadcast_object_list(values, src=0)
    if not isinstance(values[0], dict):
        raise TwoClockConsistencyEvaluationError(
            "distributed file record is invalid"
        )
    return values[0]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TwoClockConsistencyEvaluationError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TwoClockConsistencyEvaluationError(f"{label} must contain one object")
    return value


def _exclusive_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise TwoClockConsistencyEvaluationError(f"refusing to overwrite output: {path}") from exc
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
        raise TwoClockConsistencyEvaluationError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _clean_source(repo: Path, expected_commit: str, label: str) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise TwoClockConsistencyEvaluationError(f"{label} commit must be full SHA")
    if _git(repo, "rev-parse", "--show-toplevel") != str(repo):
        raise TwoClockConsistencyEvaluationError(f"{label} is not a worktree root")
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != expected_commit or _git(
        repo, "status", "--porcelain", "--untracked-files=all"
    ):
        raise TwoClockConsistencyEvaluationError(f"{label} source is not clean/pinned")
    return {
        "path": str(repo),
        "git_commit": actual,
        "git_tree_sha": _git(repo, "rev-parse", "HEAD^{tree}"),
        "clean": True,
    }


def _fresh_lustre_root(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute() or not path.name or path.name in {".", ".."}:
        raise TwoClockConsistencyEvaluationError("output root must be an absolute named path")
    parent = path.parent.resolve(strict=True)
    canonical = parent / path.name
    if canonical != path or Path("/lustre") not in canonical.parents:
        raise TwoClockConsistencyEvaluationError("output root must be canonical under /lustre")
    if canonical.exists() or canonical.is_symlink():
        raise TwoClockConsistencyEvaluationError("output root must be fresh")
    return canonical


def _resolve_array(metadata_path: Path, metadata: Mapping[str, Any], name: str) -> Path:
    value = metadata.get(f"{name}_file")
    if not isinstance(value, str) or not value:
        raise TwoClockConsistencyEvaluationError(f"metadata lacks {name}_file")
    path = Path(value)
    if not path.is_absolute():
        path = metadata_path.parent / path
    return Path(_file_record(path, rehash=False)["path"])


def _manifest_descriptors(
    path: Path,
    *,
    expected_split: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Parse and validate the immutable clip identity/topology contract."""

    descriptors: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(
                        line,
                        parse_constant=lambda token: (_ for _ in ()).throw(
                            ValueError(f"non-finite JSON constant {token}")
                        ),
                    )
                    if not isinstance(value, dict):
                        raise ValueError("manifest row is not an object")
                    descriptors.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TwoClockConsistencyEvaluationError(
            f"{expected_split} manifest is invalid JSONL"
        ) from exc
    if len(descriptors) != expected_count:
        raise TwoClockConsistencyEvaluationError(
            f"{expected_split} manifest clip count differs"
        )
    clip_ids: set[str] = set()
    episode_dirs: set[str] = set()
    for index, descriptor in enumerate(descriptors):
        clip_id = descriptor.get("clip_id")
        episode_dir = descriptor.get("episode_dir")
        frame_indices = descriptor.get("frame_indices")
        if (
            descriptor.get("split") != expected_split
            or descriptor.get("auxiliary_index") != index
            or not isinstance(clip_id, str)
            or SHA256_RE.fullmatch(clip_id) is None
            or clip_id in clip_ids
            or not isinstance(episode_dir, str)
            or not episode_dir.startswith("/")
            or episode_dir in episode_dirs
            or descriptor.get("sample_size") != 13
            or descriptor.get("chunk_size") != 5
            or descriptor.get("action_span") != 65
            or not isinstance(frame_indices, list)
            or len(frame_indices) != 13
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in frame_indices
            )
            or frame_indices
            != list(range(frame_indices[0], frame_indices[0] + 65, 5))
            or descriptor.get("start") != frame_indices[0]
        ):
            raise TwoClockConsistencyEvaluationError(
                f"{expected_split} manifest row {index} differs"
            )
        clip_ids.add(clip_id)
        episode_dirs.add(episode_dir)
    return descriptors


def _validate_train_inputs(manifest: Path, metadata_path: Path) -> dict[str, Any]:
    manifest_record = _file_record(manifest)
    metadata_record = _file_record(metadata_path)
    if manifest_record["sha256"] != TRAIN_MANIFEST_SHA256:
        raise TwoClockConsistencyEvaluationError("train manifest changed")
    if metadata_record["sha256"] != TRAIN_METADATA_SHA256:
        raise TwoClockConsistencyEvaluationError("train metadata changed")
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
        raise TwoClockConsistencyEvaluationError("train RGB/action metadata differs")
    rgb = _resolve_array(metadata_path, metadata, "rgb")
    actions = _resolve_array(metadata_path, metadata, "actions")
    rgb_record = _file_record(rgb)
    action_record = _file_record(actions)
    if rgb_record["sha256"] != TRAIN_RGB_SHA256:
        raise TwoClockConsistencyEvaluationError("train RGB array changed")
    if action_record["sha256"] != TRAIN_ACTIONS_SHA256:
        raise TwoClockConsistencyEvaluationError("train action array changed")
    descriptors = _manifest_descriptors(
        Path(manifest_record["path"]), expected_split="train", expected_count=512
    )
    return {
        "manifest": manifest_record,
        "cache_metadata": metadata_record,
        "rgb": rgb_record,
        "actions": action_record,
        "clip_count": 512,
        "split": "train",
        "descriptors": descriptors,
        "auxiliary_target_array_opened": False,
    }


def command_register(args: argparse.Namespace) -> int:
    tool_repo = args.tool_repo.expanduser().resolve(strict=True)
    historical_repo = args.historical_repo.expanduser().resolve(strict=True)
    source = _clean_source(tool_repo, args.expected_commit, "tool")
    historical = _clean_source(historical_repo, TRAINING_COMMIT, "historical model")
    _git(
        tool_repo,
        "merge-base",
        "--is-ancestor",
        TRAINING_COMMIT,
        args.expected_commit,
    )
    if tool_repo != REPO_ROOT:
        raise TwoClockConsistencyEvaluationError("registration tool repository differs")
    if not PROTOCOL_PATH.is_file() or PROTOCOL_PATH.is_symlink():
        raise TwoClockConsistencyEvaluationError("prospective protocol is absent")
    output_root = _fresh_lustre_root(args.output_root)
    validated = phase._validate_study_metadata(
        args.study_root.expanduser().resolve(strict=True),
        historical_repo,
        rehash_snapshot=True,
    )
    if validated["snapshot_sha256"] != PARENT_SNAPSHOT_SHA256:
        raise TwoClockConsistencyEvaluationError("parent VPM snapshot digest changed")
    train = _validate_train_inputs(args.train_manifest, args.train_cache_metadata)
    train_descriptors = train.pop("descriptors")
    validation_descriptors = _manifest_descriptors(
        Path(validated["validation"]["manifest"]["path"]),
        expected_split="val",
        expected_count=EXPECTED_VALIDATION_CLIPS,
    )
    if validation_descriptors != validated["descriptors"]:
        raise TwoClockConsistencyEvaluationError(
            "validation descriptors differ from the frozen parent study"
        )
    train_clip_ids = {str(row["clip_id"]) for row in train_descriptors}
    validation_clip_ids = {str(row["clip_id"]) for row in validation_descriptors}
    train_episodes = {str(row["episode_dir"]) for row in train_descriptors}
    validation_episodes = {
        str(row["episode_dir"]) for row in validation_descriptors
    }
    train_episode_starts = {
        (str(row["episode_dir"]), int(row["start"]))
        for row in train_descriptors
    }
    validation_episode_starts = {
        (str(row["episode_dir"]), int(row["start"]))
        for row in validation_descriptors
    }
    if (
        train_clip_ids & validation_clip_ids
        or train_episodes & validation_episodes
        or train_episode_starts & validation_episode_starts
    ):
        raise TwoClockConsistencyEvaluationError(
            "train and validation manifests overlap"
        )
    train["validation_disjointness"] = {
        "clip_id": True,
        "episode_dir": True,
        "episode_dir_and_start": True,
        "train_unique_clip_ids": len(train_clip_ids),
        "train_unique_episode_dirs": len(train_episodes),
        "validation_unique_clip_ids": len(validation_clip_ids),
        "validation_unique_episode_dirs": len(validation_episodes),
    }
    wandb = dual_abc_pilot._wandb_private_project(
        "zijiandu", "dual-video-diffusion-private"
    )
    # The authenticated email is not needed for provenance and should not be
    # copied into durable study receipts or redirected registration stdout.
    wandb.pop("viewer_email", None)
    python = args.python.expanduser().resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise TwoClockConsistencyEvaluationError("runtime Python is not executable")
    payload = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_REGISTRATION,
            "created_at_utc": _now(),
            "status": "registered_before_candidate_metrics",
            "output_root": str(output_root),
            "tool_repository": source,
            "historical_model_repository": historical,
            "protocol": _file_record(PROTOCOL_PATH),
            "controlled_study": {
                "study_root": str(args.study_root.resolve(strict=True)),
                "study_identity_sha256": validated["study"]["identity_sha256"],
                "arm_identity_sha256": validated["arm"]["identity_sha256"],
                "stage_identity_sha256": validated["stage"]["identity_sha256"],
                "parent_snapshot": _file_record(validated["paths"]["snapshot"]),
                "parent_completed_updates": 1000,
                "parent_total_observations": 8000,
                "training_git_commit": TRAINING_COMMIT,
            },
            "training": train,
            "validation": validated["validation"],
            "validation_descriptors": validated["descriptors"],
            "runtime": {
                **validated["runtime"],
                "python": str(python),
            },
            "fixed_protocol": fixed_protocol(),
            "wandb": {**wandb, "group": None, "mode": "online"},
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
    disjointness = registration.get("training", {}).get(
        "validation_disjointness"
    )
    parent = registration.get("controlled_study", {})
    if (
        not identity_valid(registration)
        or registration.get("kind") != KIND_REGISTRATION
        or registration.get("status") != "registered_before_candidate_metrics"
        or registration.get("protected_test_accessed") is not False
        or registration.get("auxiliary_target_array_opened") is not False
        or registration.get("fixed_protocol") != fixed_protocol()
        or parent.get("parent_completed_updates") != 1000
        or parent.get("parent_total_observations") != 8000
        or parent.get("training_git_commit") != TRAINING_COMMIT
        or parent.get("parent_snapshot", {}).get("sha256")
        != PARENT_SNAPSHOT_SHA256
        or registration.get("training", {}).get("split") != "train"
        or registration.get("training", {}).get("clip_count") != 512
        or registration.get("training", {}).get(
            "auxiliary_target_array_opened"
        )
        is not False
        or disjointness
        != {
            "clip_id": True,
            "episode_dir": True,
            "episode_dir_and_start": True,
            "train_unique_clip_ids": 512,
            "train_unique_episode_dirs": 512,
            "validation_unique_clip_ids": 64,
            "validation_unique_episode_dirs": 64,
        }
        or registration.get("wandb", {}).get("entity") != "zijiandu"
        or registration.get("wandb", {}).get("project")
        != "dual-video-diffusion-private"
        or registration.get("wandb", {}).get("access") != "PRIVATE"
        or registration.get("wandb", {}).get("viewer_username") != "zijiandu"
        or registration.get("wandb", {}).get("group") is not None
        or registration.get("wandb", {}).get("mode") != "online"
    ):
        raise TwoClockConsistencyEvaluationError("registration identity/protocol differs")
    canonical = Path(registration["output_root"]) / "registration.json"
    if path != canonical.resolve(strict=True) or _sha256(path) != _sha256(canonical):
        raise TwoClockConsistencyEvaluationError("registration path differs from canonical root")
    return registration


def _revalidate_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = Path(str(record.get("path", "")))
    observed = _file_record(path)
    if any(
        observed.get(field) != record.get(field)
        for field in ("path", "bytes", "sha256")
    ):
        raise TwoClockConsistencyEvaluationError(f"registered {label} changed")
    return observed


def revalidate_registered_inputs(
    registration: Mapping[str, Any],
    *,
    include_parent: bool,
    include_train: bool,
    include_validation: bool,
) -> dict[str, Any]:
    """Rehash immutable inputs at the arm/evaluation point of use."""

    verified: dict[str, Any] = {}
    if include_parent:
        verified["parent_snapshot"] = _revalidate_record(
            registration["controlled_study"]["parent_snapshot"],
            "parent snapshot",
        )
    if include_train:
        training = registration["training"]
        verified["training"] = {
            key: _revalidate_record(training[key], f"training {key}")
            for key in ("manifest", "cache_metadata", "rgb", "actions")
        }
    if include_validation:
        validation = registration["validation"]
        verified["validation"] = {
            "manifest": _revalidate_record(
                validation["manifest"], "validation manifest"
            ),
            "cache_metadata": _revalidate_record(
                validation["cache_metadata"], "validation cache metadata"
            ),
            "rgb": _revalidate_record(
                validation["arrays"]["rgb"], "validation RGB"
            ),
            "actions": _revalidate_record(
                validation["arrays"]["actions"], "validation actions"
            ),
        }
    return {
        "verified": verified,
        "all_selected_registered_inputs_rehashed": True,
        "auxiliary_target_array_opened": False,
        "protected_test_accessed": False,
    }


def _validate_training_trace(
    run_dir: Path, arm: Arm, registration: Mapping[str, Any]
) -> dict[str, Any]:
    trace = run_dir / "two_clock_consistency_training_trace.jsonl"
    complete_path = (
        run_dir / "two_clock_consistency_training_trace_complete.json"
    )
    trace_record = _file_record(trace)
    complete = _read_json(complete_path, "training trace completion")
    expected_run_identity = arm_run_identity(registration, arm)
    with trace.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    if (
        complete.get("kind") != "two_clock_consistency_training_trace_complete"
        or complete.get("arm") != arm.code
        or complete.get("completed_updates") != 200
        or complete.get("trace_sha256") != trace_record["sha256"]
        or complete.get("rows") != len(lines)
        or complete.get("protected_test_accessed") is not False
    ):
        raise TwoClockConsistencyEvaluationError("training trace completion differs")
    with trace.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
    if (
        header.get("kind") != "two_clock_consistency_training_trace_header"
        or header.get("arm") != arm.code
        or header.get("consistency_weight") != arm.consistency_weight
        or header.get("clock_bands")
        != {"high": [0.8, 1.0], "low": [0.0, 0.4]}
        or header.get("model_calls_per_update") != 2
        or header.get("shared_epsilon_trajectory") is not True
        or header.get("parent_snapshot_sha256") != PARENT_SNAPSHOT_SHA256
        or header.get("parent_completed_updates") != 1000
        or header.get("parent_total_observations") != 8000
        or header.get("continuation_updates") != 200
        or header.get("run_identity_sha256") != expected_run_identity
        or header.get("optimizer_state_policy") != "fresh_identical_adamw"
        or header.get("initial_optimizer_state_entries") != 0
        or header.get("ema_policy")
        != "none_in_historical_lacwm_and_none_in_both_arms"
    ):
        raise TwoClockConsistencyEvaluationError("training trace header differs")
    return {
        "trace": trace_record,
        "completion": _file_record(complete_path),
        "header": header,
        "registration_identity_sha256": registration["identity_sha256"],
    }


def _validate_arm_plan(
    registration: Mapping[str, Any], arm: Arm, run_dir: Path
) -> dict[str, Any]:
    root = Path(registration["output_root"])
    path = root / "arm_plans" / f"{arm.code.lower()}.json"
    plan = _read_json(path, "arm execution plan")
    expected_identity = arm_run_identity(registration, arm)
    if (
        not identity_valid(plan)
        or plan.get("kind") != "two_clock_consistency_arm_execution_plan"
        or plan.get("status") != "planned_before_arm_training_or_metrics"
        or plan.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or plan.get("arm") != asdict(arm)
        or plan.get("run_identity_sha256") != expected_identity
        or plan.get("paths", {}).get("run_dir") != str(run_dir)
        or plan.get("training", {}).get("updates") != 200
        or plan.get("training", {}).get("world_size") != 8
        or plan.get("training", {}).get("global_batch_size") != 8
        or plan.get("training", {}).get("model_calls_per_update") != 2
        or plan.get("training", {}).get("shared_epsilon_trajectory") is not True
        or plan.get("training", {}).get("clock_bands")
        != {"high": [0.8, 1.0], "low": [0.0, 0.4]}
        or plan.get("training", {}).get("consistency_direction")
        != "high_x0_to_stopgrad_low_x0"
        or plan.get("training", {}).get("validation")
        != {
            "immutable_clips": 64,
            "batch_size_per_rank": 2,
            "batches_per_rank_per_event": 4,
            "iterations": [0, 100, 199],
            "iterator_cycles": True,
        }
        or plan.get("evaluation", {}).get("protected_test_accessed") is not False
        or plan.get("evaluation", {}).get("target_cache_array_opened") is not False
        or plan.get("input_revalidation", {}).get(
            "all_selected_registered_inputs_rehashed"
        )
        is not True
        or plan.get("input_revalidation", {}).get("auxiliary_target_array_opened")
        is not False
        or plan.get("wandb")
        != {
            "entity": "zijiandu",
            "project": "dual-video-diffusion-private",
            "group": None,
            "mode": "online",
            "id": expected_identity,
            "resume": "never",
        }
    ):
        raise TwoClockConsistencyEvaluationError("arm execution plan differs")
    return {"record": _file_record(path), "identity_sha256": plan["identity_sha256"]}


def _validate_resolved_config(
    config: Any,
    arm: Arm,
    *,
    expected_run_identity: str | None = None,
) -> None:
    """Fail closed on every protocol-critical training/configuration field."""

    dual = config.model.dual_diffusion
    trainer = config.trainer.config
    optimizer = config.optimizer_factory
    scheduler = config.lr_scheduler_factory.lr_lambda
    train_abc = config.dataset.datasets.ABC
    val_abc = config.val_dataset.datasets.ABC
    val_loader = config.val_data_loader[0]
    expected_betas = [0.9, 0.95]
    if (
        str(config.name) != arm.run_name
        or int(config.seed) != 1234
        or not str(config.model.get("_target_", "")).endswith(
            ".TwoClockConsistencyVPM"
        )
        or float(config.model.two_clock_consistency.weight)
        != arm.consistency_weight
        or float(config.model.two_clock_consistency.epsilon) != 1e-6
        or float(config.model.two_clock_consistency.high_sigma_min) != 0.8
        or float(config.model.two_clock_consistency.high_sigma_max) != 1.0
        or float(config.model.two_clock_consistency.low_sigma_min) != 0.0
        or float(config.model.two_clock_consistency.low_sigma_max) != 0.4
        or int(config.model.num_history_frames) != 5
        or int(config.model.num_future_frames) != 8
        or not bool(dual.enabled)
        or not bool(dual.parameter_matched_control)
        or bool(dual.video_only_control)
        or bool(dual.condition_on_tf)
        or bool(dual.condition_on_tf_clock)
        or not bool(dual.head_condition_on_tf_clock)
        or str(dual.auxiliary_history_mode) != "diffuse_all"
        or str(dual.condition_mode) != "off"
        or str(dual.schedule_mode) != "aligned"
        or float(dual.tf_lead_logit) != 0.0
        or int(dual.tf_channels) != 64
        or float(dual.tf_loss_weight) != 0.0
        or float(dual.state_gate_init) != 0.0
        or not bool(dual.state_gate_trainable)
        or float(dual.clock_gate_init) != 0.0
        or not bool(dual.clock_gate_trainable)
        or list(dual.evaluation_nfe_steps) != list(NFE_GRID)
        or int(dual.evaluation_noise_seed) != 20_260_726
        or list(dual.evaluation_condition_sources) != ["autonomous"]
        or int(config.dataset.padding_dim) != 157
        or int(config.dataset.transform.sample_size) != 13
        or int(config.dataset.transform.chunk_size) != 5
        or not bool(config.dataset.infinite)
        or bool(config.dataset.img_augment)
        or bool(config.dataset.emit_crop_rgb)
        or bool(config.dataset.future_validity.enabled)
        or not str(train_abc.get("_target_", "")).endswith(
            ".ABCTwoClockConsistencyDataset"
        )
        or train_abc.expected_split != "train"
        or int(train_abc.expected_clip_count) != 512
        or train_abc.expected_manifest_sha256 != TRAIN_MANIFEST_SHA256
        or train_abc.expected_rgb_sha256 != TRAIN_RGB_SHA256
        or train_abc.expected_actions_sha256 != TRAIN_ACTIONS_SHA256
        or not bool(config.val_dataset.infinite)
        or val_abc.expected_split != "val"
        or int(val_abc.expected_clip_count) != EXPECTED_VALIDATION_CLIPS
        or val_abc.expected_manifest_sha256
        != phase.EXPECTED_VALIDATION_MANIFEST_SHA256
        or val_abc.expected_rgb_sha256 != phase.EXPECTED_VALIDATION_RGB_SHA256
        or val_abc.expected_actions_sha256
        != phase.EXPECTED_VALIDATION_ACTIONS_SHA256
        or len(config.val_data_loader) != 1
        or int(val_loader.batch_size) != 2
        or int(val_loader.num_workers) != 2
        or len(config.viz_data_loader) != 0
        or int(config.data_loader.batch_size) != 1
        or int(config.data_loader.num_workers) != 4
        or not bool(config.data_loader.pin_memory)
        or int(config.data_loader.prefetch_factor) != 2
        or not bool(config.data_loader.persistent_workers)
        or int(trainer.max_iter) != 200
        or int(trainer.gradient_accumulation_steps) != 1
        or list(trainer.exclude_keys) != []
        or bool(trainer.get("share_spatial_attention", False))
        or trainer.transition_handoff_path is not None
        or int(trainer.logging.log_every) != 1
        or int(trainer.validation.val_every) != 100
        or int(trainer.validation.n_val_samples) != 4
        or bool(trainer.validation.save_best)
        or str(trainer.dtype) != "bfloat16"
        or not bool(trainer.amp_enabled)
        or float(trainer.gradient_clipping.max_norm) != 1.0
        or float(trainer.gradient_clipping.norm_type) != 2.0
        or bool(trainer.gradient_clipping.error_if_nonfinite)
        or float(optimizer.lr) != 1e-4
        or list(optimizer.betas) != expected_betas
        or float(optimizer.eps) != 1e-8
        or float(optimizer.weight_decay) != 0.01
        or int(scheduler.warmup_steps) != 20
        or int(scheduler.total_steps) != 200
        or float(scheduler.final_learning_rate) != 1e-6
        or not bool(config.wandb.enabled)
        or str(config.wandb.mode) != "online"
        or config.wandb.entity != "zijiandu"
        or config.wandb.project != "dual-video-diffusion-private"
        or config.wandb.group is not None
        or str(config.wandb.settings.start_method) != "thread"
        or bool(config.wandb.settings.save_code)
        or float(config.wandb.settings.finish_timeout) != 120.0
        or bool(config.wandb.settings.finish_timeout_raises)
        or (
            expected_run_identity is not None
            and (
                config.wandb.get("id") != expected_run_identity
                or config.wandb.get("resume") != "never"
            )
        )
    ):
        raise TwoClockConsistencyEvaluationError("resolved arm configuration differs")


def _load_model(
    registration: Mapping[str, Any], arm: Arm, run_dir: Path, device: Any
) -> tuple[Any, Any, dict[str, Any]]:
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
    expected_run_identity = arm_run_identity(registration, arm)
    _validate_resolved_config(
        config, arm, expected_run_identity=expected_run_identity
    )
    model = instantiate(config.model)
    snapshot_path = run_dir / "snapshot.pt"
    snapshot_record = _distributed_file_record(snapshot_path)
    snapshot = torch.load(
        snapshot_path, map_location="cpu", weights_only=True, mmap=True
    )
    rank_states = snapshot.get("rank_states")
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("world_size") != EXPECTED_WORLD_SIZE
        or snapshot.get("_start_iter") != 200
        or snapshot.get("_total_observations") != 1600
        or snapshot.get("gradient_accumulation_steps") != 1
        or snapshot.get("run_identity_sha256") != expected_run_identity
        or not isinstance(rank_states, Sequence)
        or isinstance(rank_states, (str, bytes))
        or len(rank_states) != EXPECTED_WORLD_SIZE
        or any(not isinstance(state, Mapping) for state in rank_states)
        or [state.get("global_rank") for state in rank_states]
        != list(range(EXPECTED_WORLD_SIZE))
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
        or not isinstance(snapshot.get("model"), Mapping)
    ):
        raise TwoClockConsistencyEvaluationError("trained arm snapshot differs")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise TwoClockConsistencyEvaluationError("trained arm strict load failed")
    del snapshot
    model = model.to(device=device).eval()
    model._ensure_video_only_runtime_contract()
    if (
        model.tf_condition_mode != "off"
        or model.condition_on_tf
        or bool(model.condition_on_tf_clock)
        or model.tf_loss_weight != 0.0
        or not model.parameter_matched_control
        or not bool(model.forward_model.tf_head_condition_on_clock)
        or float(model.consistency_weight) != arm.consistency_weight
        or float(model.high_sigma_min) != 0.8
        or float(model.high_sigma_max) != 1.0
        or float(model.low_sigma_min) != 0.0
        or float(model.low_sigma_max) != 0.4
        or tuple(model.evaluation_nfe_steps) != NFE_GRID
        or tuple(model.evaluation_condition_sources) != ("autonomous",)
    ):
        raise TwoClockConsistencyEvaluationError("trained model is not deployable VPM")
    suspicious = []
    for name, module in model.named_modules():
        identity = f"{name}:{type(module).__module__}.{type(module).__name__}".lower()
        if any(term in identity for term in ("vjepa", "teacher", "dino")):
            suspicious.append(identity)
    if suspicious or getattr(model, "time_frequency_transform", None) is not None:
        raise TwoClockConsistencyEvaluationError("online feature/teacher module is registered")
    training = _validate_training_trace(run_dir, arm, registration)
    completion_path = run_dir / "training_complete.json"
    completion = _read_json(completion_path, "training completion")
    if (
        completion.get("status") != "completed"
        or completion.get("completed_updates") != 200
        or completion.get("max_iter") != 200
        or completion.get("run_identity_sha256") != expected_run_identity
        or completion.get("snapshot") != str(snapshot_path.resolve(strict=True))
    ):
        raise TwoClockConsistencyEvaluationError("training completion differs")
    arm_plan = _validate_arm_plan(registration, arm, run_dir)
    return model, config, {
        "snapshot": snapshot_record,
        "resolved_config": config_record,
        "training_trace": training,
        "training_completion": _file_record(completion_path),
        "arm_execution_plan": arm_plan,
        "run_identity_sha256": expected_run_identity,
    }


def _dataset(registration: Mapping[str, Any]) -> Any:
    validation = registration["validation"]
    return phase._RegisteredValidationInputs(
        rgb_path=Path(validation["arrays"]["rgb"]["path"]),
        actions_path=Path(validation["arrays"]["actions"]["path"]),
        descriptors=registration["validation_descriptors"],
        padding_dim=157,
    )


def _uint8_future(model: Any, video: Any, out_hw: tuple[int, int]) -> Any:
    import torch

    decoded = model.rgb_tokenizer.decode_temporal(video, out_hw=out_hw)
    return (
        ((decoded[:, :, -8:].float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(dtype=torch.uint8)
        .cpu()
    )


def _endpoint_rows(
    *,
    arm: Arm,
    endpoint: Endpoint,
    result: Mapping[str, Any],
    prepared: Mapping[str, Any],
    decoded: Any,
    scoring: Mapping[str, Any],
    clip_indexes: Sequence[int],
    clip_ids: Sequence[str],
    sampling_ids: Any,
    observed_calls: int,
    registration: Mapping[str, Any],
    history_input: Any,
    action_input: Any,
    action_donor_ids: Sequence[int | None],
    arm_artifacts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    final = result["video"].detach().cpu().to(torch.float16)
    clean = scoring["video_clean"]
    history_tokens = int(prepared["history_frames"])
    if observed_calls != endpoint.nfe or result["calls"] != endpoint.nfe:
        raise TwoClockConsistencyEvaluationError("actual transformer calls differ from NFE")
    latent_nmse = phase._per_sample_nmse(final, clean, history_tokens)
    latent_delta_nmse = phase._per_sample_future_delta_nmse(
        final, clean, history_tokens
    )
    decoded_metrics = phase._per_sample_decoded(
        decoded, scoring["ground_truth"], scoring["history_last"]
    )
    sampling_values = [int(value) for value in sampling_ids.detach().cpu().tolist()]
    hashes = {
        "cached_rgb_input_sha256": scoring["rgb_hashes"],
        "sampler_history_rgb_sha256": phase._slice_hashes(history_input),
        "cached_actions_input_sha256": scoring["actions_hashes"],
        "sampler_actions_sha256": phase._slice_hashes(action_input),
        "video_clean_scoring_sha256": phase._slice_hashes(clean),
        "raw_ground_truth_sha256": phase._slice_hashes(scoring["ground_truth"]),
        "raw_history_last_sha256": phase._slice_hashes(scoring["history_last"]),
        "video_initial_noise_sha256": phase._slice_hashes(
            prepared["initial_video"].detach().cpu().to(torch.float16)
        ),
        "tf_initial_noise_sha256": phase._slice_hashes(
            prepared["initial_tf"].detach().cpu().to(torch.float16)
        ),
        "video_final_sha256": phase._slice_hashes(final),
        "decoded_final_sha256": phase._slice_hashes(decoded),
    }
    rows = []
    for offset, (clip_index, clip_id) in enumerate(zip(clip_indexes, clip_ids)):
        rows.append(
            identity_payload(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": KIND_ROW,
                    "registration_identity_sha256": registration[
                        "identity_sha256"
                    ],
                    "tool_git_commit": registration["tool_repository"][
                        "git_commit"
                    ],
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
                    "history_rgb_frames": 5,
                    "future_rgb_frames": 8,
                    "history_video_latent_tokens": history_tokens,
                    "future_video_latent_tokens": int(final.shape[2] - history_tokens),
                    "actual_transformer_call_count": observed_calls,
                    "declared_nfe": endpoint.nfe,
                    "metrics": {
                        "video_future_nmse": latent_nmse[offset],
                        "video_future_temporal_delta_nmse": latent_delta_nmse[
                            offset
                        ],
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


def _expected_rank_transformer_calls() -> int:
    return (
        len(_expected_rank_indexes(0))
        // EXPECTED_BATCH_SIZE_PER_RANK
        * sum(endpoint.nfe for endpoint in ENDPOINTS)
    )


def _validate_rank_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_rank: int,
    arm: Arm,
    registration: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    """Validate one rank receipt and return its canonical row path."""

    expected_rows = len(_expected_rank_indexes(expected_rank)) * len(ENDPOINTS)
    row_record = manifest.get("rows")
    expected_row_path = output_dir / f"rank_{expected_rank:03d}.jsonl"
    if (
        not identity_valid(manifest)
        or manifest.get("kind") != KIND_RANK
        or manifest.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or manifest.get("arm") != asdict(arm)
        or manifest.get("rank") != expected_rank
        or manifest.get("world_size") != EXPECTED_WORLD_SIZE
        or manifest.get("batch_size_per_rank") != EXPECTED_BATCH_SIZE_PER_RANK
        or manifest.get("assigned_clip_indexes")
        != _expected_rank_indexes(expected_rank)
        or manifest.get("endpoints")
        != [asdict(endpoint) for endpoint in ENDPOINTS]
        or manifest.get("actual_transformer_call_count")
        != _expected_rank_transformer_calls()
        or manifest.get("protected_test_accessed") is not False
        or manifest.get("target_cache_array_opened") is not False
        or not isinstance(row_record, Mapping)
        or row_record.get("path") != str(expected_row_path)
        or row_record.get("count") != expected_rows
        or isinstance(row_record.get("bytes"), bool)
        or not isinstance(row_record.get("bytes"), int)
        or row_record.get("bytes", -1) <= 0
        or not isinstance(row_record.get("sha256"), str)
        or SHA256_RE.fullmatch(str(row_record.get("sha256"))) is None
    ):
        raise TwoClockConsistencyEvaluationError("rank manifest differs")
    if (
        not expected_row_path.is_file()
        or expected_row_path.is_symlink()
        or expected_row_path.stat().st_size != row_record["bytes"]
        or _sha256(expected_row_path) != row_record["sha256"]
    ):
        raise TwoClockConsistencyEvaluationError("rank rows changed")
    return expected_row_path


def _expected_action_donor_index(clip_index: int) -> int:
    """Return the roll-by-one donor in the fixed rank-local batch of two."""

    rank = int(clip_index) % EXPECTED_WORLD_SIZE
    assigned = _expected_rank_indexes(rank)
    position = assigned.index(int(clip_index))
    batch_start = (
        position // EXPECTED_BATCH_SIZE_PER_RANK
    ) * EXPECTED_BATCH_SIZE_PER_RANK
    batch = assigned[batch_start : batch_start + EXPECTED_BATCH_SIZE_PER_RANK]
    if len(batch) != EXPECTED_BATCH_SIZE_PER_RANK:
        raise TwoClockConsistencyEvaluationError(
            "action diagnostic batch is incomplete"
        )
    return batch[(position - batch_start - 1) % EXPECTED_BATCH_SIZE_PER_RANK]


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    arm: Arm,
    registration: Mapping[str, Any],
) -> None:
    expected = {
        (index, endpoint.code)
        for index in range(EXPECTED_VALIDATION_CLIPS)
        for endpoint in ENDPOINTS
    }
    observed: dict[tuple[int, str], Mapping[str, Any]] = {}
    descriptors = registration.get("validation_descriptors")
    if (
        not isinstance(descriptors, Sequence)
        or len(descriptors) != EXPECTED_VALIDATION_CLIPS
    ):
        raise TwoClockConsistencyEvaluationError(
            "validation descriptors are incomplete"
        )
    required_hashes = (
        "cached_rgb_input_sha256",
        "sampler_history_rgb_sha256",
        "cached_actions_input_sha256",
        "sampler_actions_sha256",
        "video_clean_scoring_sha256",
        "raw_ground_truth_sha256",
        "raw_history_last_sha256",
        "video_initial_noise_sha256",
        "tf_initial_noise_sha256",
        "video_final_sha256",
        "decoded_final_sha256",
    )
    for row in rows:
        endpoint_value = row.get("endpoint")
        code = endpoint_value.get("code") if isinstance(endpoint_value, Mapping) else None
        endpoint = ENDPOINT_BY_CODE.get(str(code))
        key = (row.get("clip_index"), str(code))
        clip_index = row.get("clip_index")
        expected_clip_id = (
            descriptors[clip_index].get("clip_id")
            if isinstance(clip_index, int)
            and not isinstance(clip_index, bool)
            and 0 <= clip_index < EXPECTED_VALIDATION_CLIPS
            and isinstance(descriptors[clip_index], Mapping)
            else None
        )
        expected_sampling_id = (
            VALIDATION_SAMPLE_ID_OFFSET + clip_index
            if expected_clip_id is not None
            else None
        )
        expected_donor = (
            None
            if endpoint is None
            or endpoint.action_source == "matched"
            or expected_clip_id is None
            else VALIDATION_SAMPLE_ID_OFFSET
            + _expected_action_donor_index(clip_index)
        )
        hashes = row.get("tensor_sha256")
        if (
            not identity_valid(row)
            or row.get("kind") != KIND_ROW
            or row.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or row.get("arm") != asdict(arm)
            or endpoint is None
            or dict(endpoint_value) != asdict(endpoint)
            or row.get("evaluation_split") != "validation"
            or row.get("clip_id") != expected_clip_id
            or row.get("sampling_id") != expected_sampling_id
            or row.get("action_donor_sampling_id") != expected_donor
            or row.get("protected_test_accessed") is not False
            or row.get("clean_future_rgb_passed_to_sampler") is not False
            or row.get("clean_video_latent_passed_to_sampler") is not False
            or row.get("clean_auxiliary_passed_to_sampler") is not False
            or row.get("target_cache_array_opened") is not False
            or row.get("online_feature_or_teacher_call_count") != 0
            or row.get("scoring_constructed_after_all_sampling") is not True
            or row.get("history_rgb_frames") != 5
            or row.get("future_rgb_frames") != 8
            or row.get("actual_transformer_call_count") != endpoint.nfe
            or row.get("declared_nfe") != endpoint.nfe
            or row.get("history_video_latent_tokens") != 2
            or row.get("future_video_latent_tokens") != 2
            or not isinstance(hashes, Mapping)
            or any(
                not isinstance(hashes.get(field), str)
                or SHA256_RE.fullmatch(hashes[field]) is None
                for field in required_hashes
            )
            or key in observed
        ):
            raise TwoClockConsistencyEvaluationError("validation row violates protocol")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise TwoClockConsistencyEvaluationError("row metrics are absent")
        for metric in (
            "video_future_nmse",
            "video_future_temporal_delta_nmse",
            "decoded_mse_unit_range",
            "decoded_psnr_db",
            "decoded_temporal_difference_mse_unit_range",
        ):
            value = metrics.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TwoClockConsistencyEvaluationError(f"invalid row metric {metric}")
        observed[key] = row
    if set(observed) != expected:
        raise TwoClockConsistencyEvaluationError("validation row inventory is incomplete")
    invariant_hashes = (
        "cached_rgb_input_sha256",
        "sampler_history_rgb_sha256",
        "cached_actions_input_sha256",
        "video_clean_scoring_sha256",
        "raw_ground_truth_sha256",
        "raw_history_last_sha256",
        "video_initial_noise_sha256",
        "tf_initial_noise_sha256",
    )
    for clip_index in range(EXPECTED_VALIDATION_CLIPS):
        clip_rows = [
            observed[(clip_index, endpoint.code)] for endpoint in ENDPOINTS
        ]
        reference = clip_rows[0]["tensor_sha256"]
        if any(
            row["tensor_sha256"].get(field) != reference.get(field)
            for row in clip_rows[1:]
            for field in invariant_hashes
        ):
            raise TwoClockConsistencyEvaluationError(
                f"paired input/noise changed within arm for clip {clip_index}"
            )
        matched_action_hash = reference.get("sampler_actions_sha256")
        for nfe in NFE_GRID:
            row = observed[(clip_index, f"autonomous_nfe_{nfe}")]
            if row["tensor_sha256"].get("sampler_actions_sha256") != matched_action_hash:
                raise TwoClockConsistencyEvaluationError("matched action input changed across NFE")
        diagnostic = observed[(clip_index, "actions_shuffled_nfe_1")]
        if diagnostic.get("action_donor_sampling_id") in {
            None,
            diagnostic.get("sampling_id"),
        }:
            raise TwoClockConsistencyEvaluationError("action-shuffled donor is not distinct")
        if matched_action_hash != reference.get("cached_actions_input_sha256"):
            raise TwoClockConsistencyEvaluationError(
                "matched sampler actions differ from cache"
            )
        donor_index = _expected_action_donor_index(clip_index)
        donor_matched = observed[(donor_index, "autonomous_nfe_1")]
        if (
            diagnostic["tensor_sha256"].get("sampler_actions_sha256")
            != donor_matched["tensor_sha256"].get(
                "cached_actions_input_sha256"
            )
        ):
            raise TwoClockConsistencyEvaluationError(
                "shuffled action tensor differs from donor"
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
        raise TwoClockConsistencyEvaluationError("evaluation requires eight ranks")
    if args.batch_size_per_rank != EXPECTED_BATCH_SIZE_PER_RANK:
        raise TwoClockConsistencyEvaluationError("evaluation batch size is fixed at two")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if "B200" not in torch.cuda.get_device_properties(device).name.upper():
        raise TwoClockConsistencyEvaluationError("evaluation requires B200 GPUs")
    registration = _validate_registration(args.registration)
    if rank == 0:
        _clean_source(
            Path(registration["tool_repository"]["path"]),
            registration["tool_repository"]["git_commit"],
            "tool",
        )
        revalidate_registered_inputs(
            registration,
            include_parent=False,
            include_train=False,
            include_validation=True,
        )
    dist.barrier()
    arm = ARM_BY_CODE.get(args.arm)
    if arm is None:
        raise TwoClockConsistencyEvaluationError("unknown arm")
    output_root = Path(registration["output_root"])
    expected_output = output_root / "evaluation" / arm.code.lower()
    if args.output_dir.expanduser().absolute() != expected_output:
        raise TwoClockConsistencyEvaluationError("evaluation output path differs")
    run_dir = (output_root / "training" / arm.run_name).resolve(strict=True)
    assigned = _expected_rank_indexes(rank)
    if rank == 0:
        if expected_output.exists() or expected_output.is_symlink():
            raise TwoClockConsistencyEvaluationError("fresh evaluation output exists")
        expected_output.mkdir(parents=True, mode=0o700)
    dist.barrier()
    model, _config, arm_artifacts = _load_model(
        registration, arm, run_dir, device
    )
    dataset = _dataset(registration)
    rows: list[dict[str, Any]] = []
    hook_calls = 0
    total_calls = 0

    def count_calls(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal hook_calls
        hook_calls += 1

    hook = model.forward_model.register_forward_hook(count_calls)
    try:
        for start in range(0, len(assigned), EXPECTED_BATCH_SIZE_PER_RANK):
            indexes = assigned[start : start + EXPECTED_BATCH_SIZE_PER_RANK]
            samples = [dataset[index] for index in indexes]
            batch = phase._move_batch(samples, device)
            history = batch["rgb"][:, :5].clone()
            sampling_ids = batch["clip_index"] + VALIDATION_SAMPLE_ID_OFFSET
            shuffled_actions = batch["actions"].roll(1, dims=0)
            donor_ids = sampling_ids.reshape(-1).roll(1, dims=0)
            clip_ids = [
                str(registration["validation_descriptors"][index]["clip_id"])
                for index in indexes
            ]
            completed: list[
                tuple[
                    Endpoint,
                    Mapping[str, Any],
                    Mapping[str, Any],
                    Any,
                    Any,
                    list[int | None],
                    int,
                ]
            ] = []
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                prepared_matched = phase._prepare_deployable_rollout(
                    model,
                    history,
                    batch["actions"],
                    batch["morphology_index"],
                    sampling_ids,
                )
                prepared_shuffled = phase._prepare_deployable_rollout(
                    model,
                    history,
                    shuffled_actions,
                    batch["morphology_index"],
                    sampling_ids,
                )
                for endpoint in ENDPOINTS:
                    prepared = (
                        prepared_matched
                        if endpoint.action_source == "matched"
                        else prepared_shuffled
                    )
                    action_input = (
                        batch["actions"]
                        if endpoint.action_source == "matched"
                        else shuffled_actions
                    )
                    action_donors: list[int | None] = (
                        [None] * len(indexes)
                        if endpoint.action_source == "matched"
                        else [int(value) for value in donor_ids.detach().cpu().tolist()]
                    )
                    hook_calls = 0
                    result = phase._run_trajectory(
                        model, prepared, steps=endpoint.nfe
                    )
                    observed_calls = hook_calls
                    decoded = _uint8_future(
                        model,
                        result["video"],
                        (int(history.shape[-2]), int(history.shape[-1])),
                    )
                    completed.append(
                        (
                            endpoint,
                            result,
                            prepared,
                            decoded,
                            action_input,
                            action_donors,
                            observed_calls,
                        )
                    )
                    total_calls += observed_calls

            # Only now may the evaluator encode the complete clip for metrics.
            scoring = phase._scoring_targets(model, batch)
            for (
                endpoint,
                result,
                prepared,
                decoded,
                action_input,
                action_donors,
                observed_calls,
            ) in completed:
                rows.extend(
                    _endpoint_rows(
                        arm=arm,
                        endpoint=endpoint,
                        result=result,
                        prepared=prepared,
                        decoded=decoded,
                        scoring=scoring,
                        clip_indexes=indexes,
                        clip_ids=clip_ids,
                        sampling_ids=sampling_ids,
                        observed_calls=observed_calls,
                        registration=registration,
                        history_input=history,
                        action_input=action_input,
                        action_donor_ids=action_donors,
                        arm_artifacts=arm_artifacts,
                    )
                )
            del completed, scoring, prepared_matched, prepared_shuffled, batch, samples
    finally:
        hook.remove()

    expected_rows = len(assigned) * len(ENDPOINTS)
    expected_calls = (
        len(assigned)
        // EXPECTED_BATCH_SIZE_PER_RANK
        * sum(endpoint.nfe for endpoint in ENDPOINTS)
    )
    if len(rows) != expected_rows or total_calls != expected_calls:
        raise TwoClockConsistencyEvaluationError("rank row/call totals differ")
    rows_path = expected_output / f"rank_{rank:03d}.jsonl"
    rows_bytes = b"".join(_canonical_json(row) + b"\n" for row in rows)
    _exclusive_bytes(rows_path, rows_bytes)
    rank_manifest = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_RANK,
            "created_at_utc": _now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "arm": asdict(arm),
            "rank": rank,
            "world_size": world_size,
            "batch_size_per_rank": EXPECTED_BATCH_SIZE_PER_RANK,
            "assigned_clip_indexes": assigned,
            "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
            "actual_transformer_call_count": total_calls,
            "rows": {
                "path": str(rows_path),
                "bytes": len(rows_bytes),
                "sha256": hashlib.sha256(rows_bytes).hexdigest(),
                "count": len(rows),
            },
            "protected_test_accessed": False,
            "target_cache_array_opened": False,
        }
    )
    _exclusive_json(expected_output / f"rank_{rank:03d}.json", rank_manifest)
    dist.barrier()
    if rank == 0:
        global_rows: list[dict[str, Any]] = []
        rank_records = []
        for expected_rank in range(world_size):
            rank_path = expected_output / f"rank_{expected_rank:03d}.json"
            record = _read_json(rank_path, "rank manifest")
            row_path = _validate_rank_manifest(
                record,
                expected_rank=expected_rank,
                arm=arm,
                registration=registration,
                output_dir=expected_output,
            )
            with row_path.open(encoding="utf-8") as handle:
                global_rows.extend(json.loads(line) for line in handle if line.strip())
            rank_records.append(_file_record(rank_path))
        _validate_rows(global_rows, arm, registration)
        inventory = identity_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": KIND_INVENTORY,
                "created_at_utc": _now(),
                "registration": _file_record(args.registration),
                "registration_identity_sha256": registration["identity_sha256"],
                "arm": asdict(arm),
                "arm_artifacts": arm_artifacts,
                "evaluation_split": "validation",
                "validation_clips": EXPECTED_VALIDATION_CLIPS,
                "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
                "row_count": len(global_rows),
                "rank_manifests": rank_records,
                "paired_inputs_and_noise_within_arm": True,
                "actual_transformer_call_count": sum(
                    int(_read_json(Path(record["path"]), "rank")[
                        "actual_transformer_call_count"
                    ])
                    for record in rank_records
                ),
                "protected_test_accessed": False,
                "target_cache_array_opened": False,
                "online_feature_or_teacher_call_count": 0,
            }
        )
        _exclusive_json(expected_output / "inventory.json", inventory)
    dist.barrier()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--tool-repo", type=Path, required=True)
    register.add_argument("--historical-repo", type=Path, required=True)
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--study-root", type=Path, required=True)
    register.add_argument("--train-manifest", type=Path, required=True)
    register.add_argument("--train-cache-metadata", type=Path, required=True)
    register.add_argument("--python", type=Path, required=True)
    register.add_argument("--output-root", type=Path, required=True)
    register.set_defaults(func=command_register)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--registration", type=Path, required=True)
    evaluate.add_argument("--arm", choices=tuple(ARM_BY_CODE), required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument(
        "--batch-size-per-rank", type=int, default=EXPECTED_BATCH_SIZE_PER_RANK
    )
    evaluate.set_defaults(func=command_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
