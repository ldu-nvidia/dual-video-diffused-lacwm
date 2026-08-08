#!/usr/bin/env python3
"""Prospective contracts and analysis for the Haar Frequency-Forcing screen.

This module is stdlib-only so registration and launch validation do not import
the training stack.  It never submits a Slurm job and never accepts a test
split.  All candidate metrics are validation-only.
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
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "docs/experiments/VIDEO_FREQUENCY_FORCING_PROTOCOL.md"
EVALUATOR = REPO_ROOT / "tools/evaluate_frequency_forcing.py"
WARMSTART_PREFLIGHT = REPO_ROOT / "tools/validate_frequency_forcing_warmstart.py"
TRANSFORM = (
    REPO_ROOT / "robot_wm/modeling/dual_diffusion/haar_lowpass.py"
)
DATASET = REPO_ROOT / "robot_wm/datasets/abc/fixed_rgb_action_dataset.py"
MODEL_CONFIG = (
    REPO_ROOT
    / "projects/latent_action_models/configs/models/dual_explicit_action_dit_frequency.yaml"
)
COMMON_CONFIG = (
    REPO_ROOT
    / "projects/latent_action_models/configs/experiments_0908/frequency_forcing_common.yaml"
)
DATASET_CONFIG = (
    REPO_ROOT
    / "robot_wm/datasets/configs/datasets/transformed_fixed_abc_rgb_action.yaml"
)
TRAIN_SBATCH = REPO_ROOT / "tools/slurm/frequency_forcing_screen.sbatch"
EVALUATE_SBATCH = REPO_ROOT / "tools/slurm/frequency_forcing_evaluate.sbatch"

SCHEMA = "video-frequency-forcing-screen-v1"
ARM_SCHEMA = "video-frequency-forcing-arm-v1"
RESULT_SCHEMA = "video-frequency-forcing-validation-row-v1"
EVALUATION_COMPLETE_SCHEMA = "video-frequency-forcing-evaluation-complete-v1"
ANALYSIS_SCHEMA = "video-frequency-forcing-analysis-v1"
SEED = 1234
EVALUATION_SEED = 20_260_807
BOOTSTRAP_SEED = 20_260_807
TRAIN_UPDATES = 200
WORLD_SIZE = 8
MICRO_BATCH = 1
GLOBAL_BATCH = WORLD_SIZE * MICRO_BATCH
EXPECTED_TRAIN_CLIPS = 512
EXPECTED_VALIDATION_CLIPS = 64
TARGET_SHAPE = [6, 4, 24, 120]
NFE_GRID = [1, 2, 4, 8]
SOURCES = [
    "autonomous",
    "off",
    "autonomous_shuffled",
    "oracle_matched",
    "oracle_shuffled",
]
DEPLOYABLE_SOURCES = ["autonomous", "off", "autonomous_shuffled"]
WANDB_ENTITY = "zijiandu"
WANDB_PROJECT = "dual-video-diffusion-private"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")


class ContractError(RuntimeError):
    """A frozen input or prospective comparison violated the protocol."""


@dataclass(frozen=True)
class Arm:
    code: str
    slug: str
    selector: str
    condition_mode: str
    condition_on_state: bool
    condition_on_clock: bool
    schedule_mode: str
    lead_logit: float
    auxiliary_loss_weight: float
    state_gate_init: float
    clock_gate_init: float
    parameter_matched_control: bool
    estimand: str


@dataclass(frozen=True)
class ManifestIdentity:
    """Exact clip and episode identities used to prove split isolation."""

    clip_ids: frozenset[str]
    episode_dirs: frozenset[str]
    episode_starts: frozenset[tuple[str, int]]
    summary: Mapping[str, Any]


ARMS = (
    Arm(
        "FPM",
        "fpm_video_only",
        "ravenhuang/wan-dit/frequency_ff_fpm.yaml",
        "off",
        False,
        False,
        "aligned",
        0.0,
        0.0,
        0.0,
        0.0,
        True,
        "same-schema parameter-matched continued-training baseline",
    ),
    Arm(
        "FAUX",
        "faux_multitask_only",
        "ravenhuang/wan-dit/frequency_ff_aux.yaml",
        "off",
        False,
        False,
        "aligned",
        0.0,
        1.0,
        0.0,
        0.0,
        False,
        "auxiliary multitask regularization without video fusion",
    ),
    Arm(
        "FSYNC",
        "fsync_joint_aligned",
        "ravenhuang/wan-dit/frequency_ff_sync.yaml",
        "matched",
        True,
        True,
        "aligned",
        0.0,
        1.0,
        0.02,
        0.02,
        False,
        "synchronous joint low-frequency denoising and fusion",
    ),
    Arm(
        "FLEAD",
        "flead_joint_leading",
        "ravenhuang/wan-dit/frequency_ff_lead.yaml",
        "matched",
        True,
        True,
        "tf_leads",
        1.0,
        1.0,
        0.02,
        0.02,
        False,
        "same joint model with an earlier-maturing auxiliary clock",
    ),
)
ARM_BY_CODE = {arm.code: arm for arm in ARMS}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    return {
        **value,
        "identity_sha256": hashlib.sha256(canonical_json(value)).hexdigest(),
    }


def validate_identity(value: Mapping[str, Any]) -> bool:
    recorded = value.get("identity_sha256")
    if not isinstance(recorded, str) or not SHA_RE.fullmatch(recorded):
        return False
    unsigned = dict(value)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest() == recorded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink():
        raise ContractError(f"{label} must be an absolute non-symlink file")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ContractError(f"{label} must be a canonical nonempty file: {path}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def executable_record(path: Path, label: str) -> dict[str, Any]:
    """Record a venv launcher without resolving away its environment context."""
    if not path.is_absolute() or not path.exists() or not os.access(path, os.X_OK):
        raise ContractError(f"{label} must be an absolute executable")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ContractError(f"{label} does not resolve to a regular file")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def source_file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if REPO_ROOT not in resolved.parents or resolved.is_symlink():
        raise ContractError(f"implementation file is outside the repository: {path}")
    return {
        "path": str(resolved.relative_to(REPO_ROOT)),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain one JSON object")
    return value


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(json.dumps(value, indent=2, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def git_output(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ContractError(f"git command failed: {' '.join(args)}") from exc


def validate_source(expected_commit: str) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise ContractError("expected commit must be a full lowercase SHA")
    if Path(git_output("rev-parse", "--show-toplevel")).resolve() != REPO_ROOT:
        raise ContractError("helper is not running from its own repository")
    observed = git_output("rev-parse", "HEAD")
    if observed != expected_commit:
        raise ContractError(f"HEAD differs: {observed} != {expected_commit}")
    if git_output("status", "--porcelain", "--untracked-files=all"):
        raise ContractError("registration/execution requires a clean worktree")
    return {"root": str(REPO_ROOT), "commit": observed, "clean": True}


def _manifest_identity(
    rows: Sequence[Any],
    *,
    split: str,
    expected_clips: int,
) -> ManifestIdentity:
    if len(rows) != expected_clips:
        raise ContractError(
            f"{split} manifest has {len(rows)} rows, expected {expected_clips}"
        )
    clip_ids: set[str] = set()
    episode_dirs: set[str] = set()
    episode_starts: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ContractError(f"{split} manifest row {index} is not an object")
        clip_id = row.get("clip_id")
        episode_dir = row.get("episode_dir")
        start = row.get("start")
        auxiliary_index = row.get("auxiliary_index")
        if (
            not isinstance(clip_id, str)
            or SHA_RE.fullmatch(clip_id) is None
            or not isinstance(episode_dir, str)
            or not episode_dir
            or isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or isinstance(auxiliary_index, bool)
            or not isinstance(auxiliary_index, int)
            or auxiliary_index != index
            or row.get("split") != split
        ):
            raise ContractError(
                f"{split} manifest row {index} violates its identity/split contract"
            )
        episode_start = (episode_dir, start)
        if clip_id in clip_ids or episode_start in episode_starts:
            raise ContractError(f"{split} manifest repeats a clip identity")
        # This screen bootstraps clip-level intervals. Requiring one clip per
        # episode prevents treating correlated windows as independent samples.
        if episode_dir in episode_dirs:
            raise ContractError(f"{split} manifest repeats an episode")
        clip_ids.add(clip_id)
        episode_dirs.add(episode_dir)
        episode_starts.add(episode_start)
    summary = {
        "unique_clip_ids": len(clip_ids),
        "unique_episode_dirs": len(episode_dirs),
        "unique_episode_starts": len(episode_starts),
        "clip_ids_sha256": hashlib.sha256(
            canonical_json(sorted(clip_ids))
        ).hexdigest(),
        "episode_dirs_sha256": hashlib.sha256(
            canonical_json(sorted(episode_dirs))
        ).hexdigest(),
        "episode_starts_sha256": hashlib.sha256(
            canonical_json(sorted([list(value) for value in episode_starts]))
        ).hexdigest(),
    }
    return ManifestIdentity(
        clip_ids=frozenset(clip_ids),
        episode_dirs=frozenset(episode_dirs),
        episode_starts=frozenset(episode_starts),
        summary=summary,
    )


def _metadata_record(
    manifest_path: Path,
    metadata_path: Path,
    *,
    split: str,
    expected_clips: int,
) -> tuple[dict[str, Any], ManifestIdentity]:
    manifest = file_record(manifest_path, f"{split} clip manifest")
    metadata_record = file_record(metadata_path, f"{split} cache metadata")
    metadata = read_json(metadata_path, f"{split} cache metadata")
    try:
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as exc:
        raise ContractError(f"{split} clip manifest contains invalid JSON") from exc
    identity = _manifest_identity(rows, split=split, expected_clips=expected_clips)
    if (
        metadata.get("split") != split
        or metadata.get("clip_count") != expected_clips
        or metadata.get("clip_manifest_sha256") != manifest["sha256"]
        or metadata.get("rgb_shape")
        != [expected_clips, 13, 3, 180, 960]
        or metadata.get("actions_shape")
        != [expected_clips, 13, 5, 23]
        or metadata.get("rgb_dtype") != "float16"
        or metadata.get("actions_dtype") != "float32"
        or metadata.get("complete") is not True
    ):
        raise ContractError(f"{split} immutable RGB/action cache contract changed")

    arrays = {}
    for name, file_key, sha_key, bytes_per_value in (
        ("rgb", "rgb_file", "rgb_sha256", 2),
        ("actions", "actions_file", "actions_sha256", 4),
    ):
        raw_path = Path(str(metadata.get(file_key, "")))
        if not raw_path.is_absolute():
            raw_path = metadata_path.parent / raw_path
        path = raw_path.resolve(strict=True)
        digest = metadata.get(sha_key)
        if not isinstance(digest, str) or SHA_RE.fullmatch(digest) is None:
            raise ContractError(f"{split} metadata lacks a valid {name} digest")
        shape = metadata[f"{name}_shape"]
        expected_bytes = math.prod(int(value) for value in shape) * bytes_per_value
        # NumPy .npy adds a small header; exact on-disk bytes are recorded, while
        # the lower bound detects truncation without rehashing multi-GB arrays.
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size < expected_bytes
        ):
            raise ContractError(f"{split} {name} cache is missing or truncated")
        arrays[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "metadata_sha256": digest,
            "large_array_rehash": False,
            "digest_inherited_from_fully_hashed_source_cache": True,
        }
    record = {
        "split": split,
        "clips": expected_clips,
        "manifest": manifest,
        "metadata": metadata_record,
        "arrays": arrays,
        "manifest_identity": identity.summary,
        "protected_test": False,
    }
    return record, identity


def _split_isolation(
    train: ManifestIdentity,
    validation: ManifestIdentity,
) -> dict[str, Any]:
    overlaps = {
        "clip_ids": sorted(train.clip_ids & validation.clip_ids),
        "episode_dirs": sorted(train.episode_dirs & validation.episode_dirs),
        "episode_starts": sorted(
            [
                list(value)
                for value in (train.episode_starts & validation.episode_starts)
            ]
        ),
    }
    if any(overlaps.values()):
        raise ContractError(
            "train/validation manifests overlap by clip or episode identity"
        )
    return {
        "clip_id_overlap": 0,
        "episode_dir_overlap": 0,
        "episode_start_overlap": 0,
        "one_clip_per_episode": True,
    }


def _arm(task_id: int) -> Arm:
    if task_id < 0 or task_id >= len(ARMS):
        raise ContractError(f"array task must be in [0,{len(ARMS) - 1}]")
    return ARMS[task_id]


def command_arm(args: argparse.Namespace) -> int:
    arm = _arm(args.array_task_id)
    values = asdict(arm)
    if args.format == "json":
        print(json.dumps(values, sort_keys=True))
    else:
        fields = [
            arm.code,
            arm.slug,
            arm.selector,
            arm.condition_mode,
            str(arm.condition_on_state).lower(),
            str(arm.condition_on_clock).lower(),
            arm.schedule_mode,
            str(arm.lead_logit),
            str(arm.auxiliary_loss_weight),
            str(arm.state_gate_init),
            str(arm.clock_gate_init),
            str(arm.parameter_matched_control).lower(),
        ]
        if any("|" in field or "\n" in field for field in fields):
            raise ContractError("arm field contains an unsafe delimiter")
        print("|".join(fields))
    return 0


def command_register(args: argparse.Namespace) -> int:
    expected_commit = args.expected_commit
    source = validate_source(expected_commit)
    study_root = args.study_root
    if not study_root.is_absolute() or study_root.is_symlink():
        raise ContractError("study root must be an absolute non-symlink path")
    if study_root.exists():
        raise ContractError("prospective study root must not already exist")
    if not study_root.parent.resolve(strict=True).is_dir():
        raise ContractError("study parent is missing")
    if SAFE_ID_RE.fullmatch(study_root.name) is None:
        raise ContractError("study ID is not path-safe")

    runtime_python = executable_record(args.python, "runtime Python")
    wan_dir = args.wan_dir.resolve(strict=True)
    videox_home = args.videox_home.resolve(strict=True)
    if not wan_dir.is_dir() or not videox_home.is_dir():
        raise ContractError("Wan and VideoX-Fun paths must be directories")
    warmstart = file_record(args.warmstart, "historical VPM warm start")
    if args.warmstart_sha256 and warmstart["sha256"] != args.warmstart_sha256:
        raise ContractError("historical VPM warm-start digest differs")
    train, train_identity = _metadata_record(
        args.train_manifest,
        args.train_metadata,
        split="train",
        expected_clips=EXPECTED_TRAIN_CLIPS,
    )
    validation, validation_identity = _metadata_record(
        args.validation_manifest,
        args.validation_metadata,
        split="val",
        expected_clips=EXPECTED_VALIDATION_CLIPS,
    )
    split_isolation = _split_isolation(train_identity, validation_identity)
    if set(arm.code for arm in ARMS) != set(ARM_BY_CODE):
        raise ContractError("arm codes are not unique")

    implementation_paths = [
        Path(__file__),
        PROTOCOL,
        EVALUATOR,
        WARMSTART_PREFLIGHT,
        TRANSFORM,
        DATASET,
        MODEL_CONFIG,
        COMMON_CONFIG,
        DATASET_CONFIG,
        TRAIN_SBATCH,
        EVALUATE_SBATCH,
        REPO_ROOT / "projects/latent_action_models/train.py",
        REPO_ROOT
        / "projects/latent_action_models/lam/dual_explicit_action_dit_model.py",
        REPO_ROOT / "robot_wm/modeling/networks/wan_forward_model.py",
        REPO_ROOT / "robot_wm/modeling/dual_diffusion/adapters.py",
        REPO_ROOT / "robot_wm/modeling/dual_diffusion/conditioning.py",
        REPO_ROOT / "robot_wm/modeling/dual_diffusion/flow.py",
        REPO_ROOT / "robot_wm/utils/trainer.py",
    ] + [
        REPO_ROOT
        / "projects/latent_action_models/configs/experiments_0908"
        / arm.selector
        for arm in ARMS
    ]
    implementation = {
        str(path.relative_to(REPO_ROOT)): source_file_record(path)
        for path in implementation_paths
    }
    payload = identity_payload(
        {
            "schema": SCHEMA,
            "study_id": study_root.name,
            "source": source,
            "study_root": str(study_root),
            "runtime": {
                "python": runtime_python,
                "wan_dir": str(wan_dir),
                "videox_home": str(videox_home),
            },
            "warm_start": {
                **warmstart,
                "role": "common model-only initialization; not a reused metric baseline",
                "excluded_prefixes": [
                    "forward_model.tf_token_adapter",
                    "forward_model.tf_clock_embedding",
                    "forward_model.tf_velocity_head",
                ],
                "historical_schema_channels": 64,
                "new_schema_channels": 6,
            },
            "data": {
                "train": train,
                "validation": validation,
                "split_isolation": split_isolation,
            },
            "representation": {
                "name": "per-view spatial Haar low-pass plus temporal DC/coarse motion",
                "input_shape": ["B", 13, 3, 180, 960],
                "target_shape": ["B", *TARGET_SHAPE],
                "channel_order": ["R_dc", "G_dc", "B_dc", "R_motion", "G_motion", "B_motion"],
                "learned_encoder": False,
                "external_checkpoint": False,
                "autonomous_transform_calls": 0,
                "all_auxiliary_bins_start_from_noise": True,
            },
            "training": {
                "seed": SEED,
                "updates": TRAIN_UPDATES,
                "world_size": WORLD_SIZE,
                "micro_batch_per_rank": MICRO_BATCH,
                "global_batch": GLOBAL_BATCH,
                "same_optimizer_and_data_order": True,
                "same_non_auxiliary_warm_start": True,
                "arms": [asdict(arm) for arm in ARMS],
            },
            "evaluation": {
                "split": "validation",
                "clips": EXPECTED_VALIDATION_CLIPS,
                "nfe": NFE_GRID,
                "sources": SOURCES,
                "deployable_sources": DEPLOYABLE_SOURCES,
                "same_total_wan_calls_at_each_nfe": True,
                "teacher_calls": 0,
                "clean_future_feature_available_to_deployable_sampler": False,
                "oracle_is_leakage_diagnostic_only": True,
                "bootstrap_replicates": 10_000,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            "decision": {
                "primary_nfe": [1, 2, 4],
                "minimum_relative_gain": 0.03,
                "guardrail_max_regression": 0.01,
                "mechanism_requires_autonomous_better_than_both_off_and_shuffled": True,
                "leading_requires_flead_better_than_fsync": True,
                "no_protected_test": True,
            },
            "wandb": {
                "entity": WANDB_ENTITY,
                "project": WANDB_PROJECT,
                "group": None,
            },
            "implementation": implementation,
            "protected_test": {
                "accepted_by_registration_cli": False,
                "accessed": False,
                "allowed": False,
            },
        }
    )
    study_root.mkdir(mode=0o700)
    exclusive_json(study_root / "protocol_registration.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def load_registration(path: Path, *, verify_files: bool = True) -> dict[str, Any]:
    registration = read_json(path.resolve(strict=True), "registration")
    if registration.get("schema") != SCHEMA or not validate_identity(registration):
        raise ContractError("registration schema or identity is invalid")
    if registration.get("protected_test", {}).get("allowed") is not False:
        raise ContractError("registration does not fail closed on protected test")
    if verify_files:
        validate_source(str(registration["source"]["commit"]))
        for record in registration["implementation"].values():
            path = REPO_ROOT / record["path"]
            if (
                path.stat().st_size != record["bytes"]
                or sha256_file(path) != record["sha256"]
            ):
                raise ContractError(f"registered implementation changed: {path}")
        for split in ("train", "validation"):
            for label in ("manifest", "metadata"):
                record = registration["data"][split][label]
                path = Path(record["path"])
                if (
                    path.stat().st_size != record["bytes"]
                    or sha256_file(path) != record["sha256"]
                ):
                    raise ContractError(f"registered {split} {label} changed")
        warm = registration["warm_start"]
        warm_path = Path(warm["path"])
        if warm_path.stat().st_size != warm["bytes"]:
            raise ContractError("warm-start byte count changed")
        # The 4.2 GB checkpoint is fully hashed at registration. Execution
        # validates its digest again once per allocation before model loading.
        if sha256_file(warm_path) != warm["sha256"]:
            raise ContractError("warm-start digest changed")
        runtime = registration["runtime"]["python"]
        runtime_resolved = Path(runtime["path"]).resolve(strict=True)
        if (
            str(runtime_resolved) != runtime["resolved_path"]
            or runtime_resolved.stat().st_size != runtime["bytes"]
            or sha256_file(runtime_resolved) != runtime["sha256"]
        ):
            raise ContractError("registered runtime Python changed")
    return registration


def command_verify(args: argparse.Namespace) -> int:
    registration = load_registration(args.registration, verify_files=True)
    print(
        json.dumps(
            {
                "status": "valid",
                "study_id": registration["study_id"],
                "identity_sha256": registration["identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def command_runtime(args: argparse.Namespace) -> int:
    registration = load_registration(args.registration, verify_files=False)
    values = [
        registration["source"]["commit"],
        registration["study_root"],
        registration["runtime"]["python"]["path"],
        registration["runtime"]["wan_dir"],
        registration["runtime"]["videox_home"],
        registration["warm_start"]["path"],
        registration["data"]["train"]["manifest"]["path"],
        registration["data"]["train"]["metadata"]["path"],
        registration["data"]["validation"]["manifest"]["path"],
        registration["data"]["validation"]["metadata"]["path"],
        registration["identity_sha256"],
    ]
    if any("|" in str(value) or "\n" in str(value) for value in values):
        raise ContractError("runtime field contains an unsafe delimiter")
    print("|".join(str(value) for value in values))
    return 0


def command_bind_arm(args: argparse.Namespace) -> int:
    registration = load_registration(args.registration, verify_files=False)
    arm = _arm(args.array_task_id)
    study_root = Path(registration["study_root"])
    expected = study_root / "runs" / arm.slug
    if args.run_dir != expected:
        raise ContractError(f"run directory must be {expected}")
    if args.run_dir.exists():
        raise ContractError("fresh arm run directory already exists")
    args.run_dir.mkdir(parents=True, mode=0o700)
    payload = identity_payload(
        {
            "schema": ARM_SCHEMA,
            "registration_identity_sha256": registration["identity_sha256"],
            "array_task_id": args.array_task_id,
            "arm": asdict(arm),
            "run_dir": str(args.run_dir),
            "run_id": f"{registration['study_id']}-{arm.code}",
            "expected_completed_updates": TRAIN_UPDATES,
            "slurm_array_job_id": args.slurm_array_job_id,
        }
    )
    exclusive_json(args.run_dir / "arm_manifest.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _percent_improvement(candidate: float, reference: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(reference) or reference <= 0:
        raise ContractError("metric means must be finite and reference positive")
    return (reference - candidate) / reference


def _bootstrap_improvement(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    seed: int,
    replicates: int = 10_000,
) -> tuple[float, float]:
    if len(candidate) != len(reference) or len(candidate) < 2:
        raise ContractError("paired bootstrap requires equal nontrivial samples")
    # Import NumPy only for offline analysis; registration/launch remain stdlib.
    import numpy as np

    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 512):
        count = min(512, replicates - start)
        indexes = rng.integers(0, len(cand), size=(count, len(cand)))
        cand_mean = cand[indexes].mean(axis=1)
        ref_mean = ref[indexes].mean(axis=1)
        values[start : start + count] = (ref_mean - cand_mean) / ref_mean
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _validate_nfe1_shuffle_negative_control(
    keyed: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
) -> None:
    """At pure noise, matched and residual-shuffled states must be identical."""
    fields = (
        "video_final_sha256",
        "auxiliary_final_sha256",
        "video_nmse",
        "decoded_mse",
        "temporal_mse",
        "auxiliary_future_nmse",
        "auxiliary_future_cosine",
        "auxiliary_dc_nmse",
        "auxiliary_motion_nmse",
    )
    for arm in ARMS:
        for clip in range(EXPECTED_VALIDATION_CLIPS):
            autonomous = keyed[(arm.code, "autonomous", 1, clip)]
            shuffled = keyed[(arm.code, "autonomous_shuffled", 1, clip)]
            changed = [
                field
                for field in fields
                if autonomous.get(field) != shuffled.get(field)
            ]
            if changed:
                raise ContractError(
                    "NFE-1 pure-noise shuffle negative control changed "
                    f"{arm.code} clip {clip}: {changed}"
                )


def _validate_evaluation_receipt(
    rows_path: Path,
    *,
    registration_identity_sha256: str,
) -> dict[str, Any]:
    """Bind one immutable result table to its rank-zero completion receipt."""
    if not rows_path.is_absolute() or rows_path.is_symlink():
        raise ContractError("evaluation rows must be an absolute non-symlink file")
    resolved_rows = rows_path.resolve(strict=True)
    if resolved_rows != rows_path or not resolved_rows.is_file():
        raise ContractError("evaluation rows must be a canonical regular file")
    receipt_path = rows_path.parent / "complete.json"
    if receipt_path.is_symlink():
        raise ContractError("evaluation receipt must not be a symlink")
    receipt = read_json(
        receipt_path.resolve(strict=True),
        f"evaluation receipt for {rows_path}",
    )
    expected_rows = (
        EXPECTED_VALIDATION_CLIPS * len(NFE_GRID) * len(SOURCES)
    )
    arm = receipt.get("arm")
    arm_identity = receipt.get("arm_identity_sha256")
    if (
        receipt.get("schema") != EVALUATION_COMPLETE_SCHEMA
        or not validate_identity(receipt)
        or receipt.get("registration_identity_sha256")
        != registration_identity_sha256
        or arm not in ARM_BY_CODE
        or not isinstance(arm_identity, str)
        or SHA_RE.fullmatch(arm_identity) is None
        or receipt.get("rows") != expected_rows
        or receipt.get("validation_clips") != EXPECTED_VALIDATION_CLIPS
        or receipt.get("nfe") != NFE_GRID
        or receipt.get("sources") != SOURCES
        or receipt.get("world_size") != WORLD_SIZE
        or receipt.get("rows_sha256") != sha256_file(resolved_rows)
        or receipt.get("protected_test_accessed") is not False
    ):
        raise ContractError(f"invalid evaluation completion receipt: {receipt_path}")
    return receipt


def command_analyze(args: argparse.Namespace) -> int:
    registration = load_registration(args.registration, verify_files=False)
    rows = []
    receipts: dict[str, dict[str, Any]] = {}
    for path in args.rows:
        receipt = _validate_evaluation_receipt(
            path,
            registration_identity_sha256=registration["identity_sha256"],
        )
        receipt_arm = str(receipt["arm"])
        if receipt_arm in receipts:
            raise ContractError(f"duplicate evaluation receipt for {receipt_arm}")
        receipts[receipt_arm] = receipt
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if (
                        row.get("schema") != RESULT_SCHEMA
                        or not validate_identity(row)
                        or row.get("registration_identity_sha256")
                        != registration["identity_sha256"]
                        or row.get("arm") != receipt_arm
                        or row.get("arm_identity_sha256")
                        != receipt["arm_identity_sha256"]
                    ):
                        raise ContractError(f"invalid result row in {path}")
                    rows.append(row)
    if set(receipts) != set(ARM_BY_CODE):
        raise ContractError("evaluation receipts do not cover all frozen arms")
    expected = len(ARMS) * len(SOURCES) * len(NFE_GRID) * EXPECTED_VALIDATION_CLIPS
    if len(rows) != expected:
        raise ContractError(f"result grid has {len(rows)} rows, expected {expected}")
    keyed: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["arm"]),
            str(row["source"]),
            int(row["nfe"]),
            int(row["clip_index"]),
        )
        if key in keyed:
            raise ContractError(f"duplicate result cell: {key}")
        deployable = row.get("source") in DEPLOYABLE_SOURCES
        if (
            row.get("actual_wan_calls") != row.get("nfe")
            or row.get("hook_wan_calls") != row.get("nfe")
            or row.get("sampler_transform_calls") != 0
            or row.get("online_teacher_calls") != 0
            or row.get("protected_test_accessed") is not False
            or row.get("deployable") is not deployable
            or (
                deployable
                and (
                    row.get("clean_auxiliary_passed_to_sampler") is not False
                    or row.get("future_rgb_passed_to_sampler") is not False
                )
            )
            or row.get("all_auxiliary_bins_initialized_from_noise") is not True
        ):
            raise ContractError(f"sampler audit failed for result cell: {key}")
        keyed[key] = row

    _validate_nfe1_shuffle_negative_control(keyed)

    for clip in range(EXPECTED_VALIDATION_CLIPS):
        clip_rows = [row for row in rows if int(row["clip_index"]) == clip]
        for field in (
            "video_initial_sha256",
            "auxiliary_initial_sha256",
            "raw_target_sha256",
        ):
            values = {str(row[field]) for row in clip_rows}
            if len(values) != 1:
                raise ContractError(
                    f"paired {field} differs across arms/cells for clip {clip}"
                )
    for arm_code in ("FPM", "FAUX"):
        arm_rows = [row for row in rows if row["arm"] == arm_code]
        if any(
            float(row["effective_state_gate"]) != 0.0
            or float(row["effective_clock_gate"]) != 0.0
            for row in arm_rows
        ):
            raise ContractError(f"{arm_code} no-fusion gates are not exact zero")

    metrics = ("video_nmse", "decoded_mse", "temporal_mse")
    summaries = []
    comparisons = []
    for arm in ARMS:
        for source in SOURCES:
            for nfe in NFE_GRID:
                cell = [keyed[(arm.code, source, nfe, clip)] for clip in range(64)]
                summaries.append(
                    {
                        "arm": arm.code,
                        "source": source,
                        "nfe": nfe,
                        **{
                            metric: sum(float(row[metric]) for row in cell) / len(cell)
                            for metric in metrics
                        },
                    }
                )

    def compare(
        candidate_arm: str,
        candidate_source: str,
        reference_arm: str,
        reference_source: str,
        nfe: int,
        label: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "label": label,
            "candidate": [candidate_arm, candidate_source, nfe],
            "reference": [reference_arm, reference_source, nfe],
        }
        for metric_index, metric in enumerate(metrics):
            candidate = [
                float(keyed[(candidate_arm, candidate_source, nfe, clip)][metric])
                for clip in range(64)
            ]
            reference = [
                float(keyed[(reference_arm, reference_source, nfe, clip)][metric])
                for clip in range(64)
            ]
            point = _percent_improvement(
                sum(candidate) / len(candidate), sum(reference) / len(reference)
            )
            low, high = _bootstrap_improvement(
                candidate,
                reference,
                seed=BOOTSTRAP_SEED + 100 * nfe + metric_index,
            )
            result[metric] = {
                "relative_improvement": point,
                "ci95": [low, high],
            }
        return result

    for nfe in NFE_GRID:
        comparisons.extend(
            [
                compare("FAUX", "autonomous", "FPM", "autonomous", nfe, "multitask_vs_video"),
                compare(
                    "FSYNC",
                    "autonomous",
                    "FAUX",
                    "autonomous",
                    nfe,
                    "fusion_package_vs_multitask",
                ),
                compare(
                    "FLEAD",
                    "autonomous",
                    "FSYNC",
                    "autonomous",
                    nfe,
                    "leading_vs_synchronous",
                ),
                compare("FSYNC", "autonomous", "FPM", "autonomous", nfe, "fsync_vs_fpm"),
                compare("FLEAD", "autonomous", "FPM", "autonomous", nfe, "flead_vs_fpm"),
                compare(
                    "FSYNC",
                    "autonomous",
                    "FSYNC",
                    "off",
                    nfe,
                    "fsync_generated_state_vs_off",
                ),
                compare(
                    "FSYNC",
                    "autonomous",
                    "FSYNC",
                    "autonomous_shuffled",
                    nfe,
                    "fsync_alignment_vs_shuffled",
                ),
                compare(
                    "FLEAD",
                    "autonomous",
                    "FLEAD",
                    "off",
                    nfe,
                    "flead_generated_state_vs_off",
                ),
                compare(
                    "FLEAD",
                    "autonomous",
                    "FLEAD",
                    "autonomous_shuffled",
                    nfe,
                    "flead_alignment_vs_shuffled",
                ),
                compare(
                    "FLEAD",
                    "oracle_matched",
                    "FLEAD",
                    "oracle_shuffled",
                    nfe,
                    "flead_oracle_alignment_headroom",
                ),
            ]
        )

    candidate_gates = []
    for arm_code in ("FSYNC", "FLEAD"):
        for nfe in (1, 2, 4):
            baseline = next(
                item
                for item in comparisons
                if item["label"] == f"{arm_code.lower()}_vs_fpm"
                and item["candidate"][2] == nfe
            )
            off = next(
                item for item in comparisons
                if item["label"] == f"{arm_code.lower()}_generated_state_vs_off"
                and item["candidate"][2] == nfe
            )
            shuffled = next(
                item for item in comparisons
                if item["label"] == f"{arm_code.lower()}_alignment_vs_shuffled"
                and item["candidate"][2] == nfe
            )
            quality = all(
                baseline[metric]["relative_improvement"] >= 0.03
                for metric in metrics
            )
            guardrails = all(
                baseline[metric]["ci95"][0] > -0.01 for metric in metrics
            )
            mechanism = (
                off["video_nmse"]["ci95"][0] > 0
                and shuffled["video_nmse"]["ci95"][0] > 0
            )
            leading_schedule = True
            if arm_code == "FLEAD":
                leading = next(
                    item
                    for item in comparisons
                    if item["label"] == "leading_vs_synchronous"
                    and item["candidate"][2] == nfe
                )
                leading_schedule = (
                    leading["video_nmse"]["ci95"][0] > 0
                    and leading["decoded_mse"]["ci95"][0] > -0.01
                    and leading["temporal_mse"]["ci95"][0] > -0.01
                )
            candidate_gates.append(
                {
                    "arm": arm_code,
                    "nfe": nfe,
                    "quality_point_3pct_all_metrics": quality,
                    "guardrails": guardrails,
                    "generated_alignment_mechanism": mechanism,
                    "leading_schedule_supported": leading_schedule,
                    "passes": (
                        quality
                        and guardrails
                        and mechanism
                        and leading_schedule
                    ),
                    "vs_fpm": baseline,
                }
            )

    payload = identity_payload(
        {
            "schema": ANALYSIS_SCHEMA,
            "registration_identity_sha256": registration["identity_sha256"],
            "evaluation_receipts": {
                arm_code: {
                    "identity_sha256": receipts[arm_code]["identity_sha256"],
                    "arm_identity_sha256": receipts[arm_code][
                        "arm_identity_sha256"
                    ],
                    "rows_sha256": receipts[arm_code]["rows_sha256"],
                    "rows": receipts[arm_code]["rows"],
                }
                for arm_code in sorted(receipts)
            },
            "rows": len(rows),
            "summaries": summaries,
            "comparisons": comparisons,
            "candidate_gates": candidate_gates,
            "passing_candidates": [
                [item["arm"], item["nfe"]]
                for item in candidate_gates if item["passes"]
            ],
            "protected_test_accessed": False,
        }
    )
    exclusive_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    arm = sub.add_parser("arm")
    arm.add_argument("--array-task-id", type=int, required=True)
    arm.add_argument("--format", choices=("json", "pipe"), default="json")
    arm.set_defaults(func=command_arm)

    register = sub.add_parser("register")
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--study-root", type=Path, required=True)
    register.add_argument("--python", type=Path, required=True)
    register.add_argument("--wan-dir", type=Path, required=True)
    register.add_argument("--videox-home", type=Path, required=True)
    register.add_argument("--warmstart", type=Path, required=True)
    register.add_argument("--warmstart-sha256")
    register.add_argument("--train-manifest", type=Path, required=True)
    register.add_argument("--train-metadata", type=Path, required=True)
    register.add_argument("--validation-manifest", type=Path, required=True)
    register.add_argument("--validation-metadata", type=Path, required=True)
    register.set_defaults(func=command_register)

    verify = sub.add_parser("verify")
    verify.add_argument("--registration", type=Path, required=True)
    verify.set_defaults(func=command_verify)

    runtime = sub.add_parser("runtime")
    runtime.add_argument("--registration", type=Path, required=True)
    runtime.set_defaults(func=command_runtime)

    bind = sub.add_parser("bind-arm")
    bind.add_argument("--registration", type=Path, required=True)
    bind.add_argument("--array-task-id", type=int, required=True)
    bind.add_argument("--run-dir", type=Path, required=True)
    bind.add_argument("--slurm-array-job-id", required=True)
    bind.set_defaults(func=command_bind_arm)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--registration", type=Path, required=True)
    analyze.add_argument("--rows", type=Path, nargs=4, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.set_defaults(func=command_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
