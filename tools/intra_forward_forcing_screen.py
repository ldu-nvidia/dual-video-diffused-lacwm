#!/usr/bin/env python3
"""Prospective contracts and analysis for one-call intra-forward forcing.

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
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "docs/experiments/VPM_INTRA_FORWARD_LATENT_FORCING_PROTOCOL.md"
RUNBOOK = REPO_ROOT / "docs/experiments/VPM_INTRA_FORWARD_LATENT_FORCING_RUNBOOK.md"
EVALUATOR = REPO_ROOT / "tools/evaluate_intra_forward_forcing.py"
WARMSTART_PREFLIGHT = REPO_ROOT / "tools/validate_intra_forward_warmstart.py"
TRANSFORM = REPO_ROOT / "robot_wm/modeling/dual_diffusion/haar_lowpass.py"
DATASET = REPO_ROOT / "robot_wm/datasets/abc/fixed_rgb_action_dataset.py"
MODEL_CONFIG = (
    REPO_ROOT
    / "projects/latent_action_models/configs/models/dual_explicit_action_dit_intra_forward.yaml"
)
COMMON_CONFIG = (
    REPO_ROOT
    / "projects/latent_action_models/configs/experiments_0908/intra_forward_forcing_common.yaml"
)
DATASET_CONFIG = (
    REPO_ROOT
    / "robot_wm/datasets/configs/datasets/transformed_fixed_abc_rgb_action.yaml"
)
TRAIN_SBATCH = REPO_ROOT / "tools/slurm/intra_forward_forcing_screen.sbatch"
EVALUATE_SBATCH = REPO_ROOT / "tools/slurm/intra_forward_forcing_evaluate.sbatch"
MEMORY_SMOKE = REPO_ROOT / "tools/smoke_intra_forward_memory.py"
MEMORY_SMOKE_SBATCH = (
    REPO_ROOT / "tools/slurm/intra_forward_forcing_memory_smoke.sbatch"
)
SUBMIT_SCRIPT = REPO_ROOT / "tools/slurm/submit_intra_forward_forcing_screen.sh"

SCHEMA = "video-intra-forward-forcing-screen-v2"
ARM_SCHEMA = "video-intra-forward-forcing-arm-v1"
RESULT_SCHEMA = "video-intra-forward-forcing-validation-row-v2"
EVALUATION_COMPLETE_SCHEMA = "video-intra-forward-forcing-evaluation-complete-v2"
ANALYSIS_SCHEMA = "video-intra-forward-forcing-analysis-v2"
PREFLIGHT_SCHEMA = "video-intra-forward-forcing-warmstart-preflight-v1"
INITIALIZATION_ANCHOR_SCHEMA = "video-intra-forward-initialization-anchor-v1"
INITIALIZATION_MATCH_SCHEMA = "video-intra-forward-initialization-match-v1"
TRAINING_CONTENT_SCHEMA = "video-intra-forward-training-content-v1"
MEMORY_SMOKE_SCHEMA = "video-intra-forward-memory-smoke-v1"
MEMORY_SMOKE_RGB_PATTERN = "deterministic_spatiotemporal_linear_v1"
EVALUATION_PREFLIGHT_SCHEMA = "video-intra-forward-evaluation-preflight-v1"
SEED = 1234
EVALUATION_SEED = 20_260_808
BOOTSTRAP_SEED = 20_260_808
TRAIN_UPDATES = 200
WORLD_SIZE = 8
MICRO_BATCH = 1
GLOBAL_BATCH = WORLD_SIZE * MICRO_BATCH
EXPECTED_TRAIN_CLIPS = 512
EXPECTED_VALIDATION_CLIPS = 64
TRAIN_VALIDATION_BATCH_SIZE_PER_RANK = 2
TRAIN_VALIDATION_NUM_WORKERS_PER_RANK = 2
TRAIN_VALIDATION_LOCAL_BATCHES_PER_EVENT = 4
TRAIN_VALIDATION_LOCAL_CLIPS_PER_EVENT = (
    TRAIN_VALIDATION_BATCH_SIZE_PER_RANK * TRAIN_VALIDATION_LOCAL_BATCHES_PER_EVENT
)
TRAIN_VALIDATION_GLOBAL_CLIPS_PER_EVENT = (
    WORLD_SIZE * TRAIN_VALIDATION_LOCAL_CLIPS_PER_EVENT
)
TRAIN_VALIDATION_ITERATIONS = [0, 50, 100, 150, 199]
TRAIN_VALIDATION_CONTRACT = {
    "dataset_infinite": True,
    "dataset_seed": SEED,
    "image_augmentation": False,
    "future_validity_enabled": False,
    "future_validity_max_retries": 0,
    "single_iterator_reused": True,
    "drop_last": False,
    "batch_size_per_rank": TRAIN_VALIDATION_BATCH_SIZE_PER_RANK,
    "loader_workers_per_rank": TRAIN_VALIDATION_NUM_WORKERS_PER_RANK,
    "persistent_workers": False,
    "local_batches_per_event": TRAIN_VALIDATION_LOCAL_BATCHES_PER_EVENT,
    "local_clips_per_event": TRAIN_VALIDATION_LOCAL_CLIPS_PER_EVENT,
    "global_clips_per_event": TRAIN_VALIDATION_GLOBAL_CLIPS_PER_EVENT,
    "iterations": TRAIN_VALIDATION_ITERATIONS,
    "one_complete_registered_validation_pass_per_event": True,
}
TARGET_SHAPE = [6, 4, 24, 120]
NFE_GRID = [1, 2, 4]
SOURCES = ["autonomous", "off", "autonomous_future_shuffled"]
DEPLOYABLE_SOURCES = ["autonomous", "off", "autonomous_future_shuffled"]
MIDPOINT_BLOCK_INDEX = 14
WAN_BLOCK_COUNT = 30
AUXILIARY_HISTORY_BINS = 2
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
    midpoint_injection: bool
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
        "MID-OFF",
        "mid_off",
        "ravenhuang/wan-dit/intra_forward_mid_off.yaml",
        "off",
        False,
        False,
        "aligned",
        0.0,
        1.0,
        0.02,
        0.0,
        False,
        False,
        "same-schema auxiliary predictor with midpoint residual exactly off",
    ),
    Arm(
        "MID-ON",
        "mid_on",
        "ravenhuang/wan-dit/intra_forward_mid_on.yaml",
        "matched",
        True,
        False,
        "aligned",
        0.0,
        1.0,
        0.02,
        0.0,
        False,
        True,
        "generated stop-gradient q0 estimate injected after block 14",
    ),
)
ARM_BY_CODE = {arm.code: arm for arm in ARMS}


def implementation_paths() -> list[Path]:
    return [
        Path(__file__).resolve(),
        PROTOCOL,
        RUNBOOK,
        EVALUATOR,
        WARMSTART_PREFLIGHT,
        TRANSFORM,
        DATASET,
        MODEL_CONFIG,
        COMMON_CONFIG,
        DATASET_CONFIG,
        TRAIN_SBATCH,
        EVALUATE_SBATCH,
        MEMORY_SMOKE,
        MEMORY_SMOKE_SBATCH,
        SUBMIT_SCRIPT,
        REPO_ROOT / "tools/env/activate_b200.sh",
        REPO_ROOT / "tools/env/verify_b200_runtime.py",
        REPO_ROOT / "projects/latent_action_models/train.py",
        REPO_ROOT
        / "projects/latent_action_models/lam/dual_explicit_action_dit_model.py",
        REPO_ROOT / "robot_wm/modeling/networks/wan_forward_model.py",
        REPO_ROOT / "robot_wm/modeling/dual_diffusion/adapters.py",
        REPO_ROOT / "robot_wm/modeling/dual_diffusion/conditioning.py",
        REPO_ROOT / "robot_wm/modeling/dual_diffusion/flow.py",
        REPO_ROOT / "robot_wm/utils/trainer.py",
        REPO_ROOT / "robot_wm/utils/wandb.py",
    ] + [
        REPO_ROOT
        / "projects/latent_action_models/configs/experiments_0908"
        / arm.selector
        for arm in ARMS
    ]


def validate_arm_table() -> None:
    if [arm.code for arm in ARMS] != ["MID-OFF", "MID-ON"]:
        raise ContractError("frozen arm order/codes changed")
    off, on = ARMS
    common_fields = (
        "condition_on_clock",
        "schedule_mode",
        "lead_logit",
        "auxiliary_loss_weight",
        "state_gate_init",
        "clock_gate_init",
        "parameter_matched_control",
    )
    changed = {
        field: [getattr(off, field), getattr(on, field)]
        for field in common_fields
        if getattr(off, field) != getattr(on, field)
    }
    if changed:
        raise ContractError(f"paired arm common factors differ: {changed}")
    if (
        off.condition_mode != "off"
        or off.condition_on_state
        or off.midpoint_injection
        or on.condition_mode != "matched"
        or not on.condition_on_state
        or not on.midpoint_injection
        or off.auxiliary_loss_weight != 1.0
        or off.condition_on_clock
    ):
        raise ContractError("frozen midpoint treatment mapping changed")


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


def validate_training_validation_contract(value: Any) -> dict[str, Any]:
    """Reject registrations/config receipts that can exhaust or oversample val64."""
    if not isinstance(value, Mapping) or dict(value) != TRAIN_VALIDATION_CONTRACT:
        raise ContractError("training validation iterator contract differs")
    if TRAIN_VALIDATION_GLOBAL_CLIPS_PER_EVENT != EXPECTED_VALIDATION_CLIPS:
        raise ContractError("training validation contract does not cover val64 exactly")
    return dict(value)


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


def contains_omegaconf_interpolation(value: Any) -> bool:
    if isinstance(value, str):
        return "${" in value
    if isinstance(value, Mapping):
        return any(contains_omegaconf_interpolation(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(contains_omegaconf_interpolation(item) for item in value)
    return False


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
        "clip_ids_sha256": hashlib.sha256(canonical_json(sorted(clip_ids))).hexdigest(),
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
        or metadata.get("rgb_shape") != [expected_clips, 13, 3, 180, 960]
        or metadata.get("actions_shape") != [expected_clips, 13, 5, 23]
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
            str(arm.midpoint_injection).lower(),
        ]
        if any("|" in field or "\n" in field for field in fields):
            raise ContractError("arm field contains an unsafe delimiter")
        print("|".join(fields))
    return 0


def command_register(args: argparse.Namespace) -> int:
    validate_arm_table()
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

    implementation = {
        str(path.relative_to(REPO_ROOT)): source_file_record(path)
        for path in implementation_paths()
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
                "name": "six-channel per-view Haar DC/coarse-motion scratchpad",
                "input_shape": ["B", 13, 3, 180, 960],
                "target_shape": ["B", *TARGET_SHAPE],
                "channel_order": [
                    "R_dc",
                    "G_dc",
                    "B_dc",
                    "R_motion",
                    "G_motion",
                    "B_motion",
                ],
                "learned_encoder": False,
                "external_checkpoint": False,
                "autonomous_transform_calls": 0,
                "all_auxiliary_bins_start_from_noise": True,
            },
            "architecture": {
                "wan_block_count": WAN_BLOCK_COUNT,
                "midpoint_block_index": MIDPOINT_BLOCK_INDEX,
                "midpoint_head_calls_per_wan_call": 1,
                "additional_wan_calls": 0,
                "generated_clean_formula": "q0_hat=q_sigma-sigma*v_hat",
                "generated_clean_stop_gradient": True,
                "history_preserved_future_shuffle": True,
                "auxiliary_history_bins": AUXILIARY_HISTORY_BINS,
                "input_level_auxiliary_state_residual": "exact_zero",
                "input_level_auxiliary_clock_residual": "exact_zero",
                "gradient_checkpointing": False,
            },
            "training": {
                "seed": SEED,
                "updates": TRAIN_UPDATES,
                "world_size": WORLD_SIZE,
                "micro_batch_per_rank": MICRO_BATCH,
                "global_batch": GLOBAL_BATCH,
                "validation_iterator": dict(TRAIN_VALIDATION_CONTRACT),
                "same_optimizer_and_data_order": True,
                "same_non_auxiliary_warm_start": True,
                "identical_initialized_model_bytes_required": True,
                "paired_clip_order_noise_and_timesteps": True,
                "update_zero_memory_smoke_required": True,
                "arms": [asdict(arm) for arm in ARMS],
            },
            "evaluation": {
                "split": "validation",
                "clips": EXPECTED_VALIDATION_CLIPS,
                "nfe": NFE_GRID,
                "sources": SOURCES,
                "deployable_sources": DEPLOYABLE_SOURCES,
                "sample_keyed_noise_seed": EVALUATION_SEED,
                "same_total_wan_calls_at_each_nfe": True,
                "midpoint_head_calls_equal_wan_calls": True,
                "teacher_calls": 0,
                "clean_future_feature_available_to_deployable_sampler": False,
                "oracle_sources": [],
                "pre_nccl_content_preflight_required": True,
                "timing": {
                    "cuda_synchronize_before_and_after": True,
                    "records_end_to_end_and_peak_memory": True,
                    "records_midpoint_head_elapsed": True,
                    "artifact_audit_batch_size": 2,
                    "same_batch_profile_equivalence_batch_size": 2,
                    "same_batch_profile_equivalence_exact": True,
                    "endpoint_timing_batch_size": 1,
                    "cross_batch_output_comparison": "diagnostic_only",
                    "generations_per_two_clip_cell": 4,
                    "latency_claim_scope": "descriptive_equal_nfe_only",
                },
                "bootstrap_replicates": 10_000,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            "decision": {
                "primary_nfe": 1,
                "primary_metric": "temporal_mse",
                "temporal_minimum_point_gain": 0.03,
                "temporal_minimum_simultaneous_lower_bound": 0.01,
                "quality_guardrail_point": 0.0,
                "quality_guardrail_simultaneous_lower_bound": -0.01,
                "references": [
                    "MID-OFF/aligned",
                    "MID-ON/off",
                    "MID-ON/history-preserved-future-shuffled",
                ],
                "mechanism_requires_aligned_better_than_all_references": True,
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
    validate_arm_table()
    registration = read_json(path.resolve(strict=True), "registration")
    if registration.get("schema") != SCHEMA or not validate_identity(registration):
        raise ContractError("registration schema or identity is invalid")
    if registration.get("protected_test", {}).get("allowed") is not False:
        raise ContractError("registration does not fail closed on protected test")
    training = registration.get("training")
    if not isinstance(training, Mapping):
        raise ContractError("registration lacks a training contract")
    validate_training_validation_contract(training.get("validation_iterator"))
    if training.get("arms") != [asdict(arm) for arm in ARMS]:
        raise ContractError("registration arm table differs from frozen treatment")
    if training.get("update_zero_memory_smoke_required") is not True:
        raise ContractError("registration does not require the memory canary")
    expected_architecture = {
        "wan_block_count": WAN_BLOCK_COUNT,
        "midpoint_block_index": MIDPOINT_BLOCK_INDEX,
        "midpoint_head_calls_per_wan_call": 1,
        "additional_wan_calls": 0,
        "generated_clean_formula": "q0_hat=q_sigma-sigma*v_hat",
        "generated_clean_stop_gradient": True,
        "history_preserved_future_shuffle": True,
        "auxiliary_history_bins": AUXILIARY_HISTORY_BINS,
        "input_level_auxiliary_state_residual": "exact_zero",
        "input_level_auxiliary_clock_residual": "exact_zero",
        "gradient_checkpointing": False,
    }
    if registration.get("architecture") != expected_architecture:
        raise ContractError("registration midpoint architecture differs")
    evaluation = registration.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("split") != "validation"
        or evaluation.get("clips") != EXPECTED_VALIDATION_CLIPS
        or evaluation.get("nfe") != NFE_GRID
        or evaluation.get("sources") != SOURCES
        or evaluation.get("deployable_sources") != DEPLOYABLE_SOURCES
        or evaluation.get("sample_keyed_noise_seed") != EVALUATION_SEED
        or evaluation.get("teacher_calls") != 0
        or evaluation.get("clean_future_feature_available_to_deployable_sampler")
        is not False
        or evaluation.get("oracle_sources") != []
        or evaluation.get("pre_nccl_content_preflight_required") is not True
        or evaluation.get("timing")
        != {
            "cuda_synchronize_before_and_after": True,
            "records_end_to_end_and_peak_memory": True,
            "records_midpoint_head_elapsed": True,
            "artifact_audit_batch_size": 2,
            "same_batch_profile_equivalence_batch_size": 2,
            "same_batch_profile_equivalence_exact": True,
            "endpoint_timing_batch_size": 1,
            "cross_batch_output_comparison": "diagnostic_only",
            "generations_per_two_clip_cell": 4,
            "latency_claim_scope": "descriptive_equal_nfe_only",
        }
    ):
        raise ContractError("registration deployable evaluation contract differs")
    if registration.get("wandb") != {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "group": None,
    }:
        raise ContractError("registration private W&B destination differs")
    if verify_files:
        validate_source(str(registration["source"]["commit"]))
        expected_implementation = {
            str(path.resolve(strict=True).relative_to(REPO_ROOT))
            for path in implementation_paths()
        }
        if set(registration.get("implementation", {})) != expected_implementation:
            raise ContractError("registered implementation set changed")
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


def command_bind_initialization(args: argparse.Namespace) -> int:
    """Bind and compare update-zero model/optimizer schemas before training."""
    registration = load_registration(args.registration, verify_files=False)
    arm = _arm(args.array_task_id)
    expected_run = Path(registration["study_root"]) / "runs" / arm.slug
    if args.run_dir != expected_run or not expected_run.is_dir():
        raise ContractError(f"run directory must be {expected_run}")
    preflight_path = args.preflight.resolve(strict=True)
    if (
        preflight_path.is_symlink()
        or preflight_path.parent != expected_run
        or preflight_path.name != "warmstart_preflight.json"
    ):
        raise ContractError("preflight must be the canonical arm-run receipt")
    preflight = read_json(preflight_path, "warm-start preflight")
    arm_manifest = read_json(
        (expected_run / "arm_manifest.json").resolve(strict=True),
        "arm manifest",
    )
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or not validate_identity(preflight)
        or preflight.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or preflight.get("arm") != arm.code
        or preflight.get("selector") != arm.selector
        or preflight.get("arm_identity_sha256") != arm_manifest.get("identity_sha256")
        or preflight.get("status") != "pass"
    ):
        raise ContractError("warm-start preflight identity or arm differs")
    compared_fields = (
        "initialized_model_state_sha256",
        "parameter_schema_sha256",
        "trainable_parameter_schema_sha256",
        "optimizer_schema_sha256",
        "total_parameter_count",
        "trainable_parameter_count",
        "initial_snapshot_tensor_count",
    )
    observed = {field: preflight.get(field) for field in compared_fields}
    for field in compared_fields[:4]:
        if (
            not isinstance(observed[field], str)
            or SHA_RE.fullmatch(observed[field]) is None
        ):
            raise ContractError(f"preflight lacks valid {field}")
    if any(
        isinstance(observed[field], bool)
        or not isinstance(observed[field], int)
        or observed[field] <= 0
        for field in compared_fields[4:]
    ):
        raise ContractError("preflight parameter counts are invalid")

    anchor_path = Path(registration["study_root"]) / "initialization_anchor.json"
    if args.array_task_id == 0:
        if arm.code != "MID-OFF" or anchor_path.exists():
            raise ContractError("MID-OFF must create a fresh initialization anchor")
        anchor = identity_payload(
            {
                "schema": INITIALIZATION_ANCHOR_SCHEMA,
                "registration_identity_sha256": registration["identity_sha256"],
                "source_arm": arm.code,
                "source_preflight_identity_sha256": preflight["identity_sha256"],
                "initialization": observed,
            }
        )
        exclusive_json(anchor_path, anchor)
    else:
        anchor = read_json(anchor_path.resolve(strict=True), "initialization anchor")
        if (
            anchor.get("schema") != INITIALIZATION_ANCHOR_SCHEMA
            or not validate_identity(anchor)
            or anchor.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or anchor.get("source_arm") != "MID-OFF"
        ):
            raise ContractError("initialization anchor is invalid")
    if anchor.get("initialization") != observed:
        raise ContractError(
            f"{arm.code} initialized model/optimizer bytes differ from MID-OFF"
        )
    match = identity_payload(
        {
            "schema": INITIALIZATION_MATCH_SCHEMA,
            "registration_identity_sha256": registration["identity_sha256"],
            "arm": arm.code,
            "arm_identity_sha256": arm_manifest["identity_sha256"],
            "preflight_identity_sha256": preflight["identity_sha256"],
            "anchor_identity_sha256": anchor["identity_sha256"],
            "initialization": observed,
            "exact_match": True,
        }
    )
    exclusive_json(expected_run / "initialization_match.json", match)
    print(json.dumps(match, sort_keys=True))
    return 0


def validate_memory_smoke(
    registration: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    """Require the exact eight-rank update-zero forward/backward canary."""
    study_root = Path(str(registration["study_root"]))
    receipt_path = study_root / "memory_smoke.json"
    if receipt_path.is_symlink():
        raise ContractError("memory smoke receipt must not be a symlink")
    receipt = read_json(receipt_path.resolve(strict=True), "memory smoke receipt")
    runtime_path = study_root / "memory_smoke_runtime.json"
    expected_batch = {
        "rgb": [1, 13, 3, 180, 960],
        "actions": [1, 13, 5, 157],
        "mask": [1, 13],
        "morphology_index": [1],
        "clip_index": [1],
    }
    if (
        receipt.get("schema") != MEMORY_SMOKE_SCHEMA
        or not validate_identity(receipt)
        or receipt.get("status") != "pass"
        or receipt.get("registration_identity_sha256")
        != registration.get("identity_sha256")
        or receipt.get("source_commit")
        != registration.get("source", {}).get("commit")
        or receipt.get("world_size") != WORLD_SIZE
        or receipt.get("selector") != ARMS[1].selector
        or receipt.get("synthetic_batch_shapes") != expected_batch
        or receipt.get("synthetic_rgb_pattern") != MEMORY_SMOKE_RGB_PATTERN
        or receipt.get("gradient_checkpointing") is not False
        or receipt.get("forward_completed") is not True
        or receipt.get("backward_completed") is not True
        or receipt.get("optimizer_step_executed") is not False
        or receipt.get("completed_optimizer_updates") != 0
        or receipt.get("optimizer_state_entries") != 0
        or not isinstance(receipt.get("maximum_peak_allocated_bytes"), int)
        or receipt.get("maximum_peak_allocated_bytes", 0) <= 0
        or not isinstance(receipt.get("minimum_headroom_bytes"), int)
        or receipt.get("minimum_headroom_bytes", 0) <= 0
        or receipt.get("scientific_metrics_emitted") is not False
        or not isinstance(receipt.get("runtime_receipt_sha256"), str)
        or SHA_RE.fullmatch(str(receipt.get("runtime_receipt_sha256"))) is None
        or not isinstance(receipt.get("ranks"), list)
        or len(receipt["ranks"]) != WORLD_SIZE
    ):
        raise ContractError("update-zero memory smoke contract differs")
    ranks = receipt["ranks"]
    if sorted(item.get("rank") for item in ranks if isinstance(item, Mapping)) != list(
        range(WORLD_SIZE)
    ) or any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("peak_allocated_bytes"), int)
        or item["peak_allocated_bytes"] <= 0
        or not isinstance(item.get("total_memory_bytes"), int)
        or item["total_memory_bytes"] <= item["peak_allocated_bytes"]
        or item.get("finite_loss") is not True
        or item.get("finite_gradients") is not True
        for item in ranks
    ):
        raise ContractError("memory smoke rank telemetry differs")
    if verify_files:
        if runtime_path.is_symlink() or not runtime_path.is_file():
            raise ContractError("memory smoke runtime receipt is missing")
        if sha256_file(runtime_path.resolve(strict=True)) != receipt[
            "runtime_receipt_sha256"
        ]:
            raise ContractError("memory smoke runtime receipt digest differs")
    return receipt


def command_verify_memory_smoke(args: argparse.Namespace) -> int:
    registration = load_registration(args.registration, verify_files=False)
    receipt = validate_memory_smoke(registration, verify_files=True)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def command_bind_training_output(args: argparse.Namespace) -> int:
    """Content-bind a completed arm before any evaluator may load it."""
    registration = load_registration(args.registration, verify_files=False)
    memory_smoke = validate_memory_smoke(registration, verify_files=True)
    arm = _arm(args.array_task_id)
    expected_run = Path(registration["study_root"]) / "runs" / arm.slug
    if args.run_dir != expected_run or not expected_run.is_dir():
        raise ContractError(f"run directory must be {expected_run}")

    def run_file(name: str, label: str) -> tuple[Path, dict[str, Any]]:
        path = expected_run / name
        if path.is_symlink():
            raise ContractError(f"{label} must not be a symlink")
        canonical = path.resolve(strict=True)
        if canonical.parent != (expected_run / Path(name).parent).resolve(strict=True):
            raise ContractError(f"{label} escaped the arm run directory")
        return canonical, file_record(canonical, label)

    manifest_path, manifest_record = run_file("arm_manifest.json", "arm manifest")
    manifest = read_json(manifest_path, "arm manifest")
    if (
        manifest.get("schema") != ARM_SCHEMA
        or not validate_identity(manifest)
        or manifest.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or manifest.get("array_task_id") != args.array_task_id
        or manifest.get("arm") != asdict(arm)
    ):
        raise ContractError("arm manifest identity differs")
    arm_identity = manifest["identity_sha256"]

    snapshot_path, snapshot_record = run_file("snapshot.pt", "final snapshot")
    hydra_config_path, hydra_config_record = run_file(
        ".hydra/config.yaml", "Hydra execution config"
    )
    resolved_config_path, resolved_config_record = run_file(
        "resolved_config.json", "fully resolved config"
    )
    runtime_path, runtime_record = run_file("runtime.json", "training runtime")
    completion_path, completion_record = run_file(
        "training_complete.json", "training completion receipt"
    )
    match_path, match_record = run_file(
        "initialization_match.json", "initialization match"
    )
    completion = read_json(completion_path, "training completion receipt")
    initialization_match = read_json(match_path, "initialization match")
    resolved_config = read_json(resolved_config_path, "fully resolved config")
    if contains_omegaconf_interpolation(resolved_config):
        raise ContractError("fully resolved config contains an interpolation")
    resolved_dual = resolved_config.get("model", {}).get("dual_diffusion", {})
    resolved_wandb = resolved_config.get("wandb", {})
    if (
        resolved_config.get("seed") != SEED
        or resolved_config.get("name") != f"{registration['study_id']}-{arm.code}"
        or resolved_config.get("trainer", {}).get("config", {}).get("max_iter")
        != TRAIN_UPDATES
        or resolved_dual.get("condition_mode") != arm.condition_mode
        or resolved_dual.get("condition_on_tf") is not arm.condition_on_state
        or resolved_dual.get("intra_forward_forcing", {}).get("history_bins")
        != AUXILIARY_HISTORY_BINS
        or resolved_wandb.get("entity") != WANDB_ENTITY
        or resolved_wandb.get("project") != WANDB_PROJECT
        or resolved_wandb.get("group") is not None
        or resolved_wandb.get("id") != arm_identity
    ):
        raise ContractError("fully resolved arm configuration differs")
    validate_runtime_receipt(runtime_path, registration, label="training runtime")
    if runtime_record["sha256"] != memory_smoke["runtime_receipt_sha256"]:
        raise ContractError("memory-smoke and training B200 runtimes differ")
    expected_completion = {
        "schema_version": 1,
        "status": "completed",
        "completed_updates": TRAIN_UPDATES,
        "max_iter": TRAIN_UPDATES,
        "run_identity_sha256": arm_identity,
        "snapshot": str(snapshot_path),
        "wandb_run_id": arm_identity,
        "wandb_training_status": "completed",
        "source_commit": registration["source"]["commit"],
        "registration_identity_sha256": registration["identity_sha256"],
        "warm_start_sha256": registration["warm_start"]["sha256"],
        "runtime_receipt_sha256": runtime_record["sha256"],
        "resolved_config_sha256": resolved_config_record["sha256"],
    }
    mismatches = {
        key: {"observed": completion.get(key), "expected": value}
        for key, value in expected_completion.items()
        if completion.get(key) != value
    }
    if mismatches:
        raise ContractError(f"training completion provenance differs: {mismatches}")
    if (
        initialization_match.get("schema") != INITIALIZATION_MATCH_SCHEMA
        or not validate_identity(initialization_match)
        or initialization_match.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or initialization_match.get("arm") != arm.code
        or initialization_match.get("arm_identity_sha256") != arm_identity
        or initialization_match.get("exact_match") is not True
    ):
        raise ContractError("initialization match identity differs")

    payload = identity_payload(
        {
            "schema": TRAINING_CONTENT_SCHEMA,
            "status": "completed",
            "registration_identity_sha256": registration["identity_sha256"],
            "source_commit": registration["source"]["commit"],
            "arm": arm.code,
            "arm_identity_sha256": arm_identity,
            "wandb": {
                "entity": WANDB_ENTITY,
                "project": WANDB_PROJECT,
                "group": None,
                "run_id": arm_identity,
                "status": "finished_by_successful_training_process",
            },
            "snapshot": snapshot_record,
            "hydra_config": hydra_config_record,
            "resolved_config": resolved_config_record,
            "training_runtime": runtime_record,
            "completion": completion_record,
            "arm_manifest": manifest_record,
            "initialization_match": match_record,
            "memory_smoke_identity_sha256": memory_smoke["identity_sha256"],
        }
    )
    exclusive_json(expected_run / "training_content.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def validate_runtime_receipt(
    path: Path, registration: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Validate that the runtime verifier observed the registered source stack."""
    if path.is_symlink():
        raise ContractError(f"{label} must not be a symlink")
    canonical = path.resolve(strict=True)
    receipt = read_json(canonical, label)
    source_paths = receipt.get("source_paths")
    gpu = receipt.get("gpus")
    environment = receipt.get("environment")
    weights = receipt.get("weights")
    registered_python = Path(
        str(registration["runtime"]["python"]["path"])
    ).resolve(strict=True)
    observed_python = Path(
        str(environment.get("sys_executable", ""))
        if isinstance(environment, Mapping)
        else ""
    )
    try:
        observed_python = observed_python.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ContractError(f"{label} has an invalid interpreter path") from exc
    expected_roots = {
        "robot_wm": REPO_ROOT,
        "lam": REPO_ROOT / "projects/latent_action_models",
        "wan_transformer": Path(str(registration["runtime"]["videox_home"])),
        "wan_vae": Path(str(registration["runtime"]["videox_home"])),
    }
    source_valid = isinstance(source_paths, Mapping)
    if source_valid:
        for name, root in expected_roots.items():
            try:
                observed = Path(str(source_paths.get(name, ""))).resolve(strict=True)
                expected_root = root.resolve(strict=True)
            except (OSError, RuntimeError):
                source_valid = False
                break
            if observed != expected_root and expected_root not in observed.parents:
                source_valid = False
                break
    devices = gpu.get("devices") if isinstance(gpu, Mapping) else None
    if (
        receipt.get("python") != "3.10.20"
        or receipt.get("python_no_user_site") is not True
        or observed_python != registered_python
        or receipt.get("videox_commit")
        != "1d6d9c3e1540968466937129fef4b288041e06de"
        or not source_valid
        or not isinstance(gpu, Mapping)
        or gpu.get("count") != WORLD_SIZE
        or gpu.get("nccl_available") is not True
        or gpu.get("inaccessible_peer_pairs") != []
        or not isinstance(devices, list)
        or len(devices) != WORLD_SIZE
        or any(
            not isinstance(device, Mapping)
            or device.get("capability") != [10, 0]
            for device in devices
        )
        or not isinstance(weights, Mapping)
        or weights.get("root") != registration["runtime"]["wan_dir"]
    ):
        raise ContractError(f"{label} differs from the registered B200 runtime")
    return {"path": str(canonical), "sha256": sha256_file(canonical)}


def validate_training_content(
    registration: Mapping[str, Any],
    arm: Arm,
    run_dir: Path,
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate the post-training content receipt and every bound file."""
    expected_run = Path(str(registration["study_root"])) / "runs" / arm.slug
    if run_dir != expected_run:
        raise ContractError(f"run directory must be {expected_run}")
    path = run_dir / "training_content.json"
    if path.is_symlink():
        raise ContractError("training content receipt must not be a symlink")
    receipt = read_json(path.resolve(strict=True), "training content receipt")
    wandb_record = receipt.get("wandb")
    if (
        receipt.get("schema") != TRAINING_CONTENT_SCHEMA
        or not validate_identity(receipt)
        or receipt.get("status") != "completed"
        or receipt.get("registration_identity_sha256")
        != registration.get("identity_sha256")
        or receipt.get("source_commit")
        != registration.get("source", {}).get("commit")
        or receipt.get("arm") != arm.code
        or not isinstance(receipt.get("arm_identity_sha256"), str)
        or SHA_RE.fullmatch(str(receipt.get("arm_identity_sha256"))) is None
        or wandb_record
        != {
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "group": None,
            "run_id": receipt.get("arm_identity_sha256"),
            "status": "finished_by_successful_training_process",
        }
        or not isinstance(receipt.get("memory_smoke_identity_sha256"), str)
        or SHA_RE.fullmatch(str(receipt.get("memory_smoke_identity_sha256"))) is None
    ):
        raise ContractError("training content identity differs")
    smoke = validate_memory_smoke(registration, verify_files=verify_files)
    training_runtime_record = receipt.get("training_runtime")
    if (
        receipt["memory_smoke_identity_sha256"] != smoke["identity_sha256"]
        or not isinstance(training_runtime_record, Mapping)
        or training_runtime_record.get("sha256")
        != smoke["runtime_receipt_sha256"]
    ):
        raise ContractError("training content references another memory smoke")
    records = {
        "snapshot": "snapshot.pt",
        "hydra_config": ".hydra/config.yaml",
        "resolved_config": "resolved_config.json",
        "training_runtime": "runtime.json",
        "completion": "training_complete.json",
        "arm_manifest": "arm_manifest.json",
        "initialization_match": "initialization_match.json",
    }
    if verify_files:
        for field, relative in records.items():
            record = receipt.get(field)
            expected = run_dir / relative
            if not isinstance(record, Mapping) or expected.is_symlink():
                raise ContractError(f"training content lacks {field}")
            canonical = expected.resolve(strict=True)
            if (
                record.get("path") != str(canonical)
                or record.get("bytes") != canonical.stat().st_size
                or record.get("sha256") != sha256_file(canonical)
            ):
                raise ContractError(f"training content-bound {field} changed")
        validate_runtime_receipt(
            run_dir / "runtime.json", registration, label="training runtime"
        )
    return receipt


def validate_evaluation_preflight(
    registration: Mapping[str, Any],
    arm: Arm,
    run_dir: Path,
    *,
    training_content: Mapping[str, Any],
    evaluation_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable content audit produced before torchrun/NCCL.

    The preflight creator performs all multi-GB warm-start and snapshot hashing
    in the single-process Slurm entrypoint. Distributed evaluation only checks
    this small, content-bound receipt, avoiding a rank-zero hashing stall before
    the first NCCL collective.
    """

    expected_run = Path(str(registration["study_root"])) / "runs" / arm.slug
    if run_dir != expected_run:
        raise ContractError(f"run directory must be {expected_run}")
    path = run_dir / "evaluation_input_preflight.json"
    if path.is_symlink():
        raise ContractError("evaluation preflight receipt must not be a symlink")
    receipt = read_json(path.resolve(strict=True), "evaluation input preflight")
    expected = {
        "schema": EVALUATION_PREFLIGHT_SCHEMA,
        "status": "pass",
        "registration_identity_sha256": registration["identity_sha256"],
        "source_commit": registration["source"]["commit"],
        "arm": arm.code,
        "arm_identity_sha256": training_content["arm_identity_sha256"],
        "training_content_identity_sha256": training_content["identity_sha256"],
        "memory_smoke_identity_sha256": training_content[
            "memory_smoke_identity_sha256"
        ],
        "snapshot_sha256": training_content["snapshot"]["sha256"],
        "hydra_config_sha256": training_content["hydra_config"]["sha256"],
        "resolved_config_sha256": training_content["resolved_config"]["sha256"],
        "training_runtime_sha256": training_content["training_runtime"]["sha256"],
        "evaluation_runtime_sha256": evaluation_runtime["sha256"],
        "verification_phase": "single_process_before_torchrun_and_nccl",
        "registration_files_fully_verified": True,
        "warm_start_sha256_verified": True,
        "training_content_files_fully_verified": True,
        "snapshot_sha256_verified": True,
        "protected_test_accessed": False,
    }
    if (
        not validate_identity(receipt)
        or any(receipt.get(key) != value for key, value in expected.items())
        or receipt.get("evaluation_runtime_sha256")
        != receipt.get("training_runtime_sha256")
    ):
        raise ContractError("evaluation input preflight identity or content differs")
    return receipt


def command_bind_evaluation_input(args: argparse.Namespace) -> int:
    """Fully hash evaluation inputs before torchrun creates an NCCL group."""

    registration = load_registration(args.registration, verify_files=True)
    arm = _arm(args.array_task_id)
    expected_run = Path(registration["study_root"]) / "runs" / arm.slug
    if args.run_dir != expected_run or not expected_run.is_dir():
        raise ContractError(f"run directory must be {expected_run}")
    output = expected_run / "evaluation_input_preflight.json"
    if output.exists() or output.is_symlink():
        raise ContractError("fresh evaluation preflight receipt already exists")
    training_content = validate_training_content(
        registration,
        arm,
        expected_run,
        verify_files=True,
    )
    expected_runtime = expected_run / "evaluation_runtime.json"
    if args.runtime_receipt != expected_runtime:
        raise ContractError(f"evaluation runtime receipt must be {expected_runtime}")
    evaluation_runtime = validate_runtime_receipt(
        args.runtime_receipt,
        registration,
        label="evaluation runtime",
    )
    if (
        evaluation_runtime["sha256"]
        != training_content["training_runtime"]["sha256"]
    ):
        raise ContractError("training and evaluation B200 runtime receipts differ")
    payload = identity_payload(
        {
            "schema": EVALUATION_PREFLIGHT_SCHEMA,
            "status": "pass",
            "registration_identity_sha256": registration["identity_sha256"],
            "source_commit": registration["source"]["commit"],
            "arm": arm.code,
            "arm_identity_sha256": training_content["arm_identity_sha256"],
            "training_content_identity_sha256": training_content["identity_sha256"],
            "memory_smoke_identity_sha256": training_content[
                "memory_smoke_identity_sha256"
            ],
            "snapshot_sha256": training_content["snapshot"]["sha256"],
            "hydra_config_sha256": training_content["hydra_config"]["sha256"],
            "resolved_config_sha256": training_content["resolved_config"]["sha256"],
            "training_runtime_sha256": training_content["training_runtime"]["sha256"],
            "evaluation_runtime_sha256": evaluation_runtime["sha256"],
            "verification_phase": "single_process_before_torchrun_and_nccl",
            "registration_files_fully_verified": True,
            "warm_start_sha256_verified": True,
            "training_content_files_fully_verified": True,
            "snapshot_sha256_verified": True,
            "protected_test_accessed": False,
        }
    )
    exclusive_json(output, payload)
    # Re-read through the same cheap validator used by distributed evaluation.
    validate_evaluation_preflight(
        registration,
        arm,
        expected_run,
        training_content=training_content,
        evaluation_runtime=evaluation_runtime,
    )
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
    simultaneous_comparisons: int = 9,
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
    alpha = 0.05 / simultaneous_comparisons
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def _validate_mid_off_noop(
    keyed: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
) -> None:
    """MID-OFF labels must be exact same-checkpoint no-op interventions."""
    fields = (
        "video_final_sha256",
        "auxiliary_final_sha256",
        "decoded_final_sha256",
        "equivalence_decoded_sha256",
        "timed_decoded_sha256",
        "video_nmse",
        "decoded_mse",
        "temporal_mse",
        "auxiliary_future_nmse",
        "auxiliary_future_cosine",
        "auxiliary_dc_nmse",
        "auxiliary_motion_nmse",
    )
    for nfe in NFE_GRID:
        for clip in range(EXPECTED_VALIDATION_CLIPS):
            aligned = keyed[("MID-OFF", "autonomous", nfe, clip)]
            for source in ("off", "autonomous_future_shuffled"):
                reference = keyed[("MID-OFF", source, nfe, clip)]
                changed = [
                    field
                    for field in fields
                    if aligned.get(field) != reference.get(field)
                ]
                if changed:
                    raise ContractError(
                        "MID-OFF source label changed an exact no-op: "
                        f"NFE{nfe} clip {clip} source={source}, fields={changed}"
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
    expected_rows = EXPECTED_VALIDATION_CLIPS * len(NFE_GRID) * len(SOURCES)
    arm = receipt.get("arm")
    arm_identity = receipt.get("arm_identity_sha256")
    if (
        receipt.get("schema") != EVALUATION_COMPLETE_SCHEMA
        or not validate_identity(receipt)
        or receipt.get("registration_identity_sha256") != registration_identity_sha256
        or arm not in ARM_BY_CODE
        or not isinstance(arm_identity, str)
        or SHA_RE.fullmatch(arm_identity) is None
        or receipt.get("rows") != expected_rows
        or receipt.get("validation_clips") != EXPECTED_VALIDATION_CLIPS
        or receipt.get("nfe") != NFE_GRID
        or receipt.get("sources") != SOURCES
        or receipt.get("world_size") != WORLD_SIZE
        or receipt.get("rows_sha256") != sha256_file(resolved_rows)
        or not isinstance(receipt.get("snapshot_sha256"), str)
        or SHA_RE.fullmatch(receipt["snapshot_sha256"]) is None
        or not isinstance(receipt.get("resolved_config_sha256"), str)
        or SHA_RE.fullmatch(receipt["resolved_config_sha256"]) is None
        or not isinstance(receipt.get("hydra_config_sha256"), str)
        or SHA_RE.fullmatch(receipt["hydra_config_sha256"]) is None
        or not isinstance(receipt.get("training_content_identity_sha256"), str)
        or SHA_RE.fullmatch(receipt["training_content_identity_sha256"]) is None
        or not isinstance(receipt.get("training_runtime_sha256"), str)
        or SHA_RE.fullmatch(receipt["training_runtime_sha256"]) is None
        or receipt.get("evaluation_runtime_sha256")
        != receipt.get("training_runtime_sha256")
        or not isinstance(receipt.get("initialization_match_identity_sha256"), str)
        or SHA_RE.fullmatch(receipt["initialization_match_identity_sha256"]) is None
        or not isinstance(receipt.get("evaluation_preflight_identity_sha256"), str)
        or SHA_RE.fullmatch(receipt["evaluation_preflight_identity_sha256"]) is None
        or receipt.get("protected_test_accessed") is not False
    ):
        raise ContractError(f"invalid evaluation completion receipt: {receipt_path}")
    return receipt


def command_analyze(args: argparse.Namespace) -> int:
    registration = load_registration(args.registration, verify_files=False)
    expected_output = Path(registration["study_root"]) / "analysis.json"
    if args.output != expected_output or args.output.exists():
        raise ContractError(f"analysis output must be fresh path {expected_output}")
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
        expected_rows_path = (
            Path(registration["study_root"])
            / "evaluation"
            / ARM_BY_CODE[receipt_arm].slug
            / "rows.jsonl"
        )
        if path != expected_rows_path:
            raise ContractError(f"{receipt_arm} rows path must be {expected_rows_path}")
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
                        or row.get("training_content_identity_sha256")
                        != receipt["training_content_identity_sha256"]
                        or row.get("resolved_config_sha256")
                        != receipt["resolved_config_sha256"]
                        or row.get("hydra_config_sha256")
                        != receipt["hydra_config_sha256"]
                        or row.get("training_runtime_sha256")
                        != receipt["training_runtime_sha256"]
                        or row.get("evaluation_runtime_sha256")
                        != receipt["evaluation_runtime_sha256"]
                        or row.get("evaluation_preflight_identity_sha256")
                        != receipt["evaluation_preflight_identity_sha256"]
                    ):
                        raise ContractError(f"invalid result row in {path}")
                    rows.append(row)
    if set(receipts) != set(ARM_BY_CODE):
        raise ContractError("evaluation receipts do not cover all frozen arms")
    anchor = read_json(
        (Path(registration["study_root"]) / "initialization_anchor.json").resolve(
            strict=True
        ),
        "initialization anchor",
    )
    if (
        anchor.get("schema") != INITIALIZATION_ANCHOR_SCHEMA
        or not validate_identity(anchor)
        or anchor.get("registration_identity_sha256") != registration["identity_sha256"]
    ):
        raise ContractError("initialization anchor is invalid")
    initialization_matches = {}
    training_contents = {}
    evaluation_preflights = {}
    for arm in ARMS:
        run_dir = Path(registration["study_root"]) / "runs" / arm.slug
        training_content = validate_training_content(
            registration,
            arm,
            run_dir,
            verify_files=True,
        )
        match = read_json(
            (
                Path(registration["study_root"])
                / "runs"
                / arm.slug
                / "initialization_match.json"
            ).resolve(strict=True),
            f"{arm.code} initialization match",
        )
        if (
            match.get("schema") != INITIALIZATION_MATCH_SCHEMA
            or not validate_identity(match)
            or match.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or match.get("arm") != arm.code
            or match.get("anchor_identity_sha256") != anchor["identity_sha256"]
            or match.get("initialization") != anchor.get("initialization")
            or match.get("exact_match") is not True
            or receipts[arm.code].get("initialization_match_identity_sha256")
            != match["identity_sha256"]
            or receipts[arm.code].get("training_content_identity_sha256")
            != training_content["identity_sha256"]
            or receipts[arm.code].get("snapshot_sha256")
            != training_content["snapshot"]["sha256"]
            or receipts[arm.code].get("resolved_config_sha256")
            != training_content["resolved_config"]["sha256"]
            or receipts[arm.code].get("hydra_config_sha256")
            != training_content["hydra_config"]["sha256"]
            or receipts[arm.code].get("training_runtime_sha256")
            != training_content["training_runtime"]["sha256"]
        ):
            raise ContractError(f"{arm.code} initialization match is invalid")
        evaluation_runtime = validate_runtime_receipt(
            run_dir / "evaluation_runtime.json",
            registration,
            label=f"{arm.code} evaluation runtime",
        )
        preflight = validate_evaluation_preflight(
            registration,
            arm,
            run_dir,
            training_content=training_content,
            evaluation_runtime=evaluation_runtime,
        )
        if (
            receipts[arm.code].get("evaluation_preflight_identity_sha256")
            != preflight["identity_sha256"]
        ):
            raise ContractError(f"{arm.code} evaluation preflight is invalid")
        initialization_matches[arm.code] = match
        training_contents[arm.code] = training_content
        evaluation_preflights[arm.code] = preflight
    expected = len(ARMS) * len(SOURCES) * len(NFE_GRID) * EXPECTED_VALIDATION_CLIPS
    if len(rows) != expected:
        raise ContractError(f"result grid has {len(rows)} rows, expected {expected}")
    keyed: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    expected_source_map = {
        "autonomous": "aligned",
        "off": "off",
        "autonomous_future_shuffled": "future_shuffled",
    }
    latency_fields = (
        "history_encode_latency_ms",
        "wan_latency_ms",
        "midpoint_overhead_latency_ms",
        "decode_latency_ms",
        "end_to_end_latency_ms",
        "profiled_internal_end_to_end_latency_ms",
    )
    nonnegative_metrics = (
        "video_nmse",
        "decoded_mse",
        "temporal_mse",
        "auxiliary_future_nmse",
        "auxiliary_dc_nmse",
        "auxiliary_motion_nmse",
    )
    hash_fields = (
        "video_initial_sha256",
        "auxiliary_initial_sha256",
        "video_final_sha256",
        "auxiliary_final_sha256",
        "decoded_final_sha256",
        "raw_target_sha256",
        "snapshot_sha256",
        "hydra_config_sha256",
        "resolved_config_sha256",
        "training_content_identity_sha256",
        "training_runtime_sha256",
        "evaluation_runtime_sha256",
        "initialization_match_identity_sha256",
        "evaluation_preflight_identity_sha256",
    )
    for row in rows:
        key = (
            str(row["arm"]),
            str(row["source"]),
            int(row["nfe"]),
            int(row["clip_index"]),
        )
        if key in keyed:
            raise ContractError(f"duplicate result cell: {key}")
        if (
            key[0] not in ARM_BY_CODE
            or key[1] not in SOURCES
            or key[2] not in NFE_GRID
            or not 0 <= key[3] < EXPECTED_VALIDATION_CLIPS
        ):
            raise ContractError(f"result cell is outside the frozen grid: {key}")
        if (
            row.get("actual_wan_calls") != row.get("nfe")
            or row.get("hook_wan_calls") != row.get("nfe")
            or row.get("artifact_midpoint_head_calls") != row.get("nfe")
            or row.get("hook_midpoint_head_calls") != row.get("nfe")
            or row.get("hook_midpoint_block_calls") != row.get("nfe")
            or row.get("timed_wan_calls") != row.get("nfe")
            or row.get("timed_midpoint_head_calls") != row.get("nfe")
            or row.get("timed_midpoint_block_calls") != row.get("nfe")
            or row.get("equivalence_wan_calls") != row.get("nfe")
            or row.get("equivalence_midpoint_head_calls") != row.get("nfe")
            or row.get("equivalence_midpoint_block_calls") != row.get("nfe")
            or row.get("equivalence_transform_calls") != 0
            or row.get("extra_wan_calls") != 0
            or row.get("evaluation_generations_per_cell") != 4
            or row.get("audit_batch_size") != 2
            or row.get("equivalence_batch_size") != 2
            or row.get("equivalence_exact_decoded_bytes") is not True
            or row.get("timed_batch_size") != 1
            or row.get("total_evaluation_wan_calls") != 4 * row.get("nfe")
            or row.get("equivalence_decoded_sha256")
            != row.get("decoded_final_sha256")
            or row.get("cross_batch_output_comparison_is_diagnostic_only") is not True
            or row.get("wan_block_count") != WAN_BLOCK_COUNT
            or row.get("midpoint_block_index") != MIDPOINT_BLOCK_INDEX
            or row.get("midpoint_condition_source")
            != expected_source_map.get(str(row.get("source")))
            or row.get("generated_clean_stop_gradient") is not True
            or row.get("sampler_transform_calls") != 0
            or row.get("online_teacher_calls") != 0
            or row.get("protected_test_accessed") is not False
            or row.get("deployable") is not True
            or row.get("oracle_leakage") is not False
            or row.get("clean_auxiliary_passed_to_sampler") is not False
            or row.get("future_rgb_passed_to_sampler") is not False
            or row.get("all_auxiliary_bins_initialized_from_noise") is not True
            or row.get("snapshot_sha256")
            != receipts[str(row["arm"])]["snapshot_sha256"]
            or row.get("initialization_match_identity_sha256")
            != receipts[str(row["arm"])]["initialization_match_identity_sha256"]
        ):
            raise ContractError(f"sampler audit failed for result cell: {key}")
        try:
            latencies = [float(row[field]) for field in latency_fields]
            losses = [float(row[field]) for field in nonnegative_metrics]
            auxiliary_cosine = float(row["auxiliary_future_cosine"])
            state_gate = float(row["effective_state_gate"])
            clock_gate = float(row["effective_clock_gate"])
            peak_memory = int(row["peak_memory_allocated_bytes"])
            cross_batch_max = int(row["cross_batch_decoded_max_abs_uint8"])
            cross_batch_mean = float(row["cross_batch_decoded_mean_abs_uint8"])
            cross_batch_fraction = float(
                row["cross_batch_decoded_differing_fraction"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(
                f"metric/latency audit is missing for result cell: {key}"
            ) from exc
        if (
            not all(math.isfinite(value) and value >= 0 for value in latencies)
            or not all(math.isfinite(value) and value >= 0 for value in losses)
            or not math.isfinite(auxiliary_cosine)
            or not -1.0 <= auxiliary_cosine <= 1.0
            or not math.isfinite(state_gate)
            or not -1.0 < state_gate < 1.0
            or clock_gate != 0.0
            or any(
                not isinstance(row.get(field), str)
                or SHA_RE.fullmatch(row[field]) is None
                for field in hash_fields
            )
            or peak_memory <= 0
            or not 0 <= cross_batch_max <= 255
            or not math.isfinite(cross_batch_mean)
            or not 0.0 <= cross_batch_mean <= 255.0
            or not math.isfinite(cross_batch_fraction)
            or not 0.0 <= cross_batch_fraction <= 1.0
            or float(row["midpoint_overhead_latency_ms"])
            > float(row["wan_latency_ms"]) + 1e-6
            or float(row["wan_latency_ms"]) > float(row["end_to_end_latency_ms"]) + 1e-6
            or float(row["profiled_internal_end_to_end_latency_ms"])
            > float(row["end_to_end_latency_ms"]) + 1e-3
            or (
                float(row["history_encode_latency_ms"])
                + float(row["wan_latency_ms"])
                + float(row["decode_latency_ms"])
            )
            > float(row["profiled_internal_end_to_end_latency_ms"]) + 1e-3
        ):
            raise ContractError(f"latency audit is invalid for result cell: {key}")
        keyed[key] = row

    _validate_mid_off_noop(keyed)

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
                        **{
                            field: sum(float(row[field]) for row in cell) / len(cell)
                            for field in latency_fields
                        },
                        "peak_memory_allocated_bytes": max(
                            int(row["peak_memory_allocated_bytes"]) for row in cell
                        ),
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
                seed=(
                    BOOTSTRAP_SEED + 1000 * nfe + 100 * len(comparisons) + metric_index
                ),
            )
            result[metric] = {
                "relative_improvement": point,
                "simultaneous_interval": [low, high],
                "one_sided_familywise_alpha": 0.05,
                "family_comparisons": 9,
            }
        return result

    reference_specs = (
        ("MID-OFF", "autonomous", "trained_mid_off"),
        ("MID-ON", "off", "same_checkpoint_off"),
        (
            "MID-ON",
            "autonomous_future_shuffled",
            "same_checkpoint_future_shuffled_history_preserved",
        ),
    )
    for nfe in NFE_GRID:
        for reference_arm, reference_source, label in reference_specs:
            comparisons.append(
                compare(
                    "MID-ON",
                    "autonomous",
                    reference_arm,
                    reference_source,
                    nfe,
                    label,
                )
            )

    primary = [item for item in comparisons if int(item["candidate"][2]) == 1]
    temporal = all(
        item["temporal_mse"]["relative_improvement"] >= 0.03
        and item["temporal_mse"]["simultaneous_interval"][0] >= 0.01
        for item in primary
    )
    quality_guardrails = all(
        item[metric]["relative_improvement"] >= 0.0
        and item[metric]["simultaneous_interval"][0] > -0.01
        for item in primary
        for metric in ("video_nmse", "decoded_mse")
    )
    decision = {
        "primary_nfe": 1,
        "comparisons": [item["label"] for item in primary],
        "temporal_gate_all_references": temporal,
        "video_and_decoded_guardrails_all_references": quality_guardrails,
        "exact_call_and_provenance_gate": True,
        "latency_gate": False,
        "latency_claim_scope": "descriptive_equal_nfe_only",
        "passes": temporal and quality_guardrails,
    }

    payload = identity_payload(
        {
            "schema": ANALYSIS_SCHEMA,
            "registration_identity_sha256": registration["identity_sha256"],
            "evaluation_receipts": {
                arm_code: {
                    "identity_sha256": receipts[arm_code]["identity_sha256"],
                    "arm_identity_sha256": receipts[arm_code]["arm_identity_sha256"],
                    "rows_sha256": receipts[arm_code]["rows_sha256"],
                    "snapshot_sha256": receipts[arm_code]["snapshot_sha256"],
                    "hydra_config_sha256": receipts[arm_code][
                        "hydra_config_sha256"
                    ],
                    "resolved_config_sha256": receipts[arm_code][
                        "resolved_config_sha256"
                    ],
                    "training_content_identity_sha256": receipts[arm_code][
                        "training_content_identity_sha256"
                    ],
                    "training_runtime_sha256": receipts[arm_code][
                        "training_runtime_sha256"
                    ],
                    "evaluation_runtime_sha256": receipts[arm_code][
                        "evaluation_runtime_sha256"
                    ],
                    "initialization_match_identity_sha256": receipts[arm_code][
                        "initialization_match_identity_sha256"
                    ],
                    "evaluation_preflight_identity_sha256": receipts[arm_code][
                        "evaluation_preflight_identity_sha256"
                    ],
                    "rows": receipts[arm_code]["rows"],
                }
                for arm_code in sorted(receipts)
            },
            "rows": len(rows),
            "summaries": summaries,
            "comparisons": comparisons,
            "initialization_anchor_identity_sha256": anchor["identity_sha256"],
            "initialization_match_identities": {
                arm: match["identity_sha256"]
                for arm, match in sorted(initialization_matches.items())
            },
            "evaluation_preflight_identities": {
                arm: preflight["identity_sha256"]
                for arm, preflight in sorted(evaluation_preflights.items())
            },
            "decision": decision,
            "conclusion": (
                "pass_one_call_future_scratchpad_forcing_screen"
                if decision["passes"]
                else "no_controlled_one_call_intra_forward_advantage_in_quick_screen"
            ),
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

    bind_initialization = sub.add_parser("bind-initialization")
    bind_initialization.add_argument("--registration", type=Path, required=True)
    bind_initialization.add_argument("--array-task-id", type=int, required=True)
    bind_initialization.add_argument("--run-dir", type=Path, required=True)
    bind_initialization.add_argument("--preflight", type=Path, required=True)
    bind_initialization.set_defaults(func=command_bind_initialization)

    verify_memory = sub.add_parser("verify-memory-smoke")
    verify_memory.add_argument("--registration", type=Path, required=True)
    verify_memory.set_defaults(func=command_verify_memory_smoke)

    bind_training = sub.add_parser("bind-training-output")
    bind_training.add_argument("--registration", type=Path, required=True)
    bind_training.add_argument("--array-task-id", type=int, required=True)
    bind_training.add_argument("--run-dir", type=Path, required=True)
    bind_training.set_defaults(func=command_bind_training_output)

    bind_evaluation = sub.add_parser("bind-evaluation-input")
    bind_evaluation.add_argument("--registration", type=Path, required=True)
    bind_evaluation.add_argument("--array-task-id", type=int, required=True)
    bind_evaluation.add_argument("--run-dir", type=Path, required=True)
    bind_evaluation.add_argument("--runtime-receipt", type=Path, required=True)
    bind_evaluation.set_defaults(func=command_bind_evaluation_input)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--registration", type=Path, required=True)
    analyze.add_argument("--rows", type=Path, nargs=2, required=True)
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
