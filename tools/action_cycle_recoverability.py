#!/usr/bin/env python3
"""Prospective clean-Wan-latent action-recoverability screen.

This Stage-0 diagnostic asks a deliberately narrower question than video
generation: does a clean, frozen Wan VAE latent displacement retain the action
segment that caused it?  A positive answer is a prerequisite for using a
train-only inverse-action critic as an auxiliary objective.  Validation data
never participates in normalization, ridge fitting, or regularization
selection, and no V-JEPA target or protected-test artifact is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
REGISTRATION_KIND = "action-cycle-recoverability-registration-v1"
ENCODING_KIND = "action-cycle-recoverability-wan-encoding-v1"
ANALYSIS_KIND = "action-cycle-recoverability-analysis-v1"
MODEL_KIND = "action-cycle-recoverability-ridge-v1"
SUBMISSION_KIND = "action-cycle-recoverability-submission-v1"

EXPECTED_ENTITY = "zijiandu"
EXPECTED_PROJECT = "dual-video-diffusion-private"
EXPECTED_LUSTRE_BASE = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train"
)
RUN_ROOT_ENVIRONMENT = "LACWM_ALLOWED_RUN_ROOTS"
EXPECTED_VIDEOX_COMMIT = "1d6d9c3e1540968466937129fef4b288041e06de"
EXPECTED_VAE_SHA256 = (
    "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
)
EXPECTED_VAE_CONFIG_SHA256 = (
    "21fe4409b664385a1c1cc5c23d92506ffb05ef3c374a18de9df67b715dca07e9"
)
EXPECTED_VAE_SOURCE_SHA256 = (
    "3e28a03770cbbb5ccbc37151fb60d2faf17ccfb9c679cb5eacd0d7bf0886a4b1"
)

TRAIN_CLIPS = 512
VALIDATION_CLIPS = 64
SAMPLE_FRAMES = 13
CAMERAS = ("top", "left_wrist", "right_wrist")
RGB_SHAPE = (SAMPLE_FRAMES, 3, 180, 960)
ACTION_SHAPE = (SAMPLE_FRAMES, 5, 23)
LATENT_SHAPE = (16, 4, 24, 120)
PER_VIEW_LATENT_SHAPE = (16, 4, 24, 40)
VAE_FRAME_SUPPORTS = ((0, 1), (1, 5), (5, 9), (9, 13))
ACTION_CHUNK_INTERVALS = ((0, 4), (4, 8), (8, 12))
TEMPORALLY_MISALIGNED_ACTION_CHUNK_INTERVALS = ((4, 8), (8, 12), (0, 4))
TEMPORAL_DONOR_TRANSITIONS = (1, 2, 0)
ALL_TRANSITIONS = (0, 1, 2)
FUTURE_RELEVANT_TRANSITIONS = (1, 2)
POOL_SHAPE = (6, 10)
FEATURE_DIM = LATENT_SHAPE[0] * POOL_SHAPE[0] * POOL_SHAPE[1]
TARGET_DIM = 4 * ACTION_SHAPE[1] * ACTION_SHAPE[2]
ALPHA_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_808
SHUFFLE_SEED = 20_260_809
STD_FLOOR = 1.0e-6
PREFIX_ATOL = 2.0e-5
PREFIX_RTOL = 1.0e-5
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
JOB_ID_RE = re.compile(r"^[1-9][0-9]*(?:_[0-9]+)?$")
PROTECTED_COMPONENTS = {"test", "tests", "protected_test", "protected-test"}


class ActionCycleProbeError(RuntimeError):
    """Raised when a prospective protocol or artifact boundary differs."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(
            canonical_json(unsigned).encode("utf-8")
        ).hexdigest(),
    }


def identity_valid(payload: Mapping[str, Any]) -> bool:
    identity = payload.get("identity_sha256")
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return bool(
        isinstance(identity, str)
        and SHA256_RE.fullmatch(identity)
        and hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()
        == identity
    )


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def numpy_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_sha256(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise ActionCycleProbeError(f"{label} must be a full lowercase SHA-256")
    return normalized


def validate_commit(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not COMMIT_RE.fullmatch(normalized):
        raise ActionCycleProbeError(f"{label} must be a full lowercase git commit")
    return normalized


def reject_protected_path(path: Path, *, label: str) -> None:
    if any(part.lower() in PROTECTED_COMPONENTS for part in path.parts):
        raise ActionCycleProbeError(f"{label} may not reference protected-test data")


def regular_file(value: str | Path, *, label: str, protected_ok: bool = False) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ActionCycleProbeError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ActionCycleProbeError(f"{label} must be a nonempty non-symlink regular file")
    path = path.resolve(strict=True)
    if not protected_ok:
        reject_protected_path(path, label=label)
    return path


def regular_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ActionCycleProbeError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ActionCycleProbeError(f"{label} must be a non-symlink directory")
    path = path.resolve(strict=True)
    reject_protected_path(path, label=label)
    return path


def python_executable(value: str | Path) -> tuple[Path, Path]:
    launcher = Path(value).expanduser().absolute()
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise ActionCycleProbeError("runtime Python launcher is not executable")
    target = launcher.resolve(strict=True)
    if not target.is_file():
        raise ActionCycleProbeError("runtime Python target is not a file")
    return launcher, target


def file_record(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    path = regular_file(path, label="recorded file", protected_ok=True)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != validate_sha256(
        expected_sha256, label=f"expected digest for {path}"
    ):
        raise ActionCycleProbeError(f"file digest differs for {path}")
    info = path.stat()
    return {"path": str(path), "sha256": digest, "bytes": int(info.st_size)}


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    path = regular_file(path, label=label, protected_ok=True)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionCycleProbeError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ActionCycleProbeError(f"{label} must contain a JSON object")
    return value


def git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True,
        check=False,
    )
    if completed.returncode:
        raise ActionCycleProbeError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def assert_clean_commit(repo: Path, expected: str, *, label: str) -> None:
    expected = validate_commit(expected, label=f"expected {label} commit")
    if git(repo, "rev-parse", "HEAD") != expected:
        raise ActionCycleProbeError(f"{label} HEAD differs from the registered commit")
    status = git(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ActionCycleProbeError(f"{label} repository is dirty: {status.replace(chr(10), '; ')}")


def validate_temporal_control_contract() -> None:
    if tuple(
        ACTION_CHUNK_INTERVALS[index] for index in TEMPORAL_DONOR_TRANSITIONS
    ) != TEMPORALLY_MISALIGNED_ACTION_CHUNK_INTERVALS:
        raise ActionCycleProbeError("temporal control is not the declared cyclic donor")
    for aligned, negative in zip(
        ACTION_CHUNK_INTERVALS, TEMPORALLY_MISALIGNED_ACTION_CHUNK_INTERVALS
    ):
        if set(range(*aligned)) & set(range(*negative)):
            raise ActionCycleProbeError(
                "temporal control must not overlap its recipient action window"
            )


def fixed_protocol() -> dict[str, Any]:
    validate_temporal_control_contract()
    gate_thresholds = {
        "normalized_mse_relative_improvement": 0.20,
        "cosine_absolute_gain": 0.10,
        "retrieval_accuracy_gain": 0.0,
        "simultaneous_lower_bound": 0.0,
    }
    return {
        "stage": "prospective_stage0_clean_wan_latent_action_recoverability",
        "scientific_scope": (
            "prerequisite_only_not_video_generation_quality_or_inference_speed"
        ),
        "seed": SHUFFLE_SEED,
        "train_clips": TRAIN_CLIPS,
        "validation_clips": VALIDATION_CLIPS,
        "protected_test_access_allowed": False,
        "cached_vjepa_target_open_allowed": False,
        "rgb_shape": list(RGB_SHAPE),
        "action_shape": list(ACTION_SHAPE),
        "camera_order": list(CAMERAS),
        "vae": {
            "latent_shape": list(LATENT_SHAPE),
            "deterministic_mode_not_sample": True,
            "causal_input_blocks": [list(value) for value in VAE_FRAME_SUPPORTS],
            "actual_runtime_prefix_equivalence_required": True,
            "prefix_atol": PREFIX_ATOL,
            "prefix_rtol": PREFIX_RTOL,
            "encode_panorama_then_split_latent_views": True,
        },
        "alignment": {
            "latent_displacements": [[index, index + 1] for index in ALL_TRANSITIONS],
            "action_chunk_intervals": [list(value) for value in ACTION_CHUNK_INTERVALS],
            "action_low_level_intervals": [[20 * i, 20 * (i + 1)] for i in ALL_TRANSITIONS],
            "unused_terminal_action_chunk": 12,
            "all_transition_primary": list(ALL_TRANSITIONS),
            "future_relevant_guardrail": list(FUTURE_RELEVANT_TRANSITIONS),
        },
        "feature": {
            "formula": "avgpool_6x10(LN(z[b+1]))-avgpool_6x10(LN(z[b]))",
            "layer_norm_axes": ["channel", "height", "per_view_width"],
            "pool_shape": list(POOL_SHAPE),
            "feature_dim": FEATURE_DIM,
            "views_fitted_separately_then_prediction_averaged": True,
        },
        "target": {
            "formula": "flatten(actions[:,4*b:4*(b+1),:,:])",
            "shape_per_transition": [4, ACTION_SHAPE[1], ACTION_SHAPE[2]],
            "target_dim": TARGET_DIM,
            "semantic_name": "action_segment_aligned_to_latent_displacement",
            "within_chunk_differencing_forbidden": True,
            "padding_to_157_excluded_from_metrics": True,
            "abc_morphology_is_constant": True,
        },
        "ridge": {
            "fit_split": "train_only",
            "feature_and_target_statistics": "train_population_only",
            "inactive_std_floor": STD_FLOOR,
            "kernel": "standardized_linear_kernel_divided_by_active_feature_count",
            "alpha_grid": list(ALPHA_GRID),
            "selection": "exact_train_only_leave_one_out_formula_mean_over_9_strata",
            "validation_observations_used_for_fit_or_selection": 0,
        },
        "controls": {
            "train_transition_mean": True,
            "episode_disjoint_train_label_shuffle_fit": True,
            "episode_disjoint_validation_target_shuffle": True,
            "shuffle_is_bijective_and_deterministic": True,
            "same_clip_task_matched_temporal_misalignment": {
                "action_chunk_intervals": [
                    list(value) for value in TEMPORALLY_MISALIGNED_ACTION_CHUNK_INTERVALS
                ],
                "donor_transition_by_recipient": list(TEMPORAL_DONOR_TRANSITIONS),
                "nonoverlapping_with_recipient": True,
                "cyclic_transition_donor": True,
                "same_clip_episode_and_task": True,
                "validation_scoring_control_only": True,
                "train_only_oracle_feasibility_required_before_registration": True,
                "oracle_minimum_cosine_gap": 0.10,
            },
        },
        "bootstrap": {
            "unit": "validation_clip_with_views_and_transitions_aggregated_inside_clip",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "familywise_confidence": 0.95,
            "method": "one_sided_bonferroni_paired_percentile",
        },
        "consumption_integrity": {
            "full_sha256_before_and_after": [
                "train_rgb_encode", "validation_rgb_encode",
                "train_features_analysis", "validation_features_analysis",
                "train_actions_analysis", "validation_actions_analysis",
            ],
            "boundary_identity_fields": [
                "device", "inode", "bytes", "mtime_ns", "sha256"
            ],
            "unchanged_window_required": True,
        },
        "gate_thresholds": gate_thresholds,
        "go_requires": (
            "all preregistered primary and future-relevant comparisons meet point "
            "threshold and simultaneous lower bound above zero"
        ),
        "distributed_encode_world_size": 8,
        "analysis_device": "cpu",
        "analysis_scheduler_bookkeeping_gpus": 1,
        "analysis_cuda_usage_allowed": False,
        "wandb": {
            "entity": EXPECTED_ENTITY,
            "project": EXPECTED_PROJECT,
            "access": "PRIVATE",
            "group": None,
        },
    }


def _manifest_rows(path: Path, *, split: str, count: int) -> list[dict[str, Any]]:
    path = regular_file(path, label=f"{split} clip manifest")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ActionCycleProbeError(f"invalid manifest row {line_number}: {exc}") from exc
        required = {
            "clip_id", "episode_dir", "start", "auxiliary_index", "frame_indices",
            "sample_size", "chunk_size", "action_span", "split",
        }
        if not isinstance(row, dict) or not required.issubset(row):
            raise ActionCycleProbeError(f"manifest row {line_number} lacks required fields")
        rows.append(row)
    if len(rows) != count:
        raise ActionCycleProbeError(f"{split} manifest must contain {count} clips")
    clip_ids: set[str] = set()
    for index, row in enumerate(rows):
        if (
            row["split"] != split
            or int(row["auxiliary_index"]) != index
            or int(row["sample_size"]) != SAMPLE_FRAMES
            or int(row["chunk_size"]) != 5
            or int(row["action_span"]) != 65
            or [int(value) for value in row["frame_indices"]]
            != list(range(int(row["start"]), int(row["start"]) + 65, 5))
        ):
            raise ActionCycleProbeError(f"manifest row {index} violates clip/action alignment")
        episode = regular_directory(row["episode_dir"], label=f"manifest episode {index}")
        row["episode_dir"] = str(episode)
        clip_id = str(row["clip_id"])
        if not SHA256_RE.fullmatch(clip_id) or clip_id in clip_ids:
            raise ActionCycleProbeError("manifest clip IDs must be unique SHA-256 values")
        clip_ids.add(clip_id)
    return rows


def _resolve_cache_array(metadata_path: Path, metadata: Mapping[str, Any], key: str) -> Path:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ActionCycleProbeError(f"cache metadata lacks {key}")
    path = Path(value)
    if not path.is_absolute():
        path = metadata_path.parent / path
    return regular_file(path, label=f"cache {key}")


def validate_cache_inputs(
    metadata_path: Path,
    manifest_path: Path,
    *,
    split: str,
    count: int,
    full_hash: bool,
) -> dict[str, Any]:
    metadata_path = regular_file(metadata_path, label=f"{split} cache metadata")
    manifest_path = regular_file(manifest_path, label=f"{split} clip manifest")
    metadata = read_json(metadata_path, label=f"{split} cache metadata")
    rows = _manifest_rows(manifest_path, split=split, count=count)
    expected_rgb = [count, *RGB_SHAPE]
    expected_actions = [count, *ACTION_SHAPE]
    if (
        metadata.get("artifact_type") != "vjepa2.1-wan-grid-cache"
        or metadata.get("complete") is not True
        or metadata.get("split") != split
        or int(metadata.get("clip_count", -1)) != count
        or metadata.get("rgb_shape") != expected_rgb
        or metadata.get("actions_shape") != expected_actions
        or metadata.get("rgb_dtype") != "float16"
        or metadata.get("actions_dtype") != "float32"
        or metadata.get("frame_offsets") != list(range(0, 65, 5))
        or metadata.get("camera_order") != list(CAMERAS)
        or int(metadata.get("sample_size", -1)) != SAMPLE_FRAMES
        or int(metadata.get("chunk_size", -1)) != 5
        or int(metadata.get("action_span", -1)) != 65
    ):
        raise ActionCycleProbeError(f"{split} cache metadata violates the fixed contract")
    manifest_digest = sha256_file(manifest_path)
    if metadata.get("clip_manifest_sha256") != manifest_digest:
        raise ActionCycleProbeError(f"{split} cache/manifest digest mismatch")
    rgb_path = _resolve_cache_array(metadata_path, metadata, "rgb_file")
    actions_path = _resolve_cache_array(metadata_path, metadata, "actions_file")
    expected_rgb_sha = validate_sha256(metadata.get("rgb_sha256", ""), label="RGB cache digest")
    expected_action_sha = validate_sha256(
        metadata.get("actions_sha256", ""), label="action cache digest"
    )
    rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    actions = np.load(actions_path, mmap_mode="r", allow_pickle=False)
    if tuple(rgb.shape) != tuple(expected_rgb) or rgb.dtype != np.float16:
        raise ActionCycleProbeError(f"{split} RGB cache shape or dtype differs")
    if tuple(actions.shape) != tuple(expected_actions) or actions.dtype != np.float32:
        raise ActionCycleProbeError(f"{split} action cache shape or dtype differs")
    probe_indices = sorted({0, count - 1})
    for index in probe_indices:
        rgb_row = np.asarray(rgb[index])
        action_row = np.asarray(actions[index])
        if (
            not np.isfinite(rgb_row).all()
            or float(rgb_row.min()) < -1.0
            or float(rgb_row.max()) > 1.0
            or not np.isfinite(action_row).all()
        ):
            raise ActionCycleProbeError(f"{split} cache contains invalid values")
    del rgb, actions
    if full_hash:
        if sha256_file(rgb_path) != expected_rgb_sha:
            raise ActionCycleProbeError(f"{split} RGB cache bytes changed")
        if sha256_file(actions_path) != expected_action_sha:
            raise ActionCycleProbeError(f"{split} action cache bytes changed")
    return {
        "split": split,
        "count": count,
        "metadata": file_record(metadata_path),
        "manifest": file_record(manifest_path),
        "rgb": {
            "path": str(rgb_path), "sha256": expected_rgb_sha,
            "bytes": int(rgb_path.stat().st_size), "shape": expected_rgb, "dtype": "float16",
        },
        "actions": {
            "path": str(actions_path), "sha256": expected_action_sha,
            "bytes": int(actions_path.stat().st_size), "shape": expected_actions,
            "dtype": "float32",
        },
        "descriptors": [
            {
                "index": i,
                "clip_id": str(row["clip_id"]),
                "episode_dir": str(row["episode_dir"]),
                "task_label": Path(str(row["episode_dir"])).parent.name,
                "start": int(row["start"]),
                "frame_indices": [int(value) for value in row["frame_indices"]],
            }
            for i, row in enumerate(rows)
        ],
        "target_cache_array_opened": False,
        "protected_test_accessed": False,
    }


def validate_train_val_disjoint(train: Mapping[str, Any], val: Mapping[str, Any]) -> None:
    for field in ("clip_id", "episode_dir"):
        left = {str(row[field]) for row in train["descriptors"]}
        right = {str(row[field]) for row in val["descriptors"]}
        if left & right:
            raise ActionCycleProbeError(f"train and validation overlap by {field}")


def _tool_records() -> dict[str, Any]:
    paths = {
        "tool": REPO_ROOT / "tools/action_cycle_recoverability.py",
        "run_root_policy": REPO_ROOT / "tools/run_root_policy.py",
        "encode_slurm": REPO_ROOT / "tools/slurm/action_cycle_recoverability_encode.sbatch",
        "analysis_slurm": REPO_ROOT / "tools/slurm/action_cycle_recoverability_analyze.sbatch",
        "submit": REPO_ROOT / "tools/slurm/submit_action_cycle_recoverability.sh",
        "protocol": REPO_ROOT / "docs/experiments/ACTION_CYCLE_RECOVERABILITY_PROTOCOL.md",
        "runbook": REPO_ROOT / "docs/experiments/ACTION_CYCLE_RECOVERABILITY_RUNBOOK.md",
    }
    return {name: file_record(path) for name, path in paths.items()}


def stage_identity(registration: Mapping[str, Any], stage: str) -> str:
    return identity_payload(
        {
            "kind": "action-cycle-recoverability-stage-identity-v1",
            "registration_identity_sha256": registration["identity_sha256"],
            "stage": stage,
        }
    )["identity_sha256"]


def _wandb_private_project() -> dict[str, Any]:
    """Verify that the authenticated W&B viewer owns the private project."""

    try:
        import wandb
        from wandb_gql import gql
    except ImportError as exc:
        raise ActionCycleProbeError("W&B packages are unavailable") from exc
    api = wandb.Api(timeout=30)
    query = gql(
        """
        query ActionCycleProjectAccess($entity: String!, $project: String!) {
          project(name: $project, entityName: $entity) {
            id
            name
            entityName
            access
          }
          viewer {
            username
            email
          }
        }
        """
    )
    result = api.client.execute(
        query,
        variable_values={"entity": EXPECTED_ENTITY, "project": EXPECTED_PROJECT},
    )
    project = result.get("project")
    viewer = result.get("viewer")
    if (
        not isinstance(project, dict)
        or project.get("entityName") != EXPECTED_ENTITY
        or project.get("name") != EXPECTED_PROJECT
        or str(project.get("access", "")).upper() != "PRIVATE"
        or not isinstance(viewer, dict)
        or viewer.get("username") != EXPECTED_ENTITY
    ):
        raise ActionCycleProbeError("W&B personal-project privacy verification failed")
    return {
        "entity": EXPECTED_ENTITY,
        "project": EXPECTED_PROJECT,
        "access": "PRIVATE",
        "viewer_username": EXPECTED_ENTITY,
        "group": None,
        "mode": "online",
    }


def _canonical_new_root(path: Path) -> Path:
    from tools.run_root_policy import canonical_allowed_run_root

    root = canonical_allowed_run_root(path)
    if root.exists() or root.is_symlink():
        raise ActionCycleProbeError("output root must be fresh and absent")
    if not root.parent.is_dir():
        raise ActionCycleProbeError("output root parent must already exist")
    return root


def validate_lustre_run_root_environment(value: str | Path) -> Path:
    """Require the one reviewed canonical Lustre root and no broader override."""

    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested != EXPECTED_LUSTRE_BASE:
        raise ActionCycleProbeError(
            f"Lustre base must be exactly {EXPECTED_LUSTRE_BASE}"
        )
    base = regular_directory(requested, label="registered Lustre base")
    if base != requested:
        raise ActionCycleProbeError("registered Lustre base is not canonical")
    configured = os.environ.get(RUN_ROOT_ENVIRONMENT)
    if configured != str(base):
        raise ActionCycleProbeError(
            f"{RUN_ROOT_ENVIRONMENT} must contain exactly the reviewed canonical Lustre base"
        )
    from tools.run_root_policy import configured_allowed_run_roots

    allowed = configured_allowed_run_roots()
    if base not in allowed or Path("/") in allowed:
        raise ActionCycleProbeError("canonical run-root policy did not admit only safe roots")
    return base


def command_register(args: argparse.Namespace) -> int:
    repo = regular_directory(args.repo, label="probe repository")
    if repo != REPO_ROOT:
        raise ActionCycleProbeError("registration must run from this exact repository")
    expected_commit = validate_commit(args.expected_commit, label="probe source commit")
    assert_clean_commit(repo, expected_commit, label="probe")
    lustre_base = validate_lustre_run_root_environment(args.lustre_base)
    output_root = _canonical_new_root(args.output_root)
    study_parent = lustre_base / "artifacts/dual_video_diffusion/action_cycle_recoverability"
    if study_parent not in output_root.parents:
        raise ActionCycleProbeError(
            f"study root must be a strict descendant of {study_parent}"
        )

    train = validate_cache_inputs(
        args.train_metadata, args.train_manifest, split="train", count=TRAIN_CLIPS,
        full_hash=True,
    )
    val = validate_cache_inputs(
        args.validation_metadata, args.validation_manifest, split="val",
        count=VALIDATION_CLIPS, full_hash=True,
    )
    validate_train_val_disjoint(train, val)
    train_actions_for_oracle = np.load(
        train["actions"]["path"], mmap_mode="r", allow_pickle=False
    )
    temporal_oracle = temporal_control_oracle_feasibility(
        train_actions_for_oracle
    )
    del train_actions_for_oracle
    validate_temporal_oracle_payload(temporal_oracle)

    videox = regular_directory(args.videox_home, label="VideoX-Fun repository")
    assert_clean_commit(videox, EXPECTED_VIDEOX_COMMIT, label="VideoX-Fun")
    wan = regular_directory(args.wan_dir, label="Wan asset directory")
    vae = regular_file(wan / "Wan2.1_VAE.pth", label="Wan VAE checkpoint")
    config = regular_file(
        videox / "config/wan2.1/wan_civitai.yaml", label="Wan VAE config"
    )
    source = regular_file(
        videox / "videox_fun/models/wan_vae.py", label="Wan VAE source"
    )
    vae_record = file_record(vae, expected_sha256=EXPECTED_VAE_SHA256)
    config_record = file_record(config, expected_sha256=EXPECTED_VAE_CONFIG_SHA256)
    source_record = file_record(source, expected_sha256=EXPECTED_VAE_SOURCE_SHA256)
    launcher, python_target = python_executable(args.python)
    wandb = _wandb_private_project()

    # Remove bulky descriptors from the cache record only after independently
    # binding them in the registration; analysis needs the exact ordered rows.
    payload = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": REGISTRATION_KIND,
            "created_at_utc": now_utc(),
            "status": "registered_before_encoding_or_validation_metrics",
            "output_root": str(output_root),
            "run_root_policy": {
                "environment_variable": RUN_ROOT_ENVIRONMENT,
                "configured_value": str(lustre_base),
                "canonical_lustre_base": str(lustre_base),
                "required_study_parent": str(study_parent),
            },
            "repository": {
                "path": str(repo),
                "commit": expected_commit,
                "clean": True,
            },
            "tools": _tool_records(),
            "runtime": {
                "python": str(launcher),
                "python_target": file_record(python_target),
                "videox_home": str(videox),
                "videox_commit": EXPECTED_VIDEOX_COMMIT,
                "wan_dir": str(wan),
                "vae_checkpoint": vae_record,
                "vae_config": config_record,
                "vae_source": source_record,
            },
            "inputs": {"train": train, "val": val},
            "train_validation_disjoint": {
                "clip_id": True,
                "episode_dir": True,
            },
            "train_only_temporal_control_oracle_feasibility": temporal_oracle,
            "fixed_protocol": fixed_protocol(),
            "wandb": wandb,
            "target_cache_array_opened": False,
            "protected_test_accessed": False,
        }
    )
    output_root.mkdir(mode=0o700)
    for relative in ("artifacts/encoded", "analysis", "logs", "wandb", "wandb-cache", "wandb-config"):
        (output_root / relative).mkdir(parents=True, mode=0o700)
    exclusive_json(output_root / "registration.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _rehash_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = regular_file(record.get("path", ""), label=label, protected_ok=True)
    if (
        record.get("sha256") != sha256_file(path)
        or int(record.get("bytes", -1)) != int(path.stat().st_size)
    ):
        raise ActionCycleProbeError(f"registered {label} changed")
    return path


def full_integrity_rehash(
    record: Mapping[str, Any],
    *,
    label: str,
    registration_identity_sha256: str,
    phase: str,
) -> dict[str, Any]:
    """Hash every byte and reject mutation during the verification window."""

    if phase not in {"immediate_preconsumption", "immediate_postconsumption"}:
        raise ActionCycleProbeError("full-integrity rehash phase is invalid")

    path = regular_file(record.get("path", ""), label=label, protected_ok=True)
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    fingerprint_before = (
        int(before.st_dev), int(before.st_ino), int(before.st_size), int(before.st_mtime_ns)
    )
    fingerprint_after = (
        int(after.st_dev), int(after.st_ino), int(after.st_size), int(after.st_mtime_ns)
    )
    if (
        fingerprint_before != fingerprint_after
        or digest != record.get("sha256")
        or int(after.st_size) != int(record.get("bytes", -1))
    ):
        raise ActionCycleProbeError(
            f"{label} changed, including a possible same-size middle mutation"
        )
    return identity_payload(
        {
            "kind": "action-cycle-consumption-boundary-full-rehash-v1",
            "created_at_utc": now_utc(),
            "registration_identity_sha256": registration_identity_sha256,
            "label": label,
            "phase": phase,
            "path": str(path),
            "sha256": digest,
            "bytes": int(after.st_size),
            "device": int(after.st_dev),
            "inode": int(after.st_ino),
            "mtime_ns": int(after.st_mtime_ns),
            "full_file_hashed": True,
            "protected_test_accessed": False,
        }
    )


def full_preconsumption_rehash(
    record: Mapping[str, Any],
    *,
    label: str,
    registration_identity_sha256: str,
) -> dict[str, Any]:
    return full_integrity_rehash(
        record,
        label=label,
        registration_identity_sha256=registration_identity_sha256,
        phase="immediate_preconsumption",
    )


def full_postconsumption_rehash(
    record: Mapping[str, Any],
    *,
    label: str,
    registration_identity_sha256: str,
) -> dict[str, Any]:
    return full_integrity_rehash(
        record,
        label=label,
        registration_identity_sha256=registration_identity_sha256,
        phase="immediate_postconsumption",
    )


def validate_consumption_window(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    invariant_fields = (
        "registration_identity_sha256", "path", "sha256", "bytes", "device",
        "inode", "mtime_ns",
    )
    if (
        not identity_valid(before)
        or not identity_valid(after)
        or before.get("phase") != "immediate_preconsumption"
        or after.get("phase") != "immediate_postconsumption"
        or before.get("full_file_hashed") is not True
        or after.get("full_file_hashed") is not True
        or any(before.get(field) != after.get(field) for field in invariant_fields)
    ):
        raise ActionCycleProbeError(
            "file identity changed across the full memmap consumption window"
        )


def validate_registration(path: Path, *, full_hash: bool) -> dict[str, Any]:
    registration_path = regular_file(path, label="probe registration", protected_ok=True)
    registration = read_json(registration_path, label="probe registration")
    if (
        not identity_valid(registration)
        or registration.get("kind") != REGISTRATION_KIND
        or registration.get("status") != "registered_before_encoding_or_validation_metrics"
        or registration.get("fixed_protocol") != fixed_protocol()
        or registration.get("target_cache_array_opened") is not False
        or registration.get("protected_test_accessed") is not False
        or registration.get("wandb", {}).get("entity") != EXPECTED_ENTITY
        or registration.get("wandb", {}).get("project") != EXPECTED_PROJECT
        or registration.get("wandb", {}).get("access") != "PRIVATE"
        or registration.get("wandb", {}).get("viewer_username") != EXPECTED_ENTITY
        or registration.get("wandb", {}).get("group") is not None
    ):
        raise ActionCycleProbeError("registration identity or protocol differs")
    validate_temporal_oracle_payload(
        registration.get("train_only_temporal_control_oracle_feasibility", {})
    )
    output_root = Path(registration.get("output_root", "")).resolve(strict=True)
    if registration_path != (output_root / "registration.json").resolve(strict=True):
        raise ActionCycleProbeError("registration is not at its canonical output root")
    run_root = registration.get("run_root_policy", {})
    lustre_base = validate_lustre_run_root_environment(
        run_root.get("canonical_lustre_base", "")
    )
    expected_parent = lustre_base / "artifacts/dual_video_diffusion/action_cycle_recoverability"
    if (
        run_root.get("environment_variable") != RUN_ROOT_ENVIRONMENT
        or run_root.get("configured_value") != str(lustre_base)
        or run_root.get("required_study_parent") != str(expected_parent)
        or expected_parent not in output_root.parents
    ):
        raise ActionCycleProbeError("registered canonical run-root policy differs")
    repo = regular_directory(registration["repository"]["path"], label="probe repository")
    if repo != REPO_ROOT:
        raise ActionCycleProbeError("registration repository differs from executing source")
    assert_clean_commit(repo, registration["repository"]["commit"], label="probe")
    for name, record in registration["tools"].items():
        _rehash_record(record, label=f"tool {name}")

    runtime = registration["runtime"]
    launcher, target = python_executable(runtime["python"])
    if str(launcher) != runtime["python"]:
        raise ActionCycleProbeError("runtime Python launcher spelling changed")
    _rehash_record(runtime["python_target"], label="runtime Python target")
    if target != Path(runtime["python_target"]["path"]):
        raise ActionCycleProbeError("runtime Python target changed")
    videox = regular_directory(runtime["videox_home"], label="VideoX-Fun repository")
    assert_clean_commit(videox, runtime["videox_commit"], label="VideoX-Fun")
    for key in ("vae_checkpoint", "vae_config", "vae_source"):
        _rehash_record(runtime[key], label=key)

    for split, count in (("train", TRAIN_CLIPS), ("val", VALIDATION_CLIPS)):
        registered_input = registration["inputs"][split]
        observed = validate_cache_inputs(
            Path(registered_input["metadata"]["path"]),
            Path(registered_input["manifest"]["path"]),
            split=split,
            count=count,
            full_hash=full_hash,
        )
        # Metadata/manifest content and the recorded array identities must stay
        # exact. Descriptors are compared as ordered scientific inputs.
        if observed != registered_input:
            raise ActionCycleProbeError(f"registered {split} inputs changed")
    validate_train_val_disjoint(
        registration["inputs"]["train"], registration["inputs"]["val"]
    )
    return registration


def command_validate_registration(args: argparse.Namespace) -> int:
    registration = validate_registration(args.registration, full_hash=args.full)
    print(
        json.dumps(
            {
                "registration_identity_sha256": registration["identity_sha256"],
                "full_hash": bool(args.full),
                "valid": True,
            },
            sort_keys=True,
        )
    )
    return 0


def command_wandb_check(_args: argparse.Namespace) -> int:
    print(json.dumps(_wandb_private_project(), sort_keys=True))
    return 0


def latent_displacement_features(latents: torch.Tensor) -> torch.Tensor:
    """Return fixed per-view features with shape ``[B,3,3,960]``.

    Wan sees the exact three-view panorama used by LACWM.  Only after encoding
    do we split its 120-wide latent grid into three 40-wide views.  Each
    endpoint is independently layer-normalized over C/H/per-view-W before a
    fixed average pool; subtracting adjacent endpoints yields all three causal
    latent displacements.
    """

    if latents.ndim != 5 or tuple(latents.shape[1:]) != LATENT_SHAPE:
        raise ActionCycleProbeError(
            f"Wan latent must have shape [B,{','.join(map(str, LATENT_SHAPE))}]"
        )
    batch, channels, bins, height, width = latents.shape
    views = len(CAMERAS)
    if width % views:
        raise ActionCycleProbeError("Wan latent width is not divisible by camera count")
    per_view_width = width // views
    view = latents.float().reshape(
        batch, channels, bins, height, views, per_view_width
    ).permute(0, 2, 4, 1, 3, 5)
    mean = view.mean(dim=(3, 4, 5), keepdim=True)
    variance = (view - mean).square().mean(dim=(3, 4, 5), keepdim=True)
    normalized = (view - mean) * torch.rsqrt(variance + 1.0e-6)
    pooled = F.adaptive_avg_pool2d(
        normalized.reshape(batch * bins * views, channels, height, per_view_width),
        POOL_SHAPE,
    ).reshape(batch, bins, views, channels, *POOL_SHAPE)
    delta = pooled[:, 1:] - pooled[:, :-1]
    result = delta.flatten(3).contiguous()
    expected = (batch, len(ALL_TRANSITIONS), views, FEATURE_DIM)
    if tuple(result.shape) != expected or not bool(torch.isfinite(result).all()):
        raise ActionCycleProbeError("latent displacement feature contract failed")
    return result


def action_targets_for_intervals(
    actions: np.ndarray | torch.Tensor,
    intervals: Sequence[tuple[int, int]],
) -> np.ndarray:
    value = np.asarray(actions)
    if value.ndim != 4 or tuple(value.shape[1:]) != ACTION_SHAPE:
        raise ActionCycleProbeError("actions must have shape [N,13,5,23]")
    if len(intervals) != 3 or any(stop - start != 4 for start, stop in intervals):
        raise ActionCycleProbeError("action-target intervals must be three 4-chunk windows")
    targets = np.stack(
        [value[:, start:stop].reshape(value.shape[0], -1) for start, stop in intervals],
        axis=1,
    ).astype(np.float64, copy=False)
    if targets.shape != (value.shape[0], 3, TARGET_DIM) or not np.isfinite(targets).all():
        raise ActionCycleProbeError("aligned action target construction failed")
    return targets


def aligned_action_targets(actions: np.ndarray | torch.Tensor) -> np.ndarray:
    """Map three Wan-bin displacements to their exact 4x5 action segments."""

    return action_targets_for_intervals(actions, ACTION_CHUNK_INTERVALS)


def temporally_misaligned_action_targets(
    actions: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Use a nonoverlapping cyclic transition donor in the same clip/task."""

    return action_targets_for_intervals(
        actions, TEMPORALLY_MISALIGNED_ACTION_CHUNK_INTERVALS
    )


def temporal_control_oracle_feasibility(
    actions: np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    """Prove on train only that the frozen 0.10 cosine gate is attainable."""

    validate_temporal_control_contract()
    aligned_raw = aligned_action_targets(actions)
    negative_raw = temporally_misaligned_action_targets(actions)
    aligned, mean, std, active = _standardize_targets(aligned_raw)
    negative = np.where(
        active[None], (negative_raw - mean[None]) / std[None], 0.0
    )
    subsets: dict[str, Any] = {}
    passed = True
    minimum = float(
        fixed_protocol()["controls"]["same_clip_task_matched_temporal_misalignment"]
        ["oracle_minimum_cosine_gap"]
    )
    for name, transitions in (
        ("all_three_transitions", ALL_TRANSITIONS),
        ("future_relevant_transitions", FUTURE_RELEVANT_TRANSITIONS),
    ):
        oracle_cosine = _per_clip_cosine(
            aligned, aligned, active, transitions
        )
        negative_cosine = _per_clip_cosine(
            aligned, negative, active, transitions
        )
        aligned_mse = _per_clip_mse(aligned, aligned, active, transitions)
        negative_mse = _per_clip_mse(aligned, negative, active, transitions)
        cosine_gap = float(oracle_cosine.mean() - negative_cosine.mean())
        mse_gap = float(negative_mse.mean() - aligned_mse.mean())
        subset_passed = bool(
            cosine_gap >= minimum
            and mse_gap > 0.0
            and float(aligned_mse.max()) == 0.0
        )
        passed = passed and subset_passed
        subsets[name] = {
            "clips": int(aligned.shape[0]),
            "transitions": list(transitions),
            "perfect_predictor_cosine_mean": float(oracle_cosine.mean()),
            "temporal_negative_cosine_mean": float(negative_cosine.mean()),
            "perfect_predictor_cosine_gap": cosine_gap,
            "minimum_required_cosine_gap": minimum,
            "perfect_predictor_mse_mean": float(aligned_mse.mean()),
            "temporal_negative_mse_mean": float(negative_mse.mean()),
            "perfect_predictor_mse_gap": mse_gap,
            "passed": subset_passed,
        }
    payload = identity_payload(
        {
            "kind": "action-cycle-temporal-control-train-oracle-feasibility-v1",
            "split": "train",
            "fit_or_validation_metrics_used": False,
            "action_chunk_intervals": [list(value) for value in ACTION_CHUNK_INTERVALS],
            "temporal_negative_action_chunk_intervals": [
                list(value) for value in TEMPORALLY_MISALIGNED_ACTION_CHUNK_INTERVALS
            ],
            "donor_transition_by_recipient": list(TEMPORAL_DONOR_TRANSITIONS),
            "nonoverlapping_with_recipient": True,
            "subsets": subsets,
            "passed": passed,
            "protected_test_accessed": False,
        }
    )
    if not passed:
        raise ActionCycleProbeError(
            "train-only temporal-control oracle cannot attain the frozen cosine gate"
        )
    return payload


def validate_temporal_oracle_payload(payload: Mapping[str, Any]) -> None:
    control = fixed_protocol()["controls"][
        "same_clip_task_matched_temporal_misalignment"
    ]
    if (
        not identity_valid(payload)
        or payload.get("kind")
        != "action-cycle-temporal-control-train-oracle-feasibility-v1"
        or payload.get("split") != "train"
        or payload.get("fit_or_validation_metrics_used") is not False
        or payload.get("action_chunk_intervals")
        != [list(value) for value in ACTION_CHUNK_INTERVALS]
        or payload.get("temporal_negative_action_chunk_intervals")
        != [list(value) for value in TEMPORALLY_MISALIGNED_ACTION_CHUNK_INTERVALS]
        or payload.get("donor_transition_by_recipient")
        != list(TEMPORAL_DONOR_TRANSITIONS)
        or payload.get("nonoverlapping_with_recipient") is not True
        or payload.get("passed") is not True
        or payload.get("protected_test_accessed") is not False
    ):
        raise ActionCycleProbeError("train-only temporal oracle certificate differs")
    for name in ("all_three_transitions", "future_relevant_transitions"):
        row = payload.get("subsets", {}).get(name, {})
        if (
            row.get("passed") is not True
            or float(row.get("perfect_predictor_cosine_gap", -math.inf))
            < float(control["oracle_minimum_cosine_gap"])
            or float(row.get("perfect_predictor_mse_gap", -math.inf)) <= 0.0
            or float(row.get("perfect_predictor_mse_mean", math.inf)) != 0.0
        ):
            raise ActionCycleProbeError("train-only temporal oracle invariant failed")


def _distributed_context() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != fixed_protocol()["distributed_encode_world_size"]:
        raise ActionCycleProbeError("encoding requires exactly eight torchrun ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= local_rank:
        raise ActionCycleProbeError("encoding rank lacks its assigned CUDA device")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return rank, world_size, torch.device("cuda", local_rank)


def _exclusive_npy(path: Path, value: np.ndarray) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_wan_tokenizer(registration: Mapping[str, Any], device: torch.device):
    runtime = registration["runtime"]
    os.environ["VIDEOX_HOME"] = runtime["videox_home"]
    os.environ["WAN_DIR"] = runtime["wan_dir"]
    from robot_wm.modeling.tokenizers.rgb.wan_vae import WanVAETokenizer

    tokenizer = WanVAETokenizer(
        model_path=runtime["wan_dir"],
        config_path=str(Path(runtime["videox_home"]) / "config/wan2.1/wan_civitai.yaml"),
    ).to(device)
    tokenizer.eval()
    if any(parameter.requires_grad for parameter in tokenizer.parameters()):
        raise ActionCycleProbeError("Wan VAE must remain frozen")
    return tokenizer


@torch.no_grad()
def runtime_prefix_alignment_audit(tokenizer: Any, rgb: torch.Tensor) -> dict[str, Any]:
    """Empirically bind the source-declared 1,4,4,4 causal emission map."""

    if tuple(rgb.shape) != (1, *RGB_SHAPE):
        raise ActionCycleProbeError("alignment canary requires one exact cached RGB clip")
    video = rgb.permute(0, 2, 1, 3, 4).contiguous()
    full = tokenizer.encode_temporal(video, sample=False).float()
    if tuple(full.shape[1:]) != LATENT_SHAPE:
        raise ActionCycleProbeError(f"actual Wan output shape differs: {tuple(full.shape)}")
    comparisons: list[dict[str, Any]] = []
    for bin_index, (_start, stop) in enumerate(VAE_FRAME_SUPPORTS):
        prefix = tokenizer.encode_temporal(video[:, :, :stop], sample=False).float()
        expected = full[:, :, : bin_index + 1]
        maximum = float((prefix - expected).abs().max())
        close = bool(torch.allclose(prefix, expected, rtol=PREFIX_RTOL, atol=PREFIX_ATOL))
        if tuple(prefix.shape[2:]) != (bin_index + 1, 24, 120) or not close:
            raise ActionCycleProbeError(
                f"actual Wan prefix equivalence failed at bin {bin_index}: max={maximum}"
            )
        comparisons.append(
            {
                "bin": bin_index,
                "input_frame_interval": list(VAE_FRAME_SUPPORTS[bin_index]),
                "prefix_frame_count": stop,
                "prefix_latent_shape": list(prefix.shape),
                "max_abs_error": maximum,
                "allclose": close,
                "prefix_sha256": tensor_sha256(prefix),
                "full_prefix_sha256": tensor_sha256(expected),
            }
        )
    return {
        "actual_full_latent_shape": list(full.shape),
        "actual_full_latent_sha256": tensor_sha256(full),
        "source_declared_input_blocks": [list(value) for value in VAE_FRAME_SUPPORTS],
        "emitted_bins": 4,
        "adjacent_displacements": 3,
        "prefix_equivalence": comparisons,
        "passed": True,
    }


def _wandb_log(
    registration: Mapping[str, Any], *, stage: str, summary: Mapping[str, Any]
) -> None:
    try:
        import wandb
    except ImportError as exc:
        raise ActionCycleProbeError("W&B is required by the registered workflow") from exc
    run_id = stage_identity(registration, stage)
    root = Path(registration["output_root"])
    run = wandb.init(
        entity=EXPECTED_ENTITY,
        project=EXPECTED_PROJECT,
        group=None,
        id=run_id,
        resume="never",
        mode="online",
        dir=str(root / "wandb"),
        config={
            "registration_identity_sha256": registration["identity_sha256"],
            "stage": stage,
            "protected_test_accessed": False,
            "target_cache_array_opened": False,
        },
    )
    try:
        run.log(dict(summary))
        run.summary["registration_identity_sha256"] = registration["identity_sha256"]
        run.summary["stage_identity_sha256"] = run_id
        run.summary["protected_test_accessed"] = False
        run.summary["target_cache_array_opened"] = False
    finally:
        run.finish(exit_code=0)


def _merge_encoding_shards(
    output: Path,
    *,
    split: str,
    count: int,
    world_size: int,
    registration: Mapping[str, Any],
    canary: Mapping[str, Any],
    rgb_pre_rehash: Mapping[str, Any],
    rgb_post_rehash: Mapping[str, Any],
) -> dict[str, Any]:
    validate_consumption_window(rgb_pre_rehash, rgb_post_rehash)
    all_indices: list[np.ndarray] = []
    all_features: list[np.ndarray] = []
    shard_records: list[dict[str, Any]] = []
    for rank in range(world_size):
        index_path = output / f"indices.rank{rank:02d}.int64.npy"
        feature_path = output / f"features.rank{rank:02d}.float32.npy"
        receipt_path = output / f"receipt.rank{rank:02d}.json"
        receipt = read_json(receipt_path, label=f"encoding rank {rank} receipt")
        indices = np.load(index_path, allow_pickle=False)
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if (
            receipt.get("registration_identity_sha256") != registration["identity_sha256"]
            or receipt.get("split") != split
            or int(receipt.get("rank", -1)) != rank
            or int(receipt.get("world_size", -1)) != world_size
            or tuple(features.shape) != (len(indices), 3, 3, FEATURE_DIM)
            or features.dtype != np.float32
            or indices.dtype != np.int64
            or receipt.get("features_sha256") != sha256_file(feature_path)
            or receipt.get("indices_sha256") != sha256_file(index_path)
        ):
            raise ActionCycleProbeError(f"encoding shard {rank} failed integrity validation")
        all_indices.append(np.asarray(indices))
        all_features.append(np.asarray(features))
        shard_records.append(
            {
                "rank": rank,
                "indices": file_record(index_path),
                "features": file_record(feature_path),
                "receipt": file_record(receipt_path),
            }
        )
    indexes = np.concatenate(all_indices)
    values = np.concatenate(all_features)
    order = np.argsort(indexes)
    indexes = indexes[order]
    values = values[order]
    if not np.array_equal(indexes, np.arange(count, dtype=np.int64)):
        raise ActionCycleProbeError("distributed encoding indexes are not exact and complete")
    if not np.isfinite(values).all():
        raise ActionCycleProbeError("distributed encoding contains non-finite features")
    final_path = output / "features.float32.npy"
    _exclusive_npy(final_path, values.astype(np.float32, copy=False))
    metadata = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": ENCODING_KIND,
            "created_at_utc": now_utc(),
            "complete": True,
            "split": split,
            "registration_identity_sha256": registration["identity_sha256"],
            "source_rgb": registration["inputs"][split]["rgb"],
            "rgb_full_rehash_consumption_window": {
                "before": dict(rgb_pre_rehash),
                "after": dict(rgb_post_rehash),
                "unchanged": True,
            },
            "source_actions": registration["inputs"][split]["actions"],
            "features": file_record(final_path),
            "feature_shape": [count, 3, 3, FEATURE_DIM],
            "feature_dtype": "float32",
            "feature_formula": fixed_protocol()["feature"],
            "alignment": fixed_protocol()["alignment"],
            "actual_vae_alignment_canary": dict(canary),
            "shards": shard_records,
            "target_cache_array_opened": False,
            "protected_test_accessed": False,
        }
    )
    exclusive_json(output / "metadata.json", metadata)
    return metadata


def command_encode(args: argparse.Namespace) -> int:
    registration = validate_registration(args.registration, full_hash=False)
    split = args.split
    count = TRAIN_CLIPS if split == "train" else VALIDATION_CLIPS
    source = registration["inputs"][split]
    rank, world_size, device = _distributed_context()
    output = Path(registration["output_root"]) / "artifacts/encoded" / split
    if rank == 0:
        if output.exists() or output.is_symlink():
            raise ActionCycleProbeError(f"{split} encoding output must be fresh")
        output.mkdir(mode=0o700)
    torch.distributed.barrier()

    tokenizer = _load_wan_tokenizer(registration, device)
    rehash_path = output / "rgb.preconsumption-full-rehash.json"
    if rank == 0:
        rgb_rehash = full_preconsumption_rehash(
            source["rgb"],
            label=f"{split} RGB array immediately before encode consumption",
            registration_identity_sha256=registration["identity_sha256"],
        )
        exclusive_json(rehash_path, rgb_rehash)
    torch.distributed.barrier()
    rgb_rehash = read_json(rehash_path, label=f"{split} RGB preconsumption rehash")
    if (
        not identity_valid(rgb_rehash)
        or rgb_rehash.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or rgb_rehash.get("sha256") != source["rgb"]["sha256"]
        or rgb_rehash.get("full_file_hashed") is not True
    ):
        raise ActionCycleProbeError("RGB preconsumption full rehash receipt differs")
    rgbs = np.load(source["rgb"]["path"], mmap_mode="r", allow_pickle=False)
    indexes = np.arange(rank, count, world_size, dtype=np.int64)
    features = np.empty((len(indexes), 3, 3, FEATURE_DIM), dtype=np.float32)
    canary: dict[str, Any] | None = None
    started = time.monotonic()
    for offset, index in enumerate(indexes.tolist()):
        rgb = torch.from_numpy(np.array(rgbs[index], copy=True)).unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        if rank == 0 and offset == 0:
            canary = runtime_prefix_alignment_audit(tokenizer, rgb)
        latent = tokenizer.encode_temporal(
            rgb.permute(0, 2, 1, 3, 4).contiguous(), sample=False
        )
        features[offset] = latent_displacement_features(latent).cpu().numpy()[0]
    torch.cuda.synchronize(device)
    torch.distributed.barrier()
    post_rehash_path = output / "rgb.postconsumption-full-rehash.json"
    if rank == 0:
        rgb_post_rehash = full_postconsumption_rehash(
            source["rgb"],
            label=f"{split} RGB array immediately after encode consumption",
            registration_identity_sha256=registration["identity_sha256"],
        )
        validate_consumption_window(rgb_rehash, rgb_post_rehash)
        exclusive_json(post_rehash_path, rgb_post_rehash)
    torch.distributed.barrier()
    rgb_post_rehash = read_json(
        post_rehash_path, label=f"{split} RGB postconsumption rehash"
    )
    validate_consumption_window(rgb_rehash, rgb_post_rehash)
    del tokenizer, rgbs
    elapsed = time.monotonic() - started
    index_path = output / f"indices.rank{rank:02d}.int64.npy"
    feature_path = output / f"features.rank{rank:02d}.float32.npy"
    _exclusive_npy(index_path, indexes)
    _exclusive_npy(feature_path, features)
    receipt = identity_payload(
        {
            "kind": "action-cycle-recoverability-encoding-rank-v1",
            "registration_identity_sha256": registration["identity_sha256"],
            "split": split,
            "rank": rank,
            "world_size": world_size,
            "device": str(device),
            "indices_sha256": sha256_file(index_path),
            "features_sha256": sha256_file(feature_path),
            "rows": len(indexes),
            "elapsed_seconds": elapsed,
            "target_cache_array_opened": False,
            "protected_test_accessed": False,
        }
    )
    exclusive_json(output / f"receipt.rank{rank:02d}.json", receipt)
    torch.distributed.barrier()
    if rank == 0:
        if not isinstance(canary, dict) or canary.get("passed") is not True:
            raise ActionCycleProbeError("rank-zero actual VAE alignment canary is missing")
        metadata = _merge_encoding_shards(
            output, split=split, count=count, world_size=world_size,
            registration=registration, canary=canary,
            rgb_pre_rehash=rgb_rehash, rgb_post_rehash=rgb_post_rehash,
        )
        if args.wandb:
            _wandb_log(
                registration,
                stage=f"encode-{split}",
                summary={
                    "clips": count,
                    "world_size": world_size,
                    "feature_dim": FEATURE_DIM,
                    "elapsed_seconds_rank0": elapsed,
                    "actual_adjacent_displacements": 3,
                    "alignment_canary_passed": 1,
                    "encoding_identity_sha256": metadata["identity_sha256"],
                },
            )
        print(json.dumps(metadata, sort_keys=True))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return 0


def episode_disjoint_permutation(
    episodes: Sequence[str], *, seed: int
) -> np.ndarray:
    """Return a deterministic bijection whose donors are from other episodes."""

    episode_array = np.asarray([str(value) for value in episodes], dtype=object)
    count = len(episode_array)
    if count < 2:
        raise ActionCycleProbeError("episode-disjoint shuffling needs at least two rows")
    unique, frequencies = np.unique(episode_array, return_counts=True)
    if int(frequencies.max()) * 2 > count:
        raise ActionCycleProbeError("no episode-disjoint bijection can exist")
    rng = np.random.default_rng(seed)
    for _ in range(100_000):
        donor = rng.permutation(count)
        if bool(np.all(episode_array != episode_array[donor])):
            return donor.astype(np.int64)
    # The registered ABC manifests have one clip per episode.  This fallback
    # makes the generic implementation deterministic for repeated episodes.
    order = np.argsort(episode_array, kind="stable")
    for shift in range(1, count):
        donor = np.empty(count, dtype=np.int64)
        donor[order] = np.roll(order, -shift)
        if bool(np.all(episode_array != episode_array[donor])):
            return donor
    raise ActionCycleProbeError(f"could not construct disjoint shuffle over {len(unique)} episodes")


def _validate_encoding(
    registration: Mapping[str, Any], split: str
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    root = Path(registration["output_root"]) / "artifacts/encoded" / split
    metadata = read_json(root / "metadata.json", label=f"{split} encoding metadata")
    count = TRAIN_CLIPS if split == "train" else VALIDATION_CLIPS
    rgb_window = metadata.get("rgb_full_rehash_consumption_window", {})
    rgb_pre_rehash = rgb_window.get("before", {})
    rgb_post_rehash = rgb_window.get("after", {})
    if (
        not identity_valid(metadata)
        or metadata.get("kind") != ENCODING_KIND
        or metadata.get("complete") is not True
        or metadata.get("split") != split
        or metadata.get("registration_identity_sha256") != registration["identity_sha256"]
        or metadata.get("feature_shape") != [count, 3, 3, FEATURE_DIM]
        or metadata.get("feature_dtype") != "float32"
        or metadata.get("feature_formula") != fixed_protocol()["feature"]
        or metadata.get("alignment") != fixed_protocol()["alignment"]
        or metadata.get("actual_vae_alignment_canary", {}).get("passed") is not True
        or metadata.get("actual_vae_alignment_canary", {}).get("adjacent_displacements") != 3
        or rgb_window.get("unchanged") is not True
        or not identity_valid(rgb_pre_rehash)
        or not identity_valid(rgb_post_rehash)
        or rgb_pre_rehash.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or rgb_pre_rehash.get("sha256")
        != registration["inputs"][split]["rgb"]["sha256"]
        or rgb_post_rehash.get("sha256")
        != registration["inputs"][split]["rgb"]["sha256"]
        or metadata.get("target_cache_array_opened") is not False
        or metadata.get("protected_test_accessed") is not False
    ):
        raise ActionCycleProbeError(f"{split} encoding protocol differs")
    validate_consumption_window(rgb_pre_rehash, rgb_post_rehash)
    feature_pre_rehash = full_preconsumption_rehash(
        metadata["features"],
        label=f"{split} encoded features immediately before analysis consumption",
        registration_identity_sha256=registration["identity_sha256"],
    )
    feature_path = Path(metadata["features"]["path"])
    values = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if tuple(values.shape) != (count, 3, 3, FEATURE_DIM) or values.dtype != np.float32:
        raise ActionCycleProbeError(f"{split} feature array differs")
    return metadata, values, feature_pre_rehash


def _standardize_targets(train_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_targets.mean(axis=0)
    std = train_targets.std(axis=0, ddof=0)
    active = std > STD_FLOOR
    if not bool(np.all(active.sum(axis=1) > 0)):
        raise ActionCycleProbeError("at least one transition has no active action coordinate")
    safe = np.where(active, std, 1.0)
    normalized = (train_targets - mean[None]) / safe[None]
    normalized = np.where(active[None], normalized, 0.0)
    return normalized, mean, safe, active


def _standardize_features(
    train: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=0)
    active = std > STD_FLOOR
    safe = np.where(active, std, 1.0)
    train_norm = np.where(active, (train - mean) / safe, 0.0)
    val_norm = np.where(active, (validation - mean) / safe, 0.0)
    return train_norm, val_norm, mean, safe, active


def _ridge_eigendecomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    active_count = int(np.count_nonzero(x.std(axis=0, ddof=0) > 0.0))
    if active_count <= 0:
        raise ActionCycleProbeError("ridge stratum has no active feature")
    kernel = (x @ x.T) / float(active_count)
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return eigenvalues, eigenvectors, active_count


def _loo_scores(
    x: np.ndarray, y: np.ndarray, alphas: Sequence[float]
) -> np.ndarray:
    eigenvalues, eigenvectors, _ = _ridge_eigendecomposition(x)
    projected = eigenvectors.T @ y
    scores: list[float] = []
    squared_vectors = np.square(eigenvectors)
    for alpha in alphas:
        shrink = eigenvalues / (eigenvalues + float(alpha))
        fitted = eigenvectors @ (shrink[:, None] * projected)
        leverage = squared_vectors @ shrink
        denominator = np.maximum(1.0 - leverage, 1.0e-8)
        residual = (y - fitted) / denominator[:, None]
        scores.append(float(np.mean(np.square(residual))))
    return np.asarray(scores, dtype=np.float64)


def _fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors, active_count = _ridge_eigendecomposition(x_train)
    dual = eigenvectors @ (
        (eigenvectors.T @ y_train) / (eigenvalues[:, None] + float(alpha))
    )
    weight = (x_train.T @ dual) / float(active_count)
    prediction = x_validation @ weight
    return prediction, weight


def _per_clip_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
    transitions: Sequence[int],
) -> np.ndarray:
    values = []
    for transition in transitions:
        mask = active[transition]
        values.append(np.mean((prediction[:, transition, mask] - target[:, transition, mask]) ** 2, axis=1))
    return np.mean(np.stack(values, axis=1), axis=1)


def _per_clip_cosine(
    prediction: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
    transitions: Sequence[int],
) -> np.ndarray:
    prediction_parts = [prediction[:, index, active[index]] for index in transitions]
    target_parts = [target[:, index, active[index]] for index in transitions]
    left = np.concatenate(prediction_parts, axis=1)
    right = np.concatenate(target_parts, axis=1)
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1.0e-12)


def _retrieval_hits(
    prediction: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
    transitions: Sequence[int],
    correct: np.ndarray,
) -> np.ndarray:
    distances = np.zeros((prediction.shape[0], target.shape[0]), dtype=np.float64)
    scalar_count = 0
    for transition in transitions:
        mask = active[transition]
        left = prediction[:, transition, mask]
        right = target[:, transition, mask]
        distances += (
            np.sum(np.square(left), axis=1)[:, None]
            + np.sum(np.square(right), axis=1)[None]
            - 2.0 * (left @ right.T)
        )
        scalar_count += int(mask.sum())
    distances /= float(scalar_count)
    nearest = np.argmin(distances, axis=1)
    return (nearest == correct).astype(np.float64)


def _comparison_vectors(
    aligned: np.ndarray,
    shuffled_fit: np.ndarray,
    targets: np.ndarray,
    shuffled_targets: np.ndarray,
    temporally_misaligned_targets: np.ndarray,
    active: np.ndarray,
    transitions: Sequence[int],
    prefix: str,
    shuffled_indexes: np.ndarray,
) -> tuple[dict[str, tuple[str, np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    zero = np.zeros_like(aligned)
    mse_aligned = _per_clip_mse(aligned, targets, active, transitions)
    mse_mean = _per_clip_mse(zero, targets, active, transitions)
    mse_shuffled_fit = _per_clip_mse(shuffled_fit, targets, active, transitions)
    mse_shuffled_target = _per_clip_mse(aligned, shuffled_targets, active, transitions)
    mse_temporally_misaligned = _per_clip_mse(
        aligned, temporally_misaligned_targets, active, transitions
    )
    cosine_aligned = _per_clip_cosine(aligned, targets, active, transitions)
    cosine_shuffled_fit = _per_clip_cosine(shuffled_fit, targets, active, transitions)
    cosine_shuffled_target = _per_clip_cosine(aligned, shuffled_targets, active, transitions)
    cosine_temporally_misaligned = _per_clip_cosine(
        aligned, temporally_misaligned_targets, active, transitions
    )
    retrieval_aligned = _retrieval_hits(
        aligned, targets, active, transitions, np.arange(len(aligned), dtype=np.int64)
    )
    retrieval_shuffled = _retrieval_hits(
        aligned, targets, active, transitions, shuffled_indexes
    )
    comparisons = {
        f"{prefix}/mse_vs_train_mean": ("relative", mse_mean, mse_aligned),
        f"{prefix}/mse_vs_shuffled_fit": ("relative", mse_shuffled_fit, mse_aligned),
        f"{prefix}/mse_vs_shuffled_target": ("relative", mse_shuffled_target, mse_aligned),
        f"{prefix}/mse_vs_same_clip_temporal_misalignment": (
            "relative", mse_temporally_misaligned, mse_aligned
        ),
        f"{prefix}/cosine_vs_shuffled_fit": ("difference", cosine_aligned, cosine_shuffled_fit),
        f"{prefix}/cosine_vs_shuffled_target": ("difference", cosine_aligned, cosine_shuffled_target),
        f"{prefix}/cosine_vs_same_clip_temporal_misalignment": (
            "difference", cosine_aligned, cosine_temporally_misaligned
        ),
        f"{prefix}/retrieval_vs_shuffled_target": (
            "difference", retrieval_aligned, retrieval_shuffled
        ),
    }
    metrics = {
        "mse_aligned": mse_aligned,
        "mse_train_mean": mse_mean,
        "mse_shuffled_fit": mse_shuffled_fit,
        "mse_shuffled_target": mse_shuffled_target,
        "mse_same_clip_temporal_misalignment": mse_temporally_misaligned,
        "cosine_aligned": cosine_aligned,
        "cosine_shuffled_fit": cosine_shuffled_fit,
        "cosine_shuffled_target": cosine_shuffled_target,
        "cosine_same_clip_temporal_misalignment": cosine_temporally_misaligned,
        "retrieval_aligned": retrieval_aligned,
        "retrieval_shuffled_target": retrieval_shuffled,
    }
    return comparisons, metrics


def paired_bootstrap_gate(
    comparisons: Mapping[str, tuple[str, np.ndarray, np.ndarray]],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    names = sorted(comparisons)
    if not names:
        raise ActionCycleProbeError("bootstrap comparison family is empty")
    count = len(next(iter(comparisons.values()))[1])
    if count != VALIDATION_CLIPS:
        raise ActionCycleProbeError("bootstrap must use all 64 validation clips")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, count, size=(samples, count), dtype=np.int64)
    family_alpha = 0.05
    quantile = family_alpha / len(names)
    results: dict[str, Any] = {}
    all_pass = True
    for name in names:
        mode, favorable, candidate = comparisons[name]
        if len(favorable) != count or len(candidate) != count:
            raise ActionCycleProbeError("bootstrap comparison length differs")
        favorable_boot = favorable[indexes].mean(axis=1)
        candidate_boot = candidate[indexes].mean(axis=1)
        if mode == "relative":
            point = float((favorable.mean() - candidate.mean()) / max(favorable.mean(), 1.0e-12))
            distribution = (favorable_boot - candidate_boot) / np.maximum(favorable_boot, 1.0e-12)
            threshold = fixed_protocol()["gate_thresholds"]["normalized_mse_relative_improvement"]
        elif mode == "difference":
            point = float(favorable.mean() - candidate.mean())
            distribution = favorable_boot - candidate_boot
            threshold = (
                fixed_protocol()["gate_thresholds"]["retrieval_accuracy_gain"]
                if "retrieval" in name
                else fixed_protocol()["gate_thresholds"]["cosine_absolute_gain"]
            )
        else:
            raise ActionCycleProbeError(f"unknown comparison mode {mode}")
        lower = float(np.quantile(distribution, quantile, method="linear"))
        passed = bool(point >= threshold and lower > 0.0)
        all_pass = all_pass and passed
        results[name] = {
            "mode": mode,
            "point": point,
            "threshold": threshold,
            "simultaneous_lower_95": lower,
            "bonferroni_one_sided_quantile": quantile,
            "passed": passed,
        }
    return {
        "samples": samples,
        "seed": seed,
        "unit": "validation_clip",
        "family_size": len(names),
        "familywise_confidence": 0.95,
        "comparisons": results,
        "all_passed": all_pass,
    }


def _exclusive_npz(path: Path, **arrays: np.ndarray) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def command_analyze(args: argparse.Namespace) -> int:
    registration = validate_registration(args.registration, full_hash=False)
    train_encoding, train_features_mmap, train_feature_pre_rehash = _validate_encoding(
        registration, "train"
    )
    train_features = np.asarray(train_features_mmap, dtype=np.float64)
    if not np.isfinite(train_features).all():
        raise ActionCycleProbeError("train feature array is non-finite")
    del train_features_mmap
    train_feature_post_rehash = full_postconsumption_rehash(
        train_encoding["features"],
        label="train encoded features immediately after analysis consumption",
        registration_identity_sha256=registration["identity_sha256"],
    )
    validate_consumption_window(
        train_feature_pre_rehash, train_feature_post_rehash
    )
    val_encoding, val_features_mmap, val_feature_pre_rehash = _validate_encoding(
        registration, "val"
    )
    val_features = np.asarray(val_features_mmap, dtype=np.float64)
    if not np.isfinite(val_features).all():
        raise ActionCycleProbeError("validation feature array is non-finite")
    del val_features_mmap
    val_feature_post_rehash = full_postconsumption_rehash(
        val_encoding["features"],
        label="validation encoded features immediately after analysis consumption",
        registration_identity_sha256=registration["identity_sha256"],
    )
    validate_consumption_window(val_feature_pre_rehash, val_feature_post_rehash)
    train_action_rehash = full_preconsumption_rehash(
        registration["inputs"]["train"]["actions"],
        label="train actions immediately before analysis consumption",
        registration_identity_sha256=registration["identity_sha256"],
    )
    train_actions = np.load(
        registration["inputs"]["train"]["actions"]["path"], mmap_mode="r",
        allow_pickle=False,
    )
    train_targets_raw = aligned_action_targets(train_actions)
    del train_actions
    train_action_post_rehash = full_postconsumption_rehash(
        registration["inputs"]["train"]["actions"],
        label="train actions immediately after analysis consumption",
        registration_identity_sha256=registration["identity_sha256"],
    )
    validate_consumption_window(train_action_rehash, train_action_post_rehash)
    val_action_rehash = full_preconsumption_rehash(
        registration["inputs"]["val"]["actions"],
        label="validation actions immediately before analysis consumption",
        registration_identity_sha256=registration["identity_sha256"],
    )
    val_actions = np.load(
        registration["inputs"]["val"]["actions"]["path"], mmap_mode="r",
        allow_pickle=False,
    )
    val_targets_raw = aligned_action_targets(val_actions)
    val_temporally_misaligned_raw = temporally_misaligned_action_targets(val_actions)
    del val_actions
    val_action_post_rehash = full_postconsumption_rehash(
        registration["inputs"]["val"]["actions"],
        label="validation actions immediately after analysis consumption",
        registration_identity_sha256=registration["identity_sha256"],
    )
    validate_consumption_window(val_action_rehash, val_action_post_rehash)
    train_targets, target_mean, target_std, target_active = _standardize_targets(
        train_targets_raw
    )
    val_targets = np.where(
        target_active[None],
        (val_targets_raw - target_mean[None]) / target_std[None],
        0.0,
    )
    val_temporally_misaligned = np.where(
        target_active[None],
        (val_temporally_misaligned_raw - target_mean[None]) / target_std[None],
        0.0,
    )

    train_episodes = [row["episode_dir"] for row in registration["inputs"]["train"]["descriptors"]]
    val_episodes = [row["episode_dir"] for row in registration["inputs"]["val"]["descriptors"]]
    train_shuffle = episode_disjoint_permutation(train_episodes, seed=SHUFFLE_SEED)
    val_shuffle = episode_disjoint_permutation(val_episodes, seed=SHUFFLE_SEED + 1)
    shuffled_train_targets = train_targets[train_shuffle]
    shuffled_val_targets = val_targets[val_shuffle]

    x_train_norm = np.empty_like(train_features, dtype=np.float64)
    x_val_norm = np.empty_like(val_features, dtype=np.float64)
    x_mean = np.empty((3, 3, FEATURE_DIM), dtype=np.float64)
    x_std = np.empty_like(x_mean)
    x_active = np.empty_like(x_mean, dtype=bool)
    alpha_score = np.zeros(len(ALPHA_GRID), dtype=np.float64)
    for transition in ALL_TRANSITIONS:
        y = train_targets[:, transition, target_active[transition]]
        for view in range(len(CAMERAS)):
            normalized = _standardize_features(
                train_features[:, transition, view], val_features[:, transition, view]
            )
            x_train_norm[:, transition, view] = normalized[0]
            x_val_norm[:, transition, view] = normalized[1]
            x_mean[transition, view] = normalized[2]
            x_std[transition, view] = normalized[3]
            x_active[transition, view] = normalized[4]
            alpha_score += _loo_scores(normalized[0], y, ALPHA_GRID)
    alpha_score /= float(len(ALL_TRANSITIONS) * len(CAMERAS))
    selected_index = int(np.argmin(alpha_score))
    selected_alpha = float(ALPHA_GRID[selected_index])

    aligned_predictions_by_view = np.zeros(
        (VALIDATION_CLIPS, 3, 3, TARGET_DIM), dtype=np.float64
    )
    shuffled_predictions_by_view = np.zeros_like(aligned_predictions_by_view)
    aligned_weights = np.zeros((3, 3, FEATURE_DIM, TARGET_DIM), dtype=np.float32)
    shuffled_weights = np.zeros_like(aligned_weights)
    for transition in ALL_TRANSITIONS:
        mask = target_active[transition]
        for view in range(len(CAMERAS)):
            prediction, weight = _fit_predict_ridge(
                x_train_norm[:, transition, view],
                train_targets[:, transition, mask],
                x_val_norm[:, transition, view],
                selected_alpha,
            )
            shuffled_prediction, shuffled_weight = _fit_predict_ridge(
                x_train_norm[:, transition, view],
                shuffled_train_targets[:, transition, mask],
                x_val_norm[:, transition, view],
                selected_alpha,
            )
            aligned_predictions_by_view[:, transition, view, mask] = prediction
            shuffled_predictions_by_view[:, transition, view, mask] = shuffled_prediction
            aligned_weights[transition, view][:, mask] = weight.astype(np.float32)
            shuffled_weights[transition, view][:, mask] = shuffled_weight.astype(np.float32)
    aligned = aligned_predictions_by_view.mean(axis=2)
    shuffled_fit = shuffled_predictions_by_view.mean(axis=2)

    primary_comparisons, primary_metrics = _comparison_vectors(
        aligned, shuffled_fit, val_targets, shuffled_val_targets,
        val_temporally_misaligned, target_active,
        ALL_TRANSITIONS, "all_three_transitions", val_shuffle,
    )
    future_comparisons, future_metrics = _comparison_vectors(
        aligned, shuffled_fit, val_targets, shuffled_val_targets,
        val_temporally_misaligned, target_active,
        FUTURE_RELEVANT_TRANSITIONS, "future_relevant_transitions", val_shuffle,
    )
    gate = paired_bootstrap_gate({**primary_comparisons, **future_comparisons})

    analysis_dir = Path(registration["output_root"]) / "analysis" / "stage0"
    if analysis_dir.exists() or analysis_dir.is_symlink():
        raise ActionCycleProbeError("Stage-0 analysis output must be fresh")
    analysis_dir.mkdir(mode=0o700)
    model_path = analysis_dir / "frozen_ridge.npz"
    _exclusive_npz(
        model_path,
        selected_alpha=np.asarray(selected_alpha, dtype=np.float64),
        alpha_grid=np.asarray(ALPHA_GRID, dtype=np.float64),
        alpha_train_loo_scores=alpha_score,
        feature_mean=x_mean.astype(np.float32),
        feature_std=x_std.astype(np.float32),
        feature_active=x_active,
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
        target_active=target_active,
        aligned_weight=aligned_weights,
        shuffled_control_weight=shuffled_weights,
        train_episode_disjoint_permutation=train_shuffle,
        validation_episode_disjoint_permutation=val_shuffle,
    )
    model_metadata = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": MODEL_KIND,
            "created_at_utc": now_utc(),
            "registration_identity_sha256": registration["identity_sha256"],
            "train_encoding_identity_sha256": train_encoding["identity_sha256"],
            "validation_encoding_identity_sha256": val_encoding["identity_sha256"],
            "artifact": file_record(model_path),
            "selected_alpha": selected_alpha,
            "selection_split": "train_only_exact_loo",
            "validation_fit_observations": 0,
            "fit_observations_per_transition_view": TRAIN_CLIPS,
            "model_strata": 9,
            "train_shuffle_sha256": numpy_sha256(train_shuffle),
            "validation_shuffle_sha256": numpy_sha256(val_shuffle),
            "full_rehash_consumption_windows": {
                "train_features": {
                    "before": train_feature_pre_rehash,
                    "after": train_feature_post_rehash,
                    "unchanged": True,
                },
                "validation_features": {
                    "before": val_feature_pre_rehash,
                    "after": val_feature_post_rehash,
                    "unchanged": True,
                },
                "train_actions": {
                    "before": train_action_rehash,
                    "after": train_action_post_rehash,
                    "unchanged": True,
                },
                "validation_actions": {
                    "before": val_action_rehash,
                    "after": val_action_post_rehash,
                    "unchanged": True,
                },
            },
            "same_clip_temporal_misalignment": fixed_protocol()["controls"][
                "same_clip_task_matched_temporal_misalignment"
            ],
            "train_only_temporal_control_oracle_feasibility": registration[
                "train_only_temporal_control_oracle_feasibility"
            ],
            "target_cache_array_opened": False,
            "protected_test_accessed": False,
        }
    )
    exclusive_json(analysis_dir / "frozen_ridge.json", model_metadata)

    rows_path = analysis_dir / "validation_rows.jsonl"
    descriptor = os.open(rows_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            for index, clip in enumerate(registration["inputs"]["val"]["descriptors"]):
                row = {
                    "index": index,
                    "clip_id": clip["clip_id"],
                    "episode_dir": clip["episode_dir"],
                    "task_label": clip["task_label"],
                    "same_clip_temporal_misalignment": {
                        "task_label": clip["task_label"],
                        "episode_dir": clip["episode_dir"],
                        "action_chunk_intervals": [
                            list(value)
                            for value in TEMPORALLY_MISALIGNED_ACTION_CHUNK_INTERVALS
                        ],
                    },
                    "shuffled_donor_index": int(val_shuffle[index]),
                    "shuffled_donor_clip_id": registration["inputs"]["val"]["descriptors"][int(val_shuffle[index])]["clip_id"],
                    "all_three_transitions": {
                        name: float(values[index]) for name, values in primary_metrics.items()
                    },
                    "future_relevant_transitions": {
                        name: float(values[index]) for name, values in future_metrics.items()
                    },
                    "protected_test_accessed": False,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    aggregate_metrics = {
        "all_three_transitions": {
            name: float(values.mean()) for name, values in primary_metrics.items()
        },
        "future_relevant_transitions": {
            name: float(values.mean()) for name, values in future_metrics.items()
        },
    }
    result = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": ANALYSIS_KIND,
            "created_at_utc": now_utc(),
            "registration_identity_sha256": registration["identity_sha256"],
            "model": file_record(analysis_dir / "frozen_ridge.json"),
            "validation_rows": file_record(rows_path),
            "selected_alpha": selected_alpha,
            "alpha_train_loo_scores": {
                str(alpha): float(score) for alpha, score in zip(ALPHA_GRID, alpha_score)
            },
            "active_target_coordinates": [int(value) for value in target_active.sum(axis=1)],
            "aggregate_metrics": aggregate_metrics,
            "paired_bootstrap_gate": gate,
            "decision": (
                "go_to_generated_latent_action_cycle_stage1"
                if gate["all_passed"]
                else "stop_or_revise_action_cycle_path"
            ),
            "scientific_scope": (
                "clean_latent_action_recoverability_only_no_video_quality_claim"
            ),
            "validation_fit_observations": 0,
            "train_only_temporal_control_oracle_feasibility": registration[
                "train_only_temporal_control_oracle_feasibility"
            ],
            "full_rehash_consumption_windows": {
                "train_features": {
                    "before": train_feature_pre_rehash,
                    "after": train_feature_post_rehash,
                    "unchanged": True,
                },
                "validation_features": {
                    "before": val_feature_pre_rehash,
                    "after": val_feature_post_rehash,
                    "unchanged": True,
                },
                "train_actions": {
                    "before": train_action_rehash,
                    "after": train_action_post_rehash,
                    "unchanged": True,
                },
                "validation_actions": {
                    "before": val_action_rehash,
                    "after": val_action_post_rehash,
                    "unchanged": True,
                },
            },
            "target_cache_array_opened": False,
            "protected_test_accessed": False,
        }
    )
    exclusive_json(analysis_dir / "result.json", result)
    if args.wandb:
        flat_log: dict[str, Any] = {
            "selected_alpha": selected_alpha,
            "gate_passed": int(gate["all_passed"]),
            "analysis_identity_sha256": result["identity_sha256"],
        }
        for family, metrics in aggregate_metrics.items():
            for name, value in metrics.items():
                flat_log[f"{family}/{name}"] = value
        for name, comparison in gate["comparisons"].items():
            flat_log[f"gate/{name}/point"] = comparison["point"]
            flat_log[f"gate/{name}/simultaneous_lower_95"] = comparison[
                "simultaneous_lower_95"
            ]
        _wandb_log(registration, stage="analyze", summary=flat_log)
    print(json.dumps(result, sort_keys=True))
    return 0


def command_record_submission(args: argparse.Namespace) -> int:
    registration = validate_registration(args.registration, full_hash=False)
    if not JOB_ID_RE.fullmatch(args.encode_job_id) or not JOB_ID_RE.fullmatch(
        args.analysis_job_id
    ):
        raise ActionCycleProbeError("Slurm job IDs are invalid")
    payload = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": SUBMISSION_KIND,
            "created_at_utc": now_utc(),
            "registration_identity_sha256": registration["identity_sha256"],
            "encode_job_id": args.encode_job_id,
            "analysis_job_id": args.analysis_job_id,
            "analysis_dependency": f"afterok:{args.encode_job_id}",
            "protected_test_accessed": False,
        }
    )
    exclusive_json(Path(registration["output_root"]) / "submission.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register", help="immutably preregister the Stage-0 probe")
    register.add_argument("--repo", type=Path, required=True)
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--output-root", type=Path, required=True)
    register.add_argument("--lustre-base", type=Path, required=True)
    register.add_argument("--train-metadata", type=Path, required=True)
    register.add_argument("--train-manifest", type=Path, required=True)
    register.add_argument("--validation-metadata", type=Path, required=True)
    register.add_argument("--validation-manifest", type=Path, required=True)
    register.add_argument("--python", type=Path, required=True)
    register.add_argument("--videox-home", type=Path, required=True)
    register.add_argument("--wan-dir", type=Path, required=True)
    register.set_defaults(func=command_register)

    validate = sub.add_parser("validate-registration")
    validate.add_argument("--registration", type=Path, required=True)
    validate.add_argument("--full", action="store_true")
    validate.set_defaults(func=command_validate_registration)

    wandb_check = sub.add_parser("wandb-check")
    wandb_check.set_defaults(func=command_wandb_check)

    encode = sub.add_parser("encode")
    encode.add_argument("--registration", type=Path, required=True)
    encode.add_argument("--split", choices=("train", "val"), required=True)
    encode.add_argument("--wandb", action="store_true")
    encode.set_defaults(func=command_encode)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--registration", type=Path, required=True)
    analyze.add_argument("--wandb", action="store_true")
    analyze.set_defaults(func=command_analyze)

    submission = sub.add_parser("record-submission")
    submission.add_argument("--registration", type=Path, required=True)
    submission.add_argument("--encode-job-id", required=True)
    submission.add_argument("--analysis-job-id", required=True)
    submission.set_defaults(func=command_record_submission)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ActionCycleProbeError as exc:
        raise SystemExit(f"action-cycle recoverability error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
