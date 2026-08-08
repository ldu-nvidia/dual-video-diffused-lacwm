#!/usr/bin/env python3
"""Fail-closed orchestration for the preregistered temporal-target DOE.

The executable is intentionally stdlib-only.  It renders the complete command
graph without importing the training stack, and it runs that graph only after
an explicit ``--execute`` inside one non-requeued, eight-GPU Slurm allocation.
It never submits a job and never opens the protected test split.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "tools" / "causal_vjepa2_temporal_targets.py"
EVALUATOR = REPO_ROOT / "tools" / "evaluate_causal_vjepa2_temporal_targets.py"
ANALYZER = REPO_ROOT / "tools" / "analyze_causal_vjepa2_temporal_targets.py"

SCHEMA = "causal-vjepa2-temporal-doe-workflow-v1"
STATE_SCHEMA = "causal-vjepa2-temporal-doe-workflow-state-v1"
TRAINING_SCHEMA = "causal-vjepa2-temporal-target-training-v1"
EVALUATION_SCHEMA = "causal-vjepa2-temporal-target-evaluation-v1"

TRAIN_SEED = 1234
EVALUATION_SEED = 20260801
BOOTSTRAP_SEED = 20260807
WORLD_SIZE = 8
GLOBAL_BATCH_SIZE = 256
MICRO_BATCH_SIZE = 32
WORKERS_PER_RANK = 4
CALIBRATION_UPDATES = 200
PRIMARY_UPDATES = 5_000
TARGET_SHAPE = [48, 8, 8, 14]
NFE_GRID = [1, 2, 4, 8, 12, 20, 25]
CONTROLS = 9
VALIDATION_CLIPS = 890
MODEL_PARAMETERS = 41_963_760
EMA_SCHEDULE = "min(target_decay,(1+completed_updates)/(10+completed_updates))-v1"

WANDB_ENTITY = "zijiandu"
WANDB_PROJECT = "dual-video-diffusion-private"

TRAIN_MANIFEST_SHA256 = (
    "cc10bccece1ac0e20abacf30ee0db60339145ec54ab2e28af977ded21e02f27e"
)
VALIDATION_MANIFEST_SHA256 = (
    "b8773a8627e887bf0c0a31cfae6ff537ba6a99e0b1b4f11efec011e3983d8d99"
)
TRAIN_CACHE_METADATA_FILE_SHA256 = (
    "7eb40233b10f001ad5f49beb037af0e787414fcd4fc206e8334e6845d6611182"
)
VALIDATION_CACHE_METADATA_FILE_SHA256 = (
    "9c9f8e7d0f67fd50b9f6fbe4696eb13bf39b897aa0e5d061bc4daa22bf882a55"
)
TRAIN_CACHE_METADATA_IDENTITY_SHA256 = (
    "2ab673f034d9e83ebf19eb08c3254f73b2bdb1074ed936023521e2bd77abb4f2"
)
TRAIN_TARGET_SHA256 = (
    "547c4579cf978ac2b9527cb038693259af678a2b07268ab1434706dc128051c4"
)
VALIDATION_TARGET_SHA256 = (
    "ed0ee6c76233b6072691854f767ce0e524867826b6d811c43988791f4011b034"
)
PCA_SHA256 = "52348899c2e1c7aa73434c0db7fa9679e1c450d37fdcfa10e64a7d95ac86a5df"
NORMALIZATION_SHA256 = (
    "157b9dedbd95c8381eda5a69dfa1c646c79b8e408604975c369a8e1e1f54cff9"
)

APPROVED_ARTIFACT_ROOTS = (Path("/lustre"),)
SAFE_ENV_EXACT = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "WANDB_API_KEY",
    "WANDB_BASE_URL",
}
SAFE_ENV_PREFIXES = (
    "SLURM_",
    "CUDA_",
    "NVIDIA_",
    "NCCL_",
    "UCX_",
    "PMI_",
    "PMIX_",
    "OMPI_",
)


class WorkflowError(RuntimeError):
    """A frozen input, execution transition, or output receipt failed closed."""


@dataclass(frozen=True)
class Arm:
    name: str
    slug: str
    target_mode: str
    temporal_weight: float
    rollin_probability: float
    uses_normalization: bool


ARMS = (
    Arm("ABS", "abs", "absolute", 0.0, 0.0, False),
    Arm("ABS-T", "abs-t", "absolute", 1.0, 0.0, True),
    Arm("DELTA", "delta", "delta_pack", 0.0, 0.0, True),
    Arm("DELTA-T", "delta-t", "delta_pack", 1.0, 0.0, True),
    Arm("DELTA-R", "delta-r", "delta_pack", 0.0, 0.5, True),
)
ARM_BY_NAME = {arm.name: arm for arm in ARMS}


@dataclass(frozen=True)
class WorkflowInputs:
    expected_commit: str
    study_root: Path
    data_root: Path
    semantic_cache_root: Path
    train_manifest: Path
    validation_manifest: Path
    normalization: Path


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def python_executable() -> str:
    """Preserve the venv launcher path; resolving its symlink can escape the venv."""
    path = Path(sys.executable)
    if not path.is_absolute() or not path.exists() or not os.access(path, os.X_OK):
        raise WorkflowError("the workflow Python must be an absolute executable path")
    return str(path)


def file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise WorkflowError(f"required file must not be symlinked: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise WorkflowError(f"required file is missing, empty, or symlinked: {path}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def normalization_file_record(path: Path) -> dict[str, Any]:
    """Match ``load_temporal_normalization``'s augmented immutable record."""
    payload = read_json(path, "D1 normalization")
    return {**file_record(path), "payload_sha256": sha256_json(payload)}


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise WorkflowError(f"{label} must contain one JSON object")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise WorkflowError(f"refusing to replace immutable artifact: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise WorkflowError(f"git preflight failed: {' '.join(arguments)}") from exc


def validate_source(expected_commit: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise WorkflowError("--expected-commit must be a full lowercase commit SHA")
    top = Path(_git("rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != REPO_ROOT:
        raise WorkflowError(f"script repository changed: {top} != {REPO_ROOT}")
    head = _git("rev-parse", "HEAD")
    if head != expected_commit:
        raise WorkflowError(f"repository HEAD {head} differs from {expected_commit}")
    if _git("status", "--porcelain"):
        raise WorkflowError("DOE execution requires a clean committed worktree")
    for path in (TRAINER, EVALUATOR, ANALYZER):
        if not path.is_file() or path.is_symlink():
            raise WorkflowError(f"required implementation is missing or symlinked: {path}")
    return {"root": str(REPO_ROOT), "commit": head, "dirty": False}


def _canonical_existing_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or not path.exists() or path.is_symlink() or not path.is_dir():
        raise WorkflowError(f"{label} must be an existing canonical absolute directory")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise WorkflowError(f"{label} is not canonical: {path} -> {resolved}")
    return resolved


def _canonical_existing_file(path: Path, label: str) -> Path:
    if not path.is_absolute() or not path.exists() or path.is_symlink() or not path.is_file():
        raise WorkflowError(f"{label} must be an existing non-symlink absolute file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise WorkflowError(f"{label} is not canonical: {path} -> {resolved}")
    return resolved


def _approved_artifact_path(path: Path) -> bool:
    return any(path == root or root in path.parents for root in APPROVED_ARTIFACT_ROOTS)


def _validate_cache_metadata(
    metadata: Mapping[str, Any],
    *,
    split: str,
    clips: int,
    manifest_sha256: str,
    target_sha256: str,
) -> None:
    expected_shape = [clips, *TARGET_SHAPE]
    if (
        metadata.get("schema") != "droid-causal-vjepa2.1-v1"
        or metadata.get("complete") is not True
        or metadata.get("split") != split
        or metadata.get("clip_count") != clips
        or metadata.get("manifest_sha256") != manifest_sha256
        or metadata.get("train_manifest_sha256") != TRAIN_MANIFEST_SHA256
        or metadata.get("target_file") != "targets.fp16.npy"
        or metadata.get("target_sha256") != target_sha256
        or metadata.get("target_shape") != expected_shape
        or metadata.get("auxiliary_target_shape") != TARGET_SHAPE
        or metadata.get("target_dtype") != "float16"
        or metadata.get("pca_sha256") != PCA_SHA256
        or metadata.get("allowed_splits") != ["train", "val"]
        or metadata.get("protected_test_access") is not False
        or metadata.get("test_rows_extracted") != 0
        or metadata.get("world_size") != WORLD_SIZE
    ):
        raise WorkflowError(f"{split} semantic cache metadata changed")


def _validate_normalization(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema") != "causal-vjepa2-temporal-normalization-v1"
        or payload.get("status") != "complete"
        or payload.get("complete") is not True
        or payload.get("split") != "train"
        or payload.get("clips") != 64_000
        or payload.get("train_manifest_sha256") != TRAIN_MANIFEST_SHA256
        or payload.get("pca_sha256") != PCA_SHA256
        or payload.get("cache_metadata_sha256")
        != TRAIN_CACHE_METADATA_IDENTITY_SHA256
        or payload.get("target_shape") != TARGET_SHAPE
        or payload.get("source_storage_dtype") != "float16"
        or payload.get("encode_decode_dtype") != "float32"
        or payload.get("declared_roundtrip_max_abs_tolerance") != 2e-5
        or payload.get("protected_test_accessed") is not False
    ):
        raise WorkflowError("D1 normalization identity or scientific contract changed")


def validate_inputs(inputs: WorkflowInputs, *, phase: str) -> dict[str, Any]:
    source = validate_source(inputs.expected_commit)
    data_root = _canonical_existing_directory(inputs.data_root, "DROID data root")
    cache_root = _canonical_existing_directory(
        inputs.semantic_cache_root, "semantic cache root"
    )
    train_manifest = _canonical_existing_file(inputs.train_manifest, "train manifest")
    validation_manifest = _canonical_existing_file(
        inputs.validation_manifest, "validation manifest"
    )
    normalization = _canonical_existing_file(inputs.normalization, "normalization")

    if not inputs.study_root.is_absolute():
        raise WorkflowError("study root must be absolute")
    if not _approved_artifact_path(inputs.study_root):
        raise WorkflowError("study root must be under Lustre")
    if REPO_ROOT == inputs.study_root or REPO_ROOT in inputs.study_root.parents:
        raise WorkflowError("large DOE artifacts must be outside the Git repository")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", inputs.study_root.name) is None:
        raise WorkflowError("study root basename is not a safe immutable study ID")
    if phase in {"train", "all"}:
        if inputs.study_root.exists():
            raise WorkflowError("fresh training/all phase requires a nonexistent study root")
        parent = _canonical_existing_directory(inputs.study_root.parent, "study parent")
        if parent / inputs.study_root.name != inputs.study_root:
            raise WorkflowError("study root has a noncanonical parent")
    else:
        _canonical_existing_directory(inputs.study_root, "existing study root")

    train_manifest_record = file_record(train_manifest)
    validation_manifest_record = file_record(validation_manifest)
    normalization_payload = read_json(normalization, "D1 normalization")
    normalization_record = {
        **file_record(normalization),
        "payload_sha256": sha256_json(normalization_payload),
    }
    expected_records = (
        (train_manifest_record, TRAIN_MANIFEST_SHA256, "train manifest"),
        (validation_manifest_record, VALIDATION_MANIFEST_SHA256, "validation manifest"),
        (normalization_record, NORMALIZATION_SHA256, "D1 normalization"),
    )
    for record, digest, label in expected_records:
        if record["sha256"] != digest:
            raise WorkflowError(f"{label} SHA-256 changed")

    train_metadata_path = cache_root / "train" / "metadata.json"
    validation_metadata_path = cache_root / "val" / "metadata.json"
    train_target_path = cache_root / "train" / "targets.fp16.npy"
    validation_target_path = cache_root / "val" / "targets.fp16.npy"
    records = {
        "train_cache_metadata": file_record(train_metadata_path),
        "validation_cache_metadata": file_record(validation_metadata_path),
        "train_target": file_record(train_target_path),
        "validation_target": file_record(validation_target_path),
    }
    expected_cache_hashes = {
        "train_cache_metadata": TRAIN_CACHE_METADATA_FILE_SHA256,
        "validation_cache_metadata": VALIDATION_CACHE_METADATA_FILE_SHA256,
        "train_target": TRAIN_TARGET_SHA256,
        "validation_target": VALIDATION_TARGET_SHA256,
    }
    for name, expected in expected_cache_hashes.items():
        if records[name]["sha256"] != expected:
            raise WorkflowError(f"frozen {name} SHA-256 changed")
    train_metadata = read_json(train_metadata_path, "train cache metadata")
    validation_metadata = read_json(validation_metadata_path, "validation cache metadata")
    _validate_cache_metadata(
        train_metadata,
        split="train",
        clips=64_000,
        manifest_sha256=TRAIN_MANIFEST_SHA256,
        target_sha256=TRAIN_TARGET_SHA256,
    )
    _validate_cache_metadata(
        validation_metadata,
        split="val",
        clips=VALIDATION_CLIPS,
        manifest_sha256=VALIDATION_MANIFEST_SHA256,
        target_sha256=VALIDATION_TARGET_SHA256,
    )
    if sha256_json(train_metadata) != TRAIN_CACHE_METADATA_IDENTITY_SHA256:
        raise WorkflowError("canonical train cache metadata identity changed")
    _validate_normalization(normalization_payload)

    return {
        "repository": source,
        "data_root": str(data_root),
        "semantic_cache_root": str(cache_root),
        "train_manifest": train_manifest_record,
        "validation_manifest": validation_manifest_record,
        "normalization": normalization_record,
        **records,
        "pca_sha256": PCA_SHA256,
        "protected_test_accessed": False,
    }


def distributed_prefix() -> list[str]:
    return [
        python_executable(),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
    ]


def training_run_id(kind: str, arm: Arm) -> str:
    if kind == "calibration":
        return f"calibration-{arm.slug}-seed{TRAIN_SEED}-u{CALIBRATION_UPDATES:06d}"
    if kind == "primary":
        return f"primary-{arm.slug}-seed{TRAIN_SEED}-u{PRIMARY_UPDATES:06d}"
    raise ValueError(f"unknown training kind: {kind}")


def training_run_dir(study_root: Path, kind: str, arm: Arm) -> Path:
    return study_root / "training" / kind / training_run_id(kind, arm)


def evaluation_run_id(arm: Arm) -> str:
    return f"development-{arm.slug}-seed{EVALUATION_SEED}"


def evaluation_run_dir(study_root: Path, arm: Arm) -> Path:
    return study_root / "evaluation" / evaluation_run_id(arm)


def build_train_command(inputs: WorkflowInputs, kind: str, arm: Arm) -> list[str]:
    updates = CALIBRATION_UPDATES if kind == "calibration" else PRIMARY_UPDATES
    checkpoint_every = updates if kind == "calibration" else 500
    command = [
        *distributed_prefix(),
        str(TRAINER),
        "train",
        "--artifact-root",
        str(inputs.study_root / "training" / kind),
        "--run-id",
        training_run_id(kind, arm),
        "--data-root",
        str(inputs.data_root),
        "--semantic-cache-root",
        str(inputs.semantic_cache_root),
        "--train-manifest",
        str(inputs.train_manifest),
        "--validation-manifest",
        str(inputs.validation_manifest),
        "--target-mode",
        arm.target_mode,
        "--flow-loss-weight",
        "1.0",
        "--temporal-velocity-loss-weight",
        str(arm.temporal_weight),
        "--action-margin-loss-weight",
        "0.0",
        "--self-rollin-probability",
        str(arm.rollin_probability),
        "--self-rollin-later-time-rule",
        "sampled_final_clock_fraction",
        "--updates",
        str(updates),
        "--checkpoint-every",
        str(checkpoint_every),
        "--seed",
        str(TRAIN_SEED),
        "--global-batch-size",
        str(GLOBAL_BATCH_SIZE),
        "--micro-batch-size",
        str(MICRO_BATCH_SIZE),
        "--workers",
        str(WORKERS_PER_RANK),
        "--log-every",
        "10",
        "--wandb",
        "--wandb-entity",
        WANDB_ENTITY,
        "--wandb-project",
        WANDB_PROJECT,
        "--wandb-private-project-ack",
    ]
    if arm.uses_normalization:
        command.extend(("--normalization", str(inputs.normalization)))
    return command


def build_evaluation_command(inputs: WorkflowInputs, arm: Arm) -> list[str]:
    return [
        *distributed_prefix(),
        str(EVALUATOR),
        "eval",
        "--artifact-root",
        str(inputs.study_root / "evaluation"),
        "--run-id",
        evaluation_run_id(arm),
        "--data-root",
        str(inputs.data_root),
        "--semantic-cache-root",
        str(inputs.semantic_cache_root),
        "--manifest",
        str(inputs.validation_manifest),
        "--checkpoint",
        str(training_run_dir(inputs.study_root, "primary", arm) / "checkpoints" / "update_005000.pt"),
        "--calibration-checkpoint",
        str(training_run_dir(inputs.study_root, "calibration", arm) / "checkpoints" / "update_000200.pt"),
        "--implementation-registration",
        str(inputs.study_root / "implementation_registration.json"),
        "--seed",
        str(EVALUATION_SEED),
        "--workers",
        "2",
        "--eval-batch-size",
        "8",
        "--wandb",
        "--wandb-entity",
        WANDB_ENTITY,
        "--wandb-project",
        WANDB_PROJECT,
        "--wandb-private-project-ack",
    ]


def build_analysis_command(inputs: WorkflowInputs) -> list[str]:
    command = [
        python_executable(),
        str(ANALYZER),
        "--artifact-root",
        str(inputs.study_root / "analysis"),
        "--run-id",
        f"development-gate-seed{BOOTSTRAP_SEED}",
    ]
    flags = {
        "ABS": "--abs-summary",
        "ABS-T": "--abs-t-summary",
        "DELTA": "--delta-summary",
        "DELTA-T": "--delta-t-summary",
        "DELTA-R": "--delta-r-summary",
    }
    for arm in ARMS:
        command.extend((flags[arm.name], str(evaluation_run_dir(inputs.study_root, arm) / "summary.json")))
    return command


def build_registration_command(inputs: WorkflowInputs) -> list[str]:
    return [
        python_executable(),
        str(EVALUATOR),
        "register",
        "--output",
        str(inputs.study_root / "implementation_registration.json"),
    ]


def build_steps(inputs: WorkflowInputs) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {
            "kind": "registration",
            "arm": None,
            "command": build_registration_command(inputs),
            "output": str(inputs.study_root / "implementation_registration.json"),
        }
    ]
    for kind in ("calibration", "primary"):
        for arm in ARMS:
            updates = CALIBRATION_UPDATES if kind == "calibration" else PRIMARY_UPDATES
            steps.append(
                {
                    "kind": kind,
                    "arm": arm.name,
                    "command": build_train_command(inputs, kind, arm),
                    "output": str(training_run_dir(inputs.study_root, kind, arm)),
                    "checkpoint": str(
                        training_run_dir(inputs.study_root, kind, arm)
                        / "checkpoints"
                        / f"update_{updates:06d}.pt"
                    ),
                }
            )
    for arm in ARMS:
        steps.append(
            {
                "kind": "evaluation",
                "arm": arm.name,
                "command": build_evaluation_command(inputs, arm),
                "output": str(evaluation_run_dir(inputs.study_root, arm)),
            }
        )
    steps.append(
        {
            "kind": "analysis",
            "arm": None,
            "command": build_analysis_command(inputs),
            "output": str(
                inputs.study_root
                / "analysis"
                / f"development-gate-seed{BOOTSTRAP_SEED}"
            ),
        }
    )
    return steps


def build_plan(inputs: WorkflowInputs, input_records: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "schema": SCHEMA,
        "status": "planned_before_candidate_metrics",
        "clock_convention": "sigma=1 is noise; sigma=0 is clean; trainer uses clean-time t=1-sigma",
        "study_root": str(inputs.study_root),
        "source": input_records["repository"],
        "frozen_inputs": dict(input_records),
        "geometry": {
            "nodes": 1,
            "gpus": WORLD_SIZE,
            "world_size": WORLD_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "local_optimizer_batch_size": MICRO_BATCH_SIZE,
            "micro_batch_size_per_rank": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": 1,
            "workers_per_rank": WORKERS_PER_RANK,
            "seed": TRAIN_SEED,
        },
        "wandb": {
            "enabled": True,
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "group": None,
            "private_project_acknowledgement_required": True,
            "credentials_persisted": False,
        },
        "order": "register; all five 200-update calibrations; all five 5000-update primary runs; optional five-arm development evaluation; optional analysis",
        "arms": [arm.name for arm in ARMS],
        "development_nfe_grid": NFE_GRID,
        "protected_test_accessed": False,
        "steps": build_steps(inputs),
    }
    return {**unsigned, "identity_sha256": sha256_json(unsigned)}


def selected_steps(plan: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    steps = plan.get("steps")
    if not isinstance(steps, list):
        raise WorkflowError("workflow plan lacks steps")
    if phase == "train":
        kinds = {"registration", "calibration", "primary"}
    elif phase == "evaluate":
        kinds = {"evaluation", "analysis"}
    elif phase == "all":
        kinds = {"registration", "calibration", "primary", "evaluation", "analysis"}
    else:
        raise WorkflowError(f"unknown workflow phase: {phase}")
    return [dict(step) for step in steps if step.get("kind") in kinds]


def render_plan(plan: Mapping[str, Any], phase: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "dry_run_no_files_created_no_commands_executed",
        "plan_identity_sha256": plan["identity_sha256"],
        "phase": phase,
        "fixed_environment": {
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WANDB_ENTITY": WANDB_ENTITY,
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_RUN_GROUP": None,
            "WANDB_MODE": "online",
        },
        "commands": [
            {
                "kind": step["kind"],
                "arm": step.get("arm"),
                "command_argv": step["command"],
                "command_shell": shlex.join(step["command"]),
                "output": step["output"],
            }
            for step in selected_steps(plan, phase)
        ],
        "secrets_persisted": False,
        "protected_test_accessed": False,
    }


def sanitized_environment(study_root: Path, source: Mapping[str, str] | None = None) -> dict[str, str]:
    original = dict(os.environ if source is None else source)
    result = {
        key: value
        for key, value in original.items()
        if key in SAFE_ENV_EXACT or key.startswith(SAFE_ENV_PREFIXES)
    }
    result.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "8",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "WANDB_ENTITY": WANDB_ENTITY,
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_MODE": "online",
            "WANDB_SILENT": "true",
            "WANDB_DIR": str(study_root / "wandb"),
            "WANDB_CACHE_DIR": str(study_root / "wandb-cache"),
            "WANDB_CONFIG_DIR": str(study_root / "wandb-config"),
        }
    )
    for forbidden in ("WANDB_RUN_GROUP", "WANDB_GROUP", "WANDB_JOB_TYPE", "WANDB_TAGS", "WANDB_NOTES"):
        result.pop(forbidden, None)
    return result


def validate_slurm_runtime() -> dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise WorkflowError("--execute is permitted only inside a Slurm allocation")
    if os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
        raise WorkflowError("the DOE is non-requeueable")
    if os.environ.get("SLURM_JOB_NUM_NODES", "1") != "1":
        raise WorkflowError("the DOE requires exactly one node")
    if os.environ.get("SLURM_NTASKS", "1") != "1":
        raise WorkflowError("the allocation must expose one launcher task")
    probe = (
        "import json,torch; "
        "n=torch.cuda.device_count(); "
        "names=[torch.cuda.get_device_name(i) for i in range(n)]; "
        "print(json.dumps({'available':torch.cuda.is_available(),'count':n,'names':names}))"
    )
    try:
        output = subprocess.run(
            [python_executable(), "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout.strip()
        gpu = json.loads(output)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise WorkflowError("cannot attest the allocated CUDA runtime") from exc
    if gpu.get("available") is not True or gpu.get("count") != WORLD_SIZE:
        raise WorkflowError("the allocation must expose exactly eight CUDA devices")
    names = gpu.get("names")
    if not isinstance(names, list) or len(names) != WORLD_SIZE or any("B200" not in str(name) for name in names):
        raise WorkflowError("all eight allocated GPUs must be NVIDIA B200 devices")
    return {
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "nodes": 1,
        "tasks": 1,
        "gpus": gpu,
        "requeued": False,
    }


def _expected_checkpoint_updates(updates: int) -> list[int]:
    if updates == CALIBRATION_UPDATES:
        return [CALIBRATION_UPDATES]
    return list(range(500, PRIMARY_UPDATES + 1, 500))


def _arm_config_valid(config: Mapping[str, Any], arm: Arm) -> bool:
    loss = config.get("loss")
    rollin = config.get("self_rollin")
    normalization = config.get("normalization")
    return bool(
        isinstance(loss, Mapping)
        and isinstance(rollin, Mapping)
        and config.get("doe_arm") == arm.name
        and config.get("target_mode") == arm.target_mode
        and loss.get("flow_weight") == 1.0
        and loss.get("normalized_temporal_velocity_weight") == arm.temporal_weight
        and loss.get("action_shuffle_margin_weight") == 0.0
        and rollin.get("probability") == arm.rollin_probability
        and rollin.get("later_time_rule") == "sampled_final_clock_fraction"
        and ((normalization is not None) == arm.uses_normalization)
        and (
            not arm.uses_normalization
            or (
                isinstance(normalization, Mapping)
                and normalization.get("sha256") == NORMALIZATION_SHA256
            )
        )
    )


def validate_training_receipt(
    run_dir: Path,
    *,
    kind: str,
    arm: Arm,
    inputs: WorkflowInputs,
) -> dict[str, Any]:
    updates = CALIBRATION_UPDATES if kind == "calibration" else PRIMARY_UPDATES
    config_path = run_dir / "resolved_config.json"
    complete_path = run_dir / "complete.json"
    checkpoint_path = run_dir / "checkpoints" / f"update_{updates:06d}.pt"
    config = read_json(config_path, f"{kind} {arm.name} config")
    complete = read_json(complete_path, f"{kind} {arm.name} completion")
    config_record = file_record(config_path)
    complete_record = file_record(complete_path)
    checkpoint_record = file_record(checkpoint_path)
    source = config.get("source")
    datasets = config.get("datasets")
    manifests = datasets if isinstance(datasets, Mapping) else {}
    caches = manifests.get("semantic_cache") if isinstance(manifests, Mapping) else {}
    train_cache = caches.get("train") if isinstance(caches, Mapping) else {}
    validation_cache = caches.get("validation") if isinstance(caches, Mapping) else {}
    wandb = config.get("wandb")
    optimizer = config.get("optimizer")
    ema = config.get("ema")
    normalization = config.get("normalization")
    expected_role = "numerical_calibration" if kind == "calibration" else "primary_5k"
    if (
        config.get("schema") != TRAINING_SCHEMA
        or config.get("command") != "train"
        or config.get("run_role") != expected_role
        or config.get("updates") != updates
        or config.get("checkpoint_updates") != _expected_checkpoint_updates(updates)
        or config.get("seed") != TRAIN_SEED
        or config.get("global_batch_size") != GLOBAL_BATCH_SIZE
        or config.get("world_size") != WORLD_SIZE
        or config.get("local_optimizer_batch_size") != MICRO_BATCH_SIZE
        or config.get("micro_batch_size_per_rank") != MICRO_BATCH_SIZE
        or config.get("gradient_accumulation_steps") != 1
        or config.get("workers_per_rank") != WORKERS_PER_RANK
        or config.get("initialization") != "from_scratch_deterministic_no_pretrained_weights"
        or config.get("dtype") != "bfloat16"
        or config.get("target_shape") != TARGET_SHAPE
        or config.get("parameter_count") != MODEL_PARAMETERS
        or config.get("entrypoint") != file_record(TRAINER)
        or optimizer
        != {
            "name": "AdamW",
            "learning_rate": 5e-5,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "warmup_updates": 500,
            "after_warmup": "constant",
            "gradient_clip_norm": 1.0,
        }
        or ema
        != {
            "decay": 0.9999,
            "schedule": EMA_SCHEDULE,
            "reported_samples_use": True,
            "short_run_initialization_bias_corrected": True,
        }
        or config.get("video_loss_enabled") is not False
        or config.get("future_rgb_model_input") is not False
        or config.get("teacher_model_calls") != 0
        or config.get("protected_test_accessed") is not False
        or not isinstance(source, Mapping)
        or source.get("commit") != inputs.expected_commit
        or source.get("dirty") is not False
        or not _arm_config_valid(config, arm)
        or manifests.get("train") != file_record(inputs.train_manifest)
        or manifests.get("validation") != file_record(inputs.validation_manifest)
        or train_cache.get("target_sha256") != TRAIN_TARGET_SHA256
        or validation_cache.get("target_sha256") != VALIDATION_TARGET_SHA256
        or train_cache.get("pca_sha256") != PCA_SHA256
        or validation_cache.get("pca_sha256") != PCA_SHA256
        or not isinstance(wandb, Mapping)
        or wandb.get("enabled") is not True
        or wandb.get("entity") != WANDB_ENTITY
        or wandb.get("project") != WANDB_PROJECT
        or wandb.get("group") is not None
        or wandb.get("private_project_acknowledged") is not True
        or (
            arm.uses_normalization
            and normalization != normalization_file_record(inputs.normalization)
        )
    ):
        raise WorkflowError(f"{kind} {arm.name} resolved config violates the DOE contract")
    expected_config_identity = sha256_json(
        {
            key: value
            for key, value in config.items()
            if key not in {"checkpoint_updates", "experiment_identity_sha256"}
        }
    )
    if config.get("experiment_identity_sha256") != expected_config_identity:
        raise WorkflowError(f"{kind} {arm.name} experiment identity changed")
    if (
        complete.get("schema") != TRAINING_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("completed_updates") != updates
        or complete.get("nonfinite_updates") != 0
        or complete.get("target_mode") != arm.target_mode
        or complete.get("video_loss_enabled") is not False
        or complete.get("future_rgb_model_input") is not False
        or complete.get("teacher_model_calls") != 0
        or complete.get("protected_test_accessed") is not False
        or complete.get("resolved_config_sha256") != sha256_json(config)
        or complete.get("experiment_identity_sha256") != expected_config_identity
        or complete.get("resolved_config") != config_record
        or complete.get("checkpoint") != checkpoint_record
    ):
        raise WorkflowError(f"{kind} {arm.name} completion/checkpoint receipt is invalid")
    return {
        "kind": kind,
        "arm": arm.name,
        "updates": updates,
        "resolved_config": config_record,
        "complete": complete_record,
        "checkpoint": checkpoint_record,
        "experiment_identity_sha256": expected_config_identity,
        "config": config,
    }


def _calibration_primary_common(config: Mapping[str, Any]) -> dict[str, Any]:
    common = copy.deepcopy(dict(config))
    for key in ("run_role", "updates", "checkpoint_updates", "experiment_identity_sha256"):
        common.pop(key, None)
    return common


def _doe_common(config: Mapping[str, Any]) -> dict[str, Any]:
    common = copy.deepcopy(dict(config))
    for key in ("doe_arm", "promotion_status", "target_mode", "normalization", "experiment_identity_sha256"):
        common.pop(key, None)
    loss = common.get("loss")
    rollin = common.get("self_rollin")
    if not isinstance(loss, dict) or not isinstance(rollin, dict):
        raise WorkflowError("training config lacks DOE mechanism fields")
    loss.pop("normalized_temporal_velocity_weight", None)
    rollin.pop("probability", None)
    return common


def validate_training_set(inputs: WorkflowInputs) -> dict[str, Any]:
    calibration: dict[str, dict[str, Any]] = {}
    primary: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        calibration[arm.name] = validate_training_receipt(
            training_run_dir(inputs.study_root, "calibration", arm),
            kind="calibration",
            arm=arm,
            inputs=inputs,
        )
        primary[arm.name] = validate_training_receipt(
            training_run_dir(inputs.study_root, "primary", arm),
            kind="primary",
            arm=arm,
            inputs=inputs,
        )
        if _calibration_primary_common(calibration[arm.name]["config"]) != _calibration_primary_common(primary[arm.name]["config"]):
            raise WorkflowError(f"{arm.name} calibration is not primary-equivalent")
    calibration_common = {sha256_json(_doe_common(item["config"])) for item in calibration.values()}
    primary_common = {sha256_json(_doe_common(item["config"])) for item in primary.values()}
    if len(calibration_common) != 1 or len(primary_common) != 1:
        raise WorkflowError("DOE arms do not share exact non-mechanism training geometry")
    return {
        "schema": "causal-vjepa2-temporal-doe-training-receipts-v1",
        "status": "complete",
        "calibration_before_primary": True,
        "calibration_common_identity_sha256": next(iter(calibration_common)),
        "primary_common_identity_sha256": next(iter(primary_common)),
        "calibration": {name: {k: v for k, v in item.items() if k != "config"} for name, item in calibration.items()},
        "primary": {name: {k: v for k, v in item.items() if k != "config"} for name, item in primary.items()},
        "nonfinite_updates": 0,
        "protected_test_accessed": False,
    }


def validate_evaluation_receipt(
    run_dir: Path, arm: Arm, *, inputs: WorkflowInputs
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path, f"{arm.name} development summary")
    metrics_path = run_dir / "per_clip_metrics.jsonl"
    metrics_record = file_record(metrics_path)
    checkpoint = summary.get("checkpoint")
    manifest = summary.get("manifest")
    cache = summary.get("semantic_cache")
    if (
        summary.get("schema") != EVALUATION_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("split") != "val"
        or summary.get("development_selection_split") is not True
        or summary.get("protected_test_accessed") is not False
        or summary.get("arm") != arm.name
        or summary.get("record_count") != VALIDATION_CLIPS * len(NFE_GRID) * CONTROLS
        or summary.get("cell_count") != len(NFE_GRID) * CONTROLS
        or summary.get("per_clip_metrics") != metrics_record
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("sha256")
        != file_record(
            training_run_dir(inputs.study_root, "primary", arm)
            / "checkpoints"
            / "update_005000.pt"
        )["sha256"]
        or not isinstance(manifest, Mapping)
        or manifest.get("sha256") != VALIDATION_MANIFEST_SHA256
        or not isinstance(cache, Mapping)
        or cache.get("target_sha256") != VALIDATION_TARGET_SHA256
        or summary.get("clean_future_target_entered_deployable_sampler") is not False
        or summary.get("future_rgb_entered_deployable_sampler") is not False
        or summary.get("teacher_model_calls") != 0
    ):
        raise WorkflowError(f"{arm.name} development evaluation receipt is invalid")
    return {
        "arm": arm.name,
        "summary": file_record(summary_path),
        "per_clip_metrics": metrics_record,
        "record_count": summary["record_count"],
        "protected_test_accessed": False,
    }


def validate_analysis_receipt(study_root: Path) -> dict[str, Any]:
    run_dir = study_root / "analysis" / f"development-gate-seed{BOOTSTRAP_SEED}"
    analysis_path = run_dir / "development_analysis.json"
    selection_path = run_dir / "frozen_selection.json"
    analysis = read_json(analysis_path, "development analysis")
    selection = read_json(selection_path, "frozen development selection")
    if (
        analysis.get("status") not in {"one_candidate_selected", "no_candidate_passed"}
        or analysis.get("split") != "val"
        or analysis.get("protected_test_accessed") is not False
        or analysis.get("paired_clips") != VALIDATION_CLIPS
        or analysis.get("protocol_frozen") is not True
        or selection.get("status") not in {"frozen_candidate", "frozen_no_selection"}
        or selection.get("selection_split") != "val"
        or selection.get("selection_used_protected_test") is not False
        or selection.get("protected_test_accessed") is not False
        or selection.get("selection_count") not in {0, 1}
    ):
        raise WorkflowError("development analysis/selection receipt is invalid")
    return {
        "schema": "causal-vjepa2-temporal-doe-development-receipts-v1",
        "status": "complete",
        "development_analysis": file_record(analysis_path),
        "frozen_selection": file_record(selection_path),
        "selection_status": selection["status"],
        "selection_count": selection["selection_count"],
        "protected_test_accessed": False,
    }


def _run(command: Sequence[str], *, environment: Mapping[str, str]) -> None:
    print(f"RUN {shlex.join(command)}", flush=True)
    try:
        subprocess.run(
            list(command),
            cwd=str(REPO_ROOT),
            env=dict(environment),
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WorkflowError(f"workflow command failed: {shlex.join(command)}") from exc


def _load_and_match_plan(inputs: WorkflowInputs, current_plan: Mapping[str, Any]) -> None:
    path = inputs.study_root / "workflow_plan.json"
    prior = read_json(path, "existing workflow plan")
    if prior != current_plan or prior.get("identity_sha256") != sha256_json(
        {key: value for key, value in prior.items() if key != "identity_sha256"}
    ):
        raise WorkflowError("existing workflow plan differs from current frozen inputs")


def execute_workflow(
    inputs: WorkflowInputs,
    *,
    phase: str,
    plan: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    if phase in {"train", "all"}:
        inputs.study_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        for path in (
            inputs.study_root / "training" / "calibration",
            inputs.study_root / "training" / "primary",
            inputs.study_root / "evaluation",
            inputs.study_root / "analysis",
            inputs.study_root / "wandb",
            inputs.study_root / "wandb-cache",
            inputs.study_root / "wandb-config",
        ):
            path.mkdir(parents=True, exist_ok=False)
        atomic_write_json(inputs.study_root / "workflow_plan.json", plan, exclusive=True)
    else:
        _load_and_match_plan(inputs, plan)

    environment = sanitized_environment(inputs.study_root)
    state_path = inputs.study_root / "workflow_state.json"
    state: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "status": "running",
        "phase": phase,
        "plan_identity_sha256": plan["identity_sha256"],
        "runtime": dict(runtime),
        "completed_steps": [],
        "secrets_persisted": False,
        "protected_test_accessed": False,
    }
    if state_path.exists():
        prior_state = read_json(state_path, "workflow state")
        state["prior_phase"] = prior_state.get("phase")
        state["prior_status"] = prior_state.get("status")
    atomic_write_json(state_path, state, exclusive=False)

    try:
        if phase in {"train", "all"}:
            _run(build_registration_command(inputs), environment=environment)
            state["completed_steps"].append("registration")
            atomic_write_json(state_path, state, exclusive=False)
            for kind in ("calibration", "primary"):
                for arm in ARMS:
                    _run(build_train_command(inputs, kind, arm), environment=environment)
                    validate_training_receipt(
                        training_run_dir(inputs.study_root, kind, arm),
                        kind=kind,
                        arm=arm,
                        inputs=inputs,
                    )
                    state["completed_steps"].append(f"{kind}:{arm.name}")
                    atomic_write_json(state_path, state, exclusive=False)
            receipts = validate_training_set(inputs)
            atomic_write_json(
                inputs.study_root / "training_receipts.json", receipts, exclusive=True
            )
            state["training_receipts"] = file_record(
                inputs.study_root / "training_receipts.json"
            )

        if phase in {"evaluate", "all"}:
            if not (inputs.study_root / "implementation_registration.json").is_file():
                raise WorkflowError("evaluation requires the pre-metric implementation registration")
            receipts = validate_training_set(inputs)
            if phase == "evaluate":
                training_receipts_path = inputs.study_root / "training_receipts.json"
                if not training_receipts_path.is_file():
                    raise WorkflowError(
                        "evaluate-only phase requires frozen training_receipts.json"
                    )
                if read_json(
                    training_receipts_path, "frozen training receipts"
                ) != receipts:
                    raise WorkflowError(
                        "frozen training receipts differ from recomputed evidence"
                    )
            evaluation_receipts: dict[str, Any] = {}
            for arm in ARMS:
                _run(build_evaluation_command(inputs, arm), environment=environment)
                evaluation_receipts[arm.name] = validate_evaluation_receipt(
                    evaluation_run_dir(inputs.study_root, arm), arm, inputs=inputs
                )
                state["completed_steps"].append(f"evaluation:{arm.name}")
                atomic_write_json(state_path, state, exclusive=False)
            _run(build_analysis_command(inputs), environment=environment)
            analysis_receipt = validate_analysis_receipt(inputs.study_root)
            development = {
                "schema": "causal-vjepa2-temporal-doe-development-workflow-receipts-v1",
                "status": "complete",
                "training_common_identity_sha256": receipts[
                    "primary_common_identity_sha256"
                ],
                "evaluations": evaluation_receipts,
                "analysis": analysis_receipt,
                "protected_test_accessed": False,
            }
            atomic_write_json(
                inputs.study_root / "development_receipts.json",
                development,
                exclusive=True,
            )
            state["completed_steps"].append("analysis")
            state["development_receipts"] = file_record(
                inputs.study_root / "development_receipts.json"
            )

        state["status"] = (
            "training_complete" if phase == "train" else "development_complete"
        )
        atomic_write_json(state_path, state, exclusive=False)
    except Exception as exc:
        state["status"] = "operational_failure_no_scientific_retry_authorized"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        atomic_write_json(state_path, state, exclusive=False)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="train")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--semantic-cache-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument(
        "--ack-private-wandb-project",
        action="store_true",
        help="Acknowledge that zijiandu/dual-video-diffusion-private is private.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.ack_private_wandb_project:
            raise WorkflowError("explicit private W&B project acknowledgement is required")
        inputs = WorkflowInputs(
            expected_commit=args.expected_commit,
            study_root=args.study_root,
            data_root=args.data_root,
            semantic_cache_root=args.semantic_cache_root,
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            normalization=args.normalization,
        )
        records = validate_inputs(inputs, phase=args.phase)
        plan = build_plan(inputs, records)
        if not args.execute:
            print(json.dumps(render_plan(plan, args.phase), indent=2, sort_keys=True))
            return 0
        runtime = validate_slurm_runtime()
        execute_workflow(inputs, phase=args.phase, plan=plan, runtime=runtime)
        return 0
    except (WorkflowError, OSError, ValueError) as exc:
        print(f"Temporal follow-up DOE workflow error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
