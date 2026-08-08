#!/usr/bin/env python3
"""Fail-closed workflow for the matched C-ABS/AINC-OFF continuation.

Run the phases in order: ``cache``, ``train``, then ``evaluate``.  Each phase
must execute inside a fresh one-node/eight-B200 Slurm allocation.  The workflow
registers and trains its own same-commit absolute control, never reuses the
running temporal DOE's ABS metrics, never submits another job, and contains no
protected-test command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_causal_vjepa2_observed_anchor as analyzer  # noqa: E402
from tools import analyze_causal_vjepa2_temporal_targets as temporal_analysis  # noqa: E402
from tools import causal_vjepa2_observed_anchor as anchor  # noqa: E402
from tools import causal_vjepa2_observed_anchor_screen as ainc  # noqa: E402
from tools import video_latent_forcing_poc as vlf  # noqa: E402
from tools.slurm import temporal_followup_doe as temporal_workflow  # noqa: E402


SCHEMA = "causal-vjepa2-observed-anchor-workflow-v1"
STATE_SCHEMA = "causal-vjepa2-observed-anchor-workflow-state-v1"
PHASES = ("cache", "train", "evaluate")
WORLD_SIZE = 8
GLOBAL_BATCH_SIZE = 256
MICRO_BATCH_SIZE = 32
TRAIN_WORKERS = 4
EVAL_BATCH_SIZE = 8
EVAL_WORKERS = 2
TRAIN_SEED = 1234
EVALUATION_SEED = 20260801
LOCAL_ABS = temporal_workflow.ARM_BY_NAME["ABS"]
WANDB_ENTITY = "zijiandu"
WANDB_PROJECT = "dual-video-diffusion-private"
TRAIN_MANIFEST_SHA256 = (
    "cc10bccece1ac0e20abacf30ee0db60339145ec54ab2e28af977ded21e02f27e"
)
VALIDATION_MANIFEST_SHA256 = (
    "b8773a8627e887bf0c0a31cfae6ff537ba6a99e0b1b4f11efec011e3983d8d99"
)
TRAIN_CACHE_METADATA_SHA256 = (
    "7eb40233b10f001ad5f49beb037af0e787414fcd4fc206e8334e6845d6611182"
)
VALIDATION_CACHE_METADATA_SHA256 = (
    "9c9f8e7d0f67fd50b9f6fbe4696eb13bf39b897aa0e5d061bc4daa22bf882a55"
)
TRAIN_TARGET_SHA256 = (
    "547c4579cf978ac2b9527cb038693259af678a2b07268ab1434706dc128051c4"
)
VALIDATION_TARGET_SHA256 = (
    "ed0ee6c76233b6072691854f767ce0e524867826b6d811c43988791f4011b034"
)
SAFE_ENV_EXACT = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "WANDB_API_KEY",
    "WANDB_BASE_URL",
}
SAFE_ENV_PREFIXES = ("SLURM_", "CUDA_", "NVIDIA_", "NCCL_", "UCX_", "PMI_", "PMIX_")


class WorkflowError(RuntimeError):
    """A source, input, execution, or receipt contract failed closed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise WorkflowError(f"required regular file is missing: {path}")
    return vlf.file_record(path)


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = vlf.load_json(path, label)
    if not isinstance(value, dict):  # defensive around imported helper
        raise WorkflowError(f"{label} must be a JSON object")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any], *, exclusive: bool) -> None:
    vlf.atomic_write_json(path, value, exclusive=exclusive)


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


def _git_external(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise WorkflowError(f"cannot validate external Git checkout: {root}") from exc


def validate_source(expected_commit: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise WorkflowError("--expected-commit must be a full lowercase SHA")
    if Path(_git("rev-parse", "--show-toplevel")).resolve() != REPO_ROOT:
        raise WorkflowError("workflow repository root changed")
    if _git("rev-parse", "HEAD") != expected_commit:
        raise WorkflowError("workflow checkout is not the expected commit")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise WorkflowError("execution requires one clean committed checkout")
    required = (
        Path(anchor.__file__),
        Path(ainc.__file__),
        Path(analyzer.__file__),
        Path(temporal_analysis.__file__),
        Path(temporal_workflow.__file__),
        temporal_workflow.TRAINER,
        REPO_ROOT / "tools" / "evaluate_causal_vjepa2_temporal_targets.py",
    )
    for path in required:
        if path.is_symlink() or not path.is_file():
            raise WorkflowError(f"required implementation is missing: {path}")
    return {"root": str(REPO_ROOT), "commit": expected_commit, "dirty": False}


def _canonical_file(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise WorkflowError(f"{label} must be an absolute regular file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise WorkflowError(f"{label} must be canonical")
    return resolved


def _canonical_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise WorkflowError(f"{label} must be an absolute directory")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise WorkflowError(f"{label} must be canonical")
    return resolved


def execution_arguments(args: argparse.Namespace) -> list[str]:
    result = ["--execution-mode", args.execution_mode]
    if args.execution_mode == "post-temporal-no-pass":
        result.extend(("--temporal-selection-record", str(args.temporal_selection_record)))
    return result


def execution_condition(args: argparse.Namespace) -> dict[str, Any]:
    namespace = argparse.Namespace(
        execution_mode=args.execution_mode,
        temporal_selection_record=(
            str(args.temporal_selection_record)
            if args.temporal_selection_record is not None
            else None
        ),
    )
    try:
        return ainc._execution_condition(namespace)  # noqa: SLF001
    except ainc.ObservedAnchorScreenError as exc:
        raise WorkflowError(str(exc)) from exc


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    source = validate_source(args.expected_commit)
    data_root = _canonical_directory(args.data_root, "DROID data root")
    semantic_root = _canonical_directory(args.semantic_cache_root, "semantic cache root")
    vjepa_source = _canonical_directory(args.vjepa_source, "V-JEPA source checkout")
    vjepa_checkpoint = _canonical_file(args.vjepa_checkpoint, "V-JEPA checkpoint")
    train_manifest = _canonical_file(args.train_manifest, "train manifest")
    validation_manifest = _canonical_file(args.validation_manifest, "validation manifest")
    if args.temporal_selection_record is not None:
        _canonical_file(
            args.temporal_selection_record, "temporal frozen selection record"
        )
    if file_record(train_manifest)["sha256"] != TRAIN_MANIFEST_SHA256:
        raise WorkflowError("train manifest bytes changed")
    if file_record(validation_manifest)["sha256"] != VALIDATION_MANIFEST_SHA256:
        raise WorkflowError("validation manifest bytes changed")
    frozen = {
        "train_cache_metadata": (
            semantic_root / "train" / "metadata.json",
            TRAIN_CACHE_METADATA_SHA256,
        ),
        "validation_cache_metadata": (
            semantic_root / "val" / "metadata.json",
            VALIDATION_CACHE_METADATA_SHA256,
        ),
        "train_target": (
            semantic_root / "train" / "targets.fp16.npy",
            TRAIN_TARGET_SHA256,
        ),
        "validation_target": (
            semantic_root / "val" / "targets.fp16.npy",
            VALIDATION_TARGET_SHA256,
        ),
    }
    frozen_records: dict[str, Any] = {}
    for name, (path, digest) in frozen.items():
        record = file_record(path)
        if record["sha256"] != digest:
            raise WorkflowError(f"frozen {name} bytes changed")
        frozen_records[name] = record
    if not args.study_root.is_absolute() or Path("/lustre") not in args.study_root.parents:
        raise WorkflowError("study root must be a new canonical path under /lustre")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", args.study_root.name) is None:
        raise WorkflowError("study root basename is unsafe")
    if args.phase == "cache":
        if args.study_root.exists():
            raise WorkflowError("cache phase requires a nonexistent study root")
        parent = _canonical_directory(args.study_root.parent, "study parent")
        if parent / args.study_root.name != args.study_root:
            raise WorkflowError("study root must have a canonical parent")
    else:
        _canonical_directory(args.study_root, "existing study root")
    condition = execution_condition(args)
    if (
        _git_external(vjepa_source, "rev-parse", "HEAD")
        != anchor.VJEPA2_SOURCE_COMMIT
        or _git_external(vjepa_source, "status", "--porcelain", "--untracked-files=all")
    ):
        raise WorkflowError("V-JEPA source checkout is not the frozen release commit")
    checkpoint_record = file_record(vjepa_checkpoint)
    if checkpoint_record["sha256"] != anchor.VJEPA2_CHECKPOINT_SHA256:
        raise WorkflowError("V-JEPA checkpoint bytes changed")
    return {
        "source": source,
        "data_root": str(data_root),
        "semantic_cache_root": str(semantic_root),
        "vjepa_source": {
            "path": str(vjepa_source),
            "commit": anchor.VJEPA2_SOURCE_COMMIT,
        },
        "vjepa_checkpoint": checkpoint_record,
        "train_manifest": file_record(train_manifest),
        "validation_manifest": file_record(validation_manifest),
        "comparison_design": {
            "kind": "self_contained_matched_two_arm_continuation",
            "control": "continuation-local C-ABS (machine arm ABS)",
            "external_temporal_abs_numeric_baseline_allowed": False,
        },
        "execution_condition": condition,
        **frozen_records,
        "protected_test_accessed": False,
    }


def distributed_prefix() -> list[str]:
    return [
        str(Path(sys.executable).absolute()),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
    ]


def anchor_cache_root(study: Path) -> Path:
    return study / "anchor_cache"


def normalization_path(study: Path) -> Path:
    return study / "normalization" / "train_increment_normalization.json"


def calibration_dir(study: Path) -> Path:
    return study / "training" / "calibration" / "calibration-ainc-off-seed1234-u000200"


def primary_dir(study: Path) -> Path:
    return study / "training" / "primary" / "primary-ainc-off-seed1234-u005000"


def evaluation_dir(study: Path) -> Path:
    return study / "evaluation" / "development-ainc-off-seed20260801"


def analysis_dir(study: Path) -> Path:
    return study / "analysis" / "development-gate-seed20260807"


def local_abs_inputs(args: argparse.Namespace) -> temporal_workflow.WorkflowInputs:
    """Adapt frozen continuation inputs to the registered ABS implementation."""
    return temporal_workflow.WorkflowInputs(
        expected_commit=args.expected_commit,
        study_root=args.study_root,
        data_root=args.data_root,
        semantic_cache_root=args.semantic_cache_root,
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        # ABS does not consume a normalization.  This path is present only
        # because the shared typed input record covers the full temporal DOE.
        normalization=normalization_path(args.study_root),
    )


def local_abs_training_dir(study: Path, kind: str) -> Path:
    return temporal_workflow.training_run_dir(study, kind, LOCAL_ABS)


def local_abs_evaluation_dir(study: Path) -> Path:
    return temporal_workflow.evaluation_run_dir(study, LOCAL_ABS)


def implementation_registration_path(study: Path) -> Path:
    return study / "implementation_registration.json"


def common_data(args: argparse.Namespace) -> list[str]:
    return [
        "--data-root",
        str(args.data_root),
        "--semantic-cache-root",
        str(args.semantic_cache_root),
        "--anchor-cache-root",
        str(anchor_cache_root(args.study_root)),
        "--train-manifest",
        str(args.train_manifest),
        "--normalization",
        str(normalization_path(args.study_root)),
        *execution_arguments(args),
    ]


def wandb_arguments() -> list[str]:
    return [
        "--wandb",
        "--wandb-entity",
        WANDB_ENTITY,
        "--wandb-project",
        WANDB_PROJECT,
        "--wandb-private-project-ack",
    ]


def cache_commands(args: argparse.Namespace) -> list[list[str]]:
    commands = [
        temporal_workflow.build_registration_command(local_abs_inputs(args))
    ]
    for manifest in (args.train_manifest, args.validation_manifest):
        commands.append(
            [
                *distributed_prefix(),
                str(Path(anchor.__file__)),
                "build-cache",
                "--manifest",
                str(manifest),
                "--data-root",
                str(args.data_root),
                "--semantic-cache-root",
                str(args.semantic_cache_root),
                "--vjepa-source",
                str(args.vjepa_source),
                "--vjepa-checkpoint",
                str(args.vjepa_checkpoint),
                "--output-root",
                str(anchor_cache_root(args.study_root)),
                "--batch-size",
                "1",
            ]
        )
    commands.append(
        [
            str(Path(sys.executable).absolute()),
            str(Path(anchor.__file__)),
            "fit-normalization",
            "--output",
            str(normalization_path(args.study_root)),
            "--train-manifest",
            str(args.train_manifest),
            "--data-root",
            str(args.data_root),
            "--semantic-cache-root",
            str(args.semantic_cache_root),
            "--anchor-cache-root",
            str(anchor_cache_root(args.study_root)),
            "--batch-size",
            "64",
            "--workers",
            str(TRAIN_WORKERS),
        ]
    )
    return commands


def training_command(args: argparse.Namespace, kind: str) -> list[str]:
    if kind == "calibration":
        command_name, run_dir, artifact_root = "calibrate", calibration_dir(args.study_root), args.study_root / "training" / "calibration"
    elif kind == "primary":
        command_name, run_dir, artifact_root = "train", primary_dir(args.study_root), args.study_root / "training" / "primary"
    else:
        raise ValueError(kind)
    command = [
        *distributed_prefix(),
        str(Path(ainc.__file__)),
        command_name,
        *common_data(args),
        "--validation-manifest",
        str(args.validation_manifest),
        "--artifact-root",
        str(artifact_root),
        "--run-id",
        run_dir.name,
        "--global-batch-size",
        str(GLOBAL_BATCH_SIZE),
        "--micro-batch-size",
        str(MICRO_BATCH_SIZE),
        "--workers",
        str(TRAIN_WORKERS),
        "--seed",
        str(TRAIN_SEED),
        "--log-every",
        "10",
        *wandb_arguments(),
    ]
    if kind == "primary":
        command.extend(("--calibration-record", str(calibration_dir(args.study_root) / "complete.json")))
    return command


def local_abs_training_command(args: argparse.Namespace, kind: str) -> list[str]:
    temporal_kind = "calibration" if kind == "calibration" else "primary"
    if kind not in {"calibration", "primary"}:
        raise ValueError(kind)
    return temporal_workflow.build_train_command(
        local_abs_inputs(args), temporal_kind, LOCAL_ABS
    )


def evaluation_commands(args: argparse.Namespace) -> list[list[str]]:
    evaluate_abs = temporal_workflow.build_evaluation_command(
        local_abs_inputs(args), LOCAL_ABS
    )
    evaluate = [
        *distributed_prefix(),
        str(Path(ainc.__file__)),
        "evaluate",
        *common_data(args),
        "--manifest",
        str(args.validation_manifest),
        "--artifact-root",
        str(args.study_root / "evaluation"),
        "--run-id",
        evaluation_dir(args.study_root).name,
        "--checkpoint",
        str(primary_dir(args.study_root) / "checkpoints" / "update_005000.pt"),
        "--eval-batch-size",
        str(EVAL_BATCH_SIZE),
        "--workers",
        str(EVAL_WORKERS),
        "--seed",
        str(EVALUATION_SEED),
        *wandb_arguments(),
    ]
    analyze = [
        str(Path(sys.executable).absolute()),
        str(Path(analyzer.__file__)),
        "--artifact-root",
        str(args.study_root / "analysis"),
        "--run-id",
        analysis_dir(args.study_root).name,
        "--ainc-summary",
        str(evaluation_dir(args.study_root) / "summary.json"),
        "--abs-summary",
        str(local_abs_evaluation_dir(args.study_root) / "summary.json"),
    ]
    return [evaluate_abs, evaluate, analyze]


def phase_commands(args: argparse.Namespace) -> list[list[str]]:
    if args.phase == "cache":
        return cache_commands(args)
    if args.phase == "train":
        # Both numerical calibrations finish before either from-scratch primary.
        return [
            local_abs_training_command(args, "calibration"),
            training_command(args, "calibration"),
            local_abs_training_command(args, "primary"),
            training_command(args, "primary"),
        ]
    return evaluation_commands(args)


def build_plan(args: argparse.Namespace, records: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "schema": SCHEMA,
        "status": "planned_before_observed_anchor_metrics",
        "study_root": str(args.study_root),
        "source": records["source"],
        "frozen_inputs": dict(records),
        "execution_mode": args.execution_mode,
        "comparison_design": {
            "kind": "self_contained_matched_two_arm_continuation",
            "arms": ["C-ABS", "AINC-OFF"],
            "machine_arm_labels": {"C-ABS": "ABS", "AINC-OFF": "AINC-OFF"},
            "same_clean_commit": True,
            "calibrations_before_primaries": True,
            "running_temporal_doe_used_for_authorization_only": (
                args.execution_mode == "post-temporal-no-pass"
            ),
            "external_temporal_abs_numeric_baseline_allowed": False,
        },
        "geometry": {
            "nodes": 1,
            "gpus": WORLD_SIZE,
            "gpu_model": "NVIDIA B200",
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "local_batch_size": MICRO_BATCH_SIZE,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": 1,
            "workers_per_rank": TRAIN_WORKERS,
        },
        "wandb": {
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "group": None,
            "private_project_acknowledged": True,
        },
        "ordered_phases": list(PHASES),
        "commands": {
            phase: [
                # Render all phases into one immutable plan.
                command
                for command in phase_commands(argparse.Namespace(**{**vars(args), "phase": phase}))
            ]
            for phase in PHASES
        },
        "protected_test_commands": [],
        "protected_test_accessed": False,
    }
    return {**unsigned, "identity_sha256": sha256_json(unsigned)}


def sanitized_environment(study_root: Path) -> dict[str, str]:
    result = {
        key: value
        for key, value in os.environ.items()
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
    for name in ("WANDB_RUN_GROUP", "WANDB_GROUP", "WANDB_JOB_TYPE", "WANDB_TAGS", "WANDB_NOTES"):
        result.pop(name, None)
    return result


def validate_slurm_runtime() -> dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise WorkflowError("--execute is allowed only inside Slurm")
    if os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
        raise WorkflowError("observed-anchor phases are non-requeueable")
    if os.environ.get("SLURM_JOB_NUM_NODES", "1") != "1" or os.environ.get("SLURM_NTASKS", "1") != "1":
        raise WorkflowError("phase requires one node and one launcher task")
    required_scheduler = {
        "SLURM_JOB_PARTITION": "batch",
        "SLURM_JOB_ACCOUNT": "coreai_chef_posttrain",
        "SLURM_JOB_QOS": "short",
    }
    if any(os.environ.get(name) != expected for name, expected in required_scheduler.items()):
        raise WorkflowError("phase requires batch/coreai_chef_posttrain/short")
    probe = "import json,torch; print(json.dumps({'count':torch.cuda.device_count(),'names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"
    try:
        gpu = json.loads(
            subprocess.run(
                [str(Path(sys.executable).absolute()), "-c", probe],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise WorkflowError("cannot attest allocated CUDA devices") from exc
    names = gpu.get("names")
    if (
        gpu.get("count") != WORLD_SIZE
        or not isinstance(names, list)
        or len(names) != WORLD_SIZE
        or any("B200" not in str(name) for name in names)
    ):
        raise WorkflowError("phase requires exactly eight NVIDIA B200 GPUs")
    return {"job_id": os.environ["SLURM_JOB_ID"], "gpus": gpu, "protected_test_accessed": False}


def _run(command: Sequence[str], environment: Mapping[str, str]) -> None:
    print(f"RUN {shlex.join(command)}", flush=True)
    try:
        subprocess.run(list(command), cwd=str(REPO_ROOT), env=dict(environment), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WorkflowError(f"phase command failed: {shlex.join(command)}") from exc


def validate_cache_receipts(args: argparse.Namespace) -> dict[str, Any]:
    registration, registration_record = (
        temporal_analysis.evaluator.load_implementation_registration(
            implementation_registration_path(args.study_root),
            current_source=temporal_analysis.screen._source_record(),  # noqa: SLF001
        )
    )
    records = {}
    for split, manifest in (("train", args.train_manifest), ("val", args.validation_manifest)):
        metadata, _ = anchor.validate_anchor_cache(
            manifest_path=manifest,
            anchor_cache_root=anchor_cache_root(args.study_root),
            expected_split=split,
        )
        records[split] = {
            "metadata": file_record(anchor_cache_root(args.study_root) / split / "metadata.json"),
            "cache_id": metadata["cache_id"],
            "anchor_file": dict(metadata["anchor_file"]),
        }
    normalization, normalization_record = anchor.load_increment_normalization(
        normalization_path(args.study_root),
        expected_train_manifest_sha256=TRAIN_MANIFEST_SHA256,
        expected_semantic_cache_metadata_sha256=TRAIN_CACHE_METADATA_SHA256,
        expected_anchor_cache_metadata_sha256=records["train"]["metadata"]["sha256"],
    )
    del normalization
    return {
        "schema": "causal-vjepa2-observed-anchor-cache-receipts-v1",
        "status": "complete",
        "local_abs_implementation_registration": registration_record,
        "local_abs_implementation_registration_identity_sha256": registration[
            "identity_sha256"
        ],
        "anchor_cache": records,
        "normalization": normalization_record,
        "protected_test_accessed": False,
    }


def _training_receipt(run_dir: Path, *, command: str, args: argparse.Namespace) -> dict[str, Any]:
    config_path = run_dir / "resolved_config.json"
    complete_path = run_dir / "complete.json"
    updates = ainc.CALIBRATION_UPDATES if command == "calibrate" else ainc.TRAIN_UPDATES
    checkpoint_path = run_dir / "checkpoints" / f"update_{updates:06d}.pt"
    config = read_json(config_path, f"AINC {command} config")
    complete = read_json(complete_path, f"AINC {command} completion")
    science = config.get("science_identity")
    wandb = config.get("wandb")
    if (
        config.get("schema") != ainc.RUN_SCHEMA
        or config.get("command") != command
        or config.get("doe_arm") != ainc.DOE_ARM
        or config.get("updates") != updates
        or not isinstance(science, Mapping)
        or config.get("science_identity_sha256") != sha256_json(science)
        or science.get("world_size") != WORLD_SIZE
        or science.get("global_batch_size") != GLOBAL_BATCH_SIZE
        or science.get("local_optimizer_batch_size") != MICRO_BATCH_SIZE
        or science.get("micro_batch_size_per_rank") != MICRO_BATCH_SIZE
        or science.get("gradient_accumulation_steps") != 1
        or science.get("workers_per_rank") != TRAIN_WORKERS
        or science.get("seed") != TRAIN_SEED
        or science.get("execution_condition") != execution_condition(args)
        or not isinstance(wandb, Mapping)
        or wandb.get("enabled") is not True
        or wandb.get("entity") != WANDB_ENTITY
        or wandb.get("project") != WANDB_PROJECT
        or wandb.get("group") is not None
        or complete.get("schema") != ainc.RUN_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("command") != command
        or complete.get("completed_updates") != updates
        or complete.get("nonfinite_updates") != 0
        or complete.get("checkpoint") != file_record(checkpoint_path)
        or complete.get("resolved_config") != file_record(config_path)
        or complete.get("protected_test_accessed") is not False
    ):
        raise WorkflowError(f"AINC {command} receipt violates frozen geometry")
    return {
        "command": command,
        "config": file_record(config_path),
        "complete": file_record(complete_path),
        "checkpoint": file_record(checkpoint_path),
        "science_identity_sha256": config["science_identity_sha256"],
        "science_identity": dict(science),
    }


def validate_training_receipts(args: argparse.Namespace) -> dict[str, Any]:
    ainc_calibration = _training_receipt(
        calibration_dir(args.study_root), command="calibrate", args=args
    )
    ainc_primary = _training_receipt(
        primary_dir(args.study_root), command="train", args=args
    )
    abs_inputs = local_abs_inputs(args)
    abs_calibration = temporal_workflow.validate_training_receipt(
        local_abs_training_dir(args.study_root, "calibration"),
        kind="calibration",
        arm=LOCAL_ABS,
        inputs=abs_inputs,
    )
    abs_primary = temporal_workflow.validate_training_receipt(
        local_abs_training_dir(args.study_root, "primary"),
        kind="primary",
        arm=LOCAL_ABS,
        inputs=abs_inputs,
    )
    if ainc_calibration["science_identity"] != ainc_primary["science_identity"]:
        raise WorkflowError("calibration and primary science identities differ")
    if temporal_workflow._calibration_primary_common(  # noqa: SLF001
        abs_calibration["config"]
    ) != temporal_workflow._calibration_primary_common(abs_primary["config"]):  # noqa: SLF001
        raise WorkflowError("C-ABS calibration and primary configurations differ")
    primary_config = read_json(
        Path(ainc_primary["config"]["path"]), "AINC primary config"
    )
    if primary_config.get("calibration_record") != ainc_calibration["complete"]:
        raise WorkflowError("primary run is not bound to the completed calibration")
    analyzer._validate_training_fairness(  # noqa: SLF001
        primary_config, abs_primary["config"]
    )
    return {
        "schema": "causal-vjepa2-observed-anchor-training-receipts-v1",
        "status": "complete",
        "comparison_design": "self_contained_same_commit_C-ABS_vs_AINC-OFF",
        "calibration_before_primary": True,
        "calibration": {
            "C-ABS": {
                key: value
                for key, value in abs_calibration.items()
                if key != "config"
            },
            "AINC-OFF": {
                key: value
                for key, value in ainc_calibration.items()
                if key != "science_identity"
            },
        },
        "primary": {
            "C-ABS": {
                key: value for key, value in abs_primary.items() if key != "config"
            },
            "AINC-OFF": {
                key: value
                for key, value in ainc_primary.items()
                if key != "science_identity"
            },
        },
        "same_commit_training_fairness_validated": True,
        "protected_test_accessed": False,
    }


def validate_development_receipts(args: argparse.Namespace) -> dict[str, Any]:
    local_abs_receipt = temporal_workflow.validate_evaluation_receipt(
        local_abs_evaluation_dir(args.study_root),
        LOCAL_ABS,
        inputs=local_abs_inputs(args),
    )
    abs_evaluation = temporal_analysis.load_evaluation(
        local_abs_evaluation_dir(args.study_root) / "summary.json",
        expected_arm="ABS",
    )
    ainc_evaluation = analyzer.load_evaluation(
        evaluation_dir(args.study_root) / "summary.json",
        abs_evaluation=abs_evaluation,
    )
    analysis_path = analysis_dir(args.study_root) / "development_analysis.json"
    selection_path = analysis_dir(args.study_root) / "frozen_selection.json"
    analysis_payload = read_json(analysis_path, "AINC development analysis")
    selection = read_json(selection_path, "AINC frozen selection")
    analysis_unsigned = {
        key: value for key, value in analysis_payload.items() if key != "identity_sha256"
    }
    selection_unsigned = {
        key: value for key, value in selection.items() if key != "identity_sha256"
    }
    comparison = analysis_payload.get("comparison_design")
    if (
        analysis_payload.get("schema") != analyzer.ANALYSIS_SCHEMA
        or not isinstance(comparison, Mapping)
        or comparison.get("control_display_name")
        != "C-ABS"
        or comparison.get("external_temporal_abs_numeric_baseline_allowed")
        is not False
        or analysis_payload.get("protected_test_accessed") is not False
        or analysis_payload.get("protected_test_cache_opened") is not False
        or analysis_payload.get("identity_sha256") != sha256_json(analysis_unsigned)
        or selection.get("schema") != analyzer.SELECTION_SCHEMA
        or selection.get("identity_sha256") != sha256_json(selection_unsigned)
        or selection.get("development_analysis") != file_record(analysis_path)
        or selection.get("development_analysis_identity_sha256")
        != analysis_payload.get("identity_sha256")
        or selection.get("selection_used_protected_test") is not False
        or selection.get("protected_test_accessed") is not False
        or selection.get("lockbox_not_opened_by_this_tool") is not True
        or selection.get("selection_count") not in {0, 1}
        or selection.get("selection_count")
        != int(selection.get("selected_cell") is not None)
    ):
        raise WorkflowError("development gate receipt is malformed or accessed test")
    return {
        "schema": "causal-vjepa2-observed-anchor-development-receipts-v1",
        "status": "complete",
        "ainc_evaluation": dict(ainc_evaluation.summary_record),
        "abs_evaluation": dict(abs_evaluation.summary_record),
        "local_abs_evaluation_receipt": local_abs_receipt,
        "analysis": file_record(analysis_path),
        "selection": file_record(selection_path),
        "selection_count": selection["selection_count"],
        "protected_test_accessed": False,
    }


def execute_phase(args: argparse.Namespace, plan: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
    plan_path = args.study_root / "workflow_plan.json"
    if args.phase == "cache":
        args.study_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        for path in (
            args.study_root / "normalization",
            args.study_root / "training" / "calibration",
            args.study_root / "training" / "primary",
            args.study_root / "evaluation",
            args.study_root / "analysis",
            args.study_root / "wandb",
            args.study_root / "wandb-cache",
            args.study_root / "wandb-config",
        ):
            path.mkdir(parents=True, exist_ok=False)
        atomic_write_json(plan_path, plan, exclusive=True)
    else:
        prior = read_json(plan_path, "observed-anchor workflow plan")
        if prior != plan or prior.get("identity_sha256") != sha256_json(
            {key: value for key, value in prior.items() if key != "identity_sha256"}
        ):
            raise WorkflowError("current inputs differ from the immutable cache-phase plan")
    prior_phase_receipt = {
        "cache": None,
        "train": args.study_root / "cache_receipts.json",
        "evaluate": args.study_root / "training_receipts.json",
    }[args.phase]
    if prior_phase_receipt is not None and not prior_phase_receipt.is_file():
        raise WorkflowError(f"missing prior-phase receipt: {prior_phase_receipt}")
    if args.phase == "train":
        recomputed_prior = validate_cache_receipts(args)
        if read_json(prior_phase_receipt, "frozen cache receipts") != recomputed_prior:
            raise WorkflowError("cache receipts differ before matched training")
    elif args.phase == "evaluate":
        recomputed_prior = validate_training_receipts(args)
        if (
            read_json(prior_phase_receipt, "frozen matched-training receipts")
            != recomputed_prior
        ):
            raise WorkflowError("matched-training receipts differ before evaluation")
    state_path = args.study_root / "workflow_state.json"
    state: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "status": "running",
        "phase": args.phase,
        "plan_identity_sha256": plan["identity_sha256"],
        "runtime": dict(runtime),
        "completed_commands": 0,
        "protected_test_accessed": False,
    }
    atomic_write_json(state_path, state, exclusive=False)
    environment = sanitized_environment(args.study_root)
    try:
        for command in phase_commands(args):
            _run(command, environment)
            state["completed_commands"] += 1
            atomic_write_json(state_path, state, exclusive=False)
        if args.phase == "cache":
            receipt = validate_cache_receipts(args)
            receipt_path = args.study_root / "cache_receipts.json"
        elif args.phase == "train":
            validate_cache_receipts(args)
            receipt = validate_training_receipts(args)
            receipt_path = args.study_root / "training_receipts.json"
        else:
            validate_training_receipts(args)
            receipt = validate_development_receipts(args)
            receipt_path = args.study_root / "development_receipts.json"
        atomic_write_json(receipt_path, receipt, exclusive=True)
        state["status"] = f"{args.phase}_complete"
        state["receipt"] = file_record(receipt_path)
        atomic_write_json(state_path, state, exclusive=False)
    except Exception as exc:
        state["status"] = "operational_failure_no_scientific_retry_authorized"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        atomic_write_json(state_path, state, exclusive=False)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--semantic-cache-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--vjepa-source", type=Path, required=True)
    parser.add_argument("--vjepa-checkpoint", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=ainc.EXECUTION_MODES, required=True)
    parser.add_argument("--temporal-selection-record", type=Path)
    parser.add_argument("--ack-private-wandb-project", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.ack_private_wandb_project:
            raise WorkflowError("private W&B project acknowledgement is required")
        if args.execution_mode == "post-temporal-no-pass" and args.temporal_selection_record is None:
            raise WorkflowError("post-temporal mode requires the frozen no-pass record")
        if args.execution_mode == "cheap-proxy-validity" and args.temporal_selection_record is not None:
            raise WorkflowError("cheap proxy mode cannot use a temporal selection record")
        records = validate_inputs(args)
        plan = build_plan(args, records)
        if not args.execute:
            print(
                json.dumps(
                    {
                        "mode": "dry_run_no_files_created_no_commands_executed",
                        "phase": args.phase,
                        "plan_identity_sha256": plan["identity_sha256"],
                        "commands": [
                            {"argv": command, "shell": shlex.join(command)}
                            for command in phase_commands(args)
                        ],
                        "protected_test_accessed": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        runtime = validate_slurm_runtime()
        execute_phase(args, plan, runtime)
        return 0
    except (
        WorkflowError,
        temporal_workflow.WorkflowError,
        analyzer.ObservedAnchorAnalysisError,
        temporal_analysis.TemporalAnalysisError,
        temporal_analysis.evaluator.TemporalEvaluationError,
        ainc.ObservedAnchorScreenError,
        anchor.ObservedAnchorError,
        vlf.PocError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Observed-anchor workflow error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
