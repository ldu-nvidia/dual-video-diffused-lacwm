#!/usr/bin/env python3
"""Identity, readiness, registration, and dry-run plan for ACD-P0.

No command submits a job. Registration is possible only after an exact-source
test report and a three-copy B200 memory-smoke receipt both pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import invertible_two_clock_pilot as shared  # noqa: E402


BASE_COMMIT = "92fd9ed8480f12084015c043f1fd5dfcfe40205a"
PARENT_SNAPSHOT_SHA256 = (
    "de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a"
)
LEGACY_REJECTED_SNAPSHOT_SHA256 = (
    "f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21"
)
PARENT_RUN_IDENTITY_SHA256 = (
    "d79c3699f0c68dcd17321fcb0ea2846fca4d96f8c815f91dff02a19d3140787f"
)
PARENT_CANONICAL_MODEL_STATE_SHA256 = (
    "d1231b8bc13a2391a94f2ade8ff216de3fbe5e91e7242b35c39c60197fd897a0"
)
PARENT_RUNTIME_TENSOR_STATE_SHA256 = (
    "82ffa76e99574f5202831897411e4b22eb281c1e2512dc358238bef72d5a4d5a"
)
PARENT_CANONICAL_MODEL_STATE_HASH_ALGORITHM = "snapshot_model_state_receipt_v1"
PARENT_RUNTIME_TENSOR_STATE_HASH_ALGORITHM = "tensor_state_sha256_v1"
PARENT_RESOLVED_CONFIG_SHA256 = (
    "ae3ffd27146883917472b828c18568b72cfc7c6f2888fbca3eaa2e980a8ffd38"
)
PARENT_TRAINING_SOURCE_COMMIT = "656086686dae723c942a4209a9d71cdb17ed6ccc"
TRAIN_MANIFEST_SHA256 = shared.TRAIN_MANIFEST_SHA256
TRAIN_METADATA_SHA256 = shared.TRAIN_METADATA_SHA256
TRAIN_RGB_SHA256 = shared.TRAIN_RGB_SHA256
TRAIN_ACTIONS_SHA256 = shared.TRAIN_ACTIONS_SHA256
VAL_MANIFEST_SHA256 = shared.VAL_MANIFEST_SHA256
VAL_RGB_SHA256 = shared.VAL_RGB_SHA256
VAL_ACTIONS_SHA256 = shared.VAL_ACTIONS_SHA256
NFE_GRID = (1, 2, 4)
NOISE_SEEDS = (20260809, 20260810, 20260811, 20260812)
SCHEMA_VERSION = 1
KIND_REGISTRATION = "acd_p0_registration"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
APPROVED_ARTIFACT_ROOTS = (
    Path("/mnt/data1"),
    Path("/mnt/data2"),
    Path(
        "/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/"
        "users/ldu/lacwm_train"
    ),
)


class ACDPilotError(RuntimeError):
    """An exact-source, lineage, memory, or access contract changed."""


def approved_artifact_path(path: Path, label: str) -> Path:
    """Require an absolute path within one of the three approved site roots."""

    path = path.expanduser()
    if ".." in path.parts or not path.is_absolute() or not any(
        path == root or root in path.parents for root in APPROVED_ARTIFACT_ROOTS
    ):
        roots = ", ".join(str(root) for root in APPROVED_ARTIFACT_ROOTS)
        raise ACDPilotError(f"{label} must be under an approved artifact root: {roots}")
    return path


@dataclass(frozen=True)
class Arm:
    code: str
    config_name: str
    run_name: str
    arm_mode: str


ARMS = (
    Arm(
        "ACD-RF-CONT",
        "ravenhuang/wan-dit/adjacent_consistency_rf_control",
        "acd-p0-rf-control-seed1234-u000400",
        "rf_control",
    ),
    Arm(
        "ACD-CONS",
        "ravenhuang/wan-dit/adjacent_consistency_candidate",
        "acd-p0-consistency-seed1234-u000400",
        "consistency",
    ),
)
ARM_BY_CODE = {arm.code: arm for arm in ARMS}


@dataclass(frozen=True)
class Endpoint:
    code: str
    checkpoint: str
    state: str
    sampler_mode: str
    objective: str
    primary: bool
    action_source: str = "aligned"


ENDPOINTS = (
    Endpoint("PARENT_NATIVE_RF", "parent", "online", "native_rf", "rf", True),
    Endpoint("RF_ONLINE_NATIVE_RF", "ACD-RF-CONT", "online", "native_rf", "rf", True),
    Endpoint("RF_EMA_NATIVE_RF", "ACD-RF-CONT", "ema", "native_rf", "rf", True),
    Endpoint("RF_ONLINE_CONSISTENCY", "ACD-RF-CONT", "online", "consistency", "rf", False),
    Endpoint("RF_EMA_CONSISTENCY", "ACD-RF-CONT", "ema", "consistency", "rf", False),
    Endpoint("CONS_ONLINE_CONSISTENCY", "ACD-CONS", "online", "consistency", "consistency", True),
    Endpoint("CONS_EMA_CONSISTENCY", "ACD-CONS", "ema", "consistency", "consistency", True),
    Endpoint("CONS_ONLINE_NATIVE_RF", "ACD-CONS", "online", "native_rf", "consistency", False),
    Endpoint("CONS_EMA_NATIVE_RF", "ACD-CONS", "ema", "native_rf", "consistency", False),
    Endpoint(
        "PARENT_NATIVE_RF_ACTION_SHUFFLED",
        "parent",
        "online",
        "native_rf",
        "rf",
        False,
        "episode_shuffled",
    ),
    Endpoint(
        "RF_EMA_NATIVE_RF_ACTION_SHUFFLED",
        "ACD-RF-CONT",
        "ema",
        "native_rf",
        "rf",
        False,
        "episode_shuffled",
    ),
    Endpoint(
        "CONS_EMA_CONSISTENCY_ACTION_SHUFFLED",
        "ACD-CONS",
        "ema",
        "consistency",
        "consistency",
        False,
        "episode_shuffled",
    ),
)
ENDPOINT_BY_CODE = {endpoint.code: endpoint for endpoint in ENDPOINTS}


PRODUCTION_SHA256 = {
    "projects/latent_action_models/lam/dual_explicit_action_dit_model.py": (
        "a10fe3730f7bb3bacd20bd14ebbcfab2b3cf8c63a2db27783f1e8a9787b87ee6"
    ),
    "robot_wm/modeling/networks/wan_forward_model.py": (
        "4d2d49b23c7efc391379b7eb0e2437e6555b40770402d82c17dd03f777e814c1"
    ),
    "robot_wm/modeling/dual_diffusion/flow.py": (
        "8445982876f327178b733f19ae3ccf5e6dbb7a92f004afc5322cc4b2a5ff64e4"
    ),
    "robot_wm/modeling/dual_diffusion/adapters.py": (
        "f905840fe0ffe54da47279eba01029ef7c228ae8b5595624c88a66b1f317733e"
    ),
    "docs/experiments/VPM_FEW_STEP_TEACHER_DOMINANCE_AUDIT.md": (
        "54af854b964706261984797bfac5bd8c94bc97a37a1829bfadb9537c115a005d"
    ),
}
EXTERNAL_REFERENCES = (
    {
        "name": "Causal-Forcing",
        "commit": "1fc7bbc19a503c1bce80ecef08158b20e702f386",
        "file": "model/naive_consistency.py",
        "sha256": "b9a1c8612c2e735675a4c754a30c8f1ec84cb17eb1c2425659297b1a80d84a40",
    },
    {
        "name": "Flash-WAM",
        "commit": "5b8df13e9db24fb15ce42ff5ccc60a4015195960",
        "file": "distillation/step.py",
        "sha256": "4b60072f4a81760705bd7126f8fc8ec914f80767d56a857eb2cc97f6ec8d0cb2",
    },
    {
        "name": "Flash-WAM",
        "commit": "5b8df13e9db24fb15ce42ff5ccc60a4015195960",
        "file": "distillation/consistency.py",
        "sha256": "74f7c71ad1c80e4c26ceb1666f4d9ce524026f1a63f1f511db0c2dff1d2cfe26",
    },
    {
        "name": "Flash-WAM",
        "commit": "5b8df13e9db24fb15ce42ff5ccc60a4015195960",
        "file": "distillation/config.py",
        "sha256": "6e71325ae8369f649287e61803d1da7225c6a6af05fbb66204888276e91630bf",
    },
    {
        "name": "rcm",
        "commit": "ed3cb14dd936f92cdc9f9381af7369991509b41f",
        "file": "rcm/inference/wan2pt1_t2v_causal_quant_infer.py",
        "sha256": "d453fdf396d7939d05497b22abb741376c4bbf6ce427defce0aa9599aa6a0c67",
        "bound_formula_line": 1218,
        "bound_formula": "x=(1-t_next)*(x-t_cur*v_pred)+t_next*noise",
        "scope": "re-noising evidence; not a substitute for the trained Karras map",
    },
)
EXTERNAL_REPO_DIR = {
    "Causal-Forcing": "Causal-Forcing",
    "Flash-WAM": "Flash-WAM",
    "rcm": "rcm",
}


REQUIRED_IMPLEMENTATION_FILES = (
    "docs/experiments/VPM_SOLVER_MATCHED_DIRECT_DISTILLATION_PROTOCOL.md",
    "docs/experiments/VPM_SOLVER_MATCHED_DIRECT_DISTILLATION_RUNBOOK.md",
    "robot_wm/modeling/low_nfe/__init__.py",
    "robot_wm/modeling/low_nfe/adjacent_consistency.py",
    "projects/latent_action_models/lam/adjacent_consistency_model.py",
    "robot_wm/utils/adjacent_consistency_trainer.py",
    "projects/latent_action_models/train_adjacent_consistency.py",
    "projects/latent_action_models/configs/models/dual_explicit_action_dit_adjacent_consistency.yaml",
    "projects/latent_action_models/configs/experiments_0908/adjacent_consistency_common.yaml",
    "projects/latent_action_models/configs/experiments_0908/ravenhuang/wan-dit/adjacent_consistency_rf_control.yaml",
    "projects/latent_action_models/configs/experiments_0908/ravenhuang/wan-dit/adjacent_consistency_candidate.yaml",
    "tools/low_nfe_direct_baseline.py",
    "tools/adjacent_consistency_memory_smoke.py",
    "tools/adjacent_consistency_evaluate.py",
    "tools/analyze_adjacent_consistency.py",
    "tools/slurm/adjacent_consistency_memory_smoke.sbatch",
    "tools/slurm/adjacent_consistency.sbatch",
    "tools/slurm/adjacent_consistency_evaluate.sbatch",
    "tools/slurm/adjacent_consistency_workflow.py",
    "robot_wm/tests/test_adjacent_consistency.py",
    "robot_wm/tests/test_adjacent_consistency_model.py",
    "robot_wm/tests/test_adjacent_consistency_ordering.py",
    "robot_wm/tests/test_low_nfe_direct_baseline.py",
    "robot_wm/tests/test_analyze_adjacent_consistency.py",
    "robot_wm/tests/test_adjacent_consistency_workflow.py",
)
FORBIDDEN_MODIFIED_PRODUCTION_FILES = tuple(PRODUCTION_SHA256)
EXACT_SOURCE_TESTS = (
    "robot_wm/tests/test_adjacent_consistency.py",
    "robot_wm/tests/test_adjacent_consistency_model.py",
    "robot_wm/tests/test_adjacent_consistency_ordering.py",
    "robot_wm/tests/test_low_nfe_direct_baseline.py",
    "robot_wm/tests/test_analyze_adjacent_consistency.py",
    "robot_wm/tests/test_adjacent_consistency_workflow.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parent_model_state_lineage() -> dict[str, str]:
    """Return the two named, deliberately non-interchangeable parent hashes."""

    return {
        "canonical_model_state_sha256": PARENT_CANONICAL_MODEL_STATE_SHA256,
        "canonical_hash_algorithm": PARENT_CANONICAL_MODEL_STATE_HASH_ALGORITHM,
        "runtime_tensor_state_sha256": PARENT_RUNTIME_TENSOR_STATE_SHA256,
        "runtime_hash_algorithm": PARENT_RUNTIME_TENSOR_STATE_HASH_ALGORITHM,
    }


def _require_parent_model_state_lineage(
    value: Mapping[str, Any], *, label: str
) -> None:
    expected = parent_model_state_lineage()
    if any(
        value.get(key) != expected_value
        for key, expected_value in expected.items()
    ):
        raise ACDPilotError(f"{label} dual model-state lineage differs")


canonical_json = shared.canonical_json
identity_payload = shared.identity_payload
identity_valid = shared.identity_valid
sha256 = shared.sha256
file_record = shared.file_record
read_json = shared.read_json
exclusive_json = shared.exclusive_json


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ACDPilotError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def clean_source(repo: Path, expected_commit: str) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise ACDPilotError("expected commit must be a full lowercase SHA")
    repo = repo.expanduser().resolve(strict=True)
    if _git(repo, "rev-parse", "HEAD") != expected_commit:
        raise ACDPilotError("source HEAD differs from expected commit")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ACDPilotError("source worktree is not clean")
    return {
        "path": str(repo),
        "git_commit": expected_commit,
        "git_tree_sha": _git(repo, "rev-parse", "HEAD^{tree}"),
        "clean": True,
    }


def resource_plan() -> dict[str, Any]:
    return {
        "memory_smoke_allocations": 1,
        "memory_smoke_b200_each": 1,
        "memory_smoke_time_limit_hours_each": 1,
        "training_allocations": 2,
        "training_b200_each": 8,
        "training_time_limit_hours_each": 6,
        "evaluation_allocations": 1,
        "evaluation_b200_each": 8,
        "evaluation_time_limit_hours_each": 2,
        "maximum_reserved_b200_hours": 113,
        "expected_parallel_wall_hours": "8-10",
        "expected_serial_wall_hours": "14-16",
        "artifact_ceiling_bytes": 100 * (1 << 30),
        "expected_persistent_artifact_bytes": 40 * (1 << 30),
        "requeue": False,
        "wandb": False,
    }


def fixed_protocol() -> dict[str, Any]:
    return {
        "protocol_version": "acd-p0-v1",
        "updates_per_arm": 400,
        "world_size": 8,
        "local_batch": 1,
        "global_batch": 8,
        "optimizer": "AdamW(lr=5e-6,betas=(0.9,0.95))",
        "warmup_updates": 40,
        "final_lr": 1e-6,
        "stride": 500,
        "sigma_data": 0.5,
        "pseudo_huber_c": 0.001,
        "ema_decay": 0.995,
        "training_wan_calls": {"teacher": 1, "online": 1, "ema_target": 1},
        "nfe_grid": list(NFE_GRID),
        "noise_seeds": list(NOISE_SEEDS),
        "endpoints": [asdict(endpoint) for endpoint in ENDPOINTS],
        "protected_test_accessed": False,
    }


def arm_identity(registration: Mapping[str, Any], arm: Arm) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "registration_core_identity_sha256": registration[
                    "registration_core_identity_sha256"
                ],
                "arm": asdict(arm),
                "fixed_protocol": registration["fixed_protocol"],
                "resolved_config_semantic_sha256": registration[
                    "arm_resolved_configs"
                ][arm.code]["semantic_sha256"],
            }
        )
    ).hexdigest()


def semantic_config_sha256(value: Mapping[str, Any]) -> str:
    """Hash every resolved Hydra job field, independent of YAML formatting."""

    return hashlib.sha256(canonical_json(value)).hexdigest()


def _parse_resolved_config(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(content)
    except Exception as exc:
        raise ACDPilotError(f"invalid resolved Hydra config: {label}") from exc
    if not isinstance(value, dict):
        raise ACDPilotError(f"resolved Hydra config is not a mapping: {label}")
    # canonical_json also rejects NaN/Infinity and non-JSON types.
    canonical_json(value)
    return value


def _compose_resolved_arm_config(
    *,
    python: Path,
    source_repo: Path,
    arm: Arm,
    run_dir: Path,
    environment: Mapping[str, str],
) -> tuple[bytes, dict[str, Any]]:
    """Ask the registered runtime/Hydra to print the exact resolved job config."""

    command = [
        str(python),
        str(
            source_repo
            / "projects/latent_action_models/train_adjacent_consistency.py"
        ),
        "--cfg",
        "job",
        "--resolve",
        f"+experiments_0908={arm.config_name}",
        f"hydra.run.dir={run_dir}",
        f"trainer.config.saving.save_path={run_dir / 'snapshot.pt'}",
        f"trainer.config.visualization.viz_path={run_dir / 'visualization'}",
    ]
    child_environment = os.environ.copy()
    child_environment.update(environment)
    python_path = [
        str(source_repo),
        str(source_repo / "projects/latent_action_models"),
        environment["VIDEOX_HOME"],
    ]
    if child_environment.get("PYTHONPATH"):
        python_path.append(child_environment["PYTHONPATH"])
    child_environment["PYTHONPATH"] = os.pathsep.join(python_path)
    result = subprocess.run(
        command,
        cwd=source_repo,
        env=child_environment,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise ACDPilotError(
            f"Hydra resolution failed for {arm.code}: "
            f"{result.stderr.decode(errors='replace')[-2000:]}"
        )
    payload = _parse_resolved_config(result.stdout, label=arm.code)
    return result.stdout, payload


def _expected_bytes_record(path: Path, content: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _exclusive_bytes(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _bind_training_array(
    record: Mapping[str, Any],
    *,
    logical: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Hash actual train bytes and bind the immutable NumPy schema."""

    import numpy as np

    path = Path(str(record.get("path", "")))
    observed = file_record(path)
    if observed["sha256"] != expected_sha256:
        raise ACDPilotError(f"actual training {logical} bytes differ")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_shape, expected_dtype = {
        "rgb": ((512, 13, 3, 180, 960), np.dtype("float16")),
        "actions": ((512, 13, 5, 23), np.dtype("float32")),
    }[logical]
    if tuple(array.shape) != expected_shape or array.dtype != expected_dtype:
        raise ACDPilotError(f"training {logical} array schema differs")
    del array
    return {
        **observed,
        "shape": list(expected_shape),
        "dtype": str(expected_dtype),
        "actual_bytes_hashed_at_registration": True,
        "rehash_required_before_each_training_arm": True,
    }


def _verify_external_references(repos_root: Path) -> list[dict[str, Any]]:
    repos_root = repos_root.expanduser().resolve(strict=True)
    records = []
    observed_commits: dict[str, str] = {}
    for reference in EXTERNAL_REFERENCES:
        name = str(reference["name"])
        repo = repos_root / EXTERNAL_REPO_DIR[name]
        commit = observed_commits.setdefault(name, _git(repo, "rev-parse", "HEAD"))
        if commit != reference["commit"]:
            raise ACDPilotError(f"external reference commit changed: {name}")
        path = repo / str(reference["file"])
        record = file_record(path)
        if record["sha256"] != reference["sha256"]:
            raise ACDPilotError(f"external reference file changed: {name}/{reference['file']}")
        records.append({**reference, "record": record})
    return records


def _remote_receipt(repo: Path, commit: str, remote: str) -> dict[str, Any]:
    refs = sorted(
        line.split("\t", 1)[1]
        for line in _git(repo, "ls-remote", remote).splitlines()
        if line.startswith(commit + "\t") and "\t" in line
    )
    if not refs:
        raise ACDPilotError("exact source commit is not reachable from remote")
    return {
        "remote": remote,
        "url": _git(repo, "remote", "get-url", remote),
        "exact_commit": commit,
        "refs": refs,
        "reachable": True,
    }


def _validate_test_report(path: Path, commit: str) -> dict[str, Any]:
    path = approved_artifact_path(path.expanduser(), "exact-source test receipt")
    path = approved_artifact_path(
        path.resolve(strict=True), "resolved exact-source test receipt"
    )
    report = read_json(path, "ACD exact-source test report")
    if (
        not identity_valid(report)
        or report.get("kind") != "acd_p0_exact_source_test_report"
        or report.get("source_commit") != commit
        or report.get("passed") is not True
        or report.get("jobs_submitted") != 0
        or report.get("wandb_writes") != 0
        or report.get("outcome_rows_opened") != 0
        or report.get("protected_test_accessed") is not False
    ):
        raise ACDPilotError("test report does not seal this exact source")
    return report


def command_test_report(args: argparse.Namespace) -> int:
    source = clean_source(args.source_repo, args.expected_commit)
    output = approved_artifact_path(
        args.output.expanduser(), "exact-source test output"
    )
    python = args.python.expanduser()
    if (
        not output.is_absolute()
        or not output.parent.is_dir()
        or output.exists()
        or output.is_symlink()
        or not python.is_absolute()
        or not os.access(python, os.X_OK)
    ):
        raise ACDPilotError("exact-source test output/runtime is invalid")
    output = approved_artifact_path(
        output.parent.resolve(strict=True) / output.name,
        "resolved exact-source test output",
    )
    command = [
        str(python),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *EXACT_SOURCE_TESTS,
    ]
    environment = os.environ.copy()
    python_path = [
        source["path"],
        str(Path(source["path"]) / "projects/latent_action_models"),
    ]
    if args.videox_home is not None:
        videox = args.videox_home.expanduser().resolve(strict=True)
        if not videox.is_dir():
            raise ACDPilotError("VideoX runtime path is unavailable")
        python_path.append(str(videox))
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "WANDB_SILENT": "true",
        }
    )
    environment.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        command,
        cwd=Path(source["path"]),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    post_source = None
    post_source_error = None
    try:
        post_source = clean_source(args.source_repo, args.expected_commit)
    except BaseException as exc:
        post_source_error = f"{type(exc).__name__}: {exc}"
    passed = (
        result.returncode == 0
        and post_source == source
        and post_source_error is None
    )
    payload = identity_payload(
        {
            "kind": "acd_p0_exact_source_test_report",
            "created_at_utc": now(),
            "source_commit": source["git_commit"],
            "source_tree_sha": source["git_tree_sha"],
            "python": str(python),
            "python_target": file_record(python.resolve(strict=True)),
            "command": command,
            "test_files": list(EXACT_SOURCE_TESTS),
            "returncode": result.returncode,
            "passed": passed,
            "post_test_source": post_source,
            "post_test_source_error": post_source_error,
            "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
            "jobs_submitted": 0,
            "wandb_writes": 0,
            "outcome_rows_opened": 0,
            "protected_test_accessed": False,
        }
    )
    exclusive_json(output, payload)
    print(str(output))
    if not passed:
        raise ACDPilotError("exact-source tests failed; failure receipt was written")
    return 0


def _validate_memory_receipt(path: Path, commit: str) -> dict[str, Any]:
    path = approved_artifact_path(path.expanduser(), "memory-smoke receipt")
    path = approved_artifact_path(
        path.resolve(strict=True), "resolved memory-smoke receipt"
    )
    receipt = read_json(path, "ACD memory smoke receipt")
    parent_state = receipt.get("parent_model_state_hash_receipt")
    loaded_states = receipt.get("strict_loaded_runtime_tensor_state_sha256")
    synthetic_support = receipt.get("synthetic_support")
    if (
        not identity_valid(receipt)
        or receipt.get("kind") != "acd_p0_memory_smoke_receipt"
        or receipt.get("schema_version") != 2
        or receipt.get("source_commit") != commit
        or receipt.get("parent_snapshot_sha256") != PARENT_SNAPSHOT_SHA256
        or not isinstance(parent_state, Mapping)
        or any(
            parent_state.get(key) != value
            for key, value in parent_model_state_lineage().items()
        )
        or not isinstance(parent_state.get("state_tensors"), int)
        or parent_state.get("state_tensors", 0) <= 0
        or not isinstance(parent_state.get("state_values"), int)
        or parent_state.get("state_values", 0) <= 0
        or loaded_states
        != {
            "student": PARENT_RUNTIME_TENSOR_STATE_SHA256,
            "teacher": PARENT_RUNTIME_TENSOR_STATE_SHA256,
            "ema_target": PARENT_RUNTIME_TENSOR_STATE_SHA256,
        }
        or receipt.get("status") != "PASS"
        or receipt.get("model_copies") != 3
        or receipt.get("synthetic_full_geometry") != [1, 13, 3, 180, 960]
        or not isinstance(synthetic_support, Mapping)
        or synthetic_support.get("fixture_version") != "acd-p0-full-support-v2"
        or synthetic_support.get("rgb_shape") != [1, 13, 3, 180, 960]
        or synthetic_support.get("rgb_dtype") != "torch.float32"
        or not isinstance(synthetic_support.get("rgb_min"), (int, float))
        or not isinstance(synthetic_support.get("rgb_max"), (int, float))
        or not math.isfinite(float(synthetic_support.get("rgb_min", float("nan"))))
        or not math.isfinite(float(synthetic_support.get("rgb_max", float("nan"))))
        or float(synthetic_support.get("rgb_min", -2.0)) < -1.0
        or float(synthetic_support.get("rgb_max", 2.0)) > 1.0
        or float(synthetic_support.get("rgb_min", 1.0))
        > float(synthetic_support.get("rgb_max", -1.0))
        or not isinstance(
            synthetic_support.get("minimum_view_std"), (int, float)
        )
        or not math.isfinite(
            float(synthetic_support.get("minimum_view_std", float("nan")))
        )
        or float(synthetic_support.get("minimum_view_std", 0.0)) <= 1e-3
        or synthetic_support.get("valid_views") != 3
        or synthetic_support.get("temporal_mask_shape") != [1, 13]
        or synthetic_support.get("temporal_mask_dtype") != "torch.bool"
        or synthetic_support.get("temporal_valid_frames") != 13
        or synthetic_support.get("history_valid_frames") != 5
        or synthetic_support.get("future_valid_frames") != 8
        or synthetic_support.get("latent_loss_mask_shape") != [1, 1, 4, 1, 120]
        or synthetic_support.get("history_latent_tokens") != 2
        or synthetic_support.get("future_latent_tokens") != 2
        or synthetic_support.get("expanded_future_support_per_sample")
        != [92160]
        or synthetic_support.get("production_build_loss_mask_exercised") is not True
        or synthetic_support.get("expanded_mask_exercised") is not True
        or synthetic_support.get("dataset_accessed") is not False
        or receipt.get("teacher_calls") != 1
        or receipt.get("online_calls") != 1
        or receipt.get("ema_target_calls") != 1
        or receipt.get("backward_completed") is not True
        or receipt.get("optimizer_step_completed") is not True
        or receipt.get("ema_step_completed") is not True
        or receipt.get("peak_reserved_fraction", 1.0) > 0.90
        or receipt.get("headroom_bytes", 0) < 16 * (1 << 30)
        or receipt.get("training_data_opened") is not False
        or receipt.get("outcomes_opened") != 0
        or receipt.get("wandb_writes") != 0
    ):
        raise ACDPilotError("three-copy memory smoke receipt is absent or insufficient")
    return receipt


def command_readiness(args: argparse.Namespace) -> int:
    source = clean_source(args.source_repo, args.expected_commit)
    source_repo = Path(source["path"])
    remote = _remote_receipt(source_repo, args.expected_commit, args.remote)
    missing = [
        relative
        for relative in REQUIRED_IMPLEMENTATION_FILES
        if not (source_repo / relative).is_file()
    ]
    if missing:
        raise ACDPilotError(f"required implementation files missing: {missing}")
    changed = set(
        filter(
            None,
            _git(
                source_repo,
                "diff",
                "--name-only",
                f"{BASE_COMMIT}..{args.expected_commit}",
            ).splitlines(),
        )
    )
    forbidden = sorted(changed & set(FORBIDDEN_MODIFIED_PRODUCTION_FILES))
    if forbidden:
        raise ACDPilotError(f"production source was modified: {forbidden}")
    for relative, expected in PRODUCTION_SHA256.items():
        if sha256(source_repo / relative) != expected:
            raise ACDPilotError(f"production/reference source changed: {relative}")
    required = {
        relative: file_record(source_repo / relative)
        for relative in REQUIRED_IMPLEMENTATION_FILES
    }
    external = _verify_external_references(args.external_repos_root)
    test_report = (
        None
        if args.test_report is None
        else _validate_test_report(args.test_report, args.expected_commit)
    )
    memory = (
        None
        if args.memory_smoke_receipt is None
        else _validate_memory_receipt(args.memory_smoke_receipt, args.expected_commit)
    )
    status = (
        "READY_FOR_REGISTRATION"
        if test_report is not None and memory is not None
        else "READY_SOURCE_MEMORY_PREFLIGHT_REQUIRED"
    )
    payload = identity_payload(
        {
            "kind": "acd_p0_readiness_seal",
            "created_at_utc": now(),
            "status": status,
            "source": source,
            "remote_reachability": remote,
            "base_commit": BASE_COMMIT,
            "changed_files": sorted(changed),
            "required_files": required,
            "forbidden_production_files_modified": forbidden,
            "production_source_parity_sha256": PRODUCTION_SHA256,
            "external_references": external,
            "parent": {
                "snapshot_sha256": PARENT_SNAPSHOT_SHA256,
                "legacy_rejected_snapshot_sha256": LEGACY_REJECTED_SNAPSHOT_SHA256,
                "run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                **parent_model_state_lineage(),
                "resolved_config_sha256": PARENT_RESOLVED_CONFIG_SHA256,
                "training_source_commit": PARENT_TRAINING_SOURCE_COMMIT,
            },
            "fixed_protocol": fixed_protocol(),
            "resource_plan": resource_plan(),
            "test_report": test_report,
            "test_report_record": (
                None if args.test_report is None else file_record(args.test_report)
            ),
            "memory_smoke": memory,
            "memory_smoke_record": (
                None
                if args.memory_smoke_receipt is None
                else file_record(args.memory_smoke_receipt)
            ),
            "registration_created": False,
            "jobs_submitted": 0,
            "wandb_writes": 0,
            "data_rows_opened": 0,
            "outcome_rows_opened": 0,
            "protected_test_accessed": False,
        }
    )
    if args.output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        output = approved_artifact_path(args.output.expanduser(), "readiness output")
        if not output.is_absolute() or not output.parent.is_dir():
            raise ACDPilotError("readiness output requires an existing absolute parent")
        output = approved_artifact_path(
            output.parent.resolve(strict=True) / output.name,
            "resolved readiness output",
        )
        if output.exists() or output.is_symlink():
            raise ACDPilotError("readiness output must be fresh")
        exclusive_json(output, payload)
        print(str(output))
    return 0


def _validate_registration_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = clean_source(args.source_repo, args.expected_commit)
    readiness_path = approved_artifact_path(
        args.readiness_seal.expanduser(), "readiness receipt"
    )
    readiness_path = approved_artifact_path(
        readiness_path.resolve(strict=True), "resolved readiness receipt"
    )
    readiness = read_json(readiness_path, "ACD readiness")
    if (
        not identity_valid(readiness)
        or readiness.get("kind") != "acd_p0_readiness_seal"
        or readiness.get("status") != "READY_FOR_REGISTRATION"
        or readiness.get("source", {}).get("git_commit") != args.expected_commit
        or readiness.get("jobs_submitted") != 0
        or readiness.get("outcome_rows_opened") != 0
    ):
        raise ACDPilotError("exact-source readiness gate is not open")
    parent = readiness.get("parent")
    if not isinstance(parent, Mapping):
        raise ACDPilotError("readiness parent lineage is absent")
    _require_parent_model_state_lineage(parent, label="readiness parent")
    memory = _validate_memory_receipt(args.memory_smoke_receipt, args.expected_commit)
    if readiness.get("memory_smoke", {}).get("identity_sha256") != memory.get(
        "identity_sha256"
    ):
        raise ACDPilotError("memory receipt differs from readiness seal")
    return source, readiness, memory


def command_register(args: argparse.Namespace) -> int:
    import torch

    from robot_wm.modeling.low_nfe.adjacent_consistency import (
        CANONICAL_MODEL_STATE_HASH_ALGORITHM,
        RUNTIME_TENSOR_STATE_HASH_ALGORITHM,
        require_model_state_hashes,
    )

    source, readiness, memory = _validate_registration_inputs(args)
    output = args.output.expanduser()
    if not output.is_absolute():
        raise ACDPilotError("output must be an absolute approved large-artifact path")
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    approved_artifact_path(output, "output")
    if output.exists() or output.is_symlink():
        raise ACDPilotError("registered output root must be fresh")
    if shutil.disk_usage(parent).free < 100 * (1 << 30):
        raise ACDPilotError("less than 100 GiB is free at output parent")
    snapshot_record = file_record(args.parent_snapshot)
    config_record = file_record(args.parent_resolved_config)
    if snapshot_record["sha256"] != PARENT_SNAPSHOT_SHA256:
        raise ACDPilotError("parent snapshot is not faithful de65")
    if config_record["sha256"] != PARENT_RESOLVED_CONFIG_SHA256:
        raise ACDPilotError("parent resolved config differs")
    snapshot = torch.load(
        snapshot_record["path"], map_location="cpu", weights_only=True, mmap=True
    )
    parent_state = snapshot.get("model")
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("run_identity_sha256") != PARENT_RUN_IDENTITY_SHA256
        or snapshot.get("_start_iter") != 1000
        or not isinstance(parent_state, Mapping)
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
    ):
        raise ACDPilotError("parent snapshot metadata differs")
    if (
        CANONICAL_MODEL_STATE_HASH_ALGORITHM
        != PARENT_CANONICAL_MODEL_STATE_HASH_ALGORITHM
        or RUNTIME_TENSOR_STATE_HASH_ALGORITHM
        != PARENT_RUNTIME_TENSOR_STATE_HASH_ALGORITHM
    ):
        raise ACDPilotError("source model-state hash algorithm identity differs")
    try:
        parent_state_hash_receipt = require_model_state_hashes(
            parent_state,
            expected_canonical_sha256=PARENT_CANONICAL_MODEL_STATE_SHA256,
            expected_runtime_sha256=PARENT_RUNTIME_TENSOR_STATE_SHA256,
            label="registered raw parent model state",
        )
    except Exception as exc:
        raise ACDPilotError("registered raw parent model-state hashes differ") from exc
    del snapshot, parent_state
    train_rows = shared._manifest(
        args.train_manifest, "train", 512, TRAIN_MANIFEST_SHA256
    )
    val_rows = shared._manifest(args.val_manifest, "val", 64, VAL_MANIFEST_SHA256)
    train_meta, train_arrays = shared._metadata(
        args.train_cache_metadata,
        split="train",
        count=512,
        expected_metadata_sha=TRAIN_METADATA_SHA256,
        expected_rgb_sha=TRAIN_RGB_SHA256,
        expected_actions_sha=TRAIN_ACTIONS_SHA256,
    )
    val_meta, val_arrays = shared._metadata(
        args.val_cache_metadata,
        split="val",
        count=64,
        expected_metadata_sha=None,
        expected_rgb_sha=VAL_RGB_SHA256,
        expected_actions_sha=VAL_ACTIONS_SHA256,
    )
    # Unlike the target-blind validation split, train inputs are admissible at
    # registration and must be bound by their actual bytes, not by metadata
    # strings supplied by the cache producer.
    train_arrays = {
        "rgb": _bind_training_array(
            train_arrays["rgb"], logical="rgb", expected_sha256=TRAIN_RGB_SHA256
        ),
        "actions": _bind_training_array(
            train_arrays["actions"],
            logical="actions",
            expected_sha256=TRAIN_ACTIONS_SHA256,
        ),
    }
    if (
        {row["clip_id"] for row in train_rows}
        & {row["clip_id"] for row in val_rows}
        or {row["episode_dir"] for row in train_rows}
        & {row["episode_dir"] for row in val_rows}
    ):
        raise ACDPilotError("training and development manifests overlap")
    lpips_linear = file_record(args.lpips_linear_weight)
    alexnet = file_record(args.alexnet_weight)
    if lpips_linear["sha256"] != (
        "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
    ) or alexnet["sha256"] != (
        "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
    ):
        raise ACDPilotError("pinned LPIPS/AlexNet weights differ")
    python = args.python.expanduser()
    if not python.is_absolute():
        raise ACDPilotError("registered Python path must be absolute")
    wan_dir = args.wan_dir.expanduser().resolve(strict=True)
    videox_home = args.videox_home.expanduser().resolve(strict=True)
    if not os.access(python, os.X_OK) or not wan_dir.is_dir() or not videox_home.is_dir():
        raise ACDPilotError("registered runtime path is unavailable")
    compose_environment = {
        "ACD_P0_TRAIN_CLIP_MANIFEST": str(args.train_manifest.resolve(strict=True)),
        "ACD_P0_TRAIN_CACHE_METADATA": str(
            args.train_cache_metadata.resolve(strict=True)
        ),
        "ACD_P0_PARENT_SNAPSHOT": snapshot_record["path"],
        "ACD_P0_PARENT_RESOLVED_CONFIG": config_record["path"],
        "ACD_P0_RUN_ROOT": str(output / "training"),
        "WAN_DIR": str(wan_dir),
        "VIDEOX_HOME": str(videox_home),
    }
    resolved_contents: dict[str, bytes] = {}
    resolved_configs: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        run_dir = output / "training" / arm.run_name
        content, payload = _compose_resolved_arm_config(
            python=python,
            source_repo=Path(source["path"]),
            arm=arm,
            run_dir=run_dir,
            environment=compose_environment,
        )
        if (
            payload.get("name") != arm.run_name
            or payload.get("model", {})
            .get("adjacent_consistency", {})
            .get("arm_mode")
            != arm.arm_mode
            or payload.get("trainer", {})
            .get("config", {})
            .get("saving", {})
            .get("save_path")
            != str(run_dir / "snapshot.pt")
        ):
            raise ACDPilotError(f"prospective resolved config differs: {arm.code}")
        config_path = output / "registered_resolved_configs" / f"{arm.code}.yaml"
        resolved_contents[arm.code] = content
        resolved_configs[arm.code] = {
            "yaml": _expected_bytes_record(config_path, content),
            "semantic_sha256": semantic_config_sha256(payload),
            "fully_resolved": True,
            "normalization": "none; registered absolute output paths are bound",
        }
    registration = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_REGISTRATION,
            "created_at_utc": now(),
            "output_root": str(output),
            "tool_repository": source,
            "readiness": file_record(args.readiness_seal),
            "readiness_payload_identity_sha256": readiness["identity_sha256"],
            "memory_smoke": file_record(args.memory_smoke_receipt),
            "memory_smoke_payload": memory,
            "fixed_protocol": fixed_protocol(),
            "arm_resolved_configs": resolved_configs,
            "protocol_files": {
                relative: file_record(Path(source["path"]) / relative)
                for relative in REQUIRED_IMPLEMENTATION_FILES
            },
            "parent": {
                "snapshot": snapshot_record,
                "resolved_config": config_record,
                "run_identity_sha256": PARENT_RUN_IDENTITY_SHA256,
                **parent_model_state_lineage(),
                "model_state_hash_receipt": parent_state_hash_receipt,
                "training_source_commit": PARENT_TRAINING_SOURCE_COMMIT,
                "legacy_rejected_snapshot_sha256": LEGACY_REJECTED_SNAPSHOT_SHA256,
            },
            "training": {
                "manifest": file_record(args.train_manifest),
                "cache_metadata": train_meta,
                "arrays": train_arrays,
                "rows": 512,
            },
            "validation": {
                "manifest": file_record(args.val_manifest),
                "cache_metadata": val_meta,
                "arrays": val_arrays,
                "rows": 64,
                "unopened_until_both_training_completions": True,
            },
            "validation_descriptors": val_rows,
            "runtime": {
                "python": str(python),
                "python_target": file_record(python.resolve(strict=True)),
                "wan_dir": str(wan_dir),
                "videox_home": str(videox_home),
            },
            "metrics": {
                "lpips_linear_weight": lpips_linear,
                "alexnet_weight": alexnet,
                "lpips_camera_preprocess": (
                    "split three 180x320 views; bilinear resize each to 64x112; "
                    "mean LPIPS-Alex over views and eight future frames"
                ),
            },
            "resource_plan": resource_plan(),
            "wandb": {"enabled": False, "writes": 0},
            "registration_created_before_training": True,
            "jobs_submitted": 0,
            "data_rows_opened_before_registration": 0,
            "outcome_rows_opened": 0,
            "protected_test_accessed": False,
        }
    )
    registration["registration_core_identity_sha256"] = registration[
        "identity_sha256"
    ]
    registration["arm_run_identity_sha256"] = {
        arm.code: arm_identity(registration, arm) for arm in ARMS
    }
    registration.pop("identity_sha256")
    registration = identity_payload(registration)
    output.mkdir(mode=0o700)
    config_directory = output / "registered_resolved_configs"
    config_directory.mkdir(mode=0o700)
    for arm in ARMS:
        path = Path(registration["arm_resolved_configs"][arm.code]["yaml"]["path"])
        _exclusive_bytes(path, resolved_contents[arm.code])
        if file_record(path) != registration["arm_resolved_configs"][arm.code]["yaml"]:
            raise ACDPilotError(f"persisted resolved config differs: {arm.code}")
    exclusive_json(output / "registration.json", registration)
    print(str(output / "registration.json"))
    return 0


def validate_registration(path: Path) -> dict[str, Any]:
    value = read_json(path.resolve(strict=True), "ACD registration")
    core = dict(value)
    core.pop("identity_sha256", None)
    core.pop("registration_core_identity_sha256", None)
    core.pop("arm_run_identity_sha256", None)
    expected_core_identity = identity_payload(core)["identity_sha256"]
    if (
        not identity_valid(value)
        or value.get("kind") != KIND_REGISTRATION
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("fixed_protocol") != fixed_protocol()
        or value.get("wandb") != {"enabled": False, "writes": 0}
        or value.get("jobs_submitted") != 0
        or value.get("protected_test_accessed") is not False
        or value.get("registration_core_identity_sha256")
        != expected_core_identity
    ):
        raise ACDPilotError("registration identity/protocol differs")
    parent = value.get("parent")
    if not isinstance(parent, Mapping):
        raise ACDPilotError("registration parent lineage is absent")
    _require_parent_model_state_lineage(parent, label="registration parent")
    parent_receipt = parent.get("model_state_hash_receipt")
    if (
        not isinstance(parent_receipt, Mapping)
        or any(
            parent_receipt.get(key) != expected
            for key, expected in parent_model_state_lineage().items()
        )
        or not isinstance(parent_receipt.get("state_tensors"), int)
        or parent_receipt.get("state_tensors", 0) <= 0
        or not isinstance(parent_receipt.get("state_values"), int)
        or parent_receipt.get("state_values", 0) <= 0
    ):
        raise ACDPilotError("registration parent dual-hash receipt differs")
    python_path = Path(str(value.get("runtime", {}).get("python", "")))
    if (
        not python_path.is_absolute()
        or not os.access(python_path, os.X_OK)
        or file_record(python_path.resolve(strict=True))
        != value.get("runtime", {}).get("python_target")
    ):
        raise ACDPilotError("registered Python runtime differs")
    for arm in ARMS:
        config = value.get("arm_resolved_configs", {}).get(arm.code, {})
        yaml_record = config.get("yaml", {})
        yaml_path = Path(str(yaml_record.get("path", "")))
        if file_record(yaml_path) != yaml_record:
            raise ACDPilotError(f"registered resolved config bytes differ: {arm.code}")
        payload = _parse_resolved_config(yaml_path.read_bytes(), label=arm.code)
        if config.get("semantic_sha256") != semantic_config_sha256(payload):
            raise ACDPilotError(f"registered resolved config semantics differ: {arm.code}")
        if value.get("arm_run_identity_sha256", {}).get(arm.code) != arm_identity(
            value, arm
        ):
            raise ACDPilotError(f"arm identity differs: {arm.code}")
    return value


def command_plan(args: argparse.Namespace) -> int:
    if args.registration is None:
        print(json.dumps(resource_plan(), indent=2, sort_keys=True))
        return 0
    registration = validate_registration(args.registration)
    root = Path(registration["output_root"])
    source = Path(registration["tool_repository"]["path"])
    python = registration["runtime"]["python"]
    commands = []
    for arm in ARMS:
        run = root / "training" / arm.run_name
        commands.append(
            {
                "phase": "train",
                "arm": arm.code,
                "command": [
                    python,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=8",
                    str(source / "projects/latent_action_models/train_adjacent_consistency.py"),
                    f"+experiments_0908={arm.config_name}",
                    f"hydra.run.dir={run}",
                    f"trainer.config.saving.save_path={run / 'snapshot.pt'}",
                    f"trainer.config.visualization.viz_path={run / 'visualization'}",
                ],
            }
        )
    commands.append(
        {
            "phase": "evaluate",
            "command": [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                str(source / "tools/adjacent_consistency_evaluate.py"),
                "--registration",
                str(args.registration),
                "--output",
                str(root / "evaluation"),
            ],
        }
    )
    print(
        json.dumps(
            {
                "kind": "acd_p0_dry_run_plan",
                "mode": "no_commands_executed_no_files_created",
                "source_commit": registration["tool_repository"]["git_commit"],
                "commands": commands,
                "resource_plan": resource_plan(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    test_report = commands.add_parser("test-report")
    test_report.add_argument("--source-repo", type=Path, required=True)
    test_report.add_argument("--expected-commit", required=True)
    test_report.add_argument("--python", type=Path, required=True)
    test_report.add_argument("--videox-home", type=Path)
    test_report.add_argument("--output", type=Path, required=True)
    test_report.set_defaults(func=command_test_report)
    readiness = commands.add_parser("readiness")
    readiness.add_argument("--source-repo", type=Path, required=True)
    readiness.add_argument("--expected-commit", required=True)
    readiness.add_argument("--external-repos-root", type=Path, required=True)
    readiness.add_argument("--test-report", type=Path)
    readiness.add_argument("--memory-smoke-receipt", type=Path)
    readiness.add_argument("--remote", default="github")
    readiness.add_argument("--output", type=Path)
    readiness.set_defaults(func=command_readiness)
    register = commands.add_parser("register")
    register.add_argument("--output", type=Path, required=True)
    register.add_argument("--source-repo", type=Path, required=True)
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--readiness-seal", type=Path, required=True)
    register.add_argument("--memory-smoke-receipt", type=Path, required=True)
    register.add_argument("--parent-snapshot", type=Path, required=True)
    register.add_argument("--parent-resolved-config", type=Path, required=True)
    register.add_argument("--train-manifest", type=Path, required=True)
    register.add_argument("--train-cache-metadata", type=Path, required=True)
    register.add_argument("--val-manifest", type=Path, required=True)
    register.add_argument("--val-cache-metadata", type=Path, required=True)
    register.add_argument("--python", type=Path, required=True)
    register.add_argument("--wan-dir", type=Path, required=True)
    register.add_argument("--videox-home", type=Path, required=True)
    register.add_argument("--lpips-linear-weight", type=Path, required=True)
    register.add_argument("--alexnet-weight", type=Path, required=True)
    register.set_defaults(func=command_register)
    plan = commands.add_parser("plan")
    plan.add_argument("--registration", type=Path)
    plan.set_defaults(func=command_plan)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
