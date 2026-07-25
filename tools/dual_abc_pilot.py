#!/usr/bin/env python3
"""Safety and provenance helpers for the matched ABC ZTF conditioning pilot.

The pilot uses the LACWM clock convention throughout: ``sigma=1`` is Gaussian
noise and ``sigma=0`` is clean data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}$")
EXPECTED_ENTITY = "zijiandu"
EXPECTED_PROJECT = "dual-video-diffusion-private"
EXPECTED_CHECKPOINT_SHA256 = (
    "5c132cb5ed6df7b840eef075d13140a73b1c6d5b3d1be4299b01e8365866224b"
)
ARMS = {
    "no-ztf-condition": {
        "condition_on_tf": False,
        "config": (
            "ravenhuang/wan-dit/dual_abc_no_ztf_condition.yaml"
        ),
    },
    "with-ztf-condition": {
        "condition_on_tf": True,
        "config": (
            "ravenhuang/wan-dit/dual_abc_with_ztf_condition.yaml"
        ),
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_regular_file(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a non-symlink regular file: {path}")
    if info.st_size <= 0:
        raise ValueError(f"{label} is empty: {path}")
    return path.resolve(strict=True)


def _canonical_directory(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be a non-symlink directory: {path}")
    return path.resolve(strict=True)


def _python_executable(value: str) -> Path:
    """Preserve the venv entrypoint spelling while validating its target.

    Resolving ``venv/bin/python`` before execution can discard the environment's
    site-packages, so a symlink is expected and intentional for this one path.
    """

    path = Path(value).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Python executable is missing or not executable: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Python symlink target is not a regular file: {resolved}")
    return path


def _validated_sha256(value: str, label: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _validated_commit(value: str) -> str:
    if not COMMIT_RE.fullmatch(value):
        raise ValueError("git commit must be 40 lowercase hexadecimal characters")
    return value


def _validated_id(value: str, label: str) -> str:
    if not SAFE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{label} must begin with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or dash (maximum 127 characters)"
        )
    return value


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _assert_clean_commit(repo_root: Path, expected_commit: str) -> None:
    actual_commit = _git(repo_root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"repository HEAD changed: {actual_commit} != {expected_commit}"
        )
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "repository is dirty; the pilot requires an exact committed source tree: "
            + status.replace("\n", "; ")
        )


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
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


def _exclusive_bytes(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _identity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return {
        **payload,
        "identity_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _wandb_private_project(entity: str, project: str) -> dict[str, str]:
    if entity != EXPECTED_ENTITY or project != EXPECTED_PROJECT:
        raise ValueError(
            f"this pilot is locked to {EXPECTED_ENTITY}/{EXPECTED_PROJECT}"
        )

    import wandb
    from wandb_gql import gql

    api = wandb.Api(timeout=30)
    query = gql(
        """
        query PilotProjectAccess($entity: String!, $project: String!) {
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
        variable_values={"entity": entity, "project": project},
    )
    project_result = result.get("project")
    viewer = result.get("viewer")
    if not isinstance(project_result, dict):
        raise RuntimeError(f"W&B project does not exist: {entity}/{project}")
    if project_result.get("entityName") != entity:
        raise RuntimeError(
            f"W&B project entity mismatch: {project_result.get('entityName')!r}"
        )
    if project_result.get("name") != project:
        raise RuntimeError(
            f"W&B project name mismatch: {project_result.get('name')!r}"
        )
    if str(project_result.get("access", "")).upper() != "PRIVATE":
        raise RuntimeError(
            f"W&B project is not private: access={project_result.get('access')!r}"
        )
    if not isinstance(viewer, dict) or viewer.get("username") != entity:
        raise RuntimeError(
            "authenticated W&B viewer is not the personal entity owner: "
            f"{None if not isinstance(viewer, dict) else viewer.get('username')!r}"
        )
    return {
        "entity": entity,
        "project": project,
        "access": "PRIVATE",
        "viewer_username": str(viewer["username"]),
        "viewer_email": str(viewer.get("email", "")),
    }


def _abc_manifest_summary(data_root: Path) -> dict[str, Any]:
    manifest = _canonical_regular_file(
        str(data_root / "abc_pp" / "manifest.txt"), "ABC manifest"
    )
    entries = [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(entries) != 10_000 or len(set(entries)) != 10_000:
        raise RuntimeError(
            "ABC manifest must contain exactly 10,000 unique nonempty entries; "
            f"found {len(entries)} entries and {len(set(entries))} unique paths"
        )
    if any(not Path(entry).is_absolute() for entry in entries):
        raise RuntimeError("ABC manifest contains a non-absolute episode path")
    return {
        "path": str(manifest),
        "sha256": _sha256(manifest),
        "entries": len(entries),
        "first_entry": entries[0],
        "last_entry": entries[-1],
    }


def _verify_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected_sha256 = _validated_sha256(
        expected_sha256, "expected checkpoint SHA-256"
    )
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"warm-start checkpoint changed: {actual_sha256} != {expected_sha256}"
        )
    if actual_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "checkpoint is not the reviewed production warm start: "
            f"{actual_sha256} != {EXPECTED_CHECKPOINT_SHA256}"
        )
    info = path.stat()
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "bytes": info.st_size,
        "model_only_load": True,
        "optimizer_state_loaded": False,
    }


def command_wandb_private(args: argparse.Namespace) -> int:
    print(json.dumps(_wandb_private_project(args.entity, args.project), sort_keys=True))
    return 0


def command_create_pair(args: argparse.Namespace) -> int:
    pair_id = _validated_id(args.pair_id, "pair ID")
    expected_commit = _validated_commit(args.git_commit)
    repo_root = _canonical_directory(args.repo_root, "repository root")
    run_root = _canonical_directory(args.run_root, "run root")
    pair_root = _canonical_directory(args.pair_root, "pair root")
    if pair_root.parent != run_root:
        raise ValueError(f"pair root must be a direct child of run root: {pair_root}")
    if pair_root.name != pair_id:
        raise ValueError(
            f"pair root basename {pair_root.name!r} does not match pair ID {pair_id!r}"
        )
    _assert_clean_commit(repo_root, expected_commit)

    checkpoint = _canonical_regular_file(args.checkpoint, "warm-start checkpoint")
    checkpoint_summary = _verify_checkpoint(
        checkpoint, args.checkpoint_sha256
    )
    data_root = _canonical_directory(args.data_root, "data root")
    python = _python_executable(args.python)
    wan_dir = _canonical_directory(args.wan_dir, "Wan directory")
    videox_home = _canonical_directory(args.videox_home, "VideoX-Fun checkout")
    wandb_summary = _wandb_private_project(args.wandb_entity, args.wandb_project)

    common_config = _canonical_regular_file(
        args.common_config, "common experiment config"
    )
    arm_configs = {}
    for arm, path_value in (
        ("no-ztf-condition", args.no_ztf_config),
        ("with-ztf-condition", args.with_ztf_config),
    ):
        path = _canonical_regular_file(path_value, f"{arm} config")
        arm_configs[arm] = {
            "path": str(path),
            "sha256": _sha256(path),
            "condition_on_tf": ARMS[arm]["condition_on_tf"],
            "selector": ARMS[arm]["config"],
        }
    smoke_reports = {}
    for arm, variant, path_value in (
        ("no-ztf-condition", "dual-no-ztf", args.no_ztf_smoke_report),
        ("with-ztf-condition", "dual-with-ztf", args.with_ztf_smoke_report),
    ):
        path = _canonical_regular_file(path_value, f"{arm} gradient smoke report")
        smoke_reports[arm] = {
            "path": str(path),
            "sha256": _sha256(path),
            "variant": variant,
            "data_mode": "real",
            "warmstart_sha256": args.checkpoint_sha256,
        }

    payload = _identity_payload(
        {
            "schema_version": 1,
            "kind": "dual_abc_ztf_matched_pair",
            "created_at_utc": _now(),
            "pair_id": pair_id,
            "git_commit": expected_commit,
            "repository_root": str(repo_root),
            "paths": {
                "python": str(python),
                "wan_dir": str(wan_dir),
                "videox_home": str(videox_home),
                "data_root": str(data_root),
                "run_root": str(run_root),
                "pair_root": str(pair_root),
            },
            "checkpoint": checkpoint_summary,
            "data": {
                "datasets": ["ABC"],
                "abc_manifest": _abc_manifest_summary(data_root),
            },
            "config": {
                "common": {
                    "path": str(common_config),
                    "sha256": _sha256(common_config),
                },
                "arms": arm_configs,
                "sole_mechanism_difference": "model.dual_diffusion.condition_on_tf",
            },
            "gradient_smoke_reports": smoke_reports,
            "schedule": {
                "seed": 1234,
                "optimizer_updates": 100,
                "warmup_updates": 20,
                "batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "gpus": 8,
                "effective_global_batch_size": 8,
                "log_every": 5,
                "validate_every": 10,
                "save_every": 50,
                "visualize_every": 50,
            },
            "diffusion_clock": {
                "convention": "sigma=1 noise, sigma=0 clean",
                "tf_schedule": "sigmoid(logit(video_sigma)-1), exact endpoints",
            },
            "wandb": {
                **wandb_summary,
                "group": None,
            },
            "slurm": {
                "nodes_per_arm": 1,
                "gpus_per_node": 8,
                "array": "0-1%2",
                "requeue": False,
            },
        }
    )
    output = Path(args.output)
    if output.parent.resolve(strict=True) != pair_root:
        raise ValueError(f"pair manifest must be written directly under {pair_root}")
    _exclusive_json(output, payload)
    print(payload["identity_sha256"])
    return 0


def _compose_resolved_config(
    python: Path,
    project_root: Path,
    overrides: list[str],
) -> bytes:
    command = [
        str(python),
        "train.py",
        *overrides,
        "--cfg",
        "job",
        "--resolve",
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Hydra config resolution failed:\n"
            + completed.stderr.decode(errors="replace")
        )
    if not completed.stdout.strip():
        raise RuntimeError("Hydra emitted an empty resolved configuration")
    return completed.stdout


def _config_value(config: Any, dotted: str) -> Any:
    current = config
    for part in dotted.split("."):
        if part not in current:
            raise RuntimeError(f"resolved config is missing {dotted}")
        current = current[part]
    return current


def _assert_resolved_contract(
    content: bytes,
    *,
    arm: str,
    condition_on_tf: bool,
    checkpoint: Path,
    run_dir: Path,
    wandb_run_id: str,
) -> None:
    from omegaconf import OmegaConf

    config = OmegaConf.create(content.decode())
    expected = {
        "name": wandb_run_id,
        "seed": 1234,
        "data_loader.batch_size": 1,
        "trainer.config.max_iter": 100,
        "trainer.config.gradient_accumulation_steps": 1,
        "trainer.config.load_path": str(checkpoint),
        "trainer.config.logging.log_every": 5,
        "trainer.config.saving.save_every": 50,
        "trainer.config.validation.val_every": 10,
        "trainer.config.visualization.viz_every": 50,
        "trainer.config.visualization.require_success": True,
        "lr_scheduler_factory.lr_lambda.warmup_steps": 20,
        "lr_scheduler_factory.lr_lambda.total_steps": 100,
        "model.dual_diffusion.enabled": True,
        "model.dual_diffusion.condition_on_tf": condition_on_tf,
        "model.dual_diffusion.schedule_mode": "tf_leads",
        "model.dual_diffusion.tf_lead_logit": 1.0,
        "model.dual_diffusion.capture_latent_trajectories": True,
        "model.time_frequency_transform.num_views": 3,
        "model.time_frequency_transform.window_size": 4,
        "model.time_frequency_transform.normalization": "none",
        "wandb.enabled": True,
        "wandb.mode": "online",
        "wandb.entity": EXPECTED_ENTITY,
        "wandb.project": EXPECTED_PROJECT,
        "wandb.id": wandb_run_id,
        "wandb.group": None,
        "wandb.resume": "never",
    }
    problems = []
    for dotted, wanted in expected.items():
        actual = _config_value(config, dotted)
        if actual != wanted:
            problems.append(f"{dotted}: {actual!r} != {wanted!r}")

    for loader in ("dataset", "val_dataset", "viz_dataset"):
        names = list(_config_value(config, f"{loader}.datasets").keys())
        if names != ["ABC"]:
            problems.append(f"{loader}.datasets: {names!r} != ['ABC']")

    save_path = Path(str(_config_value(config, "trainer.config.saving.save_path")))
    viz_path = Path(
        str(_config_value(config, "trainer.config.visualization.viz_path"))
    )
    if save_path != run_dir / "snapshot.pt":
        problems.append(f"saving.save_path: {save_path} != {run_dir / 'snapshot.pt'}")
    if viz_path != run_dir / "visualization":
        problems.append(f"visualization.viz_path: {viz_path} != {run_dir / 'visualization'}")

    expected_tag = arm
    tags = list(_config_value(config, "wandb.tags"))
    if expected_tag not in tags:
        problems.append(f"wandb.tags does not contain {expected_tag!r}: {tags!r}")
    if problems:
        raise RuntimeError(
            "resolved pilot configuration violates the matched-pair contract: "
            + "; ".join(problems)
        )


def command_prepare_arm(args: argparse.Namespace) -> int:
    arm = args.arm
    arm_contract = ARMS[arm]
    expected_condition = bool(arm_contract["condition_on_tf"])
    if args.condition_on_tf != ("true" if expected_condition else "false"):
        raise ValueError(
            f"arm {arm} requires condition_on_tf={str(expected_condition).lower()}"
        )
    if args.config_selector != arm_contract["config"]:
        raise ValueError(
            f"arm {arm} requires selector {arm_contract['config']!r}, "
            f"got {args.config_selector!r}"
        )

    pair_id = _validated_id(args.pair_id, "pair ID")
    wandb_run_id = _validated_id(args.wandb_run_id, "W&B run ID")
    expected_commit = _validated_commit(args.git_commit)
    repo_root = _canonical_directory(args.repo_root, "repository root")
    project_root = _canonical_directory(args.project_root, "project root")
    run_dir = _canonical_directory(args.run_dir, "arm run directory")
    pair_root = _canonical_directory(args.pair_root, "pair root")
    if run_dir.parent != pair_root:
        raise ValueError(f"arm run directory must be directly beneath {pair_root}")
    if pair_root.name != pair_id:
        raise ValueError("pair root and pair ID disagree")
    _assert_clean_commit(repo_root, expected_commit)

    python = _python_executable(args.python)
    checkpoint = _canonical_regular_file(args.checkpoint, "warm-start checkpoint")
    checkpoint_summary = _verify_checkpoint(checkpoint, args.checkpoint_sha256)
    data_root = _canonical_directory(args.data_root, "data root")
    wan_dir = _canonical_directory(args.wan_dir, "Wan directory")
    videox_home = _canonical_directory(args.videox_home, "VideoX-Fun checkout")
    wandb_summary = _wandb_private_project(args.wandb_entity, args.wandb_project)

    pair_manifest = _canonical_regular_file(args.pair_manifest, "pair manifest")
    pair_payload = json.loads(pair_manifest.read_text(encoding="utf-8"))
    if pair_payload.get("pair_id") != pair_id:
        raise RuntimeError("pair manifest has a different pair ID")
    if pair_payload.get("git_commit") != expected_commit:
        raise RuntimeError("pair manifest has a different Git commit")
    if pair_payload.get("checkpoint", {}).get("sha256") != args.checkpoint_sha256:
        raise RuntimeError("pair manifest has a different checkpoint SHA-256")
    smoke_report = _canonical_regular_file(
        args.smoke_report, f"{arm} gradient smoke report"
    )
    recorded_smoke = pair_payload.get("gradient_smoke_reports", {}).get(arm, {})
    if recorded_smoke.get("sha256") != _sha256(smoke_report):
        raise RuntimeError(
            "gradient smoke report differs from the submitted pair manifest"
        )

    arm_config = _canonical_regular_file(args.arm_config, "arm config")
    common_config = _canonical_regular_file(args.common_config, "common config")
    recorded_configs = pair_payload.get("config", {})
    recorded_arm = recorded_configs.get("arms", {}).get(arm, {})
    recorded_common = recorded_configs.get("common", {})
    if recorded_arm.get("sha256") != _sha256(arm_config):
        raise RuntimeError("arm config differs from the submitted pair manifest")
    if recorded_common.get("sha256") != _sha256(common_config):
        raise RuntimeError("common config differs from the submitted pair manifest")
    current_abc = _abc_manifest_summary(data_root)
    if (
        pair_payload.get("data", {}).get("abc_manifest", {}).get("sha256")
        != current_abc["sha256"]
    ):
        raise RuntimeError("ABC manifest differs from the submitted pair manifest")

    resolved_content = _compose_resolved_config(
        python, project_root, args.override
    )
    _assert_resolved_contract(
        resolved_content,
        arm=arm,
        condition_on_tf=expected_condition,
        checkpoint=checkpoint,
        run_dir=run_dir,
        wandb_run_id=wandb_run_id,
    )
    resolved_output = Path(args.resolved_config_output)
    manifest_output = Path(args.manifest_output)
    if (
        resolved_output.parent.resolve(strict=True) != run_dir
        or manifest_output.parent.resolve(strict=True) != run_dir
    ):
        raise ValueError("arm provenance files must be written directly in run_dir")
    _exclusive_bytes(resolved_output, resolved_content)
    resolved_sha256 = _sha256(resolved_output)

    payload = _identity_payload(
        {
            "schema_version": 1,
            "kind": "dual_abc_ztf_arm",
            "created_at_utc": _now(),
            "pair_id": pair_id,
            "pair_manifest": {
                "path": str(pair_manifest),
                "sha256": _sha256(pair_manifest),
                "identity_sha256": pair_payload.get("identity_sha256"),
            },
            "arm": arm,
            "condition_on_tf": expected_condition,
            "git_commit": expected_commit,
            "repository_root": str(repo_root),
            "config": {
                "selector": args.config_selector,
                "arm_path": str(arm_config),
                "arm_sha256": _sha256(arm_config),
                "common_path": str(common_config),
                "common_sha256": _sha256(common_config),
                "resolved_path": str(resolved_output),
                "resolved_sha256": resolved_sha256,
                "hydra_overrides": list(args.override),
            },
            "checkpoint": checkpoint_summary,
            "gradient_smoke_report": {
                "path": str(smoke_report),
                "sha256": _sha256(smoke_report),
                "variant": recorded_smoke.get("variant"),
                "data_mode": "real",
            },
            "data": {
                "root": str(data_root),
                "datasets": ["ABC"],
                "abc_manifest": current_abc,
            },
            "assets": {
                "wan_dir": str(wan_dir),
                "videox_home": str(videox_home),
            },
            "run": {
                "run_dir": str(run_dir),
                "wandb_run_id": wandb_run_id,
                "seed": 1234,
                "optimizer_updates": 100,
                "fresh_optimizer": True,
                "world_size": 8,
            },
            "wandb": {
                **wandb_summary,
                "run_id": wandb_run_id,
                "group": None,
            },
            "diffusion_clock": {
                "convention": "sigma=1 noise, sigma=0 clean",
            },
            "slurm": {
                "job_id": args.slurm_job_id,
                "array_job_id": args.slurm_array_job_id,
                "array_task_id": int(args.slurm_array_task_id),
                "requeue": False,
            },
        }
    )
    _exclusive_json(manifest_output, payload)
    print(payload["identity_sha256"])
    return 0


def command_record_submission(args: argparse.Namespace) -> int:
    output = Path(args.output)
    pair_manifest = _canonical_regular_file(args.pair_manifest, "pair manifest")
    payload = {
        "schema_version": 1,
        "kind": "dual_abc_ztf_slurm_submission",
        "created_at_utc": _now(),
        "pair_manifest": {
            "path": str(pair_manifest),
            "sha256": _sha256(pair_manifest),
        },
        "slurm_array_job_id": args.job_id,
        "array": "0-1%2",
        "requeue": False,
    }
    _exclusive_json(output, payload)
    return 0


def command_record_outcome(args: argparse.Namespace) -> int:
    manifest = _canonical_regular_file(args.manifest, "arm manifest")
    arm_payload = json.loads(manifest.read_text(encoding="utf-8"))
    snapshot = Path(args.snapshot)
    status = int(args.exit_status)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "dual_abc_ztf_arm_outcome",
        "created_at_utc": _now(),
        "manifest": {
            "path": str(manifest),
            "sha256": _sha256(manifest),
        },
        "exit_status": status,
        "completed": status == 0,
    }
    if status == 0:
        run_dir = snapshot.parent.resolve(strict=True)
        completion_path = _canonical_regular_file(
            str(run_dir / "training_complete.json"), "training completion marker"
        )
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected_identity = arm_payload.get("identity_sha256")
        expected_completion = {
            "schema_version": 1,
            "status": "completed",
            "completed_updates": 100,
            "max_iter": 100,
            "run_identity_sha256": expected_identity,
            "snapshot": str(snapshot.resolve(strict=False)),
        }
        completion_problems = [
            f"{key}: {completion.get(key)!r} != {value!r}"
            for key, value in expected_completion.items()
            if completion.get(key) != value
        ]
        if completion_problems:
            raise RuntimeError(
                "training completion marker is invalid: "
                + "; ".join(completion_problems)
            )

        visualization_summary: dict[str, Any] = {}
        for iteration in (0, 50, 99):
            iteration_dir = _canonical_directory(
                str(run_dir / "visualization" / f"iter_{iteration}"),
                f"iteration {iteration} visualization directory",
            )
            trajectories = sorted(
                iteration_dir.glob("*/latent_trajectory_rank_*.safetensors")
            )
            trajectory_manifests = sorted(
                iteration_dir.glob("*/latent_trajectory_rank_*.json")
            )
            decoded_videos = sorted(iteration_dir.glob("*/*.mp4"))
            if (
                len(trajectories) != 8
                or len(trajectory_manifests) != 8
                or len(decoded_videos) != 8
            ):
                raise RuntimeError(
                    f"iteration {iteration} artifact count mismatch: "
                    f"trajectories={len(trajectories)}, "
                    f"manifests={len(trajectory_manifests)}, "
                    f"decoded_videos={len(decoded_videos)}; expected 8 each"
                )

            trajectory_ranks: set[int] = set()
            manifest_ranks: set[int] = set()
            video_ranks: set[int] = set()
            trajectory_records = []
            for trajectory in trajectories:
                if trajectory.is_symlink() or trajectory.stat().st_size <= 0:
                    raise RuntimeError(
                        f"invalid latent trajectory at iteration {iteration}: {trajectory}"
                    )
                match = re.search(
                    r"latent_trajectory_rank_([0-9]+)[.]safetensors$",
                    trajectory.name,
                )
                if match is None:
                    raise RuntimeError(f"unexpected trajectory name: {trajectory}")
                rank = int(match.group(1))
                trajectory_ranks.add(rank)
                manifest_path = trajectory.with_suffix(".json")
                record = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    record.get("iteration") != iteration
                    or record.get("global_rank") != rank
                    or record.get("sigma_convention") != "1=noise,0=clean"
                ):
                    raise RuntimeError(
                        f"invalid trajectory manifest: {manifest_path}"
                    )
                actual_sha256 = _sha256(trajectory)
                if record.get("safetensors_sha256") != actual_sha256:
                    raise RuntimeError(
                        f"trajectory SHA-256 mismatch: {trajectory}"
                    )
                trajectory_records.append(
                    {
                        "path": str(trajectory),
                        "sha256": actual_sha256,
                        "bytes": trajectory.stat().st_size,
                        "rank": rank,
                    }
                )
            for manifest_path in trajectory_manifests:
                match = re.search(
                    r"latent_trajectory_rank_([0-9]+)[.]json$",
                    manifest_path.name,
                )
                if match is None or manifest_path.stat().st_size <= 0:
                    raise RuntimeError(
                        f"invalid trajectory manifest filename: {manifest_path}"
                    )
                manifest_ranks.add(int(match.group(1)))
            for video in decoded_videos:
                if video.is_symlink() or video.stat().st_size <= 0:
                    raise RuntimeError(
                        f"invalid decoded video at iteration {iteration}: {video}"
                    )
                match = re.search(r"_([0-9]+)_[0-9]+[.]mp4$", video.name)
                if match is None:
                    raise RuntimeError(f"unexpected decoded video name: {video}")
                video_ranks.add(int(match.group(1)))

            expected_ranks = set(range(8))
            if (
                trajectory_ranks != expected_ranks
                or manifest_ranks != expected_ranks
                or video_ranks != expected_ranks
            ):
                raise RuntimeError(
                    f"iteration {iteration} is missing rank artifacts: "
                    f"trajectories={sorted(trajectory_ranks)}, "
                    f"manifests={sorted(manifest_ranks)}, "
                    f"videos={sorted(video_ranks)}"
                )
            visualization_summary[str(iteration)] = {
                "trajectory_count": 8,
                "manifest_count": 8,
                "decoded_video_count": 8,
                "ranks": list(range(8)),
                "trajectories": trajectory_records,
            }

        payload["training_completion"] = {
            "path": str(completion_path),
            "sha256": _sha256(completion_path),
            "completed_updates": 100,
            "max_iter": 100,
            "run_identity_sha256": expected_identity,
        }
        payload["visualization_artifacts"] = visualization_summary
    if snapshot.is_file() and not snapshot.is_symlink():
        payload["snapshot"] = {
            "path": str(snapshot.resolve(strict=True)),
            "bytes": snapshot.stat().st_size,
            "sha256": _sha256(snapshot),
        }
    elif status == 0:
        raise RuntimeError(f"training exited successfully without a snapshot: {snapshot}")
    _exclusive_json(Path(args.output), payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    wandb_parser = subparsers.add_parser(
        "wandb-private", help="verify the authenticated personal project is private"
    )
    wandb_parser.add_argument("--entity", default=EXPECTED_ENTITY)
    wandb_parser.add_argument("--project", default=EXPECTED_PROJECT)
    wandb_parser.set_defaults(func=command_wandb_private)

    pair_parser = subparsers.add_parser(
        "create-pair", help="write the immutable matched-pair submission manifest"
    )
    for name in (
        "pair-id",
        "git-commit",
        "repo-root",
        "run-root",
        "pair-root",
        "python",
        "wan-dir",
        "videox-home",
        "data-root",
        "checkpoint",
        "checkpoint-sha256",
        "no-ztf-smoke-report",
        "with-ztf-smoke-report",
        "common-config",
        "no-ztf-config",
        "with-ztf-config",
        "wandb-entity",
        "wandb-project",
        "output",
    ):
        pair_parser.add_argument(f"--{name}", required=True)
    pair_parser.set_defaults(func=command_create_pair)

    arm_parser = subparsers.add_parser(
        "prepare-arm", help="resolve and validate one arm before torchrun"
    )
    arm_parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    arm_parser.add_argument("--condition-on-tf", choices=("true", "false"), required=True)
    for name in (
        "pair-id",
        "git-commit",
        "repo-root",
        "project-root",
        "pair-root",
        "run-dir",
        "python",
        "wan-dir",
        "videox-home",
        "data-root",
        "checkpoint",
        "checkpoint-sha256",
        "config-selector",
        "arm-config",
        "common-config",
        "wandb-entity",
        "wandb-project",
        "wandb-run-id",
        "pair-manifest",
        "smoke-report",
        "slurm-job-id",
        "slurm-array-job-id",
        "slurm-array-task-id",
        "resolved-config-output",
        "manifest-output",
    ):
        arm_parser.add_argument(f"--{name}", required=True)
    arm_parser.add_argument("--override", action="append", default=[], required=True)
    arm_parser.set_defaults(func=command_prepare_arm)

    submission_parser = subparsers.add_parser(
        "record-submission", help="record the accepted Slurm array job ID"
    )
    submission_parser.add_argument("--pair-manifest", required=True)
    submission_parser.add_argument("--job-id", required=True)
    submission_parser.add_argument("--output", required=True)
    submission_parser.set_defaults(func=command_record_submission)

    outcome_parser = subparsers.add_parser(
        "record-outcome", help="write a terminal arm outcome without mutating its manifest"
    )
    outcome_parser.add_argument("--manifest", required=True)
    outcome_parser.add_argument("--snapshot", required=True)
    outcome_parser.add_argument("--exit-status", required=True, type=int)
    outcome_parser.add_argument("--output", required=True)
    outcome_parser.set_defaults(func=command_record_outcome)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
