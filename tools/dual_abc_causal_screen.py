#!/usr/bin/env python3
"""Provenance and contract checks for the five-arm ABC causal TF screen.

This helper is intentionally separate from ``dual_abc_pilot.py``.  The
completed two-arm pilot remains immutable, while this screen tests whether
matched TF content helps more than either an exact-off control or a shuffled
TF control at explicitly nonzero, fixed conditioning scales.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dual_abc_pilot as pilot


EXPECTED_ENTITY = "zijiandu"
EXPECTED_PROJECT = "dual-video-diffusion-private"
CONFIG_SELECTOR = "ravenhuang/wan-dit/dual_abc_with_ztf_condition.yaml"
EVALUATION_NFE_STEPS = [1, 2, 4, 8]
EVALUATION_NOISE_SEED = 20260726
EVALUATION_CONDITION_SOURCES = [
    "autonomous",
    "off",
    "oracle_matched",
    "oracle_shuffled",
]
OPTIMIZER_UPDATES = 200
WARMUP_UPDATES = 20
VISUALIZATION_UPDATES = (0, 50, 100, 150, 199)
ARRAY_JOB_ID_RE = re.compile(r"^[0-9]+([_;][A-Za-z0-9_.%+-]+)?$")

# Array order is part of the immutable experimental contract.
ARMS: tuple[dict[str, Any], ...] = (
    {
        "name": "off_s000",
        "condition_on_tf": False,
        "condition_mode": "off",
        "state_gate_init": 0.0,
        "smoke_variant": "dual-no-ztf",
        "smoke_report_key": "no_ztf",
    },
    {
        "name": "matched_s003",
        "condition_on_tf": True,
        "condition_mode": "matched",
        "state_gate_init": 0.03,
        "smoke_variant": "dual-with-ztf",
        "smoke_report_key": "with_ztf",
    },
    {
        "name": "shuffled_s003",
        "condition_on_tf": True,
        "condition_mode": "shuffled",
        "state_gate_init": 0.03,
        "smoke_variant": "dual-with-ztf",
        "smoke_report_key": "with_ztf",
    },
    {
        "name": "matched_s010",
        "condition_on_tf": True,
        "condition_mode": "matched",
        "state_gate_init": 0.10,
        "smoke_variant": "dual-with-ztf",
        "smoke_report_key": "with_ztf",
    },
    {
        "name": "shuffled_s010",
        "condition_on_tf": True,
        "condition_mode": "shuffled",
        "state_gate_init": 0.10,
        "smoke_variant": "dual-with-ztf",
        "smoke_report_key": "with_ztf",
    },
)


def _arm(task_id: int) -> dict[str, Any]:
    if task_id < 0 or task_id >= len(ARMS):
        raise ValueError(f"array task ID must be in [0, {len(ARMS) - 1}]")
    return ARMS[task_id]


def _identity_is_valid(payload: dict[str, Any]) -> bool:
    recorded = payload.get("identity_sha256")
    if not isinstance(recorded, str):
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(canonical).hexdigest() == recorded


def _read_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return payload


def _report_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "no_ztf": pilot._canonical_regular_file(
            args.no_ztf_smoke_report, "no-ZTF gradient smoke report"
        ),
        "with_ztf": pilot._canonical_regular_file(
            args.with_ztf_smoke_report, "with-ZTF gradient smoke report"
        ),
    }


def _arm_manifest_contract() -> dict[str, dict[str, Any]]:
    return {
        str(task_id): {
            "array_task_id": task_id,
            "name": arm["name"],
            "condition_on_tf": arm["condition_on_tf"],
            "condition_mode": arm["condition_mode"],
            "state_gate_init": arm["state_gate_init"],
            "state_gate_trainable": False,
            "evaluation_nfe_steps": EVALUATION_NFE_STEPS,
            "evaluation_noise_seed": EVALUATION_NOISE_SEED,
            "evaluation_condition_sources": EVALUATION_CONDITION_SOURCES,
            "smoke_variant": arm["smoke_variant"],
        }
        for task_id, arm in enumerate(ARMS)
    }


def command_arm_contract(args: argparse.Namespace) -> int:
    arm = _arm(args.array_task_id)
    payload = {
        "array_task_id": args.array_task_id,
        **arm,
        "state_gate_trainable": False,
        "evaluation_nfe_steps": EVALUATION_NFE_STEPS,
        "evaluation_noise_seed": EVALUATION_NOISE_SEED,
        "evaluation_condition_sources": EVALUATION_CONDITION_SOURCES,
    }
    if args.format == "json":
        print(json.dumps(payload, sort_keys=True))
    else:
        fields = (
            payload["name"],
            str(payload["condition_on_tf"]).lower(),
            payload["condition_mode"],
            f"{payload['state_gate_init']:.2f}",
            payload["smoke_variant"],
            payload["smoke_report_key"],
        )
        if any("\t" in str(field) or "\n" in str(field) for field in fields):
            raise RuntimeError("arm contract contains a shell-unsafe delimiter")
        print("\t".join(str(field) for field in fields))
    return 0


def command_wandb_private(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            pilot._wandb_private_project(args.entity, args.project),
            sort_keys=True,
        )
    )
    return 0


def command_create_screen(args: argparse.Namespace) -> int:
    screen_id = pilot._validated_id(args.screen_id, "screen ID")
    expected_commit = pilot._validated_commit(args.git_commit)
    repo_root = pilot._canonical_directory(args.repo_root, "repository root")
    run_root = pilot._canonical_directory(args.run_root, "run root")
    screen_root = pilot._canonical_directory(args.screen_root, "screen root")
    if screen_root.parent != run_root:
        raise ValueError(
            f"screen root must be a direct child of run root: {screen_root}"
        )
    if screen_root.name != screen_id:
        raise ValueError(
            f"screen root basename {screen_root.name!r} does not match "
            f"screen ID {screen_id!r}"
        )
    pilot._assert_clean_commit(repo_root, expected_commit)

    checkpoint = pilot._canonical_regular_file(
        args.checkpoint, "warm-start checkpoint"
    )
    checkpoint_summary = pilot._verify_checkpoint(
        checkpoint, args.checkpoint_sha256
    )
    data_root = pilot._canonical_directory(args.data_root, "data root")
    python = pilot._python_executable(args.python)
    wan_dir = pilot._canonical_directory(args.wan_dir, "Wan directory")
    videox_home = pilot._canonical_directory(
        args.videox_home, "VideoX-Fun checkout"
    )
    wandb_summary = pilot._wandb_private_project(
        args.wandb_entity, args.wandb_project
    )
    common_config = pilot._canonical_regular_file(
        args.common_config, "common experiment config"
    )
    arm_config = pilot._canonical_regular_file(
        args.arm_config, "base arm experiment config"
    )
    smoke_paths = _report_paths(args)
    max_concurrent_arms = int(args.max_concurrent_arms)
    if max_concurrent_arms < 1 or max_concurrent_arms > len(ARMS):
        raise ValueError(
            f"max concurrent arms must be between 1 and {len(ARMS)}"
        )

    payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "dual_abc_tf_causal_screen",
            "created_at_utc": pilot._now(),
            "screen_id": screen_id,
            "git_commit": expected_commit,
            "repository_root": str(repo_root),
            "paths": {
                "python": str(python),
                "wan_dir": str(wan_dir),
                "videox_home": str(videox_home),
                "data_root": str(data_root),
                "run_root": str(run_root),
                "screen_root": str(screen_root),
            },
            "checkpoint": checkpoint_summary,
            "data": {
                "datasets": ["ABC"],
                "abc_manifest": pilot._abc_manifest_summary(data_root),
            },
            "config": {
                "common": {
                    "path": str(common_config),
                    "sha256": pilot._sha256(common_config),
                },
                "base_arm": {
                    "selector": CONFIG_SELECTOR,
                    "path": str(arm_config),
                    "sha256": pilot._sha256(arm_config),
                },
                "arms": _arm_manifest_contract(),
                "controlled_factors": [
                    "model.dual_diffusion.condition_mode",
                    "model.dual_diffusion.state_gate_init",
                ],
                "matched_factors": [
                    "source_commit",
                    "warm_start",
                    "ABC_manifest",
                    "seed",
                    "optimizer",
                    "schedule",
                    "evaluation_noise",
                    "NFE_schedules",
                ],
            },
            "gradient_smoke_reports": {
                key: {
                    "path": str(path),
                    "sha256": pilot._sha256(path),
                    "variant": (
                        "dual-no-ztf" if key == "no_ztf" else "dual-with-ztf"
                    ),
                    "data_mode": "real",
                    "warmstart_sha256": args.checkpoint_sha256,
                }
                for key, path in smoke_paths.items()
            },
            "schedule": {
                "seed": 1234,
                "optimizer_updates": OPTIMIZER_UPDATES,
                "warmup_updates": WARMUP_UPDATES,
                "batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "gpus_per_arm": 8,
                "effective_global_batch_size_per_arm": 8,
                "log_every": 5,
                "validate_every": 10,
                "save_every": 50,
                "visualize_every": 50,
                "evaluation_nfe_steps": EVALUATION_NFE_STEPS,
                "evaluation_noise_seed": EVALUATION_NOISE_SEED,
                "evaluation_condition_sources": EVALUATION_CONDITION_SOURCES,
            },
            "diffusion_clock": {
                "convention": "sigma=1 noise, sigma=0 clean",
                "tf_schedule": (
                    "sigmoid(logit(video_sigma)-1), exact endpoints"
                ),
            },
            "causal_question": (
                "Does matched TF content outperform both exact-off and "
                "same-strength shuffled-TF controls?"
            ),
            "wandb": {
                **wandb_summary,
                "group": None,
            },
            "slurm": {
                "nodes_per_arm": 1,
                "gpus_per_node": 8,
                "array": f"0-{len(ARMS) - 1}%{max_concurrent_arms}",
                "requeue": False,
            },
        }
    )
    output = Path(args.output)
    if output.parent.resolve(strict=True) != screen_root:
        raise ValueError(
            f"screen manifest must be written directly under {screen_root}"
        )
    pilot._exclusive_json(output, payload)
    print(payload["identity_sha256"])
    return 0


def _config_value(config: Any, dotted: str) -> Any:
    return pilot._config_value(config, dotted)


def _assert_equal(
    problems: list[str],
    config: Any,
    dotted: str,
    wanted: Any,
) -> None:
    actual = _config_value(config, dotted)
    if actual != wanted:
        problems.append(f"{dotted}: {actual!r} != {wanted!r}")


def _assert_float(
    problems: list[str],
    config: Any,
    dotted: str,
    wanted: float,
) -> None:
    actual = _config_value(config, dotted)
    try:
        matches = math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        matches = False
    if not matches:
        problems.append(f"{dotted}: {actual!r} != {wanted!r}")


def _assert_resolved_contract(
    content: bytes,
    *,
    arm: dict[str, Any],
    checkpoint: Path,
    run_dir: Path,
    wandb_run_id: str,
) -> None:
    from omegaconf import OmegaConf

    config = OmegaConf.create(content.decode())
    problems: list[str] = []
    expected = {
        "name": wandb_run_id,
        "seed": 1234,
        "data_loader.batch_size": 1,
        "trainer.config.max_iter": OPTIMIZER_UPDATES,
        "trainer.config.gradient_accumulation_steps": 1,
        "trainer.config.load_path": str(checkpoint),
        "trainer.config.logging.log_every": 5,
        "trainer.config.saving.save_every": 50,
        "trainer.config.validation.val_every": 10,
        "trainer.config.visualization.viz_every": 50,
        "trainer.config.visualization.require_success": True,
        "lr_scheduler_factory.lr_lambda.warmup_steps": WARMUP_UPDATES,
        "lr_scheduler_factory.lr_lambda.total_steps": OPTIMIZER_UPDATES,
        "model.dual_diffusion.enabled": True,
        "model.dual_diffusion.condition_on_tf": arm["condition_on_tf"],
        "model.dual_diffusion.condition_mode": arm["condition_mode"],
        "model.dual_diffusion.state_gate_trainable": False,
        "model.dual_diffusion.evaluation_nfe_steps": EVALUATION_NFE_STEPS,
        "model.dual_diffusion.evaluation_noise_seed": EVALUATION_NOISE_SEED,
        "model.dual_diffusion.evaluation_condition_sources": (
            EVALUATION_CONDITION_SOURCES
        ),
        "model.dual_diffusion.schedule_mode": "tf_leads",
        "model.dual_diffusion.tf_lead_logit": 1.0,
        "model.dual_diffusion.capture_latent_trajectories": True,
        "model.forward_model.dual_diffusion.condition_mode": arm[
            "condition_mode"
        ],
        "model.forward_model.dual_diffusion.state_gate_trainable": False,
        "model.forward_model.dual_diffusion.evaluation_nfe_steps": (
            EVALUATION_NFE_STEPS
        ),
        "model.forward_model.dual_diffusion.evaluation_noise_seed": (
            EVALUATION_NOISE_SEED
        ),
        "model.forward_model.dual_diffusion.evaluation_condition_sources": (
            EVALUATION_CONDITION_SOURCES
        ),
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
    for dotted, wanted in expected.items():
        _assert_equal(problems, config, dotted, wanted)
    _assert_float(
        problems,
        config,
        "model.dual_diffusion.state_gate_init",
        float(arm["state_gate_init"]),
    )
    _assert_float(
        problems,
        config,
        "model.forward_model.dual_diffusion.state_gate_init",
        float(arm["state_gate_init"]),
    )

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
        problems.append(
            f"visualization.viz_path: {viz_path} != {run_dir / 'visualization'}"
        )

    tags = list(_config_value(config, "wandb.tags"))
    for expected_tag in ("ztf-causal-screen", str(arm["name"]), "seed-1234"):
        if expected_tag not in tags:
            problems.append(
                f"wandb.tags does not contain {expected_tag!r}: {tags!r}"
            )
    if problems:
        raise RuntimeError(
            "resolved causal-screen configuration violates its contract: "
            + "; ".join(problems)
        )


def _validate_screen_manifest(
    path: Path,
    *,
    screen_id: str,
    expected_commit: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    payload = _read_json_file(path, "screen manifest")
    problems = []
    expected = {
        "kind": "dual_abc_tf_causal_screen",
        "screen_id": screen_id,
        "git_commit": expected_commit,
    }
    for key, wanted in expected.items():
        if payload.get(key) != wanted:
            problems.append(f"{key}: {payload.get(key)!r} != {wanted!r}")
    if payload.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        problems.append("checkpoint SHA-256 differs")
    if payload.get("config", {}).get("arms") != _arm_manifest_contract():
        problems.append("arm contract differs")
    if not _identity_is_valid(payload):
        problems.append("identity SHA-256 is invalid")
    if problems:
        raise RuntimeError("invalid screen manifest: " + "; ".join(problems))
    return payload


def command_prepare_arm(args: argparse.Namespace) -> int:
    arm = _arm(args.array_task_id)
    screen_id = pilot._validated_id(args.screen_id, "screen ID")
    wandb_run_id = pilot._validated_id(args.wandb_run_id, "W&B run ID")
    expected_commit = pilot._validated_commit(args.git_commit)
    repo_root = pilot._canonical_directory(args.repo_root, "repository root")
    project_root = pilot._canonical_directory(args.project_root, "project root")
    run_dir = pilot._canonical_directory(args.run_dir, "arm run directory")
    screen_root = pilot._canonical_directory(args.screen_root, "screen root")
    if run_dir.parent != screen_root:
        raise ValueError(f"arm run directory must be directly beneath {screen_root}")
    if run_dir.name != arm["name"]:
        raise ValueError(
            f"run directory basename {run_dir.name!r} != arm {arm['name']!r}"
        )
    if screen_root.name != screen_id:
        raise ValueError("screen root and screen ID disagree")
    pilot._assert_clean_commit(repo_root, expected_commit)

    python = pilot._python_executable(args.python)
    checkpoint = pilot._canonical_regular_file(
        args.checkpoint, "warm-start checkpoint"
    )
    checkpoint_summary = pilot._verify_checkpoint(
        checkpoint, args.checkpoint_sha256
    )
    data_root = pilot._canonical_directory(args.data_root, "data root")
    wan_dir = pilot._canonical_directory(args.wan_dir, "Wan directory")
    videox_home = pilot._canonical_directory(
        args.videox_home, "VideoX-Fun checkout"
    )
    wandb_summary = pilot._wandb_private_project(
        args.wandb_entity, args.wandb_project
    )

    screen_manifest = pilot._canonical_regular_file(
        args.screen_manifest, "screen manifest"
    )
    screen_payload = _validate_screen_manifest(
        screen_manifest,
        screen_id=screen_id,
        expected_commit=expected_commit,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    smoke_report = pilot._canonical_regular_file(
        args.smoke_report, f"{arm['name']} gradient smoke report"
    )
    recorded_smoke = screen_payload.get("gradient_smoke_reports", {}).get(
        arm["smoke_report_key"], {}
    )
    if recorded_smoke.get("sha256") != pilot._sha256(smoke_report):
        raise RuntimeError(
            "gradient smoke report differs from the submitted screen manifest"
        )
    if recorded_smoke.get("variant") != arm["smoke_variant"]:
        raise RuntimeError("gradient smoke variant differs from the arm contract")

    arm_config = pilot._canonical_regular_file(args.arm_config, "base arm config")
    common_config = pilot._canonical_regular_file(
        args.common_config, "common config"
    )
    recorded_configs = screen_payload.get("config", {})
    if recorded_configs.get("base_arm", {}).get("sha256") != pilot._sha256(
        arm_config
    ):
        raise RuntimeError("base arm config differs from the screen manifest")
    if recorded_configs.get("common", {}).get("sha256") != pilot._sha256(
        common_config
    ):
        raise RuntimeError("common config differs from the screen manifest")
    current_abc = pilot._abc_manifest_summary(data_root)
    if (
        screen_payload.get("data", {}).get("abc_manifest", {}).get("sha256")
        != current_abc["sha256"]
    ):
        raise RuntimeError("ABC manifest differs from the screen manifest")

    resolved_content = pilot._compose_resolved_config(
        python, project_root, args.override
    )
    _assert_resolved_contract(
        resolved_content,
        arm=arm,
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
    pilot._exclusive_bytes(resolved_output, resolved_content)
    resolved_sha256 = pilot._sha256(resolved_output)

    payload = pilot._identity_payload(
        {
            "schema_version": 1,
            "kind": "dual_abc_tf_causal_screen_arm",
            "created_at_utc": pilot._now(),
            "screen_id": screen_id,
            "screen_manifest": {
                "path": str(screen_manifest),
                "sha256": pilot._sha256(screen_manifest),
                "identity_sha256": screen_payload.get("identity_sha256"),
            },
            "array_task_id": args.array_task_id,
            "arm": arm["name"],
            "condition_on_tf": arm["condition_on_tf"],
            "condition_mode": arm["condition_mode"],
            "state_gate_init": arm["state_gate_init"],
            "state_gate_trainable": False,
            "git_commit": expected_commit,
            "repository_root": str(repo_root),
            "config": {
                "selector": CONFIG_SELECTOR,
                "base_arm_path": str(arm_config),
                "base_arm_sha256": pilot._sha256(arm_config),
                "common_path": str(common_config),
                "common_sha256": pilot._sha256(common_config),
                "resolved_path": str(resolved_output),
                "resolved_sha256": resolved_sha256,
                "hydra_overrides": list(args.override),
            },
            "checkpoint": checkpoint_summary,
            "gradient_smoke_report": {
                "path": str(smoke_report),
                "sha256": pilot._sha256(smoke_report),
                "variant": arm["smoke_variant"],
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
                "optimizer_updates": OPTIMIZER_UPDATES,
                "fresh_optimizer": True,
                "world_size": 8,
                "evaluation_nfe_steps": EVALUATION_NFE_STEPS,
                "evaluation_noise_seed": EVALUATION_NOISE_SEED,
                "evaluation_condition_sources": EVALUATION_CONDITION_SOURCES,
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
                "array_task_id": args.array_task_id,
                "requeue": False,
            },
        }
    )
    pilot._exclusive_json(manifest_output, payload)
    print(payload["identity_sha256"])
    return 0


def command_record_submission(args: argparse.Namespace) -> int:
    screen_manifest = pilot._canonical_regular_file(
        args.screen_manifest, "screen manifest"
    )
    screen_payload = _read_json_file(screen_manifest, "screen manifest")
    if screen_payload.get("kind") != "dual_abc_tf_causal_screen":
        raise RuntimeError("screen manifest kind is invalid")
    if not _identity_is_valid(screen_payload):
        raise RuntimeError("screen manifest identity SHA-256 is invalid")
    if not ARRAY_JOB_ID_RE.fullmatch(args.job_id):
        raise ValueError("Slurm returned an invalid array job ID")
    max_concurrent_arms = int(args.max_concurrent_arms)
    if max_concurrent_arms < 1 or max_concurrent_arms > len(ARMS):
        raise ValueError(
            f"max concurrent arms must be between 1 and {len(ARMS)}"
        )
    payload = {
        "schema_version": 1,
        "kind": "dual_abc_tf_causal_screen_slurm_submission",
        "created_at_utc": pilot._now(),
        "screen_manifest": {
            "path": str(screen_manifest),
            "sha256": pilot._sha256(screen_manifest),
            "identity_sha256": screen_payload.get("identity_sha256"),
        },
        "slurm_array_job_id": args.job_id,
        "array": f"0-{len(ARMS) - 1}%{max_concurrent_arms}",
        "requeue": False,
    }
    output = Path(args.output)
    if output.parent.resolve(strict=True) != screen_manifest.parent:
        raise ValueError("submission record must be directly under screen root")
    pilot._exclusive_json(output, payload)
    return 0


def _required_trajectory_tensor_names() -> set[str]:
    required_tensor_names = {
        "video_clean",
        "tf_clean",
        "video_initial_state",
        "tf_initial_state",
        "tf_initial_noise",
        "ground_truth_future_uint8",
        "history_latent_frames",
        "condition_on_tf",
        "condition_mode_code",
        "evaluation_noise_seed",
        "evaluation_nfe_steps",
        "evaluation_condition_source_codes",
        "oracle_sources_are_leakage",
    }
    for source in EVALUATION_CONDITION_SOURCES:
        source_infix = "" if source == "autonomous" else f"_{source}"
        for nfe in EVALUATION_NFE_STEPS:
            required_tensor_names.update(
                {
                    f"video_final{source_infix}_nfe_{nfe}",
                    f"tf_final{source_infix}_nfe_{nfe}",
                    f"decoded_future{source_infix}_nfe_{nfe}",
                }
            )
    return required_tensor_names


def _trajectory_summary(run_dir: Path) -> dict[str, Any]:
    from safetensors import safe_open

    required_tensor_names = _required_trajectory_tensor_names()

    summary: dict[str, Any] = {}
    for iteration in VISUALIZATION_UPDATES:
        iteration_dir = pilot._canonical_directory(
            str(run_dir / "visualization" / f"iter_{iteration}"),
            f"iteration {iteration} visualization directory",
        )
        trajectories = sorted(
            iteration_dir.glob("*/latent_trajectory_rank_*.safetensors")
        )
        sidecars = sorted(iteration_dir.glob("*/latent_trajectory_rank_*.json"))
        decoded_videos = sorted(iteration_dir.glob("*/*.mp4"))
        if (
            len(trajectories) != 8
            or len(sidecars) != 8
            or len(decoded_videos) != 8
        ):
            raise RuntimeError(
                f"iteration {iteration} artifact count mismatch: "
                f"trajectories={len(trajectories)}, sidecars={len(sidecars)}, "
                f"decoded_videos={len(decoded_videos)}; expected 8 each"
            )

        records = []
        ranks: set[int] = set()
        for trajectory in trajectories:
            if trajectory.is_symlink() or trajectory.stat().st_size <= 0:
                raise RuntimeError(f"invalid trajectory file: {trajectory}")
            match = re.search(
                r"latent_trajectory_rank_([0-9]+)[.]safetensors$",
                trajectory.name,
            )
            if match is None:
                raise RuntimeError(f"unexpected trajectory filename: {trajectory}")
            rank = int(match.group(1))
            ranks.add(rank)
            sidecar_path = trajectory.with_suffix(".json")
            sidecar = _read_json_file(sidecar_path, "trajectory sidecar")
            sidecar_expected = {
                "iteration": iteration,
                "global_rank": rank,
                "sigma_convention": "1=noise,0=clean",
            }
            problems = [
                f"{key}: {sidecar.get(key)!r} != {wanted!r}"
                for key, wanted in sidecar_expected.items()
                if sidecar.get(key) != wanted
            ]
            actual_sha256 = pilot._sha256(trajectory)
            if sidecar.get("safetensors_sha256") != actual_sha256:
                problems.append("safetensors_sha256 differs")
            with safe_open(trajectory, framework="pt", device="cpu") as handle:
                tensor_names = set(handle.keys())
            missing = sorted(required_tensor_names - tensor_names)
            if missing:
                problems.append(f"missing tensors: {missing}")
            if problems:
                raise RuntimeError(
                    f"invalid trajectory artifact {trajectory}: "
                    + "; ".join(problems)
                )
            records.append(
                {
                    "path": str(trajectory),
                    "sha256": actual_sha256,
                    "bytes": trajectory.stat().st_size,
                    "rank": rank,
                    "tensor_names": sorted(tensor_names),
                }
            )
        if ranks != set(range(8)):
            raise RuntimeError(
                f"iteration {iteration} trajectory ranks differ: {sorted(ranks)}"
            )
        summary[str(iteration)] = {
            "trajectory_count": 8,
            "sidecar_count": 8,
            "decoded_video_count": 8,
            "ranks": list(range(8)),
            "trajectories": records,
        }
    return summary


def command_record_outcome(args: argparse.Namespace) -> int:
    manifest = pilot._canonical_regular_file(args.manifest, "arm manifest")
    arm_payload = _read_json_file(manifest, "arm manifest")
    if arm_payload.get("kind") != "dual_abc_tf_causal_screen_arm":
        raise RuntimeError("arm manifest kind is invalid")
    if not _identity_is_valid(arm_payload):
        raise RuntimeError("arm manifest identity SHA-256 is invalid")

    snapshot = Path(args.snapshot)
    status = int(args.exit_status)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "dual_abc_tf_causal_screen_arm_outcome",
        "created_at_utc": pilot._now(),
        "manifest": {
            "path": str(manifest),
            "sha256": pilot._sha256(manifest),
            "identity_sha256": arm_payload.get("identity_sha256"),
        },
        "exit_status": status,
        "completed": status == 0,
    }
    if status == 0:
        run_dir = snapshot.parent.resolve(strict=True)
        completion_path = pilot._canonical_regular_file(
            str(run_dir / "training_complete.json"), "training completion marker"
        )
        completion = _read_json_file(completion_path, "training completion marker")
        expected_completion = {
            "schema_version": 1,
            "status": "completed",
            "completed_updates": OPTIMIZER_UPDATES,
            "max_iter": OPTIMIZER_UPDATES,
            "run_identity_sha256": arm_payload.get("identity_sha256"),
            "snapshot": str(snapshot.resolve(strict=False)),
        }
        problems = [
            f"{key}: {completion.get(key)!r} != {wanted!r}"
            for key, wanted in expected_completion.items()
            if completion.get(key) != wanted
        ]
        if problems:
            raise RuntimeError(
                "training completion marker is invalid: " + "; ".join(problems)
            )
        payload["training_completion"] = {
            "path": str(completion_path),
            "sha256": pilot._sha256(completion_path),
            "completed_updates": OPTIMIZER_UPDATES,
            "max_iter": OPTIMIZER_UPDATES,
            "run_identity_sha256": arm_payload.get("identity_sha256"),
        }
        payload["visualization_artifacts"] = _trajectory_summary(run_dir)

    if snapshot.is_file() and not snapshot.is_symlink():
        payload["snapshot"] = {
            "path": str(snapshot.resolve(strict=True)),
            "bytes": snapshot.stat().st_size,
            "sha256": pilot._sha256(snapshot),
        }
    elif status == 0:
        raise RuntimeError(
            f"training exited successfully without a snapshot: {snapshot}"
        )
    output = Path(args.output)
    if output.parent.resolve(strict=True) != manifest.parent:
        raise ValueError(
            "outcome must be written directly under the arm run directory"
        )
    pilot._exclusive_json(output, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    arm_contract = subparsers.add_parser(
        "arm-contract", help="emit the immutable mapping for one array task"
    )
    arm_contract.add_argument("--array-task-id", required=True, type=int)
    arm_contract.add_argument("--format", choices=("json", "tsv"), default="json")
    arm_contract.set_defaults(func=command_arm_contract)

    wandb_parser = subparsers.add_parser(
        "wandb-private", help="verify the authenticated personal project is private"
    )
    wandb_parser.add_argument("--entity", default=EXPECTED_ENTITY)
    wandb_parser.add_argument("--project", default=EXPECTED_PROJECT)
    wandb_parser.set_defaults(func=command_wandb_private)

    screen_parser = subparsers.add_parser(
        "create-screen", help="write the immutable causal-screen manifest"
    )
    for name in (
        "screen-id",
        "git-commit",
        "repo-root",
        "run-root",
        "screen-root",
        "python",
        "wan-dir",
        "videox-home",
        "data-root",
        "checkpoint",
        "checkpoint-sha256",
        "no-ztf-smoke-report",
        "with-ztf-smoke-report",
        "common-config",
        "arm-config",
        "wandb-entity",
        "wandb-project",
        "max-concurrent-arms",
        "output",
    ):
        screen_parser.add_argument(f"--{name}", required=True)
    screen_parser.set_defaults(func=command_create_screen)

    prepare_parser = subparsers.add_parser(
        "prepare-arm", help="resolve and validate one arm before torchrun"
    )
    prepare_parser.add_argument("--array-task-id", required=True, type=int)
    for name in (
        "screen-id",
        "git-commit",
        "repo-root",
        "project-root",
        "screen-root",
        "run-dir",
        "python",
        "wan-dir",
        "videox-home",
        "data-root",
        "checkpoint",
        "checkpoint-sha256",
        "arm-config",
        "common-config",
        "wandb-entity",
        "wandb-project",
        "wandb-run-id",
        "screen-manifest",
        "smoke-report",
        "slurm-job-id",
        "slurm-array-job-id",
        "resolved-config-output",
        "manifest-output",
    ):
        prepare_parser.add_argument(f"--{name}", required=True)
    prepare_parser.add_argument(
        "--override", action="append", default=[], required=True
    )
    prepare_parser.set_defaults(func=command_prepare_arm)

    submission_parser = subparsers.add_parser(
        "record-submission", help="record the accepted Slurm array job ID"
    )
    submission_parser.add_argument("--screen-manifest", required=True)
    submission_parser.add_argument("--job-id", required=True)
    submission_parser.add_argument("--max-concurrent-arms", required=True, type=int)
    submission_parser.add_argument("--output", required=True)
    submission_parser.set_defaults(func=command_record_submission)

    outcome_parser = subparsers.add_parser(
        "record-outcome", help="write a terminal arm outcome"
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
