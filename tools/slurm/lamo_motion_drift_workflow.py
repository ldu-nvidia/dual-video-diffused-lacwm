#!/usr/bin/env python3
"""Fail-closed workflow helpers for the two-arm LaMo VPM screen.

This tool never submits work itself.  It validates a prospective registration,
creates immutable per-arm execution receipts, renders exact commands, and runs
the final analysis only after both validation inventories exist.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_lamo_motion_drift as analysis  # noqa: E402
from tools import dual_abc_pilot  # noqa: E402
from tools import lamo_motion_drift_evaluate as evaluation  # noqa: E402


SCHEMA_VERSION = 1
KIND_ARM_PLAN = "lamo_motion_drift_arm_execution_plan"


class MotionDriftWorkflowError(RuntimeError):
    """A workflow action is not fresh or differs from registration."""


def _registration(path: Path) -> dict[str, Any]:
    value = evaluation._validate_registration(path)
    source = value["tool_repository"]
    observed = evaluation._clean_source(
        Path(source["path"]), source["git_commit"], "tool"
    )
    if observed != source:
        raise MotionDriftWorkflowError("tool source changed after registration")
    return value


def _arm(code: str) -> evaluation.Arm:
    arm = evaluation.ARM_BY_CODE.get(code)
    if arm is None:
        raise MotionDriftWorkflowError(f"unknown arm: {code}")
    return arm


def arm_paths(
    registration: dict[str, Any], arm: evaluation.Arm
) -> dict[str, Path]:
    root = Path(registration["output_root"])
    return {
        "training_root": root / "training",
        "run_dir": root / "training" / arm.run_name,
        "evaluation_dir": root / "evaluation" / arm.code.lower(),
        "arm_plan": root / "arm_plans" / f"{arm.code.lower()}.json",
    }


def arm_identity(registration: dict[str, Any], arm: evaluation.Arm) -> str:
    return evaluation.arm_run_identity(registration, arm)


def arm_values(registration: dict[str, Any], arm: evaluation.Arm) -> dict[str, Any]:
    paths = arm_paths(registration, arm)
    validation = registration["validation"]
    return {
        "arm_code": arm.code,
        "config_name": arm.config_name,
        "run_name": arm.run_name,
        "motion_drift_weight": arm.motion_drift_weight,
        "run_identity_sha256": arm_identity(registration, arm),
        **{key: str(value) for key, value in paths.items()},
        "parent_snapshot": registration["controlled_study"]["parent_snapshot"][
            "path"
        ],
        "train_manifest": registration["training"]["manifest"]["path"],
        "train_cache_metadata": registration["training"]["cache_metadata"][
            "path"
        ],
        "validation_manifest": validation["manifest"]["path"],
        "validation_cache_metadata": validation["cache_metadata"]["path"],
        "wan_dir": registration["runtime"]["wan_dir"],
        "videox_home": registration["runtime"]["videox_home"],
        "python": registration["runtime"]["python"],
    }


def _assert_fresh_arm(values: dict[str, Any]) -> None:
    for key in ("run_dir", "evaluation_dir"):
        path = Path(values[key])
        if path.exists() or path.is_symlink():
            raise MotionDriftWorkflowError(f"fresh arm output exists: {path}")
    training_root = Path(values["training_root"])
    evaluation_root = Path(values["evaluation_dir"]).parent
    for root in (training_root, evaluation_root):
        root.mkdir(parents=True, exist_ok=True, mode=0o700)


def command_arm_values(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    values = arm_values(registration, _arm(args.arm))
    if args.require_fresh:
        _assert_fresh_arm(values)
    if args.format == "json":
        print(json.dumps(values, sort_keys=True))
    else:
        order = (
            "arm_code",
            "config_name",
            "run_name",
            "run_identity_sha256",
            "training_root",
            "run_dir",
            "evaluation_dir",
            "parent_snapshot",
            "train_manifest",
            "train_cache_metadata",
            "validation_manifest",
            "validation_cache_metadata",
            "wan_dir",
            "videox_home",
            "python",
        )
        fields = [str(values[key]) for key in order]
        if any("\t" in field or "\n" in field for field in fields):
            raise MotionDriftWorkflowError("arm value contains a shell delimiter")
        print("\t".join(fields))
    return 0


def command_write_arm_plan(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    arm = _arm(args.arm)
    values = arm_values(registration, arm)
    _assert_fresh_arm(values)
    input_revalidation = evaluation.revalidate_registered_inputs(
        registration,
        include_parent=True,
        include_train=True,
        include_validation=True,
    )
    plan = evaluation.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_ARM_PLAN,
            "status": "planned_before_arm_training_or_metrics",
            "registration_identity_sha256": registration["identity_sha256"],
            "tool_repository": registration["tool_repository"],
            "arm": {
                "code": arm.code,
                "config_name": arm.config_name,
                "run_name": arm.run_name,
                "motion_drift_weight": arm.motion_drift_weight,
            },
            "run_identity_sha256": values["run_identity_sha256"],
            "paths": {
                key: values[key]
                for key in ("run_dir", "evaluation_dir", "parent_snapshot")
            },
            "input_revalidation": input_revalidation,
            "training": {
                "updates": 200,
                "seed": 1234,
                "world_size": 8,
                "batch_size_per_rank": 1,
                "global_batch_size": 8,
                "optimizer_state_policy": "fresh_identical_adamw",
                "ema_policy": "none_in_historical_lacwm_and_none_in_both_arms",
                "same_data_order_and_rng_seed_contract": True,
            },
            "evaluation": {
                "validation_clips": 64,
                "nfe_grid": list(evaluation.NFE_GRID),
                "endpoints": [
                    {
                        "code": endpoint.code,
                        "nfe": endpoint.nfe,
                        "action_source": endpoint.action_source,
                        "primary_gate": endpoint.primary_gate,
                    }
                    for endpoint in evaluation.ENDPOINTS
                ],
                "protected_test_accessed": False,
                "target_cache_array_opened": False,
            },
            "wandb": {
                "entity": "zijiandu",
                "project": "dual-video-diffusion-private",
                "group": None,
                "mode": "online",
                "run_id": values["run_identity_sha256"],
            },
        }
    )
    plan_path = Path(values["arm_plan"])
    evaluation._exclusive_json(plan_path, plan)
    print(json.dumps(plan, sort_keys=True))
    return 0


def command_wandb_check(_args: argparse.Namespace) -> int:
    result = dual_abc_pilot._wandb_private_project(
        "zijiandu", "dual-video-diffusion-private"
    )
    if result.get("access") != "PRIVATE" or result.get("viewer_username") != "zijiandu":
        raise MotionDriftWorkflowError("W&B personal project privacy check failed")
    print(json.dumps({**result, "group": None}, sort_keys=True))
    return 0


def _train_command(values: dict[str, Any], repo: Path) -> list[str]:
    return [
        values["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(
            repo
            / "projects"
            / "latent_action_models"
            / "train_lamo_motion_drift.py"
        ),
        f"+experiments_0908={values['config_name']}",
        f"hydra.run.dir={values['run_dir']}",
        f"hydra.sweep.dir={values['run_dir']}",
        f"trainer.config.saving.save_path={values['run_dir']}/snapshot.pt",
        "wandb.enabled=true",
        "wandb.mode=online",
        "wandb.entity=zijiandu",
        "wandb.project=dual-video-diffusion-private",
        "wandb.group=null",
        f"+wandb.id={values['run_identity_sha256']}",
        "+wandb.resume=never",
    ]


def _evaluation_command(
    values: dict[str, Any], repo: Path, registration: Path
) -> list[str]:
    return [
        values["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(repo / "tools" / "lamo_motion_drift_evaluate.py"),
        "evaluate",
        "--registration",
        str(registration),
        "--arm",
        values["arm_code"],
        "--output-dir",
        values["evaluation_dir"],
        "--batch-size-per-rank",
        "2",
    ]


def command_plan(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    repo = Path(registration["tool_repository"]["path"])
    arms = []
    for arm in evaluation.ARMS:
        values = arm_values(registration, arm)
        arms.append(
            {
                "arm": arm.code,
                "run_identity_sha256": values["run_identity_sha256"],
                "environment": {
                    "WAN_DIR": values["wan_dir"],
                    "VIDEOX_HOME": values["videox_home"],
                    "LAMO_VPM_SNAPSHOT": values["parent_snapshot"],
                    "LAMO_VPM_SNAPSHOT_SHA256": evaluation.PARENT_SNAPSHOT_SHA256,
                    "LAMO_TRAIN_CLIP_MANIFEST": values["train_manifest"],
                    "LAMO_TRAIN_CACHE_METADATA": values["train_cache_metadata"],
                    "LAMO_VAL_CLIP_MANIFEST": values["validation_manifest"],
                    "LAMO_VAL_CACHE_METADATA": values[
                        "validation_cache_metadata"
                    ],
                    "LAMO_RUN_ROOT": values["training_root"],
                    "LACWM_RUN_IDENTITY_SHA256": values[
                        "run_identity_sha256"
                    ],
                    "WANDB_ENTITY": "zijiandu",
                    "WANDB_PROJECT": "dual-video-diffusion-private",
                    "WANDB_MODE": "online",
                    "WANDB_RUN_GROUP": None,
                },
                "train_command_argv": _train_command(values, repo),
                "train_command_shell": shlex.join(_train_command(values, repo)),
                "evaluation_command_argv": _evaluation_command(
                    values, repo, args.registration
                ),
                "evaluation_command_shell": shlex.join(
                    _evaluation_command(values, repo, args.registration)
                ),
                "outputs": {
                    "training": values["run_dir"],
                    "evaluation": values["evaluation_dir"],
                },
            }
        )
    root = Path(registration["output_root"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "lamo_motion_drift_dry_run",
        "mode": "no_commands_executed_no_files_created",
        "registration_identity_sha256": registration["identity_sha256"],
        "arms": arms,
        "analysis": {
            "baseline_inventory": str(
                root / "evaluation" / "vpm-cont" / "inventory.json"
            ),
            "drift_inventory": str(
                root / "evaluation" / "vpm-drift" / "inventory.json"
            ),
            "output": str(root / "analysis" / "analysis.json"),
        },
        "slurm": {
            "nodes_per_arm": 1,
            "b200_per_arm": 8,
            "qos": "short",
            "time_limit": "02:00:00",
            "requeue": False,
        },
        "protected_test_accessed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    root = Path(registration["output_root"])
    output = root / "analysis" / "analysis.json"
    result = evaluation.identity_payload(
        analysis.analyze(
            root / "evaluation" / "vpm-drift" / "inventory.json",
            root / "evaluation" / "vpm-cont" / "inventory.json",
        )
    )
    analysis._exclusive_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


def _add_registration(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registration", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    values = commands.add_parser("arm-values")
    _add_registration(values)
    values.add_argument("--arm", choices=tuple(evaluation.ARM_BY_CODE), required=True)
    values.add_argument("--format", choices=("json", "tsv"), default="json")
    values.add_argument("--require-fresh", action="store_true")
    values.set_defaults(func=command_arm_values)
    arm_plan = commands.add_parser("write-arm-plan")
    _add_registration(arm_plan)
    arm_plan.add_argument("--arm", choices=tuple(evaluation.ARM_BY_CODE), required=True)
    arm_plan.set_defaults(func=command_write_arm_plan)
    wandb = commands.add_parser("wandb-check")
    wandb.set_defaults(func=command_wandb_check)
    plan = commands.add_parser("plan")
    _add_registration(plan)
    plan.set_defaults(func=command_plan)
    analyze = commands.add_parser("analyze")
    _add_registration(analyze)
    analyze.set_defaults(func=command_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
