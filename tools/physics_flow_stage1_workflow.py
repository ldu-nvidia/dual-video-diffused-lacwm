#!/usr/bin/env python3
"""Fail-closed execution planning for raw physics-flow Stage-1.

The helper never submits a job.  It converts the prospective registration into
exact environment/argv contracts consumed by the guarded Slurm entrypoint.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import dual_abc_pilot  # noqa: E402
from tools import physics_flow_stage1 as stage  # noqa: E402


class PhysicsFlowWorkflowError(RuntimeError):
    """A planned launch is stale or would violate the registration."""


def _registration(path: Path) -> dict[str, Any]:
    return stage.validate_study_registration(path)


def _arm(code: str) -> stage.Arm:
    value = stage.ARM_BY_CODE.get(code)
    if value is None:
        raise PhysicsFlowWorkflowError(f"unknown arm: {code}")
    return value


def arm_values(registration: dict[str, Any], arm: stage.Arm) -> dict[str, str]:
    root = Path(registration["output_root"])
    cache_registration = stage.read_json(
        Path(registration["flow_caches"]["train"]["metadata"]["path"])
        .parent.parent
        / "cache_registration.json",
        "cache registration",
    )
    train_input = cache_registration["inputs"]["train"]
    validation_input = cache_registration["inputs"]["validation"]
    return {
        "arm": arm.code,
        "config_name": arm.config_name,
        "run_name": arm.run_name,
        "run_identity": registration["arm_run_identity_sha256"][arm.code],
        "run_dir": str(root / "training" / arm.run_name),
        "training_root": str(root / "training"),
        "arm_plan": str(root / "arm_plans" / f"{arm.code.lower()}.json"),
        "parent_snapshot": registration["parent"]["snapshot"]["path"],
        "parent_resolved_config": registration["parent"]["resolved_config"][
            "path"
        ],
        "train_manifest": train_input["manifest"]["path"],
        "train_rgb_metadata": train_input["cache"]["metadata"]["path"],
        "train_flow_metadata": registration["flow_caches"]["train"]["metadata"][
            "path"
        ],
        "train_flow_metadata_sha256": registration["flow_caches"]["train"][
            "metadata"
        ]["sha256"],
        "train_raw_sha256": registration["flow_caches"]["train"]["arrays"]["raw"][
            "sha256"
        ],
        "val_manifest": validation_input["manifest"]["path"],
        "val_rgb_metadata": validation_input["cache"]["metadata"]["path"],
        "val_flow_metadata": registration["flow_caches"]["val"]["metadata"]["path"],
        "val_flow_metadata_sha256": registration["flow_caches"]["val"]["metadata"][
            "sha256"
        ],
        "val_raw_sha256": registration["flow_caches"]["val"]["arrays"]["raw"][
            "sha256"
        ],
        "renderer_analysis_identity": stage.RENDERER_ANALYSIS_IDENTITY,
        "study_registration_identity": registration["identity_sha256"],
        "python": registration["runtime"]["python"],
        "wan_dir": registration["runtime"]["wan_dir"],
        "videox_home": registration["runtime"]["videox_home"],
    }


def _assert_fresh_training(values: dict[str, str]) -> None:
    run_dir = Path(values["run_dir"])
    if run_dir.exists() or run_dir.is_symlink():
        raise PhysicsFlowWorkflowError(f"fresh arm run already exists: {run_dir}")
    Path(values["training_root"]).mkdir(parents=True, exist_ok=True, mode=0o700)


def command_arm_values(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    values = arm_values(registration, _arm(args.arm))
    if args.require_fresh:
        _assert_fresh_training(values)
    if args.format == "json":
        print(json.dumps(values, sort_keys=True))
    else:
        order = (
            "arm",
            "config_name",
            "run_name",
            "run_identity",
            "run_dir",
            "training_root",
            "arm_plan",
            "parent_snapshot",
            "parent_resolved_config",
            "train_manifest",
            "train_rgb_metadata",
            "train_flow_metadata",
            "train_flow_metadata_sha256",
            "train_raw_sha256",
            "val_manifest",
            "val_rgb_metadata",
            "val_flow_metadata",
            "val_flow_metadata_sha256",
            "val_raw_sha256",
            "renderer_analysis_identity",
            "study_registration_identity",
            "python",
            "wan_dir",
            "videox_home",
        )
        fields = [values[key] for key in order]
        if any("\t" in value or "\n" in value for value in fields):
            raise PhysicsFlowWorkflowError("arm value contains a shell delimiter")
        print("\t".join(fields))
    return 0


def command_write_arm_plan(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    arm = _arm(args.arm)
    values = arm_values(registration, arm)
    _assert_fresh_training(values)
    path = Path(values["arm_plan"])
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    plan = stage.identity_payload(
        {
            "schema_version": stage.SCHEMA_VERSION,
            "kind": "raw_physics_flow_stage1_arm_plan",
            "status": "planned_before_arm_training_or_generated_video_outcomes",
            "registration_identity_sha256": registration["identity_sha256"],
            "arm": stage.asdict(arm),
            "run_identity_sha256": values["run_identity"],
            "run_dir": values["run_dir"],
            "parent_snapshot_sha256": stage.PARENT_SNAPSHOT_SHA256,
            "parent_resolved_config_sha256": (
                stage.PARENT_RESOLVED_CONFIG_SHA256
            ),
            "train_flow_metadata_sha256": values["train_flow_metadata_sha256"],
            "train_raw_sha256": values["train_raw_sha256"],
            "seed": 1234,
            "updates": 200,
            "world_size": 8,
            "global_batch_size": 8,
            "optimizer_state_policy": "fresh_identical_adamw",
            "ema": False,
            "future_rgb_opened": False,
            "future_measured_state_opened": False,
            "generator_outcome_opened": False,
            "protected_test_accessed": False,
        }
    )
    stage.exclusive_json(path, plan)
    print(json.dumps(plan, sort_keys=True))
    return 0


def command_wandb_check(_args: argparse.Namespace) -> int:
    result = dual_abc_pilot._wandb_private_project(
        "zijiandu", "dual-video-diffusion-private"
    )
    if result.get("access") != "PRIVATE" or result.get("viewer_username") != "zijiandu":
        raise PhysicsFlowWorkflowError("W&B personal privacy check failed")
    print(json.dumps({**result, "group": None}, sort_keys=True))
    return 0


def _train_command(
    registration_path: Path,
    registration: dict[str, Any],
    arm: stage.Arm,
) -> list[str]:
    values = arm_values(registration, arm)
    repo = Path(registration["source_repository"]["path"])
    return [
        values["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(repo / "projects" / "latent_action_models" / "physics_flow_train.py"),
        f"+experiments_0908={values['config_name']}",
        f"hydra.run.dir={values['run_dir']}",
        f"hydra.sweep.dir={values['run_dir']}",
        f"trainer.config.saving.save_path={values['run_dir']}/snapshot.pt",
        "wandb.enabled=true",
        "wandb.mode=online",
        "wandb.entity=zijiandu",
        "wandb.project=dual-video-diffusion-private",
        "wandb.group=null",
        f"+wandb.id={values['run_name']}",
        "+wandb.resume=never",
    ]


def command_plan(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    repo = Path(registration["source_repository"]["path"])
    root = Path(registration["output_root"])
    arms = []
    for arm in stage.ARMS:
        values = arm_values(registration, arm)
        command = _train_command(args.registration, registration, arm)
        arms.append(
            {
                "arm": arm.code,
                "run_identity_sha256": values["run_identity"],
                "argv": command,
                "shell": shlex.join(command),
                "output": values["run_dir"],
            }
        )
    evaluation = [
        registration["runtime"]["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(repo / "tools" / "physics_flow_stage1_evaluate.py"),
        "--registration",
        str(args.registration),
        "--output-dir",
        str(root / "evaluation"),
        "--batch-size",
        "2",
    ]
    parent_parity = [
        registration["runtime"]["python"],
        str(repo / "tools" / "physics_flow_parent_parity.py"),
        "--registration",
        str(args.registration),
        "--output",
        str(root / stage.PARENT_PARITY_FILENAME),
    ]
    print(
        json.dumps(
            {
                "kind": "raw_physics_flow_stage1_dry_run",
                "mode": "no_commands_executed_no_files_created",
                "registration_identity_sha256": registration["identity_sha256"],
                "arms": arms,
                "native_parent_sampler_parity": {
                    "argv": parent_parity,
                    "shell": shlex.join(parent_parity),
                    "must_complete_before_validation_open": True,
                },
                "evaluation": {"argv": evaluation, "shell": shlex.join(evaluation)},
                "finalization": [
                    str(repo / "tools" / "physics_flow_stage1.py"),
                    "compare-traces/analyze/audit-study",
                ],
                "slurm": {
                    "training_nodes": 2,
                    "evaluation_nodes": 1,
                    "b200_per_node": 8,
                    "non_requeueable": True,
                },
                "protected_test_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    values = subparsers.add_parser("arm-values")
    values.add_argument("--registration", type=Path, required=True)
    values.add_argument("--arm", choices=tuple(stage.ARM_BY_CODE), required=True)
    values.add_argument("--format", choices=("json", "tsv"), default="json")
    values.add_argument("--require-fresh", action="store_true")
    values.set_defaults(handler=command_arm_values)

    plan_arm = subparsers.add_parser("write-arm-plan")
    plan_arm.add_argument("--registration", type=Path, required=True)
    plan_arm.add_argument("--arm", choices=tuple(stage.ARM_BY_CODE), required=True)
    plan_arm.set_defaults(handler=command_write_arm_plan)

    wandb = subparsers.add_parser("wandb-check")
    wandb.set_defaults(handler=command_wandb_check)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--registration", type=Path, required=True)
    plan.set_defaults(handler=command_plan)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (PhysicsFlowWorkflowError, stage.PhysicsFlowStage1Error) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
