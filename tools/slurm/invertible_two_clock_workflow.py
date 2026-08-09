#!/usr/bin/env python3
"""Fail-closed, non-submitting workflow helpers for IPQ-TC1."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import analyze_invertible_two_clock as analyzer  # noqa: E402
from tools import dual_abc_pilot  # noqa: E402
from tools import invertible_two_clock_pilot as pilot  # noqa: E402


class IPQWorkflowError(RuntimeError):
    """A registered workflow path, identity, or privacy boundary changed."""


def _registration(path: Path) -> dict[str, Any]:
    return pilot.validate_registration(path)


def _arm(code: str) -> pilot.Arm:
    try:
        return pilot.ARM_BY_CODE[code]
    except KeyError as exc:
        raise IPQWorkflowError(f"unsupported arm: {code}") from exc


def arm_values(
    registration: Mapping[str, Any], arm: pilot.Arm
) -> dict[str, Any]:
    root = Path(registration["output_root"])
    run_dir = root / "training" / arm.run_name
    evaluation_dir = root / "evaluation" / arm.code.lower()
    return {
        "arm_code": arm.code,
        "arm_mode": arm.arm_mode,
        "config_name": arm.config_name,
        "run_name": arm.run_name,
        "run_identity_sha256": registration["arm_run_identity_sha256"][arm.code],
        "output_root": str(root),
        "run_dir": str(run_dir),
        "evaluation_dir": str(evaluation_dir),
        "arm_plan": str(root / "arm_plans" / f"{arm.code.lower()}.json"),
        "slurm_log_dir": str(root / "_slurm_logs"),
        "parent_snapshot": registration["parent"]["snapshot"]["path"],
        "parent_resolved_config": registration["parent"]["resolved_config"]["path"],
        "train_manifest": registration["training"]["manifest"]["path"],
        "train_cache_metadata": registration["training"]["cache_metadata"]["path"],
        "python": registration["runtime"]["python"],
        "wan_dir": registration["runtime"]["wan_dir"],
        "videox_home": registration["runtime"]["videox_home"],
        "wandb_enabled": bool(registration["wandb"]["enabled"]),
    }


def _fresh(path: str, label: str) -> None:
    value = Path(path)
    if value.exists() or value.is_symlink():
        raise IPQWorkflowError(f"fresh {label} already exists: {value}")


def command_arm_values(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    values = arm_values(registration, _arm(args.arm))
    if args.require_training_fresh:
        _fresh(values["run_dir"], "training output")
    if args.require_evaluation_fresh:
        _fresh(values["evaluation_dir"], "evaluation output")
    if args.format == "json":
        print(json.dumps(values, sort_keys=True))
    else:
        order = (
            "arm_code",
            "arm_mode",
            "config_name",
            "run_name",
            "run_identity_sha256",
            "output_root",
            "run_dir",
            "evaluation_dir",
            "arm_plan",
            "slurm_log_dir",
            "parent_snapshot",
            "parent_resolved_config",
            "train_manifest",
            "train_cache_metadata",
            "python",
            "wan_dir",
            "videox_home",
            "wandb_enabled",
        )
        fields = [str(values[key]) for key in order]
        if any("\t" in value or "\n" in value for value in fields):
            raise IPQWorkflowError("TSV value contains a delimiter")
        print("\t".join(fields))
    return 0


def command_write_arm_plan(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    arm = _arm(args.arm)
    values = arm_values(registration, arm)
    plan_path = Path(values["arm_plan"])
    if plan_path.exists() or plan_path.is_symlink():
        raise IPQWorkflowError("arm execution plan already exists")
    plan_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    Path(values["slurm_log_dir"]).mkdir(parents=True, mode=0o700, exist_ok=True)
    payload = pilot.identity_payload(
        {
            "kind": "ipq_tc1_arm_execution_plan",
            "registration_identity_sha256": registration["identity_sha256"],
            "source_commit": registration["tool_repository"]["git_commit"],
            "arm": asdict(arm),
            "run_identity_sha256": values["run_identity_sha256"],
            "paths": {
                key: values[key]
                for key in (
                    "output_root",
                    "run_dir",
                    "evaluation_dir",
                    "slurm_log_dir",
                )
            },
            "training": {
                "updates": 400,
                "world_size": 8,
                "global_batch_size": 8,
                "wan_calls_per_update": 1,
                "endpoint_data_reachable": False,
            },
            "evaluation": {
                "world_size": 8,
                "batch_size_per_rank": 2,
                "nfe_grid": list(pilot.NFE_GRID),
                "noise_seeds": list(pilot.NOISE_SEEDS),
                "endpoints": [
                    asdict(value) for value in pilot.ENDPOINTS_BY_ARM[arm.code]
                ],
                "global_endpoint_barrier_before_full_rgb": True,
            },
            "wandb": {
                "enabled": values["wandb_enabled"],
                "entity": "zijiandu",
                "project": "dual-video-diffusion-private",
                "group": None,
                "id": values["run_identity_sha256"],
                "resume": "never",
            },
            "resource_plan": registration["resource_plan"],
            "jobs_submitted_by_workflow": 0,
        }
    )
    pilot.exclusive_json(plan_path, payload)
    print(str(plan_path))
    return 0


def command_wandb_check(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    if not registration["wandb"]["enabled"]:
        print(json.dumps({"enabled": False, "mode": "disabled", "group": None}))
        return 0
    result = dual_abc_pilot._wandb_private_project(
        "zijiandu", "dual-video-diffusion-private"
    )
    if (
        result.get("access") != "PRIVATE"
        or result.get("viewer_username") != "zijiandu"
    ):
        raise IPQWorkflowError("private personal W&B project check failed")
    print(json.dumps({**result, "enabled": True, "group": None}, sort_keys=True))
    return 0


def _train_command(
    registration_path: Path,
    registration: Mapping[str, Any],
    arm: pilot.Arm,
) -> list[str]:
    values = arm_values(registration, arm)
    repo = Path(registration["tool_repository"]["path"])
    command = [
        values["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(repo / "projects/latent_action_models/train_invertible_two_clock.py"),
        f"+experiments_0908={values['config_name']}",
        f"hydra.run.dir={values['run_dir']}",
        f"hydra.sweep.dir={values['run_dir']}",
        f"trainer.config.saving.save_path={values['run_dir']}/snapshot.pt",
    ]
    if values["wandb_enabled"]:
        command.extend(
            (
                "wandb.enabled=true",
                "wandb.mode=online",
                "wandb.entity=zijiandu",
                "wandb.project=dual-video-diffusion-private",
                "wandb.group=null",
                f"+wandb.id={values['run_identity_sha256']}",
                "+wandb.resume=never",
            )
        )
    return command


def _evaluation_command(
    registration_path: Path,
    registration: Mapping[str, Any],
    arm: pilot.Arm,
) -> list[str]:
    values = arm_values(registration, arm)
    repo = Path(registration["tool_repository"]["path"])
    return [
        values["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(repo / "tools/invertible_two_clock_evaluate.py"),
        "--registration",
        str(registration_path),
        "--arm",
        arm.code,
        "--output",
        values["evaluation_dir"],
    ]


def command_plan(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    root = Path(registration["output_root"])
    payload = {
        "kind": "ipq_tc1_dry_run_plan",
        "mode": "no_commands_executed_no_files_created",
        "source_commit": registration["tool_repository"]["git_commit"],
        "arms": [
            {
                "arm": arm.code,
                "train_argv": _train_command(
                    args.registration, registration, arm
                ),
                "train_shell": shlex.join(
                    _train_command(args.registration, registration, arm)
                ),
                "evaluate_argv": _evaluation_command(
                    args.registration, registration, arm
                ),
                "evaluate_shell": shlex.join(
                    _evaluation_command(args.registration, registration, arm)
                ),
                "submission_cwd": str(root / "_slurm_logs"),
            }
            for arm in pilot.ARMS
        ],
        "analysis_argv": [
            registration["runtime"]["python"],
            str(Path(registration["tool_repository"]["path"]) / "tools/slurm/invertible_two_clock_workflow.py"),
            "analyze",
            "--registration",
            str(args.registration),
        ],
        "resource_plan": pilot.resource_plan(),
        "jobs_submitted": 0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    root = Path(registration["output_root"])
    result = analyzer.analyze(
        root / "evaluation/ipq-sync/inventory.json",
        root / "evaluation/ipq-indep/inventory.json",
    )
    output = root / "analysis/analysis.json"
    output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    pilot.exclusive_json(output, result)
    print(str(output))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    values = commands.add_parser("arm-values")
    values.add_argument("--registration", type=Path, required=True)
    values.add_argument("--arm", choices=tuple(pilot.ARM_BY_CODE), required=True)
    values.add_argument("--format", choices=("json", "tsv"), default="json")
    values.add_argument("--require-training-fresh", action="store_true")
    values.add_argument("--require-evaluation-fresh", action="store_true")
    values.set_defaults(func=command_arm_values)
    arm_plan = commands.add_parser("write-arm-plan")
    arm_plan.add_argument("--registration", type=Path, required=True)
    arm_plan.add_argument("--arm", choices=tuple(pilot.ARM_BY_CODE), required=True)
    arm_plan.set_defaults(func=command_write_arm_plan)
    wandb = commands.add_parser("wandb-check")
    wandb.add_argument("--registration", type=Path, required=True)
    wandb.set_defaults(func=command_wandb_check)
    plan = commands.add_parser("plan")
    plan.add_argument("--registration", type=Path, required=True)
    plan.set_defaults(func=command_plan)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--registration", type=Path, required=True)
    analyze.set_defaults(func=command_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
