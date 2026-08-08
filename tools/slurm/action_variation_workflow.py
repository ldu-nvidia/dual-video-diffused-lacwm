#!/usr/bin/env python3
"""Render immutable commands for the paired action-variation screen.

This helper validates and plans; it never submits a job.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import action_variation_screen as screen  # noqa: E402
from tools import analyze_action_variation as analysis  # noqa: E402
from tools import dual_abc_pilot  # noqa: E402
from tools import two_clock_consistency_evaluate as base  # noqa: E402


KIND_PLAN = "action_variation_arm_execution_plan"


class ActionVariationWorkflowError(RuntimeError):
    """A launch plan is stale, non-fresh, or differs from registration."""


def _registration(path: Path) -> dict[str, Any]:
    value = screen.validate_registration(path)
    source = value["tool_repository"]
    if base._clean_source(Path(source["path"]), source["git_commit"], "tool") != source:
        raise ActionVariationWorkflowError("tool source changed after registration")
    return value


def _arm(code: str) -> screen.Arm:
    arm = screen.ARM_BY_CODE.get(code)
    if arm is None:
        raise ActionVariationWorkflowError("unknown arm")
    return arm


def arm_values(registration: dict[str, Any], arm: screen.Arm) -> dict[str, str]:
    root = Path(registration["output_root"])
    run_identity = screen.arm_run_identity(registration, arm)
    return {
        "arm_code": arm.code,
        "config_name": arm.config_name,
        "run_name": arm.run_name,
        "run_identity": run_identity,
        "training_root": str(root / "training"),
        "run_dir": str(root / "training" / arm.run_name),
        "evaluation_dir": str(root / "evaluation" / arm.code.lower()),
        "plan": str(root / "arm_plans" / f"{arm.code.lower()}.json"),
        "parent_snapshot": registration["controlled_study"]["parent_snapshot"]["path"],
        "train_manifest": registration["training"]["manifest"]["path"],
        "train_metadata": registration["training"]["cache_metadata"]["path"],
        "validation_manifest": registration["validation"]["manifest"]["path"],
        "validation_metadata": registration["validation"]["cache_metadata"]["path"],
        "stats": registration["action_delta_stats"]["file"]["path"],
        "stats_sha256": registration["action_delta_stats"]["file"]["sha256"],
        "python": registration["runtime"]["python"],
        "wan_dir": registration["runtime"]["wan_dir"],
        "videox_home": registration["runtime"]["videox_home"],
    }


def _train_argv(values: dict[str, str], repo: Path) -> list[str]:
    return [
        values["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(repo / "projects" / "latent_action_models" / "train_action_variation.py"),
        f"+experiments_0908={values['config_name']}",
        f"hydra.run.dir={values['run_dir']}",
        f"hydra.sweep.dir={values['run_dir']}",
        f"trainer.config.saving.save_path={values['run_dir']}/snapshot.pt",
        "wandb.enabled=true",
        "wandb.mode=online",
        "wandb.entity=zijiandu",
        "wandb.project=dual-video-diffusion-private",
        "wandb.group=null",
        f"+wandb.id={values['run_identity']}",
        "+wandb.resume=never",
    ]


def _evaluation_argv(
    values: dict[str, str], repo: Path, registration: Path
) -> list[str]:
    return [
        values["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        str(repo / "tools" / "action_variation_evaluate.py"),
        "evaluate",
        "--registration",
        str(registration.resolve(strict=True)),
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
    payload = {
        "registration_identity_sha256": registration["identity_sha256"],
        "required_environment": {
            "WAN_DIR": registration["runtime"]["wan_dir"],
            "VIDEOX_HOME": registration["runtime"]["videox_home"],
            "ACTION_VARIATION_VPM_SNAPSHOT_SHA256": screen.PARENT_SNAPSHOT_SHA256,
            "WANDB_ENTITY": "zijiandu",
            "WANDB_PROJECT": "dual-video-diffusion-private",
            "WANDB_MODE": "online",
            "WANDB_RUN_GROUP": None,
        },
        "arms": [],
    }
    for arm in screen.ARMS:
        values = arm_values(registration, arm)
        payload["arms"].append(
            {
                "values": values,
                "train_argv": _train_argv(values, repo),
                "train_shell": shlex.join(_train_argv(values, repo)),
                "evaluation_argv": _evaluation_argv(values, repo, args.registration),
                "evaluation_shell": shlex.join(
                    _evaluation_argv(values, repo, args.registration)
                ),
                "per_arm_environment": {
                    "ACTION_VARIATION_VPM_SNAPSHOT": values["parent_snapshot"],
                    "ACTION_VARIATION_TRAIN_CLIP_MANIFEST": values["train_manifest"],
                    "ACTION_VARIATION_TRAIN_CACHE_METADATA": values["train_metadata"],
                    "ACTION_VARIATION_VAL_CLIP_MANIFEST": values["validation_manifest"],
                    "ACTION_VARIATION_VAL_CACHE_METADATA": values[
                        "validation_metadata"
                    ],
                    "ACTION_VARIATION_STATS": values["stats"],
                    "ACTION_VARIATION_STATS_SHA256": values["stats_sha256"],
                    "ACTION_VARIATION_RUN_ROOT": values["training_root"],
                    "LACWM_RUN_IDENTITY_SHA256": values["run_identity"],
                },
            }
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_write_arm_plan(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    arm = _arm(args.arm)
    values = arm_values(registration, arm)
    for key in ("run_dir", "evaluation_dir"):
        path = Path(values[key])
        if path.exists() or path.is_symlink():
            raise ActionVariationWorkflowError(f"fresh output exists: {path}")
    revalidated = screen.revalidate_registered_inputs(registration)
    plan = screen.identity_payload(
        {
            "schema_version": 1,
            "kind": KIND_PLAN,
            "status": "planned_before_arm_training_or_metrics",
            "registration_identity_sha256": registration["identity_sha256"],
            "arm": {
                "code": arm.code,
                "config_name": arm.config_name,
                "run_name": arm.run_name,
                "residual_enabled": arm.residual_enabled,
            },
            "run_identity_sha256": values["run_identity"],
            "paths": values,
            "input_revalidation": revalidated,
            "training": {
                "updates": 200,
                "seed": 1234,
                "world_size": 8,
                "global_batch_size": 8,
                "wan_calls_per_update": 1,
                "fresh_identical_optimizer": True,
                "same_schema_and_forward_compute": True,
            },
            "evaluation": {
                "validation_clips": 64,
                "endpoints": [
                    {
                        "code": endpoint.code,
                        "nfe": endpoint.nfe,
                        "action_source": endpoint.action_source,
                        "primary_gate": endpoint.primary_gate,
                    }
                    for endpoint in screen.ENDPOINTS
                ],
                "protected_test_accessed": False,
                "future_or_clean_feature_used_at_sampling": False,
            },
            "wandb": {
                "entity": "zijiandu",
                "project": "dual-video-diffusion-private",
                "group": None,
                "mode": "online",
                "id": values["run_identity"],
                "resume": "never",
            },
        }
    )
    path = Path(values["plan"])
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    base._exclusive_json(path, plan)
    print(json.dumps(plan, sort_keys=True))
    return 0


def command_wandb_check(_args: argparse.Namespace) -> int:
    result = dual_abc_pilot._wandb_private_project(
        "zijiandu", "dual-video-diffusion-private"
    )
    if result.get("access") != "PRIVATE" or result.get("viewer_username") != "zijiandu":
        raise ActionVariationWorkflowError("W&B personal project privacy check failed")
    print(json.dumps({**result, "group": None}, sort_keys=True))
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    registration = _registration(args.registration)
    root = Path(registration["output_root"])
    return analysis.command_analyze(
        argparse.Namespace(
            control_inventory=root / "evaluation" / "av-cont" / "inventory.json",
            candidate_inventory=root / "evaluation" / "av-delta" / "inventory.json",
            output=root / "analysis.json",
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (
        ("plan", command_plan),
        ("wandb-check", command_wandb_check),
        ("analyze", command_analyze),
    ):
        value = sub.add_parser(name)
        if name != "wandb-check":
            value.add_argument("--registration", type=Path, required=True)
        value.set_defaults(func=func)
    write = sub.add_parser("write-arm-plan")
    write.add_argument("--registration", type=Path, required=True)
    write.add_argument("--arm", choices=tuple(screen.ARM_BY_CODE), required=True)
    write.set_defaults(func=command_write_arm_plan)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
