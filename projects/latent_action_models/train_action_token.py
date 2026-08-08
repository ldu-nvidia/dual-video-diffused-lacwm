"""Dedicated Hydra entrypoint for the paired action-token VPM screen."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import hydra
import numpy as np
import torch
from custom_resolvers import *  # noqa: F403
from omegaconf import DictConfig

import robot_wm.utils.distributed as dist
from robot_wm.utils.action_token_trainer import ActionTokenTrainer
from tools import action_token_screen as screen
from tools import two_clock_consistency_evaluate as evidence

logger = logging.getLogger(__name__)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _validate_launch_contract(cfg: DictConfig) -> str:
    registration_value = os.environ.get("ACTION_TOKEN_REGISTRATION", "")
    arm_code = os.environ.get("ACTION_TOKEN_ARM_CODE", "")
    if not registration_value or not Path(registration_value).is_absolute():
        raise RuntimeError("ACTION_TOKEN_REGISTRATION must be an absolute file")
    registration = screen.validate_registration(Path(registration_value))
    screen.revalidate_execution_environment(registration)
    arm = screen.ARM_BY_CODE.get(arm_code)
    if arm is None:
        raise RuntimeError("ACTION_TOKEN_ARM_CODE is absent or invalid")
    expected_identity = screen.arm_run_identity(registration, arm)
    plan_path = (
        Path(registration["output_root"]) / "arm_plans" / f"{arm.code.lower()}.json"
    )
    if not plan_path.is_file() or plan_path.is_symlink():
        raise RuntimeError("action-token arm plan is not a regular file")
    plan = evidence._read_json(plan_path, "action-token arm plan")
    observed_contract = screen.canonical_config_contract(cfg)
    runtime_verification = screen.validate_runtime_receipt(
        registration, arm, require_current_slurm=True
    )
    registered_inputs = screen.validate_registered_input_revalidation(
        plan.get("input_revalidation"), registration
    )
    if (
        not screen.identity_valid(plan)
        or plan.get("kind") != "action_token_arm_execution_plan"
        or plan.get("status") != "planned_before_arm_training_or_metrics"
        or plan.get("arm") != asdict(arm)
        or plan.get("registration_identity_sha256") != registration["identity_sha256"]
        or plan.get("run_identity_sha256") != expected_identity
        or plan.get("resolved_config_contract") != observed_contract
        or plan.get("input_revalidation") != registered_inputs
        or plan.get("runtime_verification") != runtime_verification
        or plan.get("evaluation", {}).get("endpoints")
        != screen.endpoint_records()
        or os.environ.get("LACWM_RUN_IDENTITY_SHA256") != expected_identity
        or not bool(cfg.wandb.enabled)
        or str(cfg.wandb.id) != expected_identity
        or str(cfg.wandb.resume) != "never"
    ):
        raise RuntimeError("resolved launch config/identity differs from arm plan")
    # These values are derived only after every rank has validated the immutable
    # pre-torchrun plan. The trainer can therefore avoid rank-zero hashing of the
    # multi-GB parent after NCCL has been initialized.
    os.environ["ACTION_TOKEN_PARENT_PREFLIGHT_SHA256"] = (
        screen.PARENT_SNAPSHOT_SHA256
    )
    os.environ["ACTION_TOKEN_RUNTIME_RECEIPT_BINDING_JSON"] = json.dumps(
        runtime_verification, sort_keys=True, separators=(",", ":")
    )
    return expected_identity


def _teardown(trainer: ActionTokenTrainer) -> None:
    failures = []

    def destroy() -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    for name, cleanup in (
        ("data-loader shutdown", trainer.shutdown_data_loaders),
        ("W&B finalization", trainer.finalize_wandb),
        ("distributed teardown", destroy),
    ):
        try:
            cleanup()
        except BaseException as exc:
            failures.append((name, exc))
            logger.error("%s failed", name, exc_info=True)
    if failures:
        raise failures[0][1]


def _setup(cfg: DictConfig) -> ActionTokenTrainer:
    expected_identity = _validate_launch_contract(cfg)
    dist.init_process_group()
    _seed_all(int(cfg.seed))
    trainer = None
    try:
        trainer = hydra.utils.instantiate(cfg.trainer)
        if not isinstance(trainer, ActionTokenTrainer):
            raise RuntimeError("entrypoint requires ActionTokenTrainer")
        trainer.initialize_wandb(cfg)
        observed_wandb_id = trainer.wandb_run_id if trainer.is_main_process else None
        observed_ids = [observed_wandb_id]
        torch.distributed.broadcast_object_list(observed_ids, src=0)
        if (
            bool(trainer.use_wandb) is not bool(trainer.is_main_process)
            or observed_ids[0] != expected_identity
            or (trainer.is_main_process and trainer.wandb_run_id != expected_identity)
            or trainer.run_identity_sha256 != expected_identity
        ):
            raise RuntimeError("actual W&B/run identity differs from registered arm")
        _seed_all(int(cfg.seed) + dist.get_global_rank())
        trainer.start_data_loaders()
        return trainer
    except BaseException:
        if trainer is not None:
            try:
                _teardown(trainer)
            except BaseException:
                logger.error("setup teardown also failed", exc_info=True)
        raise


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    torch.multiprocessing.set_start_method("spawn", force=True)
    trainer = None
    try:
        trainer = _setup(cfg)
        trainer.train()
    except BaseException:
        logger.error(traceback.format_exc())
        try:
            if trainer is not None:
                _teardown(trainer)
                trainer = None
            elif dist.is_initialized():
                dist.destroy_process_group()
        except BaseException:
            logger.error("teardown also failed", exc_info=True)
        raise
    else:
        _teardown(trainer)


if __name__ == "__main__":
    sys.exit(main())
