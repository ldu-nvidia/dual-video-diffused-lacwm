"""Dedicated Hydra entrypoint for the prospective ACD-P0 matched arms."""

from __future__ import annotations

import logging
import random
import sys
import traceback

import hydra
import numpy as np
import torch
from custom_resolvers import *  # noqa: F403
from omegaconf import DictConfig

import robot_wm.utils.distributed as dist
from robot_wm.utils.adjacent_consistency_trainer import (
    AdjacentConsistencyTrainer,
    validate_runtime_resolved_config,
)


logger = logging.getLogger(__name__)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _teardown(trainer: AdjacentConsistencyTrainer) -> None:
    failures = []
    for name, cleanup in (
        ("data-loader shutdown", trainer.shutdown_data_loaders),
        ("W&B finalization", trainer.finalize_wandb),
        (
            "distributed process-group teardown",
            lambda: dist.destroy_process_group() if dist.is_initialized() else None,
        ),
    ):
        try:
            cleanup()
        except BaseException as exc:
            failures.append((name, exc))
            logger.error("%s failed", name, exc_info=True)
    if failures:
        raise failures[0][1]


def _setup(cfg: DictConfig) -> AdjacentConsistencyTrainer:
    if bool(cfg.wandb.enabled):
        raise RuntimeError("ACD-P0 prepared source fixes W&B disabled")
    dist.init_process_group()
    _seed_all(int(cfg.seed))
    trainer = None
    try:
        trainer = hydra.utils.instantiate(cfg.trainer)
        if not isinstance(trainer, AdjacentConsistencyTrainer):
            raise RuntimeError(
                "ACD-P0 entrypoint requires AdjacentConsistencyTrainer"
            )
        trainer.initialize_wandb(cfg)
        # Construction/full-state hashing cannot shift paired corruption RNG.
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
    validate_runtime_resolved_config(cfg)
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
