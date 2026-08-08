"""Dedicated Hydra entrypoint for the video residual-anchor screen."""

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
from robot_wm.utils.video_residual_anchor_trainer import VideoResidualAnchorTrainer


logger = logging.getLogger(__name__)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _setup(cfg: DictConfig) -> VideoResidualAnchorTrainer:
    dist.init_process_group()
    _seed_all(int(cfg.seed))
    trainer = None
    try:
        trainer = hydra.utils.instantiate(cfg.trainer)
        if not isinstance(trainer, VideoResidualAnchorTrainer):
            raise RuntimeError("entrypoint requires VideoResidualAnchorTrainer")
        trainer.initialize_wandb(cfg)
        # Model construction and W&B setup cannot shift paired corruption RNG.
        _seed_all(int(cfg.seed) + dist.get_global_rank())
        trainer.start_data_loaders()
        return trainer
    except BaseException:
        if trainer is not None:
            try:
                _teardown(trainer)
            except BaseException:
                logger.error(
                    "Trainer teardown also failed during setup failure",
                    exc_info=True,
                )
        raise


def _teardown(trainer: VideoResidualAnchorTrainer) -> None:
    failures = []

    def destroy_process_group() -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    for name, cleanup in (
        ("data-loader shutdown", trainer.shutdown_data_loaders),
        ("W&B finalization", trainer.finalize_wandb),
        ("distributed process-group teardown", destroy_process_group),
    ):
        try:
            cleanup()
        except BaseException as exc:
            failures.append((name, exc))
            logger.error("%s failed", name, exc_info=True)
    if failures:
        raise failures[0][1]


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
            logger.error(
                "Teardown also failed while preserving the primary error",
                exc_info=True,
            )
        raise
    else:
        _teardown(trainer)


if __name__ == "__main__":
    sys.exit(main())
