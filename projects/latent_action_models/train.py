import logging
import random
import sys
import traceback

import hydra
import numpy as np
import torch
from custom_resolvers import *
from omegaconf import DictConfig

import robot_wm.utils.distributed as dist
from robot_wm.utils.trainer import Trainer

logger = logging.getLogger(__name__)


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _setup(cfg: DictConfig) -> Trainer:
    # initialize distributed process group
    dist.init_process_group()

    # set random seeds
    _seed_all(cfg.seed)

    # initialize trainer
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    # initialize wandb
    trainer.initialize_wandb(cfg)

    # Keep model construction deterministic, then give every rank independent
    # diffusion noise/timesteps/dropout and worker base seeds. A resumed loader
    # must be started from its checkpoint RNG, after which the main-process RNG
    # is restored again to undo iterator/W&B construction.
    if trainer.resumed:
        trainer.restore_resumed_rng_state()
        trainer.start_data_loaders()
        trainer.restore_resumed_rng_state()
    else:
        _seed_all(cfg.seed + dist.get_global_rank())
        trainer.start_data_loaders()

    return trainer


def _teardown(trainer: Trainer):
    try:
        trainer.finalize_wandb()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # CUDA/NCCL are initialized before loader iterators in this program. Spawned
    # workers do not inherit that accelerator/process-group state as forked
    # workers would.
    torch.multiprocessing.set_start_method("spawn", force=True)
    if cfg.debug:
        try:
            import debugpy
        except ImportError as exc:
            raise RuntimeError(
                "debug=true requires the optional 'debugpy' package"
            ) from exc
        logger.info(f"{cfg = }")
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()

    trainer = None
    try:
        trainer = _setup(cfg)
        trainer.train()
    except Exception:
        # Do not convert a failed rank into an apparently successful torchrun.
        # The non-zero exit code is required by launchers and job schedulers to
        # stop, alert, and resume from the last atomic snapshot.
        logger.error(traceback.format_exc())
        raise
    finally:
        if trainer is not None:
            _teardown(trainer)
        elif dist.is_initialized():
            # Model/data construction can fail after process-group creation but
            # before a Trainer object is returned.
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
