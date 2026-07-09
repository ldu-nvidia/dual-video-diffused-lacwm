# References:
#     https://github.com/pytorch/examples/blob/5dfeb46902baf444010f2f54bcf4dfbea109ae4d/distributed/ddp-tutorial-series/multinode.py
#     https://github.com/pytorch/torchtitan/blob/b345557a37d7d8804e3b1cf8e9e0a36e46e689e9/torchtitan/train.py
import hashlib
import io
import json
import logging
import os
import random
import signal
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from numbers import Integral
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn
import wandb
from einops import rearrange
from omegaconf import DictConfig
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchdata.stateful_dataloader import StatefulDataLoader

import robot_wm.utils.distributed as dist
from robot_wm.modeling.modules.lora import LoraWrapper
from robot_wm.utils.model import get_model_size, log_model_size
from robot_wm.utils.partial import fix_partial
from robot_wm.utils.wandb import init_wandb_from_config
from robot_wm.modeling.modules.video_blocks import STAttentionBlock

logger = logging.getLogger(__name__)

# Average loss across all validation datasets
AVG_LOSS_KEY = "avg/loss"
TOPOLOGY_MIGRATION_KIND = "topology_migration_reset_rank_state"


@dataclass
class Metrics:
    world_size: int
    batch_size: int
    last_step_time: float = field(default_factory=time.time)
    samples_since_log: int = 0
    time_since_log: float = 0.0
    total_observations: int = 0
    warmup: int = 100
    best_val_loss: float = float("inf")

    def update(self, microbatches: int = 1, step_time: float | None = None):
        current_time = time.time()
        if step_time is None:
            step_time = current_time - self.last_step_time
        self.last_step_time = current_time
        self.time_since_log += step_time

        samples = self.batch_size * self.world_size * microbatches
        self.samples_since_log += samples
        self.total_observations += samples

    def get_train_metrics(self, iter_num, losses, operational_metrics=None):
        samples_per_second = 0
        if self.time_since_log > 0 and iter_num >= self.warmup:
            samples_per_second = self.samples_since_log / self.time_since_log

        metrics = {
            "iteration": iter_num,
            "samples_per_second": samples_per_second,
            "total_observations": self.total_observations,
        }

        for key, value in losses.items():
            metrics[f"train_loss/{key}"] = value

        if operational_metrics:
            metrics.update(operational_metrics)

        if logger.getEffectiveLevel() <= logging.DEBUG:
            metrics.update(
                {
                    "gpu_memory_allocated_gb": torch.cuda.max_memory_allocated()
                    / 1024**3,
                    "gpu_memory_reserved_gb": torch.cuda.max_memory_reserved()
                    / 1024**3,
                }
            )

        return metrics

    def get_val_metrics(self, iter_num, val_losses):
        metrics = {
            "iteration": iter_num,
            "total_observations": self.total_observations,
        }

        for key, value in val_losses.items():
            metrics[f"val_loss/{key}"] = value
        for dataset_name, value in self.best_validation_losses().items():
            metrics[f"val_loss/best/{dataset_name}"] = value
        metrics["val_loss/best_val_loss"] = self.best_val_loss

        return metrics

    def best_validation_losses(self) -> dict[str, float]:
        prefix = "best_val_loss_"
        return {
            key.removeprefix(prefix): float(value)
            for key, value in sorted(vars(self).items())
            if key.startswith(prefix)
        }

    def refresh_best_val_loss(self) -> float:
        """Keep the legacy aggregate best metric finite and meaningful.

        Multi-dataset validation emits ``avg/loss``.  Its historical minimum is
        the best jointly observed validation point, so it is a better aggregate
        than averaging per-dataset minima reached at potentially different
        iterations.  A single-dataset run falls back to that dataset's best.
        """
        best_losses = self.best_validation_losses()
        if not best_losses:
            return self.best_val_loss
        if "avg" in best_losses:
            self.best_val_loss = best_losses["avg"]
        else:
            self.best_val_loss = sum(best_losses.values()) / len(best_losses)
        return self.best_val_loss

    def reset_throughput_counters(self):
        self.samples_since_log = 0
        self.time_since_log = 0.0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer_factory: partial[Optimizer],
        lr_scheduler_factory: partial[LRScheduler],
        data_loader: StatefulDataLoader,
        val_data_loader: StatefulDataLoader,
        viz_data_loader: StatefulDataLoader,
        config: DictConfig,
    ):
        self.local_rank = dist.get_local_rank()
        self.global_rank = dist.get_global_rank()
        self.world_size = dist.get_world_size()
        self.is_main_process = self.global_rank == 0
        self.max_iter = config["max_iter"]
        self.log_every = config["logging"]["log_every"]
        self.val_every = config["validation"]["val_every"]
        self.n_val_samples = config["validation"]["n_val_samples"]
        self.save_best = self._parse_save_best(config)
        self.viz_every = config["visualization"]["viz_every"]
        self.viz_path = Path(config["visualization"]["viz_path"])
        self.save_every = config["saving"]["save_every"]
        self.save_path = Path(config["saving"]["save_path"])
        self.gradient_accumulation_steps = self._parse_gradient_accumulation_steps(
            config
        )
        self._start_iter = 0
        self._curr_iter = 0
        self.resumed = False
        self.transitioned = False
        self._resume_rng_state = None
        self.transition_parent = None
        self._checkpoint_stop_requested = False
        self._checkpoint_stop_signal = None
        request_path = os.environ.get("LACWM_CHECKPOINT_REQUEST_FILE")
        ack_path = os.environ.get("LACWM_CHECKPOINT_ACK_FILE")
        self.checkpoint_request_path = Path(request_path) if request_path else None
        self.checkpoint_ack_path = Path(ack_path) if ack_path else None
        self.completion_path = self.save_path.parent / "training_complete.json"
        self._previous_signal_handlers = {}
        # The guarded launcher supplies a digest over the immutable run
        # configuration, data fingerprints, runtime, and assets.  Storing it in
        # every checkpoint prevents an unrelated snapshot with compatible tensor
        # shapes from being silently resumed under this run's identity.
        self.run_identity_sha256 = os.environ.get("LACWM_RUN_IDENTITY_SHA256")
        transition_handoff = config.get("transition_handoff_path")
        self.transition_handoff_path = (
            Path(transition_handoff) if transition_handoff is not None else None
        )

        # daniel: I hate this but no better solution in reasonable time
        if hasattr(model, "custom_to"):
            model = model.custom_to(self.local_rank)

        # LORA
        self.lora_finetuning = config.get("lora_finetuning", False)
        if self.lora_finetuning:
            self.lora_config = config.get("lora_config", {})
            assert self.lora_config, "LORA config must be provided for LORA finetuning"
            assert (
                len(self.lora_config) > 0
            ), "LORA config must be provided for LORA finetuning"

            # Freeze all parameters in the model
            for p in model.parameters():
                p.requires_grad = False

            # Wrap the model with LORA
            model = LoraWrapper(model, self.lora_config)
            log_model_size(model, logger.info)

        self.model = DDP(
            model.to(self.local_rank),
            device_ids=[self.local_rank],
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        self.optimizer = fix_partial(optimizer_factory)(self.model.parameters())
        self.lr_scheduler = fix_partial(lr_scheduler_factory)(self.optimizer)

        # Data loaders
        self.data_loader = data_loader
        if isinstance(val_data_loader, StatefulDataLoader):
            val_data_loader = [val_data_loader]
        if isinstance(viz_data_loader, StatefulDataLoader):
            viz_data_loader = [viz_data_loader]
        self.val_data_loaders = val_data_loader
        self.viz_data_loaders = viz_data_loader
        self.batch_size = data_loader.batch_size

        # AMP and Grad Clipping
        self.dtype = self._get_torch_dtype(config["dtype"])
        self.use_amp = config["amp_enabled"]
        self.scaler = GradScaler("cuda", enabled=self.use_amp) if self.use_amp else None
        self.max_norm = config["gradient_clipping"]["max_norm"]
        self.norm_type = config["gradient_clipping"]["norm_type"]
        self.error_if_nonfinite = config["gradient_clipping"]["error_if_nonfinite"]

        # Metrics tracking
        self.metrics = Metrics(
            self.world_size,
            self.batch_size,
            warmup=self._parse_throughput_warmup_steps(config),
        )
        self._last_operational_metrics = {}

        # Wandb
        self.use_wandb = False
        self.wandb_run_id = None

        # Load checkpoint if exists
        if self.save_path.exists():
            self._load_snapshot()
        elif self.transition_handoff_path is not None:
            self._load_transition_snapshot(self.transition_handoff_path)
        logger.info(f"Initialized trainer; save path: {self.save_path}")

        if (
            "load_path" in config
            and config["load_path"] is not None
            and Path(config["load_path"]).exists()
        ):
            self._load_model_snapshot(
                config["load_path"], config.get("exclude_keys", []), config.get("share_spatial_attention", False)
            )
            logger.info(f"loaded model state from {config['load_path']}")

        # Worker processes are started explicitly by start_data_loaders() only
        # after train.py assigns a rank-local RNG stream. This also avoids
        # silently constructing an iterator before restored loader state exists.
        self._data_loader_iter = None
        self._val_data_loader_iters = None
        self._viz_data_loader_iters = None

    @staticmethod
    def _parse_gradient_accumulation_steps(config) -> int:
        value = config.get("gradient_accumulation_steps", 1)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(
                "gradient_accumulation_steps must be a positive integer, "
                f"got {value!r}"
            )
        return int(value)

    @staticmethod
    def _parse_throughput_warmup_steps(config) -> int:
        value = config.get("throughput_warmup_steps", 100)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(
                "throughput_warmup_steps must be a nonnegative integer, "
                f"got {value!r}"
            )
        return int(value)

    @staticmethod
    def _parse_save_best(config) -> bool:
        value = config.get("validation", {}).get("save_best", True)
        if not isinstance(value, bool):
            raise ValueError(
                "validation.save_best must be a boolean, "
                f"got {value!r}"
            )
        return value

    def _distributed_weighted_loss_means(
        self,
        local_sums: dict[str, torch.Tensor],
        local_counts: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        """Return deterministic global means for a dynamic metric dictionary.

        Conditional losses (for example a morphology-specific decoder loss)
        are not necessarily present on every rank or every microbatch.  All
        ranks first agree on the sorted union of keys, then reduce a sum and a
        contribution count for every key in one collective.  An absent key has
        count zero, so it never dilutes the mean as if it were a zero-valued
        observation.
        """
        if set(local_sums) != set(local_counts):
            raise ValueError("loss sums and counts must contain identical keys")
        for key, value in local_sums.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"loss metric key must be str, got {type(key).__name__}"
                )
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise ValueError(f"loss metric {key!r} must be a scalar tensor")
            count = local_counts[key]
            if (
                isinstance(count, bool)
                or not isinstance(count, Integral)
                or count < 1
            ):
                raise ValueError(
                    f"loss metric {key!r} must have a positive contribution count"
                )

        local_keys = tuple(sorted(local_sums))
        if not dist.is_initialized() or self.world_size == 1:
            return {
                key: local_sums[key] / local_counts[key] for key in local_keys
            }

        gathered_keys = [None] * self.world_size
        torch.distributed.all_gather_object(gathered_keys, local_keys)
        global_keys = set()
        for rank, rank_keys in enumerate(gathered_keys):
            if not isinstance(rank_keys, (tuple, list)) or not all(
                isinstance(key, str) for key in rank_keys
            ):
                raise RuntimeError(
                    f"rank {rank} returned an invalid loss-metric key collection"
                )
            global_keys.update(rank_keys)
        ordered_keys = tuple(sorted(global_keys))
        if not ordered_keys:
            return {}
        if not local_sums:
            raise RuntimeError(
                "distributed loss reduction needs one local tensor to select a device"
            )

        reference = next(iter(local_sums.values()))
        packed = torch.zeros(
            (len(ordered_keys), 2),
            device=reference.device,
            dtype=torch.float64,
        )
        for index, key in enumerate(ordered_keys):
            if key in local_sums:
                packed[index, 0] = local_sums[key].detach().to(dtype=torch.float64)
                packed[index, 1] = local_counts[key]

        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        if (packed[:, 1] <= 0).any():
            missing = [
                key
                for key, count in zip(ordered_keys, packed[:, 1].tolist())
                if count <= 0
            ]
            raise RuntimeError(
                f"loss metrics have no contributing samples after reduction: {missing}"
            )
        means = packed[:, 0] / packed[:, 1]
        return {key: means[index] for index, key in enumerate(ordered_keys)}

    def start_data_loaders(self):
        if self._data_loader_iter is not None:
            raise RuntimeError("data-loader iterators have already been started")
        self._data_loader_iter = iter(self.data_loader)
        self._val_data_loader_iters = [iter(dl) for dl in self.val_data_loaders]
        self._viz_data_loader_iters = [iter(dl) for dl in self.viz_data_loaders]

    def _get_torch_dtype(self, dtype_str):
        if dtype_str == "float32":
            return torch.float32
        elif dtype_str == "float16":
            return torch.float16
        elif dtype_str == "bfloat16":
            return torch.bfloat16
        elif dtype_str == "float64":
            return torch.float64
        else:
            raise Exception("Unknown torch dtype")

    def _capture_rng_state(self):
        numpy_state = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": numpy_state[0],
                "state": torch.from_numpy(numpy_state[1].copy()),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(self.local_rank),
        }

    def _restore_rng_state(self, state):
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                numpy_state["state"].cpu().numpy(),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(state["torch_cpu"].cpu())
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device=self.local_rank)

    def restore_resumed_rng_state(self):
        """Undo RNG consumed by iterator/W&B reconstruction after checkpoint load."""
        if self._resume_rng_state is None:
            raise RuntimeError("no resumed RNG state is available")
        self._restore_rng_state(self._resume_rng_state)

    def _gather_rank_states(self):
        local_state = {
            "global_rank": self.global_rank,
            "rng": self._capture_rng_state(),
            "data_loader": self.data_loader.state_dict(),
            "val_data_loaders": [loader.state_dict() for loader in self.val_data_loaders],
            "viz_data_loaders": [loader.state_dict() for loader in self.viz_data_loaders],
        }
        if not dist.is_initialized() or self.world_size == 1:
            return [local_state]

        # ``all_gather_object`` internally serializes its input with torch.save.
        # Passing a nested state containing tensors directly can hit PyTorch's
        # legacy-storage deserializer (UntypedStorage has no ``dtype``). Serialize
        # the tensor-bearing state once ourselves, then gather plain bytes; the
        # collective's outer serialization no longer contains tensor storages.
        buffer = io.BytesIO()
        torch.save(local_state, buffer)
        local_payload = buffer.getvalue()
        payloads = [None] * self.world_size
        torch.distributed.all_gather_object(payloads, local_payload)

        states = []
        for rank, payload in enumerate(payloads):
            if not isinstance(payload, bytes):
                raise RuntimeError(
                    f"rank-state collective returned {type(payload).__name__} "
                    f"for rank {rank}, expected bytes"
                )
            states.append(
                torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
            )
        return states

    def _build_snapshot(self, rank_states):
        snapshot = {
            "snapshot_schema_version": 3,
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "_start_iter": self._curr_iter + 1,
            "_total_observations": self.metrics.total_observations,
            "best_val_loss": self.metrics.best_val_loss,
            "best_val_losses": {
                key: value
                for key, value in vars(self.metrics).items()
                if key.startswith("best_val_loss_")
            },
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "world_size": self.world_size,
            "rank_states": rank_states,
        }
        if self.run_identity_sha256 is not None:
            snapshot["run_identity_sha256"] = self.run_identity_sha256
        if getattr(self, "transition_parent", None) is not None:
            snapshot["transition_parent"] = dict(self.transition_parent)

        if self.use_amp:
            snapshot["scaler"] = self.scaler.state_dict()

        if self.wandb_run_id is not None:
            snapshot["wandb_run_id"] = self.wandb_run_id

        return snapshot

    def _checkpoint_signal_handler(self, signum, _frame):
        """Only record intent here; checkpointing from a signal handler is unsafe."""
        self._checkpoint_stop_requested = True
        self._checkpoint_stop_signal = int(signum)

    def _install_checkpoint_signal_handlers(self):
        self._previous_signal_handlers = {}
        for signum in (getattr(signal, "SIGUSR1", None), signal.SIGTERM):
            if signum is None:
                continue
            self._previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._checkpoint_signal_handler)

    def _restore_checkpoint_signal_handlers(self):
        for signum, handler in self._previous_signal_handlers.items():
            signal.signal(signum, handler)
        self._previous_signal_handlers = {}

    def _checkpoint_stop_requested_across_ranks(self):
        """Agree on a stop request at a boundary shared by every training rank."""
        local_requested = self._checkpoint_stop_requested
        if (
            self.is_main_process
            and self.checkpoint_request_path is not None
            and self.checkpoint_request_path.is_file()
        ):
            local_requested = True

        if dist.is_initialized() and self.world_size > 1:
            collective_device = (
                self.local_rank
                if str(torch.distributed.get_backend()).lower() == "nccl"
                else "cpu"
            )
            vote = torch.tensor(
                [int(local_requested)],
                dtype=torch.int32,
                device=collective_device,
            )
            torch.distributed.all_reduce(vote, op=torch.distributed.ReduceOp.MAX)
            local_requested = bool(vote.item())

        if local_requested:
            self._checkpoint_stop_requested = True
        return local_requested

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _write_checkpoint_ack(self, *, checkpoint_written: bool, next_iter: int):
        if not self.is_main_process or self.checkpoint_ack_path is None:
            return
        self._atomic_write_json(
            self.checkpoint_ack_path,
            {
                "schema_version": 1,
                "status": "checkpointed_for_reschedule",
                "checkpoint_written": checkpoint_written,
                "next_iter": int(next_iter),
                "max_iter": int(self.max_iter),
                "run_identity_sha256": self.run_identity_sha256,
                "slurm_attempt_id": os.environ.get("LACWM_SLURM_ATTEMPT_ID"),
                "snapshot": str(self.save_path.resolve(strict=False)),
                "signal": self._checkpoint_stop_signal,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _write_completion_marker(self):
        if not self.is_main_process:
            return
        self._atomic_write_json(
            self.completion_path,
            {
                "schema_version": 1,
                "status": "completed",
                "completed_updates": int(self.max_iter),
                "max_iter": int(self.max_iter),
                "run_identity_sha256": self.run_identity_sha256,
                "snapshot": str(self.save_path.resolve(strict=False)),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _save_snapshot(self, is_best=False, dataset_name=None):
        # Every rank participates so rank-local stochastic state is resumable.
        rank_states = self._gather_rank_states()
        error = None
        if self.is_main_process:
            try:
                snapshot = self._build_snapshot(rank_states)
                if is_best:
                    suffix = (
                        f".{dataset_name}.best.pt"
                        if dataset_name is not None
                        else ".best.pt"
                    )
                    self._atomic_save_snapshot(
                        snapshot, self.save_path.with_suffix(suffix)
                    )
                else:
                    # The live resume file is updated at every save cadence.
                    # Milestone archives are additional copies, never replacements.
                    self._atomic_save_snapshot(snapshot, self.save_path)
                    # Iteration zero is already represented by the live resume
                    # file. Avoid a second, potentially tens-of-GB full-model
                    # copy before the smoke/run has made any progress.
                    if self._curr_iter > 0 and self._curr_iter % 10000 == 0:
                        archive = self.save_path.with_suffix(f".{self._curr_iter}.pt")
                        self._atomic_save_snapshot(snapshot, archive)
            except Exception as exc:  # synchronize failure before any rank tears down
                error = f"{type(exc).__name__}: {exc}"

        if dist.is_initialized() and self.world_size > 1:
            result = [error]
            torch.distributed.broadcast_object_list(result, src=0)
            error = result[0]
        if error is not None:
            raise RuntimeError(f"checkpoint save failed on rank 0: {error}")

    def _atomic_save_snapshot(self, snapshot, save_path):
        # Write on the destination filesystem and atomically replace the live
        # snapshot.  A preemption or ENOSPC must not leave a corrupt resume file.
        tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        torch.save(snapshot, tmp_path)
        with tmp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, save_path)
        directory_fd = os.open(save_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        logger.info(f"Saved iteration {self._curr_iter:,} to {save_path}")

    def _load_model_snapshot(self, load_path, exclude_keys=None, share_spatial_attention=False):
        if exclude_keys is None:
            exclude_keys = []

        map_loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(load_path, map_location=map_loc, weights_only=True)

        state_dict = snapshot["model"]
        
        if share_spatial_attention:
            logger.info("Sharing spatial attention between video and spatial attention blocks!!!!!!!!!!!!!!")
            # -------------------------------------------------------
            # (1) Identify all STAttentionBlocks inside forward_model
            # -------------------------------------------------------
            st_prefixes = []
            for module_name, module in self.model.module.forward_model.named_modules():
                if isinstance(module, STAttentionBlock):
                    st_prefixes.append("forward_model." + module_name)

            # Example:
            # st_prefixes = [
            #   "forward_model.blocks.0",
            #   "forward_model.blocks.1",
            #   ...
            # ]

            # -------------------------------------------------------
            # (2) Apply renaming ONLY for these blocks
            # -------------------------------------------------------
            remapped_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for prefix in st_prefixes:
                    # rename only SpatialAttentionBlock → STAttentionBlock spatial_attn
                    if k.startswith(prefix + ".video_attn."):
                        new_k = k.replace(prefix + ".video_attn.", prefix + ".spatial_attn.")
                        break
                    if k.startswith(prefix + ".video_attn_norm."):
                        new_k = k.replace(prefix + ".video_attn_norm.", prefix + ".spatial_attn_norm.")
                        break
                remapped_state_dict[new_k] = v
            state_dict = remapped_state_dict

        # Remove excluded keys
        if exclude_keys:
            remove_list = []
            for k in state_dict.keys():
                if any(k.startswith(ex_key) for ex_key in exclude_keys):
                    remove_list.append(k)
            for k in remove_list:
                del state_dict[k]
            logger.info(f"Excluded keys from load: {remove_list}")

        missing, unexpected = self.model.module.load_state_dict(
            state_dict, strict=False
        )

        logger.info(f"Loaded model state from {load_path}")
        if missing:
            logger.info(f"Missing keys (expected if excluded): {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

    def _load_snapshot(self):
        map_loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(self.save_path, map_location=map_loc, weights_only=True)
        if snapshot.get("snapshot_schema_version") != 3:
            raise RuntimeError(
                "unsupported resume checkpoint schema; exact distributed resume "
                "requires snapshot_schema_version=3"
            )
        checkpoint_identity = snapshot.get("run_identity_sha256")
        if checkpoint_identity is not None and self.run_identity_sha256 is None:
            raise RuntimeError(
                "checkpoint is bound to a guarded run identity, but "
                "LACWM_RUN_IDENTITY_SHA256 is unset"
            )
        if self.run_identity_sha256 is not None and checkpoint_identity != self.run_identity_sha256:
            raise RuntimeError(
                "checkpoint/run identity mismatch: "
                f"checkpoint={checkpoint_identity!r}, "
                f"expected={self.run_identity_sha256!r}"
            )
        self.transition_parent = self._validate_transition_parent_metadata(
            snapshot.get("transition_parent")
        )
        checkpoint_accumulation_steps = int(
            snapshot.get("gradient_accumulation_steps", 1)
        )
        if checkpoint_accumulation_steps != self.gradient_accumulation_steps:
            raise RuntimeError(
                "checkpoint gradient-accumulation mismatch: "
                f"checkpoint={checkpoint_accumulation_steps}, "
                f"current={self.gradient_accumulation_steps}"
            )

        # Reject an incompatible fixed-world-size resume before mutating model,
        # optimizer, scheduler, metrics, or loader state. Exact continuation
        # requires one saved RNG/loader stream for every current global rank.
        cpu_snapshot = torch.load(self.save_path, map_location="cpu", weights_only=True)
        rank_states = cpu_snapshot.get("rank_states")
        saved_world_size = int(cpu_snapshot.get("world_size", -1))
        if not isinstance(rank_states, list) or saved_world_size != self.world_size:
            raise RuntimeError(
                "checkpoint lacks compatible per-rank RNG/data-loader state: "
                f"saved_world_size={saved_world_size}, current_world_size={self.world_size}"
            )
        if len(rank_states) != self.world_size:
            raise RuntimeError(
                f"checkpoint has {len(rank_states)} rank states for world size {self.world_size}"
            )
        saved_rank_order = [state.get("global_rank") for state in rank_states]
        expected_rank_order = list(range(self.world_size))
        if saved_rank_order != expected_rank_order:
            raise RuntimeError(
                "checkpoint rank states are missing, duplicated, or reordered: "
                f"saved={saved_rank_order}, expected={expected_rank_order}"
            )

        self.model.module.load_state_dict(snapshot["model"])

        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.lr_scheduler.load_state_dict(snapshot["lr_scheduler"])
        self.metrics.total_observations = snapshot["_total_observations"]

        if "best_val_loss" in snapshot:
            self.metrics.best_val_loss = snapshot["best_val_loss"]
        for key, value in snapshot.get("best_val_losses", {}).items():
            setattr(self.metrics, key, value)
        if self.use_amp and "scaler" in snapshot:
            self.scaler.load_state_dict(snapshot["scaler"])
        if "wandb_run_id" in snapshot:
            self.wandb_run_id = snapshot["wandb_run_id"]
            logger.info(f"Resuming wandb run: {self.wandb_run_id}")

        rank_state = rank_states[self.global_rank]
        self.data_loader.load_state_dict(rank_state["data_loader"])
        val_states = rank_state.get("val_data_loaders", [])
        viz_states = rank_state.get("viz_data_loaders", [])
        if len(val_states) != len(self.val_data_loaders) or len(viz_states) != len(self.viz_data_loaders):
            raise RuntimeError(
                "checkpoint validation/visualization loader count differs from current config"
            )
        for loader, state in zip(self.val_data_loaders, val_states):
            loader.load_state_dict(state)
        for loader, state in zip(self.viz_data_loaders, viz_states):
            loader.load_state_dict(state)
        self._resume_rng_state = rank_state["rng"]
        self._restore_rng_state(self._resume_rng_state)
        self._start_iter = snapshot["_start_iter"]
        self.resumed = True
        logger.info(f"Resuming from iteration {self._start_iter:,}")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _sha256_file_rank_zero(self, path: Path) -> str:
        """Hash one shared snapshot once, then give every rank the result."""
        error = None
        digest = None
        if not dist.is_initialized() or self.world_size == 1 or self.is_main_process:
            try:
                digest = self._sha256_file(path)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized() and self.world_size > 1:
            result = [digest, error]
            torch.distributed.broadcast_object_list(result, src=0)
            digest, error = result
        if error is not None:
            raise RuntimeError(f"unable to hash transition snapshot: {error}")
        return digest

    @staticmethod
    def _require_sha256(value, *, field_name: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(
                f"{field_name} must be a lowercase 64-character SHA-256 digest"
            )
        return value

    @classmethod
    def _validate_transition_parent_metadata(cls, metadata):
        if metadata is None:
            return None
        if not isinstance(metadata, dict):
            raise RuntimeError("checkpoint transition_parent metadata must be a mapping")
        required = {
            "run_identity_sha256",
            "snapshot_sha256",
            "handoff_manifest_sha256",
        }
        missing = sorted(required - set(metadata))
        if missing:
            raise RuntimeError(
                f"checkpoint transition_parent metadata is missing fields: {missing}"
            )
        validated = dict(metadata)
        for field_name in sorted(required):
            validated[field_name] = cls._require_sha256(
                validated[field_name], field_name=f"transition_parent.{field_name}"
            )
        return validated

    def _validate_transition_topology(
        self,
        manifest,
        snapshot,
        *,
        checkpoint_accumulation_steps: int,
        saved_world_size: int,
    ):
        """Authorize a topology change only when the global batch is unchanged."""
        if (
            checkpoint_accumulation_steps == self.gradient_accumulation_steps
            and saved_world_size == self.world_size
        ):
            if manifest.get("schema_version") == 1:
                return None
            raise RuntimeError(
                "schema 2 topology migration requires a changed topology tuple"
            )
        if manifest.get("transition_kind") != TOPOLOGY_MIGRATION_KIND:
            raise RuntimeError(
                "transition checkpoint topology mismatch without an explicit "
                f"{TOPOLOGY_MIGRATION_KIND} handoff"
            )
        if manifest.get("schema_version") != 2:
            raise RuntimeError("topology migration requires handoff schema_version=2")
        if manifest.get("rank_local_state_policy") != "reset":
            raise RuntimeError(
                "topology migration must explicitly reset rank-local state"
            )
        authorization_basis = manifest.get("authorization_basis")
        if not isinstance(authorization_basis, str) or not authorization_basis.strip():
            raise RuntimeError("topology migration lacks authorization_basis")

        def positive_int(field_name: str) -> int:
            value = manifest.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RuntimeError(
                    f"topology migration {field_name} must be a positive integer"
                )
            return value

        parent_batch_size = positive_int("parent_batch_size")
        parent_world_size = positive_int("parent_world_size")
        parent_accumulation = positive_int(
            "parent_gradient_accumulation_steps"
        )
        parent_global_batch = positive_int(
            "parent_effective_global_batch_size"
        )
        target_batch_size = positive_int("target_batch_size")
        target_world_size = positive_int("target_world_size")
        target_accumulation = positive_int(
            "target_gradient_accumulation_steps"
        )
        target_global_batch = positive_int("target_effective_global_batch_size")
        if (
            parent_world_size != saved_world_size
            or parent_accumulation != checkpoint_accumulation_steps
        ):
            raise RuntimeError(
                "topology migration parent tuple differs from the checkpoint"
            )
        if (
            target_batch_size != self.batch_size
            or target_world_size != self.world_size
            or target_accumulation != self.gradient_accumulation_steps
        ):
            raise RuntimeError(
                "topology migration child tuple differs from the current run"
            )
        computed_parent_global_batch = (
            parent_batch_size * parent_world_size * parent_accumulation
        )
        computed_target_global_batch = (
            target_batch_size * target_world_size * target_accumulation
        )
        if (
            parent_global_batch != computed_parent_global_batch
            or target_global_batch != computed_target_global_batch
            or parent_global_batch != target_global_batch
        ):
            raise RuntimeError(
                "topology migration must preserve the effective global batch"
            )

        identity_path_value = manifest.get("parent_run_identity")
        if not isinstance(identity_path_value, str) or not identity_path_value:
            raise RuntimeError("topology migration lacks parent_run_identity")
        identity_path = Path(identity_path_value).expanduser()
        if not identity_path.is_absolute():
            raise RuntimeError("topology migration parent_run_identity must be absolute")
        identity_path = identity_path.resolve(strict=True)
        expected_identity_file_sha = self._require_sha256(
            manifest.get("parent_run_identity_file_sha256"),
            field_name="parent_run_identity_file_sha256",
        )
        actual_identity_file_sha = self._sha256_file_rank_zero(identity_path)
        if actual_identity_file_sha != expected_identity_file_sha:
            raise RuntimeError("topology migration parent run identity SHA-256 mismatch")
        try:
            parent_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"unable to read topology migration parent identity: {exc}"
            ) from exc
        if parent_identity.get("identity_sha256") != snapshot.get(
            "run_identity_sha256"
        ):
            raise RuntimeError(
                "topology migration parent identity differs from the checkpoint"
            )
        expected_parent_identity = {
            "batch_size": parent_batch_size,
            "world_size": parent_world_size,
            "gradient_accumulation_steps": parent_accumulation,
            "effective_global_batch_size": parent_global_batch,
        }
        mismatches = [
            key
            for key, value in expected_parent_identity.items()
            if parent_identity.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                "topology migration parent identity mismatch for fields: "
                f"{mismatches}"
            )

        checkpoint_ack_value = manifest.get("checkpoint_ack")
        if not isinstance(checkpoint_ack_value, str) or not checkpoint_ack_value:
            raise RuntimeError("topology migration lacks checkpoint_ack")
        checkpoint_ack_path = Path(checkpoint_ack_value).expanduser()
        if not checkpoint_ack_path.is_absolute():
            raise RuntimeError("topology migration checkpoint_ack must be absolute")
        checkpoint_ack_path = checkpoint_ack_path.resolve(strict=True)
        expected_ack_sha = self._require_sha256(
            manifest.get("checkpoint_ack_sha256"),
            field_name="checkpoint_ack_sha256",
        )
        actual_ack_sha = self._sha256_file_rank_zero(checkpoint_ack_path)
        if actual_ack_sha != expected_ack_sha:
            raise RuntimeError("topology migration checkpoint ACK SHA-256 mismatch")
        try:
            checkpoint_ack = json.loads(
                checkpoint_ack_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"unable to read topology migration checkpoint ACK: {exc}"
            ) from exc
        if (
            checkpoint_ack.get("schema_version") != 1
            or checkpoint_ack.get("checkpoint_written") is not True
            or checkpoint_ack.get("run_identity_sha256")
            != snapshot.get("run_identity_sha256")
        ):
            raise RuntimeError("topology migration checkpoint ACK is not valid")
        ack_next_iter = checkpoint_ack.get("next_iter")
        if (
            ack_next_iter != snapshot.get("_start_iter")
            or manifest.get("checkpoint_ack_next_iter") != ack_next_iter
        ):
            raise RuntimeError(
                "topology migration checkpoint ACK iteration mismatch"
            )
        ack_snapshot = Path(str(checkpoint_ack.get("snapshot", ""))).expanduser()
        if not ack_snapshot.is_absolute():
            raise RuntimeError(
                "topology migration checkpoint ACK snapshot must be absolute"
            )
        handoff_snapshot = Path(str(manifest.get("parent_snapshot", ""))).expanduser()
        if ack_snapshot.resolve(strict=True) != handoff_snapshot.resolve(strict=True):
            raise RuntimeError(
                "topology migration checkpoint ACK points to another snapshot"
            )
        return {
            "transition_kind": TOPOLOGY_MIGRATION_KIND,
            "rank_local_state_policy": "reset",
            "authorization_basis": authorization_basis.strip(),
            "checkpoint_ack_sha256": actual_ack_sha,
            "parent_world_size": parent_world_size,
            "parent_gradient_accumulation_steps": parent_accumulation,
            "target_world_size": target_world_size,
            "target_gradient_accumulation_steps": target_accumulation,
            "effective_global_batch_size": target_global_batch,
        }

    def _load_transition_snapshot(self, handoff_path: Path):
        """Warm-continue optimizer state while starting a new dataset lineage.

        A dataset-stage transition is deliberately not an exact resume.  Model,
        optimizer, scheduler, scaler, next iteration, and observation count are
        inherited from the immutable parent snapshot.  Rank-local loader/RNG
        state, W&B identity, and validation bests are intentionally reset for
        the child dataset and child run identity.
        """
        handoff_path = handoff_path.expanduser().resolve(strict=True)
        try:
            manifest_bytes = handoff_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"unable to read transition handoff manifest {handoff_path}: {exc}"
            ) from exc
        schema_version = manifest.get("schema_version")
        if (
            manifest.get("status") != "complete"
            or schema_version not in (1, 2)
            or (
                schema_version == 2
                and manifest.get("transition_kind") != TOPOLOGY_MIGRATION_KIND
            )
        ):
            raise RuntimeError(
                "transition handoff must be complete schema 1, or complete schema 2 "
                f"with transition_kind={TOPOLOGY_MIGRATION_KIND!r}"
            )

        parent_identity = self._require_sha256(
            manifest.get("parent_run_identity_sha256"),
            field_name="parent_run_identity_sha256",
        )
        expected_snapshot_sha = self._require_sha256(
            manifest.get("parent_snapshot_sha256"),
            field_name="parent_snapshot_sha256",
        )
        if self.run_identity_sha256 is None:
            raise RuntimeError(
                "dataset-stage transition requires LACWM_RUN_IDENTITY_SHA256 for "
                "the new child run"
            )
        self._require_sha256(
            self.run_identity_sha256, field_name="LACWM_RUN_IDENTITY_SHA256"
        )
        if self.run_identity_sha256 == parent_identity:
            raise RuntimeError(
                "dataset-stage transition requires a new run identity distinct "
                "from the parent"
            )

        parent_snapshot_value = manifest.get("parent_snapshot")
        if not isinstance(parent_snapshot_value, str) or not parent_snapshot_value:
            raise RuntimeError("transition handoff lacks parent_snapshot")
        parent_snapshot = Path(parent_snapshot_value).expanduser()
        if not parent_snapshot.is_absolute():
            parent_snapshot = handoff_path.parent / parent_snapshot
        parent_snapshot = parent_snapshot.resolve(strict=True)
        actual_snapshot_sha = self._sha256_file_rank_zero(parent_snapshot)
        if actual_snapshot_sha != expected_snapshot_sha:
            raise RuntimeError(
                "transition parent snapshot SHA-256 mismatch: "
                f"actual={actual_snapshot_sha!r}, expected={expected_snapshot_sha!r}"
            )

        snapshot = torch.load(parent_snapshot, map_location="cpu", weights_only=True)
        if snapshot.get("snapshot_schema_version") != 3:
            raise RuntimeError(
                "unsupported transition checkpoint schema; expected "
                "snapshot_schema_version=3"
            )
        if snapshot.get("run_identity_sha256") != parent_identity:
            raise RuntimeError(
                "transition parent identity mismatch between handoff and snapshot: "
                f"handoff={parent_identity!r}, "
                f"snapshot={snapshot.get('run_identity_sha256')!r}"
            )
        checkpoint_accumulation_steps = int(
            snapshot.get("gradient_accumulation_steps", 1)
        )
        saved_world_size = int(snapshot.get("world_size", -1))
        topology_migration = self._validate_transition_topology(
            manifest,
            snapshot,
            checkpoint_accumulation_steps=checkpoint_accumulation_steps,
            saved_world_size=saved_world_size,
        )
        for key in ("model", "optimizer", "lr_scheduler", "_start_iter", "_total_observations"):
            if key not in snapshot:
                raise RuntimeError(f"transition checkpoint is missing required key {key!r}")
        start_iter = snapshot["_start_iter"]
        total_observations = snapshot["_total_observations"]
        if isinstance(start_iter, bool) or not isinstance(start_iter, Integral) or start_iter < 0:
            raise RuntimeError("transition checkpoint _start_iter must be nonnegative")
        if (
            isinstance(total_observations, bool)
            or not isinstance(total_observations, Integral)
            or total_observations < 0
        ):
            raise RuntimeError(
                "transition checkpoint _total_observations must be nonnegative"
            )
        if self.use_amp != ("scaler" in snapshot):
            raise RuntimeError(
                "transition checkpoint AMP/scaler policy differs from the child run"
            )

        # All provenance and compatibility checks precede state mutation.
        self.model.module.load_state_dict(snapshot["model"])
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.lr_scheduler.load_state_dict(snapshot["lr_scheduler"])
        if self.use_amp:
            self.scaler.load_state_dict(snapshot["scaler"])
        self._start_iter = int(start_iter)
        self.metrics.total_observations = int(total_observations)
        self.metrics.best_val_loss = float("inf")
        for key in list(vars(self.metrics)):
            if key.startswith("best_val_loss_"):
                delattr(self.metrics, key)
        self.metrics.reset_throughput_counters()
        self._resume_rng_state = None
        self.wandb_run_id = None
        self.resumed = False
        self.transitioned = True
        self.transition_parent = {
            "run_identity_sha256": parent_identity,
            "snapshot_sha256": actual_snapshot_sha,
            "handoff_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "snapshot": str(parent_snapshot),
        }
        if topology_migration is not None:
            self.transition_parent.update(topology_migration)
        logger.info(
            "Transitioned from parent run %s at iteration %s; loader/RNG and "
            "validation-best state start fresh",
            parent_identity,
            self._start_iter,
        )

    def _log(self, metrics):
        if not self.is_main_process:
            return

        if self.use_wandb:
            wandb.log(metrics, step=self.metrics.total_observations)

        if "train_loss/loss" in metrics:
            iter_num = metrics.get("iteration", 0)
            loss = metrics.get("train_loss/loss", 0)
            samples_per_second = metrics.get("samples_per_second", 0)
            grad_norm = metrics.get("system/gradient_norm_max", float("nan"))
            step_seconds = metrics.get(
                "system/optimizer_step_seconds_max", float("nan")
            )

            logger.info(
                f"[{iter_num}] loss: {loss:0.3f}, "
                f"samples/sec: {samples_per_second:.1f}, "
                f"step: {step_seconds:.2f}s, grad norm: {grad_norm:.3f}, "
                f"total observations: {self.metrics.total_observations:,}"
            )

        if (
            logger.getEffectiveLevel() <= logging.DEBUG
            and "gpu_memory_allocated_gb" in metrics
        ):
            allocated = metrics["gpu_memory_allocated_gb"]
            reserved = metrics["gpu_memory_reserved_gb"]
            logger.debug(f"[{iter_num}] gpu memory (allocated): {allocated:0.1f} gb")
            logger.debug(f"[{iter_num}] gpu memory (reserved): {reserved:0.1f} gb")

        if "val_loss" in metrics:
            iter_num = metrics.get("iteration", 0)
            val_loss = metrics.get("val_loss", 0)
            logger.info(
                f"[{iter_num}] validation loss: {val_loss:.4f} "
                f"(best: {self.metrics.best_val_loss:.4f})"
            )

    def _clip_grad_norm(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        return torch.nn.utils.clip_grad_norm_(
            parameters=parameters,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            error_if_nonfinite=self.error_if_nonfinite,
        )

    def _collect_operational_metrics(self, grad_norm, step_time: float) -> dict:
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = float(grad_norm.detach().float().item())
        else:
            grad_norm = float(grad_norm)

        memory_allocated = 0.0
        memory_reserved = 0.0
        if self._model_uses_cuda():
            memory_allocated = torch.cuda.memory_allocated(self.local_rank) / 2**30
            memory_reserved = torch.cuda.memory_reserved(self.local_rank) / 2**30

        distributed = dist.is_initialized() and self.world_size > 1
        collective_device = "cpu"
        if distributed and str(torch.distributed.get_backend()).lower() == "nccl":
            collective_device = torch.device("cuda", self.local_rank)
        values = torch.tensor(
            [grad_norm, float(step_time), memory_allocated, memory_reserved],
            dtype=torch.float64,
            device=collective_device,
        )
        if distributed:
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)

        return {
            "system/gradient_norm_max": float(values[0].item()),
            "system/optimizer_step_seconds_max": float(values[1].item()),
            "system/gpu_memory_allocated_gib_max": float(values[2].item()),
            "system/gpu_memory_reserved_gib_max": float(values[3].item()),
            "system/world_size": self.world_size,
            "system/effective_global_batch_size": (
                self.batch_size * self.gradient_accumulation_steps * self.world_size
            ),
        }

    def _should_collect_operational_metrics(self) -> bool:
        """Collect synchronized telemetry only on iterations that will be logged."""
        if not all(
            hasattr(self, name)
            for name in ("_curr_iter", "log_every", "max_iter")
        ):
            return True
        return (
            self._curr_iter % self.log_every == 0
            or self._curr_iter + 1 == self.max_iter
        )

    def _model_uses_cuda(self) -> bool:
        if not torch.cuda.is_available() or not hasattr(self, "model"):
            return False
        try:
            return next(self.model.parameters()).is_cuda
        except StopIteration:
            return False

    def _require_finite_losses(self, loss):
        """Fail every rank together before backward if any rank is non-finite."""
        values = [loss]
        if hasattr(self.model.module, "aux_losses"):
            values.extend(self.model.module.aux_losses.values())
        local_finite = torch.stack(
            [torch.isfinite(value).all() for value in values]
        ).all().to(device=loss.device, dtype=torch.int32)
        if dist.is_initialized():
            torch.distributed.all_reduce(
                local_finite, op=torch.distributed.ReduceOp.MIN
            )
        if local_finite.item() != 1:
            raise FloatingPointError(
                "non-finite loss or auxiliary metric on at least one distributed rank"
            )

    def _to_device(self, batch):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                if self.use_amp:
                    batch[key] = batch[key].to(self.local_rank)
                else:
                    batch[key] = batch[key].to(self.local_rank, dtype=self.dtype)
            elif isinstance(batch[key], dict):
                batch[key] = self._to_device(batch[key])
            else:
                raise TypeError(f"Unsupported type in batch: {type(batch[key])}")
        return batch

    def _step(self):
        collect_operational_metrics = self._should_collect_operational_metrics()
        if collect_operational_metrics and self._model_uses_cuda():
            # CUDA execution is asynchronous. Synchronize only on logging steps
            # so the reported maximum step duration is accurate without adding a
            # device/global collective to every optimizer update.
            torch.cuda.synchronize(self.local_rank)
        step_started = time.perf_counter()
        loss_sums = {}
        loss_counts = defaultdict(int)

        for microbatch_index in range(self.gradient_accumulation_steps):
            batch = next(self._data_loader_iter)
            batch = self._to_device(batch)
            is_final_microbatch = (
                microbatch_index + 1 == self.gradient_accumulation_steps
            )
            sync_context = (
                nullcontext()
                if is_final_microbatch
                else self.model.no_sync()
            )

            # DDP requires the forward pass to be inside no_sync() as well as
            # backward. Dividing every microbatch loss gives the gradient of the
            # mean over the complete accumulation window (loaders use fixed-size
            # microbatches).
            with sync_context:
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        loss = self.model(**batch)
                    self._require_finite_losses(loss)
                    self.scaler.scale(
                        loss / self.gradient_accumulation_steps
                    ).backward()
                else:
                    loss = self.model(**batch)
                    self._require_finite_losses(loss)
                    (loss / self.gradient_accumulation_steps).backward()

            microbatch_losses = {"loss": loss.detach()}
            if hasattr(self.model.module, "aux_losses"):
                microbatch_losses.update(self.model.module.aux_losses)
            for key, value in microbatch_losses.items():
                detached = value.detach()
                if key not in loss_sums:
                    loss_sums[key] = detached.clone()
                else:
                    loss_sums[key].add_(detached)
                loss_counts[key] += 1

        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        grad_norm = self._clip_grad_norm()
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        if collect_operational_metrics and self._model_uses_cuda():
            torch.cuda.synchronize(self.local_rank)
        step_time = time.perf_counter() - step_started
        self.metrics.update(
            microbatches=self.gradient_accumulation_steps,
            step_time=step_time,
        )
        if collect_operational_metrics:
            self._last_operational_metrics = self._collect_operational_metrics(
                grad_norm, step_time
            )
        else:
            self._last_operational_metrics = {}

        # Report global means over only the microbatches/ranks that emitted each
        # key. The reducer first establishes one global key order, avoiding
        # mismatched collectives when ranks observe different morphologies.
        losses = self._distributed_weighted_loss_means(loss_sums, loss_counts)

        # Convert to Python scalars for logging
        losses = {k: v.item() for k, v in losses.items()}

        return losses

    @torch.inference_mode()
    def _validate(self):
        logger.info(f"Running validation at iteration {self._curr_iter}")
        self.model.eval()

        all_losses = {}
        avg_loss = []

        # Validate on each dataset
        for i, (val_iter, val_loader) in enumerate(
            zip(self._val_data_loader_iters, self.val_data_loaders)
        ):
            # add index to dataset name to account for multiple splits of same dataset
            dataset_name = val_loader.dataset.name + f"_{i}"

            if "MultiDataset" in dataset_name:
                dataset_name = val_loader.dataset.full_name + f"_{i}"
            loss_sums = {}
            loss_counts = defaultdict(int)

            for _ in range(self.n_val_samples):
                val_batch = next(val_iter)
                val_batch = self._to_device(val_batch)

                with torch.autocast(
                    device_type="cuda", dtype=self.dtype, enabled=self.use_amp
                ):
                    batch_loss = self.model(**val_batch)

                batch_losses = {"loss": batch_loss.detach()}
                if hasattr(self.model.module, "aux_losses"):
                    batch_losses.update(self.model.module.aux_losses)
                for key, value in batch_losses.items():
                    detached = value.detach()
                    if key not in loss_sums:
                        loss_sums[key] = detached.clone()
                    else:
                        loss_sums[key].add_(detached)
                    loss_counts[key] += 1

            # Conditional keys may differ by rank and validation sample. Weight
            # each mean by its actual number of contributing batch samples.
            dataset_losses = {
                key: value.item()
                for key, value in self._distributed_weighted_loss_means(
                    loss_sums, loss_counts
                ).items()
            }

            # Add to all_losses with dataset prefix
            for key, value in dataset_losses.items():
                all_losses[f"{dataset_name}/{key}"] = value
            avg_loss.append(dataset_losses["loss"])

        if len(avg_loss) > 1:
            all_losses[AVG_LOSS_KEY] = sum(avg_loss) / len(avg_loss)

        self.model.train()
        logger.info(
            f"Validation completed for iteration {self._curr_iter}, losses: {all_losses}"
        )
        return all_losses

    @torch.inference_mode()
    def _viz(self):
        logger.info(f"Running visualizations at iteration {self._curr_iter}")
        self.model.eval()

        for i, (viz_iter, viz_loader) in enumerate(
            zip(self._viz_data_loader_iters, self.viz_data_loaders)
        ):
            # add index to dataset name to account for multiple splits of same dataset
            dataset_name = viz_loader.dataset.name + f"_{i}"
            if "MultiDataset" in dataset_name:
                dataset_name = viz_loader.dataset.full_name + f"_{i}"
            logger.info(f"Visualizing dataset: {dataset_name}")

            try:
                viz_batch = next(viz_iter)
                viz_batch = self._to_device(viz_batch)

                # generate visualization
                with torch.autocast(
                    device_type="cuda", dtype=self.dtype, enabled=self.use_amp
                ):
                    # [N, T1, C, *, *] [0, 1]
                    visualization = self.model.module.visualize(**viz_batch)
                    assert (
                        visualization.min() >= 0.0 and visualization.max() <= 1.0
                    ), f"Visualization out of [0, 1] range: {visualization.min()} {visualization.max()}"

                # create output directory with dataset name
                output_folder: Path = (
                    self.viz_path / f"iter_{self._curr_iter}" / dataset_name
                )
                output_folder.mkdir(exist_ok=True, parents=True)

                visualization = rearrange(visualization, "N T C H W -> N T H W C")
                logger.info(
                    f"Saving {visualization.shape[0]} videos for dataset {dataset_name} at iteration {self._curr_iter}"
                )
                for batch_index in range(len(visualization)):
                    video = visualization[batch_index]

                    # convert to uint8
                    video_255 = (
                        (video * 255.0)
                        .clamp(0, 255)
                        .to(torch.uint8)
                        .detach()
                        .cpu()
                        .numpy()
                    )  # T H W C

                    # zero pad to avoid ffmpeg resizing
                    block_size = 8
                    _, H, W, _ = video_255.shape
                    if H % block_size != 0 or W % block_size != 0:
                        h_pad = (block_size - H % block_size) % block_size
                        w_pad = (block_size - W % block_size) % block_size
                        padding = [(0, 0), (0, h_pad), (0, w_pad), (0, 0)]
                        video_255 = np.pad(video_255, padding)
                        logger.info(
                            f"Adding padding to visualization for {dataset_name}: {padding}"
                        )

                    # save video to disk with dataset name in filename
                    mp4_path = (
                        output_folder
                        / f"viz_{dataset_name}_{self.global_rank}_{batch_index}.mp4"
                    )
                    imageio.mimwrite(
                        mp4_path, video_255, "mp4", macro_block_size=block_size
                    )

            except Exception as e:
                logger.warning(f"Visualization failed for {dataset_name}: {e}")

        # Ensure all writes are flushed before logging
        if dist.is_initialized():
            dist.barrier()

        # Only main process collects and logs all videos
        if self.use_wandb and self.is_main_process:
            iter_folder = self.viz_path / f"iter_{self._curr_iter}"

            if iter_folder.exists():
                wandb_videos = {}

                for dataset_folder in iter_folder.iterdir():
                    if dataset_folder.is_dir():
                        dataset_name = dataset_folder.name
                        video_paths = sorted(dataset_folder.glob("*.mp4"))

                        if video_paths:
                            wandb_videos[dataset_name] = [
                                wandb.Video(str(path)) for path in video_paths
                            ]

                # Log with dataset-specific keys
                if wandb_videos:
                    log_dict = {}
                    for dataset_name, videos in wandb_videos.items():
                        log_dict[f"viz/{dataset_name}"] = videos

                    total_videos = sum(len(v) for v in wandb_videos.values())
                    logger.info(
                        f"Sending {total_videos} video(s) to wandb from {len(wandb_videos)} dataset(s)"
                    )
                    wandb.log(log_dict, step=self.metrics.total_observations)

        self.model.train()

    def initialize_wandb(self, cfg: DictConfig):
        self.use_wandb = cfg.wandb.enabled and self.is_main_process
        if not self.use_wandb:
            return

        init_wandb_from_config(cfg, run_id=self.wandb_run_id)
        self.wandb_run_id = wandb.run.id

        # add summary information
        total_params = get_model_size(self.model)
        trainable_params = get_model_size(self.model, requires_grad=True)
        wandb.summary["total_parameters"] = total_params
        wandb.summary["trainable_parameters"] = trainable_params
        wandb.summary["frozen_parameters"] = total_params - trainable_params
        wandb.summary["world_size"] = self.world_size
        wandb.summary["nodes"] = int(os.environ.get("LACWM_NNODES", "1"))
        wandb.summary["gpus_per_node"] = int(
            os.environ.get("LACWM_GPUS_PER_NODE", str(self.world_size))
        )
        wandb.summary["effective_global_batch_size"] = (
            self.batch_size * self.gradient_accumulation_steps * self.world_size
        )
        wandb.summary["run_identity_sha256"] = getattr(
            self, "run_identity_sha256", None
        )
        if getattr(self, "transition_parent", None) is not None:
            wandb.summary["transition_parent"] = self.transition_parent

    def finalize_wandb(self):
        if not self.use_wandb:
            return
        wandb.finish()

    def train(self):
        self.model.train()
        self._install_checkpoint_signal_handlers()
        try:
            # A request can arrive while a requeued segment is starting. Do not
            # fabricate a checkpoint that would claim update zero was completed.
            if self._checkpoint_stop_requested_across_ranks():
                self._write_checkpoint_ack(
                    checkpoint_written=self.save_path.is_file(),
                    next_iter=self._start_iter,
                )
                return "rescheduled"

            for self._curr_iter in range(self._start_iter, self.max_iter):
                losses = self._step()

                is_last = self._curr_iter + 1 == self.max_iter

                # log metrics
                if self._curr_iter % self.log_every == 0 or is_last:
                    metrics = self.metrics.get_train_metrics(
                        self._curr_iter,
                        losses,
                        getattr(self, "_last_operational_metrics", {}),
                    )
                    metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                    self._log(metrics)
                    self.metrics.reset_throughput_counters()

                # Finish scheduled validation/visualization before a signal save;
                # their RNG and loader side effects are part of exact continuation.
                if self._curr_iter % self.val_every == 0 or is_last:
                    val_losses = self._validate()
                    # Track and save best validation loss per dataset
                    improved_datasets = []
                    for key, value in val_losses.items():
                        if key.endswith("/loss"):
                            dataset_name = key.removesuffix("/loss")
                            best_key = f"best_val_loss_{dataset_name}"
                            previous_best = getattr(
                                self.metrics, best_key, float("inf")
                            )
                            if value < previous_best:
                                setattr(self.metrics, best_key, value)
                                improved_datasets.append((dataset_name, value))

                    # Refresh before writing any best checkpoint so the snapshot
                    # and W&B telemetry never retain the initial global infinity.
                    self.metrics.refresh_best_val_loss()
                    for dataset_name, value in improved_datasets:
                        if self.save_best:
                            self._save_snapshot(
                                is_best=True, dataset_name=dataset_name
                            )
                        logger.info(
                            f"New best for {dataset_name}: {value:.4f}"
                        )

                    val_metrics = self.metrics.get_val_metrics(
                        self._curr_iter, val_losses
                    )
                    self._log(val_metrics)

                # save visualizations
                if self.viz_data_loaders is not None:
                    if self._curr_iter % self.viz_every == 0 or is_last:
                        self._viz()

                # Save after validation/visualization so the live checkpoint's RNG
                # state exactly matches the next uninterrupted training iteration.
                saved_this_iteration = self._curr_iter % self.save_every == 0 or is_last
                if saved_this_iteration:
                    self._save_snapshot()

                if self._checkpoint_stop_requested_across_ranks() and not is_last:
                    if not saved_this_iteration:
                        self._save_snapshot()
                    self._write_checkpoint_ack(
                        checkpoint_written=True,
                        next_iter=self._curr_iter + 1,
                    )
                    logger.warning(
                        "Checkpointed iteration %s for scheduler resubmission",
                        self._curr_iter,
                    )
                    return "rescheduled"

            self._write_completion_marker()
            return "completed"
        finally:
            self._restore_checkpoint_signal_handlers()
