# References:
#     https://github.com/pytorch/examples/blob/5dfeb46902baf444010f2f54bcf4dfbea109ae4d/distributed/ddp-tutorial-series/multinode.py
#     https://github.com/pytorch/torchtitan/blob/b345557a37d7d8804e3b1cf8e9e0a36e46e689e9/torchtitan/train.py
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
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

    def update(self):
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.last_step_time = current_time
        self.time_since_log += step_time

        samples = self.batch_size * self.world_size
        self.samples_since_log += samples
        self.total_observations += samples

    def get_train_metrics(self, iter_num, losses):
        samples_per_second = 0
        if self.time_since_log > 0 and iter_num > self.warmup:
            samples_per_second = self.samples_since_log / self.time_since_log

        metrics = {
            "iteration": iter_num,
            "samples_per_second": samples_per_second,
            "total_observations": self.total_observations,
        }

        for key, value in losses.items():
            metrics[f"train_loss/{key}"] = value

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
        metrics["val_loss/best_val_loss"] = self.best_val_loss

        return metrics

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
        self.viz_every = config["visualization"]["viz_every"]
        self.viz_path = Path(config["visualization"]["viz_path"])
        self.save_every = config["saving"]["save_every"]
        self.save_path = Path(config["saving"]["save_path"])
        self._start_iter = 0
        self._curr_iter = 0

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
        self._val_data_loader_iters = [iter(dl) for dl in self.val_data_loaders]
        self._viz_data_loader_iters = [iter(dl) for dl in self.viz_data_loaders]
        self.batch_size = data_loader.batch_size
        self._data_loader_iter = iter(data_loader)

        # AMP and Grad Clipping
        self.dtype = self._get_torch_dtype(config["dtype"])
        self.use_amp = config["amp_enabled"]
        self.scaler = GradScaler("cuda", enabled=self.use_amp) if self.use_amp else None
        self.max_norm = config["gradient_clipping"]["max_norm"]
        self.norm_type = config["gradient_clipping"]["norm_type"]
        self.error_if_nonfinite = config["gradient_clipping"]["error_if_nonfinite"]

        # Metrics tracking
        self.metrics = Metrics(self.world_size, self.batch_size)

        # Wandb
        self.use_wandb = False
        self.wandb_run_id = None

        # Load checkpoint if exists
        if self.save_path.exists():
            self._load_snapshot()
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

    def _build_snapshot(self):
        snapshot = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "data_loader": self.data_loader.state_dict(),
            "_start_iter": self._curr_iter + 1,
            "_total_observations": self.metrics.total_observations,
            "best_val_loss": self.metrics.best_val_loss,
        }

        if self.use_amp:
            snapshot["scaler"] = self.scaler.state_dict()

        if self.wandb_run_id is not None:
            snapshot["wandb_run_id"] = self.wandb_run_id

        return snapshot

    def _save_snapshot(self, is_best=False, dataset_name=None):
        if not self.is_main_process:
            return

        snapshot = self._build_snapshot()
        save_path = self.save_path
        if is_best:
            suffix = (
                f".{dataset_name}.best.pt" if dataset_name is not None else ".best.pt"
            )
            save_path = save_path.with_suffix(suffix)
        elif self._curr_iter % 10000 == 0:
            save_path = save_path.with_suffix(f".{self._curr_iter}.pt")
        torch.save(snapshot, save_path)
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
        self.model.module.load_state_dict(snapshot["model"])

        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.lr_scheduler.load_state_dict(snapshot["lr_scheduler"])
        self.metrics.total_observations = snapshot["_total_observations"]

        if "best_val_loss" in snapshot:
            self.metrics.best_val_loss = snapshot["best_val_loss"]
        if self.use_amp and "scaler" in snapshot:
            self.scaler.load_state_dict(snapshot["scaler"])
        if "wandb_run_id" in snapshot:
            self.wandb_run_id = snapshot["wandb_run_id"]
            logger.info(f"Resuming wandb run: {self.wandb_run_id}")

        # The state_dict for the data loader needs to be loaded to the cpu
        # otherwise the random number generators in datasets or transforms will
        # not load properly. It seem like there should be a better solution than
        # reloading the snapshot to the cpu.
        cpu_snapshot = torch.load(self.save_path, map_location="cpu", weights_only=True)
        try:
            self.data_loader.load_state_dict(cpu_snapshot["data_loader"])
        except Exception as e:
            logger.warning(f"Skipping data_loader state on resume (num_workers changed?): {e}")
        self._start_iter = snapshot["_start_iter"]
        logger.info(f"Resuming from iteration {self._start_iter:,}")

    def _log(self, metrics):
        if not self.is_main_process:
            return

        if self.use_wandb:
            wandb.log(metrics, step=self.metrics.total_observations)

        if "train_loss/loss" in metrics:
            iter_num = metrics.get("iteration", 0)
            loss = metrics.get("train_loss/loss", 0)
            samples_per_second = metrics.get("samples_per_second", 0)

            logger.info(
                f"[{iter_num}] loss: {loss:0.3f}, "
                f"samples/sec: {samples_per_second:.1f}, "
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
        torch.nn.utils.clip_grad_norm_(
            parameters=parameters,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            error_if_nonfinite=self.error_if_nonfinite,
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
        batch = next(self._data_loader_iter)
        batch = self._to_device(batch)

        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                loss = self.model(**batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self._clip_grad_norm()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.model(**batch)
            loss.backward()
            self._clip_grad_norm()
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        self.metrics.update()

        # Collect all losses as tensors
        losses = {"loss": loss.detach()}  # Keep as tensor
        if hasattr(self.model.module, "aux_losses"):
            losses.update(self.model.module.aux_losses)

        # Reduce all losses across processes
        if dist.is_initialized():
            for key, value in losses.items():
                torch.distributed.all_reduce(value)
                losses[key] = value / dist.get_world_size()

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
            losses = defaultdict(list)

            for _ in range(self.n_val_samples):
                val_batch = next(val_iter)
                val_batch = self._to_device(val_batch)

                with torch.autocast(
                    device_type="cuda", dtype=self.dtype, enabled=self.use_amp
                ):
                    batch_loss = self.model(**val_batch)

                # Keep as tensors for now
                losses["loss"].append(batch_loss.detach())
                if hasattr(self.model.module, "aux_losses"):
                    for aux_name, aux_value in self.model.module.aux_losses.items():
                        losses[aux_name].append(aux_value.detach())

            # Average losses for this dataset
            dataset_losses = {}
            for key, value_list in losses.items():
                # Stack tensors and compute mean
                stacked = torch.stack(value_list)
                mean_loss = stacked.mean()

                # Reduce across processes
                if dist.is_initialized():
                    torch.distributed.all_reduce(mean_loss)
                    mean_loss = mean_loss / dist.get_world_size()

                dataset_losses[key] = mean_loss.item()

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

    def finalize_wandb(self):
        if not self.use_wandb:
            return
        wandb.finish()

    def train(self):
        self.model.train()

        for self._curr_iter in range(self._start_iter, self.max_iter):
            losses = self._step()

            is_last = self._curr_iter + 1 == self.max_iter

            # log metrics
            if self._curr_iter % self.log_every == 0 or is_last:
                metrics = self.metrics.get_train_metrics(self._curr_iter, losses)
                metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                self._log(metrics)
                self.metrics.reset_throughput_counters()

            # save snapshot
            if self._curr_iter % self.save_every == 0 or is_last:
                self._save_snapshot()

            # run validation
            if self._curr_iter % self.val_every == 0 or is_last:
                val_losses = self._validate()
                # Track and save best validation loss per dataset
                for key, value in val_losses.items():
                    if key.endswith("/loss"):
                        dataset_name = key.split("/")[0]
                        best_key = f"best_val_loss_{dataset_name}"
                        previous_best = getattr(self.metrics, best_key, float("inf"))
                        if value < previous_best:
                            setattr(self.metrics, best_key, value)
                            self._save_snapshot(is_best=True, dataset_name=dataset_name)
                            logger.info(f"New best for {dataset_name}: {value:.4f}")

                val_metrics = self.metrics.get_val_metrics(self._curr_iter, val_losses)
                self._log(val_metrics)

            # save visualizations
            if self.viz_data_loaders is not None:
                if self._curr_iter % self.viz_every == 0 or is_last:
                    self._viz()
