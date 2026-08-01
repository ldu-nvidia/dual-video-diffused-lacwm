#!/usr/bin/env python3
"""Standalone training and evaluation for the Video Latent Forcing POC.

This entrypoint is intentionally isolated from the production LACWM trainer.
It implements the frozen clean-time convention

    z(t) = t * x + (1 - t) * noise,  t=0 noise, t=1 clean,
    v*   = x - noise,

and records actual transformer calls.  Checkpoints, logs, resolved configs,
and evaluation outputs must live outside this Git repository.

The optional publication metric suite uses pinned local LPIPS-Alex and
torchvision R3D-18 weights; it never downloads weights. Gate decisions are
emitted only by the separate paired analyzer, never inferred from raw means.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as torch_dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.datasets.droid.video_latent_forcing import (  # noqa: E402
    CAMERA as DROID_CAMERA,
    ELIGIBLE_INVENTORY_SHA256 as DROID_ELIGIBLE_INVENTORY_SHA256,
    SCHEMA as DROID_DATA_SCHEMA,
    SPLIT_EPISODE_IDS_SHA256 as DROID_SPLIT_EPISODE_IDS_SHA256,
    DroidVideoLatentForcingDataset,
    read_clip_manifest,
    rows_episode_ids,
    sha256_file,
    unpatchify_lowres_rgb,
)


SCHEMA = "video-latent-forcing-poc-run-v1"
CHECKPOINT_SCHEMA = "video-latent-forcing-poc-checkpoint-v1"
EVALUATION_SCHEMA = "video-latent-forcing-poc-evaluation-v1"
SELECTION_SCHEMA = "video-latent-forcing-poc-selection-v1"
GATE_SCHEMA = "video-latent-forcing-poc-gate-v1"
CLOCK_CONVENTION = "clean-time: t=0 noise, t=1 clean; velocity=clean-noise"
ARMS = ("phase1", "B0", "A1", "L1")
DUAL_CONTROLS = ("autonomous", "off", "shuffled", "oracle_clean")
PHASE1_CONTROLS = (*DUAL_CONTROLS, "context_shuffled")
CONTROLS = PHASE1_CONTROLS
PHASE1_UPDATES = 5_000
DUAL_UPDATES = 20_000
CALIBRATION_UPDATES = 200
PHASE1_CHECKPOINTS = (500, 1_000, 2_000, 5_000)
DUAL_CHECKPOINTS = (500, 1_000, 2_000, 5_000, 10_000, 16_000, 20_000)
PHASE1_FRONTIER = (1, 2, 4, 8, 12, 20, 25)
DUAL_FRONTIER_TOTALS = (2, 4, 8, 12, 20, 50)
PRIMARY_NFE = {
    "phase1": ((25, 0),),
    "B0": ((0, 50),),
    "A1": ((25, 25),),
    "L1": ((25, 25),),
}
MISSING_PRIMARY_METRICS = (
    "lpips_alex_frame",
    "lpips_alex_temporal_difference",
    "r3d18_frechet",
)
FROZEN_GLOBAL_BATCH_SIZE = 256
FROZEN_LEARNING_RATE = 5e-5
FROZEN_WARMUP_UPDATES = 500
FROZEN_WEIGHT_DECAY = 0.0
FROZEN_GRADIENT_CLIP_NORM = 1.0
FROZEN_EMA_DECAY = 0.9999
FROZEN_EMA_SCHEDULE = "min(target_decay,(1+completed_updates)/(10+completed_updates))-v1"
FROZEN_INITIALIZATION = "latent-forcing-zero-adaln-and-output-heads-v1"
FROZEN_CLEAN_TIME_EPS = 0.05
FROZEN_MODEL_WIDTH = 512
FROZEN_MODEL_DEPTH = 12
FROZEN_MODEL_HEADS = 8
FROZEN_MODEL_MLP_RATIO = 4.0
FROZEN_PARAMETER_COUNT = 41_963_760
FROZEN_TRAIN_CLIPS = 64_000
FROZEN_VALIDATION_CLIPS = 890
FROZEN_OPTIMIZER_SEEDS = (1234, 2234, 3234)
FROZEN_EVALUATION_SEEDS = (20260801, 20260802, 20260803)
APPROVED_ARTIFACT_ROOTS = (Path("/lustre"), Path("/mnt/data1"), Path("/mnt/data2"))
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class PocError(RuntimeError):
    """A fail-closed experiment or provenance contract was violated."""


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _atomic_write_text(path: Path, text: str, *, exclusive: bool = False) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise PocError(f"immutable artifact already exists: {path}")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise PocError(f"immutable artifact appeared while publishing: {path}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


def atomic_write_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> None:
    _atomic_write_text(path, json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", exclusive=exclusive)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(_jsonable(payload)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def approved_artifact_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if _is_relative_to(resolved, REPO_ROOT.resolve()):
        raise PocError(f"run artifacts cannot be written inside the Git repository: {resolved}")
    if not any(_is_relative_to(resolved, root.resolve()) for root in APPROVED_ARTIFACT_ROOTS):
        raise PocError(f"artifact path must be under /lustre, /mnt/data1, or /mnt/data2: {resolved}")
    return resolved


def validated_run_dir(artifact_root: str | Path, run_id: str, *, resume: bool) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        raise PocError("run ID must be 1-128 line-safe alphanumeric/._- characters")
    root = approved_artifact_path(artifact_root)
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root:
        raise PocError("run directory must be a direct child of artifact root")
    if resume:
        if not run_dir.is_dir():
            raise PocError(f"resume run directory is missing: {run_dir}")
    elif run_dir.exists():
        raise PocError(f"new run directory already exists: {run_dir}")
    return run_dir


def _git(command: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *command], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def git_record() -> dict[str, Any]:
    return {
        "commit": _git(("rev-parse", "HEAD")),
        "branch": _git(("branch", "--show-current")),
        "dirty": bool(_git(("status", "--porcelain"))),
    }


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise PocError(f"required file is missing: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def runtime_record() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "torchvision", "numpy", "av", "pandas", "wandb", "lpips"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            packages[name] = importlib.metadata.version(name)
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_QOS",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_NODELIST",
        "SLURM_TIMELIMIT",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_JOB_GRES",
        "SLURM_GPUS_ON_NODE",
        "CUDA_VISIBLE_DEVICES",
    )
    gpu_inventory: list[str] = []
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            gpu_inventory = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "packages": packages,
        "slurm": {key: os.environ[key] for key in slurm_keys if key in os.environ},
        "gpu_inventory": gpu_inventory,
    }


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    initialized_here: bool = False

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size > 1:
            torch_dist.barrier()

    def sum_tensor(self, value: Tensor) -> Tensor:
        result = value.clone()
        if self.world_size > 1:
            torch_dist.all_reduce(result, op=torch_dist.ReduceOp.SUM)
        return result

    def gather_objects(self, value: Any) -> list[Any]:
        if self.world_size == 1:
            return [value]
        gathered: list[Any] = [None] * self.world_size
        torch_dist.all_gather_object(gathered, value)
        return gathered


def initialize_distributed(*, allow_cpu: bool = False) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    elif allow_cpu:
        device = torch.device("cpu")
        backend = "gloo"
    else:
        raise PocError("training/evaluation requires CUDA; use CPU only for focused tests")
    if world_size > 1 and not torch_dist.is_initialized():
        torch_dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True
    return DistributedContext(rank, world_size, local_rank, device, initialized_here)


def close_distributed(context: DistributedContext) -> None:
    if context.initialized_here and torch_dist.is_initialized():
        torch_dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    effective = int(seed) + int(rank)
    random.seed(effective)
    np.random.seed(effective % (2**32))
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class DeterministicDistributedBatchSampler(Sampler[list[int]]):
    """Step-addressed batches whose sequence is identical after resume."""

    def __init__(
        self,
        dataset_size: int,
        *,
        global_batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        start_update: int,
        end_update: int,
        micro_batch_size: int | None = None,
    ) -> None:
        if dataset_size < 1:
            raise ValueError("dataset must be nonempty")
        if global_batch_size < world_size or global_batch_size % world_size:
            raise ValueError("global batch size must be divisible by world size")
        if not 0 <= rank < world_size:
            raise ValueError("invalid distributed rank")
        if not 0 <= start_update <= end_update:
            raise ValueError("invalid update interval")
        self.dataset_size = int(dataset_size)
        self.global_batch_size = int(global_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.start_update = int(start_update)
        self.end_update = int(end_update)
        self.local_batch_size = self.global_batch_size // self.world_size
        self.micro_batch_size = (
            self.local_batch_size if micro_batch_size is None else int(micro_batch_size)
        )
        if self.micro_batch_size < 1 or self.local_batch_size % self.micro_batch_size:
            raise ValueError("micro batch size must divide the per-rank optimizer batch")
        self.accumulation_steps = self.local_batch_size // self.micro_batch_size
        self._permutations: dict[int, Tensor] = {}

    def _permutation(self, cycle: int) -> Tensor:
        if cycle not in self._permutations:
            generator = torch.Generator(device="cpu")
            digest = hashlib.sha256(f"vlf-batch\0{self.seed}\0{cycle}".encode()).digest()
            generator.manual_seed(int.from_bytes(digest[:8], "big") & ((1 << 63) - 1))
            self._permutations = {cycle: torch.randperm(self.dataset_size, generator=generator)}
        return self._permutations[cycle]

    def global_indices(self, update: int) -> list[int]:
        start = int(update) * self.global_batch_size
        indexes = []
        for absolute in range(start, start + self.global_batch_size):
            cycle, offset = divmod(absolute, self.dataset_size)
            indexes.append(int(self._permutation(cycle)[offset]))
        return indexes

    def __iter__(self) -> Iterator[list[int]]:
        begin = self.rank * self.local_batch_size
        end = begin + self.local_batch_size
        for update in range(self.start_update, self.end_update):
            local = self.global_indices(update)[begin:end]
            for microstep in range(self.accumulation_steps):
                micro_begin = microstep * self.micro_batch_size
                micro_end = micro_begin + self.micro_batch_size
                yield local[micro_begin:micro_end]

    def __len__(self) -> int:
        return (self.end_update - self.start_update) * self.accumulation_steps


def expand_clock(clock: Tensor, reference: Tensor) -> Tensor:
    if clock.ndim != 1 or clock.shape[0] != reference.shape[0]:
        raise ValueError("clock must have shape [B]")
    return clock.reshape(clock.shape[0], *([1] * (reference.ndim - 1)))


def corrupt_clean_time(clean: Tensor, noise: Tensor, time: Tensor) -> Tensor:
    """Clean-time interpolation: t=0 noise and t=1 clean."""
    if clean.shape != noise.shape:
        raise ValueError("clean and noise tensors must have identical shapes")
    t = expand_clock(time, clean).to(device=clean.device, dtype=clean.dtype)
    return t * clean + (1.0 - t) * noise


def clean_time_velocity(clean: Tensor, noise: Tensor) -> Tensor:
    if clean.shape != noise.shape:
        raise ValueError("clean and noise tensors must have identical shapes")
    return clean - noise


def clean_time_euler_from_x(noisy: Tensor, x_prediction: Tensor, time: Tensor, next_time: Tensor) -> Tensor:
    """Euler step from a clean-state prediction with the correct positive sign."""
    if noisy.shape != x_prediction.shape:
        raise ValueError("state and clean prediction must share shape")
    t = expand_clock(time, noisy).to(noisy.dtype)
    tn = expand_clock(next_time, noisy).to(noisy.dtype)
    velocity = (x_prediction - noisy) / (1.0 - t).clamp_min(FROZEN_CLEAN_TIME_EPS)
    return noisy + (tn - t) * velocity


@dataclass(frozen=True)
class TrainingClocks:
    video_time: Tensor
    auxiliary_time: Tensor
    video_loss_mask: Tensor
    auxiliary_loss_mask: Tensor
    auxiliary_condition_mask: Tensor


def sample_video_time(batch: int, device: torch.device, *, generator: torch.Generator | None = None) -> Tensor:
    base = torch.sigmoid(torch.randn(batch, device=device, generator=generator) * 0.8 - 0.4)
    replace = torch.rand(batch, device=device, generator=generator) < 0.10
    low = torch.rand(batch, device=device, generator=generator) * 0.5
    return torch.where(replace, low, base)


def sample_training_clocks(
    arm: str,
    batch: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> TrainingClocks:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    zeros = torch.zeros(batch, device=device)
    ones = torch.ones(batch, device=device)
    if arm == "phase1":
        auxiliary_time = torch.sigmoid(torch.randn(batch, device=device, generator=generator) - 1.2)
        return TrainingClocks(zeros, auxiliary_time, zeros, ones, ones.bool())
    if arm == "B0":
        # The auxiliary clock is deliberately arbitrary: the parameter-matched
        # baseline must remain a strict no-op under any auxiliary state/clock.
        auxiliary_time = torch.rand(batch, device=device, generator=generator)
        return TrainingClocks(sample_video_time(batch, device, generator=generator), auxiliary_time, ones, zeros, zeros.bool())

    auxiliary_branch = torch.rand(batch, device=device, generator=generator) < 0.40
    auxiliary_branch_time = torch.sigmoid(
        torch.randn(batch, device=device, generator=generator) - 1.2
    )
    video_branch_time = sample_video_time(batch, device, generator=generator)
    video_condition_auxiliary_time = 0.75 + 0.25 * torch.rand(
        batch, device=device, generator=generator
    )
    video_time = torch.where(auxiliary_branch, zeros, video_branch_time)
    auxiliary_time = torch.where(
        auxiliary_branch, auxiliary_branch_time, video_condition_auxiliary_time
    )
    # A1 sees its noisy auxiliary only on auxiliary-loss examples. L1 uses
    # symmetric fusion for every example. This is a per-example model input.
    auxiliary_condition = auxiliary_branch if arm == "A1" else ones.bool()
    return TrainingClocks(
        video_time,
        auxiliary_time,
        (~auxiliary_branch).float(),
        auxiliary_branch.float(),
        auxiliary_condition,
    )


def per_example_x_prediction_flow_mse(
    prediction_x: Tensor,
    noisy: Tensor,
    clean: Tensor,
    noise: Tensor,
    time: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Return one velocity-MSE scalar per example without mask renormalization."""
    if not (prediction_x.shape == noisy.shape == clean.shape == noise.shape):
        raise ValueError("flow tensors must share shape")
    t = expand_clock(time, noisy).to(noisy.dtype)
    denominator = (1.0 - t).clamp_min(FROZEN_CLEAN_TIME_EPS)
    predicted_velocity = (prediction_x - noisy) / denominator
    # Clamp both sides identically. Using clean-noise directly after clamping
    # would change the target for t > 0.95 and is not the released LF loss.
    target_velocity = (clean - noisy) / denominator
    error = (predicted_velocity - target_velocity).square()
    if valid_mask is None:
        return error.flatten(1).mean(1)
    mask = valid_mask.to(device=error.device, dtype=error.dtype)
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    try:
        mask = mask.expand_as(error)
    except RuntimeError as exc:
        raise ValueError("validity mask does not broadcast to flow tensor") from exc
    denominator = mask.flatten(1).sum(1)
    if torch.any(denominator <= 0):
        raise ValueError("every example must contain at least one valid target element")
    return (error * mask).flatten(1).sum(1) / denominator


def masked_branch_loss(per_example: Tensor, branch_mask: Tensor) -> Tensor:
    """Mean over the unchanged batch, matching released Latent Forcing."""
    if per_example.ndim != 1 or branch_mask.shape != per_example.shape:
        raise ValueError("per-example loss and branch mask must have shape [B]")
    return (per_example * branch_mask.to(per_example.dtype)).mean()


def model_config_payload(config: Any) -> dict[str, Any]:
    payload = _jsonable(config)
    if not isinstance(payload, dict):
        raise PocError("model config must resolve to a mapping")
    payload["initialization"] = FROZEN_INITIALIZATION
    return payload


def instantiate_model(args: argparse.Namespace) -> tuple[nn.Module, dict[str, Any]]:
    try:
        from robot_wm.modeling.video_latent_forcing import (
            VideoLatentForcingConfig,
            VideoLatentForcingModel,
        )
    except ImportError as exc:
        raise PocError("robot_wm.modeling.video_latent_forcing is unavailable") from exc

    signature = inspect.signature(VideoLatentForcingConfig)
    requested = {
        "hidden_size": args.width,
        "depth": args.depth,
        "num_heads": args.heads,
        "mlp_ratio": args.mlp_ratio,
        "parameter_matched_video_only": args.arm == "B0",
    }
    kwargs: dict[str, Any] = {}
    if "hidden_size" in signature.parameters:
        kwargs["hidden_size"] = requested["hidden_size"]
    elif "model_width" in signature.parameters:
        kwargs["model_width"] = requested["hidden_size"]
    else:
        raise PocError("model config does not expose an unambiguous hidden-size field")
    if "num_heads" in signature.parameters:
        kwargs["num_heads"] = requested["num_heads"]
    elif "heads" in signature.parameters:
        kwargs["heads"] = requested["num_heads"]
    for key in ("depth", "mlp_ratio", "parameter_matched_video_only"):
        if key in signature.parameters:
            kwargs[key] = requested[key]
    config = VideoLatentForcingConfig(**kwargs)
    model = VideoLatentForcingModel(config)
    return model, model_config_payload(config)


def model_forward(
    model: nn.Module,
    *,
    noisy_video: Tensor,
    noisy_auxiliary: Tensor | None,
    t_video: Tensor,
    t_auxiliary: Tensor,
    history: Tensor,
    actions: Tensor,
    condition_on_auxiliary: bool | Tensor,
) -> tuple[Tensor, Tensor]:
    output = model(
        noisy_video,
        noisy_auxiliary,
        t_video,
        t_auxiliary,
        history,
        actions,
        auxiliary_fusion_mask=condition_on_auxiliary,
    )
    if not hasattr(output, "video_x") or not hasattr(output, "auxiliary_x"):
        raise PocError("model output must expose video_x and auxiliary_x clean-state predictions")
    return output.video_x, output.auxiliary_x


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed((seed + worker_id) % (2**32))


def build_loader(
    dataset: DroidVideoLatentForcingDataset,
    *,
    context: DistributedContext,
    global_batch_size: int,
    seed: int,
    start_update: int,
    end_update: int,
    workers: int,
    micro_batch_size: int | None = None,
) -> DataLoader:
    sampler = DeterministicDistributedBatchSampler(
        len(dataset),
        global_batch_size=global_batch_size,
        rank=context.rank,
        world_size=context.world_size,
        seed=seed,
        start_update=start_update,
        end_update=end_update,
        micro_batch_size=micro_batch_size,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + context.rank)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=context.device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=_worker_seed,
        generator=generator,
    )


def move_training_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    required_shapes = {
        "history": (3, 5, 64, 112),
        "future": (3, 8, 64, 112),
        "actions": (16, 7),
        "lowres_scratchpad": (48, 8, 8, 14),
    }
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            result[key] = value.to(device=device, non_blocking=True)
        else:
            result[key] = value
    for key, shape in required_shapes.items():
        if key not in result or tuple(result[key].shape[1:]) != shape:
            raise PocError(f"batch {key} must have trailing shape {shape}, got {getattr(result.get(key), 'shape', None)}")
    return result


def optimizer_and_scheduler(model: nn.Module, args: argparse.Namespace, total_updates: int):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    def multiplier(update: int) -> float:
        if update < args.warmup_updates:
            return float(update + 1) / max(1, args.warmup_updates)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
    return optimizer, scheduler


class ModelEMA:
    """Warm-started full-state fp32 EMA used for every reported sample.

    A fixed 0.9999 decay is appropriate for the released Latent Forcing run's
    roughly 250k updates, but would retain about 61% of random initialization
    after this screen's 5k updates.  The deterministic warm-up below preserves
    the registered target decay while preventing that short-run bias.
    """

    def __init__(self, model: nn.Module, decay: float = FROZEN_EMA_DECAY) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie in (0, 1)")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow: dict[str, Tensor] = {}
        for name, value in unwrap_model(model).state_dict().items():
            clone = value.detach().clone()
            if clone.is_floating_point():
                clone = clone.float()
            self.shadow[name] = clone

    def decay_at_update(self, completed_updates: int) -> float:
        if completed_updates < 1:
            raise ValueError("EMA decay requires at least one completed update")
        warmup_decay = (1.0 + completed_updates) / (10.0 + completed_updates)
        return min(self.decay, warmup_decay)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = unwrap_model(model).state_dict()
        if current.keys() != self.shadow.keys():
            raise PocError("EMA/model state keys diverged")
        self.num_updates += 1
        update_decay = self.decay_at_update(self.num_updates)
        for name, value in current.items():
            target = self.shadow[name]
            source = value.detach().to(device=target.device)
            if target.is_floating_point():
                target.mul_(update_decay).add_(source.float(), alpha=1.0 - update_decay)
            else:
                target.copy_(source)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "schedule": FROZEN_EMA_SCHEDULE,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        updates = payload.get("num_updates")
        if (
            float(payload.get("decay", -1.0)) != self.decay
            or payload.get("schedule") != FROZEN_EMA_SCHEDULE
            or isinstance(updates, bool)
            or not isinstance(updates, int)
            or updates < 0
        ):
            raise PocError("checkpoint EMA schedule differs from resolved config")
        shadow = payload.get("shadow")
        if not isinstance(shadow, Mapping) or shadow.keys() != self.shadow.keys():
            raise PocError("checkpoint EMA state is incomplete")
        for name, value in shadow.items():
            if not isinstance(value, Tensor) or value.shape != self.shadow[name].shape:
                raise PocError(f"checkpoint EMA tensor mismatch: {name}")
            self.shadow[name].copy_(value.to(device=self.shadow[name].device))
        self.num_updates = updates

    def copy_to(self, model: nn.Module) -> None:
        target = unwrap_model(model)
        target.load_state_dict(
            {
                name: value.to(device=target.state_dict()[name].device, dtype=target.state_dict()[name].dtype)
                for name, value in self.shadow.items()
            },
            strict=True,
        )


def checkpoint_updates(command: str, arm: str) -> tuple[int, ...]:
    if command == "calibrate":
        return (CALIBRATION_UPDATES,)
    return PHASE1_CHECKPOINTS if arm == "phase1" else DUAL_CHECKPOINTS


def expected_updates(command: str, arm: str) -> int:
    if command == "calibrate":
        return CALIBRATION_UPDATES
    return PHASE1_UPDATES if arm == "phase1" else DUAL_UPDATES


def load_json(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PocError(f"{label} is not valid JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise PocError(f"{label} must contain a JSON object")
    return payload


def _verify_embedded_file_records(value: Any, *, label: str) -> int:
    """Recursively verify every explicit ``{path,sha256}`` evidence record."""
    verified = 0
    if isinstance(value, Mapping):
        if "path" in value or "sha256" in value:
            if not isinstance(value.get("path"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", str(value.get("sha256", ""))
            ):
                raise PocError(f"{label} contains a malformed file record")
            path = Path(value["path"]).expanduser().resolve()
            if not path.is_file() or sha256_file(path) != value["sha256"]:
                raise PocError(f"{label} evidence file is missing or changed: {path}")
            verified += 1
        for child in value.values():
            verified += _verify_embedded_file_records(child, label=label)
    elif isinstance(value, (list, tuple)):
        for child in value:
            verified += _verify_embedded_file_records(child, label=label)
    return verified


def validate_phase1_gate_record(path: str | Path, *, expected_commit: str) -> dict[str, Any]:
    payload = load_json(path, "Phase-1 gate record")
    selected = payload.get("selected_nfe_pair")
    if (
        payload.get("schema") != GATE_SCHEMA
        or payload.get("phase") != "phase1"
        or payload.get("status") != "pass"
        or payload.get("frozen") is not True
        or payload.get("validation_only") is not True
        or payload.get("phase1_gate_passed") is not True
        or payload.get("source_commit") != expected_commit
        or not isinstance(selected, list)
        or len(selected) != 2
        or not isinstance(selected[0], int)
        or selected[0] not in PHASE1_FRONTIER
        or selected[0] > 12
        or selected[1] != 0
    ):
        raise PocError("Phase-1 record does not prove the frozen autonomous gate passed")
    recorded_decision = payload.get("decision_sha256")
    recomputed_decision = sha256_json(
        {key: value for key, value in payload.items() if key != "decision_sha256"}
    )
    if recorded_decision != recomputed_decision:
        raise PocError("Phase-1 gate decision digest is missing or changed")
    checkpoint_record = payload.get("checkpoint")
    if (
        not isinstance(checkpoint_record, Mapping)
        or _verify_embedded_file_records(
            checkpoint_record, label="Phase-1 checkpoint"
        )
        < 1
    ):
        raise PocError("Phase-1 gate lacks a hashed checkpoint")
    if _verify_embedded_file_records(payload.get("evaluation"), label="Phase-1 evaluation") < 1:
        raise PocError("Phase-1 gate lacks hashed validation evidence")
    checkpoint_path = Path(checkpoint_record["path"]).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_ema = checkpoint.get("ema")
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("arm") != "phase1"
        or checkpoint.get("completed_updates") != PHASE1_UPDATES
        or not isinstance(checkpoint_ema, Mapping)
        or checkpoint_ema.get("decay") != FROZEN_EMA_DECAY
        or checkpoint_ema.get("schedule") != FROZEN_EMA_SCHEDULE
        or checkpoint_ema.get("num_updates") != PHASE1_UPDATES
        or not isinstance(checkpoint_ema.get("shadow"), Mapping)
    ):
        raise PocError("Phase-1 gate checkpoint is not a Phase-1 model")
    evaluation = payload.get("evaluation")
    evaluation_root = evaluation.get("root") if isinstance(evaluation, Mapping) else None
    if not isinstance(evaluation_root, str):
        raise PocError("Phase-1 gate lacks its immutable evaluation root")
    try:
        from tools.analyze_video_latent_forcing_poc import (
            GateError,
            analyze_phase1_evaluation,
        )

        reproduced = analyze_phase1_evaluation(evaluation_root)
    except (GateError, OSError, ValueError, subprocess.SubprocessError) as exc:
        raise PocError(f"Phase-1 gate evidence cannot be reproduced: {exc}") from exc
    if canonical_json(reproduced) != canonical_json(payload):
        raise PocError("Phase-1 gate record differs from the decision recomputed from raw evidence")
    return payload


def validate_calibration_record(
    path: str | Path,
    arm: str,
    *,
    expected_experiment_identity_sha256: str,
    expected_commit: str,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = load_json(path, "calibration record")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("command") != "calibrate"
        or payload.get("arm") != arm
        or payload.get("completed_updates") != CALIBRATION_UPDATES
        or payload.get("status") != "complete"
        or payload.get("nonfinite_updates") != 0
    ):
        raise PocError(f"calibration record does not prove a clean exact-200 {arm} run")
    run_dir = resolved.parent
    config_path = run_dir / "resolved_config.json"
    checkpoint_path = run_dir / "checkpoints" / f"update_{CALIBRATION_UPDATES:06d}.pt"
    config = load_json(config_path, "calibration resolved config")
    calibration_source = config.get("source")
    if (
        config.get("experiment_identity_sha256")
        != expected_experiment_identity_sha256
        or not isinstance(calibration_source, Mapping)
        or calibration_source.get("commit") != expected_commit
        or calibration_source.get("dirty") is not False
        or payload.get("resolved_config") != file_record(config_path)
        or payload.get("checkpoint") != file_record(checkpoint_path)
    ):
        raise PocError("calibration is not bound to this exact experiment identity")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_ema = checkpoint.get("ema")
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("arm") != arm
        or checkpoint.get("completed_updates") != CALIBRATION_UPDATES
        or checkpoint.get("config_sha256") != sha256_json(config)
        or checkpoint.get("model_config") != config.get("model")
        or not isinstance(checkpoint_ema, Mapping)
        or checkpoint_ema.get("decay") != FROZEN_EMA_DECAY
        or checkpoint_ema.get("schedule") != FROZEN_EMA_SCHEDULE
        or checkpoint_ema.get("num_updates") != CALIBRATION_UPDATES
    ):
        raise PocError("calibration checkpoint does not match its resolved config")
    return payload


def _manifest_record(
    path: str | Path,
    split: str,
    *,
    data_root: str | Path,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    rows = read_clip_manifest(resolved, expected_split=split)
    provenance_path = resolved.parent / "provenance.json"
    provenance = load_json(provenance_path, "DROID POC data provenance")
    expected_clips = {
        "train": FROZEN_TRAIN_CLIPS,
        "val": FROZEN_VALIDATION_CLIPS,
        "test": FROZEN_VALIDATION_CLIPS,
    }
    expected_episodes = {"train": 8_000, "val": 890, "test": 890}
    ordered_episode_ids = list(
        dict.fromkeys(int(row["episode_index"]) for row in rows)
    )
    episode_ids_sha256 = hashlib.sha256(
        json.dumps(ordered_episode_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    provenance_manifests = provenance.get("manifests")
    builder_source = provenance.get("builder_source")
    manifest_entry = (
        provenance_manifests.get(split)
        if isinstance(provenance_manifests, Mapping)
        else None
    )
    expected_manifest_name = "test.identifiers.jsonl" if split == "test" else f"{split}.jsonl"
    if (
        provenance.get("schema") != DROID_DATA_SCHEMA
        or provenance.get("complete") is not True
        or provenance.get("camera") != DROID_CAMERA
        or provenance.get("camera_count") != 1
        or not isinstance(builder_source, Mapping)
        or builder_source.get("commit") != git_record()["commit"]
        or builder_source.get("dirty") is not False
        or builder_source.get("builder_tool_sha256")
        != sha256_file(REPO_ROOT / "tools/build_video_latent_forcing_droid.py")
        or provenance.get("split_rank_expression")
        != "sha256('video-latent-forcing-poc-v1:<episode_id>')"
        or provenance.get("clip_start_seed") != 20260801
        or provenance.get("clips_per_episode")
        != {"train": 8, "val": 1, "test": 1}
        or provenance.get("minimum_episode_frames") != 66
        or provenance.get("eligible_episode_count") != 9_780
        or provenance.get("eligible_inventory_sha256")
        != DROID_ELIGIBLE_INVENTORY_SHA256
        or provenance.get("split_counts") != {"train": 8_000, "val": 890, "test": 890}
        or provenance.get("split_episode_ids_sha256")
        != DROID_SPLIT_EPISODE_IDS_SHA256
        or episode_ids_sha256 != DROID_SPLIT_EPISODE_IDS_SHA256[split]
        or Path(str(provenance.get("source_root", ""))).expanduser().resolve()
        != Path(data_root).expanduser().resolve()
        or not isinstance(manifest_entry, Mapping)
        or manifest_entry.get("path") != expected_manifest_name
        or manifest_entry.get("sha256") != sha256_file(resolved)
        or manifest_entry.get("clip_count") != expected_clips[split]
        or manifest_entry.get("episode_count") != expected_episodes[split]
        or len(rows) != expected_clips[split]
        or (split in {"train", "val"} and manifest_entry.get("cached") is not True)
        or (split == "test" and manifest_entry.get("cached") is not False)
    ):
        raise PocError(f"{split} manifest does not match the frozen DROID POC population")
    return resolved, rows, {
        **file_record(resolved),
        "split": split,
        "clips": len(rows),
        "data_provenance": file_record(provenance_path),
    }


def validate_training_manifests(
    train: str | Path,
    validation: str | Path,
    *,
    data_root: str | Path,
):
    train_path, train_rows, train_record = _manifest_record(
        train, "train", data_root=data_root
    )
    val_path, val_rows, val_record = _manifest_record(
        validation, "val", data_root=data_root
    )
    if not rows_episode_ids(train_rows).isdisjoint(rows_episode_ids(val_rows)):
        raise PocError("train and validation manifests overlap by episode")
    return train_path, val_path, {"train": train_record, "validation": val_record}


def resolved_training_config(
    args: argparse.Namespace,
    context: DistributedContext,
    *,
    model_config: Mapping[str, Any],
    manifests: Mapping[str, Any],
    total_updates: int,
) -> dict[str, Any]:
    local_optimizer_batch = args.global_batch_size // context.world_size
    micro_batch_size = args.micro_batch_size or local_optimizer_batch
    config = {
        "schema": SCHEMA,
        "source": git_record(),
        "command": args.command,
        "arm": args.arm,
        "clock_convention": CLOCK_CONVENTION,
        "clean_time_epsilon": FROZEN_CLEAN_TIME_EPS,
        "seed": args.seed,
        "updates": total_updates,
        "checkpoint_updates": list(checkpoint_updates(args.command, args.arm)),
        "global_batch_size": args.global_batch_size,
        "world_size": context.world_size,
        "local_optimizer_batch_size": local_optimizer_batch,
        "micro_batch_size_per_rank": micro_batch_size,
        "gradient_accumulation_steps": local_optimizer_batch // micro_batch_size,
        "dtype": "bfloat16",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "warmup_updates": args.warmup_updates,
            "after_warmup": "constant",
            "betas": [0.9, 0.95],
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
        },
        "ema": {
            "decay": args.ema_decay,
            "schedule": FROZEN_EMA_SCHEDULE,
            "reported_samples_use": True,
            "short_run_initialization_bias_corrected": True,
        },
        "schedule": {
            "auxiliary_branch_probability": 0.4,
            "auxiliary_logit_normal_mean": -1.2,
            "auxiliary_logit_normal_std": 1.0,
            "video_logit_normal_mean": -0.4,
            "video_logit_normal_std": 0.8,
            "video_low_time_replacement_probability": 0.1,
            "video_low_time_interval": [0.0, 0.5],
            "video_branch_auxiliary_time_interval": [0.75, 1.0],
            "auxiliary_loss_coefficient": 0.333,
            "loss_mask_normalization": "unchanged_global_batch",
        },
        "model": dict(model_config),
        "manifests": dict(manifests),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "workers_per_rank": args.workers,
        "wandb": {
            "enabled": args.wandb,
            "entity": args.wandb_entity if args.wandb else None,
            "project": args.wandb_project if args.wandb else None,
            "private_project_acknowledged": args.wandb_private_project_ack,
        },
        "phase1_gate_record": (
            file_record(args.phase1_gate_record)
            if args.phase1_gate_record is not None
            else None
        ),
        "calibration_record": (
            file_record(args.calibration_record)
            if args.command == "train" and args.calibration_record is not None
            else None
        ),
        "primary_metrics_implemented": ["rgb_mse", "rgb_psnr", "temporal_difference_mse", "auxiliary_nmse", "auxiliary_cosine"],
        "required_primary_metrics_missing": list(MISSING_PRIMARY_METRICS),
    }
    identity_keys = (
        "source",
        "arm",
        "clock_convention",
        "clean_time_epsilon",
        "seed",
        "global_batch_size",
        "world_size",
        "local_optimizer_batch_size",
        "micro_batch_size_per_rank",
        "gradient_accumulation_steps",
        "dtype",
        "optimizer",
        "ema",
        "schedule",
        "model",
        "manifests",
        "data_root",
        "workers_per_rank",
    )
    config["experiment_identity_sha256"] = sha256_json(
        {key: config[key] for key in identity_keys}
    )
    return config


class LocalAndOptionalWandbLogger:
    def __init__(self, run_dir: Path, args: argparse.Namespace, config: Mapping[str, Any], *, primary: bool) -> None:
        self.path = run_dir / "metrics.jsonl"
        self.run = None
        self._wandb = None
        if not primary or not args.wandb:
            return
        if not args.wandb_entity or not args.wandb_project or not args.wandb_private_project_ack:
            raise PocError("W&B requires explicit entity/project and --wandb-private-project-ack")
        try:
            import wandb
        except ImportError as exc:
            raise PocError("--wandb requested but wandb is not installed") from exc
        self.run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.run_id,
            group=None,
            dir=str(run_dir),
            config=_jsonable(config),
            settings=wandb.Settings(start_method="thread", save_code=False),
        )
        self._wandb = wandb

    def log(self, payload: Mapping[str, Any], *, primary: bool) -> None:
        if not primary:
            return
        append_jsonl(self.path, payload)
        if self.run is not None:
            self.run.log(dict(payload), step=int(payload.get("update", 0)))

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

    def log_media(self, records: Sequence[Mapping[str, Any]], *, step: int) -> None:
        """Mirror already-persisted local media to W&B when explicitly enabled."""
        if self.run is None or self._wandb is None:
            return
        for index, record in enumerate(records):
            payload: dict[str, Any] = {}
            stem = (
                f"samples/{record['arm']}/{record['control']}/"
                f"a{record['auxiliary_nfe']}_v{record['video_nfe']}/{index}"
            )
            if record.get("rgb_mp4"):
                payload[f"{stem}/rgb_video"] = self._wandb.Video(
                    record["rgb_mp4"]["path"], fps=4, format="mp4"
                )
            if record.get("rgb_grid"):
                payload[f"{stem}/rgb_grid"] = self._wandb.Image(record["rgb_grid"]["path"])
            if record.get("scratchpad_mp4"):
                payload[f"{stem}/scratchpad_video"] = self._wandb.Video(
                    record["scratchpad_mp4"]["path"], fps=4, format="mp4"
                )
            if payload:
                self.run.log(payload, step=step)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    ema: ModelEMA,
    update: int,
    arm: str,
    model_config: Mapping[str, Any],
    config_sha256: str,
    context: DistributedContext,
    cumulative_optimizer_wall_seconds: float = 0.0,
) -> None:
    if ema.num_updates != update:
        raise PocError(
            f"EMA update count {ema.num_updates} differs from checkpoint update {update}"
        )
    rng_by_rank = context.gather_objects(capture_rng_state())
    if context.is_primary:
        atomic_torch_save(
            path,
            {
                "schema": CHECKPOINT_SCHEMA,
                "completed_updates": int(update),
                "arm": arm,
                "model_config": dict(model_config),
                "config_sha256": config_sha256,
                "cumulative_optimizer_wall_seconds": float(
                    cumulative_optimizer_wall_seconds
                ),
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.state_dict(),
                "rng_by_rank": rng_by_rank,
            },
        )
        atomic_write_json(path.with_suffix(path.suffix + ".json"), {**file_record(path), "completed_updates": update})
    context.barrier()


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    ema: ModelEMA | None,
    expected_config_sha256: str | None,
    context: DistributedContext,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise PocError(f"cannot load checkpoint {resolved}: {exc}") from exc
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise PocError("checkpoint schema mismatch")
    completed_updates = payload.get("completed_updates")
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or completed_updates < 0
    ):
        raise PocError("checkpoint completed-update count is invalid")
    if expected_config_sha256 is not None and payload.get("config_sha256") != expected_config_sha256:
        raise PocError("checkpoint resolved-config identity mismatch")
    unwrap_model(model).load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if ema is not None:
        if "ema" not in payload:
            raise PocError("checkpoint is missing required EMA state")
        ema.load_state_dict(payload["ema"])
        if ema.num_updates != completed_updates:
            raise PocError(
                "checkpoint EMA update count differs from completed updates"
            )
    states = payload.get("rng_by_rank")
    if not isinstance(states, list) or len(states) != context.world_size:
        raise PocError("checkpoint RNG state does not match current world size")
    restore_rng_state(states[context.rank])
    return payload


def reconcile_resume_artifacts(run_dir: Path, completed_updates: int) -> int:
    """Drop only post-checkpoint log rows and reject later checkpoints."""
    later_checkpoints = []
    for path in (run_dir / "checkpoints").glob("update_*.pt"):
        match = re.fullmatch(r"update_(\d+)\.pt", path.name)
        if match and int(match.group(1)) > completed_updates:
            later_checkpoints.append(path)
    if later_checkpoints:
        raise PocError(
            "resume checkpoint is not the latest saved checkpoint; refusing to overwrite "
            + ", ".join(str(path) for path in sorted(later_checkpoints))
        )
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return 0
    kept: list[str] = []
    removed = 0
    for line_number, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PocError(f"invalid metrics JSONL at line {line_number}") from exc
        if int(row.get("update", -1)) <= completed_updates:
            kept.append(canonical_json(row))
        else:
            removed += 1
    if removed:
        _atomic_write_text(metrics_path, "".join(f"{line}\n" for line in kept))
    return removed


def _autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def training_step(model: nn.Module, batch: Mapping[str, Any], arm: str) -> tuple[Tensor, dict[str, Tensor]]:
    clean_video = batch["future"]
    clean_auxiliary = batch["lowres_scratchpad"]
    batch_size = clean_video.shape[0]
    clocks = sample_training_clocks(arm, batch_size, clean_video.device)
    video_noise = torch.randn_like(clean_video)
    auxiliary_noise = torch.randn_like(clean_auxiliary)
    noisy_video = corrupt_clean_time(clean_video, video_noise, clocks.video_time)
    noisy_auxiliary = corrupt_clean_time(clean_auxiliary, auxiliary_noise, clocks.auxiliary_time)
    condition: bool | Tensor
    if arm == "B0":
        condition = False
    elif arm == "L1" or arm == "phase1":
        condition = True
    else:
        condition = clocks.auxiliary_condition_mask
    video_x, auxiliary_x = model_forward(
        model,
        noisy_video=noisy_video,
        noisy_auxiliary=noisy_auxiliary,
        t_video=clocks.video_time,
        t_auxiliary=clocks.auxiliary_time,
        history=batch["history"],
        actions=batch["actions"],
        condition_on_auxiliary=condition,
    )
    video_per_example = per_example_x_prediction_flow_mse(
        video_x, noisy_video, clean_video, video_noise, clocks.video_time
    )
    auxiliary_per_example = per_example_x_prediction_flow_mse(
        auxiliary_x, noisy_auxiliary, clean_auxiliary, auxiliary_noise, clocks.auxiliary_time
    )
    video_loss = masked_branch_loss(video_per_example, clocks.video_loss_mask)
    auxiliary_loss = masked_branch_loss(auxiliary_per_example, clocks.auxiliary_loss_mask)
    loss = video_loss + 0.333 * auxiliary_loss
    return loss, {
        "video_loss": video_loss.detach(),
        "auxiliary_loss": auxiliary_loss.detach(),
        "video_branch_count": clocks.video_loss_mask.sum().detach(),
        "auxiliary_branch_count": clocks.auxiliary_loss_mask.sum().detach(),
    }


def training_command(args: argparse.Namespace) -> int:
    context = initialize_distributed()
    logger: LocalAndOptionalWandbLogger | None = None
    try:
        total_updates = expected_updates(args.command, args.arm)
        if args.global_batch_size % context.world_size:
            raise PocError("global batch size must divide by torchrun world size")
        run_dir = validated_run_dir(args.artifact_root, args.run_id, resume=args.resume is not None)
        train_manifest, val_manifest, manifest_records = validate_training_manifests(
            args.train_manifest,
            args.validation_manifest,
            data_root=args.data_root,
        )
        # Every rank constructs identical parameters. Rank-specific stochastic
        # training streams are installed only after model/optimizer/EMA setup.
        seed_everything(args.seed, 0)
        model, model_config = instantiate_model(args)
        source_record = git_record()
        if source_record["dirty"]:
            raise PocError("training requires a clean, committed Git source tree")
        if count_parameters(model)["total"] != FROZEN_PARAMETER_COUNT:
            raise PocError("model parameter count differs from the frozen 41,963,760")
        model.to(context.device)
        optimizer, scheduler = optimizer_and_scheduler(model, args, total_updates)
        ema = ModelEMA(model, decay=args.ema_decay)
        start_update = 0
        prior_optimizer_wall_seconds = 0.0
        config = resolved_training_config(
            args, context, model_config=model_config, manifests=manifest_records, total_updates=total_updates
        )
        config_sha256 = sha256_json(config)
        if args.command == "train":
            validate_calibration_record(
                args.calibration_record,
                args.arm,
                expected_experiment_identity_sha256=config[
                    "experiment_identity_sha256"
                ],
                expected_commit=source_record["commit"],
            )
            if args.arm in {"B0", "A1", "L1"}:
                gate_error: str | None = None
                if context.is_primary:
                    try:
                        validate_phase1_gate_record(
                            args.phase1_gate_record,
                            expected_commit=source_record["commit"],
                        )
                    except Exception as exc:  # broadcast one fail-closed verdict
                        gate_error = f"{type(exc).__name__}: {exc}"
                if context.world_size > 1:
                    verdict: list[Any] = [gate_error]
                    torch_dist.broadcast_object_list(verdict, src=0)
                    gate_error = verdict[0]
                if gate_error is not None:
                    raise PocError(f"distributed Phase-1 gate validation failed: {gate_error}")
        if args.resume is not None:
            existing = load_json(run_dir / "resolved_config.json", "resolved config")
            if sha256_json(existing) != config_sha256:
                raise PocError("resume arguments differ from the immutable resolved config")
            payload = load_checkpoint(
                args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                expected_config_sha256=config_sha256,
                context=context,
            )
            start_update = int(payload["completed_updates"])
            prior_optimizer_wall_seconds = float(
                payload.get("cumulative_optimizer_wall_seconds", 0.0)
            )
            if not 0 <= start_update <= total_updates:
                raise PocError("resume checkpoint has an invalid completed-update count")
            if start_update == total_updates and (run_dir / "complete.json").exists():
                raise PocError("run is already complete; finalize-only resume is unnecessary")
            if context.is_primary:
                removed_rows = reconcile_resume_artifacts(run_dir, start_update)
                if removed_rows:
                    atomic_write_json(
                        run_dir / f"resume_reconciliation_{start_update:06d}.json",
                        {
                            "checkpoint": file_record(args.resume),
                            "completed_updates": start_update,
                            "discarded_post_checkpoint_metric_rows": removed_rows,
                        },
                        exclusive=True,
                    )
            context.barrier()
        else:
            if context.is_primary:
                run_dir.mkdir(parents=True, exist_ok=False)
                atomic_write_json(run_dir / "resolved_config.json", config, exclusive=True)
                atomic_write_json(
                    run_dir / "provenance.json",
                    {
                        "schema": SCHEMA,
                        "git": git_record(),
                        "runtime": runtime_record(),
                        "command": [sys.executable, *sys.argv],
                        "resolved_config_sha256": config_sha256,
                        "secrets_persisted": False,
                    },
                    exclusive=True,
                )
            context.barrier()
            seed_everything(args.seed, context.rank)

        dataset = DroidVideoLatentForcingDataset(train_manifest, args.data_root)
        loader = build_loader(
            dataset,
            context=context,
            global_batch_size=args.global_batch_size,
            seed=args.seed,
            start_update=start_update,
            end_update=total_updates,
            workers=args.workers,
            micro_batch_size=args.micro_batch_size,
        )
        if context.world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=args.arm in {"phase1", "B0", "A1"},
            )
        logger = LocalAndOptionalWandbLogger(run_dir, args, config, primary=context.is_primary)
        checkpoints = set(checkpoint_updates(args.command, args.arm))
        local_optimizer_batch = args.global_batch_size // context.world_size
        micro_batch_size = args.micro_batch_size or local_optimizer_batch
        accumulation_steps = local_optimizer_batch // micro_batch_size
        nonfinite_updates = 0
        model.train()
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        loader_iterator = iter(loader)
        optimizer_wall_start = time.perf_counter()
        cumulative_optimizer_wall_seconds = prior_optimizer_wall_seconds
        for update_index in range(start_update + 1, total_updates + 1):
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = torch.zeros((), device=context.device)
            accumulated_video_loss = torch.zeros((), device=context.device)
            accumulated_auxiliary_loss = torch.zeros((), device=context.device)
            video_branch_count = torch.zeros((), device=context.device)
            auxiliary_branch_count = torch.zeros((), device=context.device)
            for microstep in range(accumulation_steps):
                raw_batch = next(loader_iterator)
                batch = move_training_batch(raw_batch, context.device)
                sync_context = (
                    model.no_sync()
                    if isinstance(model, DistributedDataParallel) and microstep + 1 < accumulation_steps
                    else contextlib.nullcontext()
                )
                with sync_context, _autocast(context.device):
                    micro_loss, telemetry = training_step(model, batch, args.arm)
                    loss = micro_loss / accumulation_steps
                if not torch.isfinite(loss):
                    nonfinite_updates += 1
                    raise PocError(f"nonfinite loss at update {update_index}, microstep {microstep}")
                loss.backward()
                accumulated_loss += micro_loss.detach() / accumulation_steps
                accumulated_video_loss += telemetry["video_loss"] / accumulation_steps
                accumulated_auxiliary_loss += telemetry["auxiliary_loss"] / accumulation_steps
                video_branch_count += telemetry["video_branch_count"]
                auxiliary_branch_count += telemetry["auxiliary_branch_count"]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                nonfinite_updates += 1
                raise PocError(f"nonfinite gradient norm at update {update_index}")
            optimizer.step()
            scheduler.step()
            ema.update(model)
            packed = torch.stack(
                [
                    accumulated_loss.float(),
                    accumulated_video_loss.float(),
                    accumulated_auxiliary_loss.float(),
                    video_branch_count.float(),
                    auxiliary_branch_count.float(),
                ]
            )
            packed = context.sum_tensor(packed)
            should_observe = (
                update_index == 1
                or update_index % args.log_every == 0
                or update_index in checkpoints
            )
            if should_observe:
                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                cumulative_optimizer_wall_seconds = (
                    prior_optimizer_wall_seconds
                    + time.perf_counter()
                    - optimizer_wall_start
                )
            if should_observe:
                logger.log(
                    {
                        "update": update_index,
                        "loss": float(packed[0] / context.world_size),
                        "video_loss": float(packed[1] / context.world_size),
                        "auxiliary_loss": float(packed[2] / context.world_size),
                        "video_branch_examples": int(packed[3].item()),
                        "auxiliary_branch_examples": int(packed[4].item()),
                        "gradient_norm": float(gradient_norm.detach()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "cumulative_optimizer_wall_seconds": cumulative_optimizer_wall_seconds,
                        "cumulative_examples_per_second": (
                            update_index
                            * args.global_batch_size
                            / max(cumulative_optimizer_wall_seconds, 1e-12)
                        ),
                        "peak_gpu_memory_allocated_bytes": (
                            int(torch.cuda.max_memory_allocated(context.device))
                            if context.device.type == "cuda"
                            else None
                        ),
                        "peak_gpu_memory_reserved_bytes": (
                            int(torch.cuda.max_memory_reserved(context.device))
                            if context.device.type == "cuda"
                            else None
                        ),
                    },
                    primary=context.is_primary,
                )
            if update_index in checkpoints:
                save_checkpoint(
                    run_dir / "checkpoints" / f"update_{update_index:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema,
                    update=update_index,
                    arm=args.arm,
                    model_config=model_config,
                    config_sha256=config_sha256,
                    context=context,
                    cumulative_optimizer_wall_seconds=cumulative_optimizer_wall_seconds,
                )

        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        cumulative_optimizer_wall_seconds = (
            prior_optimizer_wall_seconds + time.perf_counter() - optimizer_wall_start
        )
        if context.is_primary:
            atomic_write_json(
                run_dir / "complete.json",
                {
                    "schema": SCHEMA,
                    "status": "complete",
                    "command": args.command,
                    "arm": args.arm,
                    "completed_updates": total_updates,
                    "nonfinite_updates": nonfinite_updates,
                    "cumulative_optimizer_wall_seconds": cumulative_optimizer_wall_seconds,
                    "cumulative_examples_per_second": (
                        total_updates
                        * args.global_batch_size
                        / max(cumulative_optimizer_wall_seconds, 1e-12)
                    ),
                    "clock_convention": CLOCK_CONVENTION,
                    "resolved_config_sha256": config_sha256,
                    "experiment_identity_sha256": config[
                        "experiment_identity_sha256"
                    ],
                    "source": source_record,
                    "resolved_config": file_record(run_dir / "resolved_config.json"),
                    "checkpoint": file_record(
                        run_dir / "checkpoints" / f"update_{total_updates:06d}.pt"
                    ),
                    "parameter_counts": count_parameters(unwrap_model(model)),
                    "validation_manifest": file_record(val_manifest),
                    "quality_claim_evaluable": False,
                    "required_primary_metrics_missing": list(MISSING_PRIMARY_METRICS),
                },
                exclusive=True,
            )
        context.barrier()
        return 0
    finally:
        if logger is not None:
            logger.finish()
        close_distributed(context)


def stable_noise_like(reference: Tensor, clip_ids: Sequence[str], seed: int, stream: str) -> Tensor:
    if reference.shape[0] != len(clip_ids):
        raise ValueError("clip ID count must match noise batch")
    if stream not in {"video", "aux"}:
        raise ValueError("fixed-noise stream must be 'video' or 'aux'")
    samples = []
    for index, clip_id in enumerate(clip_ids):
        # This exact clip-addressed key is part of the paired-evaluation
        # contract and is intentionally independent of loader/rank ordering.
        digest = hashlib.sha256(f"{clip_id}:{seed}:{stream}".encode("utf-8")).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "big") & ((1 << 63) - 1))
        samples.append(torch.randn(reference[index].shape, generator=generator, dtype=torch.float32))
    return torch.stack(samples).to(device=reference.device, dtype=reference.dtype)


@dataclass(frozen=True)
class CascadeResult:
    video: Tensor
    generated_auxiliary: Tensor | None
    conditioning_auxiliary: Tensor | None
    initial_video_noise: Tensor
    initial_auxiliary_noise: Tensor | None
    model_calls: int
    phase_boundary_sha256: str | None


def tensor_sha256(tensor: Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _rgb_video_uint8(video: Tensor) -> np.ndarray:
    """Convert normalized ``[3,T,H,W]`` RGB to contiguous ``[T,H,W,3]``."""
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError("RGB video must have shape [3,T,H,W]")
    return (
        video.detach()
        .float()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .cpu()
        .contiguous()
        .numpy()
    )


def write_rgb_mp4(path: Path, video: Tensor, *, fps: int = 4) -> None:
    """Write a small H.264 artifact with the system ffmpeg binary."""
    frames = _rgb_video_uint8(video)
    frame_count, height, width, channels = frames.shape
    if frame_count < 1 or channels != 3 or height % 2 or width % 2:
        raise PocError("MP4 media requires nonempty RGB frames with even dimensions")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(temporary),
        ]
        result = subprocess.run(command, input=frames.tobytes(), capture_output=True)
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace")[-2000:]
            raise PocError(f"ffmpeg failed to encode {path}: {detail}")
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def write_rgb_grid(path: Path, video: Tensor) -> None:
    """Persist all temporal frames as a single lossless horizontal PNG grid."""
    from PIL import Image

    frames = _rgb_video_uint8(video)
    grid = np.concatenate([frame for frame in frames], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".png", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        Image.fromarray(grid, mode="RGB").save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def save_evaluation_media(
    output_dir: Path,
    *,
    clip_id: str,
    arm: str,
    control: str,
    auxiliary_nfe: int,
    video_nfe: int,
    video: Tensor,
    generated_auxiliary: Tensor | None,
    conditioning_auxiliary: Tensor | None,
    initial_video_noise: Tensor,
    initial_auxiliary_noise: Tensor | None,
) -> dict[str, Any]:
    """Save deployable/control media while refusing clean-future oracle payloads."""
    if control == "oracle_clean":
        raise PocError("oracle-clean RGB/scratchpad media must not be persisted")
    contains_clean_future_rgb = False
    sample_dir = (
        output_dir
        / "samples"
        / f"a{auxiliary_nfe:03d}_v{video_nfe:03d}"
        / clip_id
    )
    rgb_mp4 = sample_dir / f"{control}_rgb.mp4"
    rgb_grid = sample_dir / f"{control}_rgb_grid.png"
    write_rgb_mp4(rgb_mp4, video)
    write_rgb_grid(rgb_grid, video)
    scratchpad_record: dict[str, Any] | None = None
    if conditioning_auxiliary is not None:
        scratchpad_video = unpatchify_lowres_rgb(conditioning_auxiliary.unsqueeze(0))[0]
        scratchpad_mp4 = sample_dir / f"{control}_scratchpad.mp4"
        write_rgb_mp4(scratchpad_mp4, scratchpad_video)
        scratchpad_record = file_record(scratchpad_mp4)
    boundary_path = sample_dir / f"{control}_phase_boundary.pt"
    atomic_torch_save(
        boundary_path,
        {
            "clip_id": clip_id,
            "arm": arm,
            "control": control,
            "deployable": control != "oracle_clean",
            "generated_auxiliary": (
                generated_auxiliary.detach().cpu() if generated_auxiliary is not None else None
            ),
            "conditioning_auxiliary": (
                conditioning_auxiliary.detach().cpu()
                if conditioning_auxiliary is not None
                else None
            ),
            "initial_video_noise": initial_video_noise.detach().cpu(),
            "initial_auxiliary_noise": (
                initial_auxiliary_noise.detach().cpu()
                if initial_auxiliary_noise is not None
                else None
            ),
            "contains_clean_future_rgb": contains_clean_future_rgb,
        },
    )
    return {
        "clip_id": clip_id,
        "arm": arm,
        "control": control,
        "auxiliary_nfe": auxiliary_nfe,
        "video_nfe": video_nfe,
        "rgb_mp4": file_record(rgb_mp4),
        "rgb_grid": file_record(rgb_grid),
        "scratchpad_mp4": scratchpad_record,
        "phase_boundary_tensor": file_record(boundary_path),
        "local_artifacts_authoritative": True,
        "contains_clean_future_rgb": contains_clean_future_rgb,
    }


@torch.inference_mode()
def sample_auxiliary_phase(
    model: nn.Module,
    history: Tensor,
    actions: Tensor,
    *,
    video_noise: Tensor,
    auxiliary_noise: Tensor,
    steps: int,
) -> tuple[Tensor, int]:
    if steps < 1:
        raise ValueError("auxiliary phase requires at least one model call")
    video = video_noise.clone()
    auxiliary = auxiliary_noise.clone()
    batch = video.shape[0]
    schedule = torch.linspace(0.0, 1.0, steps + 1, device=video.device)
    for index in range(steps):
        ta = torch.full((batch,), schedule[index], device=video.device)
        ta_next = torch.full((batch,), schedule[index + 1], device=video.device)
        tv = torch.zeros(batch, device=video.device)
        _, auxiliary_x = model_forward(
            model,
            noisy_video=video,
            noisy_auxiliary=auxiliary,
            t_video=tv,
            t_auxiliary=ta,
            history=history,
            actions=actions,
            condition_on_auxiliary=True,
        )
        auxiliary = clean_time_euler_from_x(auxiliary, auxiliary_x, ta, ta_next)
        if not torch.equal(video, video_noise):
            raise PocError("video changed during auxiliary-only phase")
    return auxiliary, steps


@torch.inference_mode()
def sample_video_phase(
    model: nn.Module,
    history: Tensor,
    actions: Tensor,
    *,
    video_noise: Tensor,
    frozen_auxiliary: Tensor | None,
    steps: int,
    condition_on_auxiliary: bool,
) -> tuple[Tensor, int]:
    if steps < 1:
        raise ValueError("video phase requires at least one model call")
    video = video_noise.clone()
    auxiliary = frozen_auxiliary.clone() if frozen_auxiliary is not None else None
    boundary = auxiliary.clone() if auxiliary is not None else None
    batch = video.shape[0]
    schedule = torch.linspace(0.0, 1.0, steps + 1, device=video.device)
    for index in range(steps):
        tv = torch.full((batch,), schedule[index], device=video.device)
        tv_next = torch.full((batch,), schedule[index + 1], device=video.device)
        ta = torch.ones(batch, device=video.device)
        video_x, _ = model_forward(
            model,
            noisy_video=video,
            noisy_auxiliary=auxiliary,
            t_video=tv,
            t_auxiliary=ta,
            history=history,
            actions=actions,
            condition_on_auxiliary=condition_on_auxiliary,
        )
        video = clean_time_euler_from_x(video, video_x, tv, tv_next)
        if auxiliary is not None and not torch.equal(auxiliary, boundary):
            raise PocError("auxiliary state changed during video-only phase")
    return video, steps


@torch.inference_mode()
def sample_control(
    model: nn.Module,
    arm: str,
    history: Tensor,
    actions: Tensor,
    clean_auxiliary: Tensor | None,
    *,
    control: str,
    auxiliary_steps: int,
    video_steps: int,
    video_noise: Tensor,
    auxiliary_noise: Tensor | None,
    generated_auxiliary: Tensor | None = None,
    shuffle_indices: Tensor | None = None,
) -> CascadeResult:
    if control not in CONTROLS:
        raise ValueError(f"unknown control: {control}")
    calls = 0
    if arm == "B0":
        if auxiliary_steps != 0 or video_steps < 1 or control != "off":
            raise PocError("B0 evaluation is video-only and must use control=off")
        generated = None
        conditioning = None
        video, used = sample_video_phase(
            model,
            history,
            actions,
            video_noise=video_noise,
            frozen_auxiliary=conditioning,
            steps=video_steps,
            condition_on_auxiliary=False,
        )
        calls += used
    elif arm == "phase1":
        if video_steps != 0 or auxiliary_steps < 1:
            raise PocError("phase1 evaluation is scratchpad-only")
        if generated_auxiliary is None:
            if auxiliary_noise is None:
                raise PocError("auxiliary noise is required outside B0")
            generated, used = sample_auxiliary_phase(
                model,
                history,
                actions,
                video_noise=video_noise,
                auxiliary_noise=auxiliary_noise,
                steps=auxiliary_steps,
            )
            calls += used
        else:
            generated = generated_auxiliary
        if control in {"autonomous", "context_shuffled"}:
            conditioning = generated
        elif control == "off":
            conditioning = torch.zeros_like(generated)
        elif control == "shuffled":
            if generated.shape[0] < 2:
                raise PocError("shuffled control requires evaluation batch size at least two")
            conditioning = (
                generated.roll(1, dims=0)
                if shuffle_indices is None
                else generated.index_select(0, shuffle_indices)
            )
        else:
            if clean_auxiliary is None:
                raise PocError("oracle_clean requires its explicitly nondeployable target")
            conditioning = clean_auxiliary
        video = unpatchify_lowres_rgb(conditioning)
    else:
        if control == "context_shuffled":
            raise PocError("context_shuffled is a phase1-only causality control")
        if auxiliary_steps < 1 or video_steps < 1:
            raise PocError("A1/L1 require nonempty auxiliary and video phases")
        if generated_auxiliary is None:
            if auxiliary_noise is None:
                raise PocError("auxiliary noise is required outside B0")
            generated, used = sample_auxiliary_phase(
                model,
                history,
                actions,
                video_noise=video_noise,
                auxiliary_noise=auxiliary_noise,
                steps=auxiliary_steps,
            )
            calls += used
        else:
            generated = generated_auxiliary
        if control in {"autonomous", "off"}:
            conditioning = generated
        elif control == "shuffled":
            if generated.shape[0] < 2:
                raise PocError("shuffled control requires evaluation batch size at least two")
            conditioning = (
                generated.roll(1, dims=0)
                if shuffle_indices is None
                else generated.index_select(0, shuffle_indices)
            )
        else:
            if clean_auxiliary is None:
                raise PocError("oracle_clean requires its explicitly nondeployable target")
            conditioning = clean_auxiliary
        condition_on = arm == "L1" and control != "off"
        video, used = sample_video_phase(
            model,
            history,
            actions,
            video_noise=video_noise,
            frozen_auxiliary=conditioning,
            steps=video_steps,
            condition_on_auxiliary=condition_on,
        )
        calls += used
    return CascadeResult(
        video=video,
        generated_auxiliary=generated,
        conditioning_auxiliary=conditioning,
        initial_video_noise=video_noise,
        initial_auxiliary_noise=auxiliary_noise,
        model_calls=calls,
        phase_boundary_sha256=tensor_sha256(generated) if generated is not None else None,
    )


def stable_within_batch_shuffle_indices(
    clip_ids: Sequence[str], device: torch.device
) -> tuple[Tensor, list[str]]:
    """Choose a deterministic derangement for a fixed rank-local batch."""
    if len(clip_ids) < 2 or len(set(clip_ids)) != len(clip_ids):
        raise PocError("shuffled control requires at least two unique clip IDs per batch")
    ordered = sorted(range(len(clip_ids)), key=lambda index: clip_ids[index])
    source_for_destination = [0] * len(clip_ids)
    for position, destination in enumerate(ordered):
        source_for_destination[destination] = ordered[position - 1]
    source_ids = [clip_ids[index] for index in source_for_destination]
    return torch.tensor(source_for_destination, device=device, dtype=torch.long), source_ids


def per_example_metrics(
    prediction_video: Tensor,
    target_video: Tensor,
    generated_auxiliary: Tensor | None,
    target_auxiliary: Tensor,
) -> dict[str, Tensor | None]:
    if prediction_video.shape != target_video.shape:
        raise ValueError("predicted and target video shapes differ")
    video_mse = (prediction_video.float() - target_video.float()).square().flatten(1).mean(1)
    psnr = 10.0 * torch.log10(4.0 / video_mse.clamp_min(1e-12))
    temporal_error = (
        prediction_video.float().diff(dim=2) - target_video.float().diff(dim=2)
    ).square().flatten(1).mean(1)
    aux_nmse: Tensor | None = None
    aux_cosine: Tensor | None = None
    if generated_auxiliary is not None:
        aux_error = (
            generated_auxiliary.float() - target_auxiliary.float()
        ).square().flatten(1).mean(1)
        aux_power = target_auxiliary.float().square().flatten(1).mean(1).clamp_min(1e-12)
        aux_nmse = aux_error / aux_power
        # Channels are the 48-D pixel-unshuffled feature vector at each of
        # the 896 aligned spatiotemporal tokens. Average token cosines so a
        # high-energy token cannot dominate the representation gate.
        token_cosine = torch.nn.functional.cosine_similarity(
            generated_auxiliary.float(),
            target_auxiliary.float(),
            dim=1,
            eps=1e-8,
        )
        aux_cosine = token_cosine.flatten(1).mean(1)
    return {
        "rgb_mse": video_mse,
        "rgb_psnr": psnr,
        "temporal_difference_mse": temporal_error,
        "auxiliary_nmse": aux_nmse,
        "auxiliary_cosine": aux_cosine,
    }


def canonical_quality_video(video: Tensor, *, upsample_lowres: bool) -> tuple[Tensor, Tensor]:
    """Return canonical clipped video and per-example out-of-range fraction."""
    from robot_wm.evaluation.video_latent_forcing_quality import (
        clamp_video_for_quality,
        upsample_lowres_video_for_quality,
    )

    if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] != 8:
        raise ValueError("quality video must have shape [B,3,8,H,W]")
    outside = ((video < -1.0) | (video > 1.0)).float().flatten(1).mean(1)
    if upsample_lowres:
        # The library's low-resolution helper intentionally rejects out-of-range
        # inputs, so apply the same explicit clamp operation before resizing and
        # run its audited canonical clamp helper on the resulting metric video.
        result = upsample_lowres_video_for_quality(video.float().clamp(-1.0, 1.0))
        result, _ = clamp_video_for_quality(result)
    else:
        result, _ = clamp_video_for_quality(video.float())
    if tuple(result.shape[1:]) != (3, 8, 64, 112):
        raise PocError(f"canonical quality shape mismatch: {tuple(result.shape)}")
    return result.contiguous(), outside


def build_quality_extractors(
    args: argparse.Namespace, device: torch.device
) -> tuple[Any, Any, dict[str, Any]]:
    from robot_wm.evaluation.video_latent_forcing_quality import (
        FrozenLPIPSAlex,
        FrozenR3D18AvgPool,
        QualityMetricError,
        quality_metric_provenance,
    )

    supplied = {
        "lpips_linear": (args.lpips_linear_weight, args.lpips_linear_sha256),
        "alexnet": (args.alexnet_weight, args.alexnet_sha256),
        "r3d18": (args.r3d18_weight, args.r3d18_sha256),
    }
    for label, (path, expected) in supplied.items():
        if not path or not expected or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise PocError(
                f"--quality-metrics requires {label} path and lowercase full SHA-256"
            )
        actual = sha256_file(path)
        if actual != expected:
            raise PocError(f"explicit {label} SHA-256 mismatch: expected {expected}, got {actual}")
    try:
        perceptual = FrozenLPIPSAlex(
            linear_weight_path=args.lpips_linear_weight,
            alexnet_weight_path=args.alexnet_weight,
            device=device,
        )
        video_feature = FrozenR3D18AvgPool(
            weight_path=args.r3d18_weight,
            expected_sha256=args.r3d18_sha256,
            device=device,
        )
        provenance = quality_metric_provenance(perceptual, video_feature)
    except QualityMetricError as exc:
        raise PocError(f"cannot initialize publication metrics: {exc}") from exc
    return perceptual, video_feature, provenance


def parse_nfe_pair(value: str) -> tuple[int, int]:
    try:
        auxiliary, video = (int(part) for part in value.split(":", 1))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("NFE pair must be AUX:VIDEO") from exc
    if auxiliary < 0 or video < 0 or auxiliary + video < 1:
        raise argparse.ArgumentTypeError("NFE counts must be nonnegative with positive total")
    return auxiliary, video


def frontier_pairs(arm: str) -> tuple[tuple[int, int], ...]:
    if arm == "phase1":
        return tuple((steps, 0) for steps in PHASE1_FRONTIER)
    if arm == "B0":
        return tuple((0, steps) for steps in (*PHASE1_FRONTIER, 50))
    return tuple((total // 2, total - total // 2) for total in DUAL_FRONTIER_TOTALS)


def validate_nfe_pairs(arm: str, pairs: Sequence[tuple[int, int]]) -> None:
    for auxiliary, video in pairs:
        valid = (
            (arm == "phase1" and auxiliary >= 1 and video == 0)
            or (arm == "B0" and auxiliary == 0 and video >= 1)
            or (arm in {"A1", "L1"} and auxiliary >= 1 and video >= 1)
        )
        if not valid:
            raise PocError(f"NFE pair {auxiliary}:{video} is invalid for arm {arm}")


def evaluation_batch_indexes(
    size: int, batch_size: int, *, require_derangement: bool
) -> list[list[int]]:
    if size < 1:
        raise PocError("evaluation rank received no clips")
    batches = [list(range(start, min(start + batch_size, size))) for start in range(0, size, batch_size)]
    if require_derangement and len(batches[-1]) == 1:
        if len(batches) == 1:
            raise PocError("shuffled control requires at least two clips on every rank")
        batches[-2].extend(batches.pop())
    return batches


def paired_global_derangement(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], str]:
    """Pair adjacent manifest clips into a world/batch-invariant derangement."""
    if len(rows) < 2 or len(rows) % 2:
        raise PocError("global shuffled control requires an even manifest population")
    clip_ids = [str(row["clip_id"]) for row in rows]
    if len(set(clip_ids)) != len(clip_ids):
        raise PocError("global shuffled-control clip IDs must be unique")
    mapping: dict[str, str] = {}
    for index in range(0, len(clip_ids), 2):
        left, right = clip_ids[index : index + 2]
        mapping[left] = right
        mapping[right] = left
    return mapping, sha256_json(mapping)


def paired_rank_evaluation_layout(
    size: int,
    batch_size: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[list[int], list[list[int]]]:
    """Co-locate immutable donor pairs on ranks and keep every batch pair-complete."""
    if size < 2 or size % 2 or batch_size < 2 or batch_size % 2:
        raise PocError("paired evaluation requires even size and even batch size >=2")
    pairs = [(index, index + 1) for index in range(0, size, 2)]
    local_pairs = pairs[rank::world_size]
    if not local_pairs:
        raise PocError("evaluation rank received no donor pairs")
    local_indexes = [index for pair in local_pairs for index in pair]
    batches = [
        list(range(start, min(start + batch_size, len(local_indexes))))
        for start in range(0, len(local_indexes), batch_size)
    ]
    if any(len(batch) % 2 for batch in batches):
        raise PocError("evaluation batching split a global donor pair")
    return local_indexes, batches


def controls_for_arm(arm: str, requested: Sequence[str] | None) -> tuple[str, ...]:
    if requested:
        controls = tuple(requested)
    elif arm == "B0":
        controls = ("off",)
    elif arm == "phase1":
        controls = PHASE1_CONTROLS
    else:
        controls = DUAL_CONTROLS
    if arm == "B0" and controls != ("off",):
        raise PocError("B0 permits only its strict off control")
    if arm != "phase1" and "context_shuffled" in controls:
        raise PocError("context_shuffled is valid only for phase1")
    if len(set(controls)) != len(controls) or any(control not in CONTROLS for control in controls):
        raise PocError("evaluation controls must be unique supported values")
    return controls


def validate_selection_record(
    path: str | Path,
    *,
    arm: str,
    checkpoint: str | Path,
    nfe_pairs: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    payload = load_json(path, "frozen selection record")
    required_true = (
        "frozen",
        "validation_only_selection",
        "three_seed_gate_passed",
        "protected_test_authorized",
    )
    if payload.get("schema") != SELECTION_SCHEMA or any(payload.get(key) is not True for key in required_true):
        raise PocError("selection record is not a frozen post-validation three-seed authorization")
    if payload.get("selected_arm") != arm:
        raise PocError("selection record arm differs from requested test evaluation")
    checkpoint_record = payload.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping) or checkpoint_record.get("sha256") != sha256_file(checkpoint):
        raise PocError("selection record checkpoint digest differs")
    selected_pair = payload.get("selected_nfe_pair")
    if selected_pair != [nfe_pairs[0][0], nfe_pairs[0][1]] or len(nfe_pairs) != 1:
        raise PocError("protected test must evaluate exactly the one frozen NFE pair")
    validation_metrics = payload.get("validation_metrics")
    if not isinstance(validation_metrics, Mapping):
        raise PocError("selection record lacks frozen validation metrics")
    metrics_path = Path(str(validation_metrics.get("path", ""))).expanduser().resolve()
    if not metrics_path.is_file() or validation_metrics.get("sha256") != sha256_file(metrics_path):
        raise PocError("selection record validation metrics are missing or changed")
    required = payload.get("required_primary_metrics")
    if not isinstance(required, Mapping) or any(required.get(name) != "passed" for name in MISSING_PRIMARY_METRICS):
        raise PocError("selection record does not prove every required primary metric passed")
    return payload


def evaluation_command(args: argparse.Namespace) -> int:
    context = initialize_distributed()
    logger: LocalAndOptionalWandbLogger | None = None
    try:
        output_dir = validated_run_dir(args.artifact_root, args.run_id, resume=False)
        manifest_path, rows, manifest_record = _manifest_record(
            args.manifest,
            args.split,
            data_root=args.data_root,
        )
        nfe_pairs = tuple(args.nfe_pair or ())
        if args.frontier:
            if nfe_pairs:
                raise PocError("use either --frontier or explicit --nfe-pair, not both")
            nfe_pairs = frontier_pairs(args.arm)
        if not nfe_pairs:
            nfe_pairs = PRIMARY_NFE[args.arm]
        validate_nfe_pairs(args.arm, nfe_pairs)
        controls = controls_for_arm(args.arm, args.control)
        selection_record = None
        if args.split == "test":
            raise PocError(
                "protected test is code-locked until the three-seed selector is implemented "
                "and validation evidence freezes the exact endpoint matrix"
            )
        elif args.allow_protected_test_after_selection or args.selection_record:
            raise PocError("protected-test flags are invalid for validation evaluation")

        seed_everything(args.seed, 0)
        model, model_config = instantiate_model(args)
        source_record = git_record()
        if source_record["dirty"]:
            raise PocError("evaluation requires a clean, committed Git source tree")
        if count_parameters(model)["total"] != FROZEN_PARAMETER_COUNT:
            raise PocError("model parameter count differs from the frozen 41,963,760")
        model.to(context.device)
        # Every reported sample is generated from the checkpoint EMA, never raw
        # training weights. Arm/model bindings prevent same-shaped relabeling.
        payload = torch.load(Path(args.checkpoint).expanduser().resolve(), map_location="cpu", weights_only=False)
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise PocError("checkpoint schema mismatch")
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if checkpoint_path.parent.name != "checkpoints":
            raise PocError("checkpoint must remain inside its immutable run/checkpoints directory")
        training_config_path = checkpoint_path.parent.parent / "resolved_config.json"
        training_config = load_json(training_config_path, "checkpoint training config")
        checkpoint_file_record = file_record(checkpoint_path)
        training_config_file_record = file_record(training_config_path)
        if sha256_json(training_config) != payload.get("config_sha256"):
            raise PocError("checkpoint is not bound to its immutable training config")
        training_source = training_config.get("source")
        expected_optimizer_seed = FROZEN_OPTIMIZER_SEEDS[
            FROZEN_EVALUATION_SEEDS.index(args.seed)
        ]
        if args.arm == "phase1":
            expected_optimizer_seed = FROZEN_OPTIMIZER_SEEDS[0]
        if (
            not isinstance(training_source, Mapping)
            or training_source.get("commit") != source_record["commit"]
            or training_source.get("dirty") is not False
            or training_config.get("arm") != args.arm
            or training_config.get("model") != model_config
            or training_config.get("seed") != expected_optimizer_seed
        ):
            raise PocError("checkpoint training identity differs from the frozen evaluation")
        if payload.get("arm") != args.arm:
            raise PocError("checkpoint arm does not match requested evaluation arm")
        if payload.get("model_config") != model_config:
            raise PocError("checkpoint model config does not match evaluation model config")
        ema_payload = payload.get("ema")
        if not isinstance(ema_payload, Mapping):
            raise PocError("reported evaluation requires checkpoint EMA weights")
        if float(ema_payload.get("decay", -1.0)) != FROZEN_EMA_DECAY:
            raise PocError("checkpoint EMA decay violates the frozen protocol")
        if (
            ema_payload.get("schedule") != FROZEN_EMA_SCHEDULE
            or ema_payload.get("num_updates") != payload.get("completed_updates")
        ):
            raise PocError("checkpoint EMA warm-up state violates the frozen protocol")
        ema_shadow = ema_payload.get("shadow")
        if not isinstance(ema_shadow, Mapping):
            raise PocError("checkpoint EMA shadow state is missing")
        model.load_state_dict(ema_shadow, strict=True)
        model.eval()
        perceptual_extractor = None
        video_feature_extractor = None
        quality_provenance = None
        if args.quality_metrics:
            perceptual_extractor, video_feature_extractor, quality_provenance = (
                build_quality_extractors(args, context.device)
            )
        dataset = DroidVideoLatentForcingDataset(
            manifest_path,
            args.data_root,
            allow_protected_test=args.split == "test",
            protected_test_purpose=(
                f"frozen post-selection evaluation {sha256_file(args.selection_record)}"
                if args.split == "test"
                else None
            ),
        )
        derangement, derangement_sha256 = paired_global_derangement(rows)
        local_indexes, local_batches = paired_rank_evaluation_layout(
            len(dataset),
            args.eval_batch_size,
            rank=context.rank,
            world_size=context.world_size,
        )
        subset = torch.utils.data.Subset(dataset, local_indexes)
        loader = DataLoader(
            subset,
            batch_sampler=local_batches,
            num_workers=args.workers,
            pin_memory=context.device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        config = {
            "schema": EVALUATION_SCHEMA,
            "source": source_record,
            "arm": args.arm,
            "split": args.split,
            "clock_convention": CLOCK_CONVENTION,
            "clean_time_epsilon": FROZEN_CLEAN_TIME_EPS,
            "checkpoint": checkpoint_file_record,
            "training_config": training_config_file_record,
            "checkpoint_update": int(payload["completed_updates"]),
            "weights": {
                "kind": "ema",
                "decay": FROZEN_EMA_DECAY,
                "schedule": FROZEN_EMA_SCHEDULE,
                "num_updates": int(ema_payload["num_updates"]),
            },
            "manifest": manifest_record,
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "seed": args.seed,
            "nfe_pairs": [list(pair) for pair in nfe_pairs],
            "controls": list(controls),
            "model": model_config,
            "fixed_noise_by_clip_id": True,
            "fixed_noise_key": "sha256(f'{clip_id}:{eval_seed}:video|aux')",
            "shuffle_assignment": (
                "manifest-global adjacent-pair swap; donor pairs are co-located "
                "without changing the mapping across world size or batch size"
            ),
            "shuffle_mapping_sha256": derangement_sha256,
            "world_size": context.world_size,
            "eval_batch_size": args.eval_batch_size,
            "media_clip_ids": [str(row["clip_id"]) for row in rows[: args.media_clips]],
            "media_clean_future_policy": (
                "oracle-clean raw RGB/scratchpad/phase-boundary media are never persisted "
                "or uploaded; auditable metric scalars and pinned R3D features remain local"
            ),
            "publication_quality_metrics": {
                "enabled": args.quality_metrics,
                "provenance": quality_provenance,
                "generated_range_policy": "clamp_to_[-1,1]_and_record_fraction",
                "phase1_spatial_policy": "bilinear_upsample_prediction_and_target_to_64x112",
                "raw_rgb_mse_is_secondary": True,
            },
            "selection_record": file_record(args.selection_record) if selection_record else None,
            "basic_metrics_only": not args.quality_metrics,
            "required_primary_metrics_missing": (
                [] if args.quality_metrics else list(MISSING_PRIMARY_METRICS)
            ),
            "quality_metric_suite_complete": args.quality_metrics,
            "quality_claim_evaluable": False,
            "quality_claim_requires_gate_analyzer": True,
        }
        if context.is_primary:
            output_dir.mkdir(parents=True, exist_ok=False)
            atomic_write_json(output_dir / "resolved_config.json", config, exclusive=True)
            atomic_write_json(
                output_dir / "provenance.json",
                {
                    "schema": EVALUATION_SCHEMA,
                    "git": git_record(),
                    "runtime": runtime_record(),
                    "command": [sys.executable, *sys.argv],
                    "resolved_config_sha256": sha256_json(config),
                },
                exclusive=True,
            )
        context.barrier()
        logger = LocalAndOptionalWandbLogger(output_dir, args, config, primary=context.is_primary)

        local_records: list[dict[str, Any]] = []
        local_media_records: list[dict[str, Any]] = []
        local_quality_batches: dict[tuple[int, int, str], list[Any]] = {}
        media_clip_ids = {str(row["clip_id"]) for row in rows[: args.media_clips]}
        for raw_batch in loader:
            batch = move_training_batch(raw_batch, context.device)
            clip_ids = [str(value) for value in raw_batch["clip_id"]]
            video_noise = stable_noise_like(batch["future"], clip_ids, args.seed, "video")
            auxiliary_noise = (
                None
                if args.arm == "B0"
                else stable_noise_like(batch["lowres_scratchpad"], clip_ids, args.seed, "aux")
            )
            paired_input_hashes = [
                {
                    "history_sha256": tensor_sha256(batch["history"][index]),
                    "actions_sha256": tensor_sha256(batch["actions"][index]),
                    "initial_video_noise_sha256": tensor_sha256(video_noise[index]),
                    "initial_auxiliary_noise_sha256": (
                        tensor_sha256(auxiliary_noise[index])
                        if auxiliary_noise is not None
                        else None
                    ),
                }
                for index in range(len(clip_ids))
            ]
            shuffle_indices: Tensor | None = None
            shuffled_source_ids: list[str] | None = None
            shuffled_source_indexes: list[int] | None = None
            if any(control in {"shuffled", "context_shuffled"} for control in controls):
                if len(clip_ids) % 2:
                    raise PocError("global donor-pair batch was split")
                shuffle_indices = torch.arange(
                    len(clip_ids), device=context.device, dtype=torch.long
                ).bitwise_xor(1)
                shuffled_source_indexes = shuffle_indices.cpu().tolist()
                shuffled_source_ids = [
                    clip_ids[int(index)] for index in shuffled_source_indexes
                ]
                if any(
                    derangement[destination] != source
                    for destination, source in zip(
                        clip_ids, shuffled_source_ids, strict=True
                    )
                ):
                    raise PocError("runtime donor mapping differs from its frozen global hash")
            quality_reference: Tensor | None = None
            cached_real_features: Tensor | None = None
            if args.quality_metrics:
                from robot_wm.evaluation.video_latent_forcing_quality import (
                    r3d18_avgpool_features,
                )

                quality_reference, _ = canonical_quality_video(
                    batch["lowres_rgb"] if args.arm == "phase1" else batch["future"],
                    upsample_lowres=args.arm == "phase1",
                )
                assert video_feature_extractor is not None
                cached_real_features = r3d18_avgpool_features(
                    quality_reference, video_feature_extractor
                ).detach().cpu()
            for auxiliary_steps, video_steps in nfe_pairs:
                generated_shared: Tensor | None = None
                boundary_digest: str | None = None
                auxiliary_generation_calls = 0
                if args.arm not in {"B0"}:
                    assert auxiliary_noise is not None
                    with _autocast(context.device):
                        generated_shared, auxiliary_generation_calls = sample_auxiliary_phase(
                            model,
                            batch["history"],
                            batch["actions"],
                            video_noise=video_noise,
                            auxiliary_noise=auxiliary_noise,
                            steps=auxiliary_steps,
                        )
                    boundary_digest = tensor_sha256(generated_shared)
                for control in controls:
                    control_generated = generated_shared
                    control_auxiliary_calls = auxiliary_generation_calls
                    if control == "context_shuffled":
                        if args.arm != "phase1" or shuffle_indices is None:
                            raise PocError("context_shuffled requires phase1 and a derangement")
                        assert auxiliary_noise is not None
                        with _autocast(context.device):
                            control_generated, control_auxiliary_calls = sample_auxiliary_phase(
                                model,
                                batch["history"].index_select(0, shuffle_indices),
                                batch["actions"].index_select(0, shuffle_indices),
                                video_noise=video_noise,
                                auxiliary_noise=auxiliary_noise,
                                steps=auxiliary_steps,
                            )
                    # Clean future-derived state enters only the explicitly
                    # nondeployable oracle call; deployable controls receive None.
                    oracle_target = batch["lowres_scratchpad"] if control == "oracle_clean" else None
                    with _autocast(context.device):
                        result = sample_control(
                            model,
                            args.arm,
                            batch["history"],
                            batch["actions"],
                            oracle_target,
                            control=control,
                            auxiliary_steps=auxiliary_steps,
                            video_steps=video_steps,
                            video_noise=video_noise,
                            auxiliary_noise=auxiliary_noise,
                            generated_auxiliary=control_generated,
                            shuffle_indices=shuffle_indices,
                        )
                    if (
                        control != "context_shuffled"
                        and boundary_digest is not None
                        and result.phase_boundary_sha256 != boundary_digest
                    ):
                        raise PocError("controls did not share an identical generated auxiliary boundary")
                    conceptual_calls = control_auxiliary_calls + result.model_calls
                    if conceptual_calls != auxiliary_steps + video_steps:
                        raise PocError("actual transformer call count differs from requested NFE")
                    target_video = (
                        batch["lowres_rgb"] if args.arm == "phase1" else batch["future"]
                    )
                    metrics = per_example_metrics(
                        result.video,
                        target_video,
                        None if args.arm == "B0" else result.generated_auxiliary,
                        batch["lowres_scratchpad"],
                    )
                    quality_frame: Tensor | None = None
                    quality_temporal: Tensor | None = None
                    clipped_fraction = (
                        ((result.video < -1.0) | (result.video > 1.0))
                        .float()
                        .flatten(1)
                        .mean(1)
                    )
                    if args.quality_metrics:
                        from robot_wm.evaluation.video_latent_forcing_quality import (
                            QualityBatch,
                            lpips_alex_per_frame_per_example,
                            r3d18_avgpool_features,
                            temporal_difference_lpips_per_example,
                        )

                        assert quality_reference is not None
                        assert cached_real_features is not None
                        assert perceptual_extractor is not None
                        assert video_feature_extractor is not None
                        quality_candidate, clipped_fraction = canonical_quality_video(
                            result.video, upsample_lowres=args.arm == "phase1"
                        )
                        quality_frame = lpips_alex_per_frame_per_example(
                            quality_reference, quality_candidate, perceptual_extractor
                        ).detach().cpu()
                        quality_temporal = temporal_difference_lpips_per_example(
                            quality_reference, quality_candidate, perceptual_extractor
                        ).detach().cpu()
                        generated_features = r3d18_avgpool_features(
                            quality_candidate, video_feature_extractor
                        ).detach().cpu()
                        local_quality_batches.setdefault(
                            (auxiliary_steps, video_steps, control), []
                        ).append(
                            QualityBatch(
                                clip_ids=tuple(clip_ids),
                                lpips_alex_frame=quality_frame.double(),
                                lpips_alex_temporal_difference=quality_temporal.double(),
                                r3d18_real_features=cached_real_features.double(),
                                r3d18_generated_features=generated_features.double(),
                            )
                        )
                    for item_index, clip_id in enumerate(clip_ids):
                        phase_boundary_sha256 = (
                            tensor_sha256(result.generated_auxiliary[item_index])
                            if result.generated_auxiliary is not None
                            else None
                        )
                        conditioning_auxiliary_sha256 = (
                            tensor_sha256(result.conditioning_auxiliary[item_index])
                            if result.conditioning_auxiliary is not None
                            else None
                        )
                        context_source_index = (
                            shuffled_source_indexes[item_index]
                            if control == "context_shuffled"
                            and shuffled_source_indexes is not None
                            else item_index
                        )
                        target_auxiliary_sha256 = tensor_sha256(
                            batch["lowres_scratchpad"][item_index]
                        )
                        zero_reference = (
                            result.conditioning_auxiliary[item_index]
                            if result.conditioning_auxiliary is not None
                            else batch["lowres_scratchpad"][item_index]
                        )
                        record = {
                            "clip_id": clip_id,
                            "episode_index": int(raw_batch["episode_index"][item_index]),
                            "arm": args.arm,
                            "control": control,
                            "auxiliary_nfe": auxiliary_steps,
                            "video_nfe": video_steps,
                            "path_model_calls": conceptual_calls,
                            "optimizer_seed": int(training_config["seed"]),
                            "evaluation_seed": args.seed,
                            "checkpoint_sha256": checkpoint_file_record["sha256"],
                            "training_config_sha256": training_config_file_record["sha256"],
                            "ema_decay": FROZEN_EMA_DECAY,
                            "ema_schedule": FROZEN_EMA_SCHEDULE,
                            "ema_updates": int(ema_payload["num_updates"]),
                            # Shared auxiliary generation is counted once in
                            # actual evaluator work, but every deployable path
                            # pays for it once.
                            "evaluation_auxiliary_generation_calls": control_auxiliary_calls,
                            "phase_boundary_sha256": phase_boundary_sha256,
                            "conditioning_auxiliary_sha256": conditioning_auxiliary_sha256,
                            "pre_video_auxiliary_sha256": conditioning_auxiliary_sha256,
                            "post_video_auxiliary_sha256": conditioning_auxiliary_sha256,
                            "target_auxiliary_sha256": target_auxiliary_sha256,
                            "zero_auxiliary_sha256": tensor_sha256(
                                torch.zeros_like(zero_reference)
                            ),
                            "auxiliary_frozen_assertion_executed": video_steps > 0,
                            "shuffle_mapping_sha256": derangement_sha256,
                            "conditioning_source_clip_id": (
                                shuffled_source_ids[item_index]
                                if control in {"shuffled", "context_shuffled"}
                                and shuffled_source_ids is not None
                                else clip_id
                            ),
                            "auxiliary_conditioning_source_clip_id": (
                                shuffled_source_ids[item_index]
                                if control == "shuffled"
                                and shuffled_source_ids is not None
                                else clip_id
                            ),
                            "history_action_source_clip_id": (
                                shuffled_source_ids[item_index]
                                if control == "context_shuffled"
                                and shuffled_source_ids is not None
                                else clip_id
                            ),
                            "deployable": control != "oracle_clean",
                            "clean_future_used_as_condition": control == "oracle_clean",
                            "teacher_model_calls": 0,
                            "history_sha256": paired_input_hashes[context_source_index][
                                "history_sha256"
                            ],
                            "actions_sha256": paired_input_hashes[context_source_index][
                                "actions_sha256"
                            ],
                            "initial_video_noise_sha256": paired_input_hashes[item_index][
                                "initial_video_noise_sha256"
                            ],
                            "initial_auxiliary_noise_sha256": paired_input_hashes[item_index][
                                "initial_auxiliary_noise_sha256"
                            ],
                            "generated_pixel_clipped_fraction": float(
                                clipped_fraction[item_index]
                            ),
                            "lpips_alex_frame": (
                                None
                                if quality_frame is None
                                else float(quality_frame[item_index])
                            ),
                            "lpips_alex_temporal_difference": (
                                None
                                if quality_temporal is None
                                else float(quality_temporal[item_index])
                            ),
                            **{
                                name: (None if values is None else float(values[item_index]))
                                for name, values in metrics.items()
                            },
                        }
                        local_records.append(record)
                        if clip_id in media_clip_ids and control != "oracle_clean":
                            local_media_records.append(
                                save_evaluation_media(
                                    output_dir,
                                    clip_id=clip_id,
                                    arm=args.arm,
                                    control=control,
                                    auxiliary_nfe=auxiliary_steps,
                                    video_nfe=video_steps,
                                    video=result.video[item_index],
                                    generated_auxiliary=(
                                        result.generated_auxiliary[item_index]
                                        if result.generated_auxiliary is not None
                                        else None
                                    ),
                                    conditioning_auxiliary=(
                                        result.conditioning_auxiliary[item_index]
                                        if result.conditioning_auxiliary is not None
                                        else None
                                    ),
                                    initial_video_noise=result.initial_video_noise[item_index],
                                    initial_auxiliary_noise=(
                                        result.initial_auxiliary_noise[item_index]
                                        if result.initial_auxiliary_noise is not None
                                        else None
                                    ),
                                )
                            )

        gathered = context.gather_objects(local_records)
        gathered_media = context.gather_objects(local_media_records)
        gathered_quality = context.gather_objects(local_quality_batches)
        if context.is_primary:
            records = [record for rank_records in gathered for record in rank_records]
            media_records = [record for rank_records in gathered_media for record in rank_records]
            records.sort(
                key=lambda row: (
                    row["auxiliary_nfe"], row["video_nfe"], row["control"], row["clip_id"]
                )
            )
            records_path = output_dir / "per_clip_metrics.jsonl"
            for record in records:
                append_jsonl(records_path, record)
            media_manifest_path = output_dir / "sample_artifacts.jsonl"
            for record in sorted(
                media_records,
                key=lambda row: (
                    row["auxiliary_nfe"], row["video_nfe"], row["control"], row["clip_id"]
                ),
            ):
                append_jsonl(media_manifest_path, record)
            if not media_records:
                _atomic_write_text(media_manifest_path, "", exclusive=True)
            groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
            for record in records:
                groups.setdefault(
                    (record["auxiliary_nfe"], record["video_nfe"], record["control"]), []
                ).append(record)
            quality_summaries: dict[tuple[int, int, str], dict[str, float]] = {}
            quality_feature_artifacts: list[dict[str, Any]] = []
            if args.quality_metrics:
                from robot_wm.evaluation.video_latent_forcing_quality import (
                    merge_quality_batches,
                    quality_summary,
                )

                quality_groups: dict[tuple[int, int, str], list[Any]] = {}
                for rank_groups in gathered_quality:
                    for key, batches in rank_groups.items():
                        quality_groups.setdefault(tuple(key), []).extend(batches)
                if quality_groups.keys() != groups.keys():
                    raise PocError("quality metric groups differ from basic metric groups")
                for key, batches in sorted(quality_groups.items()):
                    merged_quality = merge_quality_batches(batches)
                    quality_summaries[key] = quality_summary(merged_quality)
                    auxiliary_nfe, video_nfe, control = key
                    feature_path = (
                        output_dir
                        / "quality_features"
                        / f"a{auxiliary_nfe:03d}_v{video_nfe:03d}_{control}.pt"
                    )
                    atomic_torch_save(
                        feature_path,
                        {
                            "clip_ids": merged_quality.clip_ids,
                            "r3d18_real_features": merged_quality.r3d18_real_features.float(),
                            "r3d18_generated_features": (
                                merged_quality.r3d18_generated_features.float()
                            ),
                            "feature_name": "torchvision_r3d18_kinetics400_v1_avgpool",
                            "feature_dimension": 512,
                            "stored_dtype": "float32_exact_extractor_output",
                        },
                    )
                    quality_feature_artifacts.append(
                        {
                            "auxiliary_nfe": auxiliary_nfe,
                            "video_nfe": video_nfe,
                            "control": control,
                            "file": file_record(feature_path),
                        }
                    )
            summaries = []
            for (auxiliary_nfe, video_nfe, control), group in sorted(groups.items()):
                key = (auxiliary_nfe, video_nfe, control)
                summaries.append(
                    {
                        "auxiliary_nfe": auxiliary_nfe,
                        "video_nfe": video_nfe,
                        "total_nfe": auxiliary_nfe + video_nfe,
                        "control": control,
                        "clips": len(group),
                        **{
                            name: (
                                float(np.mean([row[name] for row in group if row[name] is not None]))
                                if any(row[name] is not None for row in group)
                                else None
                            )
                            for name in (
                                "rgb_mse",
                                "rgb_psnr",
                                "temporal_difference_mse",
                                "auxiliary_nmse",
                                "auxiliary_cosine",
                            )
                        },
                        **quality_summaries.get(key, {}),
                    }
                )
            summary = {
                "schema": EVALUATION_SCHEMA,
                "status": "complete",
                "split": args.split,
                "arm": args.arm,
                "checkpoint": checkpoint_file_record,
                "manifest": manifest_record,
                "record_count": len(records),
                "summaries": summaries,
                "per_clip_metrics": file_record(records_path),
                "sample_artifacts": file_record(media_manifest_path),
                "sample_artifact_count": len(media_records),
                "reported_weight_source": "ema",
                "ema_decay": FROZEN_EMA_DECAY,
                "ema_schedule": FROZEN_EMA_SCHEDULE,
                "ema_updates": int(ema_payload["num_updates"]),
                "phase_boundary_shared_across_controls": (
                    "autonomous/off/shuffled/oracle_clean share the aligned generated "
                    "boundary; context_shuffled intentionally regenerates from shuffled conditions"
                ),
                "actual_nfe_reported": True,
                "basic_metrics_only": not args.quality_metrics,
                "quality_metric_suite_complete": args.quality_metrics,
                "quality_claim_evaluable": False,
                "quality_claim_requires_gate_analyzer": True,
                "acceleration_claim_evaluable": False,
                "required_primary_metrics_missing": (
                    [] if args.quality_metrics else list(MISSING_PRIMARY_METRICS)
                ),
                "quality_metric_provenance": quality_provenance,
                "quality_feature_artifacts": quality_feature_artifacts,
                "note": (
                    "RGB/auxiliary metrics are mechanism telemetry. The optional pinned "
                    "LPIPS-Alex and R3D18-Frechet suite is publication-gate quality; "
                    "R3D18-Frechet is not labeled as FVD."
                ),
            }
            atomic_write_json(output_dir / "summary.json", summary, exclusive=True)
            for item in summaries:
                logger.log(
                    {"update": int(payload["completed_updates"]), "event": "evaluation", **item},
                    primary=True,
                )
            logger.log_media(media_records, step=int(payload["completed_updates"]))
        context.barrier()
        return 0
    finally:
        if logger is not None:
            logger.finish()
        close_distributed(context)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)


def add_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--run-id", required=True)


def add_data_arguments(parser: argparse.ArgumentParser, *, training: bool) -> None:
    parser.add_argument("--data-root", required=True)
    if training:
        parser.add_argument("--train-manifest", required=True)
        parser.add_argument("--validation-manifest", required=True)
    else:
        parser.add_argument("--manifest", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("calibrate", "train"):
        subparser = subparsers.add_parser(command)
        add_model_arguments(subparser)
        add_artifact_arguments(subparser)
        add_data_arguments(subparser, training=True)
        subparser.add_argument("--seed", type=int, default=1234)
        subparser.add_argument("--global-batch-size", type=int, default=FROZEN_GLOBAL_BATCH_SIZE)
        subparser.add_argument(
            "--micro-batch-size",
            type=int,
            help="per-rank microbatch; accumulation preserves frozen global batch 256",
        )
        subparser.add_argument("--workers", type=int, default=4)
        subparser.add_argument("--learning-rate", type=float, default=FROZEN_LEARNING_RATE)
        subparser.add_argument("--warmup-updates", type=int, default=FROZEN_WARMUP_UPDATES)
        subparser.add_argument("--weight-decay", type=float, default=FROZEN_WEIGHT_DECAY)
        subparser.add_argument(
            "--gradient-clip-norm", type=float, default=FROZEN_GRADIENT_CLIP_NORM
        )
        subparser.add_argument("--ema-decay", type=float, default=FROZEN_EMA_DECAY)
        subparser.add_argument("--log-every", type=int, default=10)
        subparser.add_argument("--resume")
        subparser.add_argument("--calibration-record", required=command == "train")
        subparser.add_argument("--phase1-gate-record")
        subparser.add_argument("--wandb", action="store_true")
        subparser.add_argument("--wandb-entity")
        subparser.add_argument("--wandb-project")
        subparser.add_argument("--wandb-private-project-ack", action="store_true")

    evaluate = subparsers.add_parser("eval")
    add_model_arguments(evaluate)
    add_artifact_arguments(evaluate)
    add_data_arguments(evaluate, training=False)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="val")
    evaluate.add_argument("--seed", type=int, default=20260801)
    evaluate.add_argument("--workers", type=int, default=2)
    evaluate.add_argument("--eval-batch-size", type=int, default=8)
    evaluate.add_argument("--media-clips", type=int, default=4)
    evaluate.add_argument("--nfe-pair", action="append", type=parse_nfe_pair)
    evaluate.add_argument("--frontier", action="store_true")
    evaluate.add_argument("--control", action="append", choices=CONTROLS)
    evaluate.add_argument("--allow-protected-test-after-selection", action="store_true")
    evaluate.add_argument("--selection-record")
    evaluate.add_argument("--quality-metrics", action="store_true")
    evaluate.add_argument("--lpips-linear-weight")
    evaluate.add_argument("--lpips-linear-sha256")
    evaluate.add_argument("--alexnet-weight")
    evaluate.add_argument("--alexnet-sha256")
    evaluate.add_argument("--r3d18-weight")
    evaluate.add_argument("--r3d18-sha256")
    evaluate.add_argument("--wandb", action="store_true")
    evaluate.add_argument("--wandb-entity")
    evaluate.add_argument("--wandb-project")
    evaluate.add_argument("--wandb-private-project-ack", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.width < 1 or args.depth < 1 or args.heads < 1 or args.width % args.heads:
        raise PocError("model width/depth/heads must be positive and width must divide by heads")
    if (
        args.width != FROZEN_MODEL_WIDTH
        or args.depth != FROZEN_MODEL_DEPTH
        or args.heads != FROZEN_MODEL_HEADS
        or args.mlp_ratio != FROZEN_MODEL_MLP_RATIO
    ):
        raise PocError(
            "model contract is frozen: width512, depth12, heads8, MLP ratio4"
        )
    if args.command in {"calibrate", "train"}:
        if args.global_batch_size < 1 or args.workers < 0:
            raise PocError("batch size must be positive and workers nonnegative")
        frozen = (
            args.global_batch_size == FROZEN_GLOBAL_BATCH_SIZE
            and args.learning_rate == FROZEN_LEARNING_RATE
            and args.warmup_updates == FROZEN_WARMUP_UPDATES
            and args.weight_decay == FROZEN_WEIGHT_DECAY
            and args.gradient_clip_norm == FROZEN_GRADIENT_CLIP_NORM
            and args.ema_decay == FROZEN_EMA_DECAY
        )
        if not frozen:
            raise PocError(
                "optimizer contract is frozen: batch256, seed1234, AdamW lr5e-5 "
                "betas(.9,.95), wd0, warmup500 then constant, clip1.0, EMA.9999"
            )
        if args.seed not in FROZEN_OPTIMIZER_SEEDS or (
            args.arm == "phase1" and args.seed != FROZEN_OPTIMIZER_SEEDS[0]
        ):
            raise PocError(
                "optimizer seed must be 1234 for Phase-1 or one of the frozen "
                "dual confirmation seeds 1234/2234/3234"
            )
        if args.arm in {"B0", "A1", "L1"} and args.seed != FROZEN_OPTIMIZER_SEEDS[0]:
            raise PocError(
                "confirmation seeds 2234/3234 are code-locked until the one-seed "
                "dual gate emits a reproducible selection record"
            )
        if args.micro_batch_size is not None and args.micro_batch_size < 1:
            raise PocError("micro batch size must be positive")
        if args.command == "calibrate" and args.resume is not None:
            raise PocError("the exact 200-update calibration restarts rather than resuming")
        if args.command == "train" and args.arm in {"B0", "A1", "L1"}:
            if not args.phase1_gate_record:
                raise PocError("dual-arm training requires a passed frozen Phase-1 gate record")
        elif args.phase1_gate_record is not None:
            raise PocError("a Phase-1 gate record is valid only for B0/A1/L1 full training")
    else:
        if (
            args.workers < 0
            or args.eval_batch_size < 2
            or args.eval_batch_size % 2
            or args.media_clips < 1
        ):
            raise PocError(
                "evaluation requires workers >=0 and an even batch size >=2 for frozen donor pairs"
            )
        if args.seed not in FROZEN_EVALUATION_SEEDS:
            raise PocError("evaluation seed must be one of 20260801/20260802/20260803")
        quality_values = (
            args.lpips_linear_weight,
            args.lpips_linear_sha256,
            args.alexnet_weight,
            args.alexnet_sha256,
            args.r3d18_weight,
            args.r3d18_sha256,
        )
        if args.quality_metrics != all(value is not None for value in quality_values):
            raise PocError(
                "quality metrics and all three explicit weight path/full-hash pairs "
                "must be enabled together"
            )
    wandb_values = (args.wandb_entity, args.wandb_project, args.wandb_private_project_ack)
    if args.wandb != all(bool(value) for value in wandb_values):
        raise PocError(
            "W&B is optional, but enabling it requires entity, project, and private-project acknowledgement"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        if args.command in {"calibrate", "train"}:
            return training_command(args)
        return evaluation_command(args)
    except (PocError, ValueError, OSError) as exc:
        print(f"Video Latent Forcing POC error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
