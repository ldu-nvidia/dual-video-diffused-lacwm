#!/usr/bin/env python3
"""Real-data, official-batch DDP qualification before a long B200 run."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "projects" / "latent_action_models"
CONFIG_ROOT = PROJECT_ROOT / "configs"
sys.path[:0] = [
    str(REPO_ROOT / "tools" / "env" / "videox_shim"),
    os.environ["VIDEOX_HOME"],
    str(PROJECT_ROOT),
    str(REPO_ROOT),
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=False)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    raise TypeError(f"unsupported batch value: {type(value)}")


def group_grad_norm(model, token: str) -> tuple[float, int]:
    total = 0.0
    count = 0
    for name, parameter in model.named_parameters():
        if token not in name or parameter.grad is None:
            continue
        value = parameter.grad.detach().float().norm().item()
        total += value * value
        count += 1
    return math.sqrt(total), count


def parameter_signature(model, device) -> torch.Tensor:
    """Small deterministic signature used to compare every rank's parameters."""
    samples = []
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        flat = parameter.detach().reshape(-1)
        indices = sorted({0, flat.numel() // 2, flat.numel() - 1})
        samples.append(flat[indices].float())
    if not samples:
        raise RuntimeError("model has no trainable parameter signature")
    return torch.cat(samples).to(device)


def require_all_ranks(condition: bool, device, message: str) -> None:
    flag = torch.tensor(int(condition), device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if flag.item() != 1:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("latent", "explicit"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="microbatches accumulated per optimizer update (default: 1)",
    )
    parser.add_argument("--steps", type=int, default=3, help="number of optimizer updates")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0 or args.steps < 3:
        parser.error(
            "batch size and gradient accumulation steps must be positive, "
            "and --steps must be at least 3"
        )

    torch.multiprocessing.set_start_method("spawn", force=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        seed_all(1234)
        import custom_resolvers  # noqa: F401
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from robot_wm.utils.partial import fix_partial

        experiment = {
            "latent": "ravenhuang/wan-dit/wan_dit_abc_agibot_droid_egodex.yaml",
            "explicit": "ravenhuang/wan-dit/wan_dit_explicit_abc_agibot_droid_egodex.yaml",
        }[args.variant]
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
            cfg = compose(
                config_name="train",
                overrides=[f"+experiments_0908={experiment}"],
            )

        if int(cfg.data_loader.batch_size) != args.batch_size:
            cfg.data_loader.batch_size = args.batch_size
        # Instantiate the exact production loader settings, including worker
        # count, prefetch depth, pinning, and persistence.
        loader = instantiate(cfg.data_loader)
        model = instantiate(cfg.model).to(device)
        ddp = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        # Match Trainer.train(): enable LoRA dropout and training-time gradient
        # checkpointing while WanVAETokenizer.train() keeps the frozen VAE eval.
        ddp.train()
        optimizer = fix_partial(instantiate(cfg.optimizer_factory))(ddp.parameters())
        scheduler = fix_partial(instantiate(cfg.lr_scheduler_factory))(optimizer)
        trainer_cfg = cfg.trainer.config
        amp_enabled = bool(trainer_cfg.amp_enabled)
        dtype = getattr(torch, str(trainer_cfg.dtype))
        scaler = GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None
        seed_all(1234 + rank)

        loader_iter = iter(loader)
        initial_signature = parameter_signature(ddp, device)
        expected_groups = [
            "lora_",
            "forward_model.action_to_control",
            "action_pool",
            "morphology_tokens",
        ]
        expected_groups += (
            ["action_encoder"]
            if args.variant == "explicit"
            else ["inverse_model", "action_decoder"]
        )
        ever_nonzero = {name: False for name in expected_groups}
        seen_morphologies: set[int] = set()
        losses = []
        grad_norm = torch.tensor(float("nan"), device=device)

        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            for microbatch_index in range(args.gradient_accumulation_steps):
                batch = move_to_device(next(loader_iter), device)
                seen_morphologies.update(
                    int(value) for value in batch["morphology_index"].detach().cpu().flatten()
                )
                input_finite = all(
                    torch.isfinite(batch[name]).all().item()
                    for name in ("rgb", "actions")
                    if name in batch
                )
                require_all_ranks(
                    input_finite,
                    device,
                    "non-finite DDP smoke input on at least one rank",
                )

                # DDP must wrap both forward and backward in no_sync(). The final
                # microbatch synchronizes the accumulated gradients once.
                synchronize = microbatch_index + 1 == args.gradient_accumulation_steps
                synchronization_context = nullcontext() if synchronize else ddp.no_sync()
                with synchronization_context:
                    with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp_enabled):
                        loss = ddp(**batch)
                        backward_loss = loss / args.gradient_accumulation_steps
                    flow_loss = ddp.module.aux_losses.get("flow_loss")
                    local_loss_ok = bool(
                        torch.isfinite(loss).all().item()
                        and loss.detach().item() > 0.0
                        and flow_loss is not None
                        and torch.isfinite(flow_loss).all().item()
                        and flow_loss.detach().item() > 0.0
                    )
                    require_all_ranks(
                        local_loss_ok,
                        device,
                        "non-positive/non-finite loss or flow loss on at least one rank",
                    )

                    if amp_enabled:
                        scaler.scale(backward_loss).backward()
                    else:
                        backward_loss.backward()
                losses.append(loss.detach().float())

            # Unscale, validate, clip, update, and schedule exactly once per
            # optimizer update, after the final microbatch synchronized DDP.
            if amp_enabled:
                scaler.unscale_(optimizer)
            trainable_grads = [
                parameter.grad
                for parameter in ddp.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            gradients_ok = bool(
                trainable_grads
                and all(torch.isfinite(grad).all().item() for grad in trainable_grads)
            )
            require_all_ranks(gradients_ok, device, "missing or non-finite trainable gradients")

            step_norms = {}
            for group in expected_groups:
                norm, _ = group_grad_norm(ddp, group)
                step_norms[group] = norm
                ever_nonzero[group] = ever_nonzero[group] or norm > 0.0
            require_all_ranks(
                step_norms["lora_"] > 0.0
                and step_norms["forward_model.action_to_control"] > 0.0,
                device,
                f"zero shared conditioning/LoRA gradient at optimizer step {step}",
            )

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in ddp.parameters() if parameter.requires_grad],
                max_norm=float(trainer_cfg.gradient_clipping.max_norm),
                norm_type=float(trainer_cfg.gradient_clipping.norm_type),
                error_if_nonfinite=True,
            )
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

        delayed_groups = ["action_pool", "morphology_tokens"]
        delayed_groups += (
            ["action_encoder"]
            if args.variant == "explicit"
            else ["inverse_model", "action_decoder"]
        )
        for group in delayed_groups:
            require_all_ranks(
                ever_nonzero[group],
                device,
                f"no gradient reached required group {group!r} on at least one rank",
            )

        gathered_morphologies = [None] * world_size
        dist.all_gather_object(gathered_morphologies, sorted(seen_morphologies))
        global_morphologies = sorted(
            {value for values in gathered_morphologies for value in values}
        )
        if global_morphologies != [0, 2, 6, 9]:
            raise RuntimeError(
                f"DDP smoke did not cover all morphologies: {global_morphologies}"
            )

        final_signature = parameter_signature(ddp, device)
        require_all_ranks(
            not torch.equal(initial_signature, final_signature),
            device,
            "optimizer/scheduler produced no sampled parameter update",
        )
        gathered_signatures = [torch.empty_like(final_signature) for _ in range(world_size)]
        dist.all_gather(gathered_signatures, final_signature)
        max_parameter_difference = max(
            float((value - gathered_signatures[0]).abs().max().item())
            for value in gathered_signatures
        )
        if max_parameter_difference > 1e-6:
            raise RuntimeError(
                f"trainable parameters diverged across ranks: {max_parameter_difference}"
            )

        loss_sum = torch.stack(losses).mean()
        dist.all_reduce(loss_sum)
        max_memory = torch.tensor(
            [torch.cuda.max_memory_allocated(device)], device=device, dtype=torch.float64
        )
        dist.all_reduce(max_memory, op=dist.ReduceOp.MAX)
        dist.barrier(device_ids=[local_rank])
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "world_size": world_size,
                        "variant": args.variant,
                        # Keep the original field for report consumers while
                        # making physical versus effective batch semantics explicit.
                        "per_gpu_batch_size": args.batch_size,
                        "physical_per_gpu_batch_size": args.batch_size,
                        "physical_global_batch_size": args.batch_size * world_size,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "microbatches_per_optimizer_step": (
                            args.gradient_accumulation_steps
                        ),
                        "effective_per_gpu_batch_size": (
                            args.batch_size * args.gradient_accumulation_steps
                        ),
                        "effective_global_batch_size": (
                            args.batch_size * args.gradient_accumulation_steps * world_size
                        ),
                        "optimizer_steps": args.steps,
                        "microbatch_steps": args.steps * args.gradient_accumulation_steps,
                        "steps": args.steps,
                        "mean_loss": float(loss_sum.item() / world_size),
                        "rank0_grad_norm": float(grad_norm),
                        "max_memory_allocated_gib": float(max_memory.item() / 2**30),
                        "loader": {
                            "num_workers": int(cfg.data_loader.num_workers),
                            "prefetch_factor": int(cfg.data_loader.prefetch_factor),
                            "persistent_workers": bool(cfg.data_loader.persistent_workers),
                        },
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "morphologies": global_morphologies,
                        "groups_ever_nonzero_rank0": ever_nonzero,
                        "max_parameter_difference": max_parameter_difference,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
