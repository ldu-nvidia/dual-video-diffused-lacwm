#!/usr/bin/env python3
"""Exact update-zero B200 forward/backward memory canary for MID-ON.

The canary instantiates the production model and optimizer on eight ranks,
strictly applies the registered model-only warm start, and executes one BF16
forward/backward on the production per-rank tensor shapes.  It deliberately
does not call ``optimizer.step`` and emits no scientific metric.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "projects/latent_action_models"
CONFIG_ROOT = PROJECT_ROOT / "configs"
for root in (str(REPO_ROOT / "tools"), str(REPO_ROOT), str(PROJECT_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

import custom_resolvers  # noqa: E402,F401
import intra_forward_forcing_screen as contract  # noqa: E402
from robot_wm.utils.partial import fix_partial  # noqa: E402
from validate_intra_forward_warmstart import EXCLUDED_PREFIXES  # noqa: E402


class SmokeError(RuntimeError):
    pass


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _synthetic_rgb(device: torch.device | str) -> torch.Tensor:
    """Return a deterministic, non-constant production-shape RGB fixture.

    The model's loss mask treats a constant width-stacked camera view as a
    padded/missing view.  A zero image therefore exercises the full forward
    graph but deliberately masks every supervised pixel.  This bounded linear
    pattern keeps all three views valid while remaining deterministic and
    independent of any dataset example.
    """

    time = torch.linspace(-0.10, 0.10, 13, device=device).view(1, 13, 1, 1, 1)
    channel = torch.tensor([-0.05, 0.0, 0.05], device=device).view(
        1, 1, 3, 1, 1
    )
    height = torch.linspace(-0.25, 0.25, 180, device=device).view(
        1, 1, 1, 180, 1
    )
    width = torch.linspace(-0.25, 0.25, 960, device=device).view(
        1, 1, 1, 1, 960
    )
    return (time + channel + height + width).contiguous()


def _strict_model_only_load(model, checkpoint: Path) -> None:
    current = model.state_dict()
    expected_missing = sorted(
        key for key in current if key.startswith(EXCLUDED_PREFIXES)
    )
    snapshot = torch.load(
        checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("_start_iter") != 1000
        or not isinstance(snapshot.get("model"), dict)
    ):
        raise SmokeError("historical warm-start schema/cursor differs")
    filtered = {
        key: value
        for key, value in snapshot["model"].items()
        if not key.startswith(EXCLUDED_PREFIXES)
    }
    incompatible = model.load_state_dict(filtered, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if missing != expected_missing or unexpected:
        raise SmokeError(
            "strict model-only warm start differs: "
            f"missing={missing}, expected={expected_missing}, "
            f"unexpected={unexpected}"
        )


def run(args: argparse.Namespace) -> int:
    if not args.output.is_absolute() or args.output.exists():
        raise SmokeError("output must be a fresh absolute path")
    registration = contract.load_registration(
        args.registration.resolve(strict=True), verify_files=False
    )
    expected_output = Path(registration["study_root"]) / "memory_smoke.json"
    if args.output != expected_output or not args.output.parent.is_dir():
        raise SmokeError(f"output must be {expected_output}")
    runtime_path = args.runtime_receipt.resolve(strict=True)
    if runtime_path != Path(registration["study_root"]) / "memory_smoke_runtime.json":
        raise SmokeError("runtime receipt path differs from the study contract")
    contract.validate_runtime_receipt(
        runtime_path, registration, label="memory smoke runtime"
    )

    if not torch.cuda.is_available():
        raise SmokeError("CUDA is unavailable")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world != contract.WORLD_SIZE or not 0 <= local_rank < world:
        raise SmokeError("memory smoke requires one eight-GPU torchrun node")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    try:
        # Production construction uses the same seed on every rank. DDP then
        # verifies/broadcasts the common initialized state before rank-local
        # diffusion RNG is selected.
        _seed_all(contract.SEED)
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
            config = compose(
                config_name="train",
                overrides=[f"+experiments_0908={contract.ARMS[1].selector}"],
            )
        model_config = OmegaConf.to_container(config.model, resolve=True)
        dual = model_config["dual_diffusion"]
        intra = dual["intra_forward_forcing"]
        arm = contract.ARMS[1]
        if (
            intra
            != {
                "enabled": True,
                "block_index": contract.MIDPOINT_BLOCK_INDEX,
                "stop_gradient": True,
                "history_bins": contract.AUXILIARY_HISTORY_BINS,
            }
            or bool(model_config["forward_model"]["gradient_checkpointing"])
            or dual.get("condition_mode") != arm.condition_mode
            or dual.get("condition_on_tf") is not arm.condition_on_state
            or dual.get("condition_on_tf_clock") is not arm.condition_on_clock
            or dual.get("schedule_mode") != arm.schedule_mode
            or dual.get("tf_lead_logit") != arm.lead_logit
            or dual.get("tf_loss_weight") != arm.auxiliary_loss_weight
            or dual.get("evaluation_condition_sources") != contract.SOURCES
        ):
            raise SmokeError("resolved MID-ON memory-smoke architecture differs")
        model = instantiate(config.model)
        _strict_model_only_load(model, Path(registration["warm_start"]["path"]))
        model.train()
        ddp = DDP(
            model.to(device),
            device_ids=[local_rank],
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        optimizer = fix_partial(instantiate(config.optimizer_factory))(
            ddp.parameters()
        )
        if optimizer.state:
            raise SmokeError("optimizer state is nonempty before update zero")

        _seed_all(contract.SEED + rank)
        batch = {
            "rgb": _synthetic_rgb(device),
            "actions": torch.zeros(
                (1, 13, 5, 157), dtype=torch.float32, device=device
            ),
            "mask": torch.ones((1, 13), dtype=torch.bool, device=device),
            "morphology_index": torch.zeros((1,), dtype=torch.long, device=device),
            "clip_index": torch.tensor([rank], dtype=torch.long, device=device),
        }
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = ddp(**batch)
        finite_loss = bool(torch.isfinite(loss.detach()).all().item())
        if not finite_loss:
            raise SmokeError("update-zero loss is non-finite")
        loss.backward()
        torch.cuda.synchronize(device)
        gradients = [
            parameter.grad
            for parameter in ddp.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        finite_gradients = bool(gradients) and all(
            bool(torch.isfinite(gradient).all().item()) for gradient in gradients
        )
        if not finite_gradients:
            raise SmokeError("update-zero gradients are absent or non-finite")
        if optimizer.state:
            raise SmokeError("optimizer state changed without optimizer.step")
        properties = torch.cuda.get_device_properties(device)
        local = {
            "rank": rank,
            "local_rank": local_rank,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "total_memory_bytes": int(properties.total_memory),
            "finite_loss": finite_loss,
            "finite_gradients": finite_gradients,
        }
        gathered: list[dict | None] = [None for _ in range(world)]
        dist.all_gather_object(gathered, local)
        dist.barrier()
        if rank == 0:
            ranks = sorted(
                (dict(item) for item in gathered if item),
                key=lambda item: item["rank"],
            )
            payload = contract.identity_payload(
                {
                    "schema": contract.MEMORY_SMOKE_SCHEMA,
                    "status": "pass",
                    "registration_identity_sha256": registration[
                        "identity_sha256"
                    ],
                    "source_commit": registration["source"]["commit"],
                    "selector": contract.ARMS[1].selector,
                    "world_size": world,
                    "synthetic_batch_shapes": {
                        "rgb": [1, 13, 3, 180, 960],
                        "actions": [1, 13, 5, 157],
                        "mask": [1, 13],
                        "morphology_index": [1],
                        "clip_index": [1],
                    },
                    "synthetic_rgb_pattern": contract.MEMORY_SMOKE_RGB_PATTERN,
                    "dtype": "torch.bfloat16 autocast",
                    "gradient_checkpointing": False,
                    "forward_completed": True,
                    "backward_completed": True,
                    "optimizer_step_executed": False,
                    "completed_optimizer_updates": 0,
                    "optimizer_state_entries": 0,
                    "runtime_receipt_sha256": contract.sha256_file(runtime_path),
                    "ranks": ranks,
                    "maximum_peak_allocated_bytes": max(
                        item["peak_allocated_bytes"] for item in ranks
                    ),
                    "minimum_headroom_bytes": min(
                        item["total_memory_bytes"] - item["peak_allocated_bytes"]
                        for item in ranks
                    ),
                    "scientific_metrics_emitted": False,
                }
            )
            contract.exclusive_json(args.output, payload)
            print(json.dumps(payload, sort_keys=True))
        dist.barrier()
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (SmokeError, contract.ContractError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
