#!/usr/bin/env python3
"""Small torchrun/NCCL collective probe used before an official launch."""

from __future__ import annotations

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    print(f"rank={rank} local_rank={local_rank} stage=init", flush=True)
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    try:
        print(f"rank={rank} stage=all_reduce", flush=True)
        value = torch.tensor([rank + 1], device="cuda", dtype=torch.bfloat16)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) // 2
        if float(value.item()) != float(expected):
            raise RuntimeError(
                f"rank {rank}: all-reduce returned {value.item()}, expected {expected}"
            )
        print(f"rank={rank} stage=barrier", flush=True)
        dist.barrier(device_ids=[local_rank])
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "backend": dist.get_backend(),
                        "world_size": world_size,
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "nccl": list(torch.cuda.nccl.version()),
                        "all_reduce_sum": float(value.item()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
