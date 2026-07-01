#!/usr/bin/env python3
"""Topology and realistic-payload NCCL probe used before an official launch."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.distributed_topology import validate_rank_topology


def payload_mib(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 1024:
        raise argparse.ArgumentTypeError("payload MiB must be between 1 and 1024")
    return parsed


def positive_timeout_seconds(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("timeout seconds must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument(
        "--timeout-seconds",
        type=positive_timeout_seconds,
        default=1200,
        help="process-group initialization timeout (default: 1200 seconds)",
    )
    parser.add_argument(
        "--payload-mib",
        type=payload_mib,
        default=64,
        help="FP32 all-reduce payload per rank (default: 64 MiB)",
    )
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    print(f"rank={rank} local_rank={local_rank} stage=init", flush=True)
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=args.timeout_seconds)
    )
    try:
        records = [None] * world_size
        dist.all_gather_object(
            records,
            {
                "hostname": socket.gethostname(),
                "rank": rank,
                "local_rank": local_rank,
            },
        )
        topology = validate_rank_topology(
            records,
            expected_nodes=args.expected_nodes,
            gpus_per_node=args.gpus_per_node,
        )
        element_count = args.payload_mib * 1024 * 1024 // torch.float32.itemsize
        print(
            f"rank={rank} stage=all_reduce payload_mib={args.payload_mib}",
            flush=True,
        )
        value = torch.full(
            (element_count,),
            rank + 1,
            device=torch.device("cuda", local_rank),
            dtype=torch.float32,
        )
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) // 2
        probes = value[[0, element_count // 2, element_count - 1]].tolist()
        if probes != [float(expected)] * len(probes):
            raise RuntimeError(
                f"rank {rank}: all-reduce probes returned {probes}, expected {expected}"
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
                        "expected_nodes": args.expected_nodes,
                        "gpus_per_node": args.gpus_per_node,
                        "topology": topology,
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "nccl": list(torch.cuda.nccl.version()),
                        "all_reduce_sum": float(probes[0]),
                        "payload_mib": args.payload_mib,
                        "payload_dtype": "float32",
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
