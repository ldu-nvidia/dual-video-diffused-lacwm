"""Validation helpers shared by the NCCL and real-data DDP qualification gates."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


def validate_rank_topology(
    records: Iterable[Mapping[str, object]],
    *,
    expected_nodes: int,
    gpus_per_node: int,
) -> dict[str, list[int]]:
    """Return host -> local ranks after validating a dense fixed-size topology."""
    if not 1 <= expected_nodes <= 32:
        raise ValueError(f"expected_nodes must be in [1, 32], got {expected_nodes}")
    if gpus_per_node <= 0:
        raise ValueError(f"gpus_per_node must be positive, got {gpus_per_node}")

    rows = list(records)
    expected_world_size = expected_nodes * gpus_per_node
    if len(rows) != expected_world_size:
        raise RuntimeError(
            f"rank topology contains {len(rows)} records, expected {expected_world_size}"
        )

    by_host: dict[str, list[int]] = defaultdict(list)
    global_ranks = []
    for row in rows:
        hostname = row.get("hostname")
        rank = row.get("rank")
        local_rank = row.get("local_rank")
        if not isinstance(hostname, str) or not hostname:
            raise RuntimeError(f"invalid topology hostname: {hostname!r}")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise RuntimeError(f"invalid global rank for {hostname}: {rank!r}")
        if isinstance(local_rank, bool) or not isinstance(local_rank, int):
            raise RuntimeError(f"invalid local rank for {hostname}: {local_rank!r}")
        global_ranks.append(rank)
        by_host[hostname].append(local_rank)

    if sorted(global_ranks) != list(range(expected_world_size)):
        raise RuntimeError(
            f"global ranks are not exactly 0..{expected_world_size - 1}: "
            f"{sorted(global_ranks)}"
        )
    if len(by_host) != expected_nodes:
        raise RuntimeError(
            f"topology spans {len(by_host)} host(s), expected {expected_nodes}: "
            f"{sorted(by_host)}"
        )

    expected_local_ranks = list(range(gpus_per_node))
    normalized = {}
    for hostname, local_ranks in sorted(by_host.items()):
        local_ranks = sorted(local_ranks)
        if local_ranks != expected_local_ranks:
            raise RuntimeError(
                f"host {hostname!r} has local ranks {local_ranks}, "
                f"expected {expected_local_ranks}"
            )
        normalized[hostname] = local_ranks
    return normalized
