#!/usr/bin/env python3
"""Seal a completed dataset-stage checkpoint for a child training lineage.

The resulting ``handoff_complete.json`` is an immutable provenance boundary.
The child trainer verifies both the parent run identity and the complete
snapshot byte digest before loading any state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


def _require_sha256(value: str, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest"
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(stat_result) -> tuple[int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
    )


def build_handoff(
    *,
    parent_snapshot: Path,
    expected_parent_run_identity_sha256: str,
) -> dict:
    expected_identity = _require_sha256(
        expected_parent_run_identity_sha256,
        field_name="expected_parent_run_identity_sha256",
    )
    snapshot_path = parent_snapshot.expanduser().resolve(strict=True)
    if not snapshot_path.is_file():
        raise ValueError(f"parent snapshot is not a regular file: {snapshot_path}")
    if snapshot_path.name.endswith(".tmp"):
        raise ValueError("refusing to seal a temporary checkpoint")

    stat_before = snapshot_path.stat()
    snapshot_sha256 = _sha256_file(snapshot_path)
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    stat_after = snapshot_path.stat()
    if _file_identity(stat_before) != _file_identity(stat_after):
        raise RuntimeError("parent snapshot changed while the handoff was being sealed")

    if snapshot.get("snapshot_schema_version") != 3:
        raise ValueError(
            "parent snapshot must use snapshot_schema_version=3 for an exact "
            "distributed checkpoint"
        )
    actual_identity = snapshot.get("run_identity_sha256")
    if actual_identity != expected_identity:
        raise ValueError(
            "parent run identity mismatch: "
            f"snapshot={actual_identity!r}, expected={expected_identity!r}"
        )
    required = {
        "model",
        "optimizer",
        "lr_scheduler",
        "_start_iter",
        "_total_observations",
        "gradient_accumulation_steps",
        "world_size",
        "rank_states",
    }
    missing = sorted(required - set(snapshot))
    if missing:
        raise ValueError(f"parent snapshot is missing required fields: {missing}")
    world_size = snapshot["world_size"]
    rank_states = snapshot["rank_states"]
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
    ):
        raise ValueError("parent snapshot world_size must be a positive integer")
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise ValueError(
            "parent snapshot rank-state count does not match its world size"
        )
    rank_order = [state.get("global_rank") for state in rank_states]
    if rank_order != list(range(world_size)):
        raise ValueError("parent snapshot rank states are missing or reordered")

    return {
        "schema_version": 1,
        "status": "complete",
        "parent_snapshot": str(snapshot_path),
        "parent_snapshot_sha256": snapshot_sha256,
        "parent_snapshot_size_bytes": int(stat_after.st_size),
        "parent_run_identity_sha256": expected_identity,
        "parent_next_iteration": int(snapshot["_start_iter"]),
        "parent_total_observations": int(snapshot["_total_observations"]),
        "parent_world_size": int(world_size),
        "parent_gradient_accumulation_steps": int(
            snapshot["gradient_accumulation_steps"]
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        immutable_keys = (
            "parent_snapshot",
            "parent_snapshot_sha256",
            "parent_run_identity_sha256",
        )
        if (
            existing.get("schema_version") == 1
            and existing.get("status") == "complete"
            and all(
                existing.get(key) == payload.get(key) for key in immutable_keys
            )
        ):
            return
        raise FileExistsError(
            f"refusing to replace an existing handoff with different provenance: {path}"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-snapshot", type=Path, required=True)
    parser.add_argument(
        "--expected-parent-run-identity-sha256",
        required=True,
        help="Guarded identity recorded in the parent snapshot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to handoff_complete.json beside the parent snapshot",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_handoff(
        parent_snapshot=args.parent_snapshot,
        expected_parent_run_identity_sha256=(
            args.expected_parent_run_identity_sha256
        ),
    )
    output = args.output
    if output is None:
        output = args.parent_snapshot.expanduser().resolve().parent / "handoff_complete.json"
    _atomic_write_json(output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
