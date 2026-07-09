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


TOPOLOGY_MIGRATION_KIND = "topology_migration_reset_rank_state"


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
    parent_run_identity: Path | None = None,
    checkpoint_ack: Path | None = None,
    transition_kind: str | None = None,
    target_batch_size: int | None = None,
    target_gradient_accumulation_steps: int | None = None,
    target_world_size: int | None = None,
    authorization_basis: str | None = None,
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

    payload = {
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
    migration_values = (
        parent_run_identity,
        checkpoint_ack,
        transition_kind,
        target_batch_size,
        target_gradient_accumulation_steps,
        target_world_size,
        authorization_basis,
    )
    if any(value is not None for value in migration_values):
        if not all(value is not None for value in migration_values):
            raise ValueError(
                "topology migration requires parent_run_identity, checkpoint_ack, "
                "transition_kind, target batching/topology, and authorization_basis"
            )
        if transition_kind != TOPOLOGY_MIGRATION_KIND:
            raise ValueError(
                f"transition_kind must be {TOPOLOGY_MIGRATION_KIND!r}"
            )
        if not isinstance(authorization_basis, str) or not authorization_basis.strip():
            raise ValueError("authorization_basis must be a non-empty string")
        identity_path = parent_run_identity.expanduser().resolve(strict=True)
        if not identity_path.is_file():
            raise ValueError(
                f"parent run identity is not a regular file: {identity_path}"
            )
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"unable to read parent run identity {identity_path}: {exc}"
            ) from exc
        if identity.get("identity_sha256") != expected_identity:
            raise ValueError(
                "parent run identity file does not match the checkpoint identity"
            )

        def positive_int(value, *, field_name: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
            return value

        parent_batch_size = positive_int(
            identity.get("batch_size"), field_name="parent batch_size"
        )
        parent_accumulation = positive_int(
            identity.get("gradient_accumulation_steps"),
            field_name="parent gradient_accumulation_steps",
        )
        parent_world_size = positive_int(
            identity.get("world_size"), field_name="parent world_size"
        )
        if parent_accumulation != int(snapshot["gradient_accumulation_steps"]):
            raise ValueError(
                "parent run identity gradient accumulation differs from snapshot"
            )
        if parent_world_size != int(snapshot["world_size"]):
            raise ValueError("parent run identity world size differs from snapshot")
        recorded_global_batch = positive_int(
            identity.get("effective_global_batch_size"),
            field_name="parent effective_global_batch_size",
        )
        parent_global_batch = (
            parent_batch_size * parent_accumulation * parent_world_size
        )
        if parent_global_batch != recorded_global_batch:
            raise ValueError(
                "parent run identity has an inconsistent effective global batch"
            )
        target_batch_size = positive_int(
            target_batch_size, field_name="target_batch_size"
        )
        target_gradient_accumulation_steps = positive_int(
            target_gradient_accumulation_steps,
            field_name="target_gradient_accumulation_steps",
        )
        target_world_size = positive_int(
            target_world_size, field_name="target_world_size"
        )
        target_global_batch = (
            target_batch_size
            * target_gradient_accumulation_steps
            * target_world_size
        )
        if target_global_batch != parent_global_batch:
            raise ValueError(
                "topology migration must preserve the effective global batch: "
                f"parent={parent_global_batch}, target={target_global_batch}"
            )
        if (
            target_batch_size == parent_batch_size
            and target_gradient_accumulation_steps == parent_accumulation
            and target_world_size == parent_world_size
        ):
            raise ValueError(
                "topology migration requires a changed batch/topology tuple"
            )

        checkpoint_ack_path = checkpoint_ack.expanduser().resolve(strict=True)
        if not checkpoint_ack_path.is_file():
            raise ValueError(
                f"checkpoint ACK is not a regular file: {checkpoint_ack_path}"
            )
        try:
            ack = json.loads(checkpoint_ack_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"unable to read checkpoint ACK {checkpoint_ack_path}: {exc}"
            ) from exc
        if ack.get("schema_version") != 1 or ack.get("checkpoint_written") is not True:
            raise ValueError("checkpoint ACK does not confirm a durable checkpoint")
        if ack.get("run_identity_sha256") != expected_identity:
            raise ValueError("checkpoint ACK run identity mismatch")
        try:
            ack_snapshot = Path(str(ack.get("snapshot", ""))).expanduser().resolve(
                strict=True
            )
        except (OSError, RuntimeError) as exc:
            raise ValueError("checkpoint ACK snapshot path is invalid") from exc
        if ack_snapshot != snapshot_path:
            raise ValueError("checkpoint ACK points to a different snapshot")
        if ack.get("next_iter") != int(snapshot["_start_iter"]):
            raise ValueError("checkpoint ACK next_iter differs from snapshot")
        payload.update(
            {
                "schema_version": 2,
                "transition_kind": TOPOLOGY_MIGRATION_KIND,
                "rank_local_state_policy": "reset",
                "authorization_basis": authorization_basis.strip(),
                "parent_run_identity": str(identity_path),
                "parent_run_identity_file_sha256": _sha256_file(identity_path),
                "checkpoint_ack": str(checkpoint_ack_path),
                "checkpoint_ack_sha256": _sha256_file(checkpoint_ack_path),
                "checkpoint_ack_next_iter": int(ack["next_iter"]),
                "parent_batch_size": parent_batch_size,
                "parent_effective_global_batch_size": parent_global_batch,
                "target_batch_size": target_batch_size,
                "target_gradient_accumulation_steps": (
                    target_gradient_accumulation_steps
                ),
                "target_world_size": target_world_size,
                "target_effective_global_batch_size": target_global_batch,
            }
        )
    return payload


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        immutable_keys = (
            "parent_snapshot",
            "parent_snapshot_sha256",
            "parent_run_identity_sha256",
            "transition_kind",
            "rank_local_state_policy",
            "authorization_basis",
            "parent_run_identity",
            "parent_run_identity_file_sha256",
            "checkpoint_ack",
            "checkpoint_ack_sha256",
            "checkpoint_ack_next_iter",
            "parent_batch_size",
            "parent_world_size",
            "parent_gradient_accumulation_steps",
            "parent_effective_global_batch_size",
            "target_batch_size",
            "target_world_size",
            "target_gradient_accumulation_steps",
            "target_effective_global_batch_size",
        )
        if (
            existing.get("schema_version") == payload.get("schema_version")
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
    parser.add_argument(
        "--parent-run-identity",
        type=Path,
        help="Required with target topology arguments for a guarded migration",
    )
    parser.add_argument("--checkpoint-ack", type=Path)
    parser.add_argument("--transition-kind", choices=(TOPOLOGY_MIGRATION_KIND,))
    parser.add_argument("--target-batch-size", type=int)
    parser.add_argument("--target-gradient-accumulation-steps", type=int)
    parser.add_argument("--target-world-size", type=int)
    parser.add_argument("--authorization-basis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_handoff(
        parent_snapshot=args.parent_snapshot,
        expected_parent_run_identity_sha256=(
            args.expected_parent_run_identity_sha256
        ),
        parent_run_identity=args.parent_run_identity,
        checkpoint_ack=args.checkpoint_ack,
        transition_kind=args.transition_kind,
        target_batch_size=args.target_batch_size,
        target_gradient_accumulation_steps=(
            args.target_gradient_accumulation_steps
        ),
        target_world_size=args.target_world_size,
        authorization_basis=args.authorization_basis,
    )
    output = args.output
    if output is None:
        output = args.parent_snapshot.expanduser().resolve().parent / "handoff_complete.json"
    _atomic_write_json(output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
