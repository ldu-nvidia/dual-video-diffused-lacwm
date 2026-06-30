#!/usr/bin/env python3
"""Validate scheduler ACK/completion files before changing Slurm state."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EXPECTED_MAX_ITER = 60_000


def _load_json(path: Path, description: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} must contain a JSON object: {path}")
    return payload


def _canonical(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def _identity_digest(identity_path: Path) -> str:
    identity = _load_json(identity_path, "run identity")
    digest = identity.get("identity_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError(f"run identity has no valid identity_sha256: {identity_path}")
    return digest


def _expect(payload: dict, key: str, expected: object, description: str) -> None:
    actual = payload.get(key)
    if actual != expected:
        raise RuntimeError(
            f"{description} field {key!r} is {actual!r}, expected {expected!r}"
        )


def _expect_int(payload: dict, key: str, expected: int, description: str) -> None:
    actual = payload.get(key)
    if type(actual) is not int or actual != expected:
        raise RuntimeError(
            f"{description} field {key!r} is {actual!r}, expected integer {expected}"
        )


def _validate_snapshot(payload: dict, run_dir: Path, *, required: bool) -> None:
    expected = run_dir / "snapshot.pt"
    raw = payload.get("snapshot")
    if not isinstance(raw, str) or _canonical(Path(raw)) != _canonical(expected):
        raise RuntimeError(
            f"state file snapshot is {raw!r}, expected {_canonical(expected)!r}"
        )
    if required and (not expected.is_file() or expected.stat().st_size <= 0):
        raise RuntimeError(f"required snapshot is missing or empty: {expected}")


def validate_ack(args: argparse.Namespace) -> None:
    payload = _load_json(args.path, "checkpoint ACK")
    digest = _identity_digest(args.identity)
    _expect_int(payload, "schema_version", 1, "checkpoint ACK")
    _expect(
        payload,
        "status",
        "checkpointed_for_reschedule",
        "checkpoint ACK",
    )
    _expect(payload, "slurm_attempt_id", args.attempt_id, "checkpoint ACK")
    _expect_int(payload, "max_iter", args.expected_max_iter, "checkpoint ACK")
    _expect(payload, "run_identity_sha256", digest, "checkpoint ACK")

    next_iter = payload.get("next_iter")
    if (
        not isinstance(next_iter, int)
        or isinstance(next_iter, bool)
        or not 0 <= next_iter <= args.expected_max_iter
    ):
        raise RuntimeError(f"checkpoint ACK next_iter is invalid: {next_iter!r}")
    checkpoint_written = payload.get("checkpoint_written")
    if not isinstance(checkpoint_written, bool):
        raise RuntimeError(
            "checkpoint ACK checkpoint_written must be a JSON boolean"
        )
    # A request received before update zero may acknowledge a clean retry without
    # a snapshot. Every positive continuation must reference a durable snapshot.
    snapshot_required = checkpoint_written or next_iter > 0
    _validate_snapshot(payload, args.run_dir, required=snapshot_required)


def validate_completion(args: argparse.Namespace) -> None:
    payload = _load_json(args.path, "completion marker")
    digest = _identity_digest(args.identity)
    _expect_int(payload, "schema_version", 1, "completion marker")
    _expect(payload, "status", "completed", "completion marker")
    _expect_int(payload, "max_iter", args.expected_max_iter, "completion marker")
    _expect_int(
        payload,
        "completed_updates",
        args.expected_max_iter,
        "completion marker",
    )
    _expect(payload, "run_identity_sha256", digest, "completion marker")
    _validate_snapshot(payload, args.run_dir, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)
    for name in ("ack", "completion"):
        child = subparsers.add_parser(name)
        child.add_argument("--path", type=Path, required=True)
        child.add_argument("--identity", type=Path, required=True)
        child.add_argument("--run-dir", type=Path, required=True)
        child.add_argument(
            "--expected-max-iter", type=int, default=EXPECTED_MAX_ITER
        )
        if name == "ack":
            child.add_argument("--attempt-id", required=True)

    args = parser.parse_args()
    try:
        if args.kind == "ack":
            validate_ack(args)
        else:
            validate_completion(args)
    except RuntimeError as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"Validated {args.kind}: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
