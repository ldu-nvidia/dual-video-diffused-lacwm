#!/usr/bin/env python3
"""Hash and compare checkpoint model tensors without pickle-byte ambiguity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    path = path.expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"regular snapshot required: {path}")
    snapshot = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = snapshot.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"snapshot lacks model dictionary: {path}")
    digest = hashlib.sha256()
    schema = []
    per_tensor = {}
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"non-tensor model entry: {name}")
        tensor = value.detach().cpu().contiguous()
        item = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        schema.append(item)
        raw = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
        tensor_digest = hashlib.sha256()
        tensor_digest.update(canonical_json(item))
        tensor_digest.update(b"\0")
        tensor_digest.update(raw)
        per_tensor[name] = tensor_digest.hexdigest()
        digest.update(canonical_json(item))
        digest.update(b"\0")
        digest.update(raw)
    receipt = {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "file_bytes": path.stat().st_size,
        "canonical_model_state_sha256": digest.hexdigest(),
        "model_schema_sha256": hashlib.sha256(canonical_json(schema)).hexdigest(),
        "model_tensor_count": len(schema),
        "snapshot_metadata": {
            key: snapshot.get(key)
            for key in (
                "snapshot_schema_version",
                "run_identity_sha256",
                "_start_iter",
                "world_size",
                "gradient_accumulation_steps",
            )
        },
    }
    return receipt, per_tensor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.snapshot) not in (1, 2):
        parser.error("provide one snapshot for a receipt or two for comparison")
    receipts = []
    hashes = []
    for path in args.snapshot:
        receipt, per_tensor = model_state_receipt(path)
        receipts.append(receipt)
        hashes.append(per_tensor)
    result: dict[str, Any] = {
        "kind": "canonical_checkpoint_model_state_comparison_v1",
        "snapshots": receipts,
    }
    if len(receipts) == 2:
        all_names = sorted(set(hashes[0]) | set(hashes[1]))
        mismatches = [
            name for name in all_names if hashes[0].get(name) != hashes[1].get(name)
        ]
        result["comparison"] = {
            "model_schema_identical": (
                receipts[0]["model_schema_sha256"]
                == receipts[1]["model_schema_sha256"]
            ),
            "model_state_bit_identical": not mismatches,
            "mismatched_tensor_count": len(mismatches),
            "mismatched_tensor_names": mismatches,
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
