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


def compare_model_state_drift(
    reference_path: Path,
    current_path: Path,
) -> dict[str, Any]:
    """Compare compatible finite states without imposing a drift threshold.

    The comparison is deliberately descriptive: schema and finiteness are
    fail-closed, while finite numerical drift is recorded. This is appropriate
    for auditing a CUDA rerun whose causal inputs replay exactly but whose
    floating-point reduction order was not deterministically controlled.
    """

    import math
    import torch

    # Receipt equality is replayed in allocations with different CPU counts.
    # Pin reductions here so descriptive norms cannot vary with thread topology.
    torch.set_num_threads(1)
    paths = [
        reference_path.expanduser().resolve(strict=True),
        current_path.expanduser().resolve(strict=True),
    ]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("regular reference/current snapshots are required")
    snapshots = [
        torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for path in paths
    ]
    states = [snapshot.get("model") for snapshot in snapshots]
    if any(not isinstance(state, dict) for state in states):
        raise ValueError("snapshot lacks model dictionary")
    names = [sorted(state) for state in states]
    if names[0] != names[1]:
        missing = sorted(set(names[0]) - set(names[1]))
        unexpected = sorted(set(names[1]) - set(names[0]))
        raise ValueError(
            f"model tensor keys differ: missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    schemas: list[list[dict[str, Any]]] = [[], []]
    state_digests = [hashlib.sha256(), hashlib.sha256()]
    per_state_hashes: list[dict[str, str]] = [{}, {}]
    mismatches: list[dict[str, Any]] = []
    reference_sq = 0.0
    delta_sq = 0.0
    global_max = 0.0
    compared_elements = 0
    for name in names[0]:
        if any(not isinstance(state[name], torch.Tensor) for state in states):
            raise ValueError(f"non-tensor model entry: {name}")
        tensors = [state[name].detach().cpu().contiguous() for state in states]
        items = [
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            for tensor in tensors
        ]
        if items[0] != items[1]:
            raise ValueError(
                f"model tensor dtype/shape differs: {name}: {items[0]} != {items[1]}"
            )
        for index, (item, tensor) in enumerate(zip(items, tensors, strict=True)):
            schemas[index].append(item)
            if tensor.is_floating_point() or tensor.is_complex():
                if not bool(torch.isfinite(tensor).all()):
                    side = "reference" if index == 0 else "current"
                    raise ValueError(f"non-finite {side} model tensor: {name}")
            raw = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
            tensor_digest = hashlib.sha256()
            tensor_digest.update(canonical_json(item))
            tensor_digest.update(b"\0")
            tensor_digest.update(raw)
            per_state_hashes[index][name] = tensor_digest.hexdigest()
            state_digests[index].update(canonical_json(item))
            state_digests[index].update(b"\0")
            state_digests[index].update(raw)

        reference_flat = tensors[0].reshape(-1)
        current_flat = tensors[1].reshape(-1)
        bitwise_equal = per_state_hashes[0][name] == per_state_hashes[1][name]
        compared_elements += int(reference_flat.numel())
        if bitwise_equal:
            continue
        tensor_delta_sq = 0.0
        tensor_reference_sq = 0.0
        tensor_max = 0.0
        value_mismatches = 0
        for start in range(0, int(reference_flat.numel()), 1_048_576):
            stop = min(start + 1_048_576, int(reference_flat.numel()))
            reference_chunk = reference_flat[start:stop]
            current_chunk = current_flat[start:stop]
            value_mismatches += int(torch.count_nonzero(reference_chunk != current_chunk))
            dtype = torch.complex128 if reference_chunk.is_complex() else torch.float64
            reference_numeric = reference_chunk.to(dtype=dtype)
            delta = current_chunk.to(dtype=dtype) - reference_numeric
            abs_delta = delta.abs()
            tensor_delta_sq += float((abs_delta * abs_delta).sum(dtype=torch.float64))
            tensor_reference_sq += float(
                (reference_numeric.abs() * reference_numeric.abs()).sum(
                    dtype=torch.float64
                )
            )
            if abs_delta.numel():
                tensor_max = max(tensor_max, float(abs_delta.max()))
        mismatches.append(
            {
                "name": name,
                "dtype": items[0]["dtype"],
                "shape": items[0]["shape"],
                "elements": int(reference_flat.numel()),
                "reference_tensor_sha256": per_state_hashes[0][name],
                "current_tensor_sha256": per_state_hashes[1][name],
                "value_mismatch_elements": value_mismatches,
                "reference_l2": math.sqrt(tensor_reference_sq),
                "l2_drift": math.sqrt(tensor_delta_sq),
                "max_abs_drift": tensor_max,
            }
        )
        reference_sq += tensor_reference_sq
        delta_sq += tensor_delta_sq
        global_max = max(global_max, tensor_max)

    if schemas[0] != schemas[1]:
        raise ValueError("model tensor schema differs")
    reference_l2 = math.sqrt(reference_sq)
    drift_l2 = math.sqrt(delta_sq)
    relative_l2 = (
        drift_l2 / reference_l2
        if reference_l2
        else (0.0 if drift_l2 == 0 else None)
    )
    receipts = []
    for path, snapshot, schema, digest in zip(
        paths, snapshots, schemas, state_digests, strict=True
    ):
        receipts.append(
            {
                "path": str(path),
                "file_sha256": sha256_file(path),
                "file_bytes": path.stat().st_size,
                "canonical_model_state_sha256": digest.hexdigest(),
                "model_schema_sha256": hashlib.sha256(
                    canonical_json(schema)
                ).hexdigest(),
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
        )
    return {
        "schema": "finite-compatible-model-state-drift-v1",
        "reference_snapshot": receipts[0],
        "current_snapshot": receipts[1],
        "model_schema_identical": True,
        "all_reference_tensors_finite": True,
        "all_current_tensors_finite": True,
        "model_tensor_count": len(names[0]),
        "compared_elements": compared_elements,
        "cpu_reduction_threads": 1,
        "bitwise_identical_tensor_count": len(names[0]) - len(mismatches),
        "mismatched_tensor_count": len(mismatches),
        "mismatched_tensors": mismatches,
        "global_drift": {
            "scope": "mismatched_tensors_only",
            "reference_l2": reference_l2,
            "drift_l2": drift_l2,
            "relative_l2_vs_reference": relative_l2,
            "max_abs_drift": global_max,
            "numerical_pass_threshold": None,
        },
    }


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
