#!/usr/bin/env python3
"""Bounded real-data smoke test for the production four-source loader.

This utility composes the exact production Hydra experiment, instantiates its
``StatefulDataLoader``, and decodes real samples until every configured source
has appeared.  It also snapshots and restores the loader so an official launch
does not discover checkpoint-continuity problems after consuming B200 time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "projects" / "latent_action_models"
CONFIG_ROOT = PROJECT_ROOT / "configs"
EXPERIMENT = "ravenhuang/wan-dit/wan_dit_abc_agibot_droid_egodex.yaml"
EXPECTED_SOURCES = ("Droid", "EgoDex", "Agibot", "ABC")
EXPECTED_LENGTHS = (10_000, 10_000, 5_671, 10_000)
EXPECTED_MORPHOLOGIES = {0: 0, 1: 2, 2: 6, 3: 9}
REQUIRED_EVIDENCE = (
    ".prepared/manifests.ready",
    "egodex_cdn/manifest.csv",
    "agibot/manifest.csv",
    "abc_pp/manifest.txt",
)

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def atomic_torch_save(path: Path, value: Any) -> None:
    import torch

    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def load_torch_state(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older supported PyTorch releases
        return torch.load(path, map_location="cpu")


def compose_config() -> Any:
    from hydra import compose, initialize_config_dir

    # Importing this module registers the mul/div OmegaConf resolvers used by
    # the production experiment.
    import custom_resolvers  # noqa: F401

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        return compose(
            config_name="train",
            overrides=[f"+experiments_0908={EXPERIMENT}"],
        )


def require_finite(name: str, value: Any, chunk_elements: int = 8_000_000) -> None:
    import torch

    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        return
    flattened = value.detach().reshape(-1)
    for offset in range(0, flattened.numel(), chunk_elements):
        if not torch.isfinite(flattened[offset : offset + chunk_elements]).all().item():
            raise RuntimeError(f"batch tensor {name!r} contains NaN or Inf")


def tensor_shape(value: Any) -> list[int]:
    return [int(dim) for dim in value.shape]


def validate_batch(batch: Any, expected_batch_size: int) -> dict[str, Any]:
    import torch

    if not isinstance(batch, dict):
        raise RuntimeError(f"loader returned {type(batch).__name__}, expected dict")
    required = {"rgb", "actions", "mask", "dataset_index", "morphology_index"}
    missing = sorted(required.difference(batch))
    if missing:
        raise RuntimeError(f"loader batch is missing required keys: {missing}")

    for name in required:
        if not isinstance(batch[name], torch.Tensor):
            raise RuntimeError(f"batch field {name!r} is not a tensor")

    rgb = batch["rgb"]
    actions = batch["actions"]
    mask = batch["mask"]
    if tuple(rgb.shape) != (expected_batch_size, 13, 3, 180, 960):
        raise RuntimeError(f"unexpected rgb shape: {tuple(rgb.shape)}")
    if tuple(actions.shape) != (expected_batch_size, 13, 5, 157):
        raise RuntimeError(f"unexpected actions shape: {tuple(actions.shape)}")
    if tuple(mask.shape[:2]) != (expected_batch_size, 13):
        raise RuntimeError(f"unexpected mask shape: {tuple(mask.shape)}")

    for name, value in batch.items():
        require_finite(name, value)

    dataset_ids = [int(item) for item in batch["dataset_index"].reshape(-1).tolist()]
    morphology_ids = [
        int(item) for item in batch["morphology_index"].reshape(-1).tolist()
    ]
    if len(dataset_ids) != expected_batch_size or len(morphology_ids) != expected_batch_size:
        raise RuntimeError(
            "dataset_index and morphology_index must contain one scalar per sample"
        )
    for dataset_id, morphology_id in zip(dataset_ids, morphology_ids):
        expected = EXPECTED_MORPHOLOGIES.get(dataset_id)
        if expected is None:
            raise RuntimeError(f"unexpected dataset_index {dataset_id}")
        if morphology_id != expected:
            raise RuntimeError(
                f"dataset_index {dataset_id} emitted morphology {morphology_id}, "
                f"expected {expected}"
            )

    return {
        "dataset_indices": dataset_ids,
        "morphology_indices": morphology_ids,
        "shapes": {
            name: tensor_shape(batch[name])
            for name in ("rgb", "actions", "mask", "dataset_index", "morphology_index")
        },
    }


def update_signature(digest: Any, value: Any) -> None:
    """Hash a bounded, deterministic representation of a nested batch."""
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous().reshape(-1)
        digest.update(f"tensor:{value.dtype}:{tuple(value.shape)}:".encode())
        if tensor.numel() > 4096:
            indices = torch.linspace(
                0, tensor.numel() - 1, steps=4096, dtype=torch.long
            )
            tensor = tensor.index_select(0, indices).contiguous()
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, dict):
        for key in sorted(value):
            digest.update(f"key:{key}:".encode())
            update_signature(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"sequence:{len(value)}:".encode())
        for item in value:
            update_signature(digest, item)
        return
    digest.update(repr(value).encode("utf-8", errors="backslashreplace"))


def batch_signature(batch: Any) -> str:
    digest = hashlib.sha256()
    update_signature(digest, batch)
    return digest.hexdigest()


def close_loader(loader: Any, iterator: Any) -> None:
    """Best-effort worker cleanup before constructing the restored loader."""
    seen: set[int] = set()
    for candidate in (iterator, getattr(loader, "_iterator", None)):
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        shutdown = getattr(candidate, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()


def run_smoke(
    data_root: Path,
    state_path: Path,
    min_batches: int,
    max_batches: int,
) -> dict[str, Any]:
    from hydra.utils import instantiate

    data_root = data_root.expanduser().resolve(strict=True)
    evidence: dict[str, Any] = {}
    for relative in REQUIRED_EVIDENCE:
        path = data_root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"required production evidence is missing: {path}")
        evidence[relative] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    os.environ["LACWM_DATA"] = str(data_root)
    cfg = compose_config()
    loader = instantiate(cfg.data_loader)
    dataset = loader.dataset
    source_names = tuple(dataset.datasets.keys())
    source_lengths = tuple(int(len(child)) for child in dataset.datasets.values())
    if source_names != EXPECTED_SOURCES:
        raise RuntimeError(
            f"production source order mismatch: {source_names} != {EXPECTED_SOURCES}"
        )
    if source_lengths != EXPECTED_LENGTHS:
        raise RuntimeError(
            f"production source lengths mismatch: {source_lengths} != {EXPECTED_LENGTHS}"
        )

    batch_size = int(cfg.data_loader.batch_size)
    if batch_size != 4:
        raise RuntimeError(f"production loader batch_size changed unexpectedly: {batch_size}")

    observed_counts = {name: 0 for name in EXPECTED_SOURCES}
    records: list[dict[str, Any]] = []
    mixed_batches = 0
    iterator = iter(loader)
    for batch_number in range(1, max_batches + 1):
        batch = next(iterator)
        record = validate_batch(batch, batch_size)
        source_ids = record["dataset_indices"]
        for source_id in source_ids:
            observed_counts[EXPECTED_SOURCES[source_id]] += 1
        if len(set(source_ids)) > 1:
            mixed_batches += 1
        record["batch"] = batch_number
        records.append(record)
        del batch

        if (
            batch_number >= min_batches
            and all(count > 0 for count in observed_counts.values())
            and mixed_batches > 0
        ):
            break
    else:
        raise RuntimeError(
            f"did not observe every source in {max_batches} production batches: "
            f"{observed_counts}"
        )

    loader_state = loader.state_dict()
    atomic_torch_save(state_path, loader_state)
    state_sha256 = sha256_file(state_path)

    reference_batch = next(iterator)
    reference_record = validate_batch(reference_batch, batch_size)
    reference_signature = batch_signature(reference_batch)
    del reference_batch
    close_loader(loader, iterator)
    del iterator, loader, dataset

    restored_loader = instantiate(cfg.data_loader)
    restored_loader.load_state_dict(load_torch_state(state_path))
    restored_iterator = iter(restored_loader)
    restored_batch = next(restored_iterator)
    restored_record = validate_batch(restored_batch, batch_size)
    restored_signature = batch_signature(restored_batch)
    del restored_batch
    close_loader(restored_loader, restored_iterator)

    if restored_signature != reference_signature:
        raise RuntimeError(
            "StatefulDataLoader continuation changed after state_dict restore: "
            f"{restored_signature} != {reference_signature}"
        )

    return {
        "config": {
            "config_name": "train",
            "experiment": EXPERIMENT,
            "loader_class": str(cfg.data_loader._target_),
            "batch_size": batch_size,
            "num_workers": int(cfg.data_loader.num_workers),
            "prefetch_factor": int(cfg.data_loader.prefetch_factor),
            "persistent_workers": bool(cfg.data_loader.persistent_workers),
        },
        "data": {
            "root": str(data_root),
            "evidence": evidence,
            "source_order": list(source_names),
            "source_lengths": list(source_lengths),
            "total_episodes": sum(source_lengths),
        },
        "mix": {
            "batches_checked": len(records),
            "samples_checked": sum(observed_counts.values()),
            "mixed_batches": mixed_batches,
            "observed_source_counts": observed_counts,
            "batches": records,
        },
        "resume": {
            "state_path": str(state_path.expanduser().absolute()),
            "state_size": state_path.stat().st_size,
            "state_sha256": state_sha256,
            "reference_batch": reference_record,
            "restored_batch": restored_record,
            "reference_signature": reference_signature,
            "restored_signature": restored_signature,
            "exact_continuation": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--state-output",
        type=Path,
        help="loader-state artifact (default: REPORT.loader_state.pt)",
    )
    parser.add_argument("--min-batches", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=64)
    args = parser.parse_args()
    if args.min_batches <= 0 or args.max_batches < args.min_batches:
        parser.error("require 0 < --min-batches <= --max-batches")
    if args.max_batches > 256:
        parser.error("--max-batches must be <= 256 for a bounded smoke test")
    if args.state_output is None:
        args.state_output = args.report.with_name(
            args.report.name + ".loader_state.pt"
        )
    return args


def main() -> int:
    args = parse_args()
    started = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "lacwm_real_mixed_stateful_dataloader_smoke",
        "status": "running",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_at_utc": started.isoformat(),
        "repo_root": str(REPO_ROOT),
        "git_commit": None,
        "git_status": None,
        "requested_data_root": str(args.data_root),
    }
    return_code = 1
    try:
        payload["git_commit"] = git_output("rev-parse", "HEAD")
        payload["git_status"] = git_output(
            "status", "--porcelain", "--untracked-files=all", "--", "."
        )
        if payload["git_status"]:
            raise RuntimeError("mixed-loader smoke requires a clean commit-bound repo")
        payload["validation"] = run_smoke(
            args.data_root,
            args.state_output,
            args.min_batches,
            args.max_batches,
        )
        payload["status"] = "passed"
        return_code = 0
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        print(payload["traceback"], file=sys.stderr)
    finally:
        finished = datetime.now(timezone.utc)
        payload["finished_at_utc"] = finished.isoformat()
        payload["elapsed_seconds"] = (finished - started).total_seconds()
        atomic_json(args.report, payload)
        print(f"mixed-loader smoke report: {args.report} ({payload['status']})")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
