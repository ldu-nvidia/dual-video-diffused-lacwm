#!/usr/bin/env python3
"""Read an immutable causal V-JEPA cache through its recorded producer.

The causal-cache validator intentionally binds an artifact to the exact clean
checkout that produced it.  A descendant research commit must not alter that
validator or pretend that the old cache was produced by its active source.
Instead, this module launches a fresh Python subprocess from the producer
checkout recorded inside ``metadata.json``.  The producer implementation
validates the manifest, PCA, metadata, and complete target-content hash.  Only
after that attestation succeeds does the descendant process open the target
array read-only and pair it with the byte-identical frozen base-DROID reader.

Protected-test caches are deliberately unsupported.  A selected experiment
must build its separately authorized test artifact rather than extending this
bridge.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.datasets.droid.video_latent_forcing import (  # noqa: E402
    DroidVideoLatentForcingDataset,
    read_clip_manifest,
    sha256_file,
)
from tools import video_latent_forcing_poc as vlf  # noqa: E402


BRIDGE_SCHEMA = "causal-vjepa2-producer-attested-cache-bridge-v1"
ATTESTATION_SCHEMA = "causal-vjepa2-producer-cache-attestation-v1"
FROZEN_CACHE_PRODUCER_COMMIT = "c11487f6e83908687f27026ce2ac2e7d8d41461c"
ALLOWED_SPLITS = ("train", "val")
TARGET_SHAPE = (48, 8, 8, 14)


class ProducerCacheBridgeError(RuntimeError):
    """A recorded producer or its attested cache failed closed."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProducerCacheBridgeError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ProducerCacheBridgeError(f"{label} must be a JSON mapping")
    return value


def _exact_file_record(path: Path, value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProducerCacheBridgeError(f"{label} record is missing")
    expected = {name: value.get(name) for name in ("path", "sha256", "bytes")}
    actual = vlf.file_record(path)
    if expected != actual:
        raise ProducerCacheBridgeError(f"{label} differs from its recorded bytes")
    return actual


def _git_output(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ProducerCacheBridgeError(
            f"cannot validate recorded producer Git checkout: {repo}"
        ) from exc


def _validate_recorded_producer(
    metadata: Mapping[str, Any], *, expected_commit: str
) -> dict[str, Any]:
    implementation = metadata.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ProducerCacheBridgeError("cache lacks implementation provenance")
    commit = implementation.get("repo_commit")
    if commit != expected_commit or commit != FROZEN_CACHE_PRODUCER_COMMIT:
        raise ProducerCacheBridgeError(
            f"cache producer commit differs from the frozen producer: {commit}"
        )
    root_value = implementation.get("repo_root")
    if not isinstance(root_value, str):
        raise ProducerCacheBridgeError("cache producer root is missing")
    root = Path(root_value).expanduser().resolve()
    if root.is_symlink() or not (root / ".git").is_dir():
        raise ProducerCacheBridgeError("recorded producer is not a concrete Git checkout")
    if _git_output(root, "rev-parse", "HEAD") != expected_commit:
        raise ProducerCacheBridgeError("recorded producer checkout moved to another commit")
    if _git_output(root, "status", "--porcelain", "--untracked-files=all"):
        raise ProducerCacheBridgeError("recorded producer checkout is dirty")

    dataset_source = root / "robot_wm" / "datasets" / "droid" / "causal_vjepa2.py"
    builder_source = root / "tools" / "build_causal_vjepa2_droid.py"
    base_source = root / "robot_wm" / "datasets" / "droid" / "video_latent_forcing.py"
    dataset_record = _exact_file_record(
        dataset_source, implementation.get("dataset_source"), label="producer dataset"
    )
    builder_record = _exact_file_record(
        builder_source, implementation.get("builder_source"), label="producer builder"
    )
    current_base_value = inspect.getsourcefile(DroidVideoLatentForcingDataset)
    if current_base_value is None:
        raise ProducerCacheBridgeError("cannot locate active base-DROID reader")
    current_base = Path(current_base_value).resolve()
    if sha256_file(current_base) != sha256_file(base_source):
        raise ProducerCacheBridgeError(
            "active base-DROID reader differs from the producer checkout"
        )
    return {
        "repo_commit": expected_commit,
        "repo_root": str(root),
        "dataset_source": dataset_record,
        "builder_source": builder_record,
        "producer_base_dataset": vlf.file_record(base_source),
        "active_base_dataset": vlf.file_record(current_base),
    }


_PRODUCER_VALIDATION_CODE = r"""
import hashlib
import json
import sys
from pathlib import Path

from robot_wm.datasets.droid.causal_vjepa2 import validate_causal_cache

manifest, metadata_path, split = sys.argv[1:]
metadata, target = validate_causal_cache(
    manifest_path=manifest,
    cache_metadata_path=metadata_path,
    expected_split=split,
    verify_target_hash=True,
)
payload = {
    "schema": "causal-vjepa2-producer-cache-attestation-v1",
    "producer_module": str(Path(sys.modules[
        "robot_wm.datasets.droid.causal_vjepa2"
    ].__file__).resolve()),
    "metadata": metadata,
    "target_path": str(Path(target.filename).resolve()),
    "target_shape": list(target.shape),
    "target_dtype": str(target.dtype),
    "target_writeable": bool(target.flags.writeable),
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""


def _run_producer_attestation(
    *,
    manifest: Path,
    metadata_path: Path,
    split: str,
    producer: Mapping[str, Any],
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(producer["repo_root"])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _PRODUCER_VALIDATION_CODE,
                str(manifest),
                str(metadata_path),
                split,
            ],
            cwd=str(producer["repo_root"]),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=1_800,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "producer validator failed").strip()
        raise ProducerCacheBridgeError(
            f"recorded producer rejected the cache: {detail[-1000:]}"
        ) from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProducerCacheBridgeError("recorded producer attestation did not complete") from exc
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ProducerCacheBridgeError("producer attestation emitted unexpected output")
    try:
        attestation = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ProducerCacheBridgeError("producer attestation is not JSON") from exc
    if not isinstance(attestation, dict):
        raise ProducerCacheBridgeError("producer attestation is not a mapping")
    expected_module = Path(producer["dataset_source"]["path"]).resolve()
    if (
        attestation.get("schema") != ATTESTATION_SCHEMA
        or Path(str(attestation.get("producer_module", ""))).resolve()
        != expected_module
        or attestation.get("target_writeable") is not False
    ):
        raise ProducerCacheBridgeError("producer attestation identity is malformed")
    return attestation


class ProducerAttestedCausalDataset(Dataset):
    """Base DROID clips paired with a producer-validated read-only target map."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        semantic_cache_root: str | Path,
        *,
        expected_producer_commit: str = FROZEN_CACHE_PRODUCER_COMMIT,
    ) -> None:
        manifest = Path(manifest_path).expanduser().resolve()
        rows = read_clip_manifest(manifest)
        split = str(rows[0]["split"])
        if split not in ALLOWED_SPLITS:
            raise ProducerCacheBridgeError(
                "producer cache bridge permits only train and validation"
            )
        metadata_path = (
            Path(semantic_cache_root).expanduser().resolve()
            / split
            / "metadata.json"
        )
        metadata = _read_json(metadata_path, label="causal cache metadata")
        producer = _validate_recorded_producer(
            metadata, expected_commit=expected_producer_commit
        )
        attestation = _run_producer_attestation(
            manifest=manifest,
            metadata_path=metadata_path,
            split=split,
            producer=producer,
        )
        attested_metadata = attestation.get("metadata")
        if not isinstance(attested_metadata, dict) or attested_metadata != metadata:
            raise ProducerCacheBridgeError(
                "producer-attested metadata differs from the parent process"
            )
        target_path = Path(str(attestation.get("target_path", ""))).resolve()
        if target_path.parent != metadata_path.parent:
            raise ProducerCacheBridgeError("attested target escaped its cache directory")
        target_digest = sha256_file(target_path)
        if target_digest != metadata.get("target_sha256"):
            raise ProducerCacheBridgeError("target changed after producer attestation")
        try:
            targets = np.load(target_path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise ProducerCacheBridgeError("cannot open attested target array") from exc
        expected_shape = (len(rows), *TARGET_SHAPE)
        if (
            tuple(targets.shape) != expected_shape
            or targets.dtype != np.float16
            or targets.flags.writeable
            or attestation.get("target_shape") != list(expected_shape)
            or attestation.get("target_dtype") != "float16"
        ):
            raise ProducerCacheBridgeError("attested target shape/dtype/access differs")
        self.base = DroidVideoLatentForcingDataset(manifest, data_root)
        if len(self.base) != len(rows):
            raise ProducerCacheBridgeError("base DROID and target population differ")
        self.cache_metadata = metadata
        self._targets = targets
        target_record = {
            "path": str(target_path),
            "sha256": target_digest,
            "bytes": target_path.stat().st_size,
        }
        self.producer_attestation = {
            "schema": BRIDGE_SCHEMA,
            "producer": producer,
            "manifest": vlf.file_record(manifest),
            "cache_metadata": vlf.file_record(metadata_path),
            "target": target_record,
            "split": split,
            "clips": len(rows),
            "protected_test_accessed": False,
            "producer_attestation_sha256": _canonical_sha256(attestation),
        }

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        auxiliary = torch.from_numpy(np.array(self._targets[index], copy=True))
        if (
            auxiliary.dtype != torch.float16
            or tuple(auxiliary.shape) != TARGET_SHAPE
            or not bool(torch.isfinite(auxiliary).all())
        ):
            raise ProducerCacheBridgeError("attested target row is malformed")
        sample["auxiliary_target"] = auxiliary
        sample["auxiliary_cache_id"] = str(self.cache_metadata["cache_id"])
        return sample

    def validated_cache_metadata(self) -> dict[str, Any]:
        return dict(self.cache_metadata)


def construct_producer_attested_dataset(
    manifest_path: str | Path,
    data_root: str | Path,
    semantic_cache_root: str | Path,
) -> ProducerAttestedCausalDataset:
    """Construct the only allowed descendant reader for the frozen c114 cache."""
    dataset = ProducerAttestedCausalDataset(
        manifest_path,
        data_root,
        semantic_cache_root,
    )
    if len(dataset) < 1:
        raise ProducerCacheBridgeError("producer-attested dataset is empty")
    return dataset
