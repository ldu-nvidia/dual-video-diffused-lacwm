import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.extract_vjepa2_targets import (
    ExtractionError,
    _atomic_torch_save,
    _pca_runtime_metadata,
    _publish_initialized_cache_directory,
)


def _metadata():
    return {
        "complete": False,
        "written_rows": 0,
        "target_file": "targets.fp16.npy",
        "target_shape": [2, 3, 1, 2, 2],
        "rgb_file": "rgb.fp16.npy",
        "rgb_shape": [2, 1, 3, 2, 4],
        "actions_file": "actions.float32.npy",
        "actions_shape": [2, 1, 1, 5],
    }


def test_retry_ignores_abandoned_prepublication_directory(tmp_path):
    cache = tmp_path / "train"
    # Simulate a process dying after it created one staging array but before
    # metadata/publication.  The untrusted staging bytes remain auditable.
    abandoned = tmp_path / ".train.initializing.crashed"
    abandoned.mkdir()
    np.lib.format.open_memmap(
        abandoned / "targets.fp16.npy",
        mode="w+",
        dtype=np.float16,
        shape=(2, 3, 1, 2, 2),
    ).flush()

    metadata = _metadata()
    _publish_initialized_cache_directory(cache, metadata)

    assert abandoned.is_dir()
    with (cache / "metadata.json").open(encoding="utf-8") as handle:
        assert json.load(handle) == metadata
    target = np.load(cache / "targets.fp16.npy", mmap_mode="r")
    rgb = np.load(cache / "rgb.fp16.npy", mmap_mode="r")
    actions = np.load(cache / "actions.float32.npy", mmap_mode="r")
    assert target.shape == tuple(metadata["target_shape"])
    assert target.dtype == np.float16
    assert rgb.shape == tuple(metadata["rgb_shape"])
    assert rgb.dtype == np.float16
    assert actions.shape == tuple(metadata["actions_shape"])
    assert actions.dtype == np.float32


def test_initialization_never_replaces_a_published_cache(tmp_path):
    cache = tmp_path / "train"
    _publish_initialized_cache_directory(cache, _metadata())

    with pytest.raises(ExtractionError, match="existing cache directory"):
        _publish_initialized_cache_directory(cache, _metadata())


def test_pca_runtime_metadata_is_weights_only_safe(tmp_path):
    metadata = _pca_runtime_metadata()
    assert type(metadata["torch_version"]) is str

    artifact = tmp_path / "pca-metadata.pt"
    _atomic_torch_save(artifact, metadata)

    loaded = torch.load(artifact, map_location="cpu", weights_only=True)
    assert loaded == metadata
