import json

import numpy as np
import torch

from robot_wm.datasets.abc.fixed_clip_dataset import ABCFixedClipDataset
from robot_wm.modeling.dual_diffusion.vjepa2_target import (
    VJEPA2_1_MODEL_NAME,
    VJEPA2_1_RELEASE_COMMIT,
    sha256_file,
)


def _write_array(path, shape, dtype, fill):
    array = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )
    array[...] = fill
    array.flush()
    del array


def test_fixed_dataset_reads_only_hashed_rgb_action_and_target_arrays(tmp_path):
    manifest = tmp_path / "test.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "clip-0",
                # This deliberately does not exist: post-extraction training
                # must never fall back to mutable source episode assets.
                "episode_dir": str(tmp_path / "missing-raw-episode"),
                "start": 17,
                "auxiliary_index": 0,
            }
        )
        + "\n"
    )
    target_path = tmp_path / "targets.fp16.npy"
    rgb_path = tmp_path / "rgb.fp16.npy"
    actions_path = tmp_path / "actions.float32.npy"
    _write_array(target_path, (1, 64, 4, 24, 120), np.float16, 0.5)
    _write_array(rgb_path, (1, 13, 3, 180, 960), np.float16, -0.25)
    _write_array(actions_path, (1, 13, 5, 23), np.float32, 0.75)

    metadata = {
        "format_version": 1,
        "artifact_type": "vjepa2.1-wan-grid-cache",
        "complete": True,
        "model_name": VJEPA2_1_MODEL_NAME,
        "source_commit": VJEPA2_1_RELEASE_COMMIT,
        "target_file": target_path.name,
        "target_dtype": "float16",
        "target_shape": [1, 64, 4, 24, 120],
        "target_sha256": sha256_file(target_path),
        "rgb_file": rgb_path.name,
        "rgb_dtype": "float16",
        "rgb_shape": [1, 13, 3, 180, 960],
        "rgb_sha256": sha256_file(rgb_path),
        "actions_file": actions_path.name,
        "actions_dtype": "float32",
        "actions_shape": [1, 13, 5, 23],
        "actions_sha256": sha256_file(actions_path),
        "sample_size": 13,
        "chunk_size": 5,
        "action_span": 65,
        "frame_offsets": list(range(0, 65, 5)),
        "camera_order": ["top", "left_wrist", "right_wrist"],
        "pca_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "train_manifest_sha256": "3" * 64,
        "cache_id": "4" * 64,
        "clip_manifest_sha256": sha256_file(manifest),
        "clip_count": 1,
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    dataset = ABCFixedClipDataset(
        clip_manifest=str(manifest),
        cache_metadata=str(metadata_path),
        transform=None,
        infinite=False,
    )
    sample = dataset._get_sample(0)

    assert sample["rgb"].shape == (13, 3, 180, 960)
    assert sample["rgb"].dtype == torch.float32
    assert torch.all(sample["rgb"] == -0.25)
    assert sample["actions"].shape == (13, 5, 23)
    assert sample["actions"].dtype == torch.float32
    assert torch.all(sample["actions"] == 0.75)
    assert sample["auxiliary_target"].shape == (64, 4, 24, 120)
    assert sample["auxiliary_target"].dtype == torch.float16
    assert sample["mask"].tolist() == [True] * 13
    assert sample["clip_index"].item() == 0
