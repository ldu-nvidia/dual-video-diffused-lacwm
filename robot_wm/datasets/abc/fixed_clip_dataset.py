"""Deterministic ABC clips paired with offline V-JEPA target tensors."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from robot_wm.datasets.base import Dataset
from robot_wm.modeling.dual_diffusion.vjepa2_target import (
    VJEPA2_1_MODEL_NAME,
    VJEPA2_1_RELEASE_COMMIT,
    sha256_file,
)

logger = logging.getLogger(__name__)


class ABCFixedClipDataset(Dataset):
    """Replay an immutable clip manifest and attach its cached auxiliary target.

    The JSONL manifest contains one row per sample:

    ``{"clip_id": ..., "episode_dir": ..., "start": ..., "auxiliary_index": ...}``

    The cache metadata points to one NumPy ``.npy`` tensor with shape
    ``[N,Caux,4,24,120]``.  The array is memory-mapped independently inside
    each loader process, so workers do not copy the complete cache.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        clip_manifest: str,
        cache_metadata: str,
        seed: int = 0,
        infinite: bool = True,
        transform: Optional[Any] = None,
        validate_target_samples: int = 2,
        expected_cache_id: Optional[str] = None,
        expected_pca_sha256: Optional[str] = None,
        expected_checkpoint_sha256: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(seed=seed, infinite=infinite, transform=transform)
        self.clip_manifest = str(Path(clip_manifest).resolve())
        self.cache_metadata = str(Path(cache_metadata).resolve())
        with Path(self.clip_manifest).open() as handle:
            self.clips = [json.loads(line) for line in handle if line.strip()]
        if not self.clips:
            raise ValueError("fixed ABC clip manifest is empty")
        with Path(self.cache_metadata).open() as handle:
            self.metadata = json.load(handle)
        self.expected_cache_id = expected_cache_id
        self.expected_pca_sha256 = expected_pca_sha256
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256
        self._validate_metadata()
        self.target_path = self._resolve_cache_file("target_file")
        self.rgb_path = self._resolve_cache_file("rgb_file")
        self.actions_path = self._resolve_cache_file("actions_file")
        self._targets = None
        self._rgbs = None
        self._actions = None
        self._target_shape = tuple(int(v) for v in self.metadata["target_shape"])
        self._rgb_shape = tuple(int(v) for v in self.metadata["rgb_shape"])
        self._actions_shape = tuple(
            int(v) for v in self.metadata["actions_shape"]
        )
        self._validate_cached_arrays(validate_target_samples)
        self.ee_action_dim = 14
        logger.info(
            "ABCFixedClipDataset: %d immutable clips, target=%s shape=%s",
            len(self.clips),
            self.target_path,
            self._target_shape,
        )

    @property
    def name(self) -> str:
        # Preserve the production morphology mapping.
        return "ABCDataset"

    def _get_length(self) -> int:
        return len(self.clips)

    def __len__(self) -> int:
        return self._get_length()

    def __getstate__(self):
        state = super().__getstate__()
        state["_targets"] = None
        state["_rgbs"] = None
        state["_actions"] = None
        return state

    def _resolve_cache_file(self, metadata_key: str) -> str:
        value = self.metadata.get(metadata_key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"cache metadata lacks {metadata_key}")
        path = Path(value)
        if not path.is_absolute():
            path = Path(self.cache_metadata).parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"cached array is missing: {path}")
        return str(path)

    def _validate_metadata(self) -> None:
        def require_sha256(key: str) -> str:
            value = self.metadata.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"cache metadata {key} is not a full SHA-256")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(
                    f"cache metadata {key} is not hexadecimal"
                ) from exc
            return value.lower()

        if int(self.metadata.get("format_version", -1)) != self.FORMAT_VERSION:
            raise ValueError("unsupported V-JEPA target cache format")
        if self.metadata.get("artifact_type") != "vjepa2.1-wan-grid-cache":
            raise ValueError("unexpected V-JEPA cache artifact type")
        if not bool(self.metadata.get("complete", False)):
            raise RuntimeError("V-JEPA target cache is not marked complete")
        if self.metadata.get("model_name") != VJEPA2_1_MODEL_NAME:
            raise ValueError("V-JEPA cache model identity mismatch")
        if self.metadata.get("source_commit") != VJEPA2_1_RELEASE_COMMIT:
            raise ValueError("V-JEPA cache source commit is not pinned")
        if self.metadata.get("target_dtype") != "float16":
            raise ValueError("V-JEPA cache target dtype metadata is not float16")
        if (
            int(self.metadata.get("sample_size", -1)) != 13
            or int(self.metadata.get("chunk_size", -1)) != 5
            or int(self.metadata.get("action_span", -1)) != 65
            or self.metadata.get("frame_offsets") != list(range(0, 65, 5))
            or self.metadata.get("camera_order")
            != ["top", "left_wrist", "right_wrist"]
        ):
            raise ValueError("V-JEPA cache clip/view alignment metadata differs")
        pca_sha256 = require_sha256("pca_sha256")
        checkpoint_sha256 = require_sha256("checkpoint_sha256")
        require_sha256("target_sha256")
        require_sha256("rgb_sha256")
        require_sha256("actions_sha256")
        require_sha256("train_manifest_sha256")
        cache_id = require_sha256("cache_id")
        if self.expected_cache_id is not None and cache_id != (
            self.expected_cache_id.lower()
        ):
            raise RuntimeError("V-JEPA cache ID differs from the pinned study")
        if self.expected_pca_sha256 is not None and pca_sha256 != (
            self.expected_pca_sha256.lower()
        ):
            raise RuntimeError("V-JEPA PCA digest differs from the pinned study")
        if self.expected_checkpoint_sha256 is not None and checkpoint_sha256 != (
            self.expected_checkpoint_sha256.lower()
        ):
            raise RuntimeError(
                "V-JEPA checkpoint digest differs from the pinned study"
            )
        manifest_digest = sha256_file(self.clip_manifest)
        if self.metadata.get("clip_manifest_sha256") != manifest_digest:
            raise RuntimeError("clip manifest does not match target cache metadata")
        target_shape = self.metadata.get("target_shape")
        if (
            not isinstance(target_shape, list)
            or len(target_shape) != 5
            or int(target_shape[0]) != len(self.clips)
            or int(self.metadata.get("clip_count", -1)) != len(self.clips)
        ):
            raise ValueError("target_shape does not match the clip manifest")
        if tuple(int(v) for v in target_shape[1:]) != (64, 4, 24, 120):
            raise ValueError(
                "cached V-JEPA targets must have shape [N,64,4,24,120]"
            )
        if self.metadata.get("rgb_dtype") != "float16":
            raise ValueError("cached RGB inputs must use float16")
        if self.metadata.get("actions_dtype") != "float32":
            raise ValueError("cached actions must use float32")
        if tuple(int(v) for v in self.metadata.get("rgb_shape", [])) != (
            len(self.clips),
            13,
            3,
            180,
            960,
        ):
            raise ValueError(
                "cached RGB inputs must have shape [N,13,3,180,960]"
            )
        if tuple(int(v) for v in self.metadata.get("actions_shape", [])) != (
            len(self.clips),
            13,
            5,
            23,
        ):
            raise ValueError(
                "cached actions must have shape [N,13,5,23]"
            )
        for index, descriptor in enumerate(self.clips):
            required = {"clip_id", "episode_dir", "start", "auxiliary_index"}
            if not required.issubset(descriptor):
                raise ValueError(
                    f"clip row {index} lacks {sorted(required - descriptor.keys())}"
                )
            if int(descriptor["auxiliary_index"]) != index:
                raise ValueError(
                    "clip manifest auxiliary indexes must be dense and ordered"
                )

    def _open_targets(self):
        if self._targets is None:
            self._targets = np.load(
                self.target_path, mmap_mode="r", allow_pickle=False
            )
            if tuple(self._targets.shape) != self._target_shape:
                raise RuntimeError(
                    "target array shape changed after dataset construction"
                )
            if self._targets.dtype != np.float16:
                raise TypeError("cached V-JEPA targets must use float16")
        return self._targets

    def _open_rgbs(self):
        if self._rgbs is None:
            self._rgbs = np.load(
                self.rgb_path, mmap_mode="r", allow_pickle=False
            )
            if tuple(self._rgbs.shape) != self._rgb_shape:
                raise RuntimeError(
                    "RGB cache shape changed after dataset construction"
                )
            if self._rgbs.dtype != np.float16:
                raise TypeError("cached RGB inputs must use float16")
        return self._rgbs

    def _open_actions(self):
        if self._actions is None:
            self._actions = np.load(
                self.actions_path, mmap_mode="r", allow_pickle=False
            )
            if tuple(self._actions.shape) != self._actions_shape:
                raise RuntimeError(
                    "action cache shape changed after dataset construction"
                )
            if self._actions.dtype != np.float32:
                raise TypeError("cached actions must use float32")
        return self._actions

    def _validate_cached_arrays(self, count: int) -> None:
        targets = self._open_targets()
        rgbs = self._open_rgbs()
        actions = self._open_actions()
        indexes = list(range(min(max(count, 0), len(self.clips))))
        if len(self.clips) > 1 and indexes:
            indexes.append(len(self.clips) - 1)
        for index in sorted(set(indexes)):
            if not np.isfinite(np.asarray(targets[index])).all():
                raise FloatingPointError(
                    f"cached V-JEPA target {index} is non-finite"
                )
            rgb = np.asarray(rgbs[index])
            if (
                not np.isfinite(rgb).all()
                or rgb.min() < -1.0
                or rgb.max() > 1.0
            ):
                raise FloatingPointError(
                    f"cached RGB input {index} is invalid"
                )
            if not np.isfinite(np.asarray(actions[index])).all():
                raise FloatingPointError(
                    f"cached actions {index} are non-finite"
                )
        self._targets = None
        self._rgbs = None
        self._actions = None

    def _get_sample(self, index: int) -> dict[str, Any]:
        descriptor = self.clips[int(index)]
        # RGB and actions are frozen alongside the JEPA target; training no
        # longer reads mutable source MP4/state files after extraction.
        rgb = torch.from_numpy(
            np.array(self._open_rgbs()[int(index)], copy=True)
        ).float()
        actions = torch.from_numpy(
            np.array(self._open_actions()[int(index)], copy=True)
        )
        target = np.array(
            self._open_targets()[int(descriptor["auxiliary_index"])],
            copy=True,
        )
        return {
            "rgb": rgb,
            "actions": actions,
            "mask": torch.ones(13, dtype=torch.bool),
            "auxiliary_target": torch.from_numpy(target),
            "clip_index": torch.tensor(int(index), dtype=torch.long),
        }
