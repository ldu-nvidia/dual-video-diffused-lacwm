"""Pinned ABC RGB/action clips for the video residual-anchor screen.

Only RGB and actions are opened.  The historical cache's auxiliary feature
array is deliberately unreachable from this dataset implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_wm.datasets.base import Dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ABCVideoResidualAnchorDataset(Dataset):
    """Replay immutable RGB/actions while preserving manifest clip order."""

    def __init__(
        self,
        *,
        clip_manifest: str,
        cache_metadata: str,
        expected_split: str,
        expected_clip_count: int,
        expected_manifest_sha256: str,
        expected_rgb_sha256: str,
        expected_actions_sha256: str,
        seed: int = 0,
        infinite: bool = True,
        transform: Any = None,
        **_kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, infinite=infinite, transform=transform)
        self.clip_manifest = str(Path(clip_manifest).resolve(strict=True))
        self.cache_metadata = str(Path(cache_metadata).resolve(strict=True))
        self.expected_split = str(expected_split)
        self.expected_clip_count = int(expected_clip_count)
        self.expected_manifest_sha256 = str(expected_manifest_sha256)
        self.expected_rgb_sha256 = str(expected_rgb_sha256)
        self.expected_actions_sha256 = str(expected_actions_sha256)
        if self.expected_split not in {"train", "val"}:
            raise ValueError("residual-anchor dataset permits only train or val")
        with Path(self.clip_manifest).open(encoding="utf-8") as handle:
            self.clips = [json.loads(line) for line in handle if line.strip()]
        with Path(self.cache_metadata).open(encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self._validate_metadata()
        metadata_dir = Path(self.cache_metadata).parent
        self.rgb_path = self._resolve_data_file(metadata_dir, "rgb_file")
        self.actions_path = self._resolve_data_file(metadata_dir, "actions_file")
        self._rgb_shape = (self.expected_clip_count, 13, 3, 180, 960)
        self._actions_shape = (self.expected_clip_count, 13, 5, 23)
        self._rgbs: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._validate_arrays()
        self.ee_action_dim = 14

    @property
    def name(self) -> str:
        return "ABCDataset"

    @property
    def auxiliary_target_array_opened(self) -> bool:
        return False

    def _get_length(self) -> int:
        return len(self.clips)

    def __len__(self) -> int:
        return len(self.clips)

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["_rgbs"] = None
        state["_actions"] = None
        return state

    def _resolve_data_file(self, parent: Path, key: str) -> str:
        value = self.metadata.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"cache metadata lacks {key}")
        path = Path(value)
        if not path.is_absolute():
            path = parent / path
        path = path.resolve(strict=True)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{key} must resolve to a regular non-symlink file")
        return str(path)

    def _validate_metadata(self) -> None:
        digests = (
            self.expected_manifest_sha256,
            self.expected_rgb_sha256,
            self.expected_actions_sha256,
        )
        if any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in digests
        ):
            raise ValueError("expected artifact digests must be lowercase SHA-256")
        if (
            self.metadata.get("complete") is not True
            or self.metadata.get("split") != self.expected_split
            or int(self.metadata.get("clip_count", -1)) != self.expected_clip_count
            or len(self.clips) != self.expected_clip_count
            or _sha256(Path(self.clip_manifest)) != self.expected_manifest_sha256
            or self.metadata.get("clip_manifest_sha256")
            != self.expected_manifest_sha256
            or self.metadata.get("rgb_sha256") != self.expected_rgb_sha256
            or self.metadata.get("actions_sha256") != self.expected_actions_sha256
            or self.metadata.get("rgb_dtype") != "float16"
            or self.metadata.get("actions_dtype") != "float32"
            or self.metadata.get("rgb_shape")
            != [self.expected_clip_count, 13, 3, 180, 960]
            or self.metadata.get("actions_shape")
            != [self.expected_clip_count, 13, 5, 23]
            or int(self.metadata.get("sample_size", -1)) != 13
            or int(self.metadata.get("chunk_size", -1)) != 5
        ):
            raise RuntimeError("immutable ABC RGB/action metadata differs")
        for index, descriptor in enumerate(self.clips):
            if (
                not isinstance(descriptor, dict)
                or descriptor.get("split") != self.expected_split
                or int(descriptor.get("auxiliary_index", -1)) != index
                or not isinstance(descriptor.get("clip_id"), str)
            ):
                raise RuntimeError(f"ABC manifest row {index} differs")

    def _open_rgbs(self) -> np.ndarray:
        if self._rgbs is None:
            self._rgbs = np.load(self.rgb_path, mmap_mode="r", allow_pickle=False)
            if self._rgbs.shape != self._rgb_shape or self._rgbs.dtype != np.float16:
                raise RuntimeError("immutable RGB array changed")
        return self._rgbs

    def _open_actions(self) -> np.ndarray:
        if self._actions is None:
            self._actions = np.load(
                self.actions_path, mmap_mode="r", allow_pickle=False
            )
            if (
                self._actions.shape != self._actions_shape
                or self._actions.dtype != np.float32
            ):
                raise RuntimeError("immutable action array changed")
        return self._actions

    def _validate_arrays(self) -> None:
        rgbs = self._open_rgbs()
        actions = self._open_actions()
        for index in sorted({0, self.expected_clip_count - 1}):
            rgb = np.asarray(rgbs[index])
            action = np.asarray(actions[index])
            if (
                not np.isfinite(rgb).all()
                or float(rgb.min()) < -1.0
                or float(rgb.max()) > 1.0
                or not np.isfinite(action).all()
            ):
                raise FloatingPointError(f"invalid immutable sample {index}")
        self._rgbs = None
        self._actions = None

    def _get_sample(self, index: int) -> dict[str, torch.Tensor]:
        index = int(index)
        if not 0 <= index < len(self.clips):
            raise IndexError(index)
        return {
            "rgb": torch.from_numpy(
                np.array(self._open_rgbs()[index], copy=True)
            ).float(),
            "actions": torch.from_numpy(
                np.array(self._open_actions()[index], copy=True)
            ),
            "mask": torch.ones(13, dtype=torch.bool),
            "clip_index": torch.tensor(index, dtype=torch.long),
        }
