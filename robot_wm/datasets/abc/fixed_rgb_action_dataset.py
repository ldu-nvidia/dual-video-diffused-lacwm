"""Target-free view of the immutable ABC RGB/action cache."""

from __future__ import annotations

import numpy as np
import torch

from robot_wm.datasets.abc.fixed_clip_dataset import ABCFixedClipDataset


class ABCFixedRGBActionDataset(ABCFixedClipDataset):
    """Replay pinned ABC clips without exposing cached V-JEPA targets.

    The historical cache already pins exact RGB clips and action chunks.  A
    Frequency-Forcing target is computed online from those RGB values, so
    loading or returning the unrelated V-JEPA target would be both wasteful
    and an opportunity for accidental feature leakage.
    """

    def _validate_cached_arrays(self, count: int) -> None:
        """Validate only the RGB/action arrays used by this dataset view."""
        rgbs = self._open_rgbs()
        actions = self._open_actions()
        indexes = list(range(min(max(int(count), 0), len(self.clips))))
        if len(self.clips) > 1 and indexes:
            indexes.append(len(self.clips) - 1)
        for index in sorted(set(indexes)):
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
        # Do not retain worker-local memmaps across dataset serialization.
        self._rgbs = None
        self._actions = None

    def _get_sample(self, index: int):
        index = int(index)
        rgb = torch.from_numpy(
            np.array(self._open_rgbs()[index], copy=True)
        ).float()
        actions = torch.from_numpy(
            np.array(self._open_actions()[index], copy=True)
        )
        return {
            "rgb": rgb,
            "actions": actions,
            "mask": torch.ones(13, dtype=torch.bool),
            "clip_index": torch.tensor(index, dtype=torch.long),
        }
