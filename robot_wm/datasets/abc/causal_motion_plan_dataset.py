"""Immutable RGB/action view with a deterministic train-only planner split."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from robot_wm.datasets.abc.fixed_rgb_action_dataset import (
    ABCFixedRGBActionDataset,
)
from robot_wm.modeling.dual_diffusion.causal_motion_plan import (
    PLANNER_SPLIT_ROLES,
    planner_partition_indexes,
)


class ABCCausalMotionPlanDataset(ABCFixedRGBActionDataset):
    """Expose planner-fit, train-calibration, or complete immutable ABC rows.

    The split is fixed by the source manifest's dense ``auxiliary_index``:
    remainder zero modulo eight is the 64-row train-only calibration set; the
    remaining 448 rows fit the planner.  Video continuations use ``all``.
    Validation data must also use ``all`` and is never consulted to decide
    whether planner/video training proceeds.
    """

    SPLIT_ROLES = PLANNER_SPLIT_ROLES

    def __init__(self, *, split_role: str = "all", **kwargs: Any) -> None:
        if split_role not in self.SPLIT_ROLES:
            raise ValueError(f"split_role must be one of {self.SPLIT_ROLES}")
        super().__init__(**kwargs)
        self.split_role = split_role
        self._source_indexes = planner_partition_indexes(len(self.clips), split_role)
        if not self._source_indexes:
            raise ValueError("causal motion-plan dataset split is empty")
        if len(self.clips) == 512:
            expected = {
                "planner_fit": 448,
                "planner_calibration": 64,
                "all": 512,
            }[split_role]
            if len(self._source_indexes) != expected:
                raise RuntimeError("train-only planner partition cardinality changed")

    def _get_length(self) -> int:
        return len(self._source_indexes)

    def _get_sample(self, index: int) -> dict[str, Any]:
        source_index = int(self._source_indexes[int(index)])
        rgb = torch.from_numpy(
            np.array(self._open_rgbs()[source_index], copy=True)
        ).float()
        actions = torch.from_numpy(
            np.array(self._open_actions()[source_index], copy=True)
        )
        return {
            "rgb": rgb,
            "actions": actions,
            "mask": torch.ones(13, dtype=torch.bool),
            "clip_index": torch.tensor(source_index, dtype=torch.long),
        }
