"""Immutable ABC RGB/action clips with a fixed causal robot-flow condition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_wm.datasets.abc.video_residual_anchor_dataset import (
    ABCVideoResidualAnchorDataset,
)
from robot_wm.modeling.dual_diffusion.physics_flow import (
    FLOW_COMPONENTS,
    FUTURE_TRANSITIONS,
    LATENT_HEIGHT,
    VIEW_LATENT_WIDTH,
    pack_top_view_flow,
    validate_packed_flow,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_identity(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class ABCPhysicsFlowDataset(ABCVideoResidualAnchorDataset):
    """Replay aligned, pre-rendered flow without exposing a future target."""

    def __init__(
        self,
        *,
        flow_metadata: str,
        expected_flow_sha256: str,
        expected_renderer_analysis_identity_sha256: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.flow_metadata_path = str(Path(flow_metadata).resolve(strict=True))
        self.expected_flow_sha256 = str(expected_flow_sha256)
        self.expected_renderer_analysis_identity_sha256 = str(
            expected_renderer_analysis_identity_sha256
        )
        with Path(self.flow_metadata_path).open(encoding="utf-8") as handle:
            self.flow_metadata = json.load(handle)
        if not isinstance(self.flow_metadata, dict):
            raise RuntimeError("flow metadata root must be an object")
        # Compact transition caches avoid materializing the exact-zero history
        # tokens and two exact-zero wrist thirds on disk.  Packing is an exact,
        # tested operation performed only after the registered row is loaded.
        self._transition_shape = (
            self.expected_clip_count,
            FUTURE_TRANSITIONS,
            len(FLOW_COMPONENTS),
            LATENT_HEIGHT,
            VIEW_LATENT_WIDTH,
        )
        self.flow_path = self._validate_flow_metadata()
        self._flows: np.ndarray | None = None
        self._validate_flow_array()

    @property
    def name(self) -> str:
        # MultiDataset uses this stable morphology identity to select the ABC
        # action/morphology token.  A conditioning wrapper must not silently
        # create a new robot morphology.
        return "ABCDataset"

    @property
    def future_measured_state_opened(self) -> bool:
        return False

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["_flows"] = None
        return state

    def _validate_flow_metadata(self) -> str:
        meta = self.flow_metadata
        digest_values = (
            self.expected_flow_sha256,
            self.expected_renderer_analysis_identity_sha256,
        )
        if any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in digest_values
        ):
            raise ValueError("flow identities must be lowercase SHA-256")
        if (
            meta.get("schema") != "raw-physics-flow-cache-v2"
            or meta.get("complete") is not True
            or meta.get("split") != self.expected_split
            or int(meta.get("clip_count", -1)) != self.expected_clip_count
            or meta.get("compact_transition_shape") != list(self._transition_shape)
            or meta.get("flow_dtype") != "float16"
            or meta.get("aligned_flow_sha256") != self.expected_flow_sha256
            or meta.get("renderer_analysis_identity_sha256")
            != self.expected_renderer_analysis_identity_sha256
            or meta.get("renderer_decision") != "GO_raw_geometry_scaffold_pass"
            or meta.get("renderer_family") != "raw_geometry_scaffold"
            or meta.get("causal_inputs_only") is not True
            or meta.get("future_rgb_opened") is not False
            or meta.get("future_measured_state_opened") is not False
            or meta.get("generator_outcome_opened") is not False
            or meta.get("protected_test_accessed") is not False
            or _canonical_identity(meta) != meta.get("identity_sha256")
        ):
            raise RuntimeError("immutable physics-flow metadata differs")
        value = meta.get("aligned_flow_file")
        if not isinstance(value, str) or not value:
            raise RuntimeError("flow metadata lacks aligned_flow_file")
        path = Path(value)
        if not path.is_absolute():
            path = Path(self.flow_metadata_path).parent / path
        path = path.resolve(strict=True)
        if not path.is_file() or path.is_symlink() or _sha256(path) != self.expected_flow_sha256:
            raise RuntimeError("aligned flow array identity differs")
        return str(path)

    def _open_flows(self) -> np.ndarray:
        if self._flows is None:
            self._flows = np.load(self.flow_path, mmap_mode="r", allow_pickle=False)
            if (
                self._flows.shape != self._transition_shape
                or self._flows.dtype != np.float16
            ):
                raise RuntimeError("immutable aligned flow array changed")
        return self._flows

    def _validate_flow_array(self) -> None:
        values = self._open_flows()
        for index in sorted({0, self.expected_clip_count - 1}):
            transitions = torch.from_numpy(
                np.array(values[index : index + 1], copy=True)
            ).float()
            validate_packed_flow(pack_top_view_flow(transitions))
        self._flows = None

    def _get_sample(self, index: int) -> dict[str, torch.Tensor]:
        sample = super()._get_sample(index)
        transitions = torch.from_numpy(
            np.array(self._open_flows()[int(index)], copy=True)
        ).float()
        flow = pack_top_view_flow(transitions.unsqueeze(0)).squeeze(0)
        validate_packed_flow(flow.unsqueeze(0))
        sample["causal_robot_flow"] = flow
        return sample
