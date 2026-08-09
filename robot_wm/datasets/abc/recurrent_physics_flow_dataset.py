"""ABC clips conditioned on the sealed recurrent-delta robot-flow cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_wm.datasets.abc.physics_flow_dataset import (
    ABCPhysicsFlowDataset,
    _canonical_identity,
    _sha256,
)


class ABCRecurrentPhysicsFlowDataset(ABCPhysicsFlowDataset):
    """Replay recurrent-delta flow without relabelling it as raw geometry."""

    def __init__(
        self,
        *,
        expected_confirmation_analysis_identity_sha256: str,
        expected_confirmation_completion_identity_sha256: str,
        **kwargs: Any,
    ) -> None:
        self.expected_confirmation_analysis_identity_sha256 = str(
            expected_confirmation_analysis_identity_sha256
        )
        self.expected_confirmation_completion_identity_sha256 = str(
            expected_confirmation_completion_identity_sha256
        )
        # The base class stores this field and dispatches metadata validation
        # virtually, so no raw-family contract is exercised by this subclass.
        super().__init__(
            expected_renderer_analysis_identity_sha256=(
                self.expected_confirmation_analysis_identity_sha256
            ),
            **kwargs,
        )

    def _validate_flow_metadata(self) -> str:
        meta = self.flow_metadata
        digest_values = (
            self.expected_flow_sha256,
            self.expected_confirmation_analysis_identity_sha256,
            self.expected_confirmation_completion_identity_sha256,
        )
        if any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in digest_values
        ):
            raise ValueError("recurrent-flow identities must be lowercase SHA-256")
        if (
            meta.get("schema") != "recurrent-physics-flow-cache-v1"
            or meta.get("complete") is not True
            or meta.get("split") != self.expected_split
            or int(meta.get("clip_count", -1)) != self.expected_clip_count
            or meta.get("compact_transition_shape") != list(self._transition_shape)
            or meta.get("flow_dtype") != "float16"
            or meta.get("aligned_flow_sha256") != self.expected_flow_sha256
            or meta.get("condition_family") != "sealed_recurrent_delta_geometry"
            or meta.get("confirmation_decision")
            != "GO_RECURRENT_DELTA_WAN_SCREEN"
            or meta.get("confirmation_analysis_identity_sha256")
            != self.expected_confirmation_analysis_identity_sha256
            or meta.get("confirmation_completion_identity_sha256")
            != self.expected_confirmation_completion_identity_sha256
            or meta.get("predictor_refit_or_tuning") is not False
            or meta.get("causal_inputs_only") is not True
            or meta.get("future_rgb_opened") is not False
            or meta.get("future_measured_state_opened") is not False
            or meta.get("generator_outcome_opened") is not False
            or meta.get("protected_test_accessed") is not False
            or _canonical_identity(meta) != meta.get("identity_sha256")
        ):
            raise RuntimeError("immutable recurrent-flow metadata differs")
        value = meta.get("aligned_flow_file")
        if not isinstance(value, str) or not value:
            raise RuntimeError("recurrent-flow metadata lacks aligned_flow_file")
        path = Path(value)
        if not path.is_absolute():
            path = Path(self.flow_metadata_path).parent / path
        if path.is_symlink():
            raise RuntimeError("recurrent aligned-flow path may not be a symlink")
        path = path.resolve(strict=True)
        if (
            not path.is_file()
            or _sha256(path) != self.expected_flow_sha256
        ):
            raise RuntimeError("recurrent aligned-flow identity differs")
        return str(path)
