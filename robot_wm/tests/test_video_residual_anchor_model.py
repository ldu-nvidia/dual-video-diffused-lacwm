from __future__ import annotations

import inspect

import pytest


torch = pytest.importorskip("torch")

from lam.video_residual_anchor_model import (  # noqa: E402
    VideoResidualAnchorVPM,
    invert_video_residual_anchor,
    pack_video_residual_anchor,
)


def _inputs():
    generator = torch.Generator().manual_seed(17)
    absolute = torch.randn(3, 16, 4, 5, 7, generator=generator)
    reference = torch.zeros_like(absolute)
    reference[:, :, :2] = torch.randn(3, 16, 2, 5, 7, generator=generator)
    return absolute, reference


@pytest.mark.parametrize("mode", ("absolute", "cumulative_residual"))
def test_pack_inverse_preserves_exact_history_and_absolute_future(mode: str) -> None:
    absolute, reference = _inputs()
    packed = pack_video_residual_anchor(absolute, reference, 2, mode=mode)
    recovered = invert_video_residual_anchor(packed, reference, 2, mode=mode)
    assert torch.equal(packed[:, :, :2], reference[:, :, :2])
    assert torch.equal(recovered[:, :, :2], reference[:, :, :2])
    torch.testing.assert_close(recovered[:, :, 2:], absolute[:, :, 2:])


def test_residual_coordinates_are_anchor_displacement_and_increment() -> None:
    absolute, reference = _inputs()
    packed = pack_video_residual_anchor(
        absolute, reference, 2, mode="cumulative_residual"
    )
    torch.testing.assert_close(
        packed[:, :, 2], absolute[:, :, 2] - reference[:, :, 1]
    )
    torch.testing.assert_close(packed[:, :, 3], absolute[:, :, 3] - absolute[:, :, 2])


def test_reference_future_must_be_zero() -> None:
    absolute, reference = _inputs()
    reference[:, :, 2] = 1.0
    with pytest.raises(ValueError, match="reference future"):
        pack_video_residual_anchor(
            absolute, reference, 2, mode="cumulative_residual"
        )


def test_deployable_sampler_signature_cannot_accept_a_clean_target() -> None:
    parameters = inspect.signature(
        VideoResidualAnchorVPM.sample_video_residual_anchor
    ).parameters
    assert set(parameters) == {
        "self",
        "history_rgb",
        "actions",
        "morphology_index",
        "video_noise",
        "auxiliary_noise",
        "steps",
    }
    assert not any(
        forbidden in name
        for name in parameters
        for forbidden in ("clean", "target", "teacher", "future")
    )
