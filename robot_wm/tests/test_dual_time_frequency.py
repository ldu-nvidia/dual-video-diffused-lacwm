import math

import pytest
import torch

from robot_wm.modeling.dual_diffusion.time_frequency import (
    PerViewCausalRFFT,
    PerViewTemporalSTFT,
)


def test_causal_rfft_shape_and_exact_round_trip():
    video = torch.randn(2, 13, 3, 6, 12, dtype=torch.float64)
    transform = PerViewCausalRFFT(
        num_views=3, output_size=(6, 12), window_size=4, pad_multiple=None
    )

    coefficients = transform(video)
    reconstructed = transform.inverse(coefficients)

    assert coefficients.shape == (2, 12, 4, 6, 12)
    torch.testing.assert_close(reconstructed, video, rtol=1e-10, atol=1e-10)


def test_causal_rfft_preserves_phase_as_real_imaginary_components():
    n = torch.arange(4, dtype=torch.float64)
    base = torch.cos(2 * math.pi * n / 4)
    shifted = torch.roll(base, shifts=1)
    first = torch.cat([base[:1], base, base, base]).reshape(1, 13, 1, 1, 1)
    second = torch.cat([shifted[:1], shifted, shifted, shifted]).reshape(1, 13, 1, 1, 1)
    transform = PerViewCausalRFFT(
        num_views=1, output_size=(1, 1), window_size=4, pad_multiple=None
    )

    first_coefficients = transform(first)[0, :, 1, 0, 0]
    second_coefficients = transform(second)[0, :, 1, 0, 0]

    first_magnitude = torch.hypot(first_coefficients[1], first_coefficients[2])
    second_magnitude = torch.hypot(second_coefficients[1], second_coefficients[2])
    torch.testing.assert_close(first_magnitude, second_magnitude)
    assert not torch.allclose(first_coefficients[1:3], second_coefficients[1:3])
    assert second_coefficients[2].abs() > 0.5


def test_transforms_never_mix_width_stacked_views():
    video = torch.zeros(1, 13, 1, 4, 12)
    video[..., :4] = torch.randn(1, 13, 1, 4, 4)

    rfft = PerViewCausalRFFT(
        num_views=3, output_size=(4, 12), window_size=4, pad_multiple=None
    )
    stft = PerViewTemporalSTFT(
        num_views=3,
        output_size=(4, 12),
        target_frames=4,
        n_fft=5,
        hop_length=2,
        pad_multiple=None,
    )

    for coefficients in (rfft(video), stft(video)):
        assert coefficients[..., :4].abs().sum() > 0
        assert torch.count_nonzero(coefficients[..., 4:]) == 0


def test_stft_contract_matches_wan_temporal_grid():
    video = torch.randn(2, 13, 3, 8, 24)
    transform = PerViewTemporalSTFT(
        num_views=3,
        output_size=(4, 12),
        target_frames=4,
        n_fft=5,
        hop_length=2,
        pad_multiple=None,
    )
    assert transform(video).shape == (2, 18, 4, 4, 12)


def test_invalid_temporal_contract_fails_closed():
    transform = PerViewCausalRFFT(num_views=3, output_size=(4, 12), window_size=4)
    with pytest.raises(ValueError, match="T=1"):
        transform(torch.randn(1, 12, 3, 4, 12))


def test_causal_history_bins_do_not_depend_on_hidden_future_frames():
    first = torch.randn(1, 13, 3, 8, 24)
    second = first.clone()
    second[:, 5:] = torch.randn_like(second[:, 5:])
    transform = PerViewCausalRFFT(
        num_views=3,
        output_size=(4, 12),
        window_size=4,
        pad_multiple=None,
        normalization="none",
    )

    first_tf = transform(first)
    second_tf = transform(second)

    torch.testing.assert_close(first_tf[:, :, :2], second_tf[:, :, :2])
    assert not torch.allclose(first_tf[:, :, 2:], second_tf[:, :, 2:])


def test_production_padding_and_shape_contract():
    video = torch.ones(1, 13, 3, 180, 960)
    transform = PerViewCausalRFFT(
        num_views=3, output_size=(24, 120), window_size=4, pad_multiple=16
    )
    coefficients = transform(video)

    assert coefficients.shape == (1, 12, 4, 24, 120)
    # Temporal constants have no non-DC energy, including the padded rows.
    packed = coefficients.reshape(1, 3, 4, 4, 24, 120)
    assert torch.count_nonzero(packed[:, :, 1:]) == 0
