"""Training-only spatiotemporal spectrum supervision for video latents.

The transform is deliberately stateless: it owns no parameters, cache, teacher,
or inference branch.  Inputs use Wan's native ``[B,C,T,H,W]`` latent layout.
We take one orthonormal 3-D real FFT over the *future* ``(T,H,W)`` axes, retain
all temporal frequencies and a preregistered low spatial band, and compare both
log amplitude and phase coherence.  Computing in float32 keeps CUDA FFT stable
when the surrounding model trains in bfloat16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class SpectralConsistencyLoss:
    total: Tensor
    log_amplitude: Tensor
    phase: Tensor
    selected_bins: int


def _validate(
    predicted: Tensor,
    target: Tensor,
    validity_mask: Tensor | None,
    spatial_band_fraction: float,
    phase_weight: float,
    epsilon: float,
) -> None:
    if predicted.ndim != 5 or target.ndim != 5:
        raise ValueError("predicted and target must have shape [B,C,T,H,W]")
    if predicted.shape != target.shape:
        raise ValueError("predicted and target shapes must match")
    if not predicted.is_floating_point() or not target.is_floating_point():
        raise TypeError("predicted and target must be floating point")
    if min(int(value) for value in predicted.shape[2:]) < 1:
        raise ValueError("the future spatiotemporal volume must be nonempty")
    if not 0.0 < float(spatial_band_fraction) <= 1.0:
        raise ValueError("spatial_band_fraction must lie in (0,1]")
    if float(phase_weight) < 0.0:
        raise ValueError("phase_weight must be non-negative")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")
    if validity_mask is not None:
        if validity_mask.ndim != 5:
            raise ValueError("validity_mask must be broadcastable [B,1,T,1,W]")
        try:
            torch.broadcast_shapes(predicted.shape, validity_mask.shape)
        except RuntimeError as exc:
            raise ValueError(
                "validity_mask is not broadcastable to the signal"
            ) from exc


def _low_spatial_band(shape: tuple[int, int, int], fraction: float, device):
    """Return ``[T,H,W//2+1]`` mask; ``fraction`` is relative to Nyquist."""

    temporal, height, width = shape
    fy = torch.fft.fftfreq(height, device=device).abs()
    fx = torch.fft.rfftfreq(width, device=device)
    cutoff = 0.5 * float(fraction) + 8.0 * torch.finfo(torch.float32).eps
    spatial = (fy[:, None] <= cutoff) & (fx[None, :] <= cutoff)
    return spatial.unsqueeze(0).expand(temporal, -1, -1)


def spatiotemporal_spectral_consistency(
    predicted: Tensor,
    target: Tensor,
    *,
    validity_mask: Tensor | None = None,
    spatial_band_fraction: float = 0.5,
    phase_weight: float = 0.25,
    epsilon: float = 1.0e-4,
) -> SpectralConsistencyLoss:
    """Compare low-band 3-D spectrum of predicted and clean future latents.

    The amplitude term is Smooth-L1 on ``log1p(|X|/s)`` where ``s`` is the
    detached target RMS spectrum for each sample/channel.  The phase term is
    ``1-cos(angle(X)-angle(Y))``.  It excludes only the global DC bin and is
    weighted by capped target amplitude, so phase at vanishing-energy bins
    cannot dominate.  Invalid future regions are zeroed in both signals before
    the FFT and therefore receive exactly zero gradient.
    """

    _validate(
        predicted,
        target,
        validity_mask,
        spatial_band_fraction,
        phase_weight,
        epsilon,
    )
    estimate = predicted.float()
    clean = target.detach().float()
    if not bool(torch.isfinite(estimate).all()) or not bool(
        torch.isfinite(clean).all()
    ):
        raise FloatingPointError("spectral consistency received non-finite latents")
    if validity_mask is not None:
        valid = validity_mask.to(device=estimate.device, dtype=estimate.dtype)
        if not bool(torch.isfinite(valid).all()):
            raise FloatingPointError("validity_mask contains non-finite values")
        if bool(((valid < 0) | (valid > 1)).any()):
            raise ValueError("validity_mask values must lie in [0,1]")
        estimate = estimate * valid
        clean = clean * valid

    estimate_fft = torch.fft.rfftn(estimate, dim=(-3, -2, -1), norm="ortho")
    clean_fft = torch.fft.rfftn(clean, dim=(-3, -2, -1), norm="ortho")
    band = _low_spatial_band(
        tuple(int(v) for v in estimate.shape[-3:]),
        spatial_band_fraction,
        estimate.device,
    )
    selected_bins = int(band.sum().item())
    if selected_bins < 2:
        raise ValueError("spectral band must contain DC plus at least one non-DC bin")
    band_f = band.to(dtype=estimate.dtype)[None, None]

    estimate_magnitude = estimate_fft.abs()
    clean_magnitude = clean_fft.abs()
    count = band_f.sum().clamp_min(1.0)
    target_rms = (
        (
            (clean_magnitude.square() * band_f).sum(dim=(-3, -2, -1), keepdim=True)
            / count
        )
        .sqrt()
        .detach()
        .clamp_min(float(epsilon))
    )
    estimate_log_amplitude = torch.log1p(estimate_magnitude / target_rms)
    clean_log_amplitude = torch.log1p(clean_magnitude / target_rms)
    amplitude_error = F.smooth_l1_loss(
        estimate_log_amplitude,
        clean_log_amplitude,
        reduction="none",
        beta=0.1,
    )
    amplitude = (amplitude_error * band_f).sum() / (
        band_f.sum() * predicted.shape[0] * predicted.shape[1]
    )

    stabilizer = (float(epsilon) * target_rms).square()
    phase_cosine = (
        (estimate_fft * clean_fft.conj()).real
        / (
            (estimate_magnitude.square() + stabilizer).sqrt()
            * (clean_magnitude.square() + stabilizer).sqrt()
        )
    ).clamp(-1.0, 1.0)
    phase_band = band.clone()
    phase_band[0, 0, 0] = False
    phase_band_f = phase_band.to(dtype=estimate.dtype)[None, None]
    mean_target_magnitude = (
        (
            (clean_magnitude * phase_band_f).sum(dim=(-3, -2, -1), keepdim=True)
            / phase_band_f.sum().clamp_min(1.0)
        )
        .detach()
        .clamp_min(float(epsilon))
    )
    energy_weight = (clean_magnitude / mean_target_magnitude).detach().clamp(max=4.0)
    phase_weights = energy_weight * phase_band_f
    phase = (
        (1.0 - phase_cosine) * phase_weights
    ).sum() / phase_weights.sum().clamp_min(float(epsilon))
    total = amplitude + float(phase_weight) * phase
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("spectral consistency produced a non-finite loss")
    return SpectralConsistencyLoss(
        total=total,
        log_amplitude=amplitude,
        phase=phase,
        selected_bins=selected_bins,
    )
