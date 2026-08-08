"""Parameter-free low-frequency motion supervision for Wan video latents.

Wan compresses the eight future RGB frames in the controlled ABC setup to only
two temporal latent tokens.  A future-only temporal derivative would therefore
contain a single transition.  This loss prepends the last *observed* latent
token, yielding two supervised transitions while remaining causal at
deployment: observed anchor -> future token 0 -> future token 1.

The three camera views are concatenated along latent width.  Each view is
spatially low-passed independently so that no convolution crosses a camera
seam.  The operation owns no parameters and is called only by the train loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LowFrequencyMotionLoss:
    total: Tensor
    anchor_transition: Tensor
    future_transition: Tensor
    target_rms_mean: Tensor
    valid_transition_count: int


def _validate(
    predicted_future: Tensor,
    clean_future: Tensor,
    clean_anchor: Tensor,
    validity_mask: Tensor | None,
    *,
    num_views: int,
    kernel_size: int,
    sigma: float,
    beta: float,
    epsilon: float,
) -> None:
    if predicted_future.ndim != 5 or clean_future.ndim != 5:
        raise ValueError("future latents must have shape [B,C,T,H,W]")
    if predicted_future.shape != clean_future.shape:
        raise ValueError("predicted and clean future shapes must match")
    if predicted_future.shape[2] != 2:
        raise ValueError("controlled Wan screen requires exactly two future tokens")
    expected_anchor = (*predicted_future.shape[:2], 1, *predicted_future.shape[3:])
    if tuple(clean_anchor.shape) != expected_anchor:
        raise ValueError("clean_anchor must be [B,C,1,H,W] on the future grid")
    if not predicted_future.is_floating_point() or not clean_future.is_floating_point():
        raise TypeError("future latents must be floating point")
    if num_views < 1 or predicted_future.shape[-1] % num_views:
        raise ValueError("latent width must divide exactly into num_views")
    if kernel_size < 3 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd and at least three")
    if min(predicted_future.shape[-2:]) <= kernel_size // 2:
        raise ValueError("spatial grid is too small for reflected Gaussian padding")
    if sigma <= 0 or beta <= 0 or epsilon <= 0:
        raise ValueError("sigma, beta, and epsilon must be positive")
    if validity_mask is not None:
        if validity_mask.ndim != 5:
            raise ValueError("validity_mask must be broadcastable [B,1,T,1,W]")
        expected = (
            predicted_future.shape[0],
            predicted_future.shape[1],
            predicted_future.shape[2] + 1,
            predicted_future.shape[3],
            predicted_future.shape[4],
        )
        try:
            torch.broadcast_shapes(expected, validity_mask.shape)
        except RuntimeError as exc:
            raise ValueError("validity_mask is not broadcastable to anchor+future") from exc


def _gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> Tensor:
    coordinates = torch.arange(kernel_size, device=device, dtype=torch.float32)
    coordinates = coordinates - (kernel_size - 1) / 2
    one_dimensional = torch.exp(-0.5 * coordinates.square() / float(sigma) ** 2)
    one_dimensional = one_dimensional / one_dimensional.sum()
    kernel = one_dimensional[:, None] * one_dimensional[None, :]
    return kernel[None, None]


def _per_view_low_pass(signal: Tensor, *, num_views: int, kernel: Tensor) -> Tensor:
    """Return [B,V,C,T,H,Wv] without filtering across view boundaries."""

    batch, channels, temporal, height, width = signal.shape
    view_width = width // num_views
    per_view = signal.reshape(batch, channels, temporal, height, num_views, view_width)
    per_view = per_view.permute(0, 4, 1, 2, 3, 5).contiguous()
    flat = per_view.reshape(batch * num_views * channels * temporal, 1, height, view_width)
    padding = kernel.shape[-1] // 2
    flat = F.pad(flat, (padding, padding, padding, padding), mode="reflect")
    filtered = F.conv2d(flat, kernel)
    return filtered.reshape(batch, num_views, channels, temporal, height, view_width)


def low_frequency_motion_consistency(
    predicted_future: Tensor,
    clean_future: Tensor,
    clean_anchor: Tensor,
    *,
    validity_mask: Tensor | None = None,
    num_views: int = 3,
    kernel_size: int = 5,
    sigma: float = 1.0,
    beta: float = 0.25,
    epsilon: float = 1.0e-4,
) -> LowFrequencyMotionLoss:
    """Compare target-RMS-normalized, low-pass adjacent latent transitions.

    ``predicted_future`` is ``x0_hat = x_sigma - sigma*v_hat``.  The clean
    observed anchor is prepended to both predicted and target trajectories.
    Smooth-L1 is then applied to the two signed adjacent-time differences.  A
    detached target RMS is computed per sample/view, preventing static clips
    or high-energy views from setting the scale for unrelated examples.
    """

    _validate(
        predicted_future,
        clean_future,
        clean_anchor,
        validity_mask,
        num_views=num_views,
        kernel_size=kernel_size,
        sigma=sigma,
        beta=beta,
        epsilon=epsilon,
    )
    estimate = predicted_future.float()
    target = clean_future.detach().float()
    anchor = clean_anchor.detach().float()
    if not all(bool(torch.isfinite(value).all()) for value in (estimate, target, anchor)):
        raise FloatingPointError("motion consistency received non-finite latents")

    trajectory_estimate = torch.cat((anchor, estimate), dim=2)
    trajectory_target = torch.cat((anchor, target), dim=2)
    if validity_mask is None:
        validity = torch.ones(
            trajectory_estimate.shape[0],
            1,
            trajectory_estimate.shape[2],
            1,
            trajectory_estimate.shape[-1],
            device=estimate.device,
            dtype=torch.float32,
        )
    else:
        validity = validity_mask.to(device=estimate.device, dtype=torch.float32)
        if not bool(torch.isfinite(validity).all()):
            raise FloatingPointError("validity_mask contains non-finite values")
        if bool(((validity < 0) | (validity > 1)).any()):
            raise ValueError("validity_mask values must lie in [0,1]")
        validity = validity.expand(
            trajectory_estimate.shape[0],
            1,
            trajectory_estimate.shape[2],
            trajectory_estimate.shape[-2],
            trajectory_estimate.shape[-1],
        )

    # Zeroing before filtering guarantees exactly zero gradients for invalid views.
    trajectory_estimate = trajectory_estimate * validity
    trajectory_target = trajectory_target * validity
    kernel = _gaussian_kernel(kernel_size, sigma, estimate.device)
    filtered_estimate = _per_view_low_pass(
        trajectory_estimate, num_views=num_views, kernel=kernel
    )
    filtered_target = _per_view_low_pass(
        trajectory_target, num_views=num_views, kernel=kernel
    )
    estimate_motion = torch.diff(filtered_estimate, dim=3)
    target_motion = torch.diff(filtered_target, dim=3)

    batch, _, temporal, height, width = validity.shape
    view_width = width // num_views
    view_validity = validity.reshape(batch, 1, temporal, height, num_views, view_width)
    view_validity = view_validity.permute(0, 4, 1, 2, 3, 5)
    transition_validity = view_validity[:, :, :, :-1] * view_validity[:, :, :, 1:]
    transition_validity = transition_validity.expand_as(target_motion)
    valid_per_view = transition_validity.sum(dim=(2, 3, 4, 5), keepdim=True)
    if bool((valid_per_view == 0).all(dim=1).any()):
        raise ValueError("each sample must contain at least one valid view transition")

    target_rms = (
        (target_motion.square() * transition_validity).sum(
            dim=(2, 3, 4, 5), keepdim=True
        )
        / valid_per_view.clamp_min(1.0)
    ).sqrt().detach().clamp_min(float(epsilon))
    error = F.smooth_l1_loss(
        estimate_motion / target_rms,
        target_motion / target_rms,
        reduction="none",
        beta=float(beta),
    )

    def reduce_transition(index: int) -> Tensor:
        weights = transition_validity[:, :, :, index : index + 1]
        return (error[:, :, :, index : index + 1] * weights).sum() / weights.sum().clamp_min(1.0)

    anchor_transition = reduce_transition(0)
    future_transition = reduce_transition(1)
    total = (error * transition_validity).sum() / transition_validity.sum().clamp_min(1.0)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("motion consistency produced a non-finite loss")
    return LowFrequencyMotionLoss(
        total=total,
        anchor_transition=anchor_transition,
        future_transition=future_transition,
        target_rms_mean=target_rms.mean(),
        valid_transition_count=int(
            (transition_validity[:, :, :1].amax(dim=(-2, -1)) > 0).sum().item()
        ),
    )
