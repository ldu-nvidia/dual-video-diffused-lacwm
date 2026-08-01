"""Clean-time rectified-flow helpers for x-predicting models.

The convention here follows Latent Forcing rather than LACWM: ``t=0`` is
Gaussian noise and ``t=1`` is clean data.  An x-predicting network is trained
with the velocity-equivalent weight ``1 / (1 - t)^2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class CleanTimeCorruption:
    """A clean/noise interpolation and its data-directed velocity target."""

    clean: Tensor
    noisy: Tensor
    noise: Tensor
    velocity_target: Tensor
    clean_time: Tensor


def _expand_clean_time(
    clean_time: Tensor | float,
    reference: Tensor,
) -> Tensor:
    time = torch.as_tensor(
        clean_time,
        device=reference.device,
        dtype=reference.dtype,
    )
    if time.ndim == 0:
        time = time.expand(reference.shape[0])
    if time.ndim != 1 or time.shape[0] != reference.shape[0]:
        raise ValueError(
            "clean_time must be scalar or have shape "
            f"[B={reference.shape[0]}], got {tuple(time.shape)}"
        )
    if not bool(torch.isfinite(time).all()) or bool(((time < 0) | (time > 1)).any()):
        raise ValueError("clean_time must contain finite values in [0, 1]")
    return time.reshape(-1, *([1] * (reference.ndim - 1)))


def corrupt_clean_time(
    clean: Tensor,
    clean_time: Tensor | float,
    *,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> CleanTimeCorruption:
    """Interpolate as ``z_t = t * clean + (1 - t) * noise``.

    The ODE moves in increasing ``t``, so its constant straight-path velocity
    has sign ``clean - noise``.
    """
    if not clean.is_floating_point():
        raise TypeError("clean state must be floating point")
    if noise is None:
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
    if noise.shape != clean.shape:
        raise ValueError("noise and clean state must have the same shape")
    if noise.device != clean.device:
        raise ValueError("noise and clean state must be on the same device")
    time = _expand_clean_time(clean_time, clean)
    noisy = time * clean + (1.0 - time) * noise
    return CleanTimeCorruption(
        clean=clean,
        noisy=noisy,
        noise=noise,
        velocity_target=clean - noise,
        clean_time=time,
    )


def x_prediction_to_velocity(
    predicted_clean: Tensor,
    state: Tensor,
    clean_time: Tensor | float,
    *,
    eps: float = 5e-2,
) -> Tensor:
    """Convert an x prediction to ``dz/dt`` at the current clean time."""
    if predicted_clean.shape != state.shape:
        raise ValueError("predicted_clean and state must have the same shape")
    if eps <= 0:
        raise ValueError("eps must be positive")
    time = _expand_clean_time(clean_time, state)
    return (predicted_clean - state) / (1.0 - time).clamp_min(eps)


def v_loss_weight(
    clean_time: Tensor | float,
    reference: Tensor,
    *,
    eps: float = 5e-2,
) -> Tensor:
    """Return the broadcastable x-error weight equivalent to velocity MSE."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    time = _expand_clean_time(clean_time, reference)
    return (1.0 - time).clamp_min(eps).reciprocal().square()


def x_prediction_v_loss(
    predicted_clean: Tensor,
    clean: Tensor,
    clean_time: Tensor | float,
    *,
    sample_weight: Tensor | None = None,
    eps: float = 5e-2,
    reduction: Literal["mean", "none"] = "mean",
) -> Tensor:
    """Velocity-equivalent MSE for a model that directly predicts clean x.

    ``sample_weight`` is applied after reducing non-batch dimensions.  A mean
    reduction intentionally divides by the full batch size, matching masked
    branch training in the released Latent Forcing implementation.
    """
    if predicted_clean.shape != clean.shape:
        raise ValueError("predicted_clean and clean must have the same shape")
    weighted_error = (
        (predicted_clean - clean).square()
        * v_loss_weight(clean_time, predicted_clean, eps=eps)
    )
    per_sample = weighted_error.flatten(1).mean(dim=1)
    if sample_weight is not None:
        if sample_weight.ndim != 1 or sample_weight.shape[0] != clean.shape[0]:
            raise ValueError(f"sample_weight must have shape [B={clean.shape[0]}]")
        per_sample = per_sample * sample_weight.to(
            device=per_sample.device,
            dtype=per_sample.dtype,
        )
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"unsupported reduction: {reduction}")


def clean_time_euler_step(
    state: Tensor,
    velocity: Tensor,
    clean_time: Tensor | float,
    next_clean_time: Tensor | float,
) -> Tensor:
    """Advance a state toward clean data using increasing clean time."""
    if state.shape != velocity.shape:
        raise ValueError("state and velocity must have the same shape")
    time = _expand_clean_time(clean_time, state)
    next_time = _expand_clean_time(next_clean_time, state)
    if bool((next_time < time).any()):
        raise ValueError("next_clean_time must not be smaller than clean_time")
    return state + (next_time - time) * velocity
