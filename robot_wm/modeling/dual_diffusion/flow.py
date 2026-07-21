"""Two-state flow-matching corruption and inference schedules.

LACWM uses ``sigma=0`` for clean data and ``sigma=1`` for Gaussian noise.  This
is the opposite direction from Latent Forcing's ``t`` convention, so all public
APIs in this module use the native LACWM sigma convention explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FlowCorruption:
    clean: Tensor
    noisy: Tensor
    noise: Tensor
    velocity_target: Tensor
    sigma: Tensor


@dataclass(frozen=True)
class DualFlowCorruption:
    video: FlowCorruption
    time_frequency: FlowCorruption


@dataclass(frozen=True)
class DualClockBatch:
    video_sigma: Tensor
    tf_sigma: Tensor
    video_loss_weight: Tensor
    tf_loss_weight: Tensor


@dataclass(frozen=True)
class PairedSigmaSchedule:
    video: Tensor
    time_frequency: Tensor

    @property
    def num_steps(self) -> int:
        return self.video.numel() - 1


def _expand_sigma(sigma: Tensor, reference: Tensor) -> Tensor:
    if sigma.ndim == 0:
        sigma = sigma.expand(reference.shape[0])
    if sigma.ndim != 1 or sigma.shape[0] != reference.shape[0]:
        raise ValueError(
            f"sigma must be scalar or [B={reference.shape[0]}], got {tuple(sigma.shape)}"
        )
    return sigma.to(device=reference.device, dtype=reference.dtype).reshape(
        -1, *([1] * (reference.ndim - 1))
    )


def corrupt_flow(
    clean: Tensor,
    sigma: Tensor,
    *,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> FlowCorruption:
    """Apply rectified-flow corruption ``(1-sigma)*clean + sigma*noise``."""
    if not clean.is_floating_point():
        raise TypeError("clean state must be floating point")
    if noise is None:
        noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
    if noise.shape != clean.shape:
        raise ValueError("noise and clean state must have the same shape")
    sigma_expanded = _expand_sigma(sigma, clean)
    noisy = (1.0 - sigma_expanded) * clean + sigma_expanded * noise
    return FlowCorruption(
        clean=clean,
        noisy=noisy,
        noise=noise,
        velocity_target=noise - clean,
        sigma=sigma_expanded,
    )


def corrupt_dual_flow(
    video: Tensor,
    time_frequency: Tensor,
    clocks: DualClockBatch,
    *,
    video_noise: Tensor | None = None,
    tf_noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> DualFlowCorruption:
    if video.shape[0] != time_frequency.shape[0]:
        raise ValueError("video and TF states must have the same batch size")
    return DualFlowCorruption(
        video=corrupt_flow(
            video, clocks.video_sigma, noise=video_noise, generator=generator
        ),
        time_frequency=corrupt_flow(
            time_frequency, clocks.tf_sigma, noise=tf_noise, generator=generator
        ),
    )


class DualClockSampler(nn.Module):
    """Sample paired training noise levels and branch loss masks.

    Modes:
        ``aligned``: both branches receive the same logit-normal sigma.
        ``independent``: independent logit-normal sigmas, both losses active.
        ``tf_leads``: TF is cleaner than video by a logit-space offset.
        ``tf_first_cascaded_noised``: branch-selected Latent-Forcing-style
            training. TF-loss examples see pure-noise video; video-loss examples
            see a mostly clean, but imperfect, TF state.
    """

    def __init__(
        self,
        *,
        mode: Literal[
            "aligned", "independent", "tf_leads", "tf_first_cascaded_noised"
        ] = "independent",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        tf_lead_logit: float = 1.0,
        tf_loss_probability: float = 0.4,
        tf_condition_max_sigma: float = 0.25,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if logit_std <= 0:
            raise ValueError("logit_std must be positive")
        if tf_lead_logit < 0:
            raise ValueError("tf_lead_logit must be non-negative")
        if not 0 <= tf_loss_probability <= 1:
            raise ValueError("tf_loss_probability must lie in [0,1]")
        if not 0 <= tf_condition_max_sigma <= 1:
            raise ValueError("tf_condition_max_sigma must lie in [0,1]")
        self.mode = mode
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.tf_lead_logit = tf_lead_logit
        self.tf_loss_probability = tf_loss_probability
        self.tf_condition_max_sigma = tf_condition_max_sigma
        self.eps = eps

    def _sample_base(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None,
    ) -> Tensor:
        logits = torch.randn(
            batch_size, device=device, dtype=torch.float32, generator=generator
        )
        return torch.sigmoid(logits * self.logit_std + self.logit_mean)

    def forward(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> DualClockBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        video = self._sample_base(batch_size, device, generator)
        ones = torch.ones(batch_size, device=device, dtype=torch.float32)

        if self.mode == "aligned":
            tf = video.clone()
            video_weight = tf_weight = ones
        elif self.mode == "independent":
            tf = self._sample_base(batch_size, device, generator)
            video_weight = tf_weight = ones
        elif self.mode == "tf_leads":
            logits = torch.logit(video.clamp(self.eps, 1.0 - self.eps))
            tf = torch.sigmoid(logits - self.tf_lead_logit)
            video_weight = tf_weight = ones
        elif self.mode == "tf_first_cascaded_noised":
            tf_draw = self._sample_base(batch_size, device, generator)
            selector = torch.rand(
                batch_size, device=device, dtype=torch.float32, generator=generator
            ) < self.tf_loss_probability
            imperfect_tf = torch.rand(
                batch_size, device=device, dtype=torch.float32, generator=generator
            ) * self.tf_condition_max_sigma
            video = torch.where(selector, ones, video)
            tf = torch.where(selector, tf_draw, imperfect_tf)
            tf_weight = selector.float()
            video_weight = (~selector).float()
        else:
            raise ValueError(f"unsupported dual clock mode: {self.mode}")

        return DualClockBatch(
            video_sigma=video.to(dtype=dtype),
            tf_sigma=tf.to(dtype=dtype),
            video_loss_weight=video_weight.to(dtype=dtype),
            tf_loss_weight=tf_weight.to(dtype=dtype),
        )


def make_paired_sigma_schedule(
    num_steps: int,
    *,
    mode: Literal["aligned", "tf_leads", "tf_first_cascaded"] = "aligned",
    tf_fraction: float = 0.5,
    tf_lead_power: float = 0.5,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> PairedSigmaSchedule:
    """Build monotonically decreasing video/TF inference noise levels."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if not 0 < tf_fraction < 1:
        raise ValueError("tf_fraction must lie strictly between 0 and 1")
    if not 0 < tf_lead_power <= 1:
        raise ValueError("tf_lead_power must lie in (0,1]")

    progress = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)
    if mode == "aligned":
        video = time_frequency = 1.0 - progress
    elif mode == "tf_leads":
        video = 1.0 - progress
        time_frequency = 1.0 - progress.pow(tf_lead_power)
    elif mode == "tf_first_cascaded":
        if num_steps < 2:
            raise ValueError("cascaded scheduling requires at least two steps")
        tf_steps = min(num_steps - 1, max(1, round(num_steps * tf_fraction)))
        video_steps = num_steps - tf_steps
        time_frequency = torch.cat(
            [
                torch.linspace(1, 0, tf_steps + 1, device=device, dtype=dtype),
                torch.zeros(video_steps, device=device, dtype=dtype),
            ]
        )
        video = torch.cat(
            [
                torch.ones(tf_steps + 1, device=device, dtype=dtype),
                torch.linspace(1, 0, video_steps + 1, device=device, dtype=dtype)[1:],
            ]
        )
    else:
        raise ValueError(f"unsupported inference schedule mode: {mode}")
    return PairedSigmaSchedule(video=video, time_frequency=time_frequency)


def euler_flow_step(state: Tensor, velocity: Tensor, sigma: Tensor, next_sigma: Tensor) -> Tensor:
    """One Euler step for the LACWM ``dx/dsigma = velocity`` convention."""
    if state.shape != velocity.shape:
        raise ValueError("state and velocity must have the same shape")
    if sigma.ndim <= 1:
        sigma = _expand_sigma(sigma, state)
    if next_sigma.ndim <= 1:
        next_sigma = _expand_sigma(next_sigma, state)
    return state + (next_sigma - sigma) * velocity
