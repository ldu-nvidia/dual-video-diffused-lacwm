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


def derive_tf_sigma(
    video_sigma: Tensor,
    *,
    mode: Literal["aligned", "tf_leads"] = "tf_leads",
    tf_lead_logit: float = 1.0,
    eps: float = 1e-5,
) -> Tensor:
    """Derive the TF clock from the production video sigma.

    This deliberately accepts the *actual* shifted Wan scheduler sigma instead
    of resampling a second video clock.  LACWM uses sigma=1 for noise and
    sigma=0 for clean data.  Endpoints are restored exactly after the logit
    transform so joint inference starts from noise and ends at data.
    """
    if not video_sigma.is_floating_point():
        raise TypeError("video_sigma must be floating point")
    if tf_lead_logit < 0:
        raise ValueError("tf_lead_logit must be non-negative")
    if not 0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")
    if mode == "aligned":
        return video_sigma.clone()
    if mode != "tf_leads":
        raise ValueError(f"unsupported derived TF clock mode: {mode}")
    clipped = video_sigma.clamp(eps, 1.0 - eps)
    tf_sigma = torch.sigmoid(torch.logit(clipped) - tf_lead_logit)
    tf_sigma = torch.where(video_sigma <= 0, torch.zeros_like(tf_sigma), tf_sigma)
    tf_sigma = torch.where(video_sigma >= 1, torch.ones_like(tf_sigma), tf_sigma)
    return tf_sigma


def pair_video_sigma_schedule(
    video_sigmas: Tensor,
    *,
    mode: Literal["aligned", "tf_leads"] = "tf_leads",
    tf_lead_logit: float = 1.0,
) -> PairedSigmaSchedule:
    """Pair a native, monotonically decreasing Wan schedule with a TF clock."""
    if video_sigmas.ndim != 1 or video_sigmas.numel() < 2:
        raise ValueError("video_sigmas must contain at least two scalar nodes")
    if torch.any(torch.diff(video_sigmas) > 0):
        raise ValueError("video sigma schedule must be monotonically non-increasing")
    return PairedSigmaSchedule(
        video=video_sigmas,
        time_frequency=derive_tf_sigma(
            video_sigmas, mode=mode, tf_lead_logit=tf_lead_logit
        ),
    )


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
        native_video_sigma: Tensor | None = None,
    ) -> DualClockBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if native_video_sigma is None:
            video = self._sample_base(batch_size, device, generator)
        else:
            if (
                native_video_sigma.ndim != 1
                or native_video_sigma.shape[0] != batch_size
                or not native_video_sigma.is_floating_point()
            ):
                raise ValueError(
                    "native_video_sigma must be floating point with shape [B]"
                )
            video = native_video_sigma.to(device=device, dtype=torch.float32)
            if not bool(torch.isfinite(video).all()) or bool(
                ((video < 0) | (video > 1)).any()
            ):
                raise ValueError("native_video_sigma must lie in [0,1]")
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
        tf_steps, video_steps = cascaded_step_counts(
            num_steps, tf_fraction=tf_fraction
        )
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


def cascaded_step_counts(
    num_steps: int,
    *,
    tf_fraction: float = 0.5,
) -> tuple[int, int]:
    """Split total model evaluations into nonempty TF-first and video phases."""
    if num_steps < 2:
        raise ValueError("cascaded scheduling requires at least two steps")
    if not 0 < tf_fraction < 1:
        raise ValueError("tf_fraction must lie strictly between 0 and 1")
    tf_steps = min(num_steps - 1, max(1, round(num_steps * tf_fraction)))
    return tf_steps, num_steps - tf_steps


def pair_native_cascaded_sigma_schedule(
    native_video_sigmas: Tensor,
    *,
    total_steps: int,
    tf_fraction: float = 0.5,
) -> PairedSigmaSchedule:
    """Prepend a TF-only phase to a native Wan video sigma schedule.

    ``native_video_sigmas`` must contain the exact scheduler nodes for the
    video phase alone. The returned schedule has ``total_steps`` model calls:
    video remains at pure noise while TF moves from noise to data, then TF
    remains clean while video follows every supplied native scheduler node.
    """
    tf_steps, video_steps = cascaded_step_counts(
        total_steps, tf_fraction=tf_fraction
    )
    if (
        native_video_sigmas.ndim != 1
        or native_video_sigmas.numel() != video_steps + 1
    ):
        raise ValueError(
            "native_video_sigmas must contain video_steps+1 scalar nodes"
        )
    if not native_video_sigmas.is_floating_point():
        raise TypeError("native_video_sigmas must be floating point")
    if torch.any(torch.diff(native_video_sigmas) > 0):
        raise ValueError(
            "native video sigma schedule must be monotonically non-increasing"
        )
    if native_video_sigmas[0] != 1 or native_video_sigmas[-1] != 0:
        raise ValueError(
            "native video sigma schedule must have exact noise/data endpoints"
        )
    video = torch.cat(
        [
            torch.ones(
                tf_steps,
                device=native_video_sigmas.device,
                dtype=native_video_sigmas.dtype,
            ),
            native_video_sigmas,
        ]
    )
    time_frequency = torch.cat(
        [
            torch.linspace(
                1,
                0,
                tf_steps + 1,
                device=native_video_sigmas.device,
                dtype=native_video_sigmas.dtype,
            ),
            torch.zeros(
                video_steps,
                device=native_video_sigmas.device,
                dtype=native_video_sigmas.dtype,
            ),
        ]
    )
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
