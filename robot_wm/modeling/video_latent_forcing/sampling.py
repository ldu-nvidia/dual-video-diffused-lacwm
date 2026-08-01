"""Strict auxiliary-first sampling for Video Latent Forcing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .flow import clean_time_euler_step, x_prediction_to_velocity
from .model import VideoLatentForcingModel


AuxiliaryCondition = Literal["generated", "off", "shuffled"]


@dataclass(frozen=True)
class AuxiliarySample:
    """Autonomous auxiliary phase boundary and its two initial noise states."""

    generated_auxiliary: Tensor
    initial_video_noise: Tensor
    initial_auxiliary_noise: Tensor
    model_calls: int


@dataclass(frozen=True)
class VideoOnlySample:
    """Video-only generation with no auxiliary trajectory or condition."""

    video: Tensor
    initial_video_noise: Tensor
    model_calls: int


@dataclass(frozen=True)
class CascadeSample:
    """Final states and phase-boundary evidence from a strict cascade."""

    video: Tensor
    generated_auxiliary: Tensor
    conditioning_auxiliary: Tensor
    initial_video_noise: Tensor
    initial_auxiliary_noise: Tensor
    model_calls: int
    auxiliary_condition: AuxiliaryCondition


def apply_auxiliary_control(
    generated_auxiliary: Tensor,
    mode: AuxiliaryCondition,
) -> tuple[Tensor, bool]:
    """Return the video-phase state and whether it may enter the transformer."""
    if mode == "generated":
        return generated_auxiliary, True
    if mode == "off":
        return generated_auxiliary, False
    if mode == "shuffled":
        if generated_auxiliary.shape[0] < 2:
            raise ValueError("shuffled auxiliary control requires batch size at least two")
        return torch.roll(generated_auxiliary, shifts=1, dims=0), True
    raise ValueError(f"unsupported auxiliary condition: {mode}")


def _initial_noise(
    shape: tuple[int, ...],
    reference: Tensor,
    supplied: Tensor | None,
    generator: torch.Generator | None,
    name: str,
) -> Tensor:
    if supplied is None:
        return torch.randn(
            shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
    if tuple(supplied.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not supplied.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    return supplied.to(device=reference.device, dtype=reference.dtype).clone()


def _validate_conditioning(
    model: VideoLatentForcingModel,
    history: Tensor,
    actions: Tensor,
) -> int:
    cfg = model.config
    if history.ndim != 5 or tuple(history.shape[1:]) != cfg.history_shape:
        raise ValueError(f"history must have shape [B,{cfg.history_shape}]")
    if not history.is_floating_point():
        raise TypeError("history must be floating point")
    batch_size = history.shape[0]
    expected_actions = (batch_size, cfg.action_steps, cfg.action_dim)
    if tuple(actions.shape) != expected_actions:
        raise ValueError(f"actions must have shape {expected_actions}")
    if not actions.is_floating_point():
        raise TypeError("actions must be floating point")
    return batch_size


@torch.no_grad()
def sample_auxiliary_only(
    model: VideoLatentForcingModel,
    history: Tensor,
    actions: Tensor,
    *,
    auxiliary_steps: int = 25,
    generator: torch.Generator | None = None,
    video_noise: Tensor | None = None,
    auxiliary_noise: Tensor | None = None,
    velocity_eps: float = 5e-2,
) -> AuxiliarySample:
    """Generate only the autonomous auxiliary phase and expose its boundary."""
    if auxiliary_steps < 1:
        raise ValueError("auxiliary_steps must be positive")
    if velocity_eps <= 0:
        raise ValueError("velocity_eps must be positive")
    cfg = model.config
    batch_size = _validate_conditioning(model, history, actions)
    initial_video = _initial_noise(
        (batch_size, *cfg.future_shape),
        history,
        video_noise,
        generator,
        "video_noise",
    )
    initial_auxiliary = _initial_noise(
        (batch_size, *cfg.auxiliary_shape),
        history,
        auxiliary_noise,
        generator,
        "auxiliary_noise",
    )
    auxiliary = initial_auxiliary.clone()
    fixed_video = initial_video.clone()
    for step in range(auxiliary_steps):
        t_auxiliary = auxiliary.new_full((batch_size,), step / auxiliary_steps)
        next_t = auxiliary.new_full((batch_size,), (step + 1) / auxiliary_steps)
        prediction = model(
            fixed_video,
            auxiliary,
            fixed_video.new_zeros((batch_size,)),
            t_auxiliary,
            history,
            actions,
            condition_on_auxiliary=True,
        )
        velocity = x_prediction_to_velocity(
            prediction.auxiliary_x,
            auxiliary,
            t_auxiliary,
            eps=velocity_eps,
        )
        auxiliary = clean_time_euler_step(
            auxiliary,
            velocity,
            t_auxiliary,
            next_t,
        )
    return AuxiliarySample(
        generated_auxiliary=auxiliary,
        initial_video_noise=initial_video,
        initial_auxiliary_noise=initial_auxiliary,
        model_calls=auxiliary_steps,
    )


@torch.no_grad()
def sample_video_only(
    model: VideoLatentForcingModel,
    history: Tensor,
    actions: Tensor,
    *,
    video_steps: int = 50,
    generator: torch.Generator | None = None,
    video_noise: Tensor | None = None,
    velocity_eps: float = 5e-2,
) -> VideoOnlySample:
    """Generate B0 video with exactly ``video_steps`` auxiliary-free calls."""
    if video_steps < 1:
        raise ValueError("video_steps must be positive")
    if velocity_eps <= 0:
        raise ValueError("velocity_eps must be positive")
    cfg = model.config
    batch_size = _validate_conditioning(model, history, actions)
    initial_video = _initial_noise(
        (batch_size, *cfg.future_shape),
        history,
        video_noise,
        generator,
        "video_noise",
    )
    video = initial_video.clone()
    for step in range(video_steps):
        t_video = video.new_full((batch_size,), step / video_steps)
        next_t = video.new_full((batch_size,), (step + 1) / video_steps)
        prediction = model(
            video,
            None,
            t_video,
            video.new_zeros((batch_size,)),
            history,
            actions,
            condition_on_auxiliary=False,
        )
        velocity = x_prediction_to_velocity(
            prediction.video_x,
            video,
            t_video,
            eps=velocity_eps,
        )
        video = clean_time_euler_step(video, velocity, t_video, next_t)
    return VideoOnlySample(
        video=video,
        initial_video_noise=initial_video,
        model_calls=video_steps,
    )


@torch.no_grad()
def sample_cascade(
    model: VideoLatentForcingModel,
    history: Tensor,
    actions: Tensor,
    *,
    auxiliary_steps: int = 25,
    video_steps: int = 25,
    auxiliary_condition: AuxiliaryCondition = "generated",
    generator: torch.Generator | None = None,
    video_noise: Tensor | None = None,
    auxiliary_noise: Tensor | None = None,
    velocity_eps: float = 5e-2,
) -> CascadeSample:
    """Run a strict auxiliary-then-video cascade from two noise draws.

    The sampler has no clean-future auxiliary input.  Every video-phase control
    therefore uses the state autonomously generated during phase one.  ``off``
    and ``shuffled`` preserve the same phase-one computation and call budget.
    """
    if auxiliary_steps < 1 or video_steps < 1:
        raise ValueError("auxiliary_steps and video_steps must be positive")
    if velocity_eps <= 0:
        raise ValueError("velocity_eps must be positive")
    batch_size = _validate_conditioning(model, history, actions)
    auxiliary_sample = sample_auxiliary_only(
        model,
        history,
        actions,
        auxiliary_steps=auxiliary_steps,
        generator=generator,
        video_noise=video_noise,
        auxiliary_noise=auxiliary_noise,
        velocity_eps=velocity_eps,
    )
    video = auxiliary_sample.initial_video_noise.clone()
    generated_auxiliary = auxiliary_sample.generated_auxiliary
    conditioning_auxiliary, condition_on_auxiliary = apply_auxiliary_control(
        generated_auxiliary,
        auxiliary_condition,
    )
    frozen_auxiliary = conditioning_auxiliary.clone()

    for step in range(video_steps):
        t_video = video.new_full((batch_size,), step / video_steps)
        next_t = video.new_full((batch_size,), (step + 1) / video_steps)
        t_auxiliary = generated_auxiliary.new_ones((batch_size,))
        prediction = model(
            video,
            frozen_auxiliary,
            t_video,
            t_auxiliary,
            history,
            actions,
            condition_on_auxiliary=condition_on_auxiliary,
        )
        velocity = x_prediction_to_velocity(
            prediction.video_x,
            video,
            t_video,
            eps=velocity_eps,
        )
        video = clean_time_euler_step(video, velocity, t_video, next_t)

    return CascadeSample(
        video=video,
        generated_auxiliary=generated_auxiliary,
        conditioning_auxiliary=frozen_auxiliary,
        initial_video_noise=auxiliary_sample.initial_video_noise,
        initial_auxiliary_noise=auxiliary_sample.initial_auxiliary_noise,
        model_calls=auxiliary_steps + video_steps,
        auxiliary_condition=auxiliary_condition,
    )
