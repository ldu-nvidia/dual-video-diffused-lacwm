"""LaMo-style macro motion-drift supervision for the frozen VPM geometry.

This module deliberately adds no trainable parameter and does not override the
deployment sampler.  It only augments the ordinary video rectified-flow loss
with a scale-normalized, spatially averaged future-latent drift loss computed
from the same predicted clean state that the video-flow objective trains.

LACWM uses ``sigma=1`` for noise and ``sigma=0`` for clean data:

``x_sigma = (1 - sigma) * x0 + sigma * epsilon``
``v_target = epsilon - x0``
``x0_hat = x_sigma - sigma * v_theta``

LaMo's DDPM signal-power weight ``alpha_bar`` therefore becomes
``(1 - sigma) ** 2`` in this rectified-flow parameterization.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from lam.dual_explicit_action_dit_model import DualExplicitActionDiTModel
from robot_wm.modeling.networks.wan_forward_model import DualWanOutput


class LamoMotionDriftError(RuntimeError):
    """A model or batch violated the preregistered motion-drift contract."""


def rf_predicted_clean(noisy: Tensor, sigma: Tensor, velocity: Tensor) -> Tensor:
    """Convert an RF velocity prediction to ``x0`` under LACWM's clock."""

    if noisy.shape != velocity.shape or noisy.ndim < 2:
        raise ValueError("noisy state and velocity must share a batched shape")
    if sigma.ndim != 1 or int(sigma.shape[0]) != int(noisy.shape[0]):
        raise ValueError("sigma must have shape [B]")
    expanded = sigma.to(device=noisy.device, dtype=torch.float32).reshape(
        -1, *([1] * (noisy.ndim - 1))
    )
    return noisy.float() - expanded * velocity.float()


def rf_lamo_schedule_weight(sigma: Tensor) -> Tensor:
    """Return LaMo Eq. 6's batch-mean RF signal-power analogue."""

    if sigma.ndim != 1 or sigma.numel() == 0:
        raise ValueError("sigma must be a nonempty [B] tensor")
    if not bool(torch.isfinite(sigma).all()) or bool(
        ((sigma < 0) | (sigma > 1)).any()
    ):
        raise ValueError("sigma values must be finite in [0,1]")
    return (1.0 - sigma.float()).square().mean()


def global_rf_lamo_schedule_weight(sigma: Tensor) -> Tensor:
    """Use the effective DDP batch, not a local batch-one approximation."""

    local = rf_lamo_schedule_weight(sigma)
    if not torch.distributed.is_initialized():
        return local
    packed = torch.stack(
        (
            local * float(sigma.numel()),
            local.new_tensor(float(sigma.numel())),
        )
    )
    torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
    if not bool(torch.isfinite(packed).all()) or packed[1].item() <= 0:
        raise LamoMotionDriftError("global RF schedule-weight reduction is invalid")
    return packed[0] / packed[1]


def tensor_sha256(value: Tensor) -> str:
    """Hash exact tensor bytes together with dtype and shape for paired audits."""

    detached = value.detach().contiguous()
    header = f"{detached.dtype}|{tuple(detached.shape)}|".encode("ascii")
    raw = detached.view(torch.uint8).cpu().numpy().tobytes(order="C")
    return hashlib.sha256(header + raw).hexdigest()


def macro_future_drift_loss(
    predicted_x0: Tensor,
    clean_x0: Tensor,
    *,
    history_tokens: int,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return LaMo Eq. 5 for the one valid VPM future-token pair.

    Inputs use Wan layout ``[B,C,T,H,W]``.  The fixed 13-frame video geometry
    produces four Wan tokens: two observed-history tokens followed by two
    future tokens.  Consequently there is exactly one future-to-future
    adjacent difference and ``tau=1`` is the only estimable lag that does not
    cross the observed/predicted boundary.
    """

    if predicted_x0.ndim != 5 or clean_x0.ndim != 5:
        raise ValueError("predicted and clean latents must be [B,C,T,H,W]")
    if predicted_x0.shape != clean_x0.shape:
        raise ValueError("predicted and clean latent shapes must match")
    if (
        isinstance(history_tokens, bool)
        or not isinstance(history_tokens, int)
        or history_tokens < 1
    ):
        raise ValueError("history_tokens must be a positive integer")
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    future_tokens = int(predicted_x0.shape[2]) - history_tokens
    if future_tokens != 2:
        raise LamoMotionDriftError(
            "the preregistered VPM geometry requires exactly two future Wan "
            f"tokens, found {future_tokens}"
        )

    predicted_future = predicted_x0[:, :, history_tokens:].float()
    clean_future = clean_x0[:, :, history_tokens:].float()
    predicted_delta = predicted_future[:, :, 1] - predicted_future[:, :, 0]
    target_delta = clean_future[:, :, 1] - clean_future[:, :, 0]
    predicted_macro = predicted_delta.mean(dim=(-2, -1))
    target_macro = target_delta.mean(dim=(-2, -1))
    numerator = (predicted_macro - target_macro).square().sum(dim=1)
    denominator = target_macro.square().sum(dim=1).add(float(epsilon)).detach()
    if not bool(torch.isfinite(denominator).all()) or bool((denominator <= 0).any()):
        raise LamoMotionDriftError("macro-drift normalization is invalid")
    per_sample = numerator / denominator
    if not bool(torch.isfinite(per_sample).all()):
        raise LamoMotionDriftError("macro-drift loss is non-finite")
    return per_sample.mean(), predicted_macro, target_macro


class LamoMotionDriftVPM(DualExplicitActionDiTModel):
    """Parameter-identical VPM with an optional training-only macro drift loss."""

    def __init__(self, *, motion_drift: Mapping[str, Any], **kwargs: Any) -> None:
        config = dict(motion_drift)
        self.motion_drift_weight = float(config.get("weight", 0.0))
        self.motion_drift_epsilon = float(config.get("epsilon", 1e-6))
        self.motion_drift_tau = int(config.get("tau", 1))
        if not math.isfinite(self.motion_drift_weight) or self.motion_drift_weight < 0:
            raise ValueError("motion_drift.weight must be finite and nonnegative")
        if (
            not math.isfinite(self.motion_drift_epsilon)
            or self.motion_drift_epsilon <= 0
        ):
            raise ValueError("motion_drift.epsilon must be finite and positive")
        if self.motion_drift_tau != 1:
            raise ValueError(
                "VPM has only two future Wan tokens; the screen fixes tau=1"
            )
        super().__init__(**kwargs)
        if (
            not bool(getattr(self, "parameter_matched_control", False))
            or self.tf_condition_mode != "off"
            or self.condition_on_tf
            or bool(getattr(self, "condition_on_tf_clock", False))
            or self.tf_loss_weight != 0.0
        ):
            raise LamoMotionDriftError(
                "motion-drift screen requires the exact VPM no-op auxiliary arm"
            )

        # Hooks observe the already-existing VPM call; they do not add a model
        # invocation or alter its input/output interface.  The capture flag is
        # enabled only around ``super().forward`` below.
        self._lamo_capture_active = False
        self._lamo_clean_full: Tensor | None = None
        self._lamo_video_noisy: Tensor | None = None
        self._lamo_timesteps: Tensor | None = None
        self._lamo_prediction: DualWanOutput | None = None
        self.paired_audit_exact: dict[str, str] = {}
        self._lamo_pre_hook_handle = self.forward_model.register_forward_pre_hook(
            self._capture_forward_input, with_kwargs=True
        )
        self._lamo_post_hook_handle = self.forward_model.register_forward_hook(
            self._capture_forward_output
        )

    def _resolve_auxiliary_clean(
        self,
        rgb: Tensor,
        latent_shape: tuple[int, ...],
        auxiliary_target: Tensor | None,
    ) -> Tensor:
        """Provide an inert state for the inherited parameter-matched VPM head.

        The historical VPM snapshot retains the unused 64-channel auxiliary
        head solely to match the parameter topology of the V-JEPA arms.  This
        screen is deliberately RGB/action-only: neither arm may open the old
        V-JEPA target cache or construct another clean feature target.  Because
        state/clock injection and the auxiliary objective are all exact no-ops,
        an all-zero placeholder preserves the inherited parameter/I/O schema
        without supplying any future information to the video prediction.
        """

        if auxiliary_target is not None:
            raise LamoMotionDriftError(
                "the motion-drift screen forbids cached auxiliary targets"
            )
        if len(latent_shape) != 5 or int(latent_shape[0]) != int(rgb.shape[0]):
            raise LamoMotionDriftError("invalid video latent geometry")
        channels = int(self.forward_model.tf_token_adapter.tf_channels)
        return rgb.new_zeros(
            int(latent_shape[0]),
            channels,
            *tuple(int(value) for value in latent_shape[2:]),
        )

    def _encode_clip(self, rgb: Tensor) -> Tensor:
        encoded = super()._encode_clip(rgb)
        expected_rgb_frames = self.num_history_frames + self.num_future_frames
        if self._lamo_capture_active and int(rgb.shape[1]) == expected_rgb_frames:
            if self._lamo_clean_full is not None:
                raise LamoMotionDriftError(
                    "full clean video was encoded more than once in one forward"
                )
            self._lamo_clean_full = encoded
        return encoded

    def _capture_forward_input(
        self,
        _module: Any,
        args: tuple[Any, ...],
        _kwargs: Mapping[str, Any],
    ) -> None:
        if not self._lamo_capture_active:
            return
        if len(args) < 2 or not isinstance(args[0], Tensor) or not isinstance(args[1], Tensor):
            raise LamoMotionDriftError("cannot capture VPM noisy state/timestep")
        if self._lamo_video_noisy is not None or self._lamo_timesteps is not None:
            raise LamoMotionDriftError("VPM forward was called more than once")
        self._lamo_video_noisy = args[0]
        self._lamo_timesteps = args[1]

    def _capture_forward_output(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not self._lamo_capture_active:
            return
        if not isinstance(output, DualWanOutput):
            raise LamoMotionDriftError("VPM did not return DualWanOutput")
        if self._lamo_prediction is not None:
            raise LamoMotionDriftError("VPM produced more than one captured output")
        self._lamo_prediction = output

    def _reset_capture(self) -> None:
        self._lamo_clean_full = None
        self._lamo_video_noisy = None
        self._lamo_timesteps = None
        self._lamo_prediction = None

    @staticmethod
    def _probe_mean(value: Tensor) -> Tensor:
        """Small deterministic audit projection without a full tensor reduction."""

        flat = value.detach().float().reshape(value.shape[0], -1)
        width = min(8, int(flat.shape[1]))
        return flat[:, :width].mean()

    def forward(
        self,
        rgb: Tensor,
        actions: Tensor | None = None,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        self._reset_capture()
        self._lamo_capture_active = True
        try:
            base_loss = super().forward(rgb, actions=actions, mask=mask, **kwargs)
        finally:
            self._lamo_capture_active = False

        clean = self._lamo_clean_full
        noisy = self._lamo_video_noisy
        timesteps = self._lamo_timesteps
        prediction = self._lamo_prediction
        if (
            clean is None
            or noisy is None
            or timesteps is None
            or prediction is None
        ):
            raise LamoMotionDriftError("motion-drift capture is incomplete")
        if tuple(clean.shape) != tuple(noisy.shape):
            raise LamoMotionDriftError("clean/noisy video geometry differs")
        if int(clean.shape[2]) != 4 or self.num_history_latent != 2:
            raise LamoMotionDriftError(
                "expected 13 RGB frames -> 4 Wan tokens and 5 history RGB -> 2 tokens"
            )
        if mask is not None:
            future_mask = mask[:, self.num_history_frames : self.num_history_frames + self.num_future_frames]
            if future_mask.numel() == 0 or not bool(future_mask.bool().all()):
                raise LamoMotionDriftError(
                    "the controlled screen requires both future Wan tokens to be valid"
                )

        sigma = self._get_sigmas(
            timesteps,
            n_dim=2,
            dtype=torch.float32,
            device=noisy.device,
        )[:, 0]
        predicted_x0 = rf_predicted_clean(
            noisy, sigma, prediction.video_velocity
        )
        drift_loss, predicted_macro, target_macro = macro_future_drift_loss(
            predicted_x0,
            clean,
            history_tokens=self.num_history_latent,
            epsilon=self.motion_drift_epsilon,
        )
        # LaMo Eq. 6 uses batch-mean alpha_bar.  In this RF interpolation the
        # clean-signal coefficient is (1-sigma), hence alpha_bar_RF=(1-sigma)^2.
        schedule_weight = global_rf_lamo_schedule_weight(sigma)
        weighted_drift = (
            self.motion_drift_weight * schedule_weight * drift_loss
        )

        self.aux_losses["motion_drift/raw_loss"] = drift_loss.detach()
        self.aux_losses["motion_drift/schedule_weight"] = schedule_weight.detach()
        self.aux_losses["motion_drift/weighted_loss"] = weighted_drift.detach()
        self.aux_losses["motion_drift/lambda"] = base_loss.new_tensor(
            self.motion_drift_weight
        )
        self.aux_losses["motion_drift/epsilon"] = base_loss.new_tensor(
            self.motion_drift_epsilon
        )
        self.aux_losses["motion_drift/tau"] = base_loss.new_tensor(1.0)
        self.aux_losses["motion_drift/future_token_count"] = base_loss.new_tensor(2.0)
        self.aux_losses["motion_drift/predicted_macro_rms"] = (
            predicted_macro.detach().square().mean().sqrt()
        )
        self.aux_losses["motion_drift/target_macro_rms"] = (
            target_macro.detach().square().mean().sqrt()
        )

        # These RNG-free projections let the workflow verify that the two arms
        # saw the same clip order, sampled clocks, and corruption stream.
        clip_index = kwargs.get("clip_index")
        if isinstance(clip_index, Tensor):
            identifiers = clip_index.detach().float().reshape(-1)
            self.aux_losses["paired_audit/clip_index_mean"] = identifiers.mean()
            self.aux_losses["paired_audit/clip_index_square_mean"] = (
                identifiers.square().mean()
            )
        self.aux_losses["paired_audit/timestep_mean"] = (
            timesteps.detach().float().mean()
        )
        self.aux_losses["paired_audit/timestep_square_mean"] = (
            timesteps.detach().float().square().mean()
        )
        self.aux_losses["paired_audit/noisy_probe"] = self._probe_mean(noisy)
        self.aux_losses["paired_audit/clean_probe"] = self._probe_mean(clean)
        clip_tensor = (
            clip_index
            if isinstance(clip_index, Tensor)
            else torch.empty(0, dtype=torch.long, device=noisy.device)
        )
        action_tensor = (
            actions
            if isinstance(actions, Tensor)
            else torch.empty(0, dtype=noisy.dtype, device=noisy.device)
        )
        cuda_rng = (
            torch.cuda.get_rng_state(noisy.device)
            if noisy.is_cuda
            else torch.empty(0, dtype=torch.uint8)
        )
        self.paired_audit_exact = {
            "clip_index": tensor_sha256(clip_tensor),
            "actions": tensor_sha256(action_tensor),
            "clean_latent": tensor_sha256(clean),
            "noisy_latent": tensor_sha256(noisy),
            "timesteps": tensor_sha256(timesteps),
            "cpu_rng_state_after_forward": tensor_sha256(torch.get_rng_state()),
            "cuda_rng_state_after_forward": tensor_sha256(cuda_rng),
        }

        # Exact no-op for the control: no zero-weight auxiliary graph is
        # attached and the ordinary VPM loss object is returned unchanged.
        if self.motion_drift_weight == 0.0:
            self._reset_capture()
            return base_loss
        total = base_loss + weighted_drift.to(base_loss.dtype)
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError("motion-drift training objective is non-finite")
        self._reset_capture()
        return total
