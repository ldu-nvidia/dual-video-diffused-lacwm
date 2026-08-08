"""Direct cumulative-residual coordinates for the VPM video latent.

This is an adjacent structural baseline, not a dual-diffusion method.  The
network, parameter schema, actions, and deployment inputs are unchanged.  Only
the coordinate system of the two generated Wan tokens changes:

``R_K     = Z_K - Z_{K-1}``
``R_{K+j} = Z_{K+j} - Z_{K+j-1}``, ``j > 0``.

The known history ``R[:K]`` is always the exact history-only VAE encoding.  At
the end of sampling, cumulative summation recovers the ordinary Wan latent
before VAE decoding and every quality metric.  No clean future, teacher, or
auxiliary feature is accepted by the deployment sampler.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from lam.dual_explicit_action_dit_model import DualExplicitActionDiTModel
from robot_wm.modeling.dual_diffusion.flow import (
    euler_flow_step,
    pair_video_sigma_schedule,
)
from robot_wm.modeling.networks.wan_forward_model import DualWanOutput


REPRESENTATION_MODES = ("absolute", "cumulative_residual")
RepresentationMode = Literal["absolute", "cumulative_residual"]


class VideoResidualAnchorError(RuntimeError):
    """The fixed VPM geometry or causal sampling contract changed."""


def _validate_latents(
    value: Tensor,
    reference: Tensor,
    history_tokens: int,
    *,
    label: str,
) -> None:
    if value.ndim != 5 or reference.ndim != 5:
        raise ValueError(f"{label} and reference must be [B,C,T,H,W]")
    if value.shape != reference.shape:
        raise ValueError(f"{label} and reference must have identical shape")
    if (
        isinstance(history_tokens, bool)
        or not isinstance(history_tokens, int)
        or history_tokens < 1
        or history_tokens >= int(value.shape[2])
    ):
        raise ValueError("history_tokens must select a nonempty proper prefix")
    if not value.is_floating_point() or not reference.is_floating_point():
        raise ValueError("video latents must be floating point")
    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(reference).all()):
        raise ValueError("video latents must be finite")


def pack_video_residual_anchor(
    absolute: Tensor,
    reference: Tensor,
    history_tokens: int,
    *,
    mode: RepresentationMode,
) -> Tensor:
    """Map an absolute Wan latent to exact-history/future-generation coordinates.

    ``reference`` is produced only from observed RGB.  Its future slots must be
    zero.  For the absolute control, generated slots remain ordinary absolute
    Wan tokens.  For the residual arm, every generated slot is an increment;
    the first is anchored at the final observed token.
    """

    _validate_latents(absolute, reference, history_tokens, label="absolute latent")
    if mode not in REPRESENTATION_MODES:
        raise ValueError(f"unsupported video representation mode: {mode}")
    if bool(reference[:, :, history_tokens:].ne(0).any()):
        raise ValueError("reference future slots must be exact zero")
    packed = absolute.clone()
    packed[:, :, :history_tokens] = reference[:, :, :history_tokens]
    if mode == "cumulative_residual":
        future = absolute[:, :, history_tokens:]
        predecessor = torch.cat(
            (reference[:, :, history_tokens - 1 : history_tokens], future[:, :, :-1]),
            dim=2,
        )
        packed[:, :, history_tokens:] = future - predecessor
    return packed


def invert_video_residual_anchor(
    representation: Tensor,
    reference: Tensor,
    history_tokens: int,
    *,
    mode: RepresentationMode,
) -> Tensor:
    """Recover ordinary absolute Wan coordinates before decode or scoring."""

    _validate_latents(
        representation, reference, history_tokens, label="video representation"
    )
    if mode not in REPRESENTATION_MODES:
        raise ValueError(f"unsupported video representation mode: {mode}")
    if bool(reference[:, :, history_tokens:].ne(0).any()):
        raise ValueError("reference future slots must be exact zero")
    absolute = representation.clone()
    absolute[:, :, :history_tokens] = reference[:, :, :history_tokens]
    if mode == "cumulative_residual":
        anchor = reference[:, :, history_tokens - 1 : history_tokens]
        absolute[:, :, history_tokens:] = (
            anchor + representation[:, :, history_tokens:].cumsum(dim=2)
        )
    return absolute


@dataclass(frozen=True)
class VideoResidualAnchorSample:
    """One deployable endpoint in both representation and scoring coordinates."""

    representation: Tensor
    absolute_video_latent: Tensor
    decoded_future: Tensor
    auxiliary_state: Tensor
    model_calls: int
    history_tokens: int


class VideoResidualAnchorVPM(DualExplicitActionDiTModel):
    """Parameter-identical VPM trained in absolute or cumulative-residual space."""

    def __init__(
        self,
        *,
        video_residual_anchor: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        config = dict(video_residual_anchor)
        self.video_representation_mode: RepresentationMode = str(
            config.get("mode", "absolute")
        )  # type: ignore[assignment]
        self.video_residual_normalization = str(
            config.get("normalization", "none")
        )
        if self.video_representation_mode not in REPRESENTATION_MODES:
            raise ValueError(
                "video_residual_anchor.mode must be absolute or cumulative_residual"
            )
        if self.video_residual_normalization != "none":
            raise ValueError(
                "the first screen preregisters no data-dependent normalization"
            )
        super().__init__(**kwargs)
        if (
            not bool(getattr(self, "parameter_matched_control", False))
            or self.tf_condition_mode != "off"
            or self.condition_on_tf
            or bool(getattr(self, "condition_on_tf_clock", False))
            or not bool(
                getattr(self.forward_model, "tf_head_condition_on_clock", False)
            )
            or self.tf_loss_weight != 0.0
            or self.auxiliary_history_mode != "diffuse_all"
            or self.tf_schedule_mode != "aligned"
            or self.tf_lead_logit != 0.0
        ):
            raise VideoResidualAnchorError(
                "residual-anchor screen requires the exact historical VPM no-op "
                "auxiliary architecture and schedule"
            )
        if self.num_history_latent != 2:
            raise VideoResidualAnchorError(
                "screen requires five observed RGB frames -> two Wan history tokens"
            )

    def _assert_screen_geometry(self, latent: Tensor, history_tokens: int) -> None:
        if int(latent.shape[2]) != 4 or history_tokens != 2:
            raise VideoResidualAnchorError(
                "screen requires 13 RGB frames -> four Wan tokens (two history, two future)"
            )

    def _zero_auxiliary(self, video: Tensor) -> Tensor:
        channels = int(self.forward_model.tf_token_adapter.tf_channels)
        return video.new_zeros(video.shape[0], channels, *video.shape[2:])

    @staticmethod
    def _probe_mean(value: Tensor) -> Tensor:
        flat = value.detach().float().reshape(value.shape[0], -1)
        return flat[:, : min(8, int(flat.shape[1]))].mean()

    def forward(
        self,
        rgb: Tensor,
        actions: Tensor | None = None,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Train only the future video representation; history stays observable."""

        self._ensure_video_only_runtime_contract()
        if kwargs.get("auxiliary_target") is not None:
            raise VideoResidualAnchorError(
                "residual-anchor training must not consume an auxiliary target"
            )
        self.aux_losses = {}
        morphology_index = kwargs.get("morphology_index")
        device = rgb.device
        absolute_clean = self._encode_clip(rgb).to(rgb.dtype)
        reference, history_tokens = self._history_reference(rgb, absolute_clean.shape)
        self._assert_screen_geometry(absolute_clean, history_tokens)
        representation_clean = pack_video_residual_anchor(
            absolute_clean,
            reference,
            history_tokens,
            mode=self.video_representation_mode,
        )
        batch_size = int(rgb.shape[0])
        _, z_control, _ = self._latent_actions(
            rgb,
            actions,
            morphology_index,
            int(absolute_clean.shape[2]),
            history_tokens,
        )
        z_control = z_control.to(rgb.dtype)

        # Both arms execute this random path in the same order.  Only the clean
        # coordinate tensor differs.  Known history is clamped exactly after
        # corruption and has zero loss weight.
        video_noise = torch.randn_like(representation_clean)
        (
            timesteps,
            video_sigma,
            tf_sigma,
            video_loss_weight,
            _tf_loss_weight,
            _tf_noise_timesteps,
        ) = self._paired_training_clocks(
            batch_size,
            device,
            representation_clean.dtype,
            sample_ids=kwargs.get("clip_index"),
        )
        sigma = self._expand_sigma(video_sigma, representation_clean)
        video_noisy = (
            (1.0 - sigma) * representation_clean + sigma * video_noise
        )
        video_noisy = video_noisy.clone()
        video_noisy[:, :, :history_tokens] = reference[:, :, :history_tokens]
        video_target = video_noise - representation_clean

        tf_state = self._zero_auxiliary(video_noisy)
        context = self._build_context(batch_size, device, rgb.dtype)
        clip_fea = self._build_clip(batch_size, device, rgb.dtype)
        prediction = self.forward_model(
            video_noisy,
            timesteps,
            z_control,
            reference,
            context,
            clip_fea,
            noisy_tf=tf_state,
            conditioning_tf=tf_state,
            tf_sigma=tf_sigma,
            condition_on_tf=False,
            condition_on_tf_clock=False,
        )
        if not isinstance(prediction, DualWanOutput):
            raise VideoResidualAnchorError(
                "parameter-matched VPM did not return both velocity heads"
            )

        loss_mask = self._build_loss_mask(rgb, mask, absolute_clean.shape)
        future_mask = loss_mask[:, :, history_tokens:]
        per_sample = self._masked_per_sample_mse(
            prediction.video_velocity[:, :, history_tokens:],
            video_target[:, :, history_tokens:],
            future_mask,
        )
        flow_loss = self._branch_weighted_mean(per_sample, video_loss_weight)
        # Diagnostic only: do not retain a second x0/inverse graph through
        # backward merely to report teacher-forced telemetry.
        with torch.no_grad():
            predicted_representation = (
                video_noisy - sigma * prediction.video_velocity
            )
            predicted_absolute = invert_video_residual_anchor(
                predicted_representation,
                reference,
                history_tokens,
                mode=self.video_representation_mode,
            )
            latent_nmse = self._masked_per_sample_nmse(
                predicted_absolute[:, :, history_tokens:],
                absolute_clean[:, :, history_tokens:],
                future_mask,
            )

        self.aux_losses["flow_loss"] = flow_loss.detach()
        self.aux_losses["video_flow_loss"] = flow_loss.detach()
        self.aux_losses["residual_anchor/mode_code"] = flow_loss.new_tensor(
            0.0 if self.video_representation_mode == "absolute" else 1.0
        )
        self.aux_losses["residual_anchor/history_exact"] = flow_loss.new_tensor(1.0)
        self.aux_losses["residual_anchor/history_tokens"] = flow_loss.new_tensor(
            float(history_tokens)
        )
        self.aux_losses["residual_anchor/future_tokens"] = flow_loss.new_tensor(
            float(absolute_clean.shape[2] - history_tokens)
        )
        self.aux_losses["residual_anchor/normalization_none"] = flow_loss.new_tensor(
            1.0
        )
        self.aux_losses["teacher_forced/video_x0_nmse"] = latent_nmse.mean().detach()
        self.aux_losses["paired_audit/timestep_mean"] = timesteps.float().mean().detach()
        self.aux_losses["paired_audit/timestep_square_mean"] = (
            timesteps.float().square().mean().detach()
        )
        self.aux_losses["paired_audit/noise_probe"] = self._probe_mean(video_noise)
        clip_index = kwargs.get("clip_index")
        if isinstance(clip_index, Tensor):
            ids = clip_index.detach().float().reshape(-1)
            self.aux_losses["paired_audit/clip_index_mean"] = ids.mean()
            self.aux_losses["paired_audit/clip_index_square_mean"] = ids.square().mean()
        if not bool(torch.isfinite(flow_loss).all()):
            raise FloatingPointError("video residual-anchor loss is non-finite")
        return flow_loss

    @torch.inference_mode()
    def sample_video_residual_anchor(
        self,
        history_rgb: Tensor,
        actions: Tensor,
        morphology_index: Tensor | None,
        *,
        video_noise: Tensor,
        auxiliary_noise: Tensor,
        steps: int,
    ) -> VideoResidualAnchorSample:
        """Generate from observables and explicit noise, then invert before decode.

        The signature intentionally has no clean-future, target, teacher, or
        feature argument.  Evaluators own ground truth and construct it only
        after all endpoint sampling for a batch has finished.
        """

        self._ensure_video_only_runtime_contract()
        if history_rgb.ndim != 5 or int(history_rgb.shape[1]) != self.num_history_frames:
            raise ValueError(
                f"sampler requires exactly {self.num_history_frames} observed RGB frames"
            )
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        batch_size = int(history_rgb.shape[0])
        history_latents = self._encode_clip(history_rgb).to(history_rgb.dtype)
        history_tokens = int(history_latents.shape[2])
        latent_tokens = int(
            self.rgb_tokenizer.latent_temporal_len(
                self.num_history_frames + self.num_future_frames
            )
        )
        shape = (
            batch_size,
            int(history_latents.shape[1]),
            latent_tokens,
            int(history_latents.shape[3]),
            int(history_latents.shape[4]),
        )
        if tuple(video_noise.shape) != shape:
            raise ValueError(
                f"video_noise shape {tuple(video_noise.shape)} != expected {shape}"
            )
        auxiliary_shape = (
            batch_size,
            int(self.forward_model.tf_token_adapter.tf_channels),
            latent_tokens,
            shape[3],
            shape[4],
        )
        if tuple(auxiliary_noise.shape) != auxiliary_shape:
            raise ValueError(
                "auxiliary_noise shape differs from the parameter-matched VPM state"
            )
        if not bool(torch.isfinite(video_noise).all()) or not bool(
            torch.isfinite(auxiliary_noise).all()
        ):
            raise ValueError("sampling noise must be finite")

        reference = history_latents.new_zeros(shape)
        reference[:, :, :history_tokens] = history_latents
        self._assert_screen_geometry(reference, history_tokens)
        _, z_control, _ = self._latent_actions(
            history_rgb,
            actions,
            morphology_index,
            latent_tokens,
            history_tokens,
        )
        z_control = z_control.to(history_rgb.dtype)
        context = self._build_context(batch_size, history_rgb.device, history_rgb.dtype)
        clip_fea = self._build_clip(batch_size, history_rgb.device, history_rgb.dtype)
        representation = video_noise.clone()
        representation[:, :, :history_tokens] = reference[:, :, :history_tokens]
        auxiliary = auxiliary_noise.clone()

        self.sample_scheduler.set_timesteps(steps, device=history_rgb.device)
        timesteps = tuple(self.sample_scheduler.timesteps)
        sigmas = self.sample_scheduler.sigmas.to(
            device=history_rgb.device, dtype=torch.float32
        )[: steps + 1]
        schedule = pair_video_sigma_schedule(
            sigmas,
            mode=self.tf_schedule_mode,
            tf_lead_logit=self.tf_lead_logit,
        )
        if len(timesteps) != steps or schedule.num_steps != steps:
            raise VideoResidualAnchorError("scheduler call grid differs from NFE")

        calls = 0
        for index, timestep in enumerate(timesteps):
            tf_sigma = schedule.time_frequency[index]
            next_tf_sigma = schedule.time_frequency[index + 1]
            tf_batch_sigma = tf_sigma.expand(batch_size).to(
                device=history_rgb.device, dtype=history_rgb.dtype
            )
            prediction = self.forward_model(
                representation,
                timestep.expand(batch_size).to(history_rgb.device),
                z_control,
                reference,
                context,
                clip_fea,
                noisy_tf=auxiliary,
                conditioning_tf=auxiliary,
                tf_sigma=tf_batch_sigma,
                condition_on_tf=False,
                condition_on_tf_clock=False,
            )
            calls += 1
            if not isinstance(prediction, DualWanOutput):
                raise VideoResidualAnchorError("VPM sampler output schema changed")
            representation = self.sample_scheduler.step(
                prediction.video_velocity.float(),
                timestep,
                representation.float(),
            ).prev_sample.to(history_rgb.dtype)
            # History is observable state, not a generated coordinate.
            representation[:, :, :history_tokens] = reference[:, :, :history_tokens]
            auxiliary = euler_flow_step(
                auxiliary.float(),
                prediction.tf_velocity.float(),
                tf_sigma,
                next_tf_sigma,
            ).to(history_rgb.dtype)

        absolute = invert_video_residual_anchor(
            representation,
            reference,
            history_tokens,
            mode=self.video_representation_mode,
        )
        decoded = self.rgb_tokenizer.decode_temporal(
            absolute, out_hw=(int(history_rgb.shape[-2]), int(history_rgb.shape[-1]))
        )
        if calls != steps or not all(
            bool(torch.isfinite(value).all())
            for value in (representation, absolute, decoded, auxiliary)
        ):
            raise VideoResidualAnchorError("sampler call count or finite-state check failed")
        return VideoResidualAnchorSample(
            representation=representation,
            absolute_video_latent=absolute,
            decoded_future=decoded[:, :, -self.num_future_frames :],
            auxiliary_state=auxiliary,
            model_calls=calls,
            history_tokens=history_tokens,
        )
