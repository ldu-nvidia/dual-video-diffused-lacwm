"""Wan video continuation with a fixed inference-causal robot-flow field.

The field is a clean deterministic condition, not a second diffusion state. It
is computed before RGB generation and held fixed for every Wan call.  The two
matched training arms differ only in the non-parametric ``fuse_flow`` switch.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from lam.dual_explicit_action_dit_model import DualExplicitActionDiTModel
from robot_wm.modeling.dual_diffusion.physics_flow import (
    FLOW_CHANNELS,
    HISTORY_LATENT_FRAMES,
    LATENT_FRAMES,
    LATENT_HEIGHT,
    LATENT_WIDTH,
    PhysicsFlowError,
    tensor_sha256,
    validate_packed_flow,
)
from robot_wm.modeling.networks.wan_forward_model import DualWanOutput


FLOW_CONDITION_SOURCES = (
    "raw",
    "off",
    "episode_shuffled",
    "timeshift_plus_one",
    "hold_current",
    # Recurrent-delta bridge labels.  These use the identical packed tensor
    # path; keeping the semantic source in the sample receipt prevents a
    # recurrent field from being silently reported as a raw-command field.
    "recurrent_aligned",
    "recurrent_shuffled",
    "recurrent_wrong_time",
    "recurrent_hold",
)
RAW_FLOW_CONDITION_SOURCES = (
    "raw",
    "off",
    "episode_shuffled",
    "timeshift_plus_one",
    "hold_current",
)
RECURRENT_FLOW_CONDITION_SOURCES = (
    "recurrent_aligned",
    "off",
    "recurrent_shuffled",
    "recurrent_wrong_time",
    "recurrent_hold",
)
FlowConditionSource = Literal[
    "raw",
    "off",
    "episode_shuffled",
    "timeshift_plus_one",
    "hold_current",
    "recurrent_aligned",
    "recurrent_shuffled",
    "recurrent_wrong_time",
    "recurrent_hold",
]


@dataclass(frozen=True)
class PhysicsFlowSample:
    video_latent: Tensor
    decoded_future: Tensor
    history_latent: Tensor
    injected_flow: Tensor
    wan_calls: int
    flow_model_calls: int
    history_tokens: int
    condition_source: str
    history_encode_seconds: float
    adapter_and_wan_seconds: float
    decode_seconds: float
    end_to_end_seconds: float


class PhysicsFlowVPM(DualExplicitActionDiTModel):
    """VPM continuation conditioned on an externally rendered causal field."""

    def __init__(
        self,
        *,
        physics_flow: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        config = dict(physics_flow)
        self.fuse_flow = bool(config.get("fuse_flow", False))
        self.condition_family = str(
            config.get("condition_family", "raw_geometry_scaffold")
        )
        if self.condition_family == "raw_geometry_scaffold":
            self.flow_condition_sources = RAW_FLOW_CONDITION_SOURCES
        elif self.condition_family == "sealed_recurrent_delta_geometry":
            self.flow_condition_sources = RECURRENT_FLOW_CONDITION_SOURCES
        else:
            raise PhysicsFlowError("unsupported fixed-flow condition family")
        super().__init__(**kwargs)
        if self.num_history_latent != HISTORY_LATENT_FRAMES:
            raise PhysicsFlowError("physics-flow screen requires two Wan history tokens")
        if int(self.forward_model.tf_token_adapter.tf_channels) != FLOW_CHANNELS:
            raise PhysicsFlowError("physics-flow screen requires a 16-channel adapter")
        if self.tf_loss_weight != 0.0 or bool(self.condition_on_tf_clock):
            raise PhysicsFlowError("fixed flow requires zero auxiliary loss and no clock")
        if self.tf_condition_mode != "off" or bool(self.condition_on_tf):
            raise PhysicsFlowError("physics_flow.fuse_flow owns the only fusion switch")
        if self.auxiliary_history_mode != "diffuse_all":
            raise PhysicsFlowError("fixed flow requires diffuse_all compatibility mode")
        if bool(self.parameter_matched_control) or bool(self.video_only_control):
            raise PhysicsFlowError("legacy auxiliary control modes must be disabled")

    @staticmethod
    def _probe_mean(value: Tensor) -> Tensor:
        flat = value.detach().float().reshape(value.shape[0], -1)
        return flat[:, : min(8, int(flat.shape[1]))].mean()

    @staticmethod
    def _synchronize(value: Tensor) -> None:
        if value.is_cuda:
            torch.cuda.synchronize(value.device)

    @staticmethod
    def _validate_video_geometry(value: Tensor, history_tokens: int) -> None:
        if tuple(value.shape[1:]) != (
            FLOW_CHANNELS,
            LATENT_FRAMES,
            LATENT_HEIGHT,
            LATENT_WIDTH,
        ) or history_tokens != HISTORY_LATENT_FRAMES:
            raise PhysicsFlowError("Wan latent geometry changed")

    @staticmethod
    def _validate_flow(value: Tensor, *, batch_size: int, device: torch.device) -> Tensor:
        validate_packed_flow(value)
        if int(value.shape[0]) != int(batch_size):
            raise PhysicsFlowError("flow/video batch sizes differ")
        if value.device != device:
            raise PhysicsFlowError("flow must already be on the video device")
        return value

    def forward(
        self,
        rgb: Tensor,
        actions: Tensor | None = None,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        if kwargs.get("auxiliary_target") is not None:
            raise PhysicsFlowError("fixed-flow training cannot consume auxiliary targets")
        if kwargs.get("future_measured_state") is not None:
            raise PhysicsFlowError("future measured state is forbidden")
        if actions is None:
            raise ValueError("physics-flow training requires candidate actions")
        condition = kwargs.get("causal_robot_flow")
        if not isinstance(condition, Tensor):
            raise PhysicsFlowError("batch lacks causal_robot_flow")
        condition = self._validate_flow(
            condition, batch_size=int(rgb.shape[0]), device=rgb.device
        ).to(dtype=rgb.dtype)
        morphology_index = kwargs.get("morphology_index")
        absolute_clean = self._encode_clip(rgb).to(rgb.dtype)
        reference, history_tokens = self._history_reference(rgb, absolute_clean.shape)
        self._validate_video_geometry(absolute_clean, history_tokens)
        batch_size = int(rgb.shape[0])
        _, z_control, _ = self._latent_actions(
            rgb,
            actions,
            morphology_index,
            int(absolute_clean.shape[2]),
            history_tokens,
        )
        z_control = z_control.to(rgb.dtype)
        video_noise = torch.randn_like(absolute_clean)
        (
            timesteps,
            video_sigma,
            _tf_sigma,
            video_loss_weight,
            _tf_loss_weight,
            _tf_noise_timesteps,
        ) = self._paired_training_clocks(
            batch_size,
            rgb.device,
            absolute_clean.dtype,
            sample_ids=kwargs.get("clip_index"),
        )
        expanded_sigma = self._expand_sigma(video_sigma, absolute_clean)
        video_noisy = (1.0 - expanded_sigma) * absolute_clean + expanded_sigma * video_noise
        video_target = video_noise - absolute_clean
        context = self._build_context(batch_size, rgb.device, rgb.dtype)
        clip_fea = self._build_clip(batch_size, rgb.device, rgb.dtype)
        zero_flow_sigma = video_sigma.new_zeros(batch_size)
        prediction = self.forward_model(
            video_noisy,
            timesteps,
            z_control,
            reference,
            context,
            clip_fea,
            noisy_tf=condition,
            conditioning_tf=condition,
            tf_sigma=zero_flow_sigma,
            condition_on_tf=self.fuse_flow,
            condition_on_tf_clock=False,
        )
        if not isinstance(prediction, DualWanOutput):
            raise PhysicsFlowError("Wan did not return the dual output schema")
        loss_mask = self._build_loss_mask(rgb, mask, absolute_clean.shape)
        future_mask = loss_mask[:, :, history_tokens:]
        per_sample = self._masked_per_sample_mse(
            prediction.video_velocity[:, :, history_tokens:],
            video_target[:, :, history_tokens:],
            future_mask,
        )
        flow_loss = self._branch_weighted_mean(per_sample, video_loss_weight)
        self.aux_losses = {
            "flow_loss": flow_loss.detach(),
            "video_flow_loss": flow_loss.detach(),
            "physics_flow/fuse_flow": flow_loss.new_tensor(float(self.fuse_flow)),
            "physics_flow/flow_model_calls": flow_loss.new_tensor(0.0),
            "physics_flow/condition_rms": condition.float().square().mean().sqrt().detach(),
            "physics_flow/condition_nonzero_fraction": condition.ne(0).float().mean().detach(),
            "physics_flow/effective_adapter_gate": (
                self.forward_model.tf_token_adapter.effective_gate().detach().float()
            ),
            "physics_flow/fixed_clean_clock": flow_loss.new_tensor(1.0),
            "physics_flow/future_measured_state_conditioned": flow_loss.new_tensor(0.0),
            "paired_audit/timestep_mean": timesteps.float().mean().detach(),
            "paired_audit/timestep_square_mean": timesteps.float().square().mean().detach(),
            "paired_audit/video_noise_probe": self._probe_mean(video_noise),
            "paired_audit/flow_probe": self._probe_mean(condition),
            "paired_audit/action_probe": self._probe_mean(actions),
        }
        clip_index = kwargs.get("clip_index")
        if isinstance(clip_index, Tensor):
            ids = clip_index.detach().float().reshape(-1)
            self.aux_losses["paired_audit/clip_index_mean"] = ids.mean()
            self.aux_losses["paired_audit/clip_index_square_mean"] = ids.square().mean()
        # Exact rank-local hashes are consumed by the fail-closed trainer and
        # deliberately remain outside differentiable state.
        self._physics_flow_step_evidence = {
            "video_noise_sha256": tensor_sha256(video_noise),
            "flow_sha256": tensor_sha256(condition),
            "actions_sha256": tensor_sha256(actions),
            "timesteps_sha256": tensor_sha256(timesteps),
            "clip_index_sha256": (
                tensor_sha256(clip_index)
                if isinstance(clip_index, Tensor)
                else None
            ),
        }
        if not bool(torch.isfinite(flow_loss)):
            raise FloatingPointError("physics-flow video loss is non-finite")
        return flow_loss

    @torch.inference_mode()
    def sample_physics_flow(
        self,
        history_rgb: Tensor,
        actions: Tensor,
        morphology_index: Tensor,
        *,
        video_noise: Tensor,
        flow_condition: Tensor,
        steps: int,
        condition_source: FlowConditionSource,
    ) -> PhysicsFlowSample:
        """Sample from observed history, actions, explicit noise, and fixed flow.

        The API intentionally has no future RGB, clean latent, teacher feature,
        optical-flow target, or future measured-state argument.
        """

        if condition_source not in self.flow_condition_sources:
            raise ValueError(f"unsupported flow source: {condition_source}")
        if history_rgb.ndim != 5 or tuple(history_rgb.shape[1:3]) != (5, 3):
            raise ValueError("sampler requires exactly five observed RGB frames")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        total_start = time.perf_counter()
        self._synchronize(history_rgb)
        started = time.perf_counter()
        history_latents = self._encode_clip(history_rgb).to(history_rgb.dtype)
        expected_video = (
            int(history_rgb.shape[0]),
            FLOW_CHANNELS,
            LATENT_FRAMES,
            LATENT_HEIGHT,
            LATENT_WIDTH,
        )
        self._validate_video_geometry(
            history_latents.new_zeros(expected_video), int(history_latents.shape[2])
        )
        if tuple(video_noise.shape) != expected_video:
            raise PhysicsFlowError("video noise shape differs from frozen Wan state")
        condition = self._validate_flow(
            flow_condition,
            batch_size=int(history_rgb.shape[0]),
            device=history_rgb.device,
        ).to(dtype=history_rgb.dtype)
        if condition_source == "off" and bool(condition.ne(0).any()):
            raise PhysicsFlowError("off condition source requires an exact-zero tensor")
        if not bool(torch.isfinite(video_noise).all()):
            raise PhysicsFlowError("video noise must be finite")
        self._synchronize(history_latents)
        history_encode_seconds = time.perf_counter() - started
        reference = video_noise.new_zeros(expected_video)
        reference[:, :, :HISTORY_LATENT_FRAMES] = history_latents
        _, z_control, _ = self._latent_actions(
            history_rgb,
            actions,
            morphology_index,
            LATENT_FRAMES,
            HISTORY_LATENT_FRAMES,
        )
        z_control = z_control.to(history_rgb.dtype)
        context = self._build_context(
            history_rgb.shape[0], history_rgb.device, history_rgb.dtype
        )
        clip_fea = self._build_clip(
            history_rgb.shape[0], history_rgb.device, history_rgb.dtype
        )
        initial_video_state = video_noise.clone()
        state = initial_video_state.clone()
        schedule, model_timesteps, auxiliary_only_steps = self._sampling_schedule(
            steps, device=history_rgb.device
        )
        timesteps = tuple(model_timesteps)
        if len(timesteps) != steps or auxiliary_only_steps != 0:
            raise PhysicsFlowError("Wan sampling grid differs from declared NFE")
        self._synchronize(state)
        started = time.perf_counter()
        wan_calls = 0
        zero_sigma = state.new_zeros(state.shape[0])
        use_flow = self.fuse_flow and condition_source != "off"
        for step_index, timestep in enumerate(timesteps):
            prediction = self.forward_model(
                state,
                timestep.expand(state.shape[0]).to(state.device),
                z_control,
                reference,
                context,
                clip_fea,
                noisy_tf=condition,
                conditioning_tf=condition,
                tf_sigma=zero_sigma,
                condition_on_tf=use_flow,
                condition_on_tf_clock=False,
            )
            if not isinstance(prediction, DualWanOutput):
                raise PhysicsFlowError("Wan sampler output schema changed")
            state = self.sample_scheduler.step(
                prediction.video_velocity.float(), timestep, state.float()
            ).prev_sample.to(history_rgb.dtype)
            # Preserve the frozen VPM state distribution.  Known history follows
            # its registered forward-noise trajectory between calls and becomes
            # clean only when the native scheduler reaches sigma zero.
            next_sigma = schedule.video[step_index + 1].to(
                device=state.device, dtype=state.dtype
            )
            state[:, :, :HISTORY_LATENT_FRAMES] = (
                (1.0 - next_sigma) * reference[:, :, :HISTORY_LATENT_FRAMES]
                + next_sigma
                * initial_video_state[:, :, :HISTORY_LATENT_FRAMES]
            )
            wan_calls += 1
        self._synchronize(state)
        adapter_and_wan_seconds = time.perf_counter() - started
        started = time.perf_counter()
        decoded = self.rgb_tokenizer.decode_temporal(
            state, out_hw=(int(history_rgb.shape[-2]), int(history_rgb.shape[-1]))
        )
        self._synchronize(decoded)
        decode_seconds = time.perf_counter() - started
        if wan_calls != steps:
            raise PhysicsFlowError("reported Wan calls differ from execution")
        if not all(bool(torch.isfinite(x).all()) for x in (state, decoded, condition)):
            raise FloatingPointError("physics-flow sampler produced non-finite output")
        return PhysicsFlowSample(
            video_latent=state,
            decoded_future=decoded[:, :, -self.num_future_frames :],
            history_latent=history_latents,
            injected_flow=condition,
            wan_calls=wan_calls,
            flow_model_calls=0,
            history_tokens=HISTORY_LATENT_FRAMES,
            condition_source=condition_source,
            history_encode_seconds=history_encode_seconds,
            adapter_and_wan_seconds=adapter_and_wan_seconds,
            decode_seconds=decode_seconds,
            end_to_end_seconds=time.perf_counter() - total_start,
        )
