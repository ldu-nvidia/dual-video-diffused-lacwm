"""Feature-free adjacent consistency distillation for the causal VPM.

This experiment-local subclass leaves production Wan/VPM sampling untouched.
Training binds two non-serialized full-model copies: an immutable parent
teacher and a frozen EMA target student.  Deployment uses only this module's
student parameters through :meth:`sample_consistency_deployable`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from lam.dual_explicit_action_dit_model import DualExplicitActionDiTModel
from robot_wm.modeling.low_nfe.adjacent_consistency import (
    AdjacentClockPair,
    AdjacentConsistencyError,
    adjacent_clock_pair,
    consistency_output,
    masked_pseudo_huber,
    rectified_flow_euler_step,
    rectified_flow_noisy_state,
    restore_history_forward_path,
    state_probe,
    tensor_sha256,
    validate_shift5_rf_clock,
)
from robot_wm.modeling.networks.wan_forward_model import DualWanOutput


ARM_MODES = ("rf_control", "consistency")
ArmMode = Literal["rf_control", "consistency"]
SamplerMode = Literal["native_rf", "consistency"]


@dataclass(frozen=True)
class AdjacentConsistencySample:
    """One feature-free consistency-sampler endpoint."""

    absolute_video_latent: Tensor
    decoded_future: Tensor
    model_calls: int
    teacher_calls: int
    feature_calls: int
    sigma_nodes: Tensor
    sampler_mode: SamplerMode


class AdjacentConsistencyVPM(DualExplicitActionDiTModel):
    """Parameter-identical VPM with teacher-step/EMA consistency training."""

    def __init__(
        self,
        *,
        adjacent_consistency: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        config = dict(adjacent_consistency)
        self.acd_arm_mode: ArmMode = str(
            config.get("arm_mode", "rf_control")
        )  # type: ignore[assignment]
        self.acd_stride = int(config.get("stride", 500))
        self.acd_sigma_data = float(config.get("sigma_data", 0.5))
        self.acd_huber_c = float(config.get("huber_c", 0.001))
        self.acd_ema_decay = float(config.get("ema_decay", 0.995))
        self.acd_protocol_version = str(
            config.get("protocol_version", "acd-p0-v1")
        )
        if self.acd_arm_mode not in ARM_MODES:
            raise ValueError(f"arm_mode must be one of {ARM_MODES}")
        if self.acd_stride != 500:
            raise ValueError("ACD-P0 fixes the Flash-WAM video stride to 500")
        if self.acd_sigma_data != 0.5:
            raise ValueError("ACD-P0 fixes sigma_data=0.5")
        if self.acd_huber_c != 0.001:
            raise ValueError("ACD-P0 fixes pseudo-Huber c=0.001")
        if self.acd_ema_decay != 0.995:
            raise ValueError("ACD-P0 fixes EMA decay=0.995")
        if self.acd_protocol_version != "acd-p0-v1":
            raise ValueError("unsupported adjacent-consistency protocol")
        super().__init__(**kwargs)
        self.acd_clock_receipt = validate_shift5_rf_clock(self.noise_scheduler)
        if (
            not bool(getattr(self, "parameter_matched_control", False))
            or self.tf_condition_mode != "off"
            or self.condition_on_tf
            or bool(getattr(self, "condition_on_tf_clock", False))
            or self.tf_loss_weight != 0.0
            or self.auxiliary_history_mode != "diffuse_all"
            or self.tf_schedule_mode != "aligned"
            or self.tf_lead_logit != 0.0
        ):
            raise AdjacentConsistencyError(
                "ACD-P0 requires the exact faithful VPM no-op auxiliary topology"
            )
        if self.num_history_latent != 2:
            raise AdjacentConsistencyError(
                "ACD-P0 requires five RGB history frames -> two Wan tokens"
            )
        object.__setattr__(self, "_acd_teacher_model", None)
        object.__setattr__(self, "_acd_target_model", None)
        self.paired_audit_exact: dict[str, str] = {}
        self._acd_call_counts: dict[str, int] = {}
        self._acd_rng_before: dict[str, tuple[str, str]] = {}

    def bind_frozen_models(
        self,
        *,
        teacher: "AdjacentConsistencyVPM",
        target: "AdjacentConsistencyVPM",
    ) -> None:
        """Bind external, non-serialized teacher and EMA target copies."""

        if teacher is self or target is self or teacher is target:
            raise ValueError("student, teacher, and EMA target must be distinct")
        for name, model in (("teacher", teacher), ("target", target)):
            if not isinstance(model, AdjacentConsistencyVPM):
                raise TypeError(f"{name} must be AdjacentConsistencyVPM")
            if model.acd_arm_mode != self.acd_arm_mode:
                raise AdjacentConsistencyError(f"{name} arm identity differs")
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise AdjacentConsistencyError(f"{name} must be frozen before binding")
            model.eval()
        student_state = self.state_dict()
        for name, model in (("teacher", teacher), ("target", target)):
            model_state = model.state_dict()
            if student_state.keys() != model_state.keys():
                raise AdjacentConsistencyError(f"{name} state keys differ")
            mismatched = [
                key
                for key in student_state
                if student_state[key].shape != model_state[key].shape
                or student_state[key].dtype != model_state[key].dtype
            ]
            if mismatched:
                raise AdjacentConsistencyError(
                    f"{name} state schema differs: {mismatched[:8]}"
                )
        object.__setattr__(self, "_acd_teacher_model", teacher)
        object.__setattr__(self, "_acd_target_model", target)

    def bound_target_model(self) -> "AdjacentConsistencyVPM":
        target = object.__getattribute__(self, "_acd_target_model")
        if not isinstance(target, AdjacentConsistencyVPM):
            raise AdjacentConsistencyError("EMA target model is not bound")
        return target

    def bound_teacher_model(self) -> "AdjacentConsistencyVPM":
        teacher = object.__getattribute__(self, "_acd_teacher_model")
        if not isinstance(teacher, AdjacentConsistencyVPM):
            raise AdjacentConsistencyError("frozen teacher model is not bound")
        return teacher

    def _conditions_for(
        self,
        model: "AdjacentConsistencyVPM",
        history_rgb: Tensor,
        actions: Tensor,
        morphology_index: Tensor | None,
        *,
        latent_frames: int,
        history_tokens: int,
    ) -> tuple[Tensor, Any, Tensor]:
        # ExplicitActionDiTModel does not inspect RGB in _latent_actions, but
        # pass only observed history so future RGB is structurally unavailable.
        _, z_control, _ = model._latent_actions(
            history_rgb,
            actions,
            morphology_index,
            latent_frames,
            history_tokens,
        )
        z_control = z_control.to(history_rgb.dtype)
        context = model._build_context(
            int(history_rgb.shape[0]), history_rgb.device, history_rgb.dtype
        )
        clip_fea = model._build_clip(
            int(history_rgb.shape[0]), history_rgb.device, history_rgb.dtype
        )
        return z_control, context, clip_fea

    @staticmethod
    def _zero_auxiliary(model: "AdjacentConsistencyVPM", state: Tensor) -> Tensor:
        channels = int(model.forward_model.tf_token_adapter.tf_channels)
        return state.new_zeros(state.shape[0], channels, *state.shape[2:])

    def _velocity_call(
        self,
        model: "AdjacentConsistencyVPM",
        state: Tensor,
        timestep: Tensor,
        sigma: Tensor,
        reference: Tensor,
        conditions: tuple[Tensor, Any, Tensor],
        *,
        role: str,
    ) -> Tensor:
        call_index = self._acd_call_counts.get(role, 0)
        ledger_role = f"{role}:{call_index}"
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state(state.device)
            if state.is_cuda
            else torch.empty(0, dtype=torch.uint8)
        )
        self._acd_rng_before[ledger_role] = (
            tensor_sha256(cpu_rng),
            tensor_sha256(cuda_rng),
        )
        z_control, context, clip_fea = conditions
        inert = self._zero_auxiliary(model, state)
        prediction = model.forward_model(
            state,
            timestep,
            z_control,
            reference,
            context,
            clip_fea,
            noisy_tf=inert,
            conditioning_tf=inert,
            tf_sigma=sigma,
            condition_on_tf=False,
            condition_on_tf_clock=False,
        )
        self._acd_call_counts[role] = self._acd_call_counts.get(role, 0) + 1
        if not isinstance(prediction, DualWanOutput):
            raise AdjacentConsistencyError(f"{role} Wan output schema changed")
        velocity = prediction.video_velocity
        if velocity.shape != state.shape or not bool(torch.isfinite(velocity).all()):
            raise AdjacentConsistencyError(f"{role} velocity is invalid")
        return velocity

    def _sample_training_clock(
        self, device: torch.device
    ) -> tuple[AdjacentClockPair, tuple[int, ...]]:
        schedule_t = self.noise_scheduler.timesteps.to(device=device)
        schedule_s = self.noise_scheduler.sigmas.to(
            device=device, dtype=torch.float32
        )[: schedule_t.numel()]
        draws: list[int] = []
        while True:
            sampled = self._sample_timesteps(1, device)
            membership = (schedule_t == sampled[0]).nonzero().flatten()
            if int(membership.numel()) != 1:
                raise AdjacentConsistencyError(
                    "sampled training timestep is absent or non-unique on RF grid"
                )
            index = int(membership.item())
            draws.append(index)
            if index < int(schedule_t.numel()) - 1:
                return (
                    adjacent_clock_pair(
                        schedule_s,
                        schedule_t,
                        start_index=index,
                        stride=self.acd_stride,
                    ),
                    tuple(draws),
                )
            if len(draws) > 65:
                raise AdjacentConsistencyError(
                    "unable to draw a nonzero adjacent consistency clock"
                )

    @staticmethod
    def _probe_hash(model: nn.Module) -> str:
        return tensor_sha256(state_probe(model, values_per_tensor=2))

    def forward(
        self,
        rgb: Tensor,
        actions: Tensor | None = None,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Execute exactly one teacher, online, and stopped EMA target call."""

        self._ensure_video_only_runtime_contract()
        if actions is None:
            raise ValueError("ACD-P0 requires explicit planned actions")
        if kwargs.get("auxiliary_target") is not None:
            raise AdjacentConsistencyError(
                "ACD-P0 forbids auxiliary targets and future-feature caches"
            )
        if int(rgb.shape[0]) != 1:
            raise AdjacentConsistencyError("ACD-P0 fixes local batch size to one")
        teacher = self.bound_teacher_model()
        target_model = self.bound_target_model()
        teacher.eval()
        target_model.eval()
        self.aux_losses = {}
        self._acd_call_counts = {}
        self._acd_rng_before = {}

        morphology_index = kwargs.get("morphology_index")
        clean = self._encode_clip(rgb).to(rgb.dtype)
        reference, history_tokens = self._history_reference(rgb, clean.shape)
        if int(clean.shape[2]) != 4 or history_tokens != 2:
            raise AdjacentConsistencyError(
                "ACD-P0 fixes 13 RGB frames -> four Wan tokens (two future)"
            )
        history_rgb = rgb[:, : self.num_history_frames]
        noise = torch.randn_like(clean)
        pair, clock_draw_indexes = self._sample_training_clock(rgb.device)
        rejected_draws = len(clock_draw_indexes) - 1
        start_sigma = pair.start_sigma.expand(1).to(rgb.device)
        end_sigma = pair.end_sigma.expand(1).to(rgb.device)
        start_timestep = pair.start_timestep.expand(1).to(rgb.device)
        end_timestep = pair.end_timestep.expand(1).to(rgb.device)

        source = rectified_flow_noisy_state(clean, noise, start_sigma).to(rgb.dtype)
        source = restore_history_forward_path(
            source,
            reference,
            noise,
            start_sigma,
            history_tokens=history_tokens,
        ).to(rgb.dtype)
        latent_frames = int(clean.shape[2])
        teacher_conditions = self._conditions_for(
            teacher,
            history_rgb,
            actions,
            morphology_index,
            latent_frames=latent_frames,
            history_tokens=history_tokens,
        )
        student_conditions = self._conditions_for(
            self,
            history_rgb,
            actions,
            morphology_index,
            latent_frames=latent_frames,
            history_tokens=history_tokens,
        )
        target_conditions = self._conditions_for(
            target_model,
            history_rgb,
            actions,
            morphology_index,
            latent_frames=latent_frames,
            history_tokens=history_tokens,
        )

        with torch.no_grad():
            teacher_velocity = self._velocity_call(
                teacher,
                source,
                start_timestep,
                start_sigma.to(rgb.dtype),
                reference,
                teacher_conditions,
                role="teacher",
            )
            stepped = rectified_flow_euler_step(
                source, teacher_velocity, start_sigma, end_sigma
            ).to(rgb.dtype)
            stepped = restore_history_forward_path(
                stepped,
                reference,
                noise,
                end_sigma,
                history_tokens=history_tokens,
            ).to(rgb.dtype)

        student_velocity = self._velocity_call(
            self,
            source,
            start_timestep,
            start_sigma.to(rgb.dtype),
            reference,
            student_conditions,
            role="online_student",
        )
        online_output = consistency_output(
            source,
            student_velocity,
            start_sigma,
            sigma_data=self.acd_sigma_data,
        )
        with torch.no_grad():
            target_velocity = self._velocity_call(
                target_model,
                stepped,
                end_timestep,
                end_sigma.to(rgb.dtype),
                reference,
                target_conditions,
                role="ema_target",
            )
            target_output = consistency_output(
                stepped,
                target_velocity,
                end_sigma,
                sigma_data=self.acd_sigma_data,
            )

        if self._acd_call_counts != {
            "teacher": 1,
            "online_student": 1,
            "ema_target": 1,
        }:
            raise AdjacentConsistencyError("training Wan call ledger differs")
        loss_mask = self._build_loss_mask(rgb, mask, clean.shape)
        future_mask = loss_mask[:, :, history_tokens:]
        consistency_loss, _ = masked_pseudo_huber(
            online_output[:, :, history_tokens:],
            target_output[:, :, history_tokens:],
            future_mask,
            c=self.acd_huber_c,
        )
        rf_target = noise - clean
        rf_per_sample = self._masked_per_sample_mse(
            student_velocity[:, :, history_tokens:],
            rf_target[:, :, history_tokens:],
            future_mask,
        )
        rf_loss = rf_per_sample.mean()
        returned = rf_loss if self.acd_arm_mode == "rf_control" else consistency_loss
        if not bool(torch.isfinite(returned).all()):
            raise FloatingPointError("ACD-P0 objective is non-finite")

        self.aux_losses.update(
            {
                "flow_loss": rf_loss.detach(),
                "video_flow_loss": rf_loss.detach(),
                "adjacent_consistency/raw_loss": consistency_loss.detach(),
                "adjacent_consistency/arm_code": returned.new_tensor(
                    0.0 if self.acd_arm_mode == "rf_control" else 1.0
                ),
                "adjacent_consistency/start_sigma": start_sigma.mean().detach(),
                "adjacent_consistency/end_sigma": end_sigma.mean().detach(),
                "adjacent_consistency/delta_sigma": (
                    end_sigma - start_sigma
                ).mean().detach(),
                "adjacent_consistency/start_index": returned.new_tensor(
                    float(pair.start_index)
                ),
                "adjacent_consistency/end_index": returned.new_tensor(
                    float(pair.end_index)
                ),
                "adjacent_consistency/rejected_clock_draws": returned.new_tensor(
                    float(rejected_draws)
                ),
                "adjacent_consistency/teacher_calls": returned.new_tensor(1.0),
                "adjacent_consistency/online_calls": returned.new_tensor(1.0),
                "adjacent_consistency/ema_target_calls": returned.new_tensor(1.0),
                "adjacent_consistency/deployment_feature_calls": returned.new_tensor(0.0),
            }
        )
        clip = kwargs.get("clip_index")
        if (
            not isinstance(clip, Tensor)
            or clip.numel() != 1
            or clip.dtype.is_floating_point
        ):
            raise AdjacentConsistencyError(
                "ACD-P0 requires one non-floating clip_index per local batch"
            )
        clip_tensor = clip.detach()
        action_tensor = actions.detach()
        cuda_rng = (
            torch.cuda.get_rng_state(rgb.device)
            if rgb.is_cuda
            else torch.empty(0, dtype=torch.uint8)
        )
        self.paired_audit_exact = {
            "clip_index": tensor_sha256(clip_tensor),
            "actions": tensor_sha256(action_tensor),
            "clean_latent": tensor_sha256(clean),
            "noise": tensor_sha256(noise),
            "clock_indexes": tensor_sha256(
                torch.tensor(
                    [*clock_draw_indexes, pair.start_index, pair.end_index],
                    dtype=torch.int64,
                )
            ),
            "clock_sigmas": tensor_sha256(
                torch.stack((pair.start_sigma, pair.end_sigma))
            ),
            "clock_timesteps": tensor_sha256(
                torch.stack((pair.start_timestep, pair.end_timestep))
            ),
            "source_state": tensor_sha256(source),
            "teacher_stepped_state": tensor_sha256(stepped),
            "online_consistency": tensor_sha256(online_output),
            "target_consistency": tensor_sha256(target_output),
            "rf_target": tensor_sha256(rf_target),
            "teacher_probe": self._probe_hash(teacher),
            "ema_probe_before_update": self._probe_hash(target_model),
            "cpu_rng_before_teacher": self._acd_rng_before["teacher:0"][0],
            "cuda_rng_before_teacher": self._acd_rng_before["teacher:0"][1],
            "cpu_rng_before_online_student": self._acd_rng_before[
                "online_student:0"
            ][0],
            "cuda_rng_before_online_student": self._acd_rng_before[
                "online_student:0"
            ][1],
            "cpu_rng_before_ema_target": self._acd_rng_before["ema_target:0"][0],
            "cuda_rng_before_ema_target": self._acd_rng_before["ema_target:0"][1],
            "cpu_rng_after_forward": tensor_sha256(torch.get_rng_state()),
            "cuda_rng_after_forward": tensor_sha256(cuda_rng),
            "call_ledger": tensor_sha256(
                torch.tensor([1, 1, 1], dtype=torch.int64)
            ),
        }
        return returned

    @torch.inference_mode()
    def sample_consistency_deployable(
        self,
        history_rgb: Tensor,
        actions: Tensor,
        morphology_index: Tensor | None,
        *,
        renoise_noises: Tensor,
        steps: int,
        sampler_mode: SamplerMode = "consistency",
    ) -> AdjacentConsistencySample:
        """Generate with only observables and sample-keyed re-noising streams.

        ``renoise_noises`` has shape ``[steps,B,C,T,H,W]``. Its first slice is
        the initial state. Later slices drive consistency re-noising; native RF
        uses only the first slice while retaining the same registered stream.
        """

        self._ensure_video_only_runtime_contract()
        if history_rgb.ndim != 5 or int(history_rgb.shape[1]) != self.num_history_frames:
            raise ValueError(
                f"sampler requires exactly {self.num_history_frames} observed frames"
            )
        if isinstance(steps, bool) or not isinstance(steps, int) or steps not in {1, 2, 4}:
            raise ValueError("ACD-P0 deployment steps must be one of {1,2,4}")
        if sampler_mode not in {"native_rf", "consistency"}:
            raise ValueError("sampler_mode must be native_rf or consistency")
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
        if tuple(renoise_noises.shape) != (steps, *shape):
            raise ValueError(
                f"renoise_noises shape {tuple(renoise_noises.shape)} != "
                f"expected {(steps, *shape)}"
            )
        if not bool(torch.isfinite(renoise_noises).all()):
            raise ValueError("renoise noises must be finite")
        reference = history_latents.new_zeros(shape)
        reference[:, :, :history_tokens] = history_latents
        conditions = self._conditions_for(
            self,
            history_rgb,
            actions,
            morphology_index,
            latent_frames=latent_tokens,
            history_tokens=history_tokens,
        )
        self.sample_scheduler.set_timesteps(steps, device=history_rgb.device)
        timesteps = tuple(self.sample_scheduler.timesteps)
        sigmas = self.sample_scheduler.sigmas.to(
            device=history_rgb.device, dtype=torch.float32
        )[: steps + 1]
        if len(timesteps) != steps or int(sigmas.numel()) != steps + 1:
            raise AdjacentConsistencyError("native deployment clock differs")
        if (
            float(sigmas[0]) != 1.0
            or float(sigmas[-1]) != 0.0
            or not bool(torch.isfinite(sigmas).all())
            or bool(((sigmas < 0) | (sigmas > 1)).any())
            or bool((sigmas[1:] >= sigmas[:-1]).any())
            or not torch.equal(
                torch.stack(timesteps).float().cpu(),
                (sigmas[:-1] * 1000.0).float().cpu(),
            )
        ):
            raise AdjacentConsistencyError(
                "deployment clock requires terminal zero and exact t=1000*sigma"
            )
        state = restore_history_forward_path(
            renoise_noises[0],
            reference,
            renoise_noises[0],
            sigmas[0],
            history_tokens=history_tokens,
        ).to(history_rgb.dtype)
        self._acd_call_counts = {}
        self._acd_rng_before = {}
        endpoint = None
        for index, timestep in enumerate(timesteps):
            sigma = sigmas[index].expand(batch_size)
            velocity = self._velocity_call(
                self,
                state,
                timestep.expand(batch_size).to(history_rgb.device),
                sigma.to(history_rgb.dtype),
                reference,
                conditions,
                role="deployment_student",
            )
            next_sigma = sigmas[index + 1].expand(batch_size)
            if sampler_mode == "consistency":
                endpoint = consistency_output(
                    state,
                    velocity,
                    sigma,
                    sigma_data=self.acd_sigma_data,
                ).to(history_rgb.dtype)
                if index + 1 < steps:
                    state = rectified_flow_noisy_state(
                        endpoint,
                        renoise_noises[index + 1],
                        next_sigma,
                    ).to(history_rgb.dtype)
                    state = restore_history_forward_path(
                        state,
                        reference,
                        renoise_noises[index + 1],
                        next_sigma,
                        history_tokens=history_tokens,
                    ).to(history_rgb.dtype)
            else:
                state = rectified_flow_euler_step(
                    state, velocity, sigma, next_sigma
                ).to(history_rgb.dtype)
                state = restore_history_forward_path(
                    state,
                    reference,
                    renoise_noises[0],
                    next_sigma,
                    history_tokens=history_tokens,
                ).to(history_rgb.dtype)
                endpoint = state
        if endpoint is None:
            raise AdjacentConsistencyError("consistency sampler made no call")
        clean_estimate = endpoint.clone()
        clean_estimate[:, :, :history_tokens] = reference[:, :, :history_tokens]
        decoded = self.rgb_tokenizer.decode_temporal(
            clean_estimate,
            out_hw=(int(history_rgb.shape[-2]), int(history_rgb.shape[-1])),
        )
        calls = self._acd_call_counts.get("deployment_student", 0)
        if calls != steps or not all(
            bool(torch.isfinite(value).all())
            for value in (clean_estimate, decoded)
        ):
            raise AdjacentConsistencyError("deployment call/finite contract failed")
        return AdjacentConsistencySample(
            absolute_video_latent=clean_estimate,
            decoded_future=decoded[:, :, -self.num_future_frames :],
            model_calls=calls,
            teacher_calls=0,
            feature_calls=0,
            sigma_nodes=sigmas.detach().clone(),
            sampler_mode=sampler_mode,
        )
