"""End-to-end native-latent P/Q training for the IPQ-TC1 pilot.

This specialization is isolated from the production LACWM sampler.  During
training it splits the future Wan VAE latent into exact spatial Haar-LL (P) and
complementary detail (Q) states, corrupts them at tied or independent clocks,
and predicts both projected velocities in one shared Wan call.  The dedicated
``sample_ipq_latent`` method is the only mixed-clock inference entrypoint.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Literal

import torch
from torch import Tensor

from lam.dual_explicit_action_dit_model import DualExplicitActionDiTModel
from robot_wm.modeling.dual_diffusion.invertible_two_clock import (
    PROTOCOL_VERSION,
    assert_projector_receipt,
    band_sigma_schedule,
    euler_band_step,
    expand_sigma,
    make_mixed_clock_state,
    project_p,
    project_q,
    project_velocities_and_loss,
    projector_receipt,
    split_pq,
)
from robot_wm.modeling.networks.wan_forward_model import DualWanOutput


ArmMode = Literal["synchronous", "independent"]
ConditionSource = Literal["aligned", "state_off", "state_shuffled", "clock_tied"]


class InvertibleTwoClockModelError(RuntimeError):
    """The preregistered IPQ-TC1 model contract was violated."""


@dataclass(frozen=True)
class IPQSamplingResult:
    latent: Tensor
    p_state: Tensor
    q_state: Tensor
    initial_noise: Tensor
    p_sigmas: Tensor
    q_sigmas: Tensor
    wan_calls: int
    trajectory_p: tuple[Tensor, ...]
    trajectory_q: tuple[Tensor, ...]
    injection_rms: tuple[Tensor, ...]


def tensor_sha256(value: Tensor) -> str:
    """Hash exact tensor bytes together with dtype and shape."""

    detached = value.detach().contiguous()
    header = f"{detached.dtype}|{tuple(detached.shape)}|".encode("ascii")
    raw = detached.reshape(-1).view(torch.uint8).cpu().numpy().tobytes(order="C")
    return hashlib.sha256(header + raw).hexdigest()


class InvertibleTwoClockVPM(DualExplicitActionDiTModel):
    """Parameter-matched tied/independent-clock P/Q Wan continuation."""

    def __init__(
        self,
        *,
        invertible_two_clock: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        config = dict(invertible_two_clock)
        self.ipq_arm_mode: ArmMode = str(config.get("arm_mode", "synchronous"))  # type: ignore[assignment]
        self.ipq_history_latent_frames = int(
            config.get("history_latent_frames", 2)
        )
        self.ipq_view_width = int(config.get("view_width", 40))
        self.ipq_num_views = int(config.get("num_views", 3))
        self.ipq_loss_identity_tolerance = float(
            config.get("loss_identity_tolerance", 2e-6)
        )
        if self.ipq_arm_mode not in {"synchronous", "independent"}:
            raise ValueError("arm_mode must be synchronous or independent")
        if (
            self.ipq_history_latent_frames != 2
            or self.ipq_view_width != 40
            or self.ipq_num_views != 3
        ):
            raise ValueError("IPQ-TC1 fixes history=2, three views, view_width=40")
        if (
            not math.isfinite(self.ipq_loss_identity_tolerance)
            or self.ipq_loss_identity_tolerance <= 0
        ):
            raise ValueError("loss identity tolerance must be finite and positive")
        super().__init__(**kwargs)
        self._assert_ipq_model_contract()
        self.paired_audit_exact: dict[str, str] = {}
        self.ipq_protocol_version = PROTOCOL_VERSION

    def _assert_ipq_model_contract(self) -> None:
        forward = self.forward_model
        problems = []
        if not bool(getattr(forward, "dual_diffusion_enabled", False)):
            problems.append("dual Wan schema is disabled")
        if int(forward.tf_token_adapter.tf_channels) != 16:
            problems.append("Q adapter must have 16 native-latent channels")
        if self.tf_condition_mode != "matched" or not self.condition_on_tf:
            problems.append("Q state conditioning must be matched and enabled")
        if not self.condition_on_tf_clock:
            problems.append("Q clock conditioning must be enabled")
        if not bool(getattr(forward, "tf_head_condition_on_clock", False)):
            problems.append("Q head must receive its raw clock")
        state_gate = forward.tf_token_adapter.effective_gate().detach().float()
        clock_gate = forward.tf_clock_embedding.effective_gate().detach().float()
        if not torch.allclose(state_gate.cpu(), torch.tensor(0.02), atol=1e-7, rtol=0):
            problems.append("state gate is not fixed open at 0.02")
        if not torch.allclose(clock_gate.cpu(), torch.tensor(0.02), atol=1e-7, rtol=0):
            problems.append("clock gate is not fixed open at 0.02")
        if forward.tf_token_adapter.gate.requires_grad:
            problems.append("state gate must be frozen")
        if forward.tf_clock_embedding.gate.requires_grad:
            problems.append("clock gate must be frozen")
        if self.auxiliary_history_mode != "diffuse_all":
            problems.append("Q history support must be diffuse_all/zero")
        if problems:
            raise InvertibleTwoClockModelError(
                "invalid IPQ-TC1 model configuration: " + "; ".join(problems)
            )

    def _validate_geometry(self, value: Tensor, history_frames: int) -> None:
        if (
            value.ndim != 5
            or value.shape[1] != 16
            or value.shape[2] != 4
            or value.shape[3] != 24
            or value.shape[4] != self.ipq_num_views * self.ipq_view_width
            or history_frames != self.ipq_history_latent_frames
        ):
            raise InvertibleTwoClockModelError(
                "IPQ-TC1 requires native [B,16,4,24,120] with two history tokens"
            )

    @staticmethod
    def _probe_mean(value: Tensor) -> Tensor:
        flat = value.detach().float().reshape(value.shape[0], -1)
        return flat[:, : min(8, int(flat.shape[1]))].mean()

    def _draw_training_clocks(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Draw both clocks in the same RNG order for both matched arms."""

        timestep_p = self._sample_timesteps(batch_size, device)
        sigma_p = self._get_sigmas(
            timestep_p, n_dim=2, dtype=dtype, device=device
        )[:, 0]
        spare_timestep_q = self._sample_timesteps(batch_size, device)
        spare_sigma_q = self._get_sigmas(
            spare_timestep_q, n_dim=2, dtype=dtype, device=device
        )[:, 0]
        effective_sigma_q = (
            sigma_p if self.ipq_arm_mode == "synchronous" else spare_sigma_q
        )
        return (
            timestep_p,
            sigma_p,
            spare_timestep_q,
            spare_sigma_q,
            effective_sigma_q,
        )

    def forward(
        self,
        rgb: Tensor,
        actions: Tensor | None = None,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Train one shared Wan call on exact complementary P/Q velocities."""

        if kwargs.get("auxiliary_target") is not None:
            raise InvertibleTwoClockModelError(
                "IPQ-TC1 forbids every cached or online auxiliary target"
            )
        self.aux_losses = {}
        device = rgb.device
        clean = self._encode_clip(rgb).to(rgb.dtype)
        batch_size, _, latent_frames, _, _ = clean.shape
        reference, history_frames = self._history_reference(rgb, clean.shape)
        self._validate_geometry(clean, history_frames)
        _, z_control, _ = self._latent_actions(
            rgb,
            actions,
            kwargs.get("morphology_index"),
            latent_frames,
            history_frames,
        )
        z_control = z_control.to(rgb.dtype)

        canonical_noise = torch.randn_like(clean)
        (
            timestep_p,
            sigma_p,
            spare_timestep_q,
            spare_sigma_q,
            sigma_q,
        ) = self._draw_training_clocks(batch_size, device, clean.dtype)
        mixed = make_mixed_clock_state(
            clean,
            canonical_noise,
            sigma_p,
            sigma_q,
            history_frames=history_frames,
            view_width=self.ipq_view_width,
        )
        receipt = projector_receipt(
            clean,
            history_frames=history_frames,
            view_width=self.ipq_view_width,
        )
        assert_projector_receipt(receipt)

        context = self._build_context(batch_size, device, rgb.dtype)
        clip_fea = self._build_clip(batch_size, device, rgb.dtype)
        rng_before_wan = torch.get_rng_state()
        cuda_rng_before_wan = (
            torch.cuda.get_rng_state(device)
            if clean.is_cuda
            else torch.empty(0, dtype=torch.uint8)
        )
        observed_wan_calls = 0

        def count_wan_call(_module, _inputs, _output):
            nonlocal observed_wan_calls
            observed_wan_calls += 1

        wan_handle = self.forward_model.transformer.register_forward_hook(
            count_wan_call
        )
        try:
            prediction = self.forward_model(
                mixed.native_noisy.to(rgb.dtype),
                timestep_p,
                z_control,
                reference,
                context,
                clip_fea,
                noisy_tf=mixed.noisy_q.to(rgb.dtype),
                conditioning_tf=mixed.noisy_q.to(rgb.dtype),
                tf_sigma=sigma_q,
                condition_on_tf=True,
                condition_on_tf_clock=True,
            )
        finally:
            wan_handle.remove()
        if observed_wan_calls != 1:
            raise InvertibleTwoClockModelError(
                f"training Wan hook observed {observed_wan_calls} calls, expected one"
            )
        if not isinstance(prediction, DualWanOutput):
            raise InvertibleTwoClockModelError(
                "IPQ-TC1 Wan call did not return both velocity heads"
            )

        loss_mask = self._build_loss_mask(rgb, mask, clean.shape)
        future_mask = loss_mask[:, :, history_frames:]
        projected = project_velocities_and_loss(
            prediction.video_velocity[:, :, history_frames:],
            prediction.tf_velocity[:, :, history_frames:],
            mixed.target_p[:, :, history_frames:],
            mixed.target_q[:, :, history_frames:],
            future_mask,
            history_frames=0,
            view_width=self.ipq_view_width,
        )
        if float(projected.loss_identity_relative_error.detach()) > (
            self.ipq_loss_identity_tolerance
        ):
            raise InvertibleTwoClockModelError(
                "orthogonal P/Q loss identity exceeded its tolerance"
            )

        expanded_p = expand_sigma(sigma_p, mixed.noisy_p)
        expanded_q = expand_sigma(sigma_q, mixed.noisy_q)
        estimate_p = (
            mixed.noisy_p[:, :, history_frames:]
            - expanded_p * projected.velocity_p
        )
        estimate_q = (
            mixed.noisy_q[:, :, history_frames:]
            - expanded_q * projected.velocity_q
        )
        clean_future = clean[:, :, history_frames:]
        clean_estimate = estimate_p + estimate_q
        video_nmse = self._masked_per_sample_nmse(
            clean_estimate, clean_future, future_mask
        ).mean()
        p_nmse = self._masked_per_sample_nmse(
            estimate_p,
            mixed.clean_p[:, :, history_frames:],
            future_mask,
        ).mean()
        q_nmse = self._masked_per_sample_nmse(
            estimate_q,
            mixed.clean_q[:, :, history_frames:],
            future_mask,
        ).mean()

        telemetry = prediction.tf_condition_telemetry
        self.aux_losses.update(
            {
                "flow_loss": projected.loss.detach(),
                "ipq/loss": projected.loss.detach(),
                "ipq/combined_loss": projected.combined_loss.detach(),
                "ipq/loss_identity_relative_error": projected.loss_identity_relative_error.detach(),
                "ipq/model_calls": projected.loss.new_tensor(
                    float(observed_wan_calls)
                ),
                "ipq/independent_clock_arm": projected.loss.new_tensor(
                    float(self.ipq_arm_mode == "independent")
                ),
                "ipq/video_x0_nmse": video_nmse.detach(),
                "ipq/p_x0_nmse": p_nmse.detach(),
                "ipq/q_x0_nmse": q_nmse.detach(),
                "ipq/sigma_p_mean": sigma_p.mean().detach(),
                "ipq/spare_sigma_q_mean": spare_sigma_q.mean().detach(),
                "ipq/effective_sigma_q_mean": sigma_q.mean().detach(),
                "ipq/reconstruction_max_abs": projected.loss.new_tensor(
                    receipt.reconstruction_max_abs
                ),
                "ipq/reconstruction_relative_energy": projected.loss.new_tensor(
                    receipt.reconstruction_relative_energy
                ),
                "ipq/normalized_inner_product": projected.loss.new_tensor(
                    receipt.normalized_inner_product
                ),
                "ipq/state_injection_rms": telemetry[
                    "state_residual_rms"
                ].detach(),
                "ipq/clock_injection_rms": telemetry[
                    "clock_residual_rms"
                ].detach(),
                "paired_audit/clean_probe": self._probe_mean(clean),
                "paired_audit/noise_probe": self._probe_mean(canonical_noise),
                "paired_audit/p_noisy_probe": self._probe_mean(mixed.noisy_p),
                "paired_audit/q_noisy_probe": self._probe_mean(mixed.noisy_q),
            }
        )
        clip_index = kwargs.get("clip_index")
        clip_tensor = (
            clip_index
            if isinstance(clip_index, Tensor)
            else torch.empty(0, dtype=torch.long, device=device)
        )
        action_tensor = (
            actions
            if isinstance(actions, Tensor)
            else torch.empty(0, dtype=clean.dtype, device=device)
        )
        cuda_rng_after_wan = (
            torch.cuda.get_rng_state(device)
            if clean.is_cuda
            else torch.empty(0, dtype=torch.uint8)
        )
        self.paired_audit_exact = {
            "clip_index": tensor_sha256(clip_tensor),
            "actions": tensor_sha256(action_tensor),
            "clean_latent": tensor_sha256(clean),
            "canonical_noise": tensor_sha256(canonical_noise),
            "timestep_p": tensor_sha256(timestep_p),
            "sigma_p": tensor_sha256(sigma_p),
            "spare_timestep_q": tensor_sha256(spare_timestep_q),
            "spare_sigma_q": tensor_sha256(spare_sigma_q),
            "effective_sigma_q": tensor_sha256(sigma_q),
            "noisy_p": tensor_sha256(mixed.noisy_p),
            "noisy_q": tensor_sha256(mixed.noisy_q),
            "native_noisy": tensor_sha256(mixed.native_noisy),
            "cpu_rng_before_wan": tensor_sha256(rng_before_wan),
            "cuda_rng_before_wan": tensor_sha256(cuda_rng_before_wan),
            "cpu_rng_after_wan": tensor_sha256(torch.get_rng_state()),
            "cuda_rng_after_wan": tensor_sha256(cuda_rng_after_wan),
            "wan_call_count": tensor_sha256(
                torch.tensor(observed_wan_calls, device=device, dtype=torch.int64)
            ),
        }
        if not bool(torch.isfinite(projected.loss)):
            raise FloatingPointError("IPQ-TC1 training loss is non-finite")
        return projected.loss

    @torch.no_grad()
    def sample_ipq_latent(
        self,
        history_rgb: Tensor,
        *,
        actions: Tensor | None,
        morphology_index: Tensor | None,
        initial_noise: Tensor,
        num_steps: int,
        schedule_mode: Literal["synchronous", "p_leads", "q_leads"],
        condition_source: ConditionSource = "aligned",
        collect_trajectory: bool = False,
    ) -> IPQSamplingResult:
        """Run the isolated target-blind P/Q sampler and return a native latent."""

        if history_rgb.ndim != 5 or history_rgb.shape[1] != self.num_history_frames:
            raise ValueError(
                "IPQ deployment accepts exactly the observed RGB history"
            )
        if num_steps not in {1, 2, 4}:
            raise ValueError("IPQ-TC1 evaluates only NFE 1, 2, or 4")
        if condition_source not in {
            "aligned",
            "state_off",
            "state_shuffled",
            "clock_tied",
        }:
            raise ValueError("unsupported IPQ conditioning intervention")
        history_latents = self._encode_clip(history_rgb).to(history_rgb.dtype)
        latent_frames = self.rgb_tokenizer.latent_temporal_len(
            self.num_history_frames + self.num_future_frames
        )
        history_frames = int(history_latents.shape[2])
        shape = (
            history_rgb.shape[0],
            history_latents.shape[1],
            latent_frames,
            history_latents.shape[3],
            history_latents.shape[4],
        )
        if tuple(initial_noise.shape) != tuple(shape):
            raise ValueError(
                f"initial noise shape {tuple(initial_noise.shape)} != {shape}"
            )
        self._validate_geometry(initial_noise, history_frames)
        reference = history_latents.new_zeros(shape)
        reference[:, :, :history_frames] = history_latents
        _, z_control, _ = self._latent_actions(
            history_rgb,
            actions,
            morphology_index,
            latent_frames,
            history_frames,
        )
        z_control = z_control.to(history_rgb.dtype)
        context = self._build_context(
            shape[0], history_rgb.device, history_rgb.dtype
        )
        clip_fea = self._build_clip(
            shape[0], history_rgb.device, history_rgb.dtype
        )

        p_state, q_state = split_pq(
            initial_noise,
            history_frames=history_frames,
            view_width=self.ipq_view_width,
        )
        self.sample_scheduler.set_timesteps(num_steps, device=history_rgb.device)
        base_sigmas = self.sample_scheduler.sigmas.to(
            device=history_rgb.device, dtype=torch.float32
        )[: num_steps + 1]
        p_sigmas, q_sigmas = band_sigma_schedule(base_sigmas, schedule_mode)
        timesteps = self.sample_scheduler.timesteps
        if timesteps.numel() != num_steps:
            raise InvertibleTwoClockModelError(
                "native sampler timestep count differs from requested NFE"
            )
        trajectory_p = [p_state.detach().clone()] if collect_trajectory else []
        trajectory_q = [q_state.detach().clone()] if collect_trajectory else []
        injection_rms = []
        wan_calls = 0
        for index, timestep in enumerate(timesteps):
            sigma_p = p_sigmas[index]
            sigma_q = q_sigmas[index]
            native_state = p_state + q_state
            native_state[:, :, :history_frames] = (
                (1.0 - sigma_p) * reference[:, :, :history_frames].float()
                + sigma_p * initial_noise[:, :, :history_frames].float()
            )
            if condition_source == "state_shuffled":
                conditioning_q = self._roll_across_global_batch(q_state)
            else:
                conditioning_q = q_state
            # ``state_off`` is a shared-trunk intervention, not deletion of Q:
            # the Q head still needs its own local noisy q state and raw q clock
            # to predict the velocity that the separate Q integrator consumes.
            use_condition = condition_source != "state_off"
            q_clock = sigma_p if condition_source == "clock_tied" else sigma_q
            q_batch_clock = q_clock.expand(shape[0]).to(history_rgb.dtype)
            prediction = self.forward_model(
                native_state.to(history_rgb.dtype),
                timestep.expand(shape[0]),
                z_control,
                reference,
                context,
                clip_fea,
                noisy_tf=q_state.to(history_rgb.dtype),
                conditioning_tf=conditioning_q.to(history_rgb.dtype),
                tf_sigma=q_batch_clock,
                condition_on_tf=use_condition,
                condition_on_tf_clock=use_condition,
            )
            wan_calls += 1
            if not isinstance(prediction, DualWanOutput):
                raise InvertibleTwoClockModelError(
                    "IPQ sampler did not receive both velocity heads"
                )
            velocity_p = project_p(
                prediction.video_velocity,
                history_frames=history_frames,
                view_width=self.ipq_view_width,
            )
            velocity_q = project_q(
                prediction.tf_velocity,
                history_frames=history_frames,
                view_width=self.ipq_view_width,
            )
            p_state, q_state = euler_band_step(
                p_state,
                q_state,
                velocity_p,
                velocity_q,
                sigma_p,
                p_sigmas[index + 1],
                sigma_q,
                q_sigmas[index + 1],
            )
            p_state[:, :, :history_frames] = 0.0
            q_state[:, :, :history_frames] = 0.0
            injection_rms.append(
                prediction.tf_condition_telemetry["combined_rms"]
                .detach()
                .float()
                .cpu()
            )
            if collect_trajectory:
                trajectory_p.append(p_state.detach().clone())
                trajectory_q.append(q_state.detach().clone())
        if wan_calls != num_steps:
            raise InvertibleTwoClockModelError(
                f"actual Wan calls {wan_calls} != declared NFE {num_steps}"
            )
        final_latent = (p_state + q_state).to(history_rgb.dtype)
        final_latent[:, :, :history_frames] = reference[
            :, :, :history_frames
        ]
        return IPQSamplingResult(
            latent=final_latent,
            p_state=p_state,
            q_state=q_state,
            initial_noise=initial_noise,
            p_sigmas=p_sigmas,
            q_sigmas=q_sigmas,
            wan_calls=wan_calls,
            trajectory_p=tuple(trajectory_p),
            trajectory_q=tuple(trajectory_q),
            injection_rms=tuple(injection_rms),
        )
