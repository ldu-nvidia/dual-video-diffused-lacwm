"""Plan-first causal motion diffusion for the VPM video generator.

The planner is calibrated once on a deterministic train-only split, loaded
from one content-addressed checkpoint, frozen, and executed identically by the
PLAN-OFF and PLAN-ON continuation arms.  The video branch is conditioned only
on the planner's autonomous two-call output.  It never consumes a clean motion
target or future-derived feature.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from lam.dual_explicit_action_dit_model import DualExplicitActionDiTModel
from robot_wm.modeling.dual_diffusion.causal_motion_plan import (
    FUTURE_TOKENS,
    HISTORY_TOKENS,
    LATENT_HEIGHT,
    LATENT_WIDTH,
    PLAN_CHANNELS,
    PLAN_HEIGHT,
    PLAN_STEPS,
    PLAN_WIDTH,
    CausalMotionPlanError,
    CausalMotionPlanner,
    MotionPlanNormalizer,
    build_plan_condition,
    motion_plan_target,
)
from robot_wm.modeling.dual_diffusion.conditioning import roll_across_global_batch
from robot_wm.modeling.networks.wan_forward_model import DualWanOutput


PLAN_CONDITION_SOURCES = ("aligned", "off", "shuffled", "action_shuffled")
PlanConditionSource = Literal["aligned", "off", "shuffled", "action_shuffled"]
PLANNER_FIT_MODULUS = 8
PLANNER_CALIBRATION_REMAINDER = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_temporal(tokenizer: nn.Module, rgb: Tensor) -> Tensor:
    with torch.no_grad():
        video = rearrange(rgb, "b t c h w -> b c t h w")
        return tokenizer.encode_temporal(video).detach()


class CausalMotionPlannerCalibrationModel(nn.Module):
    """Train only the compact planner; the Wan VAE is a frozen target encoder."""

    def __init__(
        self,
        *,
        rgb_tokenizer: nn.Module,
        causal_motion_planner: CausalMotionPlanner,
        motion_plan_normalizer: MotionPlanNormalizer,
        rollout_loss_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if rollout_loss_weight < 0:
            raise ValueError("rollout_loss_weight must be non-negative")
        self.rgb_tokenizer = rgb_tokenizer
        self.causal_motion_planner = causal_motion_planner
        self.motion_plan_normalizer = motion_plan_normalizer
        self.rollout_loss_weight = float(rollout_loss_weight)
        for parameter in self.rgb_tokenizer.parameters():
            parameter.requires_grad = False
        self.rgb_tokenizer.eval()
        self.aux_losses: dict[str, Tensor] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.rgb_tokenizer.eval()
        return self

    def _assert_train_only_partition(self, clip_index: Tensor) -> None:
        indexes = clip_index.detach().reshape(-1).to(torch.long)
        is_calibration = indexes.remainder(PLANNER_FIT_MODULUS).eq(
            PLANNER_CALIBRATION_REMAINDER
        )
        if self.training and bool(is_calibration.any()):
            raise CausalMotionPlanError(
                "planner fit batch contains a held-out train-calibration clip"
            )
        if not self.training and bool((~is_calibration).any()):
            raise CausalMotionPlanError(
                "planner diagnostic batch contains a planner-fit clip"
            )

    def forward(
        self,
        rgb: Tensor,
        actions: Tensor,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        del mask
        clip_index = kwargs.get("clip_index")
        morphology_index = kwargs.get("morphology_index")
        if not isinstance(clip_index, Tensor):
            raise CausalMotionPlanError("planner calibration requires clip_index")
        self._assert_train_only_partition(clip_index)
        if rgb.ndim != 5 or tuple(rgb.shape[1:3]) != (13, 3):
            raise ValueError("planner calibration requires [B,13,3,H,W] RGB")
        full_latents = _encode_temporal(self.rgb_tokenizer, rgb).to(rgb.dtype)
        history_latents = _encode_temporal(self.rgb_tokenizer, rgb[:, :5]).to(
            rgb.dtype
        )
        history_difference = (
            full_latents[:, :, :HISTORY_TOKENS] - history_latents
        ).abs().max()
        if float(history_difference) > 1e-4:
            raise CausalMotionPlanError(
                "full-clip observed Wan tokens differ from history-only encoding"
            )
        raw_target = motion_plan_target(full_latents, history_latents).detach()
        target = self.motion_plan_normalizer.normalize(raw_target).detach()
        noise = torch.randn_like(target)
        sigma = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        expanded = sigma[:, None, None, None, None]
        noisy = (1.0 - expanded) * target + expanded * noise
        velocity_target = noise - target
        velocity = self.causal_motion_planner(
            noisy, sigma, history_latents, actions, morphology_index
        )
        flow_loss = (velocity.float() - velocity_target.float()).square().mean()
        rollout = self.causal_motion_planner.rollout_two_step(
            noise, history_latents, actions, morphology_index
        )
        reduce_dims = tuple(range(1, target.ndim))
        numerator = (rollout.plan.float() - target.float()).square().mean(
            dim=reduce_dims
        )
        denominator = target.float().square().mean(dim=reduce_dims).clamp_min(1e-6)
        endpoint_nmse = numerator / denominator
        total = flow_loss + self.rollout_loss_weight * endpoint_nmse.mean()
        flat_prediction = rollout.plan.float().flatten(1)
        flat_target = target.float().flatten(1)
        cosine = torch.nn.functional.cosine_similarity(
            flat_prediction, flat_target, dim=1, eps=1e-8
        )
        self.aux_losses = {
            "planner/flow_loss": flow_loss.detach(),
            "planner/two_step_nmse": endpoint_nmse.mean().detach(),
            "planner/two_step_cosine": cosine.mean().detach(),
            "planner/calls": total.new_tensor(float(rollout.calls)),
            "planner/train_partition": total.new_tensor(float(self.training)),
            "planner/clean_future_conditioned": total.new_tensor(0.0),
            "planner/causal_history_max_abs": history_difference.detach(),
        }
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("causal planner loss is non-finite")
        return total


@dataclass(frozen=True)
class CausalMotionPlanSample:
    video_latent: Tensor
    decoded_future: Tensor
    history_latent: Tensor
    generated_plan: Tensor
    injected_plan: Tensor
    wan_calls: int
    planner_calls: int
    history_tokens: int
    condition_source: str
    history_encode_seconds: float
    planner_seconds: float
    wan_seconds: float
    decode_seconds: float
    end_to_end_seconds: float


class CausalMotionPlanVPM(DualExplicitActionDiTModel):
    """VPM continuation conditioned on a frozen autonomous motion planner."""

    def __init__(
        self,
        *,
        causal_motion_planner: CausalMotionPlanner,
        motion_plan_normalizer: MotionPlanNormalizer,
        causal_motion_plan: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        config = dict(causal_motion_plan)
        self.fuse_generated_plan = bool(config.get("fuse_generated_plan", False))
        self.require_planner_checkpoint = bool(
            config.get("require_planner_checkpoint", True)
        )
        self.planner_checkpoint_path = str(config.get("planner_checkpoint", ""))
        self.planner_checkpoint_sha256 = str(
            config.get("planner_checkpoint_sha256", "")
        )
        self.planner_expected_updates = int(
            config.get("planner_expected_updates", 400)
        )
        super().__init__(**kwargs)
        self.causal_motion_planner = causal_motion_planner
        self.motion_plan_normalizer = motion_plan_normalizer
        self._load_and_freeze_planner()
        if self.num_history_latent != HISTORY_TOKENS:
            raise CausalMotionPlanError("CAMP requires exactly two Wan history tokens")
        if int(self.forward_model.tf_token_adapter.tf_channels) != PLAN_CHANNELS:
            raise CausalMotionPlanError("CAMP requires a 16-channel Wan plan adapter")
        if self.tf_loss_weight != 0.0 or self.condition_on_tf_clock:
            raise CausalMotionPlanError(
                "CAMP requires zero auxiliary loss and no auxiliary clock injection"
            )
        if self.tf_condition_mode != "off" or self.condition_on_tf:
            raise CausalMotionPlanError(
                "CAMP owns the sole runtime fusion switch; base TF mode must be off"
            )
        if self.parameter_matched_control or self.video_only_control:
            raise CausalMotionPlanError("legacy VPM no-op control modes must be disabled")

    def _load_and_freeze_planner(self) -> None:
        if self.require_planner_checkpoint:
            path = Path(self.planner_checkpoint_path).expanduser()
            if (
                not path.is_absolute()
                or not path.is_file()
                or path.is_symlink()
                or len(self.planner_checkpoint_sha256) != 64
                or _sha256(path) != self.planner_checkpoint_sha256
            ):
                raise CausalMotionPlanError(
                    "planner checkpoint is absent, mutable, or differs from SHA-256"
                )
            snapshot = torch.load(path, map_location="cpu", weights_only=True)
            if (
                snapshot.get("snapshot_schema_version") != 3
                or snapshot.get("_start_iter") != self.planner_expected_updates
                or not isinstance(snapshot.get("model"), dict)
            ):
                raise CausalMotionPlanError("planner checkpoint metadata differs")
            prefix = "causal_motion_planner."
            source = {
                key[len(prefix) :]: value
                for key, value in snapshot["model"].items()
                if key.startswith(prefix)
            }
            expected = self.causal_motion_planner.state_dict()
            if set(source) != set(expected) or any(
                tuple(source[key].shape) != tuple(expected[key].shape)
                for key in expected
            ):
                raise CausalMotionPlanError("planner checkpoint parameter schema differs")
            self.causal_motion_planner.load_state_dict(source, strict=True)
            normalizer_prefix = "motion_plan_normalizer."
            normalizer_source = {
                key[len(normalizer_prefix) :]: value
                for key, value in snapshot["model"].items()
                if key.startswith(normalizer_prefix)
            }
            normalizer_expected = self.motion_plan_normalizer.state_dict()
            if set(normalizer_source) != set(normalizer_expected) or any(
                not torch.equal(
                    normalizer_source[key].detach().cpu(),
                    normalizer_expected[key].detach().cpu(),
                )
                for key in normalizer_expected
            ):
                raise CausalMotionPlanError(
                    "planner checkpoint used different normalization statistics"
                )
        for parameter in self.causal_motion_planner.parameters():
            parameter.requires_grad = False
        self.causal_motion_planner.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.causal_motion_planner.eval()
        return self

    @staticmethod
    def _plan_noise(reference: Tensor) -> Tensor:
        return torch.randn(
            reference.shape[0],
            PLAN_CHANNELS,
            FUTURE_TOKENS,
            PLAN_HEIGHT,
            PLAN_WIDTH,
            device=reference.device,
            dtype=reference.dtype,
        )

    @staticmethod
    def _probe_mean(value: Tensor) -> Tensor:
        flat = value.detach().float().reshape(value.shape[0], -1)
        return flat[:, : min(8, int(flat.shape[1]))].mean()

    @staticmethod
    def _validate_video_geometry(value: Tensor, history_tokens: int) -> None:
        if tuple(value.shape[1:]) != (
            PLAN_CHANNELS,
            HISTORY_TOKENS + FUTURE_TOKENS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
        ) or history_tokens != HISTORY_TOKENS:
            raise CausalMotionPlanError("CAMP Wan latent geometry changed")

    def _autonomous_condition(
        self,
        history_latents: Tensor,
        actions: Tensor,
        morphology_index: Tensor,
        plan_noise: Tensor,
    ) -> tuple[Tensor, int]:
        with torch.no_grad():
            rollout = self.causal_motion_planner.rollout_two_step(
                plan_noise, history_latents, actions, morphology_index
            )
            condition = build_plan_condition(rollout.plan)
        return condition.detach(), rollout.calls

    def forward(
        self,
        rgb: Tensor,
        actions: Tensor | None = None,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        if kwargs.get("auxiliary_target") is not None:
            raise CausalMotionPlanError(
                "video continuation cannot consume a clean auxiliary target"
            )
        if actions is None:
            raise ValueError("CAMP requires requested actions")
        morphology_index = kwargs.get("morphology_index")
        self.aux_losses = {}
        absolute_clean = self._encode_clip(rgb).to(rgb.dtype)
        reference, history_tokens = self._history_reference(rgb, absolute_clean.shape)
        self._validate_video_geometry(absolute_clean, history_tokens)
        history_difference = (
            absolute_clean[:, :, :history_tokens]
            - reference[:, :, :history_tokens]
        ).abs().max()
        if float(history_difference) > 1e-4:
            raise CausalMotionPlanError(
                "video target observed Wan tokens differ from history-only encoding"
            )
        history_latents = reference[:, :, :history_tokens]
        plan_noise = self._plan_noise(history_latents)
        plan_condition, planner_calls = self._autonomous_condition(
            history_latents, actions, morphology_index, plan_noise
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
        video_noisy = video_noisy.clone()
        video_noisy[:, :, :history_tokens] = reference[:, :, :history_tokens]
        video_target = video_noise - absolute_clean
        context = self._build_context(batch_size, rgb.device, rgb.dtype)
        clip_fea = self._build_clip(batch_size, rgb.device, rgb.dtype)
        zero_plan_sigma = video_sigma.new_zeros(batch_size)
        prediction = self.forward_model(
            video_noisy,
            timesteps,
            z_control,
            reference,
            context,
            clip_fea,
            noisy_tf=plan_condition,
            conditioning_tf=plan_condition,
            tf_sigma=zero_plan_sigma,
            condition_on_tf=self.fuse_generated_plan,
            condition_on_tf_clock=False,
        )
        if not isinstance(prediction, DualWanOutput):
            raise CausalMotionPlanError("Wan did not return the matched dual schema")
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
            "camp/fuse_generated_plan": flow_loss.new_tensor(
                float(self.fuse_generated_plan)
            ),
            "camp/planner_calls": flow_loss.new_tensor(float(planner_calls)),
            "camp/plan_rms": plan_condition.float().square().mean().sqrt().detach(),
            "camp/planner_frozen": flow_loss.new_tensor(
                float(not any(p.requires_grad for p in self.causal_motion_planner.parameters()))
            ),
            "camp/clean_future_plan_conditioned": flow_loss.new_tensor(0.0),
            "camp/causal_history_max_abs": history_difference.detach(),
            "paired_audit/timestep_mean": timesteps.float().mean().detach(),
            "paired_audit/timestep_square_mean": (
                timesteps.float().square().mean().detach()
            ),
            "paired_audit/video_noise_probe": self._probe_mean(video_noise),
            "paired_audit/plan_noise_probe": self._probe_mean(plan_noise),
            "paired_audit/action_probe": self._probe_mean(actions),
        }
        clip_index = kwargs.get("clip_index")
        if isinstance(clip_index, Tensor):
            ids = clip_index.detach().float().reshape(-1)
            self.aux_losses["paired_audit/clip_index_mean"] = ids.mean()
            self.aux_losses["paired_audit/clip_index_square_mean"] = (
                ids.square().mean()
            )
        if not bool(torch.isfinite(flow_loss)):
            raise FloatingPointError("CAMP video loss is non-finite")
        return flow_loss

    @staticmethod
    def _synchronize(value: Tensor) -> None:
        if value.is_cuda:
            torch.cuda.synchronize(value.device)

    @torch.inference_mode()
    def sample_causal_motion_plan(
        self,
        history_rgb: Tensor,
        actions: Tensor,
        morphology_index: Tensor,
        *,
        video_noise: Tensor,
        plan_noise: Tensor,
        steps: int,
        condition_source: PlanConditionSource,
    ) -> CausalMotionPlanSample:
        """Sample using observables and explicit noise only.

        The signature intentionally cannot accept future RGB, a clean video
        latent, a target plan, or a teacher feature.
        """

        if condition_source not in PLAN_CONDITION_SOURCES:
            raise ValueError(f"unsupported plan condition source: {condition_source}")
        if history_rgb.ndim != 5 or tuple(history_rgb.shape[1:3]) != (5, 3):
            raise ValueError("sampler requires exactly five observed RGB frames")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        total_start = time.perf_counter()
        self._synchronize(history_rgb)
        stage_start = time.perf_counter()
        history_latents = self._encode_clip(history_rgb).to(history_rgb.dtype)
        self._validate_video_geometry(
            history_latents.new_zeros(
                history_latents.shape[0],
                PLAN_CHANNELS,
                HISTORY_TOKENS + FUTURE_TOKENS,
                LATENT_HEIGHT,
                LATENT_WIDTH,
            ),
            int(history_latents.shape[2]),
        )
        self._synchronize(history_latents)
        history_encode_seconds = time.perf_counter() - stage_start
        expected_plan_noise = (
            history_latents.shape[0],
            PLAN_CHANNELS,
            FUTURE_TOKENS,
            PLAN_HEIGHT,
            PLAN_WIDTH,
        )
        expected_video_noise = (
            history_latents.shape[0],
            PLAN_CHANNELS,
            HISTORY_TOKENS + FUTURE_TOKENS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
        )
        if tuple(plan_noise.shape) != expected_plan_noise:
            raise ValueError("plan_noise shape differs from the frozen planner state")
        if tuple(video_noise.shape) != expected_video_noise:
            raise ValueError("video_noise shape differs from the frozen Wan state")
        if not bool(torch.isfinite(plan_noise).all()) or not bool(
            torch.isfinite(video_noise).all()
        ):
            raise ValueError("sampler noise must be finite")
        stage_start = time.perf_counter()
        planner_actions = (
            roll_across_global_batch(actions)
            if condition_source == "action_shuffled"
            else actions
        )
        local_rollout = self.causal_motion_planner.rollout_two_step(
            plan_noise, history_latents, planner_actions, morphology_index
        )
        local_condition = build_plan_condition(local_rollout.plan)
        if condition_source == "shuffled":
            injected_plan = roll_across_global_batch(local_rollout.plan)
            injected_condition = build_plan_condition(injected_plan)
        else:
            injected_plan = local_rollout.plan
            injected_condition = local_condition
        use_plan = condition_source != "off" and self.fuse_generated_plan
        self._synchronize(local_condition)
        planner_seconds = time.perf_counter() - stage_start
        reference = video_noise.new_zeros(expected_video_noise)
        reference[:, :, :HISTORY_TOKENS] = history_latents
        _, z_control, _ = self._latent_actions(
            history_rgb,
            actions,
            morphology_index,
            HISTORY_TOKENS + FUTURE_TOKENS,
            HISTORY_TOKENS,
        )
        z_control = z_control.to(history_rgb.dtype)
        context = self._build_context(
            history_rgb.shape[0], history_rgb.device, history_rgb.dtype
        )
        clip_fea = self._build_clip(
            history_rgb.shape[0], history_rgb.device, history_rgb.dtype
        )
        state = video_noise.clone()
        state[:, :, :HISTORY_TOKENS] = history_latents
        self.sample_scheduler.set_timesteps(steps, device=history_rgb.device)
        timesteps = tuple(self.sample_scheduler.timesteps)
        if len(timesteps) != steps:
            raise CausalMotionPlanError("Wan sampling grid differs from declared NFE")
        self._synchronize(state)
        stage_start = time.perf_counter()
        wan_calls = 0
        zero_sigma = state.new_zeros(state.shape[0])
        for timestep in timesteps:
            prediction = self.forward_model(
                state,
                timestep.expand(state.shape[0]).to(state.device),
                z_control,
                reference,
                context,
                clip_fea,
                noisy_tf=local_condition,
                conditioning_tf=injected_condition,
                tf_sigma=zero_sigma,
                condition_on_tf=use_plan,
                condition_on_tf_clock=False,
            )
            if not isinstance(prediction, DualWanOutput):
                raise CausalMotionPlanError("Wan sampler output schema changed")
            state = self.sample_scheduler.step(
                prediction.video_velocity.float(), timestep, state.float()
            ).prev_sample.to(history_rgb.dtype)
            state[:, :, :HISTORY_TOKENS] = history_latents
            wan_calls += 1
        self._synchronize(state)
        wan_seconds = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        decoded = self.rgb_tokenizer.decode_temporal(
            state, out_hw=(int(history_rgb.shape[-2]), int(history_rgb.shape[-1]))
        )
        self._synchronize(decoded)
        decode_seconds = time.perf_counter() - stage_start
        end_to_end_seconds = time.perf_counter() - total_start
        if wan_calls != steps or local_rollout.calls != PLAN_STEPS:
            raise CausalMotionPlanError("reported sampler calls differ from execution")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (state, decoded, local_rollout.plan, injected_plan)
        ):
            raise FloatingPointError("CAMP sampler produced non-finite output")
        return CausalMotionPlanSample(
            video_latent=state,
            decoded_future=decoded[:, :, -self.num_future_frames :],
            history_latent=history_latents,
            generated_plan=local_rollout.plan,
            injected_plan=injected_plan,
            wan_calls=wan_calls,
            planner_calls=local_rollout.calls,
            history_tokens=HISTORY_TOKENS,
            condition_source=condition_source,
            history_encode_seconds=history_encode_seconds,
            planner_seconds=planner_seconds,
            wan_seconds=wan_seconds,
            decode_seconds=decode_seconds,
            end_to_end_seconds=end_to_end_seconds,
        )
