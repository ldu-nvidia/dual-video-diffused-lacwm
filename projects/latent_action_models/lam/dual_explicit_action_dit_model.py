"""Causal explicit-action LACWM with a jointly denoised TF state.

The two pilot arms instantiate this exact class and train the same TF objective.
Their sole causal difference is ``condition_on_tf``: whether the current noisy
or generated TF state can enter Wan's shared video-token stream.

LACWM clock convention: sigma=1 is Gaussian noise and sigma=0 is clean data.
"""

from __future__ import annotations

import logging
import time
from typing import Mapping

import torch
import torch.distributed as dist

from lam.explicit_action_dit_model import ExplicitActionDiTModel
from robot_wm.modeling.dual_diffusion.conditioning import (
    TF_CONDITION_MODES,
    make_oracle_conditioning_tf,
    make_sampling_conditioning_tf,
    make_training_conditioning_tf,
    roll_across_global_batch,
)
from robot_wm.modeling.dual_diffusion.flow import (
    DualClockSampler,
    cascaded_step_counts,
    derive_tf_sigma,
    euler_flow_step,
    pair_native_cascaded_sigma_schedule,
    pair_video_sigma_schedule,
)
from robot_wm.modeling.networks.wan_forward_model import DualWanOutput

logger = logging.getLogger(__name__)

TF_CONDITION_MODE_CODES = {
    mode: index for index, mode in enumerate(TF_CONDITION_MODES)
}
EVALUATION_CONDITION_SOURCES = (
    "autonomous",
    "off",
    "oracle_matched",
    "oracle_shuffled",
    "autonomous_shuffled",
)
EVALUATION_CONDITION_SOURCE_CODES = {
    source: index
    for index, source in enumerate(EVALUATION_CONDITION_SOURCES)
}


class DualExplicitActionDiTModel(ExplicitActionDiTModel):
    """Explicit-action world model with video and an auxiliary flow state."""

    def __init__(
        self,
        *,
        time_frequency_transform=None,
        dual_diffusion: Mapping,
        **kwargs,
    ):
        config = dict(dual_diffusion)
        if not config.get("enabled", False):
            raise ValueError(
                "DualExplicitActionDiTModel requires dual_diffusion.enabled=true"
            )
        super().__init__(**kwargs)
        if not getattr(self.forward_model, "dual_diffusion_enabled", False):
            raise ValueError("WanForwardModel must also have dual diffusion enabled")

        self.time_frequency_transform = time_frequency_transform
        legacy_condition_on_tf = bool(config.get("condition_on_tf", False))
        self.tf_condition_mode = str(
            config.get(
                "condition_mode",
                "matched" if legacy_condition_on_tf else "off",
            )
        )
        if self.tf_condition_mode not in TF_CONDITION_MODES:
            raise ValueError(
                f"condition_mode must be one of {TF_CONDITION_MODES}, "
                f"got {self.tf_condition_mode!r}"
            )
        self.condition_on_tf = self.tf_condition_mode != "off"
        self.condition_on_tf_clock = bool(
            # The original dual implementation always injected its separately
            # gated clock, even when state conditioning was off.
            config.get("condition_on_tf_clock", True)
        )
        self.auxiliary_history_mode = str(
            config.get("auxiliary_history_mode", "clamp_clean")
        )
        if self.auxiliary_history_mode not in {"clamp_clean", "diffuse_all"}:
            raise ValueError(
                "auxiliary_history_mode must be one of "
                "{'clamp_clean', 'diffuse_all'}"
            )
        if (
            self.auxiliary_history_mode == "clamp_clean"
            and bool(getattr(time_frequency_transform, "bidirectional", False))
        ):
            raise ValueError(
                "bidirectional auxiliary targets cannot clamp clean history"
            )
        if (
            time_frequency_transform is None
            and self.auxiliary_history_mode != "diffuse_all"
        ):
            raise ValueError(
                "offline auxiliary targets require "
                "auxiliary_history_mode=diffuse_all; their causal prefix "
                "cannot be verified by the trainer"
            )
        self.tf_schedule_mode = str(config.get("schedule_mode", "tf_leads"))
        self.tf_lead_logit = float(config.get("tf_lead_logit", 1.0))
        self.tf_loss_weight = float(config.get("tf_loss_weight", 1.0))
        self.cascade_tf_loss_probability = float(
            config.get("cascade_tf_loss_probability", 0.4)
        )
        self.cascade_logit_mean = float(
            config.get("cascade_logit_mean", 1.2)
        )
        self.cascade_logit_std = float(
            config.get("cascade_logit_std", 1.0)
        )
        self.cascade_tf_condition_max_sigma = float(
            config.get("cascade_tf_condition_max_sigma", 0.25)
        )
        self.cascade_validation_tf_sigma = float(
            config.get(
                "cascade_validation_tf_sigma",
                0.5 * self.cascade_tf_condition_max_sigma,
            )
        )
        self.cascade_inference_tf_fraction = float(
            config.get("cascade_inference_tf_fraction", 0.5)
        )
        self.video_only_control = bool(config.get("video_only_control", False))
        self.parameter_matched_control = bool(
            config.get("parameter_matched_control", False)
        )
        if self.video_only_control and self.parameter_matched_control:
            raise ValueError(
                "video_only_control and parameter_matched_control are exclusive"
            )
        self.evaluation_nfe_steps = tuple(
            sorted(
                {
                    int(value)
                    for value in config.get(
                        "evaluation_nfe_steps", (self.viz_num_steps,)
                    )
                }
            )
        )
        if self.viz_num_steps not in self.evaluation_nfe_steps:
            self.evaluation_nfe_steps = tuple(
                sorted((*self.evaluation_nfe_steps, self.viz_num_steps))
            )
        if not self.evaluation_nfe_steps or any(
            value < 1 for value in self.evaluation_nfe_steps
        ):
            raise ValueError("evaluation_nfe_steps must contain positive integers")
        self.evaluation_noise_seed = int(
            config.get("evaluation_noise_seed", 20_260_726)
        )
        if self.evaluation_noise_seed < 0:
            raise ValueError("evaluation_noise_seed must be non-negative")
        self.evaluation_condition_sources = tuple(
            str(value)
            for value in config.get(
                "evaluation_condition_sources", ("autonomous",)
            )
        )
        if (
            not self.evaluation_condition_sources
            or len(set(self.evaluation_condition_sources))
            != len(self.evaluation_condition_sources)
            or any(
                source not in EVALUATION_CONDITION_SOURCES
                for source in self.evaluation_condition_sources
            )
        ):
            raise ValueError(
                "evaluation_condition_sources must be unique supported values "
                f"; supported={EVALUATION_CONDITION_SOURCES}"
            )
        if bool(
            getattr(self.forward_model, "intra_forward_forcing_enabled", False)
        ) and any(
            source not in {"autonomous", "off", "autonomous_shuffled"}
            for source in self.evaluation_condition_sources
        ):
            raise ValueError(
                "intra-forward forcing accepts deployable evaluation sources only"
            )
        self.capture_latent_trajectories = bool(
            config.get("capture_latent_trajectories", True)
        )
        artifact_batch_limit = config.get("artifact_batch_limit", 1)
        self.artifact_batch_limit = (
            None
            if artifact_batch_limit is None
            else int(artifact_batch_limit)
        )
        if (
            self.artifact_batch_limit is not None
            and self.artifact_batch_limit < 1
        ):
            raise ValueError("artifact_batch_limit must be positive or null")
        self.validation_video_sigmas = tuple(
            float(value)
            for value in config.get(
                "validation_video_sigmas", (0.90, 0.75, 0.50, 0.25)
            )
        )
        if self.tf_schedule_mode not in {
            "aligned",
            "tf_leads",
            "tf_first_cascaded",
        }:
            raise ValueError(
                "schedule_mode must be aligned, tf_leads, or "
                "tf_first_cascaded"
            )
        if self.tf_loss_weight < 0:
            raise ValueError("tf_loss_weight must be non-negative")
        if not 0 <= self.cascade_tf_loss_probability <= 1:
            raise ValueError(
                "cascade_tf_loss_probability must lie in [0,1]"
            )
        if self.cascade_logit_std <= 0:
            raise ValueError("cascade_logit_std must be positive")
        if not (
            0
            <= self.cascade_validation_tf_sigma
            <= self.cascade_tf_condition_max_sigma
            <= 1
        ):
            raise ValueError(
                "cascade validation/max auxiliary sigmas must satisfy "
                "0 <= validation <= max <= 1"
            )
        if not 0 < self.cascade_inference_tf_fraction < 1:
            raise ValueError(
                "cascade_inference_tf_fraction must lie strictly between 0 and 1"
            )
        self._assert_video_only_control_contract()
        self._assert_parameter_matched_control_contract()
        # Check once more on the first train/eval call, after any model-only
        # warm start has loaded.  Frozen gates cannot subsequently open, and
        # avoiding a scalar device sync on every update keeps speed telemetry
        # representative.
        self._video_only_runtime_validated = not (
            self.video_only_control or self.parameter_matched_control
        )
        if not self.validation_video_sigmas or any(
            not 0 <= sigma <= 1 for sigma in self.validation_video_sigmas
        ):
            raise ValueError("validation_video_sigmas must lie in [0,1]")
        self._cascade_clock_sampler = DualClockSampler(
            mode="tf_first_cascaded_noised",
            logit_mean=self.cascade_logit_mean,
            logit_std=self.cascade_logit_std,
            tf_loss_probability=self.cascade_tf_loss_probability,
            tf_condition_max_sigma=self.cascade_tf_condition_max_sigma,
        )
        self._visualization_artifacts = None
        self.profile_sampling_stages = False
        self._last_sampling_profile = None
        logger.info(
            "DualExplicitActionDiTModel: condition_mode=%s, schedule=%s, "
            "auxiliary_history_mode=%s, tf_lead_logit=%.3f, "
            "tf_loss_weight=%.3f, video_only_control=%s, "
            "parameter_matched_control=%s, "
            "evaluation_nfe=%s, evaluation_sources=%s",
            self.tf_condition_mode,
            self.tf_schedule_mode,
            self.auxiliary_history_mode,
            self.tf_lead_logit,
            self.tf_loss_weight,
            self.video_only_control,
            self.parameter_matched_control,
            self.evaluation_nfe_steps,
            self.evaluation_condition_sources,
        )

    def _assert_video_only_control_contract(self) -> None:
        """Fail closed if the declared video-only arm can consume TF signals."""
        if not self.video_only_control:
            return
        state_gate = self.forward_model.tf_token_adapter.gate
        clock_gate = self.forward_model.tf_clock_embedding.gate
        problems = []
        if self.tf_condition_mode != "off" or self.condition_on_tf:
            problems.append("condition mode is not off")
        if bool(getattr(self.forward_model, "condition_on_tf", False)):
            problems.append("forward-model TF condition is enabled")
        if self.tf_loss_weight != 0.0:
            problems.append(f"TF loss weight is {self.tf_loss_weight}, not zero")
        if state_gate.requires_grad:
            problems.append("state gate is trainable")
        if clock_gate.requires_grad:
            problems.append("clock gate is trainable")
        if float(
            self.forward_model.tf_token_adapter.effective_gate().detach().float()
        ) != 0.0:
            problems.append("state gate is not exact zero")
        if float(
            self.forward_model.tf_clock_embedding.effective_gate().detach().float()
        ) != 0.0:
            problems.append("clock gate is not exact zero")
        if problems:
            raise RuntimeError(
                "video-only control violates its causal no-op contract: "
                + "; ".join(problems)
            )

    def _ensure_video_only_runtime_contract(self) -> None:
        if getattr(self, "_video_only_runtime_validated", False):
            return
        self._assert_video_only_control_contract()
        self._assert_parameter_matched_control_contract()
        self._video_only_runtime_validated = True

    def _assert_parameter_matched_control_contract(self) -> None:
        """Keep the dual schema trainable while making its video path a no-op."""
        if not getattr(self, "parameter_matched_control", False):
            return
        problems = []
        if self.tf_condition_mode != "off" or self.condition_on_tf:
            problems.append("condition mode is not off")
        if bool(getattr(self.forward_model, "condition_on_tf", False)):
            problems.append("forward-model auxiliary state is enabled")
        if bool(getattr(self, "condition_on_tf_clock", False)) or bool(
            getattr(self.forward_model, "condition_on_tf_clock", False)
        ):
            problems.append("forward-model auxiliary clock is enabled")
        if self.tf_loss_weight != 0.0:
            problems.append(
                f"auxiliary loss weight is {self.tf_loss_weight}, not zero"
            )
        state_gate = self.forward_model.tf_token_adapter.gate
        clock_gate = self.forward_model.tf_clock_embedding.gate
        if not state_gate.requires_grad or not clock_gate.requires_grad:
            problems.append("adapter gates are not parameter matched/trainable")
        if float(
            self.forward_model.tf_token_adapter.effective_gate().detach().float()
        ) != 0.0:
            problems.append("state gate is not exact zero")
        if float(
            self.forward_model.tf_clock_embedding.effective_gate().detach().float()
        ) != 0.0:
            problems.append("clock gate is not exact zero")
        if problems:
            raise RuntimeError(
                "parameter-matched control violates its no-op contract: "
                + "; ".join(problems)
            )

    @staticmethod
    def _expand_sigma(sigma: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if sigma.ndim != 1 or sigma.shape[0] != reference.shape[0]:
            raise ValueError("sigma must have shape [B]")
        return sigma.to(device=reference.device, dtype=reference.dtype).reshape(
            -1, *([1] * (reference.ndim - 1))
        )

    def _paired_training_clocks(
        self,
        batch_size: int,
        device,
        dtype,
        *,
        sample_ids: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.training and self.tf_schedule_mode == "tf_first_cascaded":
            native_timesteps = self._sample_timesteps(batch_size, device)
            native_video_sigma = self._get_sigmas(
                native_timesteps,
                n_dim=2,
                dtype=dtype,
                device=device,
            )[:, 0]
            generator = self._cascade_clock_generator(
                sample_ids=sample_ids,
                timesteps=native_timesteps,
                device=device,
            )
            clocks = self._cascade_clock_sampler(
                batch_size,
                device=device,
                dtype=dtype,
                generator=generator,
                native_video_sigma=native_video_sigma,
            )
            tf_examples = clocks.tf_loss_weight.bool()
            schedule_timesteps = self.noise_scheduler.timesteps.to(
                device=device
            )
            timesteps = torch.where(
                tf_examples,
                schedule_timesteps[0].expand_as(native_timesteps),
                native_timesteps,
            )
            return (
                timesteps,
                clocks.video_sigma,
                clocks.tf_sigma,
                clocks.video_loss_weight,
                clocks.tf_loss_weight,
                # Keep the pre-mask native draw as the independent auxiliary
                # noise key. The model timestep above must be pure noise on
                # auxiliary-loss examples, but keying epsilon by that constant
                # would collapse each clip to one corruption direction.
                native_timesteps,
            )

        if not self.training and self.validation_video_sigmas:
            requested = torch.tensor(
                self.validation_video_sigmas, device=device, dtype=torch.float32
            )
            requested = requested[
                torch.arange(batch_size, device=device) % requested.numel()
            ]
            schedule_sigmas = self.noise_scheduler.sigmas.to(
                device=device, dtype=torch.float32
            )
            schedule_timesteps = self.noise_scheduler.timesteps.to(device=device)
            usable_sigmas = schedule_sigmas[: schedule_timesteps.numel()]
            indices = torch.stack(
                [(usable_sigmas - sigma).abs().argmin() for sigma in requested]
            )
            timesteps = schedule_timesteps[indices]
            video_sigma = usable_sigmas[indices].to(dtype=dtype)
        else:
            timesteps = self._sample_timesteps(batch_size, device)
            expanded = self._get_sigmas(
                timesteps, n_dim=2, dtype=dtype, device=device
            )
            video_sigma = expanded[:, 0]
        if self.tf_schedule_mode == "tf_first_cascaded":
            # Deterministic validation measures the video branch at native Wan
            # noise levels while exposing the imperfect, near-clean V-JEPA
            # condition used by that branch during training.
            tf_sigma = torch.full_like(
                video_sigma,
                self.cascade_validation_tf_sigma,
            )
        else:
            tf_sigma = derive_tf_sigma(
                video_sigma.float(),
                mode=self.tf_schedule_mode,
                tf_lead_logit=self.tf_lead_logit,
            ).to(dtype=dtype)
        ones = torch.ones_like(video_sigma)
        tf_loss_weight = (
            torch.zeros_like(video_sigma)
            if self.tf_schedule_mode == "tf_first_cascaded"
            else ones
        )
        return (
            timesteps,
            video_sigma,
            tf_sigma,
            ones,
            tf_loss_weight,
            timesteps,
        )

    @staticmethod
    def _cascade_clock_generator(
        *,
        sample_ids: torch.Tensor | None,
        timesteps: torch.Tensor,
        device,
    ) -> torch.Generator | None:
        """Use a separate deterministic stream for cascade branch sampling.

        Keeping selector/auxiliary-clock draws off the global generator avoids
        changing Wan LoRA-dropout masks solely because the cascade arm needs
        additional random variables.
        """
        if sample_ids is None:
            return None
        identifiers = sample_ids.reshape(-1)
        clock_steps = timesteps.reshape(-1)
        if identifiers.numel() != clock_steps.numel():
            raise ValueError(
                "sample_ids and timesteps must have the same batch length"
            )
        seed = int(torch.initial_seed()) ^ 0x4341534341444532
        modulus = (1 << 63) - 1
        for sample_id, timestep in zip(
            identifiers.detach().cpu().tolist(),
            clock_steps.detach().cpu().tolist(),
        ):
            seed = (
                seed * 0x9E3779B185EBCA87
                + int(sample_id) * 0xC2B2AE3D27D4EB4F
                + int(timestep)
            ) % modulus
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _masked_per_sample_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        squared_error = (prediction.float() - target.float()).square()
        weights = mask.to(dtype=squared_error.dtype).expand_as(squared_error)
        reduce_dims = tuple(range(1, squared_error.ndim))
        denominator = weights.sum(dim=reduce_dims)
        if (denominator <= 0).any():
            raise RuntimeError("dual flow loss received a sample with no valid future")
        return (squared_error * weights).sum(dim=reduce_dims) / denominator

    @staticmethod
    def _masked_per_sample_nmse(
        estimate: torch.Tensor,
        clean: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        weights = mask.to(dtype=torch.float32).expand_as(estimate)
        reduce_dims = tuple(range(1, estimate.ndim))
        numerator = ((estimate.float() - clean.float()).square() * weights).sum(
            dim=reduce_dims
        )
        denominator = (clean.float().square() * weights).sum(dim=reduce_dims)
        return numerator / denominator.clamp_min(eps)

    @staticmethod
    def _branch_weighted_mean(
        per_sample: torch.Tensor,
        branch_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Average a branch-selected loss over the unchanged global batch."""
        if (
            per_sample.ndim != 1
            or branch_weight.ndim != 1
            or per_sample.shape != branch_weight.shape
        ):
            raise ValueError(
                "per-sample loss and branch weight must have identical [B] shape"
            )
        if not bool(torch.isfinite(branch_weight).all()) or bool(
            (branch_weight < 0).any()
        ):
            raise ValueError("branch weights must be finite and nonnegative")
        return (per_sample * branch_weight.to(per_sample.dtype)).mean()

    @torch.no_grad()
    def _tf_clean(self, rgb: torch.Tensor, latent_shape) -> torch.Tensor:
        if self.time_frequency_transform is None:
            raise RuntimeError(
                "no online auxiliary transform is configured; "
                "the batch must provide auxiliary_target"
            )
        coefficients = self.time_frequency_transform(rgb).detach()
        return self._validate_auxiliary_clean(coefficients, latent_shape)

    def _validate_auxiliary_clean(
        self, coefficients: torch.Tensor, latent_shape
    ) -> torch.Tensor:
        if coefficients.ndim != 5:
            raise RuntimeError(
                "auxiliary target must have shape [B,C,F,H,W]"
            )
        expected_channels = int(self.forward_model.tf_token_adapter.tf_channels)
        if coefficients.shape[1] != expected_channels:
            raise RuntimeError(
                "auxiliary target channel mismatch: "
                f"expected {expected_channels}, got {coefficients.shape[1]}"
            )
        if coefficients.shape[0] != latent_shape[0] or coefficients.shape[2:] != tuple(
            latent_shape[2:]
        ):
            raise RuntimeError(
                "auxiliary target is not aligned to the Wan latent grid: "
                f"auxiliary={tuple(coefficients.shape)}, Wan={tuple(latent_shape)}"
            )
        if not torch.isfinite(coefficients).all():
            raise FloatingPointError("auxiliary target contains non-finite values")
        return coefficients.detach()

    def _resolve_auxiliary_clean(
        self,
        rgb: torch.Tensor,
        latent_shape,
        auxiliary_target: torch.Tensor | None,
    ) -> torch.Tensor:
        if auxiliary_target is None:
            coefficients = self._tf_clean(rgb, latent_shape)
        else:
            coefficients = self._validate_auxiliary_clean(
                auxiliary_target, latent_shape
            )
        return coefficients.to(device=rgb.device, dtype=rgb.dtype)

    def _auxiliary_history_frames(self, video_history_frames: int) -> int:
        mode = getattr(self, "auxiliary_history_mode", "clamp_clean")
        return video_history_frames if mode == "clamp_clean" else 0

    @torch.no_grad()
    def _history_reference(
        self,
        rgb: torch.Tensor,
        latent_shape,
    ) -> tuple[torch.Tensor, int]:
        """Encode only observable RGB when constructing the Wan reference.

        Even though the Wan VAE is described as causal, this explicit boundary
        keeps the conditioning path independent of implementation details such
        as temporal normalization.  Training, paired evaluation, and
        deployment consequently use the exact same history encoder call.
        """
        if rgb.ndim != 5 or rgb.shape[1] < self.num_history_frames:
            raise ValueError(
                f"rgb must provide at least {self.num_history_frames} history frames"
            )
        history_rgb = rgb[:, : self.num_history_frames]
        history_latents = self._encode_clip(history_rgb).to(rgb.dtype)
        expected_history = self.num_history_latent
        if history_latents.shape[2] != expected_history:
            raise RuntimeError(
                "history-only VAE latent length differs from the declared "
                f"history: {history_latents.shape[2]} != {expected_history}"
            )
        if (
            history_latents.shape[0] != int(latent_shape[0])
            or history_latents.shape[1] != int(latent_shape[1])
            or history_latents.shape[3:] != tuple(latent_shape[3:])
            or expected_history >= int(latent_shape[2])
        ):
            raise RuntimeError(
                "history-only VAE latents do not fit the requested rollout grid: "
                f"history={tuple(history_latents.shape)}, grid={tuple(latent_shape)}"
            )
        reference = history_latents.new_zeros(tuple(int(v) for v in latent_shape))
        reference[:, :, :expected_history] = history_latents
        return reference, expected_history

    @staticmethod
    @torch.no_grad()
    def _roll_across_global_batch(state: torch.Tensor) -> torch.Tensor:
        return roll_across_global_batch(state)

    def _training_conditioning_tf(
        self,
        *,
        tf_clean: torch.Tensor,
        tf_noise: torch.Tensor,
        tf_noisy: torch.Tensor,
        tf_sigma_expanded: torch.Tensor,
        history_frames: int,
    ) -> torch.Tensor:
        """Construct the video-only TF condition with matched noise statistics."""
        return make_training_conditioning_tf(
            mode=self.tf_condition_mode,
            tf_clean=tf_clean,
            tf_noise=tf_noise,
            tf_noisy=tf_noisy,
            tf_sigma_expanded=tf_sigma_expanded,
            history_frames=history_frames,
        )

    def _training_tf_noise(
        self,
        tf_clean: torch.Tensor,
        *,
        sample_ids: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw TF corruption without perturbing the video-only RNG path."""
        if self.video_only_control or getattr(
            self, "parameter_matched_control", False
        ):
            # The TF target is zero-weighted and both injection gates are
            # frozen at zero.  A deterministic placeholder avoids advancing
            # the global generator before Wan's LoRA-dropout masks are drawn,
            # preserving the ordinary video RNG/scheduler path.
            return torch.zeros_like(tf_clean)
        if sample_ids is None and timesteps is None:
            # Preserve historical RFFT behavior for datasets without immutable
            # clip identities.
            return torch.randn_like(tf_clean)
        if sample_ids is None or timesteps is None:
            raise ValueError(
                "sample_ids and timesteps must be supplied together"
            )
        sample_ids = sample_ids.reshape(-1)
        timesteps = timesteps.reshape(-1)
        if (
            sample_ids.numel() != tf_clean.shape[0]
            or timesteps.numel() != tf_clean.shape[0]
        ):
            raise ValueError(
                "auxiliary noise identities must have one value per sample"
            )

        # A stateless, independent stream prevents auxiliary corruption from
        # shifting—or reusing—the Wan LoRA-dropout RNG sequence. The seed is a
        # pure function of immutable clip identity, sampled flow timestep, rank
        # seed, and this schema constant, so staged exact resumes reproduce it.
        base_seed = int(torch.initial_seed())
        samples = []
        modulus = (1 << 63) - 1
        for sample_id, timestep in zip(
            sample_ids.detach().cpu().tolist(),
            timesteps.detach().cpu().tolist(),
        ):
            seed = (
                base_seed
                ^ (int(sample_id) * 0x9E3779B185EBCA87)
                ^ (int(timestep) * 0xC2B2AE3D27D4EB4F)
                ^ 0x564A455041323031
            ) % modulus
            generator = torch.Generator(device=tf_clean.device)
            generator.manual_seed(seed)
            samples.append(
                torch.randn(
                    tf_clean.shape[1:],
                    device=tf_clean.device,
                    dtype=tf_clean.dtype,
                    generator=generator,
                )
            )
        return torch.stack(samples)

    @staticmethod
    def _evaluation_noise(
        shape,
        *,
        device,
        dtype,
        base_seed: int,
        sample_ids: torch.Tensor | None,
        stream: int,
        rank: int,
    ) -> torch.Tensor:
        """Draw deterministic evaluation noise, keyed by immutable clip ID.

        With ``sample_ids`` this is invariant to batch packing, rank assignment,
        arm, and milestone.  The legacy rank-keyed stream remains available for
        callers without dataset identities, including the latency benchmark.
        """
        if sample_ids is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(base_seed) + int(rank) + int(stream))
            return torch.randn(
                shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        identifiers = sample_ids.reshape(-1)
        if identifiers.numel() != int(shape[0]):
            raise ValueError("sample_ids must contain one ID per batch element")
        modulus = (1 << 63) - 1
        samples = []
        for sample_id in identifiers.detach().cpu().tolist():
            seed = (
                int(base_seed)
                ^ (int(sample_id) * 0x9E3779B185EBCA87)
                ^ (int(stream) * 0xD6E8FEB86659FD93)
            ) % modulus
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            samples.append(
                torch.randn(
                    tuple(int(value) for value in shape[1:]),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        return torch.stack(samples)

    def _training_objective(
        self,
        video_loss: torch.Tensor,
        tf_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Combine losses while keeping the video-only graph TF-independent."""
        if self.video_only_control or getattr(
            self, "parameter_matched_control", False
        ):
            return video_loss
        return video_loss + self.tf_loss_weight * tf_loss

    def _sampling_conditioning_tf(
        self,
        tf_state: torch.Tensor,
        tf_noise: torch.Tensor,
        tf_sigma_expanded: torch.Tensor,
        history_frames: int,
    ) -> torch.Tensor:
        return make_sampling_conditioning_tf(
            mode=self.tf_condition_mode,
            tf_state=tf_state,
            tf_noise=tf_noise,
            tf_sigma_expanded=tf_sigma_expanded,
            history_frames=history_frames,
        )

    def _record_sigma_metrics(
        self,
        video_nmse: torch.Tensor,
        tf_nmse: torch.Tensor,
        video_sigma: torch.Tensor,
    ) -> None:
        # These one-call denoising diagnostics use TF states corrupted from the
        # same ground-truth clip.  They measure the supervised denoising problem,
        # not autonomous joint generation; keep that distinction explicit in
        # telemetry so a teacher-forced gain cannot be reported as an inference
        # acceleration result.
        self.aux_losses["teacher_forced/video_x0_nmse"] = (
            video_nmse.mean().detach()
        )
        self.aux_losses["teacher_forced/tf_x0_nmse"] = tf_nmse.mean().detach()
        if not self.training and self.validation_video_sigmas:
            for requested in self.validation_video_sigmas:
                nearest = (video_sigma.float() - requested).abs()
                selected = nearest == nearest.min()
                self.aux_losses[
                    f"teacher_forced/video_x0_nmse/sigma_{requested:.2f}"
                ] = video_nmse[selected].mean().detach()
                self.aux_losses[
                    f"teacher_forced/tf_x0_nmse/sigma_{requested:.2f}"
                ] = tf_nmse[selected].mean().detach()

    def _sampling_schedule(
        self,
        num_steps: int,
        *,
        device: torch.device,
    ):
        """Return paired sigmas, model timesteps, and auxiliary-only calls."""
        if self.tf_schedule_mode != "tf_first_cascaded":
            self.sample_scheduler.set_timesteps(num_steps, device=device)
            native_video_sigmas = self.sample_scheduler.sigmas.to(
                device=device, dtype=torch.float32
            )[: num_steps + 1]
            schedule = pair_video_sigma_schedule(
                native_video_sigmas,
                mode=self.tf_schedule_mode,
                tf_lead_logit=self.tf_lead_logit,
            )
            return schedule, self.sample_scheduler.timesteps, 0

        if num_steps == 1:
            # A strict two-phase cascade cannot exist at one call. Retain the
            # one-call point as an explicitly degenerate joint negative control
            # so the candidate is still compared with the full baseline NFE
            # frontier rather than silently omitting its cheapest endpoint.
            self.sample_scheduler.set_timesteps(1, device=device)
            native_video_sigmas = self.sample_scheduler.sigmas.to(
                device=device, dtype=torch.float32
            )[:2]
            return (
                pair_video_sigma_schedule(
                    native_video_sigmas,
                    mode="aligned",
                ),
                self.sample_scheduler.timesteps,
                0,
            )

        tf_steps, video_steps = cascaded_step_counts(
            num_steps,
            tf_fraction=self.cascade_inference_tf_fraction,
        )
        self.sample_scheduler.set_timesteps(video_steps, device=device)
        video_timesteps = self.sample_scheduler.timesteps
        native_video_sigmas = self.sample_scheduler.sigmas.to(
            device=device, dtype=torch.float32
        )[: video_steps + 1]
        schedule = pair_native_cascaded_sigma_schedule(
            native_video_sigmas,
            total_steps=num_steps,
            tf_fraction=self.cascade_inference_tf_fraction,
        )
        model_timesteps = torch.cat(
            [
                video_timesteps[:1].expand(tf_steps),
                video_timesteps,
            ]
        )
        return schedule, model_timesteps, tf_steps

    def _sampling_condition_source_for_step(
        self,
        condition_source: str,
        *,
        step_index: int,
        tf_only_steps: int,
    ) -> str:
        """Delay video-conditioning controls until the cascade phase boundary."""
        if (
            self.tf_schedule_mode == "tf_first_cascaded"
            and step_index < tf_only_steps
            and condition_source in {"off", "autonomous_shuffled"}
        ):
            return "autonomous"
        return condition_source

    @staticmethod
    def _intra_forward_condition_source(condition_source: str) -> str:
        """Map public deployable controls to the block-14 intervention."""
        mapping = {
            "autonomous": "aligned",
            "autonomous_shuffled": "shuffled",
            "off": "off",
        }
        try:
            return mapping[condition_source]
        except KeyError as exc:
            raise RuntimeError(
                "the deployable intra-forward screen forbids oracle midpoint "
                f"conditioning: {condition_source!r}"
            ) from exc

    def forward(
        self,
        rgb: torch.Tensor,
        actions: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Train both future states; never condition on clean future TF."""
        self._ensure_video_only_runtime_contract()
        self.aux_losses = {}
        morphology_index = kwargs.get("morphology_index", None)
        device = rgb.device

        video_clean = self._encode_clip(rgb).to(rgb.dtype)
        batch_size, _, latent_frames, height, width = video_clean.shape
        reference, history_frames = self._history_reference(
            rgb, video_clean.shape
        )
        auxiliary_history_frames = self._auxiliary_history_frames(
            history_frames
        )

        _, z_control, _ = self._latent_actions(
            rgb,
            actions,
            morphology_index,
            latent_frames,
            history_frames,
        )
        z_control = z_control.to(rgb.dtype)

        # Preserve the production video RNG/scheduler path, then draw TF noise.
        video_noise = torch.randn_like(video_clean)
        (
            timesteps,
            video_sigma,
            tf_sigma,
            video_loss_weight,
            tf_loss_weight,
            tf_noise_timesteps,
        ) = self._paired_training_clocks(
            batch_size,
            device,
            video_clean.dtype,
            sample_ids=kwargs.get("clip_index"),
        )
        video_sigma_expanded = self._expand_sigma(video_sigma, video_clean)
        video_noisy = (
            (1.0 - video_sigma_expanded) * video_clean
            + video_sigma_expanded * video_noise
        )
        video_target = video_noise - video_clean

        tf_clean = self._resolve_auxiliary_clean(
            rgb,
            video_clean.shape,
            kwargs.get("auxiliary_target"),
        )
        tf_noise = self._training_tf_noise(
            tf_clean,
            sample_ids=kwargs.get("clip_index"),
            timesteps=(
                tf_noise_timesteps if "clip_index" in kwargs else None
            ),
        )
        tf_sigma_expanded = self._expand_sigma(tf_sigma, tf_clean)
        tf_noisy = (1.0 - tf_sigma_expanded) * tf_clean + tf_sigma_expanded * tf_noise
        # Causal hand-designed targets may clamp observed history.  A
        # bidirectional learned target (V-JEPA) must diffuse all bins because
        # even nominal prefix tokens contain full-clip context.
        if auxiliary_history_frames:
            tf_noisy = tf_noisy.clone()
            tf_noisy[:, :, :auxiliary_history_frames] = tf_clean[
                :, :, :auxiliary_history_frames
            ]
        tf_target = tf_noise - tf_clean
        conditioning_tf = self._training_conditioning_tf(
            tf_clean=tf_clean,
            tf_noise=tf_noise,
            tf_noisy=tf_noisy,
            tf_sigma_expanded=tf_sigma_expanded,
            history_frames=auxiliary_history_frames,
        )

        context = self._build_context(batch_size, device, rgb.dtype)
        clip_fea = self._build_clip(batch_size, device, rgb.dtype)
        forward_kwargs = {}
        if bool(
            getattr(self.forward_model, "intra_forward_forcing_enabled", False)
        ):
            forward_kwargs["intra_forward_condition_source"] = (
                "aligned" if self.condition_on_tf else "off"
            )
        prediction = self.forward_model(
            video_noisy,
            timesteps,
            z_control,
            reference,
            context,
            clip_fea,
            noisy_tf=tf_noisy,
            conditioning_tf=conditioning_tf,
            tf_sigma=tf_sigma,
            condition_on_tf=self.condition_on_tf,
            condition_on_tf_clock=self.condition_on_tf_clock,
            **forward_kwargs,
        )
        if not isinstance(prediction, DualWanOutput):
            raise RuntimeError("dual Wan forward did not return both velocities")

        loss_mask = self._build_loss_mask(rgb, mask, video_clean.shape)
        future_mask = loss_mask[:, :, history_frames:]
        video_per_sample = self._masked_per_sample_mse(
            prediction.video_velocity[:, :, history_frames:],
            video_target[:, :, history_frames:],
            future_mask,
        )
        auxiliary_mask = loss_mask[:, :, auxiliary_history_frames:]
        tf_per_sample = self._masked_per_sample_mse(
            prediction.tf_velocity[:, :, auxiliary_history_frames:],
            tf_target[:, :, auxiliary_history_frames:],
            auxiliary_mask,
        )
        video_loss = self._branch_weighted_mean(
            video_per_sample,
            video_loss_weight,
        )
        tf_loss = self._branch_weighted_mean(
            tf_per_sample,
            tf_loss_weight,
        )
        # Do not attach the zero-weight TF graph to the video-only objective.
        # Besides avoiding needless TF gradients/weight decay, this prevents a
        # non-finite diagnostic TF loss from contaminating a finite video loss
        # through IEEE ``0 * NaN`` semantics.
        total_loss = self._training_objective(video_loss, tf_loss)

        video_x0 = video_noisy - video_sigma_expanded * prediction.video_velocity
        tf_x0 = tf_noisy - tf_sigma_expanded * prediction.tf_velocity
        video_nmse = self._masked_per_sample_nmse(
            video_x0[:, :, history_frames:],
            video_clean[:, :, history_frames:],
            future_mask,
        )
        tf_nmse = self._masked_per_sample_nmse(
            tf_x0[:, :, auxiliary_history_frames:],
            tf_clean[:, :, auxiliary_history_frames:],
            auxiliary_mask,
        )

        self.aux_losses["flow_loss"] = video_loss.detach()
        self.aux_losses["video_flow_loss"] = video_loss.detach()
        self.aux_losses["tf_flow_loss"] = tf_loss.detach()
        self.aux_losses["clock/video_sigma_mean"] = video_sigma.float().mean().detach()
        self.aux_losses["clock/tf_sigma_mean"] = tf_sigma.float().mean().detach()
        self.aux_losses["clock/video_loss_weight_mean"] = (
            video_loss_weight.float().mean().detach()
        )
        self.aux_losses["clock/tf_loss_weight_mean"] = (
            tf_loss_weight.float().mean().detach()
        )
        self.aux_losses["state/video_noisy_rms"] = (
            video_noisy.float().square().mean().sqrt().detach()
        )
        self.aux_losses["state/tf_noisy_rms"] = (
            tf_noisy.float().square().mean().sqrt().detach()
        )
        self.aux_losses["condition/tf_token_rms"] = (
            prediction.tf_condition_tokens.float().square().mean().sqrt().detach()
        )
        self.aux_losses["condition/ztf_enabled"] = video_loss.new_tensor(
            float(self.condition_on_tf)
        )
        self.aux_losses["condition/ztf_clock_enabled"] = video_loss.new_tensor(
            float(self.condition_on_tf_clock)
        )
        self.aux_losses[
            "condition/auxiliary_history_frames"
        ] = video_loss.new_tensor(float(auxiliary_history_frames))
        self.aux_losses["condition/mode_code"] = video_loss.new_tensor(
            float(TF_CONDITION_MODE_CODES[self.tf_condition_mode])
        )
        self.aux_losses["condition/video_only_control"] = video_loss.new_tensor(
            float(self.video_only_control)
        )
        self.aux_losses[
            "condition/parameter_matched_control"
        ] = video_loss.new_tensor(float(self.parameter_matched_control))
        self.aux_losses["condition/tf_loss_weight"] = video_loss.new_tensor(
            self.tf_loss_weight
        )
        state_gate = self.forward_model.tf_token_adapter.effective_gate()
        clock_gate = self.forward_model.tf_clock_embedding.effective_gate()
        self.aux_losses["condition/state_gate"] = torch.as_tensor(
            state_gate, device=video_loss.device, dtype=video_loss.dtype
        ).detach()
        self.aux_losses["condition/clock_gate"] = torch.as_tensor(
            clock_gate, device=video_loss.device, dtype=video_loss.dtype
        ).detach()
        for key, value in prediction.tf_condition_telemetry.items():
            self.aux_losses[f"condition/{key}"] = value.detach()
        self._record_sigma_metrics(video_nmse, tf_nmse, video_sigma)

        if not torch.isfinite(total_loss):
            logger.error(
                "non-finite dual loss: video=%s TF=%s, condition_on_tf=%s",
                bool(torch.isfinite(video_loss)),
                bool(torch.isfinite(tf_loss)),
                self.condition_on_tf,
            )
        return total_loss

    @torch.no_grad()
    def _sample_future(
        self,
        rgb,
        actions=None,
        morphology_index=None,
        auxiliary_target=None,
        collect_artifacts: bool = True,
        deployment_mode: bool = False,
        sample_ids: torch.Tensor | None = None,
    ):
        """Run independent joint samplers at each requested NFE.

        Every NFE starts from the same deterministic video/TF noise. Under an
        overlapping schedule, one step is a paired negative control because
        matched and shuffled see identical future TF noise. A strict cascade
        requires at least two calls so both branches receive a nonempty phase.

        ``collect_artifacts=False`` is the deployable latency path.  It avoids
        every trajectory and CPU evidence copy while retaining the real
        encoder, Wan calls, auxiliary Euler updates, and RGB decoder.  A small
        Python-only counter record remains available in
        ``_last_sampling_counters``; an untimed artifact-collecting call is
        still required when auditing tensors and provenance.

        ``deployment_mode=True`` accepts exactly the observed RGB history,
        allocates the unavailable future latent slots from the declared model
        horizon, and never encodes or decodes ground-truth future RGB.  It is
        therefore the only valid path for latency or closed-loop deployment.
        The ordinary full-clip path is retained for paired quality scoring.
        """
        self._ensure_video_only_runtime_contract()
        if rgb.ndim != 5:
            raise ValueError("rgb must have shape [B,T,C,H,W]")
        batch_size = rgb.shape[0]
        profile_stages = bool(getattr(self, "profile_sampling_stages", False))
        if profile_stages:
            if not rgb.is_cuda:
                raise RuntimeError("sampling-stage latency profiling requires CUDA")
            if collect_artifacts:
                raise RuntimeError(
                    "sampling-stage latency profiling forbids artifact collection"
                )
            if len(self.evaluation_condition_sources) != 1 or len(
                self.evaluation_nfe_steps
            ) != 1:
                raise RuntimeError(
                    "sampling-stage latency profiling requires one source/NFE cell"
                )
            torch.cuda.synchronize(rgb.device)
            end_to_end_started_ns = time.perf_counter_ns()
            history_encode_started_ns = end_to_end_started_ns
            self._last_sampling_profile = None
        if deployment_mode:
            if rgb.shape[1] != self.num_history_frames:
                raise ValueError(
                    "deployable sampling accepts exactly the observed history: "
                    f"expected {self.num_history_frames} frames, got {rgb.shape[1]}"
                )
            if auxiliary_target is not None:
                raise ValueError(
                    "deployable sampling cannot consume a clean auxiliary target"
                )
            nondeployable_sources = set(self.evaluation_condition_sources) - {
                "autonomous",
                "off",
                "autonomous_shuffled",
            }
            if nondeployable_sources:
                raise ValueError(
                    "deployable sampling cannot use oracle condition sources: "
                    f"{sorted(nondeployable_sources)}"
                )
            history_latents = self._encode_clip(rgb).to(rgb.dtype)
            if history_latents.shape[2] != self.num_history_latent:
                raise RuntimeError(
                    "history-only VAE latent length differs from the declared "
                    f"history: {history_latents.shape[2]} != "
                    f"{self.num_history_latent}"
                )
            latent_frames = self.rgb_tokenizer.latent_temporal_len(
                self.num_history_frames + self.num_future_frames
            )
            history_frames = history_latents.shape[2]
            if latent_frames <= history_frames:
                raise RuntimeError(
                    "declared future horizon does not create future latent slots"
                )
            video_shape = (
                batch_size,
                history_latents.shape[1],
                latent_frames,
                history_latents.shape[3],
                history_latents.shape[4],
            )
            video_clean = None
            reference = history_latents.new_zeros(video_shape)
            reference[:, :, :history_frames] = history_latents
        else:
            video_clean = self._encode_clip(rgb).to(rgb.dtype)
            _, _, latent_frames, _, _ = video_clean.shape
            video_shape = tuple(video_clean.shape)
            reference, history_frames = self._history_reference(
                rgb, video_shape
            )
        if profile_stages:
            torch.cuda.synchronize(rgb.device)
            history_encode_latency_ms = (
                time.perf_counter_ns() - history_encode_started_ns
            ) / 1_000_000.0
        auxiliary_history_frames = self._auxiliary_history_frames(
            history_frames
        )
        if deployment_mode and auxiliary_history_frames:
            raise RuntimeError(
                "deployable sampling cannot clamp a clean auxiliary history; "
                "use auxiliary_history_mode=diffuse_all"
            )
        _, z_control, _ = self._latent_actions(
            rgb,
            actions,
            morphology_index,
            latent_frames,
            history_frames,
        )
        z_control = z_control.to(rgb.dtype)
        context = self._build_context(batch_size, rgb.device, rgb.dtype)
        clip_fea = self._build_clip(batch_size, rgb.device, rgb.dtype)

        needs_auxiliary_clean = (
            auxiliary_target is not None
            or auxiliary_history_frames > 0
            or any(
                source in {"oracle_matched", "oracle_shuffled"}
                for source in self.evaluation_condition_sources
            )
        )
        tf_clean = (
            self._resolve_auxiliary_clean(
                rgb, video_shape, auxiliary_target
            )
            if needs_auxiliary_clean
            else None
        )
        ground_truth_pixels = None
        if video_clean is not None:
            ground_truth_pixels = self.rgb_tokenizer.decode_temporal(
                video_clean, out_hw=(rgb.shape[-2], rgb.shape[-1])
            )
            future_pixel_frames = min(
                self.num_future_frames, ground_truth_pixels.shape[2]
            )
        else:
            future_pixel_frames = self.num_future_frames

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        auxiliary_shape = (
            batch_size,
            int(self.forward_model.tf_token_adapter.tf_channels),
            *video_shape[2:],
        )
        if sample_ids is None:
            # Preserve the historical paired evaluation stream for callers
            # without immutable dataset identities.
            generator = torch.Generator(device=rgb.device)
            generator.manual_seed(self.evaluation_noise_seed + rank)
            initial_video_state = torch.randn(
                video_shape,
                device=rgb.device,
                dtype=rgb.dtype,
                generator=generator,
            )
            initial_tf_noise = torch.randn(
                auxiliary_shape,
                device=rgb.device,
                dtype=rgb.dtype,
                generator=generator,
            )
        else:
            initial_video_state = self._evaluation_noise(
                video_shape,
                device=rgb.device,
                dtype=rgb.dtype,
                base_seed=self.evaluation_noise_seed,
                sample_ids=sample_ids,
                stream=0,
                rank=rank,
            )
            initial_tf_noise = self._evaluation_noise(
                auxiliary_shape,
                device=rgb.device,
                dtype=rgb.dtype,
                base_seed=self.evaluation_noise_seed,
                sample_ids=sample_ids,
                stream=1,
                rank=rank,
            )
        initial_tf_state = initial_tf_noise.clone()
        if auxiliary_history_frames:
            if tf_clean is None:
                raise RuntimeError("clean auxiliary history is unavailable")
            initial_tf_state[:, :, :auxiliary_history_frames] = tf_clean[
                :, :, :auxiliary_history_frames
            ]

        artifact_batch_limit = getattr(self, "artifact_batch_limit", 1)
        evidence_count = (
            batch_size
            if artifact_batch_limit is None
            else min(batch_size, artifact_batch_limit)
        )
        evidence_slice = slice(0, evidence_count)
        artifacts = None
        if collect_artifacts:
            artifacts = {
                "video_initial_state": (
                    initial_video_state[evidence_slice]
                    .detach()
                    .cpu()
                    .to(torch.float16)
                ),
                "reference_latents": (
                    reference[evidence_slice]
                    .detach()
                    .cpu()
                    .to(torch.float16)
                ),
                "tf_initial_state": (
                    initial_tf_state[evidence_slice]
                    .detach()
                    .cpu()
                    .to(torch.float16)
                ),
                "tf_initial_noise": (
                    initial_tf_noise[evidence_slice]
                    .detach()
                    .cpu()
                    .to(torch.float16)
                ),
                "deployment_mode": torch.tensor(
                    [int(deployment_mode)], dtype=torch.int64
                ),
                "history_latent_frames": torch.tensor([history_frames]),
                "auxiliary_history_latent_frames": torch.tensor(
                    [auxiliary_history_frames]
                ),
                "auxiliary_clean_available": torch.tensor(
                    [int(tf_clean is not None)], dtype=torch.int64
                ),
                "condition_on_tf": torch.tensor([int(self.condition_on_tf)]),
                "condition_mode_code": torch.tensor(
                    [TF_CONDITION_MODE_CODES[self.tf_condition_mode]]
                ),
                "video_only_control": torch.tensor(
                    [int(self.video_only_control)], dtype=torch.int64
                ),
                "parameter_matched_control": torch.tensor(
                    [int(getattr(self, "parameter_matched_control", False))],
                    dtype=torch.int64,
                ),
                "tf_loss_weight": torch.tensor(
                    [self.tf_loss_weight], dtype=torch.float32
                ),
                "effective_state_gate": (
                    self.forward_model.tf_token_adapter.effective_gate()
                    .detach()
                    .cpu()
                    .float()
                    .reshape(1)
                ),
                "effective_clock_gate": (
                    self.forward_model.tf_clock_embedding.effective_gate()
                    .detach()
                    .cpu()
                    .float()
                    .reshape(1)
                ),
                "evaluation_noise_seed": torch.tensor(
                    [self.evaluation_noise_seed], dtype=torch.int64
                ),
                "sample_ids": (
                    sample_ids.reshape(-1)[evidence_slice]
                    .detach()
                    .cpu()
                    .to(torch.int64)
                    if sample_ids is not None
                    else torch.full(
                        (evidence_count,), -1, dtype=torch.int64
                    )
                ),
                "evaluation_nfe_steps": torch.tensor(
                    self.evaluation_nfe_steps, dtype=torch.int64
                ),
                "evaluation_condition_source_codes": torch.tensor(
                    [
                        EVALUATION_CONDITION_SOURCE_CODES[source]
                        for source in self.evaluation_condition_sources
                    ],
                    dtype=torch.int64,
                ),
                "oracle_sources_are_leakage": torch.tensor(
                    [1], dtype=torch.int64
                ),
                # The frozen teacher is not registered on this model and cannot
                # be invoked by the autonomous sampler. Keep an explicit
                # machine-checked field in every evaluation artifact.
                "online_teacher_call_count": torch.tensor(
                    [0], dtype=torch.int64
                ),
            }
            if video_clean is not None and ground_truth_pixels is not None:
                artifacts["video_clean"] = (
                    video_clean[evidence_slice]
                    .detach()
                    .cpu()
                    .to(torch.float16)
                )
                artifacts["ground_truth_future_uint8"] = (
                    (
                        ground_truth_pixels[
                            evidence_slice, :, -future_pixel_frames:
                        ]
                        .float()
                        .clamp(-1.0, 1.0)
                        + 1.0
                    )
                    .mul(127.5)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                )
            if tf_clean is not None:
                artifacts["tf_clean"] = (
                    tf_clean[evidence_slice]
                    .detach()
                    .cpu()
                    .to(torch.float16)
                )
        primary_pixels = None
        sampling_call_counts: dict[str, int] = {}

        wrong_tf_clean = None
        if "oracle_shuffled" in self.evaluation_condition_sources:
            if tf_clean is None:
                raise RuntimeError("oracle evaluation requires auxiliary_target")
            wrong_tf_clean = self._roll_across_global_batch(tf_clean)

        primary_condition_source = self.evaluation_condition_sources[0]
        for condition_source in self.evaluation_condition_sources:
            source_infix = (
                "" if condition_source == "autonomous" else f"_{condition_source}"
            )
            for num_steps in self.evaluation_nfe_steps:
                video_state = initial_video_state.clone()
                tf_state = initial_tf_state.clone()
                backbone_call_count = 0
                midpoint_head_call_count = 0
                wan_latency_ms = 0.0
                midpoint_overhead_latency_ms = 0.0
                schedule, model_timesteps, tf_only_steps = (
                    self._sampling_schedule(
                        num_steps,
                        device=rgb.device,
                    )
                )

                is_primary_result = (
                    condition_source == primary_condition_source
                    and num_steps == self.viz_num_steps
                )
                capture_this_trajectory = (
                    collect_artifacts
                    and self.capture_latent_trajectories
                    and is_primary_result
                )
                video_trajectory = (
                    [
                        video_state[evidence_slice]
                        .detach()
                        .cpu()
                        .to(torch.float16)
                    ]
                    if capture_this_trajectory
                    else []
                )
                tf_trajectory = (
                    [
                        tf_state[evidence_slice]
                        .detach()
                        .cpu()
                        .to(torch.float16)
                    ]
                    if capture_this_trajectory
                    else []
                )
                video_x0_trajectory = []
                tf_x0_trajectory = []

                for index, timestep in enumerate(model_timesteps):
                    video_sigma = schedule.video[index]
                    next_video_sigma = schedule.video[index + 1]
                    tf_sigma = schedule.time_frequency[index]
                    next_tf_sigma = schedule.time_frequency[index + 1]
                    timesteps = timestep.expand(batch_size).to(rgb.device)
                    tf_batch_sigma = tf_sigma.expand(batch_size).to(
                        dtype=rgb.dtype
                    )
                    tf_sigma_expanded = self._expand_sigma(
                        tf_batch_sigma, tf_state
                    )
                    # For a strict cascade, generate one identical auxiliary
                    # trajectory before applying the video-conditioning
                    # interventions. Otherwise ``off`` or ``shuffled`` would
                    # also change the auxiliary predictor and would not isolate
                    # whether the video phase uses the final sample-aligned
                    # V-JEPA state.
                    effective_condition_source = (
                        self._sampling_condition_source_for_step(
                            condition_source,
                            step_index=index,
                            tf_only_steps=tf_only_steps,
                        )
                    )

                    if effective_condition_source in {
                        "autonomous",
                        "autonomous_shuffled",
                    }:
                        # ``autonomous_shuffled`` is an evaluation-only,
                        # same-checkpoint intervention.  It never reads
                        # ``tf_clean``: at every call it keeps this sample's
                        # corruption noise and observed history, while rolling
                        # only the generated, noise-subtracted future residual
                        # across the effective global batch.
                        #
                        # Keep the historical ``autonomous`` path byte-for-byte
                        # equivalent when the new source is not requested.
                        if bool(
                            getattr(
                                self.forward_model,
                                "intra_forward_forcing_enabled",
                                False,
                            )
                        ):
                            # The intra-forward head consumes ``tf_state``
                            # directly and performs the sole aligned/shuffled
                            # intervention after producing q0_hat.  Avoid an
                            # unrelated pre-backbone global collective here.
                            conditioning_tf = tf_state
                        elif effective_condition_source == "autonomous":
                            conditioning_tf = self._sampling_conditioning_tf(
                                tf_state,
                                initial_tf_noise,
                                tf_sigma_expanded,
                                auxiliary_history_frames,
                            )
                        else:
                            conditioning_tf = make_sampling_conditioning_tf(
                                mode="shuffled",
                                tf_state=tf_state,
                                tf_noise=initial_tf_noise,
                                tf_sigma_expanded=tf_sigma_expanded,
                                history_frames=auxiliary_history_frames,
                            )
                        use_tf_condition = self.condition_on_tf
                        use_tf_clock_condition = getattr(
                            self,
                            "condition_on_tf_clock",
                            self.condition_on_tf,
                        )
                    elif effective_condition_source == "off":
                        conditioning_tf = tf_state
                        use_tf_condition = False
                        use_tf_clock_condition = False
                    else:
                        if tf_clean is None:
                            raise RuntimeError(
                                "oracle evaluation requires auxiliary_target"
                            )
                        oracle_clean = (
                            tf_clean
                            if condition_source == "oracle_matched"
                            else wrong_tf_clean
                        )
                        if oracle_clean is None:
                            raise RuntimeError(
                                "oracle shuffled source was not prepared"
                            )
                        conditioning_tf = make_oracle_conditioning_tf(
                            tf_clean=tf_clean,
                            tf_noise=initial_tf_noise,
                            tf_sigma_expanded=tf_sigma_expanded,
                            history_frames=auxiliary_history_frames,
                            wrong_tf_clean=(
                                None
                                if condition_source == "oracle_matched"
                                else oracle_clean
                            ),
                        )
                        use_tf_condition = True
                        use_tf_clock_condition = self.condition_on_tf_clock

                    forward_kwargs = {
                        "noisy_tf": tf_state,
                        "conditioning_tf": conditioning_tf,
                        "tf_sigma": tf_batch_sigma,
                        "condition_on_tf": use_tf_condition,
                    }
                    if bool(
                        getattr(
                            self.forward_model,
                            "intra_forward_forcing_enabled",
                            False,
                        )
                    ):
                        forward_kwargs["intra_forward_condition_source"] = (
                            self._intra_forward_condition_source(
                                effective_condition_source
                            )
                        )
                    # Preserve compatibility with historical test doubles and
                    # checkpoints that predate the separately switchable clock.
                    if hasattr(self, "condition_on_tf_clock"):
                        forward_kwargs[
                            "condition_on_tf_clock"
                        ] = use_tf_clock_condition
                    if profile_stages:
                        torch.cuda.synchronize(rgb.device)
                        wan_started_ns = time.perf_counter_ns()
                        self.forward_model.profile_intra_forward_latency = True
                    try:
                        prediction = self.forward_model(
                            video_state,
                            timesteps,
                            z_control,
                            reference,
                            context,
                            clip_fea,
                            **forward_kwargs,
                        )
                    finally:
                        if profile_stages:
                            torch.cuda.synchronize(rgb.device)
                            wan_latency_ms += (
                                time.perf_counter_ns() - wan_started_ns
                            ) / 1_000_000.0
                            self.forward_model.profile_intra_forward_latency = False
                    backbone_call_count += 1
                    if not isinstance(prediction, DualWanOutput):
                        raise RuntimeError(
                            "dual Wan sampler did not return both velocities"
                        )
                    if bool(
                        getattr(
                            self.forward_model,
                            "intra_forward_forcing_enabled",
                            False,
                        )
                    ):
                        calls = prediction.tf_condition_telemetry.get(
                            "midpoint_head_calls"
                        )
                        block = prediction.tf_condition_telemetry.get(
                            "midpoint_block_index"
                        )
                        if (
                            calls is None
                            or block is None
                            or float(calls.detach().float()) != 1.0
                            or float(block.detach().float()) != 14.0
                        ):
                            raise RuntimeError(
                                "intra-forward call telemetry violated the frozen seam"
                            )
                        midpoint_head_call_count += 1
                        midpoint_overhead_latency_ms += float(
                            prediction.tf_condition_telemetry[
                                "midpoint_overhead_latency_ms"
                            ].detach().float()
                        )

                    video_x0 = (
                        video_state.float()
                        - video_sigma.float()
                        * prediction.video_velocity.float()
                    )
                    tf_x0 = (
                        tf_state.float()
                        - tf_sigma.float() * prediction.tf_velocity.float()
                    )
                    if capture_this_trajectory:
                        video_x0_trajectory.append(
                            video_x0[evidence_slice]
                            .detach()
                            .cpu()
                            .to(torch.float16)
                        )
                        tf_x0_trajectory.append(
                            tf_x0[evidence_slice]
                            .detach()
                            .cpu()
                            .to(torch.float16)
                        )

                    if index >= tf_only_steps:
                        video_state = self.sample_scheduler.step(
                            prediction.video_velocity.float(),
                            timestep,
                            video_state.float(),
                        ).prev_sample.to(rgb.dtype)
                    # The video objective supervises future latent slots only.
                    # Keep observed history exactly on its known forward-noise
                    # trajectory so every subsequent Wan call matches the
                    # training distribution, then reaches the clean reference
                    # at sigma=0 before the causal VAE decoder runs.
                    history_sigma = next_video_sigma.to(
                        device=video_state.device,
                        dtype=video_state.dtype,
                    )
                    video_state[:, :, :history_frames] = (
                        (1.0 - history_sigma)
                        * reference[:, :, :history_frames]
                        + history_sigma
                        * initial_video_state[:, :, :history_frames]
                    )
                    tf_state = euler_flow_step(
                        tf_state.float(),
                        prediction.tf_velocity.float(),
                        tf_sigma,
                        next_tf_sigma,
                    ).to(rgb.dtype)
                    if auxiliary_history_frames:
                        if tf_clean is None:
                            raise RuntimeError(
                                "clean auxiliary history is unavailable"
                            )
                        tf_state[
                            :, :, :auxiliary_history_frames
                        ] = tf_clean[:, :, :auxiliary_history_frames]
                    if capture_this_trajectory:
                        video_trajectory.append(
                            video_state[evidence_slice]
                            .detach()
                            .cpu()
                            .to(torch.float16)
                        )
                        tf_trajectory.append(
                            tf_state[evidence_slice]
                            .detach()
                            .cpu()
                            .to(torch.float16)
                        )

                if profile_stages:
                    torch.cuda.synchronize(rgb.device)
                    decode_started_ns = time.perf_counter_ns()
                predicted_pixels = self.rgb_tokenizer.decode_temporal(
                    video_state, out_hw=(rgb.shape[-2], rgb.shape[-1])
                )
                if profile_stages:
                    torch.cuda.synchronize(rgb.device)
                    decode_latency_ms = (
                        time.perf_counter_ns() - decode_started_ns
                    ) / 1_000_000.0
                if backbone_call_count != num_steps:
                    raise RuntimeError(
                        "reported NFE does not equal actual Wan calls: "
                        f"{backbone_call_count} != {num_steps}"
                    )
                counter_key = f"{condition_source}:nfe_{num_steps}"
                sampling_call_counts[counter_key] = backbone_call_count
                if artifacts is not None:
                    artifacts[
                        f"wan_call_count{source_infix}_nfe_{num_steps}"
                    ] = torch.tensor([backbone_call_count], dtype=torch.int64)
                    if bool(
                        getattr(
                            self.forward_model,
                            "intra_forward_forcing_enabled",
                            False,
                        )
                    ):
                        artifacts[
                            f"midpoint_head_call_count{source_infix}_nfe_{num_steps}"
                        ] = torch.tensor(
                            [midpoint_head_call_count], dtype=torch.int64
                        )
                    artifacts[
                        f"video_final{source_infix}_nfe_{num_steps}"
                    ] = (
                        video_state[evidence_slice]
                        .detach()
                        .cpu()
                        .to(torch.float16)
                    )
                    artifacts[
                        f"tf_final{source_infix}_nfe_{num_steps}"
                    ] = (
                        tf_state[evidence_slice]
                        .detach()
                        .cpu()
                        .to(torch.float16)
                    )
                    artifacts[
                        f"decoded_future{source_infix}_nfe_{num_steps}"
                    ] = (
                        (
                            predicted_pixels[
                                evidence_slice, :, -future_pixel_frames:
                            ]
                            .float()
                            .clamp(-1.0, 1.0)
                            + 1.0
                        )
                        .mul(127.5)
                        .round()
                        .to(torch.uint8)
                        .cpu()
                    )

                if capture_this_trajectory:
                    assert artifacts is not None
                    artifacts.update(
                        {
                            "video_trajectory": torch.stack(
                                video_trajectory
                            ),
                            "tf_trajectory": torch.stack(tf_trajectory),
                            "video_x0_trajectory": torch.stack(
                                video_x0_trajectory
                            ),
                            "tf_x0_trajectory": torch.stack(
                                tf_x0_trajectory
                            ),
                            "video_sigmas": (
                                schedule.video.detach().cpu().float()
                            ),
                            "tf_sigmas": (
                                schedule.time_frequency.detach().cpu().float()
                            ),
                        }
                    )
                if is_primary_result:
                    primary_pixels = predicted_pixels
                if profile_stages:
                    torch.cuda.synchronize(rgb.device)
                    self._last_sampling_profile = {
                        "condition_source": condition_source,
                        "nfe": num_steps,
                        "history_encode_latency_ms": history_encode_latency_ms,
                        "wan_latency_ms": wan_latency_ms,
                        "midpoint_overhead_latency_ms": (
                            midpoint_overhead_latency_ms
                        ),
                        "decode_latency_ms": decode_latency_ms,
                        "end_to_end_latency_ms": (
                            time.perf_counter_ns() - end_to_end_started_ns
                        )
                        / 1_000_000.0,
                    }

        if primary_pixels is None:
            raise RuntimeError(
                f"viz_num_steps={self.viz_num_steps} was not evaluated"
            )
        self._last_sampling_counters = {
            "wan_calls_by_source_nfe": sampling_call_counts,
            "wan_calls_total": sum(sampling_call_counts.values()),
            "online_teacher_calls": 0,
            "auxiliary_clean_available": int(tf_clean is not None),
            "artifacts_collected": int(collect_artifacts),
            "deployment_mode": int(deployment_mode),
        }
        if artifacts is not None:
            self._visualization_artifacts = artifacts
        return primary_pixels, ground_truth_pixels

    @torch.no_grad()
    def sample_future_deployable(
        self,
        history_rgb,
        actions,
        morphology_index=None,
        *,
        collect_artifacts: bool = False,
        sample_ids: torch.Tensor | None = None,
    ):
        """Generate only the future frames from observable rollout inputs.

        ``history_rgb`` contains exactly ``num_history_frames`` frames.  Clean
        future RGB and clean V-JEPA features are neither accepted nor
        constructed.  The returned tensor is ``[B,3,num_future_frames,H,W]``.
        """
        predicted, ground_truth = self._sample_future(
            history_rgb,
            actions,
            morphology_index,
            auxiliary_target=None,
            collect_artifacts=collect_artifacts,
            deployment_mode=True,
            sample_ids=sample_ids,
        )
        if ground_truth is not None:
            raise RuntimeError(
                "deployable sampler unexpectedly constructed ground truth"
            )
        if predicted.shape[2] < self.num_future_frames:
            raise RuntimeError(
                "deployable decoder returned fewer frames than requested"
            )
        return predicted[:, :, -self.num_future_frames :]

    @torch.no_grad()
    def visualize(self, rgb, actions=None, mask=None, **kwargs):
        """Visualize while forwarding an offline auxiliary target when present."""
        del mask
        predicted, ground_truth = self._sample_future(
            rgb,
            actions,
            kwargs.get("morphology_index"),
            auxiliary_target=kwargs.get("auxiliary_target"),
            sample_ids=kwargs.get("clip_index"),
        )
        future_frames = min(self.num_future_frames, predicted.shape[2])
        side_by_side = torch.cat(
            [
                ground_truth[:, :, -future_frames:],
                predicted[:, :, -future_frames:],
            ],
            dim=-1,
        )
        side_by_side = side_by_side.permute(0, 2, 1, 3, 4)
        return torch.clamp(side_by_side * 0.5 + 0.5, 0.0, 1.0)

    def pop_visualization_artifacts(self):
        artifacts = self._visualization_artifacts
        self._visualization_artifacts = None
        return artifacts
