"""Causal explicit-action LACWM with a jointly denoised TF state.

The two pilot arms instantiate this exact class and train the same TF objective.
Their sole causal difference is ``condition_on_tf``: whether the current noisy
or generated TF state can enter Wan's shared video-token stream.

LACWM clock convention: sigma=1 is Gaussian noise and sigma=0 is clean data.
"""

from __future__ import annotations

import logging
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
    derive_tf_sigma,
    euler_flow_step,
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
)
EVALUATION_CONDITION_SOURCE_CODES = {
    source: index
    for index, source in enumerate(EVALUATION_CONDITION_SOURCES)
}


class DualExplicitActionDiTModel(ExplicitActionDiTModel):
    """Explicit-action world model with video and causal-RFFT flow states."""

    def __init__(
        self,
        *,
        time_frequency_transform,
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
        self.tf_schedule_mode = str(config.get("schedule_mode", "tf_leads"))
        self.tf_lead_logit = float(config.get("tf_lead_logit", 1.0))
        self.tf_loss_weight = float(config.get("tf_loss_weight", 1.0))
        self.video_only_control = bool(config.get("video_only_control", False))
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
            or "autonomous" not in self.evaluation_condition_sources
            or len(set(self.evaluation_condition_sources))
            != len(self.evaluation_condition_sources)
            or any(
                source not in EVALUATION_CONDITION_SOURCES
                for source in self.evaluation_condition_sources
            )
        ):
            raise ValueError(
                "evaluation_condition_sources must be unique supported values "
                f"including 'autonomous'; supported={EVALUATION_CONDITION_SOURCES}"
            )
        self.capture_latent_trajectories = bool(
            config.get("capture_latent_trajectories", True)
        )
        self.validation_video_sigmas = tuple(
            float(value)
            for value in config.get(
                "validation_video_sigmas", (0.90, 0.75, 0.50, 0.25)
            )
        )
        if self.tf_schedule_mode not in {"aligned", "tf_leads"}:
            raise ValueError(
                "pilot supports schedule_mode in {'aligned', 'tf_leads'}"
            )
        if self.tf_loss_weight < 0:
            raise ValueError("tf_loss_weight must be non-negative")
        self._assert_video_only_control_contract()
        # Check once more on the first train/eval call, after any model-only
        # warm start has loaded.  Frozen gates cannot subsequently open, and
        # avoiding a scalar device sync on every update keeps speed telemetry
        # representative.
        self._video_only_runtime_validated = not self.video_only_control
        if not self.validation_video_sigmas or any(
            not 0 <= sigma <= 1 for sigma in self.validation_video_sigmas
        ):
            raise ValueError("validation_video_sigmas must lie in [0,1]")
        self._visualization_artifacts = None
        logger.info(
            "DualExplicitActionDiTModel: condition_mode=%s, schedule=%s, "
            "tf_lead_logit=%.3f, tf_loss_weight=%.3f, video_only_control=%s, "
            "evaluation_nfe=%s, evaluation_sources=%s",
            self.tf_condition_mode,
            self.tf_schedule_mode,
            self.tf_lead_logit,
            self.tf_loss_weight,
            self.video_only_control,
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
        self._video_only_runtime_validated = True

    @staticmethod
    def _expand_sigma(sigma: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if sigma.ndim != 1 or sigma.shape[0] != reference.shape[0]:
            raise ValueError("sigma must have shape [B]")
        return sigma.to(device=reference.device, dtype=reference.dtype).reshape(
            -1, *([1] * (reference.ndim - 1))
        )

    def _paired_training_clocks(
        self, batch_size: int, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        tf_sigma = derive_tf_sigma(
            video_sigma.float(),
            mode=self.tf_schedule_mode,
            tf_lead_logit=self.tf_lead_logit,
        ).to(dtype=dtype)
        return timesteps, video_sigma, tf_sigma

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

    @torch.no_grad()
    def _tf_clean(self, rgb: torch.Tensor, latent_shape) -> torch.Tensor:
        coefficients = self.time_frequency_transform(rgb).detach()
        if coefficients.shape[0] != latent_shape[0] or coefficients.shape[2:] != tuple(
            latent_shape[2:]
        ):
            raise RuntimeError(
                "TF transform is not aligned to the Wan latent grid: "
                f"TF={tuple(coefficients.shape)}, Wan={tuple(latent_shape)}"
            )
        return coefficients.to(device=rgb.device, dtype=rgb.dtype)

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

    def _training_tf_noise(self, tf_clean: torch.Tensor) -> torch.Tensor:
        """Draw TF corruption without perturbing the video-only RNG path."""
        if self.video_only_control:
            # The TF target is zero-weighted and both injection gates are
            # frozen at zero.  A deterministic placeholder avoids advancing
            # the global generator before Wan's LoRA-dropout masks are drawn,
            # preserving the ordinary video RNG/scheduler path.
            return torch.zeros_like(tf_clean)
        return torch.randn_like(tf_clean)

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
        history_frames = min(self.num_history_latent, latent_frames)
        reference = torch.zeros_like(video_clean)
        reference[:, :, :history_frames] = video_clean[:, :, :history_frames]

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
        timesteps, video_sigma, tf_sigma = self._paired_training_clocks(
            batch_size, device, video_clean.dtype
        )
        video_sigma_expanded = self._expand_sigma(video_sigma, video_clean)
        video_noisy = (
            (1.0 - video_sigma_expanded) * video_clean
            + video_sigma_expanded * video_noise
        )
        video_target = video_noise - video_clean

        tf_clean = self._tf_clean(rgb, video_clean.shape)
        tf_noise = self._training_tf_noise(tf_clean)
        tf_sigma_expanded = self._expand_sigma(tf_sigma, tf_clean)
        tf_noisy = (1.0 - tf_sigma_expanded) * tf_clean + tf_sigma_expanded * tf_noise
        # Observed history is deterministic conditioning; only future bins diffuse.
        tf_noisy = tf_noisy.clone()
        tf_noisy[:, :, :history_frames] = tf_clean[:, :, :history_frames]
        tf_target = tf_noise - tf_clean
        conditioning_tf = self._training_conditioning_tf(
            tf_clean=tf_clean,
            tf_noise=tf_noise,
            tf_noisy=tf_noisy,
            tf_sigma_expanded=tf_sigma_expanded,
            history_frames=history_frames,
        )

        context = self._build_context(batch_size, device, rgb.dtype)
        clip_fea = self._build_clip(batch_size, device, rgb.dtype)
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
        tf_per_sample = self._masked_per_sample_mse(
            prediction.tf_velocity[:, :, history_frames:],
            tf_target[:, :, history_frames:],
            future_mask,
        )
        video_loss = video_per_sample.mean()
        tf_loss = tf_per_sample.mean()
        total_loss = video_loss + self.tf_loss_weight * tf_loss

        video_x0 = video_noisy - video_sigma_expanded * prediction.video_velocity
        tf_x0 = tf_noisy - tf_sigma_expanded * prediction.tf_velocity
        video_nmse = self._masked_per_sample_nmse(
            video_x0[:, :, history_frames:],
            video_clean[:, :, history_frames:],
            future_mask,
        )
        tf_nmse = self._masked_per_sample_nmse(
            tf_x0[:, :, history_frames:],
            tf_clean[:, :, history_frames:],
            future_mask,
        )

        self.aux_losses["flow_loss"] = video_loss.detach()
        self.aux_losses["video_flow_loss"] = video_loss.detach()
        self.aux_losses["tf_flow_loss"] = tf_loss.detach()
        self.aux_losses["clock/video_sigma_mean"] = video_sigma.float().mean().detach()
        self.aux_losses["clock/tf_sigma_mean"] = tf_sigma.float().mean().detach()
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
        self.aux_losses["condition/mode_code"] = video_loss.new_tensor(
            float(TF_CONDITION_MODE_CODES[self.tf_condition_mode])
        )
        self.aux_losses["condition/video_only_control"] = video_loss.new_tensor(
            float(self.video_only_control)
        )
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
    def _sample_future(self, rgb, actions=None, morphology_index=None):
        """Run independent joint samplers at each requested NFE.

        Every NFE starts from the same deterministic video/TF noise.  A
        one-step result is therefore a deliberate paired negative control: its
        sole Wan call sees the same local future TF noise at ``sigma=1`` for
        matched and shuffled conditioning.  The first possible causal benefit
        from an autonomously denoised TF state is on the second Wan call.
        """
        self._ensure_video_only_runtime_contract()
        video_clean = self._encode_clip(rgb).to(rgb.dtype)
        batch_size, _, latent_frames, _, _ = video_clean.shape
        history_frames = min(self.num_history_latent, latent_frames)
        reference = torch.zeros_like(video_clean)
        reference[:, :, :history_frames] = video_clean[:, :, :history_frames]
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

        tf_clean = self._tf_clean(rgb, video_clean.shape)
        ground_truth_pixels = self.rgb_tokenizer.decode_temporal(
            video_clean, out_hw=(rgb.shape[-2], rgb.shape[-1])
        )
        future_pixel_frames = min(
            self.num_future_frames, ground_truth_pixels.shape[2]
        )

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        generator = torch.Generator(device=rgb.device)
        generator.manual_seed(self.evaluation_noise_seed + rank)
        initial_video_state = torch.randn(
            video_clean.shape,
            device=video_clean.device,
            dtype=video_clean.dtype,
            generator=generator,
        )
        initial_tf_noise = torch.randn(
            tf_clean.shape,
            device=tf_clean.device,
            dtype=tf_clean.dtype,
            generator=generator,
        )
        initial_tf_state = initial_tf_noise.clone()
        initial_tf_state[:, :, :history_frames] = tf_clean[
            :, :, :history_frames
        ]

        artifacts = {
            "video_clean": video_clean[:1].detach().cpu().to(torch.float16),
            "tf_clean": tf_clean[:1].detach().cpu().to(torch.float16),
            "video_initial_state": (
                initial_video_state[:1].detach().cpu().to(torch.float16)
            ),
            "tf_initial_state": (
                initial_tf_state[:1].detach().cpu().to(torch.float16)
            ),
            "tf_initial_noise": (
                initial_tf_noise[:1].detach().cpu().to(torch.float16)
            ),
            "ground_truth_future_uint8": (
                (
                    ground_truth_pixels[:1, :, -future_pixel_frames:]
                    .float()
                    .clamp(-1.0, 1.0)
                    + 1.0
                )
                .mul(127.5)
                .round()
                .to(torch.uint8)
                .cpu()
            ),
            "history_latent_frames": torch.tensor([history_frames]),
            "condition_on_tf": torch.tensor([int(self.condition_on_tf)]),
            "condition_mode_code": torch.tensor(
                [TF_CONDITION_MODE_CODES[self.tf_condition_mode]]
            ),
            "video_only_control": torch.tensor(
                [int(self.video_only_control)], dtype=torch.int64
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
            "oracle_sources_are_leakage": torch.tensor([1], dtype=torch.int64),
        }
        primary_pixels = None

        wrong_tf_clean = None
        if "oracle_shuffled" in self.evaluation_condition_sources:
            wrong_tf_clean = self._roll_across_global_batch(tf_clean)

        for condition_source in self.evaluation_condition_sources:
            source_infix = (
                "" if condition_source == "autonomous" else f"_{condition_source}"
            )
            for num_steps in self.evaluation_nfe_steps:
                video_state = initial_video_state.clone()
                tf_state = initial_tf_state.clone()
                self.sample_scheduler.set_timesteps(num_steps, device=rgb.device)
                native_video_sigmas = self.sample_scheduler.sigmas.to(
                    device=rgb.device, dtype=torch.float32
                )[: num_steps + 1]
                schedule = pair_video_sigma_schedule(
                    native_video_sigmas,
                    mode=self.tf_schedule_mode,
                    tf_lead_logit=self.tf_lead_logit,
                )

                capture_this_trajectory = (
                    condition_source == "autonomous"
                    and num_steps == self.viz_num_steps
                )
                video_trajectory = (
                    [video_state[:1].detach().cpu().to(torch.float16)]
                    if capture_this_trajectory
                    else []
                )
                tf_trajectory = (
                    [tf_state[:1].detach().cpu().to(torch.float16)]
                    if capture_this_trajectory
                    else []
                )
                video_x0_trajectory = []
                tf_x0_trajectory = []

                for index, timestep in enumerate(
                    self.sample_scheduler.timesteps
                ):
                    video_sigma = schedule.video[index]
                    tf_sigma = schedule.time_frequency[index]
                    next_tf_sigma = schedule.time_frequency[index + 1]
                    timesteps = timestep.expand(batch_size).to(rgb.device)
                    tf_batch_sigma = tf_sigma.expand(batch_size).to(
                        dtype=rgb.dtype
                    )
                    tf_sigma_expanded = self._expand_sigma(
                        tf_batch_sigma, tf_state
                    )
                    if condition_source == "autonomous":
                        conditioning_tf = self._sampling_conditioning_tf(
                            tf_state,
                            initial_tf_noise,
                            tf_sigma_expanded,
                            history_frames,
                        )
                        use_tf_condition = self.condition_on_tf
                    elif condition_source == "off":
                        conditioning_tf = tf_state
                        use_tf_condition = False
                    else:
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
                            history_frames=history_frames,
                            wrong_tf_clean=(
                                None
                                if condition_source == "oracle_matched"
                                else oracle_clean
                            ),
                        )
                        use_tf_condition = True

                    prediction = self.forward_model(
                        video_state,
                        timesteps,
                        z_control,
                        reference,
                        context,
                        clip_fea,
                        noisy_tf=tf_state,
                        conditioning_tf=conditioning_tf,
                        tf_sigma=tf_batch_sigma,
                        condition_on_tf=use_tf_condition,
                    )
                    if not isinstance(prediction, DualWanOutput):
                        raise RuntimeError(
                            "dual Wan sampler did not return both velocities"
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
                            video_x0[:1].detach().cpu().to(torch.float16)
                        )
                        tf_x0_trajectory.append(
                            tf_x0[:1].detach().cpu().to(torch.float16)
                        )

                    video_state = self.sample_scheduler.step(
                        prediction.video_velocity.float(),
                        timestep,
                        video_state.float(),
                    ).prev_sample.to(rgb.dtype)
                    tf_state = euler_flow_step(
                        tf_state.float(),
                        prediction.tf_velocity.float(),
                        tf_sigma,
                        next_tf_sigma,
                    ).to(rgb.dtype)
                    tf_state[:, :, :history_frames] = tf_clean[
                        :, :, :history_frames
                    ]
                    if capture_this_trajectory:
                        video_trajectory.append(
                            video_state[:1].detach().cpu().to(torch.float16)
                        )
                        tf_trajectory.append(
                            tf_state[:1].detach().cpu().to(torch.float16)
                        )

                predicted_pixels = self.rgb_tokenizer.decode_temporal(
                    video_state, out_hw=(rgb.shape[-2], rgb.shape[-1])
                )
                artifacts[
                    f"video_final{source_infix}_nfe_{num_steps}"
                ] = video_state[:1].detach().cpu().to(torch.float16)
                artifacts[
                    f"tf_final{source_infix}_nfe_{num_steps}"
                ] = tf_state[:1].detach().cpu().to(torch.float16)
                artifacts[
                    f"decoded_future{source_infix}_nfe_{num_steps}"
                ] = (
                    (
                        predicted_pixels[:1, :, -future_pixel_frames:]
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
                    primary_pixels = predicted_pixels
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

        if primary_pixels is None:
            raise RuntimeError(
                f"viz_num_steps={self.viz_num_steps} was not evaluated"
            )
        if self.capture_latent_trajectories:
            self._visualization_artifacts = artifacts
        return primary_pixels, ground_truth_pixels

    def pop_visualization_artifacts(self):
        artifacts = self._visualization_artifacts
        self._visualization_artifacts = None
        return artifacts
