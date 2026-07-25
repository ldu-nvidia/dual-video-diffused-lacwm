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

from lam.explicit_action_dit_model import ExplicitActionDiTModel
from robot_wm.modeling.dual_diffusion.flow import (
    derive_tf_sigma,
    euler_flow_step,
    pair_video_sigma_schedule,
)
from robot_wm.modeling.networks.wan_forward_model import DualWanOutput

logger = logging.getLogger(__name__)


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
        self.condition_on_tf = bool(config.get("condition_on_tf", False))
        self.tf_schedule_mode = str(config.get("schedule_mode", "tf_leads"))
        self.tf_lead_logit = float(config.get("tf_lead_logit", 1.0))
        self.tf_loss_weight = float(config.get("tf_loss_weight", 1.0))
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
        if not self.validation_video_sigmas or any(
            not 0 <= sigma <= 1 for sigma in self.validation_video_sigmas
        ):
            raise ValueError("validation_video_sigmas must lie in [0,1]")
        self._visualization_artifacts = None
        logger.info(
            "DualExplicitActionDiTModel: condition_on_tf=%s, schedule=%s, "
            "tf_lead_logit=%.3f, tf_loss_weight=%.3f",
            self.condition_on_tf,
            self.tf_schedule_mode,
            self.tf_lead_logit,
            self.tf_loss_weight,
        )

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
        tf_noise = torch.randn_like(tf_clean)
        tf_sigma_expanded = self._expand_sigma(tf_sigma, tf_clean)
        tf_noisy = (1.0 - tf_sigma_expanded) * tf_clean + tf_sigma_expanded * tf_noise
        # Observed history is deterministic conditioning; only future bins diffuse.
        tf_noisy = tf_noisy.clone()
        tf_noisy[:, :, :history_frames] = tf_clean[:, :, :history_frames]
        tf_target = tf_noise - tf_clean

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
        self.aux_losses["condition/state_gate"] = (
            torch.tanh(self.forward_model.tf_token_adapter.gate).detach()
        )
        self.aux_losses["condition/clock_gate"] = (
            torch.tanh(self.forward_model.tf_clock_embedding.gate).detach()
        )
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
        """Joint Euler sampling from two future noise states."""
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
        video_state = torch.randn_like(video_clean)
        tf_state = torch.randn_like(tf_clean)
        tf_state[:, :, :history_frames] = tf_clean[:, :, :history_frames]

        self.sample_scheduler.set_timesteps(self.viz_num_steps, device=rgb.device)
        native_video_sigmas = self.sample_scheduler.sigmas.to(
            device=rgb.device, dtype=torch.float32
        )[: self.viz_num_steps + 1]
        schedule = pair_video_sigma_schedule(
            native_video_sigmas,
            mode=self.tf_schedule_mode,
            tf_lead_logit=self.tf_lead_logit,
        )

        video_trajectory = [video_state[:1].detach().cpu().to(torch.float16)]
        tf_trajectory = [tf_state[:1].detach().cpu().to(torch.float16)]
        video_x0_trajectory = []
        tf_x0_trajectory = []

        for index, timestep in enumerate(self.sample_scheduler.timesteps):
            video_sigma = schedule.video[index]
            tf_sigma = schedule.time_frequency[index]
            next_tf_sigma = schedule.time_frequency[index + 1]
            timesteps = timestep.expand(batch_size).to(rgb.device)
            tf_batch_sigma = tf_sigma.expand(batch_size).to(dtype=rgb.dtype)
            prediction = self.forward_model(
                video_state,
                timesteps,
                z_control,
                reference,
                context,
                clip_fea,
                noisy_tf=tf_state,
                tf_sigma=tf_batch_sigma,
                condition_on_tf=self.condition_on_tf,
            )
            if not isinstance(prediction, DualWanOutput):
                raise RuntimeError("dual Wan sampler did not return both velocities")

            video_x0 = (
                video_state.float()
                - video_sigma.float() * prediction.video_velocity.float()
            )
            tf_x0 = tf_state.float() - tf_sigma.float() * prediction.tf_velocity.float()
            video_x0_trajectory.append(
                video_x0[:1].detach().cpu().to(torch.float16)
            )
            tf_x0_trajectory.append(tf_x0[:1].detach().cpu().to(torch.float16))

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
            tf_state[:, :, :history_frames] = tf_clean[:, :, :history_frames]
            video_trajectory.append(
                video_state[:1].detach().cpu().to(torch.float16)
            )
            tf_trajectory.append(tf_state[:1].detach().cpu().to(torch.float16))

        if self.capture_latent_trajectories:
            self._visualization_artifacts = {
                "video_trajectory": torch.stack(video_trajectory),
                "tf_trajectory": torch.stack(tf_trajectory),
                "video_x0_trajectory": torch.stack(video_x0_trajectory),
                "tf_x0_trajectory": torch.stack(tf_x0_trajectory),
                "video_clean": video_clean[:1].detach().cpu().to(torch.float16),
                "tf_clean": tf_clean[:1].detach().cpu().to(torch.float16),
                "video_sigmas": schedule.video.detach().cpu().float(),
                "tf_sigmas": schedule.time_frequency.detach().cpu().float(),
                "history_latent_frames": torch.tensor([history_frames]),
                "condition_on_tf": torch.tensor([int(self.condition_on_tf)]),
            }

        predicted_pixels = self.rgb_tokenizer.decode_temporal(
            video_state, out_hw=(rgb.shape[-2], rgb.shape[-1])
        )
        ground_truth_pixels = self.rgb_tokenizer.decode_temporal(
            video_clean, out_hw=(rgb.shape[-2], rgb.shape[-1])
        )
        return predicted_pixels, ground_truth_pixels

    def pop_visualization_artifacts(self):
        artifacts = self._visualization_artifacts
        self._visualization_artifacts = None
        return artifacts
