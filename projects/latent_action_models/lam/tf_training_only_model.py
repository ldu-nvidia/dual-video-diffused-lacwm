"""Explicit-action Wan model with a training-only spectral endpoint loss.

There is no auxiliary state, network, input, or sampling clock.  Both study
arms instantiate this identical parameter schema; only the scalar loss weight
differs.  At training time ``x0_hat = x_sigma - sigma*v_hat`` is compared with
the clean *future* Wan latent in a low-band 3-D Fourier representation.  The
deployment sampler is an ordinary one-clock Wan Euler sampler that accepts
only observed RGB history and robot actions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lam.explicit_action_dit_model import ExplicitActionDiTModel
from robot_wm.modeling.dual_diffusion.spectral_consistency import (
    spatiotemporal_spectral_consistency,
)


class TFTrainingOnlyContractError(RuntimeError):
    """The preregistered architecture, geometry, or causal sampler changed."""


@dataclass(frozen=True)
class DeployableVideoSample:
    initial_video_noise: Tensor
    video_latent: Tensor
    decoded_future: Tensor
    history_latent_frames: int
    wan_calls: int


class TFTrainingOnlyExplicitActionDiT(ExplicitActionDiTModel):
    """Parameter-identical OFF/ON model; spectrum exists only in the train loss."""

    def __init__(self, *, spectral_regularization: Mapping[str, Any], **kwargs: Any):
        config = dict(spectral_regularization)
        self.spectral_loss_weight = float(config.get("loss_weight", 0.0))
        self.spectral_spatial_band_fraction = float(
            config.get("spatial_band_fraction", 0.5)
        )
        self.spectral_phase_weight = float(config.get("phase_weight", 0.25))
        self.spectral_epsilon = float(config.get("epsilon", 1.0e-4))
        if self.spectral_loss_weight < 0:
            raise ValueError("spectral loss_weight must be non-negative")
        if not 0 < self.spectral_spatial_band_fraction <= 1:
            raise ValueError("spectral spatial_band_fraction must lie in (0,1]")
        if self.spectral_phase_weight < 0 or self.spectral_epsilon <= 0:
            raise ValueError(
                "spectral phase_weight/epsilon must be non-negative/positive"
            )
        super().__init__(**kwargs)
        # The screen is intentionally the ordinary single-clock Wan model.
        forbidden = (
            "tf_token_adapter",
            "tf_clock_embedding",
            "tf_velocity_head",
        )
        if any(hasattr(self.forward_model, name) for name in forbidden):
            raise TFTrainingOnlyContractError(
                "training-only spectral supervision must not instantiate a TF branch"
            )
        if self.num_history_frames != 5 or self.num_future_frames != 8:
            raise TFTrainingOnlyContractError(
                "screen requires 5 history + 8 future RGB frames"
            )

    @staticmethod
    def _masked_per_sample_nmse(
        estimate: Tensor, target: Tensor, mask: Tensor
    ) -> Tensor:
        error = (estimate.float() - target.float()).square()
        denominator_signal = target.float().square()
        expanded = mask.expand_as(error).float()
        dims = tuple(range(1, error.ndim))
        numerator = (error * expanded).sum(dim=dims)
        denominator = (denominator_signal * expanded).sum(dim=dims)
        return numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)

    def forward(
        self,
        rgb: Tensor,
        actions: Tensor | None = None,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        if kwargs.get("auxiliary_target") is not None:
            raise TFTrainingOnlyContractError(
                "model must not consume an auxiliary target"
            )
        self.aux_losses = {}
        morphology_index = kwargs.get("morphology_index")
        clean = self._encode_clip(rgb).to(rgb.dtype)
        batch_size, _, latent_frames, _, _ = clean.shape
        history_frames = min(self.num_history_latent, latent_frames)
        if latent_frames != 4 or history_frames != 2:
            raise TFTrainingOnlyContractError(
                "screen requires 13 RGB frames -> 4 Wan tokens (2 history, 2 future)"
            )
        reference = torch.zeros_like(clean)
        reference[:, :, :history_frames] = clean[:, :, :history_frames]
        _, control, _ = self._latent_actions(
            rgb, actions, morphology_index, latent_frames, history_frames
        )
        control = control.to(rgb.dtype)

        noise = torch.randn_like(clean)
        timesteps = self._sample_timesteps(batch_size, rgb.device)
        sigma = self._get_sigmas(timesteps, clean.ndim, clean.dtype, rgb.device)
        noisy = (1.0 - sigma) * clean + sigma * noise
        velocity_target = noise - clean
        velocity = self.forward_model(
            noisy,
            timesteps,
            control,
            reference,
            self._build_context(batch_size, rgb.device, rgb.dtype),
            self._build_clip(batch_size, rgb.device, rgb.dtype),
        )
        loss_mask = self._build_loss_mask(rgb, mask, clean.shape)
        future_mask = loss_mask[:, :, history_frames:]
        squared_error = (
            velocity[:, :, history_frames:].float()
            - velocity_target[:, :, history_frames:].float()
        ).square()
        expanded_mask = future_mask.expand_as(squared_error).float()
        reduce_dims = tuple(range(1, squared_error.ndim))
        denominator = expanded_mask.sum(dim=reduce_dims)
        if bool((denominator <= 0).any()):
            raise TFTrainingOnlyContractError(
                "sample has no valid future latent elements"
            )
        flow_per_sample = (squared_error * expanded_mask).sum(
            dim=reduce_dims
        ) / denominator
        flow_loss = flow_per_sample.mean()

        predicted_clean = noisy - sigma * velocity
        spectral_total = flow_loss.new_zeros(())
        amplitude = flow_loss.new_zeros(())
        phase = flow_loss.new_zeros(())
        selected_bins = 0
        if self.spectral_loss_weight > 0:
            terms = spatiotemporal_spectral_consistency(
                predicted_clean[:, :, history_frames:],
                clean[:, :, history_frames:],
                validity_mask=future_mask,
                spatial_band_fraction=self.spectral_spatial_band_fraction,
                phase_weight=self.spectral_phase_weight,
                epsilon=self.spectral_epsilon,
            )
            spectral_total = terms.total
            amplitude = terms.log_amplitude
            phase = terms.phase
            selected_bins = terms.selected_bins
        total = flow_loss + self.spectral_loss_weight * spectral_total
        clean_nmse = self._masked_per_sample_nmse(
            predicted_clean[:, :, history_frames:],
            clean[:, :, history_frames:],
            future_mask,
        ).mean()
        self.aux_losses.update(
            {
                "flow_loss": flow_loss.detach(),
                "spectral/total": spectral_total.detach(),
                "spectral/log_amplitude": amplitude.detach(),
                "spectral/phase": phase.detach(),
                "spectral/loss_weight": flow_loss.new_tensor(self.spectral_loss_weight),
                "spectral/selected_bins": flow_loss.new_tensor(float(selected_bins)),
                "state/predicted_clean_future_nmse": clean_nmse.detach(),
            }
        )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("training-only spectral objective is non-finite")
        return total

    @torch.no_grad()
    def sample_future_deployable(
        self,
        history_rgb: Tensor,
        actions: Tensor,
        morphology_index: Tensor | None = None,
        *,
        nfe: int = 1,
        noise_seed: int,
    ) -> DeployableVideoSample:
        """Generate using observed history/actions and exactly ``nfe`` Wan calls."""

        if history_rgb.ndim != 5 or history_rgb.shape[1] != self.num_history_frames:
            raise ValueError(
                f"history_rgb must contain exactly {self.num_history_frames} observed frames"
            )
        if nfe < 1 or noise_seed < 0:
            raise ValueError("nfe must be positive and noise_seed non-negative")
        history_latent = self._encode_clip(history_rgb).to(history_rgb.dtype)
        if int(history_latent.shape[2]) != self.num_history_latent:
            raise TFTrainingOnlyContractError("history-only VAE geometry changed")
        latent_frames = self.rgb_tokenizer.latent_temporal_len(
            self.num_history_frames + self.num_future_frames
        )
        history_frames = int(history_latent.shape[2])
        shape = (
            int(history_rgb.shape[0]),
            int(history_latent.shape[1]),
            int(latent_frames),
            int(history_latent.shape[3]),
            int(history_latent.shape[4]),
        )
        reference = history_latent.new_zeros(shape)
        reference[:, :, :history_frames] = history_latent
        _, control, _ = self._latent_actions(
            history_rgb, actions, morphology_index, latent_frames, history_frames
        )
        control = control.to(history_rgb.dtype)
        context = self._build_context(
            history_rgb.shape[0], history_rgb.device, history_rgb.dtype
        )
        clip = self._build_clip(
            history_rgb.shape[0], history_rgb.device, history_rgb.dtype
        )
        generator = torch.Generator(device=history_rgb.device)
        generator.manual_seed(int(noise_seed))
        initial_noise = torch.randn(
            shape,
            generator=generator,
            device=history_rgb.device,
            dtype=history_rgb.dtype,
        )
        state = initial_noise.clone()
        self.sample_scheduler.set_timesteps(int(nfe), device=history_rgb.device)
        calls = 0
        native_sigmas = self.sample_scheduler.sigmas.to(
            device=history_rgb.device, dtype=history_rgb.dtype
        )
        for index, timestep in enumerate(self.sample_scheduler.timesteps):
            velocity = self.forward_model(
                state,
                timestep.expand(history_rgb.shape[0]),
                control,
                reference,
                context,
                clip,
            )
            calls += 1
            state = self.sample_scheduler.step(
                velocity.float(), timestep, state.float()
            ).prev_sample.to(history_rgb.dtype)
            next_sigma = native_sigmas[index + 1]
            state[:, :, :history_frames] = (1.0 - next_sigma) * reference[
                :, :, :history_frames
            ] + next_sigma * initial_noise[:, :, :history_frames]
        if calls != nfe:
            raise TFTrainingOnlyContractError("sampler Wan call count differs from NFE")
        decoded = self.rgb_tokenizer.decode_temporal(
            state, out_hw=(history_rgb.shape[-2], history_rgb.shape[-1])
        )
        return DeployableVideoSample(
            initial_video_noise=initial_noise,
            video_latent=state,
            decoded_future=decoded[:, :, -self.num_future_frames :],
            history_latent_frames=history_frames,
            wan_calls=calls,
        )
