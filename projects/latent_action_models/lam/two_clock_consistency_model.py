"""Matched two-clock rectified-flow self-consistency for the VPM video branch.

This training-only specialization preserves the historical parameter-matched
video model and its deployment sampler.  Every update executes two Wan calls
from the same clean video latent and the same Gaussian direction:

``x_hi = (1-sigma_hi) * x0 + sigma_hi * epsilon``, ``sigma_hi in [0.8, 1.0]``
``x_lo = (1-sigma_lo) * x0 + sigma_lo * epsilon``, ``sigma_lo in [0.0, 0.4]``

Both calls receive ordinary rectified-flow supervision.  The candidate also
trains the high-noise clean prediction toward a stopped low-noise prediction.
No auxiliary feature, teacher, new parameter, or inference-time call is added.
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


class TwoClockConsistencyError(RuntimeError):
    """The preregistered two-clock contract was violated."""


def tensor_sha256(value: Tensor) -> str:
    """Hash exact tensor bytes together with dtype and shape."""

    detached = value.detach().contiguous()
    header = f"{detached.dtype}|{tuple(detached.shape)}|".encode("ascii")
    raw = detached.view(torch.uint8).cpu().numpy().tobytes(order="C")
    return hashlib.sha256(header + raw).hexdigest()


def rf_predicted_clean(noisy: Tensor, sigma: Tensor, velocity: Tensor) -> Tensor:
    """Return ``x0_hat = x_sigma - sigma * v_theta`` in float32."""

    if noisy.shape != velocity.shape or noisy.ndim != 5:
        raise ValueError("noisy and velocity must share [B,C,T,H,W] shape")
    if sigma.ndim != 1 or sigma.shape[0] != noisy.shape[0]:
        raise ValueError("sigma must have shape [B]")
    expanded = sigma.to(device=noisy.device, dtype=torch.float32).reshape(
        -1, 1, 1, 1, 1
    )
    return noisy.float() - expanded * velocity.float()


def normalized_stopped_consistency_loss(
    high_x0: Tensor,
    low_x0: Tensor,
    clean_x0: Tensor,
    mask: Tensor,
    *,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Scale-normalized high-to-stopped-low clean-prediction consistency.

    ``mask`` is broadcast to the video latent geometry.  The denominator uses
    only the detached clean-latent energy as a scale; the consistency target is
    exclusively ``stopgrad(low_x0)``.
    """

    if high_x0.shape != low_x0.shape or high_x0.shape != clean_x0.shape:
        raise ValueError("high, low, and clean states must share shape")
    if high_x0.ndim != 5 or mask.ndim != 5:
        raise ValueError("states and mask must use Wan [B,C,T,H,W] layout")
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    weights = mask.to(device=high_x0.device, dtype=torch.float32).expand_as(
        high_x0
    )
    reduce_dims = tuple(range(1, high_x0.ndim))
    count = weights.sum(dim=reduce_dims)
    if bool((count <= 0).any()):
        raise TwoClockConsistencyError("consistency received an empty mask")
    teacher = low_x0.detach().float()
    numerator = (
        (high_x0.float() - teacher).square() * weights
    ).sum(dim=reduce_dims) / count
    denominator = (
        clean_x0.detach().float().square() * weights
    ).sum(dim=reduce_dims) / count
    denominator = denominator.add(float(epsilon))
    per_sample = numerator / denominator
    if not bool(torch.isfinite(per_sample).all()):
        raise TwoClockConsistencyError("normalized consistency is non-finite")
    return per_sample.mean(), denominator.mean()


class TwoClockConsistencyVPM(DualExplicitActionDiTModel):
    """Parameter-identical VPM with two-clock training and unchanged sampling."""

    def __init__(
        self,
        *,
        two_clock_consistency: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        config = dict(two_clock_consistency)
        self.consistency_weight = float(config.get("weight", 0.0))
        self.consistency_epsilon = float(config.get("epsilon", 1e-6))
        self.high_sigma_min = float(config.get("high_sigma_min", 0.8))
        self.high_sigma_max = float(config.get("high_sigma_max", 1.0))
        self.low_sigma_min = float(config.get("low_sigma_min", 0.0))
        self.low_sigma_max = float(config.get("low_sigma_max", 0.4))
        if self.consistency_weight not in {0.0, 0.2}:
            raise ValueError("screen fixes consistency weight to 0.0 or 0.2")
        if not math.isfinite(self.consistency_epsilon) or self.consistency_epsilon <= 0:
            raise ValueError("consistency epsilon must be finite and positive")
        if (
            self.high_sigma_min != 0.8
            or self.high_sigma_max != 1.0
            or self.low_sigma_min != 0.0
            or self.low_sigma_max != 0.4
        ):
            raise ValueError("screen fixes high=[0.8,1.0] and low=[0.0,0.4]")
        super().__init__(**kwargs)
        if (
            not bool(getattr(self, "parameter_matched_control", False))
            or self.tf_condition_mode != "off"
            or self.condition_on_tf
            or bool(getattr(self, "condition_on_tf_clock", False))
            or self.tf_loss_weight != 0.0
        ):
            raise TwoClockConsistencyError(
                "two-clock screen requires the exact VPM no-op auxiliary arm"
            )
        self.paired_audit_exact: dict[str, str] = {}

    def _resolve_auxiliary_clean(
        self,
        rgb: Tensor,
        latent_shape: tuple[int, ...],
        auxiliary_target: Tensor | None,
    ) -> Tensor:
        """Retain the VPM auxiliary topology without opening a target cache."""

        if auxiliary_target is not None:
            raise TwoClockConsistencyError(
                "two-clock consistency forbids cached auxiliary targets"
            )
        channels = int(self.forward_model.tf_token_adapter.tf_channels)
        return rgb.new_zeros(
            int(latent_shape[0]),
            channels,
            *tuple(int(value) for value in latent_shape[2:]),
        )

    def _clock_candidates(
        self,
        lower: float,
        upper: float,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        schedule_t = self.noise_scheduler.timesteps.to(device=device)
        schedule_sigma = self.noise_scheduler.sigmas.to(
            device=device, dtype=torch.float32
        )[: schedule_t.numel()]
        keep = (schedule_sigma >= lower) & (schedule_sigma <= upper)
        indexes = keep.nonzero(as_tuple=False).flatten()
        if indexes.numel() < 2:
            raise TwoClockConsistencyError(
                f"scheduler has too few clock points in [{lower}, {upper}]"
            )
        return schedule_t[indexes], schedule_sigma[indexes]

    def _sample_two_clocks(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Uniformly sample discrete scheduler points in both frozen bands."""

        hi_t, hi_s = self._clock_candidates(
            self.high_sigma_min, self.high_sigma_max, device=device
        )
        lo_t, lo_s = self._clock_candidates(
            self.low_sigma_min, self.low_sigma_max, device=device
        )
        hi_choice = torch.randint(hi_t.numel(), (batch_size,), device=device)
        lo_choice = torch.randint(lo_t.numel(), (batch_size,), device=device)
        return (
            hi_t[hi_choice],
            hi_s[hi_choice],
            lo_t[lo_choice],
            lo_s[lo_choice],
        )

    @staticmethod
    def _probe_mean(value: Tensor) -> Tensor:
        flat = value.detach().float().reshape(value.shape[0], -1)
        return flat[:, : min(8, int(flat.shape[1]))].mean()

    def _video_pass(
        self,
        noisy: Tensor,
        timesteps: Tensor,
        sigma: Tensor,
        z_control: Tensor,
        reference: Tensor,
        context: Any,
        clip_fea: Tensor,
        inert_auxiliary: Tensor,
    ) -> DualWanOutput:
        self._two_clock_training_calls += 1
        prediction = self.forward_model(
            noisy,
            timesteps,
            z_control,
            reference,
            context,
            clip_fea,
            noisy_tf=inert_auxiliary,
            conditioning_tf=inert_auxiliary,
            # Preserve the historical VPM's unused auxiliary-head clock. The
            # video-trunk clock injection remains explicitly disabled.
            tf_sigma=sigma,
            condition_on_tf=False,
            condition_on_tf_clock=False,
        )
        if not isinstance(prediction, DualWanOutput):
            raise TwoClockConsistencyError("VPM did not return DualWanOutput")
        return prediction

    def forward(
        self,
        rgb: Tensor,
        actions: Tensor | None = None,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Train two points on one RF trajectory; inference remains inherited."""

        self._ensure_video_only_runtime_contract()
        self.aux_losses = {}
        self._two_clock_training_calls = 0
        device = rgb.device
        clean = self._encode_clip(rgb).to(rgb.dtype)
        batch_size, _, latent_frames, _, _ = clean.shape
        if latent_frames != 4 or self.num_history_latent != 2:
            raise TwoClockConsistencyError(
                "screen requires 13 RGB frames -> 4 Wan tokens, 2 history tokens"
            )
        reference, history_frames = self._history_reference(rgb, clean.shape)
        _, z_control, _ = self._latent_actions(
            rgb,
            actions,
            kwargs.get("morphology_index"),
            latent_frames,
            history_frames,
        )
        z_control = z_control.to(rgb.dtype)

        # One Gaussian direction defines both trajectory points.  Clock draws
        # happen after epsilon in both arms, preserving the paired RNG order.
        epsilon = torch.randn_like(clean)
        hi_t, hi_sigma, lo_t, lo_sigma = self._sample_two_clocks(
            batch_size, device
        )
        hi_expand = self._expand_sigma(hi_sigma, clean)
        lo_expand = self._expand_sigma(lo_sigma, clean)
        noisy_hi = (1.0 - hi_expand) * clean + hi_expand * epsilon
        noisy_lo = (1.0 - lo_expand) * clean + lo_expand * epsilon
        velocity_target = epsilon - clean

        context = self._build_context(batch_size, device, rgb.dtype)
        clip_fea = self._build_clip(batch_size, device, rgb.dtype)
        inert_auxiliary = self._resolve_auxiliary_clean(
            rgb, clean.shape, kwargs.get("auxiliary_target")
        )
        # Low first supplies the stopped clean target; high second is the
        # consistency student.  Both receive ordinary RF supervision.
        prediction_lo = self._video_pass(
            noisy_lo,
            lo_t,
            lo_sigma,
            z_control,
            reference,
            context,
            clip_fea,
            inert_auxiliary,
        )
        prediction_hi = self._video_pass(
            noisy_hi,
            hi_t,
            hi_sigma,
            z_control,
            reference,
            context,
            clip_fea,
            inert_auxiliary,
        )
        if self._two_clock_training_calls != 2:
            raise TwoClockConsistencyError(
                "every training update must execute exactly two Wan calls"
            )

        loss_mask = self._build_loss_mask(rgb, mask, clean.shape)
        future_mask = loss_mask[:, :, history_frames:]
        target_future = velocity_target[:, :, history_frames:]
        flow_hi_per_sample = self._masked_per_sample_mse(
            prediction_hi.video_velocity[:, :, history_frames:],
            target_future,
            future_mask,
        )
        flow_lo_per_sample = self._masked_per_sample_mse(
            prediction_lo.video_velocity[:, :, history_frames:],
            target_future,
            future_mask,
        )
        flow_hi = flow_hi_per_sample.mean()
        flow_lo = flow_lo_per_sample.mean()
        base_loss = 0.5 * (flow_hi + flow_lo)

        x0_hi = rf_predicted_clean(
            noisy_hi, hi_sigma, prediction_hi.video_velocity
        )
        x0_lo = rf_predicted_clean(
            noisy_lo, lo_sigma, prediction_lo.video_velocity
        )
        consistency, normalization_energy = normalized_stopped_consistency_loss(
            x0_hi[:, :, history_frames:],
            x0_lo[:, :, history_frames:],
            clean[:, :, history_frames:],
            future_mask,
            epsilon=self.consistency_epsilon,
        )
        weighted_consistency = self.consistency_weight * consistency

        hi_nmse = self._masked_per_sample_nmse(
            x0_hi[:, :, history_frames:],
            clean[:, :, history_frames:],
            future_mask,
        ).mean()
        lo_nmse = self._masked_per_sample_nmse(
            x0_lo[:, :, history_frames:],
            clean[:, :, history_frames:],
            future_mask,
        ).mean()
        self.aux_losses.update(
            {
                "flow_loss": base_loss.detach(),
                "two_clock_consistency/flow_hi": flow_hi.detach(),
                "two_clock_consistency/flow_lo": flow_lo.detach(),
                "two_clock_consistency/raw_loss": consistency.detach(),
                "two_clock_consistency/weighted_loss": weighted_consistency.detach(),
                "two_clock_consistency/weight": base_loss.new_tensor(
                    self.consistency_weight
                ),
                "two_clock_consistency/model_calls": base_loss.new_tensor(2.0),
                "two_clock_consistency/shared_epsilon_trajectory": base_loss.new_tensor(
                    1.0
                ),
                "two_clock_consistency/normalization_energy": normalization_energy.detach(),
                "two_clock_consistency/x0_hi_nmse": hi_nmse.detach(),
                "two_clock_consistency/x0_lo_nmse": lo_nmse.detach(),
                "clock/sigma_hi_mean": hi_sigma.mean().detach(),
                "clock/sigma_lo_mean": lo_sigma.mean().detach(),
                "clock/sigma_hi_min_observed": hi_sigma.min().detach(),
                "clock/sigma_hi_max_observed": hi_sigma.max().detach(),
                "clock/sigma_lo_min_observed": lo_sigma.min().detach(),
                "clock/sigma_lo_max_observed": lo_sigma.max().detach(),
                "paired_audit/epsilon_probe": self._probe_mean(epsilon),
                "paired_audit/clean_probe": self._probe_mean(clean),
                "paired_audit/noisy_hi_probe": self._probe_mean(noisy_hi),
                "paired_audit/noisy_lo_probe": self._probe_mean(noisy_lo),
                "paired_audit/sigma_hi_mean": hi_sigma.mean().detach(),
                "paired_audit/sigma_lo_mean": lo_sigma.mean().detach(),
            }
        )
        clip_index = kwargs.get("clip_index")
        if isinstance(clip_index, Tensor):
            ids = clip_index.detach().float().reshape(-1)
            self.aux_losses["paired_audit/clip_index_mean"] = ids.mean()
            self.aux_losses["paired_audit/clip_index_square_mean"] = (
                ids.square().mean()
            )

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
        cuda_rng = (
            torch.cuda.get_rng_state(device)
            if clean.is_cuda
            else torch.empty(0, dtype=torch.uint8)
        )
        self.paired_audit_exact = {
            "clip_index": tensor_sha256(clip_tensor),
            "actions": tensor_sha256(action_tensor),
            "clean_latent": tensor_sha256(clean),
            "epsilon": tensor_sha256(epsilon),
            "sigma_hi": tensor_sha256(hi_sigma),
            "sigma_lo": tensor_sha256(lo_sigma),
            "timestep_hi": tensor_sha256(hi_t),
            "timestep_lo": tensor_sha256(lo_t),
            "noisy_hi": tensor_sha256(noisy_hi),
            "noisy_lo": tensor_sha256(noisy_lo),
            "cpu_rng_state_after_forward": tensor_sha256(torch.get_rng_state()),
            "cuda_rng_state_after_forward": tensor_sha256(cuda_rng),
        }

        # The control executes both calls and diagnostics but attaches no
        # consistency graph to its objective.
        if self.consistency_weight == 0.0:
            return base_loss
        total = base_loss + weighted_consistency.to(base_loss.dtype)
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError("two-clock training objective is non-finite")
        return total
