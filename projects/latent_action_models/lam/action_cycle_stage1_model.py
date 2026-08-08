"""VPM continuation with a training-only frozen inverse-action cycle loss.

The ordinary VPM Wan trunk is called exactly once.  From its rectified-flow
velocity we form ``x0_hat = x_sigma - sigma * v_theta`` and apply the frozen
train-only Stage-0 ridge critic to the same latent displacement representation
used by the recoverability probe.  Requested actions supervise only the two
transitions that touch generated future latents.  No critic parameter or
buffer is written to the world-model state dict, and deployment mode never
loads the critic bundle.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from lam.dual_explicit_action_dit_model import DualExplicitActionDiTModel
from robot_wm.modeling.dual_diffusion.action_cycle import (
    FUTURE_RELEVANT_TRANSITIONS,
    ActionCycleError,
    FrozenStage0RidgeCritic,
    critic_is_absent_from_model_state,
    rf_predicted_clean,
)
from robot_wm.modeling.networks.wan_forward_model import DualWanOutput


def tensor_sha256(value: Tensor) -> str:
    detached = value.detach().contiguous()
    header = f"{detached.dtype}|{tuple(detached.shape)}|".encode("ascii")
    raw = detached.view(torch.uint8).cpu().numpy().tobytes(order="C")
    return hashlib.sha256(header + raw).hexdigest()


class ActionCycleStage1VPM(DualExplicitActionDiTModel):
    """Parameter-identical VPM with OFF/ON training and critic-free deploy modes."""

    MODES = {"off", "on", "deploy"}

    def __init__(self, *, action_cycle: Mapping[str, Any], **kwargs: Any) -> None:
        config = dict(action_cycle)
        self.action_cycle_mode = str(config.get("mode", "off"))
        self.action_cycle_loss_weight = float(config.get("loss_weight", 0.0))
        self.action_cycle_transitions = tuple(
            int(value) for value in config.get("transitions", (1, 2))
        )
        if self.action_cycle_mode not in self.MODES:
            raise ValueError(f"action_cycle.mode must be one of {sorted(self.MODES)}")
        if (
            not math.isfinite(self.action_cycle_loss_weight)
            or self.action_cycle_loss_weight < 0
        ):
            raise ValueError("action-cycle loss weight must be finite and nonnegative")
        if self.action_cycle_transitions != FUTURE_RELEVANT_TRANSITIONS:
            raise ValueError("Stage 1 is frozen to future-relevant transitions [1,2]")
        expected_weight = {"off": 0.0, "on": 0.05, "deploy": 0.0}[
            self.action_cycle_mode
        ]
        if self.action_cycle_loss_weight != expected_weight:
            raise ValueError(
                f"{self.action_cycle_mode} mode requires loss weight {expected_weight}"
            )
        super().__init__(**kwargs)
        if (
            not bool(getattr(self, "parameter_matched_control", False))
            or self.tf_condition_mode != "off"
            or self.condition_on_tf
            or bool(getattr(self, "condition_on_tf_clock", False))
            or self.tf_loss_weight != 0.0
        ):
            raise ActionCycleError(
                "action-cycle screen requires the exact parameter-matched VPM no-op arm"
            )

        self.action_cycle_critic: FrozenStage0RidgeCritic | None = None
        critic_path = config.get("critic_path")
        critic_sha256 = str(config.get("critic_sha256", ""))
        stage0_identity = str(
            config.get("stage0_registration_identity_sha256", "")
        )
        if self.action_cycle_mode == "deploy":
            if critic_path not in (None, "") or critic_sha256 or stage0_identity:
                raise ActionCycleError(
                    "deployment mode forbids a critic path, digest, or Stage-0 identity"
                )
        else:
            if not critic_path:
                raise ActionCycleError("OFF/ON training requires the sealed critic bundle")
            self.action_cycle_critic = FrozenStage0RidgeCritic(
                critic_path,
                expected_sha256=critic_sha256,
                expected_stage0_registration_identity=stage0_identity,
            )

        self._action_cycle_capture_active = False
        self._action_cycle_clean_full: Tensor | None = None
        self._action_cycle_noisy: Tensor | None = None
        self._action_cycle_timesteps: Tensor | None = None
        self._action_cycle_prediction: DualWanOutput | None = None
        self.paired_audit_exact: dict[str, str] = {}
        self._action_cycle_pre_hook = self.forward_model.register_forward_pre_hook(
            self._capture_forward_input, with_kwargs=True
        )
        self._action_cycle_post_hook = self.forward_model.register_forward_hook(
            self._capture_forward_output
        )
        if not critic_is_absent_from_model_state(self):
            raise ActionCycleError("training-only critic leaked into model state/module schema")

    def _resolve_auxiliary_clean(
        self,
        rgb: Tensor,
        latent_shape: tuple[int, ...],
        auxiliary_target: Tensor | None,
    ) -> Tensor:
        """Keep the inherited VPM auxiliary topology inert in both train arms."""

        if auxiliary_target is not None:
            raise ActionCycleError("action-cycle training forbids clean feature targets")
        if len(latent_shape) != 5 or int(latent_shape[0]) != int(rgb.shape[0]):
            raise ActionCycleError("invalid video latent geometry")
        channels = int(self.forward_model.tf_token_adapter.tf_channels)
        return rgb.new_zeros(
            int(latent_shape[0]),
            channels,
            *tuple(int(value) for value in latent_shape[2:]),
        )

    def _encode_clip(self, rgb: Tensor) -> Tensor:
        encoded = super()._encode_clip(rgb)
        expected_frames = self.num_history_frames + self.num_future_frames
        if self._action_cycle_capture_active and int(rgb.shape[1]) == expected_frames:
            if self._action_cycle_clean_full is not None:
                raise ActionCycleError("full clean video was encoded more than once")
            self._action_cycle_clean_full = encoded
        return encoded

    def _capture_forward_input(
        self,
        _module: Any,
        args: tuple[Any, ...],
        _kwargs: Mapping[str, Any],
    ) -> None:
        if not self._action_cycle_capture_active:
            return
        if len(args) < 2 or not isinstance(args[0], Tensor) or not isinstance(args[1], Tensor):
            raise ActionCycleError("cannot capture VPM noisy state/timestep")
        if self._action_cycle_noisy is not None or self._action_cycle_timesteps is not None:
            raise ActionCycleError("VPM forward was called more than once")
        self._action_cycle_noisy = args[0]
        self._action_cycle_timesteps = args[1]

    def _capture_forward_output(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not self._action_cycle_capture_active:
            return
        if not isinstance(output, DualWanOutput):
            raise ActionCycleError("VPM did not return DualWanOutput")
        if self._action_cycle_prediction is not None:
            raise ActionCycleError("VPM produced more than one captured output")
        self._action_cycle_prediction = output

    def _reset_capture(self) -> None:
        self._action_cycle_clean_full = None
        self._action_cycle_noisy = None
        self._action_cycle_timesteps = None
        self._action_cycle_prediction = None

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
        if self.action_cycle_mode == "deploy":
            raise ActionCycleError("deployment-only model may not execute training forward")
        if actions is None or self.action_cycle_critic is None:
            raise ActionCycleError("action-cycle training requires actions and frozen critic")
        self._reset_capture()
        self._action_cycle_capture_active = True
        try:
            base_loss = super().forward(rgb, actions=actions, mask=mask, **kwargs)
        finally:
            self._action_cycle_capture_active = False

        clean = self._action_cycle_clean_full
        noisy = self._action_cycle_noisy
        timesteps = self._action_cycle_timesteps
        prediction = self._action_cycle_prediction
        if clean is None or noisy is None or timesteps is None or prediction is None:
            raise ActionCycleError("action-cycle forward capture is incomplete")
        if tuple(clean.shape) != tuple(noisy.shape) or tuple(clean.shape[1:]) != (
            16,
            4,
            24,
            120,
        ):
            raise ActionCycleError("captured Wan clean/noisy geometry differs")
        if self.num_history_latent != 2:
            raise ActionCycleError("Stage 1 requires two history and two future Wan bins")
        if mask is not None:
            future_mask = mask[
                :, self.num_history_frames : self.num_history_frames + self.num_future_frames
            ]
            if future_mask.numel() == 0 or not bool(future_mask.bool().all()):
                raise ActionCycleError("controlled screen requires every future frame valid")

        sigma = self._get_sigmas(
            timesteps, n_dim=2, dtype=torch.float32, device=noisy.device
        )[:, 0]
        predicted_clean = rf_predicted_clean(
            noisy, sigma, prediction.video_velocity
        )
        if self.action_cycle_mode == "off":
            with torch.no_grad():
                per_sample, telemetry = self.action_cycle_critic.predict_and_loss(
                    predicted_clean.detach(),
                    actions.detach(),
                    transitions=self.action_cycle_transitions,
                )
        else:
            per_sample, telemetry = self.action_cycle_critic.predict_and_loss(
                predicted_clean,
                actions,
                transitions=self.action_cycle_transitions,
            )
        cycle_loss = per_sample.mean()
        weighted = self.action_cycle_loss_weight * cycle_loss
        self.aux_losses["action_cycle/raw_inverse_action_mse"] = cycle_loss.detach()
        self.aux_losses["action_cycle/weighted_loss"] = weighted.detach()
        self.aux_losses["action_cycle/loss_weight"] = base_loss.new_tensor(
            self.action_cycle_loss_weight
        )
        self.aux_losses["action_cycle/enabled"] = base_loss.new_tensor(
            float(self.action_cycle_mode == "on")
        )
        self.aux_losses["action_cycle/future_transition_count"] = base_loss.new_tensor(2.0)
        self.aux_losses["action_cycle/critic_cosine"] = telemetry["cosine"].mean().detach()
        self.aux_losses["action_cycle/feature_rms"] = telemetry["feature_rms"].mean().detach()
        self.aux_losses["action_cycle/prediction_rms"] = telemetry[
            "prediction_rms"
        ].mean().detach()
        self.aux_losses["action_cycle/target_rms"] = telemetry["target_rms"].mean().detach()
        self.aux_losses["action_cycle/predicted_x0_rms"] = (
            predicted_clean.detach().square().mean().sqrt()
        )
        self.aux_losses["action_cycle/sigma_mean"] = sigma.detach().mean()

        clip_index = kwargs.get("clip_index")
        if isinstance(clip_index, Tensor):
            identifiers = clip_index.detach().float().reshape(-1)
            self.aux_losses["paired_audit/clip_index_mean"] = identifiers.mean()
            self.aux_losses["paired_audit/clip_index_square_mean"] = (
                identifiers.square().mean()
            )
        self.aux_losses["paired_audit/timestep_mean"] = timesteps.detach().float().mean()
        self.aux_losses["paired_audit/timestep_square_mean"] = (
            timesteps.detach().float().square().mean()
        )
        self.aux_losses["paired_audit/noisy_probe"] = self._probe_mean(noisy)
        self.aux_losses["paired_audit/clean_probe"] = self._probe_mean(clean)
        clip_tensor = (
            clip_index
            if isinstance(clip_index, Tensor)
            else torch.empty(0, dtype=torch.long, device=noisy.device)
        )
        cuda_rng = (
            torch.cuda.get_rng_state(noisy.device)
            if noisy.is_cuda
            else torch.empty(0, dtype=torch.uint8)
        )
        self.paired_audit_exact = {
            "clip_index": tensor_sha256(clip_tensor),
            "actions": tensor_sha256(actions),
            "clean_latent": tensor_sha256(clean),
            "noisy_latent": tensor_sha256(noisy),
            "timesteps": tensor_sha256(timesteps),
            "cpu_rng_state_after_forward": tensor_sha256(torch.get_rng_state()),
            "cuda_rng_state_after_forward": tensor_sha256(cuda_rng),
        }
        self._reset_capture()

        # OFF returns the exact parent loss object and attaches no zero-weight
        # auxiliary graph.  Both arms nevertheless execute the same critic
        # diagnostic and exactly one Wan call.
        if self.action_cycle_mode == "off":
            return base_loss
        total = base_loss + weighted.to(base_loss.dtype)
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError("action-cycle training objective is non-finite")
        return total

    def assert_deployable(self) -> None:
        """Fail unless this instance has no critic or persisted critic state."""

        if self.action_cycle_mode != "deploy" or self.action_cycle_critic is not None:
            raise ActionCycleError("deployment instance still owns a training critic")
        if not critic_is_absent_from_model_state(self):
            raise ActionCycleError("critic leaked into deployment state/module schema")
