"""Initial-function-preserving action-variation continuation of the VPM.

The historical explicit-action path strongly compresses clip identity before
Wan.  This model keeps that path intact and adds a zero-gated residual derived
from train-whitened within-chunk action velocities.  Both experimental arms
instantiate and execute the residual; only whether its gate may affect the
action latent differs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from lam.dual_explicit_action_dit_model import DualExplicitActionDiTModel
from robot_wm.modeling.networks.action_delta_residual import (
    WhitenedActionDeltaResidual,
)


def tensor_sha256(value: Tensor) -> str:
    detached = value.detach().contiguous()
    header = f"{detached.dtype}|{tuple(detached.shape)}|".encode("ascii")
    raw = detached.view(torch.uint8).cpu().numpy().tobytes(order="C")
    return hashlib.sha256(header + raw).hexdigest()


class ActionVariationContractError(RuntimeError):
    """The paired continuation or deployable action boundary changed."""


class ActionVariationConditionedVPM(DualExplicitActionDiTModel):
    """VPM with a train-whitened, zero-gated requested-action residual."""

    def __init__(
        self,
        *,
        action_variation: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        variation = dict(action_variation)
        required = {
            "enabled",
            "stats_path",
            "expected_stats_sha256",
            "action_dim",
            "chunk_size",
            "latent_dim",
            "morph_dim",
            "hidden",
            "num_layers",
            "clip_value",
            "initialization_seed",
        }
        if set(variation) != required:
            raise ValueError(
                "action_variation keys differ from the prospective contract: "
                f"missing={sorted(required - set(variation))}, "
                f"extra={sorted(set(variation) - required)}"
            )
        super().__init__(**kwargs)
        if (
            not bool(getattr(self, "parameter_matched_control", False))
            or self.tf_condition_mode != "off"
            or self.condition_on_tf
            or bool(getattr(self, "condition_on_tf_clock", False))
            or self.tf_loss_weight != 0.0
        ):
            raise ActionVariationContractError(
                "action-variation screen requires the exact VPM no-op auxiliary arm"
            )
        self.action_delta_residual = WhitenedActionDeltaResidual(**variation)
        self.action_variation_enabled = bool(variation["enabled"])
        if self.action_variation_enabled != self.action_delta_residual.enabled:
            raise ActionVariationContractError("action-variation arm flag differs")
        self.paired_audit_exact: dict[str, str] = {}
        self.action_variation_telemetry: dict[str, Tensor] = {}
        self._capture_action_variation_audit = False
        self._wan_input_hook = self.forward_model.register_forward_pre_hook(
            self._capture_wan_inputs,
            with_kwargs=True,
        )

    @staticmethod
    def _probe(value: Tensor) -> Tensor:
        flat = value.detach().float().reshape(value.shape[0], -1)
        return flat[:, : min(16, int(flat.shape[1]))].mean()

    def _capture_wan_inputs(
        self,
        _module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        if not self._capture_action_variation_audit:
            return
        fields = ("noisy_latent", "timesteps", "z_control", "reference")
        values: list[Any] = list(args[:4])
        if len(values) < 4:
            names = ("noisy_latents", "timesteps", "z_control", "ref_latents")
            values.extend(kwargs.get(name) for name in names[len(values) :])
        if len(values) != 4 or any(not isinstance(value, Tensor) for value in values):
            raise ActionVariationContractError(
                "Wan input audit did not receive four tensors"
            )
        for field, value in zip(fields, values):
            self.paired_audit_exact[field] = tensor_sha256(value)

    @torch.no_grad()
    def _encode_clip(self, rgb: Tensor) -> Tensor:
        latent = super()._encode_clip(rgb)
        if self._capture_action_variation_audit:
            self.paired_audit_exact["clean_latent"] = tensor_sha256(latent)
        return latent

    def _latent_actions(
        self,
        rgb: Tensor,
        actions: Tensor | None,
        morphology_index: Tensor | None,
        Fp: int,
        K: int,
    ) -> tuple[Tensor, Tensor, None]:
        if actions is None:
            raise ActionVariationContractError("requested actions are required")
        future = self._future_action_chunks(actions)
        if self.morphology_tokens is not None and morphology_index is not None:
            morphology = self.morphology_tokens(morphology_index)
        else:
            morphology = future.new_zeros(future.shape[0], self.latent_action_dim)
        base = self.action_encoder(future, morphology)
        residual, standardized = self.action_delta_residual(future, morphology)
        gate = self.action_delta_residual.effective_gate().to(
            device=base.device, dtype=base.dtype
        )
        augmented = base + gate * residual
        control = self._future_control(augmented.mean(dim=2), Fp, K)

        self.action_variation_telemetry = {
            "effective_gate": gate.detach().float(),
            "standardized_delta_rms": standardized.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "base_latent_rms": base.detach().float().square().mean().sqrt(),
            "residual_latent_rms": residual.detach().float().square().mean().sqrt(),
            "gated_residual_rms": (gate * residual)
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "control_rms": control.detach().float().square().mean().sqrt(),
            "control_sample_std_rms": control.detach()
            .float()
            .std(dim=0, unbiased=False)
            .square()
            .mean()
            .sqrt(),
        }
        if self._capture_action_variation_audit:
            for name, value in (
                ("actions", actions),
                ("future_actions", future),
                ("standardized_action_delta", standardized),
                ("base_action_latent", base),
                ("delta_action_latent", residual),
                ("augmented_action_latent", augmented),
                ("action_control", control),
            ):
                self.paired_audit_exact[name] = tensor_sha256(value)
        return augmented, control, None

    def forward(
        self,
        rgb: Tensor,
        actions: Tensor | None = None,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        self.paired_audit_exact = {}
        self._capture_action_variation_audit = bool(self.training)
        try:
            loss = super().forward(rgb, actions=actions, mask=mask, **kwargs)
        finally:
            self._capture_action_variation_audit = False
        if self.training:
            expected = {
                "actions",
                "future_actions",
                "standardized_action_delta",
                "base_action_latent",
                "delta_action_latent",
                "augmented_action_latent",
                "action_control",
                "clean_latent",
                "noisy_latent",
                "timesteps",
                "z_control",
                "reference",
            }
            if set(self.paired_audit_exact) != expected:
                raise ActionVariationContractError(
                    "paired action/noise audit is incomplete"
                )
            for key, value in self.action_variation_telemetry.items():
                self.aux_losses[f"action_variation/{key}"] = value
        return loss
