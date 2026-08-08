"""Initial-function-preserving explicit action-token continuation of the VPM.

The native VPM action route is retained byte-for-byte.  A second route emits
one token for each of the 8 x 5 requested future action samples and adds those
tokens to the existing null-text context.  Wan then consumes them through its
pretrained text embedding and native cross-attention in every transformer
block.  Both arms instantiate and execute the adapter; only the exact scalar
gate differs in whether it is allowed to affect context.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from lam.dual_explicit_action_dit_model import DualExplicitActionDiTModel
from robot_wm.modeling.networks.action_token_context import (
    ActionTokenContextAdapter,
)


def tensor_sha256(value: Tensor) -> str:
    detached = value.detach().contiguous()
    header = f"{detached.dtype}|{tuple(detached.shape)}|".encode("ascii")
    raw = detached.view(torch.uint8).cpu().numpy().tobytes(order="C")
    return hashlib.sha256(header + raw).hexdigest()


class ActionTokenContractError(RuntimeError):
    """The paired continuation or deployable action boundary changed."""


class ActionTokenConditionedVPM(DualExplicitActionDiTModel):
    """VPM with a zero-gated, per-substep native cross-attention route."""

    def __init__(
        self,
        *,
        action_token: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        config = dict(action_token)
        required = {
            "enabled",
            "stats_path",
            "expected_stats_sha256",
            "action_dim",
            "chunk_size",
            "num_transitions",
            "morph_dim",
            "hidden",
            "text_dim",
            "clip_value",
            "initialization_seed",
        }
        if set(config) != required:
            raise ValueError(
                "action_token keys differ from the prospective contract: "
                f"missing={sorted(required - set(config))}, "
                f"extra={sorted(set(config) - required)}"
            )
        super().__init__(**kwargs)
        if (
            not bool(getattr(self, "parameter_matched_control", False))
            or self.tf_condition_mode != "off"
            or self.condition_on_tf
            or bool(getattr(self, "condition_on_tf_clock", False))
            or self.tf_loss_weight != 0.0
        ):
            raise ActionTokenContractError(
                "action-token screen requires the exact VPM no-op auxiliary arm"
            )
        if int(config["text_dim"]) != int(self.text_dim):
            raise ActionTokenContractError("adapter/Wan raw context dimensions differ")
        if int(config["num_transitions"]) != int(self.num_future_frames):
            raise ActionTokenContractError("adapter/future transition counts differ")
        wan_text_len = int(getattr(self.forward_model.transformer, "text_len", -1))
        if wan_text_len != 512:
            raise ActionTokenContractError("prospective screen fixes Wan text length to 512")

        self.action_token_adapter = ActionTokenContextAdapter(**config)
        self.action_token_enabled = bool(config["enabled"])
        if self.action_token_enabled != self.action_token_adapter.enabled:
            raise ActionTokenContractError("action-token arm flag differs")
        self.paired_audit_exact: dict[str, str] = {}
        self.action_token_telemetry: dict[str, Tensor] = {}
        self._capture_action_token_audit = False
        self._current_action_context_tokens: Tensor | None = None
        self._wan_input_hook = self.forward_model.register_forward_pre_hook(
            self._capture_wan_inputs,
            with_kwargs=True,
        )

    def _capture_wan_inputs(
        self,
        _module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        if not self._capture_action_token_audit:
            return
        fields = ("noisy_latent", "timesteps", "z_control", "reference")
        values: list[Any] = list(args[:4])
        if len(values) < 4:
            names = ("noisy_latents", "timesteps", "z_control", "ref_latents")
            values.extend(kwargs.get(name) for name in names[len(values) :])
        if len(values) != 4 or any(not isinstance(value, Tensor) for value in values):
            raise ActionTokenContractError(
                "Wan input audit did not receive four tensors"
            )
        for field, value in zip(fields, values):
            self.paired_audit_exact[field] = tensor_sha256(value)

    @torch.no_grad()
    def _encode_clip(self, rgb: Tensor) -> Tensor:
        latent = super()._encode_clip(rgb)
        if self._capture_action_token_audit:
            self.paired_audit_exact["clean_latent"] = tensor_sha256(latent)
        return latent

    def _resolve_auxiliary_clean(
        self,
        rgb: Tensor,
        latent_shape: tuple[int, ...],
        auxiliary_target: Tensor | None,
    ) -> Tensor:
        """Keep the inherited parameter-matched auxiliary topology inert."""

        if auxiliary_target is not None:
            raise ActionTokenContractError(
                "action-token training forbids cached auxiliary targets"
            )
        if len(latent_shape) != 5 or int(latent_shape[0]) != int(rgb.shape[0]):
            raise ActionTokenContractError("invalid video latent geometry")
        channels = int(self.forward_model.tf_token_adapter.tf_channels)
        return rgb.new_zeros(
            int(latent_shape[0]),
            channels,
            *tuple(int(value) for value in latent_shape[2:]),
        )

    def _latent_actions(
        self,
        rgb: Tensor,
        actions: Tensor | None,
        morphology_index: Tensor | None,
        Fp: int,
        K: int,
    ) -> tuple[Tensor, Tensor, None]:
        if actions is None:
            raise ActionTokenContractError("requested actions are required")
        future = self._future_action_chunks(actions)
        if self.morphology_tokens is not None and morphology_index is not None:
            morphology = self.morphology_tokens(morphology_index)
        else:
            morphology = future.new_zeros(future.shape[0], self.latent_action_dim)

        # The historical action encoder/pool/control path is intentionally
        # unchanged.  The second route branches from the same requested input.
        base = self.action_encoder(future, morphology)
        control = self._future_control(base.mean(dim=2), Fp, K)
        raw_tokens, standardized = self.action_token_adapter(future, morphology)
        self._current_action_context_tokens = raw_tokens

        gate = self.action_token_adapter.effective_gate().to(
            device=raw_tokens.device, dtype=raw_tokens.dtype
        )
        gated_tokens = gate * raw_tokens
        self.action_token_telemetry = {
            "effective_gate": gate.detach().float(),
            "standardized_action_rms": standardized.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "base_latent_rms": base.detach().float().square().mean().sqrt(),
            "raw_context_token_rms": raw_tokens.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "gated_context_token_rms": gated_tokens.detach()
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
        if self._capture_action_token_audit:
            for name, value in (
                ("actions", actions),
                ("future_actions", future),
                ("standardized_actions", standardized),
                ("base_action_latent", base),
                ("raw_action_context_tokens", raw_tokens),
                ("gated_action_context_tokens", gated_tokens),
                ("action_control", control),
            ):
                self.paired_audit_exact[name] = tensor_sha256(value)
        return base, control, None

    def _build_context(self, batch_size: int, device: Any, dtype: torch.dtype):
        raw_tokens = self._current_action_context_tokens
        if raw_tokens is None:
            raise ActionTokenContractError(
                "action tokens must be prepared before Wan context"
            )
        if raw_tokens.shape[0] != batch_size:
            raise ActionTokenContractError("action-token/context batch differs")
        base = super()._build_context(batch_size, device, dtype)
        stacked = torch.stack(base, dim=0)
        count = int(self.action_token_adapter.token_count)
        text_len = int(self.forward_model.transformer.text_len)
        if stacked.ndim != 3 or stacked.shape[1] > text_len - count:
            raise ActionTokenContractError(
                "null context leaves too few padded positions for action tokens"
            )
        if stacked.shape[2] != self.action_token_adapter.text_dim:
            raise ActionTokenContractError("null context width differs")
        # The prepared asset contains one null UMT5 token.  Native Wan pads it
        # with raw zeros to `text_len` immediately before `text_embedding`.
        # Materialize that exact padding here, then use only the final 40 padded
        # positions.  At a zero gate the tensor entering `text_embedding` is
        # identical to the parent's internally padded raw context.
        if stacked.shape[1] < text_len:
            stacked = torch.cat(
                (
                    stacked,
                    stacked.new_zeros(
                        batch_size,
                        text_len - int(stacked.shape[1]),
                        int(stacked.shape[2]),
                    ),
                ),
                dim=1,
            )
        gate = self.action_token_adapter.effective_gate().to(
            device=device, dtype=dtype
        )
        gated = raw_tokens.to(device=device, dtype=dtype) * gate
        context = stacked.clone()
        context[:, -count:, :] = context[:, -count:, :] + gated
        if self._capture_action_token_audit:
            self.paired_audit_exact["wan_raw_context"] = tensor_sha256(stacked)
            self.paired_audit_exact["wan_action_context"] = tensor_sha256(context)
        self.action_token_telemetry["null_context_rms"] = (
            stacked.detach().float().square().mean().sqrt()
        )
        self.action_token_telemetry["conditioned_context_rms"] = (
            context.detach().float().square().mean().sqrt()
        )
        return [context[index] for index in range(batch_size)]

    def forward(
        self,
        rgb: Tensor,
        actions: Tensor | None = None,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        self.paired_audit_exact = {}
        self._current_action_context_tokens = None
        self._capture_action_token_audit = bool(self.training)
        try:
            loss = super().forward(rgb, actions=actions, mask=mask, **kwargs)
        finally:
            self._capture_action_token_audit = False
        if self.training:
            expected = {
                "actions",
                "future_actions",
                "standardized_actions",
                "base_action_latent",
                "raw_action_context_tokens",
                "gated_action_context_tokens",
                "action_control",
                "wan_raw_context",
                "wan_action_context",
                "clean_latent",
                "noisy_latent",
                "timesteps",
                "z_control",
                "reference",
            }
            if set(self.paired_audit_exact) != expected:
                raise ActionTokenContractError(
                    "paired action/noise/context audit is incomplete"
                )
            for key, value in self.action_token_telemetry.items():
                self.aux_losses[f"action_token/{key}"] = value
        return loss
