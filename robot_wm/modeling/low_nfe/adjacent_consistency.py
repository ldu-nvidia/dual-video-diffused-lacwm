"""Auditable primitives for feature-free adjacent consistency distillation.

The clock convention is the LACWM/Wan rectified-flow convention: ``sigma=1``
is Gaussian noise and ``sigma=0`` is clean data.  This module owns no model,
dataset, checkpoint, or launcher state.  Its APIs deliberately have no clean
future *conditioning* or feature argument.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn


class AdjacentConsistencyError(RuntimeError):
    """A frozen clock, boundary, mask, or EMA contract was violated."""


CANONICAL_MODEL_STATE_HASH_ALGORITHM = "snapshot_model_state_receipt_v1"
RUNTIME_TENSOR_STATE_HASH_ALGORITHM = "tensor_state_sha256_v1"


@dataclass(frozen=True)
class AdjacentClockPair:
    """One start/end pair on the immutable descending RF training grid."""

    start_index: int
    end_index: int
    stride: int
    start_sigma: Tensor
    end_sigma: Tensor
    start_timestep: Tensor
    end_timestep: Tensor

    @property
    def delta_sigma(self) -> Tensor:
        return self.end_sigma - self.start_sigma


@dataclass(frozen=True)
class EMAUpdateReceipt:
    """Structural receipt for one in-place target-student EMA update."""

    decay: float
    ema_parameter_tensors: int
    ema_parameter_values: int
    copied_parameter_tensors: int
    copied_parameter_values: int
    copied_floating_buffer_tensors: int
    copied_nonfloating_buffer_tensors: int


@dataclass(frozen=True)
class RFClockGridReceipt:
    """Exact identity of the fixed shifted 1000-node RF clock."""

    train_nodes: int
    training_sigma_nodes: int
    shift: float
    terminal_sigma: float
    timestep_sigma_relation_exact: bool


def tensor_sha256(value: Tensor) -> str:
    """Hash exact tensor bytes together with shape and dtype."""

    detached = value.detach().contiguous()
    header = f"{detached.dtype}|{tuple(detached.shape)}|".encode("ascii")
    raw = detached.view(torch.uint8).cpu().numpy().tobytes(order="C")
    return hashlib.sha256(header + raw).hexdigest()


def _validated_named_tensor_state(
    state: Mapping[str, Tensor], *, label: str
) -> list[tuple[str, Tensor]]:
    if not isinstance(state, Mapping) or not state:
        raise AdjacentConsistencyError(f"{label} must be a nonempty tensor mapping")
    result: list[tuple[str, Tensor]] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not name or not isinstance(value, Tensor):
            raise AdjacentConsistencyError(
                f"{label} must contain only named tensors"
            )
        result.append((name, value.detach().contiguous().cpu()))
    return result


def canonical_model_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Reproduce ``snapshot_model_state_receipt_v1`` exactly.

    This is the historical checkpoint-lineage hash.  It intentionally remains
    distinct from :func:`tensor_state_sha256`, the in-process runtime hash.
    """

    digest = hashlib.sha256()
    for name, value in _validated_named_tensor_state(
        state, label="canonical model state"
    ):
        item = {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        digest.update(
            json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash a complete named tensor state, including names/dtypes/shapes."""

    digest = hashlib.sha256()
    for name, value in _validated_named_tensor_state(
        state, label="runtime tensor state"
    ):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def model_state_hash_receipt(state: Mapping[str, Tensor]) -> dict[str, int | str]:
    """Seal both non-interchangeable hashes for one exact tensor state."""

    items = _validated_named_tensor_state(state, label="model state")
    normalized = {name: value for name, value in items}
    return {
        "canonical_model_state_sha256": canonical_model_state_sha256(normalized),
        "canonical_hash_algorithm": CANONICAL_MODEL_STATE_HASH_ALGORITHM,
        "runtime_tensor_state_sha256": tensor_state_sha256(normalized),
        "runtime_hash_algorithm": RUNTIME_TENSOR_STATE_HASH_ALGORITHM,
        "state_tensors": len(items),
        "state_values": sum(int(value.numel()) for _name, value in items),
    }


def require_model_state_hashes(
    state: Mapping[str, Tensor],
    *,
    expected_canonical_sha256: str,
    expected_runtime_sha256: str,
    label: str,
) -> dict[str, int | str]:
    """Fail closed unless each state hash matches its own named algorithm."""

    receipt = model_state_hash_receipt(state)
    if receipt["canonical_model_state_sha256"] != expected_canonical_sha256:
        raise AdjacentConsistencyError(
            f"{label} canonical checkpoint-lineage hash differs"
        )
    if receipt["runtime_tensor_state_sha256"] != expected_runtime_sha256:
        raise AdjacentConsistencyError(f"{label} runtime tensor-state hash differs")
    return receipt


def module_state_receipt(module: nn.Module) -> dict[str, int | str]:
    """Return a full-value hash and exact parameter/buffer capacity counts."""

    parameters = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    state = module.state_dict()
    return {
        "sha256": tensor_state_sha256(state),
        "hash_algorithm": RUNTIME_TENSOR_STATE_HASH_ALGORITHM,
        "state_tensors": len(state),
        "state_values": sum(int(value.numel()) for value in state.values()),
        "parameter_tensors": len(parameters),
        "parameter_values": sum(int(value.numel()) for value in parameters.values()),
        "trainable_parameter_values": sum(
            int(value.numel()) for value in parameters.values() if value.requires_grad
        ),
        "buffer_tensors": len(buffers),
        "buffer_values": sum(int(value.numel()) for value in buffers.values()),
    }


def _validate_state(value: Tensor, *, name: str) -> None:
    if value.ndim != 5:
        raise ValueError(f"{name} must have Wan [B,C,T,H,W] layout")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite floating point")


def validate_shift5_rf_clock(scheduler: Any) -> RFClockGridReceipt:
    """Fail unless a scheduler is the exact 1000-node, shift-5 Wan clock."""

    config = scheduler.config
    train_nodes = int(getattr(config, "num_train_timesteps"))
    shift = float(getattr(config, "shift"))
    timesteps = scheduler.timesteps.detach().float().cpu()
    sigmas = scheduler.sigmas.detach().float().cpu()
    if train_nodes != 1000 or shift != 5.0:
        raise AdjacentConsistencyError("ACD-P0 requires 1000 nodes and shift=5")
    if timesteps.ndim != 1 or sigmas.ndim != 1:
        raise AdjacentConsistencyError("RF clock tensors must be one-dimensional")
    if int(timesteps.numel()) != 1000 or int(sigmas.numel()) != 1000:
        raise AdjacentConsistencyError("RF clock node count differs")
    if not torch.equal(timesteps, sigmas * 1000.0):
        raise AdjacentConsistencyError("RF clock requires exact t=1000*sigma")
    if (
        float(sigmas[0]) != 1.0
        or not bool(torch.isfinite(sigmas).all())
        or not float(sigmas[-1]) > 0.0
    ):
        raise AdjacentConsistencyError(
            "RF training grid must start at noise and end above clean"
        )
    if bool((timesteps[1:] >= timesteps[:-1]).any()) or bool(
        (sigmas[1:] >= sigmas[:-1]).any()
    ):
        raise AdjacentConsistencyError("RF clock must be strictly descending")
    return RFClockGridReceipt(
        train_nodes=1000,
        training_sigma_nodes=1000,
        shift=5.0,
        # FlowMatchEulerDiscreteScheduler appends this explicit deployment
        # terminal in set_timesteps(); the sampler revalidates it every call.
        terminal_sigma=0.0,
        timestep_sigma_relation_exact=True,
    )


def _expanded_sigma(sigma: Tensor | float, state: Tensor) -> Tensor:
    value = torch.as_tensor(sigma, device=state.device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.expand(state.shape[0])
    if value.ndim != 1 or int(value.shape[0]) != int(state.shape[0]):
        raise ValueError("sigma must be scalar or have one value per batch sample")
    if not bool(torch.isfinite(value).all()) or bool(
        ((value < 0) | (value > 1)).any()
    ):
        raise ValueError("sigma must be finite and lie in [0,1]")
    return value.reshape(-1, 1, 1, 1, 1)


def adjacent_clock_pair(
    sigmas: Tensor,
    timesteps: Tensor,
    *,
    start_index: int,
    stride: int,
) -> AdjacentClockPair:
    """Resolve the fixed ``i -> min(i+stride,N-1)`` teacher step.

    ``sigmas`` and ``timesteps`` contain model-call nodes only; neither may
    include a terminal extra sigma.  The pair must move strictly toward clean.
    """

    if sigmas.ndim != 1 or timesteps.ndim != 1 or sigmas.shape != timesteps.shape:
        raise ValueError("sigmas and timesteps must share one-dimensional shape")
    if sigmas.numel() < 2:
        raise ValueError("clock grid must contain at least two nodes")
    if not sigmas.is_floating_point() or not timesteps.is_floating_point():
        raise ValueError("clock tensors must be floating point")
    if not bool(torch.isfinite(sigmas).all()) or not bool(
        torch.isfinite(timesteps).all()
    ):
        raise ValueError("clock tensors must be finite")
    if bool((sigmas[1:] >= sigmas[:-1]).any()) or bool(
        (timesteps[1:] >= timesteps[:-1]).any()
    ):
        raise AdjacentConsistencyError("RF clock grid must be strictly descending")
    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise TypeError("start_index must be an integer")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if start_index < 0 or start_index >= int(sigmas.numel()) - 1:
        raise ValueError("start_index must leave a strictly later clock node")
    end_index = min(start_index + stride, int(sigmas.numel()) - 1)
    if end_index <= start_index or not bool(sigmas[end_index] < sigmas[start_index]):
        raise AdjacentConsistencyError("adjacent teacher step has zero/wrong sign")
    return AdjacentClockPair(
        start_index=start_index,
        end_index=end_index,
        stride=stride,
        start_sigma=sigmas[start_index],
        end_sigma=sigmas[end_index],
        start_timestep=timesteps[start_index],
        end_timestep=timesteps[end_index],
    )


def rectified_flow_noisy_state(
    clean: Tensor,
    noise: Tensor,
    sigma: Tensor | float,
) -> Tensor:
    """Return ``(1-sigma)*clean + sigma*noise`` in float32 arithmetic."""

    _validate_state(clean, name="clean")
    _validate_state(noise, name="noise")
    if clean.shape != noise.shape:
        raise ValueError("clean and noise must share shape")
    expanded = _expanded_sigma(sigma, clean)
    return (1.0 - expanded) * clean.float() + expanded * noise.float()


def rectified_flow_euler_step(
    state: Tensor,
    velocity: Tensor,
    sigma: Tensor | float,
    next_sigma: Tensor | float,
) -> Tensor:
    """Take one decreasing-sigma Euler step with an explicit sign check."""

    _validate_state(state, name="state")
    _validate_state(velocity, name="velocity")
    if state.shape != velocity.shape:
        raise ValueError("state and velocity must share shape")
    start = _expanded_sigma(sigma, state)
    end = _expanded_sigma(next_sigma, state)
    if bool((end >= start).any()):
        raise AdjacentConsistencyError(
            "denoising requires next_sigma < sigma (1=noise,0=clean)"
        )
    return state.float() + (end - start) * velocity.float()


def restore_history_forward_path(
    state: Tensor,
    reference: Tensor,
    noise: Tensor,
    sigma: Tensor | float,
    *,
    history_tokens: int,
) -> Tensor:
    """Restore only the observed prefix to its known RF corruption path."""

    _validate_state(state, name="state")
    _validate_state(reference, name="reference")
    _validate_state(noise, name="noise")
    if state.shape != reference.shape or state.shape != noise.shape:
        raise ValueError("state, reference, and noise must share shape")
    if (
        isinstance(history_tokens, bool)
        or not isinstance(history_tokens, int)
        or history_tokens < 1
        or history_tokens >= int(state.shape[2])
    ):
        raise ValueError("history_tokens must select a nonempty proper prefix")
    if bool(reference[:, :, history_tokens:].ne(0).any()):
        raise AdjacentConsistencyError("reference future support must be exact zero")
    result = state.clone()
    expanded = _expanded_sigma(sigma, state)
    result[:, :, :history_tokens] = (
        (1.0 - expanded) * reference.float()
        + expanded * noise.float()
    )[:, :, :history_tokens]
    return result


def karras_boundary_scalings(
    sigma: Tensor | float,
    *,
    sigma_data: float,
) -> tuple[Tensor, Tensor]:
    """Return the released Flash-WAM Karras consistency scalings."""

    if not math.isfinite(float(sigma_data)) or float(sigma_data) <= 0:
        raise ValueError("sigma_data must be finite and positive")
    value = torch.as_tensor(sigma, dtype=torch.float32)
    if not bool(torch.isfinite(value).all()) or bool(
        ((value < 0) | (value > 1)).any()
    ):
        raise ValueError("sigma must be finite and lie in [0,1]")
    data_sq = float(sigma_data) ** 2
    denominator = value.square() + data_sq
    c_skip = data_sq / denominator
    c_out = value * float(sigma_data) / denominator.sqrt()
    return c_skip, c_out


def consistency_output(
    noisy: Tensor,
    velocity: Tensor,
    sigma: Tensor | float,
    *,
    sigma_data: float,
) -> Tensor:
    """Compute Karras-boundary ``F(x,sigma)`` from RF velocity prediction."""

    _validate_state(noisy, name="noisy")
    _validate_state(velocity, name="velocity")
    if noisy.shape != velocity.shape:
        raise ValueError("noisy and velocity must share shape")
    expanded = _expanded_sigma(sigma, noisy)
    c_skip, c_out = karras_boundary_scalings(
        expanded, sigma_data=sigma_data
    )
    clean = predicted_clean(noisy, velocity, sigma)
    result = c_skip.to(noisy.device) * noisy.float() + c_out.to(
        noisy.device
    ) * clean
    if not bool(torch.isfinite(result).all()):
        raise AdjacentConsistencyError("consistency output is non-finite")
    return result


def predicted_clean(
    noisy: Tensor,
    velocity: Tensor,
    sigma: Tensor | float,
) -> Tensor:
    """Return the RF clean estimate ``x_sigma - sigma*v`` in float32."""

    _validate_state(noisy, name="noisy")
    _validate_state(velocity, name="velocity")
    if noisy.shape != velocity.shape:
        raise ValueError("noisy and velocity must share shape")
    expanded = _expanded_sigma(sigma, noisy)
    result = noisy.float() - expanded * velocity.float()
    if not bool(torch.isfinite(result).all()):
        raise AdjacentConsistencyError("predicted clean state is non-finite")
    return result


def _expanded_mask(mask: Tensor, target: Tensor) -> Tensor:
    if mask.ndim != 5 or target.ndim != 5:
        raise ValueError("mask and target must use [B,C,T,H,W]-broadcast layout")
    try:
        weights = mask.to(device=target.device, dtype=torch.float32).expand_as(target)
    except RuntimeError as exc:
        raise ValueError("mask is not broadcast-compatible with target") from exc
    reduce_dims = tuple(range(1, target.ndim))
    if bool((weights.sum(dim=reduce_dims) <= 0).any()):
        raise AdjacentConsistencyError("every sample needs nonempty future support")
    return weights


def masked_pseudo_huber(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    c: float,
) -> tuple[Tensor, Tensor]:
    """Return mean and per-sample pseudo-Huber loss over the explicit mask."""

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must share [B,C,T,H,W] shape")
    if not math.isfinite(float(c)) or float(c) <= 0:
        raise ValueError("pseudo-Huber c must be finite and positive")
    weights = _expanded_mask(mask, prediction)
    difference = prediction.float() - target.detach().float()
    elementwise = torch.sqrt(difference.square() + float(c) ** 2) - float(c)
    reduce_dims = tuple(range(1, prediction.ndim))
    denominator = weights.sum(dim=reduce_dims)
    per_sample = (elementwise * weights).sum(dim=reduce_dims) / denominator
    if not bool(torch.isfinite(per_sample).all()):
        raise AdjacentConsistencyError("pseudo-Huber loss is non-finite")
    return per_sample.mean(), per_sample


def ema_update_module_(
    target: nn.Module,
    source: nn.Module,
    *,
    decay: float,
) -> EMAUpdateReceipt:
    """Update a frozen target module after a successful optimizer step.

    Only trainable source parameters use EMA. Frozen parameters and every
    buffer are copied exactly, preventing repeated floating-point arithmetic
    from drifting immutable backbone/VAE state. Module keys, shapes, and
    dtypes must match exactly. The function never mutates ``source`` and
    rejects a target that still requires gradients.
    """

    if target is source:
        raise ValueError("EMA target and source must be distinct modules")
    if not math.isfinite(float(decay)) or not 0 <= float(decay) < 1:
        raise ValueError("EMA decay must be finite and lie in [0,1)")
    target_parameters = dict(target.named_parameters())
    source_parameters = dict(source.named_parameters())
    if target_parameters.keys() != source_parameters.keys():
        raise AdjacentConsistencyError("EMA parameter names differ")
    if any(parameter.requires_grad for parameter in target_parameters.values()):
        raise AdjacentConsistencyError("EMA target parameters must be frozen")
    ema_parameter_tensors = 0
    ema_parameter_values = 0
    copied_parameter_tensors = 0
    copied_parameter_values = 0
    with torch.no_grad():
        for name in target_parameters:
            target_value = target_parameters[name]
            source_value = source_parameters[name]
            if (
                target_value.shape != source_value.shape
                or target_value.dtype != source_value.dtype
            ):
                raise AdjacentConsistencyError(f"EMA parameter differs: {name}")
            if source_value.requires_grad:
                if not target_value.is_floating_point():
                    raise AdjacentConsistencyError(
                        f"trainable parameter is unexpectedly non-floating: {name}"
                    )
                target_value.mul_(float(decay)).add_(
                    source_value.detach(), alpha=1.0 - float(decay)
                )
                ema_parameter_tensors += 1
                ema_parameter_values += int(target_value.numel())
            else:
                target_value.copy_(source_value.detach())
                copied_parameter_tensors += 1
                copied_parameter_values += int(target_value.numel())

        target_buffers = dict(target.named_buffers())
        source_buffers = dict(source.named_buffers())
        if target_buffers.keys() != source_buffers.keys():
            raise AdjacentConsistencyError("EMA buffer names differ")
        copied_floating_buffers = 0
        copied_nonfloating_buffers = 0
        for name in target_buffers:
            target_value = target_buffers[name]
            source_value = source_buffers[name]
            if (
                target_value.shape != source_value.shape
                or target_value.dtype != source_value.dtype
            ):
                raise AdjacentConsistencyError(f"EMA buffer differs: {name}")
            target_value.copy_(source_value.detach())
            if target_value.is_floating_point():
                copied_floating_buffers += 1
            else:
                copied_nonfloating_buffers += 1
    return EMAUpdateReceipt(
        decay=float(decay),
        ema_parameter_tensors=ema_parameter_tensors,
        ema_parameter_values=ema_parameter_values,
        copied_parameter_tensors=copied_parameter_tensors,
        copied_parameter_values=copied_parameter_values,
        copied_floating_buffer_tensors=copied_floating_buffers,
        copied_nonfloating_buffer_tensors=copied_nonfloating_buffers,
    )


def state_probe(module: nn.Module, *, values_per_tensor: int = 2) -> Tensor:
    """Return a cheap deterministic float64 probe for per-update EMA tracing."""

    if values_per_tensor < 1:
        raise ValueError("values_per_tensor must be positive")
    values: list[Tensor] = []
    for _name, tensor in module.state_dict().items():
        if tensor.numel() == 0:
            continue
        flat = tensor.detach().reshape(-1)
        indexes = torch.linspace(
            0,
            flat.numel() - 1,
            min(values_per_tensor, flat.numel()),
            device=flat.device,
        ).round().long()
        values.append(flat[indexes].to(dtype=torch.float64))
    if not values:
        raise AdjacentConsistencyError("module state has no probeable values")
    return torch.cat(values)


def receipt_dict(receipt: EMAUpdateReceipt) -> dict[str, Any]:
    """Serialize an EMA receipt without depending on dataclass internals."""

    return {
        "decay": receipt.decay,
        "ema_parameter_tensors": receipt.ema_parameter_tensors,
        "ema_parameter_values": receipt.ema_parameter_values,
        "copied_parameter_tensors": receipt.copied_parameter_tensors,
        "copied_parameter_values": receipt.copied_parameter_values,
        "copied_floating_buffer_tensors": receipt.copied_floating_buffer_tensors,
        "copied_nonfloating_buffer_tensors": receipt.copied_nonfloating_buffer_tensors,
    }
