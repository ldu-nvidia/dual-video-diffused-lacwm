"""Exact P/Q algebra for the IPQ-TC1 invertible two-clock pilot.

The functions in this module are deliberately independent of the Wan runtime.
They implement a view-isolated spatial Haar-LL projector, its exact complement,
mixed-clock rectified-flow corruption, rank-correct loss normalization, and
band-specific Euler integration.  Production sampling code does not import
this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor


PROTOCOL_VERSION = "ipq-tc1-v1"
ScheduleMode = Literal["synchronous", "p_leads", "q_leads"]


class InvertibleTwoClockError(RuntimeError):
    """The fixed IPQ-TC1 transform or clock contract was violated."""


def tensor_state_sha256(state: dict[str, Tensor]) -> str:
    """Hash an exact named tensor state, including dtype and shape."""

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().contiguous().cpu()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def parameter_schema(model: torch.nn.Module) -> dict[str, Any]:
    """Return a value-independent model capacity receipt, including frozen gates."""

    rows = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": parameter.numel(),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in model.named_parameters()
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "parameter_tensors": len(rows),
        "parameter_count": sum(value["numel"] for value in rows),
        "trainable_parameter_count": sum(
            value["numel"] for value in rows if value["requires_grad"]
        ),
    }


@dataclass(frozen=True)
class ProjectorReceipt:
    reconstruction_max_abs: float
    reconstruction_relative_energy: float
    normalized_inner_product: float
    p_idempotence_max_abs: float
    q_idempotence_max_abs: float
    history_nonzero: int
    rank_fraction_p: float
    rank_fraction_q: float


@dataclass(frozen=True)
class MixedClockState:
    clean_p: Tensor
    clean_q: Tensor
    noise_p: Tensor
    noise_q: Tensor
    noisy_p: Tensor
    noisy_q: Tensor
    native_noisy: Tensor
    target_p: Tensor
    target_q: Tensor


@dataclass(frozen=True)
class ProjectedPrediction:
    velocity_p: Tensor
    velocity_q: Tensor
    loss: Tensor
    combined_loss: Tensor
    loss_identity_relative_error: Tensor


def _validate_native_tensor(
    value: Tensor,
    *,
    history_frames: int,
    view_width: int,
) -> None:
    if value.ndim != 5:
        raise ValueError("native latent must have shape [B,C,T,H,W]")
    if history_frames < 0 or history_frames >= value.shape[2]:
        raise ValueError("history_frames must leave at least one future token")
    if view_width <= 0 or value.shape[-1] % view_width:
        raise ValueError("width must be an integer number of camera views")
    if value.shape[-2] % 2 or view_width % 2:
        raise ValueError("each view height and width must be divisible by two")


def _project_ll_view(value: Tensor) -> Tensor:
    """Project one `[B,C,T,H,W_view]` tensor onto 2x2 Haar LL."""

    batch, channels, frames, height, width = value.shape
    blocks = value.reshape(
        batch,
        channels,
        frames,
        height // 2,
        2,
        width // 2,
        2,
    )
    coarse = blocks.mean(dim=(4, 6), keepdim=True)
    return coarse.expand_as(blocks).reshape_as(value)


def project_p(
    value: Tensor,
    *,
    history_frames: int = 2,
    view_width: int = 40,
) -> Tensor:
    """Return the future-only, view-isolated spatial Haar-LL projection.

    Projection arithmetic is always FP32 as frozen by the protocol.  History
    is exact zero and each camera view is transformed independently.
    """

    _validate_native_tensor(
        value, history_frames=history_frames, view_width=view_width
    )
    source = value.float()
    result = torch.zeros_like(source)
    for start in range(0, source.shape[-1], view_width):
        stop = start + view_width
        result[:, :, history_frames:, :, start:stop] = _project_ll_view(
            source[:, :, history_frames:, :, start:stop]
        )
    return result


def future_only(
    value: Tensor,
    *,
    history_frames: int = 2,
    view_width: int = 40,
) -> Tensor:
    """Copy future slots in FP32 and set history to exact zero."""

    _validate_native_tensor(
        value, history_frames=history_frames, view_width=view_width
    )
    result = torch.zeros_like(value, dtype=torch.float32)
    result[:, :, history_frames:] = value[:, :, history_frames:].float()
    return result


def project_q(
    value: Tensor,
    *,
    history_frames: int = 2,
    view_width: int = 40,
) -> Tensor:
    """Return the exact future-only complement `Q=(I-S)`."""

    return future_only(
        value, history_frames=history_frames, view_width=view_width
    ) - project_p(value, history_frames=history_frames, view_width=view_width)


def split_pq(
    value: Tensor,
    *,
    history_frames: int = 2,
    view_width: int = 40,
) -> tuple[Tensor, Tensor]:
    """Split a native latent into exact future P and Q tensors."""

    p = project_p(value, history_frames=history_frames, view_width=view_width)
    q = future_only(
        value, history_frames=history_frames, view_width=view_width
    ) - p
    return p, q


def expand_sigma(sigma: Tensor, reference: Tensor) -> Tensor:
    if sigma.ndim != 1 or sigma.shape[0] != reference.shape[0]:
        raise ValueError("sigma must have shape [B]")
    return sigma.to(device=reference.device, dtype=torch.float32).reshape(
        -1, *([1] * (reference.ndim - 1))
    )


def make_mixed_clock_state(
    clean: Tensor,
    canonical_noise: Tensor,
    sigma_p: Tensor,
    sigma_q: Tensor,
    *,
    history_frames: int = 2,
    view_width: int = 40,
) -> MixedClockState:
    """Construct independently corrupted P/Q states from one Gaussian noise."""

    if clean.shape != canonical_noise.shape:
        raise ValueError("clean and canonical_noise must share native shape")
    clean_p, clean_q = split_pq(
        clean, history_frames=history_frames, view_width=view_width
    )
    noise_p, noise_q = split_pq(
        canonical_noise, history_frames=history_frames, view_width=view_width
    )
    expanded_p = expand_sigma(sigma_p, clean_p)
    expanded_q = expand_sigma(sigma_q, clean_q)
    noisy_p = (1.0 - expanded_p) * clean_p + expanded_p * noise_p
    noisy_q = (1.0 - expanded_q) * clean_q + expanded_q * noise_q
    native_noisy = noisy_p + noisy_q
    history = (
        (1.0 - expanded_p) * clean.float()
        + expanded_p * canonical_noise.float()
    )
    native_noisy[:, :, :history_frames] = history[:, :, :history_frames]
    return MixedClockState(
        clean_p=clean_p,
        clean_q=clean_q,
        noise_p=noise_p,
        noise_q=noise_q,
        noisy_p=noisy_p,
        noisy_q=noisy_q,
        native_noisy=native_noisy,
        target_p=noise_p - clean_p,
        target_q=noise_q - clean_q,
    )


def _expanded_mask(mask: Tensor, reference: Tensor) -> Tensor:
    if mask.ndim != 5 or reference.ndim != 5:
        raise ValueError("mask and reference must use [B,C,T,H,W] layout")
    try:
        return mask.to(device=reference.device, dtype=torch.float32).expand_as(
            reference
        )
    except RuntimeError as exc:
        raise ValueError("mask is not broadcastable to the native latent") from exc


def project_velocities_and_loss(
    native_velocity: Tensor,
    q_head_velocity: Tensor,
    target_p: Tensor,
    target_q: Tensor,
    mask: Tensor,
    *,
    history_frames: int = 2,
    view_width: int = 40,
) -> ProjectedPrediction:
    """Hard-project both heads and compute the rank-correct full-coordinate loss."""

    if not (
        native_velocity.shape
        == q_head_velocity.shape
        == target_p.shape
        == target_q.shape
    ):
        raise ValueError("predictions and targets must share native shape")
    velocity_p = project_p(
        native_velocity,
        history_frames=history_frames,
        view_width=view_width,
    )
    velocity_q = project_q(
        q_head_velocity,
        history_frames=history_frames,
        view_width=view_width,
    )
    weights = _expanded_mask(mask, velocity_p)
    denominator = weights.sum()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise InvertibleTwoClockError("flow loss received an empty mask")
    error_p = velocity_p - target_p.float()
    error_q = velocity_q - target_q.float()
    loss = ((error_p.square() + error_q.square()) * weights).sum() / denominator
    combined_error = (
        velocity_p + velocity_q - target_p.float() - target_q.float()
    )
    combined_loss = (combined_error.square() * weights).sum() / denominator
    relative_error = (loss - combined_loss).abs() / combined_loss.abs().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(relative_error)):
        raise InvertibleTwoClockError("projected flow loss is non-finite")
    return ProjectedPrediction(
        velocity_p=velocity_p,
        velocity_q=velocity_q,
        loss=loss,
        combined_loss=combined_loss,
        loss_identity_relative_error=relative_error,
    )


def band_sigma_schedule(base_sigmas: Tensor, mode: ScheduleMode) -> tuple[Tensor, Tensor]:
    """Return the frozen synchronous, P-leading, or Q-leading schedule."""

    if base_sigmas.ndim != 1 or base_sigmas.numel() < 2:
        raise ValueError("base_sigmas must contain at least two scalar nodes")
    base = base_sigmas.float()
    if not bool(torch.isfinite(base).all()) or bool((base < 0).any()) or bool(
        (base > 1).any()
    ):
        raise ValueError("base sigmas must be finite in [0,1]")
    if bool((base[1:] > base[:-1]).any()):
        raise ValueError("base sigmas must be monotonically nonincreasing")
    if float(base[0]) != 1.0 or float(base[-1]) != 0.0:
        raise ValueError("base schedule endpoints must be exactly one and zero")
    if mode == "synchronous":
        q = base.clone()
    elif mode == "p_leads":
        q = base.sqrt()
    elif mode == "q_leads":
        q = base.square()
    else:
        raise ValueError(f"unsupported IPQ schedule mode: {mode!r}")
    q[0] = 1.0
    q[-1] = 0.0
    return base, q


def euler_band_step(
    p_state: Tensor,
    q_state: Tensor,
    velocity_p: Tensor,
    velocity_q: Tensor,
    sigma_p: Tensor | float,
    next_sigma_p: Tensor | float,
    sigma_q: Tensor | float,
    next_sigma_q: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """Integrate complementary velocities with their own scalar step sizes."""

    if not (
        p_state.shape == q_state.shape == velocity_p.shape == velocity_q.shape
    ):
        raise ValueError("band states and velocities must share shape")
    delta_p = torch.as_tensor(
        next_sigma_p, device=p_state.device, dtype=torch.float32
    ) - torch.as_tensor(sigma_p, device=p_state.device, dtype=torch.float32)
    delta_q = torch.as_tensor(
        next_sigma_q, device=q_state.device, dtype=torch.float32
    ) - torch.as_tensor(sigma_q, device=q_state.device, dtype=torch.float32)
    return (
        p_state.float() + delta_p * velocity_p.float(),
        q_state.float() + delta_q * velocity_q.float(),
    )


def projector_receipt(
    value: Tensor,
    *,
    history_frames: int = 2,
    view_width: int = 40,
    epsilon: float = 1e-30,
) -> ProjectorReceipt:
    """Compute the fixed reconstruction/orthogonality audit scalars."""

    p, q = split_pq(
        value, history_frames=history_frames, view_width=view_width
    )
    target = future_only(
        value, history_frames=history_frames, view_width=view_width
    )
    reconstruction = p + q - target
    target_energy = target.square().sum().clamp_min(epsilon)
    inner = (p * q).sum().abs() / target_energy
    p_twice = project_p(p, history_frames=history_frames, view_width=view_width)
    q_twice = project_q(q, history_frames=history_frames, view_width=view_width)
    return ProjectorReceipt(
        reconstruction_max_abs=float(reconstruction.abs().max()),
        reconstruction_relative_energy=float(
            reconstruction.square().sum() / target_energy
        ),
        normalized_inner_product=float(inner),
        p_idempotence_max_abs=float((p_twice - p).abs().max()),
        q_idempotence_max_abs=float((q_twice - q).abs().max()),
        history_nonzero=int(
            torch.count_nonzero(p[:, :, :history_frames])
            + torch.count_nonzero(q[:, :, :history_frames])
        ),
        rank_fraction_p=0.25,
        rank_fraction_q=0.75,
    )


def assert_projector_receipt(receipt: ProjectorReceipt) -> None:
    """Fail closed under the prospective IPQ-TC1 transform tolerances."""

    failures = []
    if receipt.reconstruction_max_abs > 2e-6:
        failures.append("reconstruction_max_abs")
    if receipt.reconstruction_relative_energy > 1e-12:
        failures.append("reconstruction_relative_energy")
    if abs(receipt.normalized_inner_product) > 2e-6:
        failures.append("normalized_inner_product")
    if receipt.p_idempotence_max_abs > 2e-6:
        failures.append("p_idempotence_max_abs")
    if receipt.q_idempotence_max_abs > 2e-6:
        failures.append("q_idempotence_max_abs")
    if receipt.history_nonzero != 0:
        failures.append("history_nonzero")
    if receipt.rank_fraction_p != 0.25 or receipt.rank_fraction_q != 0.75:
        failures.append("rank_fraction")
    if failures:
        raise InvertibleTwoClockError(
            "IPQ projector receipt failed: " + ", ".join(failures)
        )
