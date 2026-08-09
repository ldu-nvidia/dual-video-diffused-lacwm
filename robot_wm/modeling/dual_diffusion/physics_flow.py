"""Exact tensor contract for fixed causal robot-flow conditioning.

The representation is deliberately small and analytic.  Eight RGB-frame
transitions are packed four-at-a-time into the two future Wan temporal tokens.
Only the preregistered top-camera third of the width-stacked latent grid is
populated; history tokens and the two wrist-view thirds are exact zero.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

import torch
from torch import Tensor


FLOW_COMPONENTS: Final[tuple[str, ...]] = (
    "dx_over_width",
    "dy_over_height",
    "visibility",
    "log_depth_ratio",
)
FUTURE_TRANSITIONS: Final[int] = 8
TRANSITIONS_PER_LATENT: Final[int] = 4
FLOW_CHANNELS: Final[int] = len(FLOW_COMPONENTS) * TRANSITIONS_PER_LATENT
LATENT_FRAMES: Final[int] = 4
HISTORY_LATENT_FRAMES: Final[int] = 2
LATENT_HEIGHT: Final[int] = 24
VIEW_LATENT_WIDTH: Final[int] = 40
LATENT_WIDTH: Final[int] = 120
TOP_VIEW_SLICE: Final[slice] = slice(0, VIEW_LATENT_WIDTH)


class PhysicsFlowError(RuntimeError):
    """The preregistered flow geometry or causal tensor contract changed."""


def _validate_transitions(transitions: Tensor) -> None:
    expected = (
        FUTURE_TRANSITIONS,
        len(FLOW_COMPONENTS),
        LATENT_HEIGHT,
        VIEW_LATENT_WIDTH,
    )
    if transitions.ndim != 5 or tuple(transitions.shape[1:]) != expected:
        raise PhysicsFlowError(
            "transition flow must be [B,8,4,24,40], got "
            f"{tuple(transitions.shape)}"
        )
    if not transitions.is_floating_point() or not bool(torch.isfinite(transitions).all()):
        raise PhysicsFlowError("transition flow must be finite floating point")
    visibility = transitions[:, :, 2]
    if bool((visibility < 0).any()) or bool((visibility > 1).any()):
        raise PhysicsFlowError("pooled visibility must lie in [0,1]")
    if bool((transitions[:, :, :2].abs() > 1).any()):
        raise PhysicsFlowError("visible normalized displacement must lie in [-1,1]")
    unsupported = visibility == 0
    for component in (0, 1, 3):
        if bool(transitions[:, :, component].masked_select(unsupported).ne(0).any()):
            raise PhysicsFlowError("unsupported flow components must be exact zero")


def pack_top_view_flow(transitions: Tensor) -> Tensor:
    """Pack `[B,8,4,24,40]` transitions into `[B,16,4,24,120]`.

    Channel order within each future latent is transition-major, then
    `(dx,dy,visibility,log-depth)`.  The operation is exactly invertible on the
    populated top-view support.
    """

    _validate_transitions(transitions)
    batch = int(transitions.shape[0])
    result = transitions.new_zeros(
        batch, FLOW_CHANNELS, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH
    )
    for latent_offset in range(2):
        start = latent_offset * TRANSITIONS_PER_LATENT
        stop = start + TRANSITIONS_PER_LATENT
        packed = transitions[:, start:stop].reshape(
            batch, FLOW_CHANNELS, LATENT_HEIGHT, VIEW_LATENT_WIDTH
        )
        result[:, :, HISTORY_LATENT_FRAMES + latent_offset, :, TOP_VIEW_SLICE] = packed
    validate_packed_flow(result)
    return result


def unpack_top_view_flow(packed: Tensor) -> Tensor:
    """Recover the eight top-view transitions and reject off-contract values."""

    validate_packed_flow(packed)
    batch = int(packed.shape[0])
    parts = []
    for latent_offset in range(2):
        value = packed[
            :, :, HISTORY_LATENT_FRAMES + latent_offset, :, TOP_VIEW_SLICE
        ].reshape(
            batch,
            TRANSITIONS_PER_LATENT,
            len(FLOW_COMPONENTS),
            LATENT_HEIGHT,
            VIEW_LATENT_WIDTH,
        )
        parts.append(value)
    transitions = torch.cat(parts, dim=1)
    _validate_transitions(transitions)
    return transitions


def timeshift_plus_one(transitions: Tensor) -> Tensor:
    """Use the next transition at each slot without cyclic wrap.

    The final slot is exact zero.  This deterministic intervention is fixed
    before any generator outcome is inspected.
    """

    _validate_transitions(transitions)
    shifted = torch.zeros_like(transitions)
    shifted[:, :-1] = transitions[:, 1:]
    return shifted


def validate_packed_flow(value: Tensor) -> None:
    expected = (FLOW_CHANNELS, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    if value.ndim != 5 or tuple(value.shape[1:]) != expected:
        raise PhysicsFlowError(
            "physics flow must be [B,16,4,24,120], got " f"{tuple(value.shape)}"
        )
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise PhysicsFlowError("physics flow must be finite floating point")
    if bool(value[:, :, :HISTORY_LATENT_FRAMES].ne(0).any()):
        raise PhysicsFlowError("observed-history flow slots must be exact zero")
    if bool(value[..., VIEW_LATENT_WIDTH:].ne(0).any()):
        raise PhysicsFlowError("unvalidated wrist-view flow columns must be exact zero")
    for latent in range(HISTORY_LATENT_FRAMES, LATENT_FRAMES):
        for substep in range(TRANSITIONS_PER_LATENT):
            channel = substep * len(FLOW_COMPONENTS) + 2
            visibility = value[:, channel, latent, :, TOP_VIEW_SLICE]
            if bool((visibility < 0).any()) or bool((visibility > 1).any()):
                raise PhysicsFlowError("packed visibility must lie in [0,1]")
            unsupported = visibility == 0
            for offset in (0, 1, 3):
                component = value[
                    :, substep * len(FLOW_COMPONENTS) + offset, latent, :, TOP_VIEW_SLICE
                ]
                if offset in (0, 1) and bool((component.abs() > 1).any()):
                    raise PhysicsFlowError(
                        "packed normalized displacement must lie in [-1,1]"
                    )
                if bool(component.masked_select(unsupported).ne(0).any()):
                    raise PhysicsFlowError(
                        "packed unsupported flow components must be exact zero"
                    )


def tensor_sha256(value: Tensor) -> str:
    """Hash dtype, shape, and contiguous bytes for immutable evidence."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(tensor.view(torch.uint8).numpy()))
    return digest.hexdigest()
