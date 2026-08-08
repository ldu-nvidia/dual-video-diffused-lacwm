"""Per-substep requested-action tokens for Wan's native cross-attention seam.

The historical VPM reduces each five-sample robot-action chunk to one latent,
then reduces four adjacent transition latents to one spatially broadcast Wan
control plane.  :class:`ActionTokenContextAdapter` preserves all 8 x 5 future
action samples as distinct context tokens.  Tokens are added to the existing
null-text sequence before Wan's frozen text embedding, so pretrained tensor
shapes and every Wan block API remain unchanged.

Only requested actions, morphology, and fixed token positions are consumed.
The path is therefore available identically during training and deployment.
This adaptive follow-up fixes the candidate's effective scalar gate at 0.1 so
the projection cannot be starved by the exact-zero cold start used in the
prospective screen; the paired control instantiates the same frozen scalar but
hard-masks its context effect.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

ACTION_TOKEN_STATS_SCHEMA = "lacwm-action-token-whitening-v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_stats(
    path: str,
    *,
    expected_sha256: str,
    action_dim: int,
    chunk_size: int,
    num_transitions: int,
) -> Mapping[str, Any]:
    stats_path = Path(path).expanduser()
    if (
        not stats_path.is_absolute()
        or not stats_path.is_file()
        or stats_path.is_symlink()
    ):
        raise ValueError("action-token statistics must be a regular absolute file")
    if _sha256(stats_path) != expected_sha256:
        raise ValueError("action-token statistics SHA-256 differs")
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("action-token statistics are invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("action-token statistics must contain one object")
    identity = payload.get("identity_sha256")
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    if (
        payload.get("schema") != ACTION_TOKEN_STATS_SCHEMA
        or payload.get("split") != "train"
        or payload.get("protected_test_accessed") is not False
        or payload.get("future_action_chunks") != [4, 12]
        or int(payload.get("chunk_size", -1)) != chunk_size
        or int(payload.get("num_transitions", -1)) != num_transitions
        or int(payload.get("token_count", -1)) != chunk_size * num_transitions
        or int(payload.get("padding_dim", -1)) != action_dim
        or not isinstance(identity, str)
        or hashlib.sha256(_canonical_json(unsigned)).hexdigest() != identity
    ):
        raise ValueError("action-token statistics identity/contract differs")
    for key in ("mean", "std", "active"):
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != action_dim:
            raise ValueError(f"action-token statistics {key} has the wrong shape")
    return payload


class ActionTokenContextAdapter(nn.Module):
    """Map each future action sample to one raw Wan context residual token.

    Input shape is ``[B, 8, 5, 157]``.  Output shape is ``[B, 40, 4096]``;
    transition-major flattening preserves the exact within-chunk ordering.
    The output dimension matches the raw UMT5 context dimension and is passed
    through Wan's native ``text_embedding`` and every block's cross-attention.
    """

    def __init__(
        self,
        *,
        stats_path: str,
        expected_stats_sha256: str,
        enabled: bool,
        action_dim: int = 157,
        chunk_size: int = 5,
        num_transitions: int = 8,
        morph_dim: int = 64,
        hidden: int = 256,
        text_dim: int = 4096,
        clip_value: float = 8.0,
        initialization_seed: int = 20_260_808,
        gate_init: float = 0.1,
        gate_trainable: bool = False,
    ) -> None:
        super().__init__()
        if (action_dim, chunk_size, num_transitions, morph_dim) != (157, 5, 8, 64):
            raise ValueError("prospective screen fixes action geometry to [8,5,157]")
        if hidden != 256 or text_dim != 4096:
            raise ValueError("prospective screen fixes hidden/text dimensions")
        if not math.isfinite(clip_value) or clip_value != 8.0:
            raise ValueError("prospective screen fixes whitening clip value to eight")
        if initialization_seed != 20_260_808:
            raise ValueError("prospective screen fixes adapter initialization seed")
        if not math.isfinite(gate_init) or gate_init != 0.1:
            raise ValueError("fixed-dose follow-up fixes effective gate to 0.1")
        if gate_trainable is not False:
            raise ValueError("fixed-dose follow-up freezes the scalar gate")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")

        payload = _read_stats(
            stats_path,
            expected_sha256=expected_stats_sha256,
            action_dim=action_dim,
            chunk_size=chunk_size,
            num_transitions=num_transitions,
        )
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32)
        active = torch.tensor(payload["active"], dtype=torch.bool)
        if (
            not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(std).all())
            or bool((std <= 0).any())
            or int(active.sum()) != int(payload.get("active_dimensions", -1))
            or bool((mean[~active] != 0).any())
            or bool((std[~active] != 1).any())
        ):
            raise ValueError("action-token statistics contain invalid tensors")
        self.register_buffer("action_mean", mean, persistent=True)
        self.register_buffer("action_std", std, persistent=True)
        self.register_buffer("action_active", active, persistent=True)

        self.raw_gate = nn.Parameter(
            torch.tensor(math.atanh(gate_init), dtype=torch.float32),
            requires_grad=gate_trainable,
        )
        self.enabled = enabled
        self._runtime_hard_mask = False
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_transitions = num_transitions
        self.token_count = chunk_size * num_transitions
        self.morph_dim = morph_dim
        self.hidden = hidden
        self.text_dim = text_dim
        self.clip_value = clip_value
        self.gate_init = gate_init
        self.gate_trainable = gate_trainable
        self.stats_identity_sha256 = str(payload["identity_sha256"])
        self.stats_file_sha256 = expected_stats_sha256

        self.input_projection = nn.Linear(action_dim + morph_dim, hidden)
        self.transition_embedding = nn.Parameter(
            torch.empty(num_transitions, hidden, dtype=torch.float32)
        )
        self.substep_embedding = nn.Parameter(
            torch.empty(chunk_size, hidden, dtype=torch.float32)
        )
        self.output_projection = nn.Linear(hidden, text_dim)

        # Local RNG isolation guarantees byte-identical new parameters in both
        # arms without perturbing parent initialization or the training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            nn.init.trunc_normal_(self.input_projection.weight, std=0.02)
            nn.init.zeros_(self.input_projection.bias)
            nn.init.trunc_normal_(self.transition_embedding, std=0.02)
            nn.init.trunc_normal_(self.substep_embedding, std=0.02)
            nn.init.trunc_normal_(self.output_projection.weight, std=0.02)
            nn.init.zeros_(self.output_projection.bias)

    def effective_gate(self) -> Tensor:
        """Return the bounded trainable gate or an exact hard zero."""

        gate = torch.tanh(self.raw_gate)
        return gate if self.enabled and not self._runtime_hard_mask else gate * 0.0

    @contextmanager
    def runtime_hard_mask(self):
        """Temporarily disable only this trained context route."""

        previous = self._runtime_hard_mask
        self._runtime_hard_mask = True
        try:
            yield
        finally:
            self._runtime_hard_mask = previous

    def standardized_actions(self, actions: Tensor) -> Tensor:
        if (
            actions.ndim != 4
            or tuple(actions.shape[1:])
            != (self.num_transitions, self.chunk_size, self.action_dim)
            or not actions.is_floating_point()
        ):
            raise ValueError("actions must have shape [B,8,5,157] and float dtype")
        active = self.action_active.view(1, 1, 1, -1)
        whitened = (
            actions.float() - self.action_mean.view(1, 1, 1, -1)
        ) / self.action_std.view(1, 1, 1, -1)
        whitened = torch.where(active, whitened, torch.zeros_like(whitened))
        whitened = whitened.clamp(-self.clip_value, self.clip_value)
        if not bool(torch.isfinite(whitened).all()):
            raise FloatingPointError("standardized action tokens are non-finite")
        return whitened.to(dtype=actions.dtype)

    def forward(self, actions: Tensor, morph_emb: Tensor) -> tuple[Tensor, Tensor]:
        standardized = self.standardized_actions(actions)
        if morph_emb.ndim != 2 or tuple(morph_emb.shape) != (
            actions.shape[0],
            self.morph_dim,
        ):
            raise ValueError("morphology embedding must have shape [B,64]")
        morphology = morph_emb[:, None, None, :].expand(
            -1, self.num_transitions, self.chunk_size, -1
        )
        hidden = self.input_projection(
            torch.cat((standardized, morphology.to(standardized.dtype)), dim=-1)
        )
        hidden = hidden + self.transition_embedding[None, :, None, :].to(hidden.dtype)
        hidden = hidden + self.substep_embedding[None, None, :, :].to(hidden.dtype)
        tokens = self.output_projection(torch.nn.functional.silu(hidden))
        tokens = tokens.flatten(1, 2)
        expected = (actions.shape[0], self.token_count, self.text_dim)
        if tuple(tokens.shape) != expected:
            raise RuntimeError(f"action context token shape changed: {tuple(tokens.shape)}")
        return tokens, standardized
