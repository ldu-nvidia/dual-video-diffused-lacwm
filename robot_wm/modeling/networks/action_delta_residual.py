"""Train-statistics-fitted action-velocity residual for explicit-action VPMs.

The residual uses only requested robot actions, so it is available unchanged at
training and deployment.  Its scalar gate is initialized to exact zero: adding
this module to a historical VPM therefore preserves the parent's initial
function while exposing a higher-rank, per-clip action coordinate to training.
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

ACTION_DELTA_STATS_SCHEMA = "lacwm-action-delta-whitening-v1"


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
    delta_steps: int,
) -> Mapping[str, Any]:
    stats_path = Path(path).expanduser()
    if (
        not stats_path.is_absolute()
        or not stats_path.is_file()
        or stats_path.is_symlink()
    ):
        raise ValueError("action-delta statistics must be a regular absolute file")
    if _sha256(stats_path) != expected_sha256:
        raise ValueError("action-delta statistics SHA-256 differs")
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("action-delta statistics are invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("action-delta statistics must contain one object")
    identity = payload.get("identity_sha256")
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    if (
        payload.get("schema") != ACTION_DELTA_STATS_SCHEMA
        or payload.get("split") != "train"
        or payload.get("protected_test_accessed") is not False
        or payload.get("future_action_chunks") != [4, 12]
        or int(payload.get("chunk_size", -1)) != delta_steps + 1
        or int(payload.get("delta_steps", -1)) != delta_steps
        or int(payload.get("padding_dim", -1)) != action_dim
        or not isinstance(identity, str)
        or hashlib.sha256(_canonical_json(unsigned)).hexdigest() != identity
    ):
        raise ValueError("action-delta statistics identity/contract differs")
    for key in ("mean", "std", "active"):
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != action_dim:
            raise ValueError(f"action-delta statistics {key} has the wrong shape")
    return payload


class WhitenedActionDeltaResidual(nn.Module):
    """Encode standardized within-chunk action differences into a latent residual.

    For each future action chunk ``a[..., 0:5, :]`` the encoder consumes
    ``a[..., 1:, :] - a[..., :-1, :]``.  Mean/std are fitted once from the
    registered training split.  Constant and padded coordinates are hard-zeroed
    by the immutable active mask.
    """

    def __init__(
        self,
        *,
        stats_path: str,
        expected_stats_sha256: str,
        enabled: bool,
        action_dim: int = 157,
        chunk_size: int = 5,
        latent_dim: int = 64,
        morph_dim: int = 64,
        hidden: int = 256,
        num_layers: int = 3,
        clip_value: float = 8.0,
        initialization_seed: int = 20_260_808,
    ) -> None:
        super().__init__()
        if chunk_size != 5:
            raise ValueError("prospective screen fixes action chunk size to five")
        if action_dim != 157 or latent_dim != 64 or morph_dim != 64:
            raise ValueError("prospective screen fixes VPM action/latent dimensions")
        if hidden != 256 or num_layers != 3:
            raise ValueError("prospective screen fixes the residual adapter topology")
        if not math.isfinite(clip_value) or clip_value != 8.0:
            raise ValueError("prospective screen fixes whitening clip value to eight")
        if initialization_seed != 20_260_808:
            raise ValueError("prospective screen fixes the adapter initialization seed")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")

        payload = _read_stats(
            stats_path,
            expected_sha256=expected_stats_sha256,
            action_dim=action_dim,
            delta_steps=chunk_size - 1,
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
            raise ValueError("action-delta statistics contain invalid tensors")
        self.register_buffer("delta_mean", mean, persistent=True)
        self.register_buffer("delta_std", std, persistent=True)
        self.register_buffer("delta_active", active, persistent=True)
        self.raw_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.enabled = enabled
        # Evaluation may temporarily mask the *trained* candidate residual to
        # measure its direct causal contribution.  This is deliberately runtime
        # state rather than a buffer/parameter so both arms retain an identical
        # serialized schema.
        self._runtime_hard_mask = False
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.clip_value = clip_value
        self.stats_identity_sha256 = str(payload["identity_sha256"])
        self.stats_file_sha256 = expected_stats_sha256

        in_dim = (chunk_size - 1) * action_dim + morph_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(max(num_layers - 2, 0)):
            layers.extend((nn.Linear(hidden, hidden), nn.SiLU()))
        layers.append(nn.Linear(hidden, latent_dim))
        self.net = nn.Sequential(*layers)
        # A local fork makes the two arms' new parameters byte-identical without
        # perturbing the parent model's global initialization/RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)

    def effective_gate(self) -> Tensor:
        """Bounded gate; control computes the branch but hard-masks its effect."""

        gate = torch.tanh(self.raw_gate)
        return gate if self.enabled and not self._runtime_hard_mask else gate * 0.0

    @contextmanager
    def runtime_hard_mask(self):
        """Temporarily force an exact-zero gate without mutating model state."""

        previous = self._runtime_hard_mask
        self._runtime_hard_mask = True
        try:
            yield
        finally:
            self._runtime_hard_mask = previous

    def standardized_deltas(self, actions: Tensor) -> Tensor:
        if (
            actions.ndim != 4
            or actions.shape[-2] != self.chunk_size
            or actions.shape[-1] != self.action_dim
            or not actions.is_floating_point()
        ):
            raise ValueError("actions must have shape [B,T,5,157] and float dtype")
        delta = actions[..., 1:, :].float() - actions[..., :-1, :].float()
        active = self.delta_active.view(1, 1, 1, -1)
        whitened = (delta - self.delta_mean.view(1, 1, 1, -1)) / self.delta_std.view(
            1, 1, 1, -1
        )
        whitened = torch.where(active, whitened, torch.zeros_like(whitened))
        whitened = whitened.clamp(-self.clip_value, self.clip_value)
        if not bool(torch.isfinite(whitened).all()):
            raise FloatingPointError("standardized action deltas are non-finite")
        return whitened.to(dtype=actions.dtype)

    def forward(self, actions: Tensor, morph_emb: Tensor) -> tuple[Tensor, Tensor]:
        whitened = self.standardized_deltas(actions)
        if morph_emb.ndim != 2 or morph_emb.shape != (
            actions.shape[0],
            64,
        ):
            raise ValueError("morphology embedding must have shape [B,64]")
        flat = whitened.flatten(start_dim=2)
        morphology = morph_emb.unsqueeze(1).expand(-1, flat.shape[1], -1)
        residual = self.net(torch.cat((flat, morphology.to(flat.dtype)), dim=-1))
        residual = residual.unsqueeze(2)
        if residual.shape != (actions.shape[0], actions.shape[1], 1, self.latent_dim):
            raise RuntimeError("action-delta residual shape changed")
        return residual, whitened
