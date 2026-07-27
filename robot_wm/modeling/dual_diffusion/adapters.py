"""Checkpoint-safe TF adapters and output heads for the Wan token grid."""

from __future__ import annotations

from math import atanh, isfinite, log, prod
from typing import Tuple

import torch
from torch import Tensor, nn


class ZeroInitTFTokenAdapter(nn.Module):
    """Project a TF state to Wan tokens behind a zero-initialized residual gate.

    The existing Wan 48-channel patch projection remains untouched.  At
    initialization this module contributes exactly zero, preserving the
    pretrained LACWM function and enabling a strict no-op regression test.
    """

    def __init__(
        self,
        tf_channels: int,
        hidden_size: int = 1536,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        gate_init: float = 0.0,
        gate_trainable: bool = True,
    ) -> None:
        super().__init__()
        if tf_channels < 1 or hidden_size < 1:
            raise ValueError("tf_channels and hidden_size must be positive")
        if len(patch_size) != 3 or any(p < 1 for p in patch_size):
            raise ValueError("patch_size must contain three positive values")
        if not isfinite(gate_init) or not -1.0 < gate_init < 1.0:
            raise ValueError("gate_init must be finite and strictly between -1 and 1")
        self.tf_channels = tf_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.projection = nn.Conv3d(
            tf_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(hidden_size)
        # Keep the parameter name and scalar shape checkpoint-compatible with
        # the original zero-gated adapter.  ``gate_init`` is the effective
        # residual multiplier, so invert tanh before storing the raw parameter.
        self.gate = nn.Parameter(
            torch.tensor(atanh(gate_init), dtype=torch.float32),
            requires_grad=gate_trainable,
        )

    def project_tokens(
        self, time_frequency: Tensor
    ) -> tuple[Tensor, tuple[int, int, int]]:
        """Return normalized TF tokens before applying the residual gate."""
        if time_frequency.ndim != 5:
            raise ValueError("TF state must have shape [B,C,F,H,W]")
        if time_frequency.shape[1] != self.tf_channels:
            raise ValueError(
                f"expected {self.tf_channels} TF channels, got {time_frequency.shape[1]}"
            )
        projected = self.projection(time_frequency)
        grid = tuple(projected.shape[2:])
        tokens = projected.flatten(2).transpose(1, 2)
        return self.norm(tokens), grid

    def residual_tokens(self, tokens: Tensor) -> Tensor:
        """Apply the checkpoint-compatible scalar gate to projected tokens."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError("tokens must have shape [B,N,hidden_size]")
        gate = self.effective_gate().to(device=tokens.device, dtype=tokens.dtype)
        return gate * tokens

    def effective_gate(self) -> Tensor:
        """Return the bounded residual multiplier used by the adapter."""
        return torch.tanh(self.gate)

    def forward(self, time_frequency: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        tokens, grid = self.project_tokens(time_frequency)
        return self.residual_tokens(tokens), grid


class TFVelocityHead(nn.Module):
    """Decode shared Wan tokens into a TF rectified-flow velocity field."""

    def __init__(
        self,
        hidden_size: int,
        tf_channels: int,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
    ) -> None:
        super().__init__()
        if hidden_size < 1 or tf_channels < 1:
            raise ValueError("hidden_size and tf_channels must be positive")
        self.hidden_size = hidden_size
        self.tf_channels = tf_channels
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, prod(patch_size) * tf_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, tokens: Tensor, grid: tuple[int, int, int]) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError("tokens must have shape [B,N,hidden_size]")
        ft, ht, wt = grid
        if tokens.shape[1] != ft * ht * wt:
            raise ValueError("token count does not match the provided 3D grid")
        pt, ph, pw = self.patch_size
        b = tokens.shape[0]
        values = self.linear(self.norm(tokens)).reshape(
            b, ft, ht, wt, pt, ph, pw, self.tf_channels
        )
        # [B,F,H,W,pF,pH,pW,C] -> [B,C,F*pF,H*pH,W*pW]
        return values.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            b, self.tf_channels, ft * pt, ht * ph, wt * pw
        )


class TFSigmaTokenEmbedding(nn.Module):
    """Embed the TF noise clock as a zero-initialized Wan-token residual.

    A noisy TF tensor is not identifiable without its own noise level when the
    video and TF clocks differ.  This module gives the shared Wan trunk that
    clock while retaining the exact pretrained function at initialization.
    """

    def __init__(
        self,
        hidden_size: int,
        embedding_dim: int = 128,
        max_period: float = 10_000.0,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if embedding_dim < 2 or embedding_dim % 2:
            raise ValueError("embedding_dim must be a positive even integer")
        if max_period <= 1:
            raise ValueError("max_period must exceed one")
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        frequencies = torch.exp(
            -log(max_period)
            * torch.arange(embedding_dim // 2, dtype=torch.float32)
            / max(embedding_dim // 2 - 1, 1)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def effective_gate(self) -> Tensor:
        """Return the bounded residual multiplier used by the clock embedding."""
        return torch.tanh(self.gate)

    def forward(self, sigma: Tensor) -> Tensor:
        if sigma.ndim > 1:
            sigma = sigma.reshape(sigma.shape[0], -1)
            if sigma.shape[1] != 1:
                raise ValueError("TF sigma must be scalar per batch element")
            sigma = sigma[:, 0]
        if sigma.ndim != 1:
            raise ValueError("TF sigma must have shape [B]")
        angles = sigma.float().unsqueeze(-1) * self.frequencies.unsqueeze(0)
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        tokens = self.net(embedding).to(dtype=sigma.dtype)
        return self.effective_gate().to(dtype=tokens.dtype) * tokens
