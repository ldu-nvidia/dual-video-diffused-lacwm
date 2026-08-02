"""A compact, PyTorch-only raw-RGB Video Latent Forcing transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class VideoLatentForcingConfig:
    """Shape and model contract for the small video proof-of-concept."""

    video_channels: int = 3
    future_frames: int = 8
    history_frames: int = 5
    height: int = 64
    width: int = 112
    patch_size: tuple[int, int, int] = (1, 8, 8)
    aux_channels: int = 48
    aux_grid: tuple[int, int, int] | None = None
    action_steps: int = 16
    action_dim: int = 7
    hidden_size: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    parameter_matched_video_only: bool = False

    def __post_init__(self) -> None:
        positive = {
            "video_channels": self.video_channels,
            "future_frames": self.future_frames,
            "history_frames": self.history_frames,
            "height": self.height,
            "width": self.width,
            "aux_channels": self.aux_channels,
            "action_steps": self.action_steps,
            "action_dim": self.action_dim,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "num_heads": self.num_heads,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if len(self.patch_size) != 3 or any(size < 1 for size in self.patch_size):
            raise ValueError("patch_size must contain three positive integers")
        if (
            self.future_frames % self.patch_size[0]
            or self.height % self.patch_size[1]
            or self.width % self.patch_size[2]
        ):
            raise ValueError("future video dimensions must be divisible by patch_size")
        if self.aux_grid is not None and tuple(self.aux_grid) != self.patch_grid:
            raise ValueError(
                f"aux_grid must match video patch grid {self.patch_grid}, "
                f"got {tuple(self.aux_grid)}"
            )
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")

    @property
    def patch_grid(self) -> tuple[int, int, int]:
        return (
            self.future_frames // self.patch_size[0],
            self.height // self.patch_size[1],
            self.width // self.patch_size[2],
        )

    @property
    def resolved_aux_grid(self) -> tuple[int, int, int]:
        return self.patch_grid if self.aux_grid is None else tuple(self.aux_grid)

    @property
    def num_patches(self) -> int:
        temporal, height, width = self.patch_grid
        return temporal * height * width

    @property
    def future_shape(self) -> tuple[int, int, int, int]:
        return self.video_channels, self.future_frames, self.height, self.width

    @property
    def history_shape(self) -> tuple[int, int, int, int]:
        return self.video_channels, self.history_frames, self.height, self.width

    @property
    def auxiliary_shape(self) -> tuple[int, int, int, int]:
        return self.aux_channels, *self.resolved_aux_grid


@dataclass(frozen=True)
class VideoLatentForcingOutput:
    """Clean-state predictions from the video and auxiliary output heads."""

    video_x: Tensor
    auxiliary_x: Tensor


class _ScalarEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, value: Tensor) -> Tensor:
        half = self.hidden_size // 2
        if half == 0:
            frequencies = value.new_empty((0,))
        else:
            frequencies = torch.exp(
                -math.log(10_000)
                * torch.arange(half, device=value.device, dtype=torch.float32)
                / max(half, 1)
            ).to(dtype=value.dtype)
        angles = value[:, None] * frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if self.hidden_size % 2:
            embedding = torch.cat((embedding, embedding.new_zeros((value.shape[0], 1))), dim=-1)
        return self.mlp(embedding)


def _modulate(tokens: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return tokens * (1.0 + scale[:, None]) + shift[:, None]


class _AdaLNTransformerBlock(nn.Module):
    """Pre-norm transformer block driven by shared per-example modulation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        mlp_size = round(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, modulation: tuple[Tensor, ...]) -> Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = modulation
        attention_input = _modulate(
            self.attention_norm(tokens),
            attention_shift,
            attention_scale,
        )
        attention_output = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )[0]
        tokens = tokens + attention_gate[:, None] * attention_output
        mlp_input = _modulate(self.mlp_norm(tokens), mlp_shift, mlp_scale)
        return tokens + mlp_gate[:, None] * self.mlp(mlp_input)


class VideoLatentForcingModel(nn.Module):
    """Shared-grid dual-state transformer with symmetric token fusion.

    The two state projections are added with coefficient one; there is no gate
    or privileged branch.  Schedule and loss masking create the intended
    auxiliary-first asymmetry.
    """

    def __init__(self, config: VideoLatentForcingConfig | None = None) -> None:
        super().__init__()
        self.config = config or VideoLatentForcingConfig()
        cfg = self.config
        self.video_patch_projection = nn.Conv3d(
            cfg.video_channels,
            cfg.hidden_size,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.auxiliary_patch_projection = nn.Conv3d(
            cfg.aux_channels,
            cfg.hidden_size,
            kernel_size=1,
            stride=1,
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, cfg.num_patches, cfg.hidden_size)
        )

        history_patch = (
            cfg.history_frames,
            cfg.patch_size[1],
            cfg.patch_size[2],
        )
        self.history_projection = nn.Conv3d(
            cfg.video_channels,
            cfg.hidden_size,
            kernel_size=history_patch,
            stride=history_patch,
        )
        self.num_history_tokens = (
            cfg.height // cfg.patch_size[1]
        ) * (cfg.width // cfg.patch_size[2])
        self.history_position_embedding = nn.Parameter(
            torch.empty(1, self.num_history_tokens, cfg.hidden_size)
        )
        self.action_projection = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        self.action_position_embedding = nn.Parameter(
            torch.empty(1, cfg.action_steps, cfg.hidden_size)
        )
        self.video_time_embedding = _ScalarEmbedding(cfg.hidden_size)
        self.auxiliary_time_embedding = _ScalarEmbedding(cfg.hidden_size)
        self.condition_norm = nn.LayerNorm(cfg.hidden_size)

        self.clock_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, 6 * cfg.hidden_size),
        )
        self.transformer = nn.ModuleList(
            _AdaLNTransformerBlock(
                cfg.hidden_size,
                cfg.num_heads,
                cfg.mlp_ratio,
                cfg.dropout,
            )
            for _ in range(cfg.depth)
        )
        self.transformer_norm = nn.LayerNorm(cfg.hidden_size)
        self.output_norm = nn.LayerNorm(cfg.hidden_size)
        patch_volume = math.prod(cfg.patch_size)
        self.video_output_head = nn.Linear(
            cfg.hidden_size,
            cfg.video_channels * patch_volume,
        )
        self.auxiliary_output_head = nn.Linear(cfg.hidden_size, cfg.aux_channels)

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.history_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.action_position_embedding, std=0.02)

        # Match the released Latent Forcing/DiT initialization: every residual
        # branch starts as an exact no-op, and both clean-state prediction heads
        # start at zero.  Unlike the released image model, this compact model
        # shares one adaptive modulation across its transformer blocks, so
        # zeroing this final modulation projection is the corresponding
        # operation for all block shifts, scales, and gates.  Keep positional
        # and causal-context projections at their ordinary non-zero
        # initialization so training can immediately learn the output heads.
        nn.init.zeros_(self.clock_modulation[-1].weight)
        nn.init.zeros_(self.clock_modulation[-1].bias)
        nn.init.zeros_(self.video_output_head.weight)
        nn.init.zeros_(self.video_output_head.bias)
        nn.init.zeros_(self.auxiliary_output_head.weight)
        nn.init.zeros_(self.auxiliary_output_head.bias)

    def _validate_inputs(
        self,
        noisy_video: Tensor,
        noisy_auxiliary: Tensor | None,
        history: Tensor,
        actions: Tensor,
    ) -> None:
        cfg = self.config
        if noisy_video.ndim != 5 or tuple(noisy_video.shape[1:]) != cfg.future_shape:
            raise ValueError(
                f"noisy_video must have shape [B,{','.join(map(str, cfg.future_shape))}]"
            )
        batch_size = noisy_video.shape[0]
        if history.ndim != 5 or tuple(history.shape) != (batch_size, *cfg.history_shape):
            raise ValueError(
                f"history must have shape [B,{','.join(map(str, cfg.history_shape))}]"
            )
        if actions.ndim != 3 or tuple(actions.shape) != (
            batch_size,
            cfg.action_steps,
            cfg.action_dim,
        ):
            raise ValueError(
                f"actions must have shape [B,{cfg.action_steps},{cfg.action_dim}]"
            )
        if noisy_auxiliary is not None and (
            noisy_auxiliary.ndim != 5
            or tuple(noisy_auxiliary.shape) != (batch_size, *cfg.auxiliary_shape)
        ):
            raise ValueError(
                "noisy_auxiliary must have shape "
                f"[B,{','.join(map(str, cfg.auxiliary_shape))}]"
            )
        for name, value in (
            ("noisy_video", noisy_video),
            ("history", history),
            ("actions", actions),
        ):
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point")
        if noisy_auxiliary is not None and not noisy_auxiliary.is_floating_point():
            raise TypeError("noisy_auxiliary must be floating point")

    @staticmethod
    def _batch_clock(value: Tensor | float, reference: Tensor, name: str) -> Tensor:
        clock = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if clock.ndim == 0:
            clock = clock.expand(reference.shape[0])
        if clock.ndim != 1 or clock.shape[0] != reference.shape[0]:
            raise ValueError(f"{name} must be scalar or have shape [B]")
        if not bool(torch.isfinite(clock).all()) or bool(((clock < 0) | (clock > 1)).any()):
            raise ValueError(f"{name} must contain finite values in [0, 1]")
        return clock

    def _auxiliary_mask(
        self,
        value: Tensor | bool,
        reference: Tensor,
    ) -> Tensor:
        if isinstance(value, bool):
            mask = torch.full(
                (reference.shape[0],),
                value,
                device=reference.device,
                dtype=torch.bool,
            )
        elif isinstance(value, Tensor):
            if value.dtype != torch.bool:
                raise TypeError("auxiliary_fusion_mask tensor must have bool dtype")
            mask = value.to(device=reference.device)
            if mask.ndim == 0:
                mask = mask.expand(reference.shape[0])
            if mask.ndim != 1 or mask.shape[0] != reference.shape[0]:
                raise ValueError("auxiliary_fusion_mask must be scalar or have shape [B]")
        else:
            raise TypeError("auxiliary_fusion_mask must be a bool or bool tensor")
        if self.config.parameter_matched_video_only:
            return torch.zeros_like(mask)
        return mask

    def project_states(
        self,
        noisy_video: Tensor,
        noisy_auxiliary: Tensor,
        *,
        auxiliary_fusion_mask: Tensor | bool = True,
    ) -> Tensor:
        """Project and sum the two aligned state grids with unit coefficients."""
        video_tokens = self.video_patch_projection(noisy_video).flatten(2).transpose(1, 2)
        mask = self._auxiliary_mask(auxiliary_fusion_mask, noisy_video)
        if self.config.parameter_matched_video_only:
            return video_tokens
        # Keep the auxiliary projection in the graph for every dual/A1
        # microbatch.  A data-dependent Python branch can otherwise make DDP
        # skip its all-reduce when the final accumulated microbatch happens to
        # contain no auxiliary examples.  Multiplication by an exact Boolean
        # zero preserves the per-example no-op contract for finite inputs.
        auxiliary_tokens = (
            self.auxiliary_patch_projection(noisy_auxiliary)
            .flatten(2)
            .transpose(1, 2)
        )
        return video_tokens + auxiliary_tokens * mask[:, None, None].to(
            dtype=auxiliary_tokens.dtype
        )

    def _unpatchify_video(self, patches: Tensor) -> Tensor:
        cfg = self.config
        grid_t, grid_h, grid_w = cfg.patch_grid
        patch_t, patch_h, patch_w = cfg.patch_size
        patches = patches.reshape(
            patches.shape[0],
            grid_t,
            grid_h,
            grid_w,
            cfg.video_channels,
            patch_t,
            patch_h,
            patch_w,
        )
        patches = patches.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return patches.reshape(patches.shape[0], *cfg.future_shape)

    def _unpatchify_auxiliary(self, patches: Tensor) -> Tensor:
        grid_t, grid_h, grid_w = self.config.resolved_aux_grid
        return (
            patches.reshape(
                patches.shape[0],
                grid_t,
                grid_h,
                grid_w,
                self.config.aux_channels,
            )
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

    def forward(
        self,
        noisy_video: Tensor,
        noisy_auxiliary: Tensor | None,
        t_video: Tensor | float,
        t_auxiliary: Tensor | float,
        history: Tensor,
        actions: Tensor,
        *,
        condition_on_auxiliary: Tensor | bool = True,
        auxiliary_fusion_mask: Tensor | bool | None = None,
        predict_video: bool = True,
    ) -> VideoLatentForcingOutput:
        """Predict clean video and auxiliary states from their noisy states."""
        if not isinstance(predict_video, bool):
            raise TypeError("predict_video must be a bool")
        self._validate_inputs(noisy_video, noisy_auxiliary, history, actions)
        cfg = self.config
        if auxiliary_fusion_mask is None:
            auxiliary_fusion_mask = condition_on_auxiliary
        elif not (isinstance(condition_on_auxiliary, bool) and condition_on_auxiliary):
            raise ValueError(
                "pass either condition_on_auxiliary or auxiliary_fusion_mask, not both"
            )
        fusion_mask = self._auxiliary_mask(auxiliary_fusion_mask, noisy_video)
        if noisy_auxiliary is None:
            if bool(fusion_mask.any()):
                raise ValueError("noisy_auxiliary is required when its condition is enabled")
            noisy_auxiliary = noisy_video.new_zeros(
                noisy_video.shape[0], *cfg.auxiliary_shape
            )
        t_video = self._batch_clock(t_video, noisy_video, "t_video")
        t_auxiliary = self._batch_clock(t_auxiliary, noisy_video, "t_auxiliary")

        tokens = self.project_states(
            noisy_video,
            noisy_auxiliary,
            auxiliary_fusion_mask=fusion_mask,
        )
        tokens = tokens + self.position_embedding.to(dtype=tokens.dtype)

        history_tokens = self.history_projection(history).flatten(2).transpose(1, 2)
        history_tokens = history_tokens + self.history_position_embedding.to(
            dtype=history_tokens.dtype
        )
        action_tokens = self.action_projection(actions)
        action_tokens = action_tokens + self.action_position_embedding.to(
            dtype=action_tokens.dtype
        )
        clock_condition = self.video_time_embedding(t_video)
        # Released Latent Forcing embeds both state clocks even when an input
        # projection is ablated.  Keep that clock/token distinction for A1 and
        # the L1 off control.  Only the strict parameter-matched B0 baseline
        # removes every auxiliary dependency, including its arbitrary clock.
        auxiliary_clock_mask = torch.full_like(
            fusion_mask,
            not cfg.parameter_matched_video_only,
        )
        clock_condition = clock_condition + self.auxiliary_time_embedding(
            t_auxiliary
        ) * auxiliary_clock_mask[:, None].to(dtype=clock_condition.dtype)
        clock_token = self.condition_norm(clock_condition)[:, None]
        context = torch.cat((clock_token, history_tokens, action_tokens), dim=1)
        hidden = torch.cat((context, tokens), dim=1)
        modulation = self.clock_modulation(clock_condition).chunk(6, dim=-1)
        for block in self.transformer:
            hidden = block(hidden, modulation)
        hidden = self.transformer_norm(hidden)
        hidden = self.output_norm(hidden[:, context.shape[1] :])
        # A semantic-only DDP pass must not expose the video head's autograd
        # graph as a forward output.  ``find_unused_parameters`` starts its
        # traversal from every returned tensor; returning an ignored
        # ``video_x`` therefore makes DDP classify the head as used even though
        # its reduction hooks can never fire.  An explicit branch keeps those
        # parameters genuinely unused (and out of the optimizer update) while
        # preserving the architecture, state-dict schema, and auxiliary path.
        video_x = (
            self._unpatchify_video(self.video_output_head(hidden))
            if predict_video
            else torch.zeros_like(noisy_video)
        )
        if cfg.parameter_matched_video_only:
            auxiliary_x = torch.zeros_like(noisy_auxiliary)
        else:
            auxiliary_x = self._unpatchify_auxiliary(
                self.auxiliary_output_head(hidden)
            )
        return VideoLatentForcingOutput(
            video_x=video_x,
            auxiliary_x=auxiliary_x,
        )


# Short alias for interactive experiments while retaining the explicit public name.
VideoLatentForcing = VideoLatentForcingModel
