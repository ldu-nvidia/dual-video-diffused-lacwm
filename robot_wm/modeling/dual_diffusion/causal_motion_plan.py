"""Causal action-conditioned low-frequency motion-plan diffusion.

The deployment boundary is deliberately narrow: the planner accepts only the
Wan latent of the observed five-frame history, requested actions, morphology,
and explicit Gaussian noise.  Clean future video is used solely by
``motion_plan_target`` while training the planner and is not accepted by the
planner or its rollout method.

Tensor convention throughout this module is ``[B,C,T,H,W]``.  The frozen
screen geometry is two observed and two future Wan tokens on a ``24 x 120``
grid containing three width-stacked views.  The motion plan is sixteen times
smaller spatially: ``[B,16,2,6,30]``.
"""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


PLAN_CHANNELS = 16
HISTORY_TOKENS = 2
FUTURE_TOKENS = 2
LATENT_HEIGHT = 24
LATENT_WIDTH = 120
NUM_VIEWS = 3
VIEW_WIDTH = LATENT_WIDTH // NUM_VIEWS
PLAN_HEIGHT = 6
PLAN_VIEW_WIDTH = 10
PLAN_WIDTH = PLAN_VIEW_WIDTH * NUM_VIEWS
PLAN_STEPS = 2
PLAN_SIGMAS = (1.0, 0.5, 0.0)
PLANNER_SPLIT_ROLES = ("planner_fit", "planner_calibration", "all")
NORMALIZATION_SCHEMA_VERSION = 1
NORMALIZATION_KIND = "camp_motion_plan_normalization"
FROZEN_TRAIN_MANIFEST_SHA256 = (
    "eeace7b0c9f5b6598f32e802d6d3678b6deccccf5b62c6cea2bb3d5478ab8b74"
)
FROZEN_TRAIN_METADATA_SHA256 = (
    "fa22a213f352ffb8cc0b4dc0d35138b35aac349c03f362c597c621fa3473da43"
)
FROZEN_TRAIN_RGB_SHA256 = (
    "b5bdde4461c75bc88653c38b737021fcbd69b0b22f4c87bc8e8097c3494b64ee"
)


class CausalMotionPlanError(RuntimeError):
    """A frozen geometry or causal-input contract was violated."""


def planner_partition_indexes(count: int, split_role: str) -> tuple[int, ...]:
    """Return the prospective train-only 448/64 planner partition."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if split_role not in PLANNER_SPLIT_ROLES:
        raise ValueError(f"split_role must be one of {PLANNER_SPLIT_ROLES}")
    indexes = tuple(range(count))
    if split_role == "planner_fit":
        return tuple(index for index in indexes if index % 8 != 0)
    if split_role == "planner_calibration":
        return tuple(index for index in indexes if index % 8 == 0)
    return indexes


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_channel_moments(
    channel_sum: Tensor, channel_square_sum: Tensor, element_count: int
) -> tuple[Tensor, Tensor]:
    """Convert exact per-channel sufficient statistics to population mean/std."""

    if (
        channel_sum.shape != (PLAN_CHANNELS,)
        or channel_square_sum.shape != (PLAN_CHANNELS,)
        or channel_sum.dtype != torch.float64
        or channel_square_sum.dtype != torch.float64
        or isinstance(element_count, bool)
        or not isinstance(element_count, int)
        or element_count < 1
        or not bool(torch.isfinite(channel_sum).all())
        or not bool(torch.isfinite(channel_square_sum).all())
    ):
        raise ValueError("motion-plan channel moments are malformed")
    mean = channel_sum / float(element_count)
    variance = channel_square_sum / float(element_count) - mean.square()
    # Small negative values can result from the two-pass algebra in finite
    # precision, but a materially negative population variance is invalid.
    if bool((variance < -1e-12).any()):
        raise ValueError("motion-plan channel variance is negative")
    std = variance.clamp_min(0.0).sqrt()
    if bool((std <= 1e-6).any()) or not bool(torch.isfinite(std).all()):
        raise ValueError("motion-plan channel standard deviation is degenerate")
    return mean, std


class MotionPlanNormalizer(nn.Module):
    """Immutable per-channel normalization fitted on planner-fit rows only."""

    def __init__(self, mean: Tensor, std: Tensor, *, artifact_sha256: str) -> None:
        super().__init__()
        if mean.shape != (PLAN_CHANNELS,) or std.shape != (PLAN_CHANNELS,):
            raise ValueError("motion-plan mean/std must each contain 16 channels")
        mean = mean.detach().float().clone()
        std = std.detach().float().clone()
        if (
            not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(std).all())
            or bool((std <= 1e-6).any())
            or len(artifact_sha256) != 64
        ):
            raise ValueError("motion-plan normalization statistics are invalid")
        self.register_buffer("mean", mean, persistent=True)
        self.register_buffer("std", std, persistent=True)
        self.artifact_sha256 = artifact_sha256

    def normalize(self, value: Tensor) -> Tensor:
        _validate_plan(value, label="unnormalized motion plan")
        mean = self.mean.to(device=value.device, dtype=value.dtype)[None, :, None, None, None]
        std = self.std.to(device=value.device, dtype=value.dtype)[None, :, None, None, None]
        result = (value - mean) / std
        _validate_plan(result, label="normalized motion plan")
        return result

    def denormalize(self, value: Tensor) -> Tensor:
        _validate_plan(value, label="normalized motion plan")
        mean = self.mean.to(device=value.device, dtype=value.dtype)[None, :, None, None, None]
        std = self.std.to(device=value.device, dtype=value.dtype)[None, :, None, None, None]
        result = value * std + mean
        _validate_plan(result, label="denormalized motion plan")
        return result


def load_motion_plan_normalizer(
    *, path: str, expected_sha256: str
) -> MotionPlanNormalizer:
    """Load and independently verify the train-fit-only stats artifact."""

    resolved = Path(path).expanduser()
    if (
        not resolved.is_absolute()
        or not resolved.is_file()
        or resolved.is_symlink()
        or len(expected_sha256) != 64
        or _sha256_file(resolved) != expected_sha256
    ):
        raise CausalMotionPlanError("motion-plan stats file is absent or differs")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CausalMotionPlanError("motion-plan stats JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise CausalMotionPlanError("motion-plan stats must contain one object")
    identity = payload.get("identity_sha256")
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    if (
        not isinstance(identity, str)
        or hashlib.sha256(_canonical_json(unsigned)).hexdigest() != identity
        or payload.get("schema_version") != NORMALIZATION_SCHEMA_VERSION
        or payload.get("kind") != NORMALIZATION_KIND
        or payload.get("status") != "complete_before_planner_training"
        or payload.get("split_rule") != "auxiliary_index_mod_8_nonzero"
        or payload.get("fit_clips") != 448
        or payload.get("calibration_clips_excluded") != 64
        or payload.get("validation_clips_read") != 0
        or payload.get("protected_test_clips_read") != 0
        or payload.get("elements_per_channel")
        != 448 * FUTURE_TOKENS * PLAN_HEIGHT * PLAN_WIDTH
        or payload.get("train_manifest_sha256") != FROZEN_TRAIN_MANIFEST_SHA256
        or payload.get("train_cache_metadata_sha256")
        != FROZEN_TRAIN_METADATA_SHA256
        or payload.get("train_rgb_sha256") != FROZEN_TRAIN_RGB_SHA256
        or payload.get("history_encoding")
        != "independent_five_frame_observed_only"
        or payload.get("future_tensor_used_for") != "statistics_target_only"
        or payload.get("causal_history_max_abs_tolerance") != 1e-4
        or isinstance(payload.get("causal_history_max_abs_observed"), bool)
        or not isinstance(payload.get("causal_history_max_abs_observed"), (int, float))
        or float(payload["causal_history_max_abs_observed"]) > 1e-4
    ):
        raise CausalMotionPlanError("motion-plan stats provenance/identity differs")
    try:
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise CausalMotionPlanError("motion-plan stats values are malformed") from exc
    return MotionPlanNormalizer(mean, std, artifact_sha256=expected_sha256)


def _finite_float(value: Tensor, *, label: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{label} contains non-finite values")


def _validate_history_latents(history: Tensor) -> None:
    expected = (
        PLAN_CHANNELS,
        HISTORY_TOKENS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    )
    if history.ndim != 5 or tuple(history.shape[1:]) != expected:
        raise ValueError(
            f"history_latents must have shape [B,{expected}], got "
            f"{tuple(history.shape)}"
        )
    _finite_float(history, label="history_latents")


def _validate_plan(plan: Tensor, *, label: str = "motion_plan") -> None:
    expected = (PLAN_CHANNELS, FUTURE_TOKENS, PLAN_HEIGHT, PLAN_WIDTH)
    if plan.ndim != 5 or tuple(plan.shape[1:]) != expected:
        raise ValueError(
            f"{label} must have shape [B,{expected}], got {tuple(plan.shape)}"
        )
    _finite_float(plan, label=label)


def pool_per_view(value: Tensor) -> Tensor:
    """Average each Wan view independently from ``24x40`` to ``6x10``.

    Splitting before pooling prevents convolution or pooling kernels from
    crossing the artificial seams between the top and wrist cameras.
    """

    if value.ndim != 4 or tuple(value.shape[1:]) != (
        PLAN_CHANNELS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    ):
        raise ValueError(
            "value must have shape [B,16,24,120], got " f"{tuple(value.shape)}"
        )
    _finite_float(value, label="per-view pooling input")
    views = value.split(VIEW_WIDTH, dim=-1)
    if len(views) != NUM_VIEWS or any(view.shape[-1] != VIEW_WIDTH for view in views):
        raise CausalMotionPlanError("Wan latent width no longer contains three views")
    pooled = [F.avg_pool2d(view, kernel_size=4, stride=4) for view in views]
    output = torch.cat(pooled, dim=-1)
    if tuple(output.shape[1:]) != (PLAN_CHANNELS, PLAN_HEIGHT, PLAN_WIDTH):
        raise CausalMotionPlanError("per-view pooled motion grid changed")
    return output


def upsample_per_view(plan: Tensor) -> Tensor:
    """Upsample a compact plan without mixing the three camera views."""

    _validate_plan(plan)
    views = plan.split(PLAN_VIEW_WIDTH, dim=-1)
    upsampled = []
    for view in views:
        batch, channels, frames, height, width = view.shape
        flattened = view.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        enlarged = F.interpolate(
            flattened,
            size=(LATENT_HEIGHT, VIEW_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        upsampled.append(
            enlarged.reshape(batch, frames, channels, LATENT_HEIGHT, VIEW_WIDTH)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
    output = torch.cat(upsampled, dim=-1)
    expected = (
        plan.shape[0],
        PLAN_CHANNELS,
        FUTURE_TOKENS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    )
    if tuple(output.shape) != expected:
        raise CausalMotionPlanError("upsampled motion-plan grid changed")
    return output


def motion_plan_target(full_latents: Tensor, history_latents: Tensor) -> Tensor:
    """Construct the clean planner target from training video only.

    This is the only public function in the deployment module that accepts a
    clean future tensor.  The first increment is anchored at the independently
    encoded final observed token so the target matches deployment history.
    """

    _validate_history_latents(history_latents)
    expected = (
        history_latents.shape[0],
        PLAN_CHANNELS,
        HISTORY_TOKENS + FUTURE_TOKENS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    )
    if full_latents.ndim != 5 or tuple(full_latents.shape) != expected:
        raise ValueError(
            f"full_latents must have shape {expected}, got {tuple(full_latents.shape)}"
        )
    _finite_float(full_latents, label="full_latents")
    first = pool_per_view(full_latents[:, :, 2] - history_latents[:, :, 1])
    second = pool_per_view(full_latents[:, :, 3] - full_latents[:, :, 2])
    target = torch.stack((first, second), dim=2)
    _validate_plan(target, label="motion-plan target")
    return target


def build_plan_condition(plan: Tensor) -> Tensor:
    """Place generated future increments on the full Wan token grid.

    History slots are exact zeros because observed appearance already enters
    Wan through its native reference channel.  This also makes it impossible
    for a shuffled-plan control to replace another sample's observed scene.
    """

    future = upsample_per_view(plan)
    condition = future.new_zeros(
        future.shape[0],
        PLAN_CHANNELS,
        HISTORY_TOKENS + FUTURE_TOKENS,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    )
    condition[:, :, HISTORY_TOKENS:] = future
    return condition


class _PlanResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, sigma_dim: int) -> None:
        super().__init__()
        groups = 16 if hidden_size % 16 == 0 else 1
        self.norm = nn.GroupNorm(groups, hidden_size)
        self.conv = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.sigma_projection = nn.Linear(sigma_dim, hidden_size)

    def forward(self, value: Tensor, sigma_embedding: Tensor) -> Tensor:
        residual = F.silu(self.norm(value))
        residual = self.conv(residual)
        residual = residual + self.sigma_projection(sigma_embedding).to(
            residual.dtype
        )[:, :, None, None, None]
        return value + F.silu(residual)


@dataclass(frozen=True)
class MotionPlanRollout:
    plan: Tensor
    calls: int


class CausalMotionPlanner(nn.Module):
    """Small RF planner conditioned only on observed history and actions."""

    def __init__(
        self,
        *,
        action_dim: int = 157,
        chunk_size: int = 5,
        morphology_count: int = 10,
        morphology_dim: int = 64,
        action_token_dim: int = 64,
        action_hidden: int = 256,
        action_map_channels: int = 32,
        hidden_size: int = 128,
        num_blocks: int = 4,
        sigma_dim: int = 128,
    ) -> None:
        super().__init__()
        if action_dim != 157 or chunk_size != 5:
            raise ValueError("the frozen ABC screen requires action_dim=157, chunk_size=5")
        if num_blocks < 1 or hidden_size < 16 or sigma_dim < 4 or sigma_dim % 2:
            raise ValueError("invalid planner width/depth/sigma embedding")
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.morphology_embedding = nn.Embedding(morphology_count, morphology_dim)
        frame_input = action_dim * chunk_size + morphology_dim
        self.frame_action_encoder = nn.Sequential(
            nn.Linear(frame_input, action_hidden),
            nn.SiLU(),
            nn.Linear(action_hidden, action_token_dim),
        )
        self.action_pool = nn.Sequential(
            nn.Linear(4 * action_token_dim, action_hidden),
            nn.SiLU(),
            nn.Linear(action_hidden, action_token_dim),
        )
        self.action_to_map = nn.Linear(action_token_dim, action_map_channels)
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(sigma_dim // 2, dtype=torch.float32)
            / max(sigma_dim // 2 - 1, 1)
        )
        self.register_buffer("sigma_frequencies", frequencies, persistent=False)
        input_channels = PLAN_CHANNELS + PLAN_CHANNELS + action_map_channels
        self.stem = nn.Conv3d(input_channels, hidden_size, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_PlanResidualBlock(hidden_size, sigma_dim) for _ in range(num_blocks)]
        )
        groups = 16 if hidden_size % 16 == 0 else 1
        self.head_norm = nn.GroupNorm(groups, hidden_size)
        self.head = nn.Conv3d(hidden_size, PLAN_CHANNELS, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.morphology_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Start from a zero RF velocity while leaving the hidden representation
        # trainable, mirroring the checkpoint-safe Wan auxiliary head.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _sigma_embedding(self, sigma: Tensor) -> Tensor:
        if sigma.ndim != 1:
            raise ValueError("planner sigma must have shape [B]")
        angles = sigma.float().unsqueeze(-1) * self.sigma_frequencies.unsqueeze(0)
        return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

    def _action_tokens(
        self, actions: Tensor, morphology_index: Tensor, *, dtype: torch.dtype
    ) -> Tensor:
        if actions.ndim != 4 or tuple(actions.shape[2:]) != (
            self.chunk_size,
            self.action_dim,
        ):
            raise ValueError("actions must have shape [B,T,5,157]")
        if actions.shape[1] < 12:
            raise ValueError("actions must contain transition chunks 4:12")
        if morphology_index is None:
            raise ValueError("the causal planner requires a morphology index")
        morphology_index = morphology_index.reshape(-1).long()
        if morphology_index.shape[0] != actions.shape[0]:
            raise ValueError("morphology batch differs from actions")
        future = actions[:, 4:12].to(dtype=dtype)
        flat = future.flatten(2)
        morphology = self.morphology_embedding(morphology_index).to(dtype=dtype)
        morphology = morphology[:, None, :].expand(-1, 8, -1)
        frame_tokens = self.frame_action_encoder(torch.cat((flat, morphology), dim=-1))
        grouped = frame_tokens.reshape(frame_tokens.shape[0], FUTURE_TOKENS, -1)
        return self.action_pool(grouped)

    def forward(
        self,
        noisy_plan: Tensor,
        sigma: Tensor,
        history_latents: Tensor,
        actions: Tensor,
        morphology_index: Tensor,
    ) -> Tensor:
        _validate_plan(noisy_plan, label="noisy motion plan")
        _validate_history_latents(history_latents)
        if noisy_plan.shape[0] != history_latents.shape[0]:
            raise ValueError("planner state and history batches differ")
        if sigma.shape != (noisy_plan.shape[0],):
            raise ValueError("planner sigma batch differs from state")
        history_map = pool_per_view(history_latents[:, :, -1]).unsqueeze(2)
        history_map = history_map.expand(-1, -1, FUTURE_TOKENS, -1, -1)
        action_tokens = self._action_tokens(
            actions, morphology_index, dtype=noisy_plan.dtype
        )
        action_map = self.action_to_map(action_tokens).permute(0, 2, 1)
        action_map = action_map[:, :, :, None, None].expand(
            -1, -1, -1, PLAN_HEIGHT, PLAN_WIDTH
        )
        hidden = self.stem(torch.cat((noisy_plan, history_map, action_map), dim=1))
        sigma_embedding = self._sigma_embedding(sigma).to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, sigma_embedding)
        velocity = self.head(F.silu(self.head_norm(hidden)))
        _validate_plan(velocity, label="planner velocity")
        return velocity

    def rollout_two_step(
        self,
        noise: Tensor,
        history_latents: Tensor,
        actions: Tensor,
        morphology_index: Tensor,
    ) -> MotionPlanRollout:
        """Generate a plan with the fixed two-call Euler schedule."""

        _validate_plan(noise, label="planner noise")
        state = noise
        calls = 0
        for sigma, next_sigma in zip(PLAN_SIGMAS[:-1], PLAN_SIGMAS[1:]):
            batch_sigma = state.new_full((state.shape[0],), sigma)
            velocity = self(
                state, batch_sigma, history_latents, actions, morphology_index
            )
            state = state + (next_sigma - sigma) * velocity
            calls += 1
        if calls != PLAN_STEPS:
            raise CausalMotionPlanError("planner rollout call count changed")
        _validate_plan(state, label="generated motion plan")
        return MotionPlanRollout(plan=state, calls=calls)
