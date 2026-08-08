#!/usr/bin/env python3
"""Train causal V-JEPA 2 predictors on motion-emphasized target variants.

This is an exploratory descendant of :mod:`tools.causal_vjepa2_screen`.  The
frozen semantic-screen executable is imported, never modified.  The default
configuration delegates its training step to that executable exactly.  The
new configuration represents a semantic sequence ``X`` by the invertible pack

``P[0] = X[0]`` and ``P[j] = X[j] - X[j-1]`` for ``j >= 1``.

Anchor and delta values can be normalized only with statistics computed from
the frozen *training* cache.  Sampling remains deployable: its public API has
no clean target, future RGB, or teacher argument.  A sampled pack is inverted
only after the generated trajectory is complete.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import causal_vjepa2_screen as screen  # noqa: E402
from tools import video_latent_forcing_poc as vlf  # noqa: E402
from tools.causal_vjepa2_cache_bridge import (  # noqa: E402
    construct_producer_attested_dataset,
)


RUN_SCHEMA = "causal-vjepa2-temporal-target-training-v1"
NORMALIZATION_SCHEMA = "causal-vjepa2-temporal-normalization-v1"
TARGET_MODES = ("absolute", "delta_pack")
TargetMode = Literal["absolute", "delta_pack"]
SELF_ROLLIN_TIME_RULES = (
    "sampled_final_clock_fraction",
    "fixed_final_clock_fraction",
)
SelfRollinTimeRule = Literal[
    "sampled_final_clock_fraction", "fixed_final_clock_fraction"
]
TARGET_SHAPE = screen.TARGET_SHAPE
TEMPORAL_AXIS = screen.TEMPORAL_AXIS
CHANNELS = TARGET_SHAPE[0]
DEFAULT_UPDATES = screen.TRAIN_UPDATES
DEFAULT_CHECKPOINT_EVERY = 500
STD_FLOOR = 1e-6
SELF_ROLLIN_HASH_SCHEMA = "causal-vjepa2-self-rollin-counter-v1"


class TemporalTargetError(RuntimeError):
    """A temporal-target scientific or provenance contract failed closed."""


def _channel_view(value: Tensor, reference: Tensor) -> Tensor:
    if tuple(value.shape) != (CHANNELS,):
        raise ValueError(f"normalization vector must have shape [{CHANNELS}]")
    return value.to(device=reference.device, dtype=reference.dtype).reshape(
        1, CHANNELS, 1, 1, 1
    )


def _validate_semantic_tensor(value: Tensor, *, label: str) -> None:
    if value.ndim != 5 or tuple(value.shape[1:]) != TARGET_SHAPE:
        raise ValueError(f"{label} must have shape [B,{TARGET_SHAPE}]")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite floating point")


def temporal_delta_pack(value: Tensor) -> Tensor:
    """Return the invertible ``[anchor, first differences]`` representation."""
    _validate_semantic_tensor(value, label="semantic target")
    packed = torch.empty_like(value)
    packed[:, :, :1] = value[:, :, :1]
    packed[:, :, 1:] = value[:, :, 1:] - value[:, :, :-1]
    return packed


def invert_temporal_delta_pack(value: Tensor) -> Tensor:
    """Invert :func:`temporal_delta_pack` without approximation."""
    _validate_semantic_tensor(value, label="temporal delta pack")
    result = torch.empty_like(value)
    result[:, :, :1] = value[:, :, :1]
    result[:, :, 1:] = value[:, :, :1] + value[:, :, 1:].cumsum(
        dim=TEMPORAL_AXIS
    )
    return result


@dataclass(frozen=True)
class TemporalNormalization:
    """Fixed channel statistics derived solely from the frozen train cache."""

    anchor_mean: Tensor
    anchor_std: Tensor
    delta_mean: Tensor
    delta_std: Tensor
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("anchor_mean", "anchor_std", "delta_mean", "delta_std"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or tuple(value.shape) != (CHANNELS,):
                raise ValueError(f"{name} must be a [{CHANNELS}] tensor")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite floating point")
        if bool((self.anchor_std <= 0).any()) or bool((self.delta_std <= 0).any()):
            raise ValueError("normalization standard deviations must be positive")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("normalization provenance must be a mapping")

    @classmethod
    def identity(cls) -> "TemporalNormalization":
        zeros = torch.zeros(CHANNELS, dtype=torch.float32)
        ones = torch.ones(CHANNELS, dtype=torch.float32)
        return cls(zeros, ones, zeros.clone(), ones.clone(), {"identity": True})

    def encode(self, semantic: Tensor, mode: TargetMode) -> Tensor:
        _validate_semantic_tensor(semantic, label="semantic target")
        if mode == "absolute":
            return semantic
        if mode != "delta_pack":
            raise ValueError(f"unknown temporal target mode: {mode}")
        packed = temporal_delta_pack(semantic)
        result = torch.empty_like(packed)
        result[:, :, :1] = (
            packed[:, :, :1] - _channel_view(self.anchor_mean, packed)
        ) / _channel_view(self.anchor_std, packed)
        result[:, :, 1:] = (
            packed[:, :, 1:] - _channel_view(self.delta_mean, packed)
        ) / _channel_view(self.delta_std, packed)
        return result

    def decode(self, representation: Tensor, mode: TargetMode) -> Tensor:
        _validate_semantic_tensor(representation, label="semantic representation")
        if mode == "absolute":
            return representation
        if mode != "delta_pack":
            raise ValueError(f"unknown temporal target mode: {mode}")
        packed = torch.empty_like(representation)
        packed[:, :, :1] = (
            representation[:, :, :1]
            * _channel_view(self.anchor_std, representation)
            + _channel_view(self.anchor_mean, representation)
        )
        packed[:, :, 1:] = (
            representation[:, :, 1:]
            * _channel_view(self.delta_std, representation)
            + _channel_view(self.delta_mean, representation)
        )
        return invert_temporal_delta_pack(packed)

    def decode_velocity(self, velocity: Tensor, mode: TargetMode) -> Tensor:
        """Apply only the linear part of ``decode`` to a tangent vector."""
        _validate_semantic_tensor(velocity, label="representation velocity")
        if mode == "absolute":
            return velocity
        if mode != "delta_pack":
            raise ValueError(f"unknown temporal target mode: {mode}")
        packed_velocity = torch.empty_like(velocity)
        packed_velocity[:, :, :1] = velocity[:, :, :1] * _channel_view(
            self.anchor_std, velocity
        )
        packed_velocity[:, :, 1:] = velocity[:, :, 1:] * _channel_view(
            self.delta_std, velocity
        )
        return invert_temporal_delta_pack(packed_velocity)


def _float_vector(payload: Mapping[str, Any], name: str) -> Tensor:
    value = payload.get(name)
    if not isinstance(value, list) or len(value) != CHANNELS:
        raise TemporalTargetError(f"normalization {name} must contain {CHANNELS} values")
    try:
        tensor = torch.tensor(value, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise TemporalTargetError(f"normalization {name} is not numeric") from exc
    return tensor


def load_temporal_normalization(
    path: str | Path,
    *,
    expected_train_manifest_sha256: str | None = None,
    expected_pca_sha256: str | None = None,
    expected_cache_metadata_sha256: str | None = None,
) -> tuple[TemporalNormalization, dict[str, Any]]:
    """Load and bind one immutable train-derived normalization artifact."""
    resolved = Path(path).expanduser().resolve()
    payload = vlf.load_json(resolved, "temporal normalization")
    if (
        payload.get("schema") != NORMALIZATION_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("complete") is not True
        or payload.get("target_kind") != screen.TARGET_KIND
        or payload.get("target_shape") != list(TARGET_SHAPE)
        or payload.get("source_storage_dtype") != "float16"
        or payload.get("encode_decode_dtype") != "float32"
        or payload.get("declared_roundtrip_max_abs_tolerance") != 2e-5
        or payload.get("temporal_axis") != TEMPORAL_AXIS
        or payload.get("channel_axis") != screen.TARGET_CHANNEL_AXIS
        or payload.get("split") != "train"
        or payload.get("clips") != screen.FROZEN_TRAIN_CLIPS
        or payload.get("std_floor") != STD_FLOOR
        or payload.get("statistics_source") != "frozen_train_cache_only"
    ):
        raise TemporalTargetError("normalization artifact contract is malformed")
    bindings = (
        ("train_manifest_sha256", expected_train_manifest_sha256),
        ("pca_sha256", expected_pca_sha256),
        ("cache_metadata_sha256", expected_cache_metadata_sha256),
    )
    for name, expected in bindings:
        value = payload.get(name)
        if not isinstance(value, str) or not screen.HEX64.fullmatch(value):
            raise TemporalTargetError(f"normalization lacks a valid {name}")
        if expected is not None and value != expected:
            raise TemporalTargetError(f"normalization {name} differs from training data")
    normalization = TemporalNormalization(
        anchor_mean=_float_vector(payload, "anchor_mean"),
        anchor_std=_float_vector(payload, "anchor_std"),
        delta_mean=_float_vector(payload, "delta_mean"),
        delta_std=_float_vector(payload, "delta_std"),
        provenance=payload,
    )
    record = {**vlf.file_record(resolved), "payload_sha256": screen.sha256_json(payload)}
    return normalization, record


def _flow_velocities(
    prediction_x: Tensor,
    noisy: Tensor,
    clean: Tensor,
    time: Tensor,
) -> tuple[Tensor, Tensor]:
    if not (prediction_x.shape == noisy.shape == clean.shape):
        raise ValueError("flow tensors must share shape")
    t = vlf.expand_clock(time, noisy).to(noisy.dtype)
    denominator = (1.0 - t).clamp_min(vlf.FROZEN_CLEAN_TIME_EPS)
    return (prediction_x - noisy) / denominator, (clean - noisy) / denominator


def per_example_normalized_temporal_velocity_mse(
    prediction_x: Tensor,
    noisy: Tensor,
    clean: Tensor,
    time: Tensor,
    *,
    target_mode: TargetMode,
    normalization: TemporalNormalization,
) -> Tensor:
    """Compare temporal changes in semantic-space flow velocity.

    Each original semantic channel is divided by its fixed train-set delta
    standard deviation.  For a normalized delta pack this is algebraically the
    velocity error of packed tokens 1..7, while the explicit decode keeps the
    definition identical for the absolute-target control.
    """
    predicted_velocity, target_velocity = _flow_velocities(
        prediction_x, noisy, clean, time
    )
    predicted_semantic_velocity = normalization.decode_velocity(
        predicted_velocity, target_mode
    )
    target_semantic_velocity = normalization.decode_velocity(
        target_velocity, target_mode
    )
    difference_error = (
        predicted_semantic_velocity.diff(dim=TEMPORAL_AXIS)
        - target_semantic_velocity.diff(dim=TEMPORAL_AXIS)
    ) / _channel_view(normalization.delta_std, predicted_semantic_velocity)
    return difference_error.square().flatten(1).mean(1)


def per_example_action_margin_loss(
    positive_error: Tensor, shuffled_error: Tensor, *, margin: float
) -> Tensor:
    """Hinge loss requiring own-action error to beat shuffled-action error."""
    if (
        positive_error.ndim != 1
        or shuffled_error.shape != positive_error.shape
        or margin <= 0
        or not math.isfinite(margin)
    ):
        raise ValueError("action margin requires finite [B] errors and margin > 0")
    return torch.relu(positive_error - shuffled_error + margin)


def self_rollin_counter_uniforms(
    *, seed: int, update: int, global_sample_ids: Tensor, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Return mask/time uniforms without reading or advancing any process RNG.

    Each value is addressed by ``(seed, update, global_sample_id, stream)``.
    Consequently enabling DELTA-R leaves the incumbent clock and Gaussian
    corruption stream bit-identical to its DELTA control.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("self-roll-in seed must be an integer")
    if isinstance(update, bool) or not isinstance(update, int) or update < 1:
        raise ValueError("self-roll-in update must be a positive integer")
    if not isinstance(global_sample_ids, Tensor) or global_sample_ids.ndim != 1:
        raise ValueError("global sample ids must be a one-dimensional tensor")
    ids = global_sample_ids.detach().to(device="cpu", dtype=torch.int64).tolist()
    if len(set(ids)) != len(ids) or any(value < 0 for value in ids):
        raise ValueError("global sample ids must be unique nonnegative integers")

    streams: list[list[float]] = [[], []]
    for global_id in ids:
        for stream_index, stream_name in enumerate(("mask", "start_time")):
            message = (
                f"{SELF_ROLLIN_HASH_SCHEMA}\0{seed}\0{update}\0"
                f"{global_id}\0{stream_name}"
            ).encode("utf-8")
            integer = int.from_bytes(hashlib.sha256(message).digest()[:8], "big")
            # Midpoint conversion gives a strict open-interval uniform and is
            # deterministic before the final explicit float32 cast.
            streams[stream_index].append((integer + 0.5) / float(1 << 64))
    return tuple(  # type: ignore[return-value]
        torch.tensor(values, device=device, dtype=torch.float32) for values in streams
    )


@dataclass(frozen=True)
class SelfRollinState:
    noisy_at_final_time: Tensor
    selected_mask: Tensor
    initial_time: Tensor
    final_time: Tensor
    hash_mask_uniform: Tensor
    hash_time_uniform: Tensor
    no_grad_model_calls: int


def stopped_one_step_self_rollin(
    model: nn.Module,
    *,
    clean: Tensor,
    noise: Tensor,
    true_noisy_at_final_time: Tensor,
    final_time: Tensor,
    noisy_video: Tensor,
    history: Tensor,
    actions: Tensor,
    probability: float,
    time_rule: SelfRollinTimeRule,
    fixed_final_clock_fraction: float,
    seed: int,
    update: int,
    global_sample_ids: Tensor,
) -> SelfRollinState:
    """Mix true ``z(t1)`` with stopped model roll-ins from ``z(t0)``.

    ``t1`` is always the incumbent auxiliary clock draw.  Selected examples
    set ``t0 = u*t1``, construct the exact forward-noised state from the same
    clean target and Gaussian noise, make one no-gradient prediction, and
    Euler-step it to ``t1``.  Unselected examples retain the true ``z(t1)``.
    """
    if not 0.0 <= probability <= 1.0 or not math.isfinite(probability):
        raise ValueError("self-roll-in probability must lie in [0,1]")
    if time_rule not in SELF_ROLLIN_TIME_RULES:
        raise ValueError(f"unknown self-roll-in time rule: {time_rule}")
    if not 0.0 <= fixed_final_clock_fraction < 1.0 or not math.isfinite(
        fixed_final_clock_fraction
    ):
        raise ValueError("fixed final-clock fraction must lie in [0,1)")
    if not (
        clean.shape
        == noise.shape
        == true_noisy_at_final_time.shape
        and clean.shape[0] == final_time.shape[0] == global_sample_ids.shape[0]
    ):
        raise ValueError("self-roll-in semantic tensors and keys must share shape")
    mask_uniform, time_uniform = self_rollin_counter_uniforms(
        seed=seed,
        update=update,
        global_sample_ids=global_sample_ids,
        device=clean.device,
    )
    selected = mask_uniform < probability
    fraction = (
        time_uniform
        if time_rule == "sampled_final_clock_fraction"
        else torch.full_like(time_uniform, fixed_final_clock_fraction)
    )
    initial_time = fraction * final_time
    mixed = true_noisy_at_final_time.clone()
    indexes = selected.nonzero(as_tuple=False).flatten()
    no_grad_calls = 0
    if indexes.numel() > 0:
        selected_clean = clean.index_select(0, indexes)
        selected_noise = noise.index_select(0, indexes)
        selected_t0 = initial_time.index_select(0, indexes)
        selected_t1 = final_time.index_select(0, indexes)
        noisy_at_t0 = vlf.corrupt_clean_time(
            selected_clean, selected_noise, selected_t0
        )
        # Bypass DDP reducer setup for this stopped auxiliary call.  Parameters
        # are synchronized by the gradient-bearing full-batch call below.
        rng_state = vlf.capture_rng_state()
        try:
            with torch.no_grad():
                _, first_prediction = vlf.model_forward(
                    vlf.unwrap_model(model),
                    noisy_video=noisy_video.index_select(0, indexes),
                    noisy_auxiliary=noisy_at_t0,
                    t_video=torch.zeros_like(selected_t0),
                    t_auxiliary=selected_t0,
                    history=history.index_select(0, indexes),
                    actions=actions.index_select(0, indexes),
                    condition_on_auxiliary=True,
                    predict_video=False,
                )
        finally:
            # The extra stopped call must not perturb dropout or any other
            # stochastic stream seen by the gradient-bearing incumbent call.
            vlf.restore_rng_state(rng_state)
        rolled = vlf.clean_time_euler_from_x(
            noisy_at_t0, first_prediction.detach(), selected_t0, selected_t1
        ).detach()
        mixed.index_copy_(0, indexes, rolled)
        no_grad_calls = 1
    return SelfRollinState(
        noisy_at_final_time=mixed,
        selected_mask=selected,
        initial_time=initial_time,
        final_time=final_time,
        hash_mask_uniform=mask_uniform,
        hash_time_uniform=time_uniform,
        no_grad_model_calls=no_grad_calls,
    )


def _is_exact_baseline(
    *,
    target_mode: TargetMode,
    normalization: TemporalNormalization | None,
    flow_loss_weight: float,
    temporal_velocity_loss_weight: float,
    action_margin_loss_weight: float,
    self_rollin_probability: float,
) -> bool:
    return (
        target_mode == "absolute"
        and normalization is None
        and flow_loss_weight == 1.0
        and temporal_velocity_loss_weight == 0.0
        and action_margin_loss_weight == 0.0
        and self_rollin_probability == 0.0
    )


def temporal_target_training_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    target_mode: TargetMode = "absolute",
    normalization: TemporalNormalization | None = None,
    flow_loss_weight: float = 1.0,
    temporal_velocity_loss_weight: float = 0.0,
    action_margin_loss_weight: float = 0.0,
    action_margin: float = 0.1,
    self_rollin_probability: float = 0.0,
    self_rollin_later_time_rule: SelfRollinTimeRule = "sampled_final_clock_fraction",
    self_rollin_fixed_final_clock_fraction: float = 0.5,
    rollin_seed: int | None = None,
    rollin_update: int | None = None,
    rollin_global_sample_ids: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Run one semantic-only update; defaults are the frozen baseline exactly."""
    if _is_exact_baseline(
        target_mode=target_mode,
        normalization=normalization,
        flow_loss_weight=flow_loss_weight,
        temporal_velocity_loss_weight=temporal_velocity_loss_weight,
        action_margin_loss_weight=action_margin_loss_weight,
        self_rollin_probability=self_rollin_probability,
    ):
        return screen.semantic_training_step(model, batch)
    if target_mode not in TARGET_MODES:
        raise ValueError(f"unknown temporal target mode: {target_mode}")
    weights = (
        flow_loss_weight,
        temporal_velocity_loss_weight,
        action_margin_loss_weight,
    )
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("loss weights must be finite and nonnegative")
    if flow_loss_weight + temporal_velocity_loss_weight <= 0:
        raise ValueError("flow or temporal-velocity supervision must be enabled")
    if not 0.0 <= self_rollin_probability <= 1.0 or not math.isfinite(
        self_rollin_probability
    ):
        raise ValueError("self-roll-in probability must lie in [0,1]")
    if self_rollin_probability > 0 and target_mode != "delta_pack":
        raise ValueError("stopped self-roll-in is preregistered only for delta_pack")
    if self_rollin_probability > 0 and action_margin_loss_weight > 0:
        raise ValueError("self-roll-in and diagnostic action margin are separate arms")
    if normalization is None and (
        target_mode == "delta_pack"
        or temporal_velocity_loss_weight > 0
        or action_margin_loss_weight > 0
    ):
        raise ValueError("this target/loss configuration requires train normalization")

    semantic_clean = batch["auxiliary_target"]
    clean = (
        semantic_clean
        if target_mode == "absolute"
        else normalization.encode(semantic_clean, target_mode)  # type: ignore[union-attr]
    )
    batch_size = clean.shape[0]
    clocks = vlf.sample_training_clocks("phase1", batch_size, clean.device)
    if (
        bool(clocks.video_time.ne(0).any())
        or bool(clocks.video_loss_mask.ne(0).any())
        or bool(clocks.auxiliary_loss_mask.ne(1).any())
    ):
        raise TemporalTargetError("shared Phase-1 clock implementation changed")
    noisy_video = torch.randn_like(batch["future"])
    auxiliary_noise = torch.randn_like(clean)
    noisy_auxiliary = vlf.corrupt_clean_time(
        clean, auxiliary_noise, clocks.auxiliary_time
    )
    training_state = noisy_auxiliary
    rollin: SelfRollinState | None = None
    if self_rollin_probability > 0:
        if (
            rollin_seed is None
            or rollin_update is None
            or rollin_global_sample_ids is None
        ):
            raise ValueError("self-roll-in requires seed/update/global-sample keys")
        rollin = stopped_one_step_self_rollin(
            model,
            clean=clean,
            noise=auxiliary_noise,
            true_noisy_at_final_time=noisy_auxiliary,
            final_time=clocks.auxiliary_time,
            noisy_video=noisy_video,
            history=batch["history"],
            actions=batch["actions"],
            probability=self_rollin_probability,
            time_rule=self_rollin_later_time_rule,
            fixed_final_clock_fraction=self_rollin_fixed_final_clock_fraction,
            seed=rollin_seed,
            update=rollin_update,
            global_sample_ids=rollin_global_sample_ids,
        )
        training_state = rollin.noisy_at_final_time

    shuffled_prediction_x: Tensor | None = None
    if action_margin_loss_weight > 0:
        if batch_size < 2:
            raise ValueError("action-shuffle margin requires a local microbatch >= 2")
        # One doubled forward keeps DDP reducer semantics valid while pairing
        # identical corruption/history with own versus rolled actions.
        _, predictions = vlf.model_forward(
            model,
            noisy_video=torch.cat((noisy_video, noisy_video), dim=0),
            noisy_auxiliary=torch.cat((training_state, training_state), dim=0),
            t_video=torch.cat((clocks.video_time, clocks.video_time), dim=0),
            t_auxiliary=torch.cat(
                (clocks.auxiliary_time, clocks.auxiliary_time), dim=0
            ),
            history=torch.cat((batch["history"], batch["history"]), dim=0),
            actions=torch.cat(
                (batch["actions"], batch["actions"].roll(shifts=1, dims=0)), dim=0
            ),
            condition_on_auxiliary=True,
            predict_video=False,
        )
        prediction_x, shuffled_prediction_x = predictions.split(batch_size, dim=0)
    else:
        _, prediction_x = vlf.model_forward(
            model,
            noisy_video=noisy_video,
            noisy_auxiliary=training_state,
            t_video=clocks.video_time,
            t_auxiliary=clocks.auxiliary_time,
            history=batch["history"],
            actions=batch["actions"],
            condition_on_auxiliary=True,
            predict_video=False,
        )
    flow_per_example = vlf.per_example_x_prediction_flow_mse(
        prediction_x,
        training_state,
        clean,
        auxiliary_noise,
        clocks.auxiliary_time,
    )
    flow_loss = vlf.masked_branch_loss(
        flow_per_example, clocks.auxiliary_loss_mask
    )

    zero = flow_loss.new_zeros(())
    temporal_per_example: Tensor | None = None
    temporal_loss = zero
    if temporal_velocity_loss_weight > 0 or action_margin_loss_weight > 0:
        assert normalization is not None
        temporal_per_example = per_example_normalized_temporal_velocity_mse(
            prediction_x,
            training_state,
            clean,
            clocks.auxiliary_time,
            target_mode=target_mode,
            normalization=normalization,
        )
        temporal_loss = vlf.masked_branch_loss(
            temporal_per_example, clocks.auxiliary_loss_mask
        )

    margin_loss = zero
    shuffled_temporal_loss = zero
    if action_margin_loss_weight > 0:
        assert (
            temporal_per_example is not None
            and normalization is not None
            and shuffled_prediction_x is not None
        )
        shuffled_per_example = per_example_normalized_temporal_velocity_mse(
            shuffled_prediction_x,
            training_state,
            clean,
            clocks.auxiliary_time,
            target_mode=target_mode,
            normalization=normalization,
        )
        shuffled_temporal_loss = vlf.masked_branch_loss(
            shuffled_per_example, clocks.auxiliary_loss_mask
        )
        margin_loss = vlf.masked_branch_loss(
            per_example_action_margin_loss(
                temporal_per_example,
                shuffled_per_example,
                margin=action_margin,
            ),
            clocks.auxiliary_loss_mask,
        )

    combined = (
        flow_loss_weight * flow_loss
        + temporal_velocity_loss_weight * temporal_loss
        + action_margin_loss_weight * margin_loss
    )
    weighted = 0.333 * combined
    selected_count = (
        rollin.selected_mask.sum().to(dtype=weighted.dtype)
        if rollin is not None
        else weighted.new_zeros(())
    )
    selected_denominator = selected_count.clamp_min(1.0)
    selected_time = (
        rollin.selected_mask.to(dtype=weighted.dtype)
        if rollin is not None
        else torch.zeros(batch_size, device=weighted.device, dtype=weighted.dtype)
    )
    initial_time_mean = (
        (rollin.initial_time * selected_time).sum() / selected_denominator
        if rollin is not None
        else weighted.new_zeros(())
    )
    time_advance_mean = (
        ((rollin.final_time - rollin.initial_time) * selected_time).sum()
        / selected_denominator
        if rollin is not None
        else weighted.new_zeros(())
    )
    initial_time_sum = (
        (rollin.initial_time * selected_time).sum()
        if rollin is not None
        else weighted.new_zeros(())
    )
    time_advance_sum = (
        ((rollin.final_time - rollin.initial_time) * selected_time).sum()
        if rollin is not None
        else weighted.new_zeros(())
    )
    return weighted, {
        "auxiliary_loss": flow_loss.detach(),
        "flow_loss": flow_loss.detach(),
        "normalized_temporal_velocity_loss": temporal_loss.detach(),
        "shuffled_temporal_velocity_loss": shuffled_temporal_loss.detach(),
        "action_margin_loss": margin_loss.detach(),
        "combined_auxiliary_loss": combined.detach(),
        "weighted_auxiliary_loss": weighted.detach(),
        "auxiliary_branch_count": clocks.auxiliary_loss_mask.sum().detach(),
        "self_rollin_selected_count": selected_count.detach(),
        "self_rollin_selected_fraction": (selected_count / batch_size).detach(),
        "self_rollin_initial_time_mean": initial_time_mean.detach(),
        "self_rollin_time_advance_mean": time_advance_mean.detach(),
        "self_rollin_initial_time_sum": initial_time_sum.detach(),
        "self_rollin_time_advance_sum": time_advance_sum.detach(),
        "self_rollin_no_grad_model_calls": weighted.new_tensor(
            0 if rollin is None else rollin.no_grad_model_calls
        ),
        "gradient_model_calls": weighted.new_tensor(1),
        "total_model_calls": weighted.new_tensor(
            1 if rollin is None else 1 + rollin.no_grad_model_calls
        ),
    }


@dataclass(frozen=True)
class TemporalTargetSample:
    representation_prediction: Tensor
    semantic_prediction: Tensor
    model_calls: int
    call_input_sha256_by_example: tuple[tuple[str, ...], ...]


@torch.inference_mode()
def sample_temporal_target(
    model: nn.Module,
    history: Tensor,
    actions: Tensor,
    *,
    video_noise: Tensor,
    auxiliary_noise: Tensor,
    steps: int,
    target_mode: TargetMode = "absolute",
    normalization: TemporalNormalization | None = None,
) -> TemporalTargetSample:
    """Generate and decode an auxiliary state without accepting clean future data."""
    if target_mode not in TARGET_MODES:
        raise ValueError(f"unknown temporal target mode: {target_mode}")
    if target_mode == "delta_pack" and normalization is None:
        raise ValueError("delta-pack sampling requires train normalization")
    sampled = screen.sample_semantic(
        model,
        history,
        actions,
        video_noise=video_noise,
        auxiliary_noise=auxiliary_noise,
        steps=steps,
    )
    semantic = (
        sampled.prediction
        if target_mode == "absolute"
        else normalization.decode(sampled.prediction, target_mode)  # type: ignore[union-attr]
    )
    return TemporalTargetSample(
        representation_prediction=sampled.prediction,
        semantic_prediction=semantic,
        model_calls=sampled.model_calls,
        call_input_sha256_by_example=sampled.call_input_sha256_by_example,
    )


def _accumulate_channel_statistics(
    sum_: Tensor, square_sum: Tensor, count: int, values: Tensor
) -> tuple[Tensor, Tensor, int]:
    values64 = values.detach().to(device="cpu", dtype=torch.float64)
    reduction = (0, 2, 3, 4)
    return (
        sum_ + values64.sum(dim=reduction),
        square_sum + values64.square().sum(dim=reduction),
        count + values64.numel() // CHANNELS,
    )


def fit_normalization_command(args: argparse.Namespace) -> int:
    """Fit channel statistics using all and only the frozen training cache."""
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise TemporalTargetError(f"immutable normalization already exists: {output}")
    # Fail before scanning the 5.5 GB cache if source cannot be bound.
    source = screen._source_record()  # noqa: SLF001
    manifest, rows, manifest_record = screen._manifest_record(  # noqa: SLF001
        args.train_manifest,
        split="train",
        expected_clips=screen.FROZEN_TRAIN_CLIPS,
    )
    dataset = construct_producer_attested_dataset(
        manifest, args.data_root, args.semantic_cache_root
    )
    cache_metadata = screen.validated_cache_metadata(dataset)
    if len(dataset) != screen.FROZEN_TRAIN_CLIPS or len(rows) != len(dataset):
        raise TemporalTargetError("normalization population is not exactly 64k train clips")
    # The dataset constructor has already hash-validated this read-only target
    # memmap against the train manifest. Reading it directly avoids decoding
    # irrelevant RGB while preserving exact manifest/cache row order.
    targets = getattr(dataset, "_targets", None)
    if (
        not isinstance(targets, np.memmap)
        or targets.flags.writeable
        or tuple(targets.shape) != (screen.FROZEN_TRAIN_CLIPS, *TARGET_SHAPE)
        or targets.dtype != np.float16
    ):
        raise TemporalTargetError("validated train target memmap is unavailable or mutable")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    anchor_sum = torch.zeros(CHANNELS, dtype=torch.float64)
    anchor_square_sum = torch.zeros_like(anchor_sum)
    delta_sum = torch.zeros_like(anchor_sum)
    delta_square_sum = torch.zeros_like(anchor_sum)
    anchor_count = 0
    delta_count = 0
    clips = 0
    for start in range(0, len(dataset), args.batch_size):
        end = min(start + args.batch_size, len(dataset))
        target = torch.from_numpy(np.array(targets[start:end], copy=True))
        _validate_semantic_tensor(target, label="cached semantic target")
        target64 = target.to(dtype=torch.float64)
        anchor_sum, anchor_square_sum, anchor_count = _accumulate_channel_statistics(
            anchor_sum, anchor_square_sum, anchor_count, target64[:, :, :1]
        )
        # Difference after conversion: no float16 subtraction may perturb the
        # train-derived delta population before float64 moments are computed.
        delta = target64[:, :, 1:] - target64[:, :, :-1]
        delta_sum, delta_square_sum, delta_count = _accumulate_channel_statistics(
            delta_sum, delta_square_sum, delta_count, delta
        )
        clips += target.shape[0]
    expected_anchor_count = screen.FROZEN_TRAIN_CLIPS * TARGET_SHAPE[2] * TARGET_SHAPE[3]
    expected_delta_count = expected_anchor_count * (TARGET_SHAPE[1] - 1)
    if (
        clips != screen.FROZEN_TRAIN_CLIPS
        or anchor_count != expected_anchor_count
        or delta_count != expected_delta_count
    ):
        raise TemporalTargetError("normalization did not consume the complete train cache")

    def moments(sum_: Tensor, square_sum: Tensor, count: int) -> tuple[Tensor, Tensor]:
        mean = sum_ / count
        variance = (square_sum / count - mean.square()).clamp_min(0.0)
        return mean, variance.sqrt().clamp_min(STD_FLOOR)

    anchor_mean, anchor_std = moments(
        anchor_sum, anchor_square_sum, anchor_count
    )
    delta_mean, delta_std = moments(delta_sum, delta_square_sum, delta_count)
    payload = {
        "schema": NORMALIZATION_SCHEMA,
        "status": "complete",
        "complete": True,
        "source": source,
        "entrypoint": vlf.file_record(__file__),
        "runtime": vlf.runtime_record(),
        "target_kind": screen.TARGET_KIND,
        "target_shape": list(TARGET_SHAPE),
        "source_storage_dtype": "float16",
        "encode_decode_dtype": "float32",
        "declared_roundtrip_max_abs_tolerance": 2e-5,
        "channel_axis": screen.TARGET_CHANNEL_AXIS,
        "temporal_axis": TEMPORAL_AXIS,
        "split": "train",
        "clips": clips,
        "statistics_source": "frozen_train_cache_only",
        "population_axes": {
            "anchor": "clip,height,width at temporal token 0; one moment per channel",
            "delta": "clip,temporal-difference,height,width; one moment per channel",
        },
        "train_manifest": manifest_record,
        "train_manifest_sha256": manifest_record["sha256"],
        "pca_sha256": cache_metadata["pca_sha256"],
        "cache_metadata_sha256": screen.sha256_json(cache_metadata),
        "cache_access_attestation": dataset.producer_attestation,
        "anchor_elements_per_channel": anchor_count,
        "delta_elements_per_channel": delta_count,
        "variance": "population",
        "std_floor": STD_FLOOR,
        "numerical_accumulation": {
            "dtype": "float64",
            "delta_subtraction_dtype": "float64",
            "serialized_moment_dtype": "float64 JSON numbers",
            "training_load_dtype": "float32",
            "target_read_order": "immutable train memmap manifest order",
            "deterministic_algorithms": True,
            "torch_cpu_threads": 1,
            "chunk_size": args.batch_size,
        },
        "anchor_mean": anchor_mean.tolist(),
        "anchor_std": anchor_std.tolist(),
        "delta_mean": delta_mean.tolist(),
        "delta_std": delta_std.tolist(),
        "protected_test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    vlf.atomic_write_json(output, payload, exclusive=True)
    print(vlf.file_record(output))
    return 0


def _normalization_for_training(
    args: argparse.Namespace, datasets: Mapping[str, Any]
) -> tuple[TemporalNormalization | None, dict[str, Any] | None]:
    required = (
        args.target_mode == "delta_pack"
        or args.temporal_velocity_loss_weight > 0
        or args.action_margin_loss_weight > 0
    )
    if args.normalization is None:
        if required:
            raise TemporalTargetError("selected target/loss requires --normalization")
        return None, None
    train_cache = datasets["semantic_cache"]["train"]
    normalization, record = load_temporal_normalization(
        args.normalization,
        expected_train_manifest_sha256=datasets["train"]["sha256"],
        expected_pca_sha256=train_cache["pca_sha256"],
        expected_cache_metadata_sha256=screen.sha256_json(train_cache),
    )
    return normalization, record


def _training_datasets(
    args: argparse.Namespace,
) -> tuple[Any, Any, dict[str, Any]]:
    """Bind c114 targets through their exact producer without relabelling them."""
    train_path, train_rows, train_record = screen._manifest_record(  # noqa: SLF001
        args.train_manifest,
        split="train",
        expected_clips=screen.FROZEN_TRAIN_CLIPS,
    )
    validation_path, validation_rows, validation_record = screen._manifest_record(  # noqa: SLF001
        args.validation_manifest,
        split="val",
        expected_clips=screen.FROZEN_VALIDATION_CLIPS,
    )
    if not screen.rows_episode_ids(train_rows).isdisjoint(  # type: ignore[attr-defined]
        screen.rows_episode_ids(validation_rows)  # type: ignore[attr-defined]
    ):
        raise TemporalTargetError("training and validation episodes overlap")
    train_dataset = construct_producer_attested_dataset(
        train_path, args.data_root, args.semantic_cache_root
    )
    validation_dataset = construct_producer_attested_dataset(
        validation_path, args.data_root, args.semantic_cache_root
    )
    train_cache = screen.validated_cache_metadata(train_dataset)
    validation_cache = screen.validated_cache_metadata(validation_dataset)
    screen._validate_training_cache_pair(  # noqa: SLF001
        train_cache,
        validation_cache,
        train_manifest=train_record,
        validation_manifest=validation_record,
    )
    return train_dataset, validation_dataset, {
        "train": train_record,
        "validation": validation_record,
        "semantic_cache": {
            "train": train_cache,
            "validation": validation_cache,
        },
        "cache_access": {
            "schema": "causal-vjepa2-producer-attested-cache-access-v1",
            "bridge": vlf.file_record(
                REPO_ROOT / "tools" / "causal_vjepa2_cache_bridge.py"
            ),
            "train": train_dataset.producer_attestation,
            "validation": validation_dataset.producer_attestation,
            "cache_relabelled_as_current_source": False,
            "protected_test_accessed": False,
        },
    }


def _checkpoint_updates(total_updates: int, every: int) -> tuple[int, ...]:
    return tuple(sorted({*range(every, total_updates + 1, every), total_updates}))


def _doe_arm(args: argparse.Namespace) -> str:
    base = (
        args.flow_loss_weight == 1.0
        and args.action_margin_loss_weight == 0.0
    )
    if base and args.target_mode == "absolute" and args.self_rollin_probability == 0:
        if args.temporal_velocity_loss_weight == 0.0:
            return "ABS"
        if args.temporal_velocity_loss_weight == 1.0:
            return "ABS-T"
    if base and args.target_mode == "delta_pack":
        if (
            args.self_rollin_probability == 0
            and args.temporal_velocity_loss_weight == 0.0
        ):
            return "DELTA"
        if (
            args.self_rollin_probability == 0
            and args.temporal_velocity_loss_weight == 1.0
        ):
            return "DELTA-T"
        if (
            args.self_rollin_probability == 0.5
            and args.temporal_velocity_loss_weight == 0.0
            and args.self_rollin_later_time_rule
            == "sampled_final_clock_fraction"
        ):
            return "DELTA-R"
    return "DIAGNOSTIC-OR-UNREGISTERED"


def _training_config(
    args: argparse.Namespace,
    context: vlf.DistributedContext,
    *,
    model_config: Mapping[str, Any],
    datasets: Mapping[str, Any],
    normalization_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    local_batch = args.global_batch_size // context.world_size
    micro_batch = args.micro_batch_size or local_batch
    doe_arm = _doe_arm(args)
    run_role = (
        "numerical_calibration"
        if args.updates == screen.CALIBRATION_UPDATES
        else "primary_5k"
        if args.updates == screen.TRAIN_UPDATES
        else "exploratory"
    )
    config = {
        "schema": RUN_SCHEMA,
        "source": screen._source_record(),  # noqa: SLF001
        "entrypoint": vlf.file_record(__file__),
        "dataset_source": screen._dataset_source_record(),  # noqa: SLF001
        "command": "train",
        "run_role": run_role,
        "doe_arm": doe_arm,
        "promotion_status": (
            "reference_not_promotable"
            if doe_arm == "ABS"
            else "primary_candidate"
            if doe_arm in {"ABS-T", "DELTA", "DELTA-T", "DELTA-R"}
            else "diagnostic_not_promotion_eligible"
        ),
        "initialization": "from_scratch_deterministic_no_pretrained_weights",
        "target_kind": screen.TARGET_KIND,
        "target_shape": list(TARGET_SHAPE),
        "target_mode": args.target_mode,
        "normalization": normalization_record,
        "clock_convention": vlf.CLOCK_CONVENTION,
        "updates": args.updates,
        "checkpoint_updates": list(
            _checkpoint_updates(args.updates, args.checkpoint_every)
        ),
        "seed": args.seed,
        "global_batch_size": args.global_batch_size,
        "world_size": context.world_size,
        "local_optimizer_batch_size": local_batch,
        "micro_batch_size_per_rank": micro_batch,
        "gradient_accumulation_steps": local_batch // micro_batch,
        "dtype": "bfloat16",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.95],
            "weight_decay": args.weight_decay,
            "warmup_updates": args.warmup_updates,
            "after_warmup": "constant",
            "gradient_clip_norm": args.gradient_clip_norm,
        },
        "ema": {
            "decay": args.ema_decay,
            "schedule": vlf.FROZEN_EMA_SCHEDULE,
            "reported_samples_use": True,
            "short_run_initialization_bias_corrected": True,
        },
        "loss": {
            "flow_weight": args.flow_loss_weight,
            "normalized_temporal_velocity_weight": args.temporal_velocity_loss_weight,
            "action_shuffle_margin_weight": args.action_margin_loss_weight,
            "action_shuffle_margin": args.action_margin,
            "action_shuffle_rule": "local_microbatch_roll_plus_one",
            "action_shuffle_margin_status": "diagnostic_only_not_promotion_eligible",
            "auxiliary_coefficient": 0.333,
        },
        "self_rollin": {
            "probability": args.self_rollin_probability,
            "later_time_rule": args.self_rollin_later_time_rule,
            "fixed_final_clock_fraction": args.self_rollin_fixed_final_clock_fraction,
            "counter_rng_schema": SELF_ROLLIN_HASH_SCHEMA,
            "counter_rng_key": "seed,update,global_sample_id,stream",
            "global_sample_id": (
                "(update-1)*global_batch_size + global_batch_position"
            ),
            "final_time": "unchanged incumbent auxiliary clock draw t1",
            "initial_time": "u*t1",
            "clean_and_noise": "bit-identical to ordinary true z(t1)",
            "first_prediction": "no_grad_stop_gradient_selected_examples_only",
            "selected_state": (
                "Euler(z(t0),xhat.detach(),t0,t1) replaces true z(t1)"
            ),
            "unselected_state": "true forward-noised z(t1)",
            "gradient_model_calls_per_microbatch": 1,
            "extra_batched_model_calls_per_microbatch": (
                "one iff at least one example is selected"
            ),
            "process_rng_restored_after_extra_call": True,
        },
        "generated_sampling_contract": {
            "eligible_checkpoint_update": screen.TRAIN_UPDATES,
            "weights": {
                "kind": "ema",
                "decay": vlf.FROZEN_EMA_DECAY,
                "schedule": vlf.FROZEN_EMA_SCHEDULE,
            },
            "nfe_grid": list(screen.NFE_GRID),
            "schedule": "uniform clean-time Euler from t=0 to t=1",
            "initial_state": "clip-addressed float32 Gaussian noise",
            "trajectory_state_dtype": "float32",
            "transformer_autocast": "cuda-bfloat16",
            "target_mode_decode": "delta pack decoded after autonomous trajectory",
            "clean_target_sampler_argument": "forbidden",
            "future_rgb_sampler_argument": "forbidden",
            "teacher_calls": 0,
        },
        "model": dict(model_config),
        "parameter_count": screen.FROZEN_MODEL_PARAMETERS,
        "datasets": dict(datasets),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "semantic_cache_root": str(Path(args.semantic_cache_root).expanduser().resolve()),
        "workers_per_rank": args.workers,
        "future_rgb_model_input": False,
        "video_loss_enabled": False,
        "teacher_model_calls": 0,
        "protected_test_accessed": False,
        "wandb": {
            "enabled": args.wandb,
            "entity": args.wandb_entity if args.wandb else None,
            "project": args.wandb_project if args.wandb else None,
            "group": None,
            "private_project_acknowledged": args.wandb_private_project_ack,
        },
    }
    config["experiment_identity_sha256"] = screen.sha256_json(
        {key: value for key, value in config.items() if key != "checkpoint_updates"}
    )
    return config


def training_command(args: argparse.Namespace) -> int:
    context = vlf.initialize_distributed()
    logger: vlf.LocalAndOptionalWandbLogger | None = None
    try:
        if args.global_batch_size % context.world_size:
            raise TemporalTargetError("global batch must divide by torchrun world size")
        local_batch = args.global_batch_size // context.world_size
        micro_batch = args.micro_batch_size or local_batch
        if local_batch % micro_batch:
            raise TemporalTargetError("microbatch must divide the per-rank optimizer batch")
        if args.action_margin_loss_weight > 0 and micro_batch < 2:
            raise TemporalTargetError("action margin requires microbatch >= 2 on every rank")
        run_dir = vlf.validated_run_dir(
            args.artifact_root, args.run_id, resume=args.resume is not None
        )
        train_dataset, validation_dataset, datasets = _training_datasets(args)
        del validation_dataset
        normalization, normalization_record = _normalization_for_training(args, datasets)
        vlf.seed_everything(args.seed, 0)
        model, model_config = screen.instantiate_model(args)
        source = screen._source_record()  # noqa: SLF001
        model.to(context.device)
        optimizer, scheduler = vlf.optimizer_and_scheduler(model, args, args.updates)
        ema = vlf.ModelEMA(model, decay=args.ema_decay)
        config = _training_config(
            args,
            context,
            model_config=model_config,
            datasets=datasets,
            normalization_record=normalization_record,
        )
        config_sha256 = screen.sha256_json(config)
        screen._assert_distributed_config(context, config_sha256)  # noqa: SLF001

        start_update = 0
        prior_wall = 0.0
        if args.resume is not None:
            existing = vlf.load_json(run_dir / "resolved_config.json", "resolved config")
            if screen.sha256_json(existing) != config_sha256:
                raise TemporalTargetError("resume arguments differ from immutable config")
            payload = vlf.load_checkpoint(
                args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                expected_config_sha256=config_sha256,
                context=context,
            )
            start_update = int(payload["completed_updates"])
            prior_wall = float(payload.get("cumulative_optimizer_wall_seconds", 0.0))
            if not 0 <= start_update < args.updates:
                raise TemporalTargetError("resume update is outside this run")
            if context.is_primary:
                vlf.reconcile_resume_artifacts(run_dir, start_update)
            context.barrier()
        else:
            if context.is_primary:
                run_dir.mkdir(parents=True, exist_ok=False)
                vlf.atomic_write_json(run_dir / "resolved_config.json", config, exclusive=True)
                vlf.atomic_write_json(
                    run_dir / "provenance.json",
                    {
                        "schema": RUN_SCHEMA,
                        "source": source,
                        "runtime": vlf.runtime_record(),
                        "command": [sys.executable, *sys.argv],
                        "resolved_config_sha256": config_sha256,
                        "secrets_persisted": False,
                    },
                    exclusive=True,
                )
            context.barrier()
            vlf.seed_everything(args.seed, context.rank)

        loader = vlf.build_loader(
            train_dataset,
            context=context,
            global_batch_size=args.global_batch_size,
            seed=args.seed,
            start_update=start_update,
            end_update=args.updates,
            workers=args.workers,
            micro_batch_size=args.micro_batch_size,
        )
        if context.world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        logger = vlf.LocalAndOptionalWandbLogger(
            run_dir, args, config, primary=context.is_primary
        )
        checkpoints = set(_checkpoint_updates(args.updates, args.checkpoint_every))
        accumulation_steps = local_batch // micro_batch
        iterator = iter(loader)
        nonfinite_updates = 0
        model.train()
        wall_start = time.perf_counter()
        cumulative_wall = prior_wall
        telemetry_names = (
            "weighted_auxiliary_loss",
            "auxiliary_loss",
            "normalized_temporal_velocity_loss",
            "shuffled_temporal_velocity_loss",
            "action_margin_loss",
            "auxiliary_branch_count",
            "self_rollin_selected_count",
            "self_rollin_initial_time_sum",
            "self_rollin_time_advance_sum",
            "self_rollin_no_grad_model_calls",
            "gradient_model_calls",
            "total_model_calls",
        )
        count_telemetry = {
            "auxiliary_branch_count",
            "self_rollin_selected_count",
            "self_rollin_initial_time_sum",
            "self_rollin_time_advance_sum",
            "self_rollin_no_grad_model_calls",
            "gradient_model_calls",
            "total_model_calls",
        }
        for update in range(start_update + 1, args.updates + 1):
            optimizer.zero_grad(set_to_none=True)
            accumulated = torch.zeros(len(telemetry_names), device=context.device)
            for microstep in range(accumulation_steps):
                batch = screen._validate_batch(next(iterator), context.device)  # noqa: SLF001
                global_position = context.rank * local_batch + microstep * micro_batch
                global_sample_ids = torch.arange(
                    (update - 1) * args.global_batch_size + global_position,
                    (update - 1) * args.global_batch_size
                    + global_position
                    + batch["auxiliary_target"].shape[0],
                    device=context.device,
                    dtype=torch.int64,
                )
                sync = (
                    model.no_sync()
                    if isinstance(model, DistributedDataParallel)
                    and microstep + 1 < accumulation_steps
                    else contextlib.nullcontext()
                )
                with sync, vlf._autocast(context.device):  # noqa: SLF001
                    weighted, telemetry = temporal_target_training_step(
                        model,
                        batch,
                        target_mode=args.target_mode,
                        normalization=normalization,
                        flow_loss_weight=args.flow_loss_weight,
                        temporal_velocity_loss_weight=args.temporal_velocity_loss_weight,
                        action_margin_loss_weight=args.action_margin_loss_weight,
                        action_margin=args.action_margin,
                        self_rollin_probability=args.self_rollin_probability,
                        self_rollin_later_time_rule=args.self_rollin_later_time_rule,
                        self_rollin_fixed_final_clock_fraction=(
                            args.self_rollin_fixed_final_clock_fraction
                        ),
                        rollin_seed=args.seed,
                        rollin_update=update,
                        rollin_global_sample_ids=global_sample_ids,
                    )
                    loss = weighted / accumulation_steps
                if not bool(torch.isfinite(loss)):
                    nonfinite_updates += 1
                    raise TemporalTargetError(
                        f"nonfinite loss at update {update}, microstep {microstep}"
                    )
                loss.backward()
                for index, name in enumerate(telemetry_names):
                    fallback = weighted.new_tensor(
                        1 if name in {"gradient_model_calls", "total_model_calls"} else 0
                    )
                    scale = 1.0 if name in count_telemetry else accumulation_steps
                    accumulated[index] += telemetry.get(name, fallback) / scale
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                nonfinite_updates += 1
                raise TemporalTargetError(f"nonfinite gradient norm at update {update}")
            optimizer.step()
            scheduler.step()
            ema.update(model)
            packed = context.sum_tensor(accumulated.float())
            observe = update == 1 or update % args.log_every == 0 or update in checkpoints
            if observe:
                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                cumulative_wall = prior_wall + time.perf_counter() - wall_start
                values = {
                    name: float(packed[index] / context.world_size)
                    for index, name in enumerate(telemetry_names)
                    if name not in count_telemetry
                }
                counts = {
                    name: float(packed[index])
                    for index, name in enumerate(telemetry_names)
                    if name in count_telemetry
                }
                selected = counts["self_rollin_selected_count"]
                logger.log(
                    {
                        "update": update,
                        **values,
                        **counts,
                        "self_rollin_selected_fraction": (
                            selected / args.global_batch_size
                        ),
                        "self_rollin_initial_time_mean": (
                            counts["self_rollin_initial_time_sum"] / max(selected, 1.0)
                        ),
                        "self_rollin_time_advance_mean": (
                            counts["self_rollin_time_advance_sum"] / max(selected, 1.0)
                        ),
                        "video_loss": 0.0,
                        "gradient_norm": float(gradient_norm.detach()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "cumulative_optimizer_wall_seconds": cumulative_wall,
                        "cumulative_gpu_hours": (
                            cumulative_wall * context.world_size / 3600.0
                            if context.device.type == "cuda"
                            else 0.0
                        ),
                    },
                    primary=context.is_primary,
                )
            if update in checkpoints:
                vlf.save_checkpoint(
                    run_dir / "checkpoints" / f"update_{update:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema,
                    update=update,
                    arm="phase1",
                    model_config=model_config,
                    config_sha256=config_sha256,
                    context=context,
                    cumulative_optimizer_wall_seconds=cumulative_wall,
                )
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        cumulative_wall = prior_wall + time.perf_counter() - wall_start
        if context.is_primary:
            checkpoint = run_dir / "checkpoints" / f"update_{args.updates:06d}.pt"
            vlf.atomic_write_json(
                run_dir / "complete.json",
                {
                    "schema": RUN_SCHEMA,
                    "status": "complete",
                    "completed_updates": args.updates,
                    "nonfinite_updates": nonfinite_updates,
                    "target_mode": args.target_mode,
                    "video_loss_enabled": False,
                    "future_rgb_model_input": False,
                    "teacher_model_calls": 0,
                    "protected_test_accessed": False,
                    "resolved_config_sha256": config_sha256,
                    "experiment_identity_sha256": config["experiment_identity_sha256"],
                    "source": source,
                    "resolved_config": vlf.file_record(run_dir / "resolved_config.json"),
                    "provenance": vlf.file_record(run_dir / "provenance.json"),
                    "checkpoint": vlf.file_record(checkpoint),
                    "parameter_counts": vlf.count_parameters(vlf.unwrap_model(model)),
                    "cumulative_optimizer_wall_seconds": cumulative_wall,
                    "cumulative_gpu_hours": (
                        cumulative_wall * context.world_size / 3600.0
                        if context.device.type == "cuda"
                        else 0.0
                    ),
                },
                exclusive=True,
            )
        context.barrier()
        return 0
    finally:
        if logger is not None:
            logger.finish()
        vlf.close_distributed(context)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=vlf.FROZEN_MODEL_WIDTH)
    parser.add_argument("--depth", type=int, default=vlf.FROZEN_MODEL_DEPTH)
    parser.add_argument("--heads", type=int, default=vlf.FROZEN_MODEL_HEADS)
    parser.add_argument("--mlp-ratio", type=float, default=vlf.FROZEN_MODEL_MLP_RATIO)


def _add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-private-project-ack", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit-normalization")
    fit.add_argument("--output", required=True)
    fit.add_argument("--data-root", required=True)
    fit.add_argument("--semantic-cache-root", required=True)
    fit.add_argument("--train-manifest", required=True)
    fit.add_argument("--batch-size", type=int, default=64)

    train = commands.add_parser("train")
    _add_model_arguments(train)
    train.add_argument("--artifact-root", required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--data-root", required=True)
    train.add_argument("--semantic-cache-root", required=True)
    train.add_argument("--train-manifest", required=True)
    train.add_argument("--validation-manifest", required=True)
    train.add_argument("--target-mode", choices=TARGET_MODES, default="absolute")
    train.add_argument("--normalization")
    train.add_argument("--flow-loss-weight", type=float, default=1.0)
    train.add_argument("--temporal-velocity-loss-weight", type=float, default=0.0)
    train.add_argument("--action-margin-loss-weight", type=float, default=0.0)
    train.add_argument("--action-margin", type=float, default=0.1)
    train.add_argument("--self-rollin-probability", type=float, default=0.0)
    train.add_argument(
        "--self-rollin-later-time-rule",
        choices=SELF_ROLLIN_TIME_RULES,
        default="sampled_final_clock_fraction",
    )
    train.add_argument(
        "--self-rollin-fixed-final-clock-fraction", type=float, default=0.5
    )
    train.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    train.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    train.add_argument("--seed", type=int, default=screen.FROZEN_SEED)
    train.add_argument(
        "--global-batch-size", type=int, default=screen.FROZEN_GLOBAL_BATCH_SIZE
    )
    train.add_argument("--micro-batch-size", type=int)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=vlf.FROZEN_LEARNING_RATE)
    train.add_argument("--warmup-updates", type=int, default=vlf.FROZEN_WARMUP_UPDATES)
    train.add_argument("--weight-decay", type=float, default=vlf.FROZEN_WEIGHT_DECAY)
    train.add_argument(
        "--gradient-clip-norm", type=float, default=vlf.FROZEN_GRADIENT_CLIP_NORM
    )
    train.add_argument("--ema-decay", type=float, default=vlf.FROZEN_EMA_DECAY)
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--resume")
    _add_wandb_arguments(train)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "fit-normalization":
        if args.batch_size < 1:
            raise TemporalTargetError("normalization batch size must be positive")
        return
    if args.workers < 0:
        raise TemporalTargetError("worker count must be nonnegative")
    if (
        args.width != vlf.FROZEN_MODEL_WIDTH
        or args.depth != vlf.FROZEN_MODEL_DEPTH
        or args.heads != vlf.FROZEN_MODEL_HEADS
        or args.mlp_ratio != vlf.FROZEN_MODEL_MLP_RATIO
    ):
        raise TemporalTargetError("model contract is frozen at width512/depth12/heads8/MLP4")
    if (
        args.seed != screen.FROZEN_SEED
        or args.global_batch_size != screen.FROZEN_GLOBAL_BATCH_SIZE
        or args.learning_rate != vlf.FROZEN_LEARNING_RATE
        or args.weight_decay != vlf.FROZEN_WEIGHT_DECAY
        or args.gradient_clip_norm != vlf.FROZEN_GRADIENT_CLIP_NORM
        or args.ema_decay != vlf.FROZEN_EMA_DECAY
        or args.updates < 1
        or args.checkpoint_every < 1
        or args.warmup_updates != vlf.FROZEN_WARMUP_UPDATES
        or args.log_every < 1
        or (args.micro_batch_size is not None and args.micro_batch_size < 1)
    ):
        raise TemporalTargetError(
            "training preserves seed1234/global-batch256/AdamW/clip1/EMA.9999"
        )
    weights = (
        args.flow_loss_weight,
        args.temporal_velocity_loss_weight,
        args.action_margin_loss_weight,
    )
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise TemporalTargetError("loss weights must be finite and nonnegative")
    if args.flow_loss_weight + args.temporal_velocity_loss_weight <= 0:
        raise TemporalTargetError("flow or temporal velocity supervision is required")
    if not math.isfinite(args.action_margin) or args.action_margin <= 0:
        raise TemporalTargetError("action margin must be finite and positive")
    if not 0.0 <= args.self_rollin_probability <= 1.0 or not math.isfinite(
        args.self_rollin_probability
    ):
        raise TemporalTargetError("self-roll-in probability must lie in [0,1]")
    if not 0.0 <= args.self_rollin_fixed_final_clock_fraction < 1.0 or not math.isfinite(
        args.self_rollin_fixed_final_clock_fraction
    ):
        raise TemporalTargetError("fixed final-clock fraction must lie in [0,1)")
    if args.self_rollin_probability > 0 and args.target_mode != "delta_pack":
        raise TemporalTargetError("self-roll-in is preregistered only for delta_pack")
    if args.self_rollin_probability > 0 and args.action_margin_loss_weight > 0:
        raise TemporalTargetError("self-roll-in and action margin are separate DOE arms")
    needs_normalization = (
        args.target_mode == "delta_pack"
        or args.temporal_velocity_loss_weight > 0
        or args.action_margin_loss_weight > 0
    )
    if needs_normalization != (args.normalization is not None):
        raise TemporalTargetError(
            "--normalization is required exactly for delta/temporal/action variants"
        )
    wandb_values = (
        args.wandb_entity,
        args.wandb_project,
        args.wandb_private_project_ack,
    )
    if args.wandb != all(bool(value) for value in wandb_values):
        raise TemporalTargetError(
            "W&B requires entity, project, and private-project acknowledgement"
        )
    if args.wandb and (
        args.wandb_entity != "zijiandu"
        or args.wandb_project != "dual-video-diffusion-private"
    ):
        raise TemporalTargetError(
            "W&B is frozen to private zijiandu/dual-video-diffusion-private"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        if args.command == "fit-normalization":
            return fit_normalization_command(args)
        return training_command(args)
    except (TemporalTargetError, screen.ScreenError, vlf.PocError, ValueError, OSError) as exc:
        print(f"Causal V-JEPA 2 temporal-target error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
