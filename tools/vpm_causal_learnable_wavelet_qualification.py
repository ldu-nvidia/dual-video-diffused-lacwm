#!/usr/bin/env python3
"""Post-selection VPM causal learnable-QMF qualification.

A spatial QMF basis is fit without residual targets on clean future VAE
latents, frozen, and compared with rank-matched Haar and FULL_DIRECT heads on
the exact VPM@1 residual. Rows 416--479 are explicitly prior-inspected;
rows 480--511, validation, and protected test are barred. LACWM's clock is
sigma=1 noise and sigma=0 clean video.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools import causal_compressibility_ladder as ladder
from tools import vpm_direct_residual_frontier as frontier


SCHEMA = "vpm-causal-learnable-wavelet-registration-v1"
FIT_SCHEMA = "vpm-causal-learnable-wavelet-fit-v1"
WEIGHT_SCHEMA = "vpm-causal-learnable-wavelet-weights-v1"
ROW_SCHEMA = "vpm-causal-learnable-wavelet-row-v1"
TIMING_SCHEMA = "vpm-causal-learnable-wavelet-timing-v1"
ENDPOINT_SCHEMA = "vpm-causal-learnable-wavelet-endpoint-v1"
ANALYSIS_SCHEMA = "vpm-causal-learnable-wavelet-analysis-v1"
COMPLETE_SCHEMA = "vpm-causal-learnable-wavelet-complete-v1"
AUDIT_SCHEMA = "vpm-causal-learnable-wavelet-audit-v1"
BASIS_SCHEMA = "vpm-causal-learnable-wavelet-basis-v1"
BASIS_WEIGHT_SCHEMA = "vpm-causal-learnable-wavelet-basis-weights-v1"
BASIS_CAL_ROW_SCHEMA = "vpm-causal-learnable-wavelet-basis-calibration-row-v1"

BASE_BANDS = ("ST_COARSE", "TEMPORAL_CHANGE", "SPATIOTEMPORAL_DETAIL")
HAAR_BANDS = tuple(f"HAAR_{band}" for band in BASE_BANDS)
QMF_BANDS = tuple(f"QMF_{band}" for band in BASE_BANDS)
BANDS = (*HAAR_BANDS, *QMF_BANDS)
TARGETS = ("FULL_DIRECT", *BANDS)
ALIGNMENTS = ("ALIGNED", "SHUFFLED")
ARMS = ("ZERO", *(f"{target}_{alignment}" for target in TARGETS for alignment in ALIGNMENTS))
DOSES = (256,)

FIT_RANGE = (128, 384)
CAL_RANGE = (384, 416)
DEV_RANGE = (416, 480)
RESERVE_RANGE = (480, 511)
STRUCTURAL_EXCLUDED_RANGE = (511, 512)
EXCLUDED_RANGE = (0, 128)
FREQUENCY_FORCING_SOURCE_SHA256 = "34ea62bf37e300012a3d5911c27a80048c81d8c58284d92d982c5983de3ce133"
BASIS_SEED = 20261001
FIT_NOISE_SEEDS = (20261002, 20261003)
CAL_NOISE_SEEDS: tuple[int, ...] = ()
DEV_NOISE_SEEDS = (20261005, 20261006, 20261007, 20261008)
PROJECTION_SEED = 20261009
BOOTSTRAP_SEED = 20261010
BOOTSTRAP_REPLICATES = 10_000
CAPACITY_RUNGS = (1024,)
RIDGE_LAMBDAS = (1.0,)
TOKEN_SAMPLE_COUNT = 256

BASIS_FILTER_LENGTH = 8
BASIS_FREE_ANGLES = 3
BASIS_OPT_STEPS = 512
BASIS_BATCH_SIZE = 8
BASIS_LEARNING_RATE = 0.02
BASIS_REGULARIZER_WEIGHT = 100.0
BASIS_SPARSITY_IMPROVEMENT_PERCENT = 5.0
BASIS_FILTER_HAAR_MAX_COSINE = 0.9999
BASIS_TERMINAL_MIN_ENERGY_FRACTION = 0.005
BASIS_ADMISSIBILITY_TOLERANCE = 2e-6
BASIS_MATRIX_ORTHOGONALITY_TOLERANCE = 3e-6
BASIS_RECONSTRUCTION_TOLERANCE = 3e-6
BASIS_RELATIVE_RECONSTRUCTION_ENERGY_TOLERANCE = 1e-11
BASIS_ENERGY_RATIO_TOLERANCE = 1e-5

LATENT_SHAPE = (16, 4, 24, 120)
HISTORY_LATENT_FRAMES = 2
VIEW_COUNT = 3
VIEW_WIDTH = 40
HAAR_MAX_ABS_TOLERANCE = 2e-6
HAAR_RELATIVE_ENERGY_TOLERANCE = 1e-12
HAAR_ORTHOGONALITY_TOLERANCE = 2e-6
LEAKAGE_TOLERANCE = 2e-6
GROUPED_BAND_MIN_ENERGY_FRACTION = 0.02
GROUPED_BAND_MAX_ENERGY_FRACTION = 0.96

ERROR_METRICS = (
    "corrected_velocity_mse",
    "video_future_nmse",
    "decoded_mse_unit_range",
    "decoded_temporal_difference_mse_unit_range",
)
BAND_METRICS = (
    "target_r2",
    "target_cosine",
    "target_energy",
    "target_rms",
    "prediction_energy",
    "correction_rms",
    "preprojection_offband_relative_energy",
    "postprojection_offband_relative_energy",
)
SPARSITY_METRICS = (
    "normalized_l1",
    "hoyer_sparsity",
    "small_coefficient_fraction_at_0p1_rms",
    "top_10_percent_energy_fraction",
    "coefficient_input_energy_ratio",
    "terminal_LL_energy_fraction",
    "terminal_LH_energy_fraction",
    "terminal_HL_energy_fraction",
    "terminal_HH_energy_fraction",
)


class WaveletError(ladder.LadderError):
    """A prospective basis, decomposition, access, or evidence contract changed."""


def endpoint_name(dose: int, target: str, alignment: str) -> str:
    if dose not in DOSES or target not in TARGETS or alignment not in ALIGNMENTS:
        raise WaveletError("invalid endpoint coordinates")
    return f"D{dose:03d}_{target}_{alignment}"


ENDPOINTS = (
    "VPM_OFF",
    "ZERO",
    *(endpoint_name(dose, target, alignment) for dose in DOSES for target in TARGETS for alignment in ALIGNMENTS),
)


def _configure_shared_helpers() -> None:
    # Imported ridge helpers read these globals at call time.
    ladder.ARMS = ARMS
    ladder.CAPACITY_RUNGS = CAPACITY_RUNGS
    ladder.RIDGE_LAMBDAS = RIDGE_LAMBDAS
    ladder.PROJECTION_SEED = PROJECTION_SEED
    ladder.TOKEN_SAMPLE_COUNT = TOKEN_SAMPLE_COUNT


def _rotation(theta: Any) -> Any:
    import torch

    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    return torch.stack(
        (torch.stack((cosine, -sine)), torch.stack((sine, cosine)))
    )


def lattice_lowpass(free_angles: Any) -> Any:
    """Length-eight orthogonal lowpass from a paraunitary lattice.

    The fourth rotation fixes the total angle at -pi/4, which gives lowpass
    sum sqrt(2) and zero-DC QMF highpass while every lattice factor remains
    paraunitary.
    """
    import torch
    import torch.nn.functional as functional

    if free_angles.ndim != 1 or free_angles.numel() != BASIS_FREE_ANGLES:
        raise WaveletError("learned basis must contain exactly three free angles")
    # The lattice is part of the numerical evidence contract. In particular,
    # an enclosing model-level bf16 autocast region must not silently quantize
    # the polynomial einsum. Preserve the caller's FP32/FP64 dtype and gradient.
    with torch.autocast(device_type=free_angles.device.type, enabled=False):
        angles = torch.cat(
            (
                free_angles,
                free_angles.new_tensor([-math.pi / 4.0])
                - free_angles.sum().reshape(1),
            )
        )
        polynomial = _rotation(angles[0]).unsqueeze(-1)
        for theta in angles[1:]:
            # Right multiplication by diag(1,z^-1) then a constant rotation.
            delayed = torch.stack(
                (
                    functional.pad(polynomial[:, 0, :], (0, 1)),
                    functional.pad(polynomial[:, 1, :], (1, 0)),
                ),
                dim=1,
            )
            polynomial = torch.einsum("ijl,jk->ikl", delayed, _rotation(theta))
        lowpass = torch.stack(
            (polynomial[0, 0], polynomial[0, 1]), dim=-1
        ).reshape(-1)
    if lowpass.numel() != BASIS_FILTER_LENGTH:
        raise WaveletError("lattice produced the wrong filter support")
    return lowpass


def qmf_highpass(lowpass: Any) -> Any:
    import torch

    if lowpass.ndim != 1 or lowpass.numel() != BASIS_FILTER_LENGTH:
        raise WaveletError("QMF lowpass shape differs")
    signs = torch.where(
        torch.arange(BASIS_FILTER_LENGTH, device=lowpass.device) % 2 == 0,
        lowpass.new_tensor(1.0),
        lowpass.new_tensor(-1.0),
    )
    return torch.flip(lowpass, dims=(0,)) * signs


def haar_lowpass(*, device: Any = None, dtype: Any = None) -> Any:
    import torch

    result = torch.zeros(BASIS_FILTER_LENGTH, device=device, dtype=dtype or torch.float32)
    result[:2] = 1.0 / math.sqrt(2.0)
    return result


def analysis_matrix(filter_taps: Any, length: int) -> Any:
    """Periodic stride-two analysis with a phase shared by Haar and QMF."""
    import torch

    if length % 2 or length < BASIS_FILTER_LENGTH:
        raise WaveletError("analysis length must be even and at least filter support")
    rows = length // 2
    indexes = (
        2 * torch.arange(rows, device=filter_taps.device)[:, None]
        + torch.arange(BASIS_FILTER_LENGTH, device=filter_taps.device)[None, :]
    ) % length
    matrix = filter_taps.new_zeros((rows, length))
    return matrix.scatter_add(1, indexes, filter_taps.expand(rows, -1))


def _spatial_lowpass_view(value: Any, lowpass: Any) -> Any:
    import torch

    if value.ndim != 5 or value.shape[-2] % 2 or value.shape[-1] % 2:
        raise WaveletError("per-view wavelet tensor must be [B,C,T,even-H,even-W]")
    # Explicit casts do not override an outer autocast context for einsum.
    # The transform and evidence tolerances are FP32 contracts.
    with torch.autocast(device_type=value.device.type, enabled=False):
        lowpass = lowpass.to(device=value.device, dtype=torch.float32)
        height_matrix = analysis_matrix(lowpass, int(value.shape[-2]))
        width_matrix = analysis_matrix(lowpass, int(value.shape[-1]))
        coefficients = value.float()
        coefficients = torch.einsum(
            "ih,bcthw,jw->bctij", height_matrix, coefficients, width_matrix
        )
        return torch.einsum(
            "ih,bctij,jw->bcthw", height_matrix, coefficients, width_matrix
        )


def _temporal_lowpass(value: Any) -> Any:
    if value.ndim != 5 or value.shape[2] % 2:
        raise WaveletError("temporal Haar tensor must have an even frame count")
    batch, channels, frames, height, width = value.shape
    pairs = value.reshape(batch, channels, frames // 2, 2, height, width)
    low = pairs.mean(dim=3, keepdim=True)
    return low.expand_as(pairs).reshape_as(value)


def _wavelet_decompose_fp32(
    value: Any, *, lowpass: Any, history_frames: int
) -> dict[str, Any]:
    """Return three orthogonal full-shape projections, computed per view."""
    import torch

    if value.ndim != 5 or tuple(value.shape[1:]) != LATENT_SHAPE:
        raise WaveletError(
            f"pinned VPM latent must be [B,{LATENT_SHAPE}], got {tuple(value.shape)}"
        )
    if history_frames != HISTORY_LATENT_FRAMES:
        raise WaveletError("pinned VPM history boundary differs")
    future = value.float()[:, :, history_frames:]
    if future.shape[2] != 2 or future.shape[-1] != VIEW_COUNT * VIEW_WIDTH:
        raise WaveletError("pinned future/view geometry differs")
    coarse_views = []
    change_views = []
    detail_views = []
    for view in range(VIEW_COUNT):
        left = view * VIEW_WIDTH
        right = left + VIEW_WIDTH
        current = future[..., left:right]
        spatial = _spatial_lowpass_view(current, lowpass)
        coarse = _temporal_lowpass(spatial)
        coarse_views.append(coarse)
        change_views.append(spatial - coarse)
        detail_views.append(current - spatial)
    future_bands = {
        "ST_COARSE": torch.cat(coarse_views, dim=-1),
        "TEMPORAL_CHANGE": torch.cat(change_views, dim=-1),
        "SPATIOTEMPORAL_DETAIL": torch.cat(detail_views, dim=-1),
    }
    result = {}
    for band, projected in future_bands.items():
        full = torch.zeros_like(value, dtype=torch.float32)
        full[:, :, history_frames:] = projected
        result[band] = full
    return result


def wavelet_decompose(
    value: Any, *, lowpass: Any, history_frames: int
) -> dict[str, Any]:
    import torch

    with torch.autocast(device_type=value.device.type, enabled=False):
        return _wavelet_decompose_fp32(
            value.float(), lowpass=lowpass.float(), history_frames=history_frames
        )


def haar_decompose(value: Any, *, history_frames: int) -> dict[str, Any]:
    import torch

    return wavelet_decompose(
        value,
        lowpass=haar_lowpass(device=value.device, dtype=torch.float32),
        history_frames=history_frames,
    )


def qmf_decompose(value: Any, *, lowpass: Any, history_frames: int) -> dict[str, Any]:
    return wavelet_decompose(value, lowpass=lowpass, history_frames=history_frames)


def _future_only(value: Any, history_frames: int) -> Any:
    import torch

    result = torch.zeros_like(value, dtype=torch.float32)
    result[:, :, history_frames:] = value.float()[:, :, history_frames:]
    return result


def _partition_contract_fp32(
    value: Any, bands: Mapping[str, Any], *, history_frames: int
) -> dict[str, Any]:
    import torch

    if set(bands) != set(BASE_BANDS):
        raise WaveletError("wavelet partition band inventory differs")
    target = _future_only(value, history_frames)
    reconstruction = sum((bands[name] for name in BASE_BANDS), torch.zeros_like(target))
    error = reconstruction - target
    energy = target.square().sum().clamp_min(1e-30)
    max_abs = float(error.abs().max().detach().cpu())
    relative_energy = float((error.square().sum() / energy).detach().cpu())
    pairs = {}
    for first_index, first in enumerate(BASE_BANDS):
        for second in BASE_BANDS[first_index + 1 :]:
            pairs[f"{first}__{second}"] = float(
                ((bands[first] * bands[second]).sum().abs() / energy).detach().cpu()
            )
    fractions = {
        name: float((bands[name].square().sum() / energy).detach().cpu())
        for name in BASE_BANDS
    }
    history_nonzero = sum(
        int(torch.count_nonzero(bands[name][:, :, :history_frames]).detach().cpu())
        for name in BASE_BANDS
    )
    if (
        max_abs > HAAR_MAX_ABS_TOLERANCE
        or relative_energy > HAAR_RELATIVE_ENERGY_TOLERANCE
        or any(value > HAAR_ORTHOGONALITY_TOLERANCE for value in pairs.values())
        or history_nonzero != 0
        or not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=5e-6)
    ):
        raise WaveletError("wavelet reconstruction/orthogonality contract failed")
    return {
        "max_abs_reconstruction_error": max_abs,
        "relative_reconstruction_energy": relative_energy,
        "normalized_pairwise_inner_products": pairs,
        "band_energy_fractions": fractions,
        "history_nonzero": history_nonzero,
    }


def partition_contract(
    value: Any, bands: Mapping[str, Any], *, history_frames: int
) -> dict[str, Any]:
    import torch

    with torch.autocast(device_type=value.device.type, enabled=False):
        return _partition_contract_fp32(
            value.float(),
            {name: tensor.float() for name, tensor in bands.items()},
            history_frames=history_frames,
        )


def _new_partition_summary() -> dict[str, Any]:
    return {
        "checked_batches": 0,
        "max_abs_reconstruction_error": 0.0,
        "max_relative_reconstruction_energy": 0.0,
        "max_abs_normalized_pairwise_inner_product": 0.0,
        "mean_band_energy_fraction_sums": {name: 0.0 for name in BASE_BANDS},
    }


def _update_partition_summary(summary: dict[str, Any], receipt: Mapping[str, Any]) -> None:
    summary["checked_batches"] += 1
    summary["max_abs_reconstruction_error"] = max(
        summary["max_abs_reconstruction_error"],
        float(receipt["max_abs_reconstruction_error"]),
    )
    summary["max_relative_reconstruction_energy"] = max(
        summary["max_relative_reconstruction_energy"],
        float(receipt["relative_reconstruction_energy"]),
    )
    summary["max_abs_normalized_pairwise_inner_product"] = max(
        summary["max_abs_normalized_pairwise_inner_product"],
        *(float(value) for value in receipt["normalized_pairwise_inner_products"].values()),
    )
    for band in BASE_BANDS:
        summary["mean_band_energy_fraction_sums"][band] += float(
            receipt["band_energy_fractions"][band]
        )


def _finalize_partition_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    count = int(summary["checked_batches"])
    if count <= 0:
        raise WaveletError("no wavelet-partition batches were checked")
    result = {
        **{key: value for key, value in summary.items() if key != "mean_band_energy_fraction_sums"},
        "mean_band_energy_fraction": {
            name: float(value) / count
            for name, value in summary["mean_band_energy_fraction_sums"].items()
        },
        "passed": (
            float(summary["max_abs_reconstruction_error"]) <= HAAR_MAX_ABS_TOLERANCE
            and float(summary["max_relative_reconstruction_energy"]) <= HAAR_RELATIVE_ENERGY_TOLERANCE
            and float(summary["max_abs_normalized_pairwise_inner_product"]) <= HAAR_ORTHOGONALITY_TOLERANCE
        ),
    }
    fractions = result["mean_band_energy_fraction"]
    result["nontrivial_grouped_band_energy"] = all(
        GROUPED_BAND_MIN_ENERGY_FRACTION <= float(value) <= GROUPED_BAND_MAX_ENERGY_FRACTION
        for value in fractions.values()
    )
    result["passed"] = bool(result["passed"] and result["nontrivial_grouped_band_energy"])
    return result


def basis_admissibility(lowpass: Any) -> dict[str, Any]:
    import torch

    lowpass = lowpass.detach().float()
    highpass = qmf_highpass(lowpass)
    shifts = {
        str(shift): float((lowpass[:-shift] * lowpass[shift:]).sum().abs().cpu())
        for shift in range(2, BASIS_FILTER_LENGTH, 2)
    }
    matrices = {}
    max_orthogonality = 0.0
    max_reconstruction = 0.0
    max_relative_reconstruction_energy = 0.0
    max_energy_ratio_error = 0.0
    generator = torch.Generator(device="cpu").manual_seed(BASIS_SEED + 97)
    for length in (LATENT_SHAPE[2], VIEW_WIDTH):
        low = analysis_matrix(lowpass.cpu(), length)
        high = analysis_matrix(highpass.cpu(), length)
        full = torch.cat((low, high), dim=0)
        gram_error = float((full @ full.T - torch.eye(length)).abs().max())
        sample = torch.randn((5, length), generator=generator)
        coefficients = sample @ full.T
        reconstructed = coefficients @ full
        error = reconstructed - sample
        energy = sample.square().sum().clamp_min(1e-30)
        energy_ratio = float(coefficients.square().sum() / energy)
        matrices[str(length)] = {
            "max_abs_orthogonality_error": gram_error,
            "max_abs_reconstruction_error": float(error.abs().max()),
            "relative_reconstruction_energy": float(error.square().sum() / energy),
            "coefficient_input_energy_ratio": energy_ratio,
            "rank": int(torch.linalg.matrix_rank(full)),
        }
        max_orthogonality = max(max_orthogonality, gram_error)
        max_reconstruction = max(max_reconstruction, matrices[str(length)]["max_abs_reconstruction_error"])
        max_relative_reconstruction_energy = max(
            max_relative_reconstruction_energy,
            matrices[str(length)]["relative_reconstruction_energy"],
        )
        max_energy_ratio_error = max(max_energy_ratio_error, abs(energy_ratio - 1.0))
    diagnostics = {
        "lowpass": [float(value) for value in lowpass.cpu().tolist()],
        "highpass": [float(value) for value in highpass.cpu().tolist()],
        "lowpass_sum_absolute_error": float(abs(float(lowpass.sum()) - math.sqrt(2.0))),
        "highpass_sum_absolute_error": float(abs(float(highpass.sum()))),
        "lowpass_norm_absolute_error": float(abs(float(lowpass.square().sum()) - 1.0)),
        "even_shift_autocorrelation_absolute": shifts,
        "max_admissibility_residual": max(
            float(abs(float(lowpass.sum()) - math.sqrt(2.0))),
            float(abs(float(highpass.sum()))),
            float(abs(float(lowpass.square().sum()) - 1.0)),
            *shifts.values(),
        ),
        "analysis_matrices": matrices,
        "max_analysis_matrix_orthogonality_error": max_orthogonality,
        "max_abs_reconstruction_error": max_reconstruction,
        "max_relative_reconstruction_energy": max_relative_reconstruction_energy,
        "max_coefficient_input_energy_ratio_error": max_energy_ratio_error,
        "threshold_gates_enabled": False,
        "terminal_band_gate_retention": [1.0, 1.0, 1.0, 1.0],
    }
    diagnostics["passed"] = bool(
        diagnostics["max_admissibility_residual"] <= BASIS_ADMISSIBILITY_TOLERANCE
        and max_orthogonality <= BASIS_MATRIX_ORTHOGONALITY_TOLERANCE
        and max_reconstruction <= BASIS_RECONSTRUCTION_TOLERANCE
        and max_relative_reconstruction_energy <= BASIS_RELATIVE_RECONSTRUCTION_ENERGY_TOLERANCE
        and max_energy_ratio_error <= BASIS_ENERGY_RATIO_TOLERANCE
        and all(value == 1.0 for value in diagnostics["terminal_band_gate_retention"])
    )
    return diagnostics


def packet_coefficients(value: Any, lowpass: Any) -> Any:
    """One-level separable packet coefficients, isolated per camera view."""
    import torch

    if (
        value.ndim != 5
        or value.shape[2] != 2
        or value.shape[-2] != LATENT_SHAPE[2]
        or value.shape[-1] != VIEW_COUNT * VIEW_WIDTH
    ):
        raise WaveletError("basis tensor must be the exact two-future-token latent marginal")
    with torch.autocast(device_type=value.device.type, enabled=False):
        lowpass = lowpass.to(device=value.device, dtype=torch.float32)
        highpass = qmf_highpass(lowpass)
        height = torch.cat(
            (
                analysis_matrix(lowpass, LATENT_SHAPE[2]),
                analysis_matrix(highpass, LATENT_SHAPE[2]),
            ),
            dim=0,
        )
        width = torch.cat(
            (
                analysis_matrix(lowpass, VIEW_WIDTH),
                analysis_matrix(highpass, VIEW_WIDTH),
            ),
            dim=0,
        )
        views = []
        for view in range(VIEW_COUNT):
            left = view * VIEW_WIDTH
            current = value.float()[..., left : left + VIEW_WIDTH]
            views.append(torch.einsum("ih,bcthw,jw->bctij", height, current, width))
        return torch.cat(views, dim=-1)


def paper_regularizers(lowpass: Any) -> dict[str, Any]:
    """The three filter losses stated in Frequency-Forcing's primary source."""
    highpass = qmf_highpass(lowpass)
    shift_terms = [
        (lowpass[:-shift] * lowpass[shift:]).sum().abs()
        for shift in range(2, BASIS_FILTER_LENGTH, 2)
    ]
    ortho = (
        (lowpass.square().sum() - 1.0).abs() + sum(shift_terms)
    ) / (1 + len(shift_terms))
    return {
        "sum": (lowpass.sum() - math.sqrt(2.0)).abs(),
        "highpass_zero_dc": highpass.sum().abs(),
        "even_shift_orthogonality": ortho,
    }


def _per_sample_sparsity_fp32(value: Any, lowpass: Any) -> dict[str, Any]:
    import torch

    coefficients = packet_coefficients(value, lowpass).flatten(1)
    source = value.float().flatten(1)
    coefficient_energy = coefficients.square().sum(dim=1).clamp_min(1e-30)
    source_energy = source.square().sum(dim=1).clamp_min(1e-30)
    rms = coefficients.square().mean(dim=1).sqrt().clamp_min(1e-30)
    normalized_l1 = coefficients.abs().mean(dim=1) / rms
    count = coefficients.shape[1]
    hoyer = (
        math.sqrt(count)
        - coefficients.abs().sum(dim=1) / coefficient_energy.sqrt()
    ) / (math.sqrt(count) - 1.0)
    small = (coefficients.abs() <= 0.1 * rms[:, None]).float().mean(dim=1)
    top_count = max(1, math.ceil(0.10 * count))
    top_energy = torch.topk(coefficients.square(), k=top_count, dim=1).values.sum(dim=1)
    height_half = LATENT_SHAPE[2] // 2
    width_half = VIEW_WIDTH // 2
    reshaped = packet_coefficients(value, lowpass).reshape(
        value.shape[0], value.shape[1], value.shape[2], LATENT_SHAPE[2], VIEW_COUNT, VIEW_WIDTH
    )
    terminal = {
        "LL": reshaped[:, :, :, :height_half, :, :width_half],
        "LH": reshaped[:, :, :, :height_half, :, width_half:],
        "HL": reshaped[:, :, :, height_half:, :, :width_half],
        "HH": reshaped[:, :, :, height_half:, :, width_half:],
    }
    reduce = tuple(range(1, reshaped.ndim))
    terminal_fraction = {
        name: tensor.square().sum(dim=tuple(range(1, tensor.ndim))) / coefficient_energy
        for name, tensor in terminal.items()
    }
    return {
        "normalized_l1": normalized_l1,
        "hoyer_sparsity": hoyer,
        "small_coefficient_fraction_at_0p1_rms": small,
        "top_10_percent_energy_fraction": top_energy / coefficient_energy,
        "coefficient_input_energy_ratio": coefficient_energy / source_energy,
        **{f"terminal_{name}_energy_fraction": fraction for name, fraction in terminal_fraction.items()},
    }


def _per_sample_sparsity(value: Any, lowpass: Any) -> dict[str, Any]:
    import torch

    with torch.autocast(device_type=value.device.type, enabled=False):
        return _per_sample_sparsity_fp32(value.float(), lowpass.float())


def _summarize_sample_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for name, values in metrics.items():
        array = values.detach().double().cpu()
        result[name] = {
            "mean": float(array.mean()),
            "std": float(array.std(unbiased=True)),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return result


def _bootstrap_sparsity_reduction(haar_values: Any, learned_values: Any) -> dict[str, Any]:
    import numpy as np

    control = np.asarray(haar_values.detach().double().cpu(), dtype=np.float64)
    candidate = np.asarray(learned_values.detach().double().cpu(), dtype=np.float64)
    if control.shape != (CAL_RANGE[1] - CAL_RANGE[0],) or candidate.shape != control.shape:
        raise WaveletError("held-out sparsity episode inventory differs")
    point = 100.0 * (control.mean() - candidate.mean()) / control.mean()
    rng = np.random.default_rng(BOOTSTRAP_SEED + 77)
    indexes = rng.integers(0, control.shape[0], size=(BOOTSTRAP_REPLICATES, control.shape[0]))
    draws = 100.0 * (
        control[indexes].mean(axis=1) - candidate[indexes].mean(axis=1)
    ) / control[indexes].mean(axis=1)
    return {
        "relative_reduction_percent": float(point),
        "paired_episode_bootstrap_95_percent": [
            float(value) for value in np.quantile(draws, (0.025, 0.975))
        ],
        "favorable_episode_fraction": float(np.mean(candidate < control)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _validated_basis_calibration_metrics(
    rows: Sequence[Mapping[str, Any]],
    registered_samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    import torch

    registered = {
        int(sample.get("clip_index", -1)): (
            sample.get("clip_id"),
            sample.get("episode_dir"),
        )
        for sample in registered_samples
    }
    expected = set(range(*CAL_RANGE))
    if (
        set(registered) != expected
        or len(registered) != len(registered_samples)
        or any(
            not isinstance(clip_id, str) or not isinstance(episode, str)
            for clip_id, episode in registered.values()
        )
    ):
        raise WaveletError("registered basis-calibration identity inventory differs")
    inventory = {}
    for row in rows:
        if row.get("schema") != BASIS_CAL_ROW_SCHEMA or not ladder.identity_valid(row):
            raise WaveletError("basis-calibration row identity/schema differs")
        clip_index = int(row.get("clip_index", -1))
        metrics = row.get("metrics")
        hashes = row.get("tensor_sha256")
        if (
            clip_index not in expected
            or clip_index in inventory
            or (row.get("clip_id"), row.get("episode_dir")) != registered[clip_index]
            or row.get("basis_was_frozen") is not True
            or int(row.get("history_frames_used", -1)) != 0
            or int(row.get("future_frames_used", -1)) != 2
            or int(row.get("vpm_forwards", -1)) != 0
            or row.get("residuals_hidden_states_or_outcomes_used") is not False
            or row.get("actions_loaded_by_dataset") is not True
            or row.get("actions_entered_basis_objective") is not False
            or row.get("prior_dev_opened") is not False
            or row.get("fresh_reserve_480_510_opened") is not False
            or row.get("validation_opened") is not False
            or row.get("protected_test_opened") is not False
            or not isinstance(metrics, Mapping)
            or set(metrics) != {"HAAR", "QMF"}
            or any(
                not isinstance(metrics[basis], Mapping)
                or set(metrics[basis]) != set(SPARSITY_METRICS)
                or any(
                    not math.isfinite(float(metrics[basis][name]))
                    for name in SPARSITY_METRICS
                )
                for basis in ("HAAR", "QMF")
            )
            or not isinstance(hashes, Mapping)
            or set(hashes) != {"clean_future_vae_latent"}
            or not isinstance(hashes["clean_future_vae_latent"], str)
            or len(hashes["clean_future_vae_latent"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in hashes["clean_future_vae_latent"]
            )
        ):
            raise WaveletError("basis-calibration row contract differs")
        for basis in ("HAAR", "QMF"):
            values = metrics[basis]
            terminal_sum = sum(
                float(values[name])
                for name in SPARSITY_METRICS
                if name.startswith("terminal_")
            )
            if (
                not 0.0 < float(values["normalized_l1"]) <= 1.0 + 1e-6
                or not -1e-5 <= float(values["hoyer_sparsity"]) <= 1.0 + 1e-5
                or not 0.0
                <= float(values["small_coefficient_fraction_at_0p1_rms"])
                <= 1.0
                or not 0.0
                <= float(values["top_10_percent_energy_fraction"])
                <= 1.0 + 1e-6
                or float(values["coefficient_input_energy_ratio"]) <= 0.0
                or any(
                    not 0.0 <= float(values[name]) <= 1.0 + 1e-6
                    for name in SPARSITY_METRICS
                    if name.startswith("terminal_")
                )
                or not math.isclose(terminal_sum, 1.0, rel_tol=0.0, abs_tol=2e-5)
            ):
                raise WaveletError("basis-calibration physical metric contract differs")
        inventory[clip_index] = row
    if set(inventory) != expected or len(rows) != len(expected):
        raise WaveletError("basis-calibration row inventory differs")
    return {
        basis: {
            name: torch.tensor(
                [float(inventory[index]["metrics"][basis][name]) for index in sorted(expected)],
                dtype=torch.float64,
            )
            for name in SPARSITY_METRICS
        }
        for basis in ("HAAR", "QMF")
    }


def recompute_basis_qualification(
    *,
    rows: Sequence[Mapping[str, Any]],
    registered_samples: Sequence[Mapping[str, Any]],
    learned_lowpass: Any,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    joined = _validated_basis_calibration_metrics(rows, registered_samples)
    sparsity_effect = _bootstrap_sparsity_reduction(
        joined["HAAR"]["normalized_l1"], joined["QMF"]["normalized_l1"]
    )
    learned_admissibility = basis_admissibility(learned_lowpass)
    fixed_haar = haar_lowpass(dtype=torch.float32)
    haar_admissibility = basis_admissibility(fixed_haar)
    filter_cosine = float(
        functional.cosine_similarity(
            learned_lowpass.detach().double().cpu(), fixed_haar.double(), dim=0
        ).abs()
    )
    learned_terminal_means = {
        name.removeprefix("terminal_").removesuffix("_energy_fraction"): float(
            joined["QMF"][name].mean()
        )
        for name in SPARSITY_METRICS
        if name.startswith("terminal_") and name.endswith("_energy_fraction")
    }
    heldout_energy_pass = all(
        float((metrics["coefficient_input_energy_ratio"] - 1.0).abs().max())
        <= BASIS_ENERGY_RATIO_TOLERANCE
        for metrics in joined.values()
    )
    gates = {
        "admissibility_reconstruction_energy_pass": bool(
            learned_admissibility["passed"] and haar_admissibility["passed"]
        ),
        "heldout_coefficient_energy_pass": heldout_energy_pass,
        "nontrivial_filter_change_pass": filter_cosine < BASIS_FILTER_HAAR_MAX_COSINE,
        "heldout_normalized_l1_sparsity_pass": bool(
            sparsity_effect["relative_reduction_percent"]
            >= BASIS_SPARSITY_IMPROVEMENT_PERCENT
            and sparsity_effect["paired_episode_bootstrap_95_percent"][0] > 0
        ),
        "nontrivial_terminal_energy_pass": all(
            value >= BASIS_TERMINAL_MIN_ENERGY_FRACTION
            for value in learned_terminal_means.values()
        ),
        "threshold_retention_pass": True,
        "basis_frozen_before_any_residual_target": True,
    }
    gates["all_passed"] = all(gates.values())
    return {
        "heldout_clean_receipt": {
            "clip_indices": list(CAL_RANGE),
            "episode_count": CAL_RANGE[1] - CAL_RANGE[0],
            "basis_was_updated": False,
            "metrics": {
                basis: _summarize_sample_metrics(metrics)
                for basis, metrics in joined.items()
            },
            "normalized_l1_learned_vs_haar": sparsity_effect,
            "learned_terminal_band_mean_energy_fraction": learned_terminal_means,
        },
        "learned_admissibility": learned_admissibility,
        "haar_admissibility": haar_admissibility,
        "learned_filter_absolute_cosine_with_haar": filter_cosine,
        "qualification_gates": gates,
        "calibration_rows_replayed": True,
    }


def fit_unsupervised_basis(clean_future_cpu: Any, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    if tuple(clean_future_cpu.shape) != (
        FIT_RANGE[1] - FIT_RANGE[0],
        LATENT_SHAPE[0],
        2,
        LATENT_SHAPE[2],
        LATENT_SHAPE[3],
    ):
        raise WaveletError("basis-prefit clean future latent inventory differs")
    clean = clean_future_cpu.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(BASIS_SEED)
    initial = (torch.randn((BASIS_FREE_ANGLES,), generator=generator) * 0.01).double()
    angles = torch.nn.Parameter(initial.to(device=device))
    optimizer = torch.optim.Adam(
        (angles,), lr=BASIS_LEARNING_RATE, weight_decay=0.0
    )
    permutation = torch.empty((0,), dtype=torch.long)
    cursor = 0
    trace = []
    for step in range(BASIS_OPT_STEPS):
        if cursor + BASIS_BATCH_SIZE > permutation.numel():
            permutation = torch.randperm(clean.shape[0], generator=generator)
            cursor = 0
        indexes = permutation[cursor : cursor + BASIS_BATCH_SIZE].to(device=device)
        cursor += BASIS_BATCH_SIZE
        batch = clean.index_select(0, indexes)
        lowpass = lattice_lowpass(angles)
        coefficients = packet_coefficients(batch, lowpass)
        sparse = coefficients.abs().mean() / batch.square().mean().sqrt().clamp_min(1e-12)
        regularizers = paper_regularizers(lowpass)
        loss = sparse + BASIS_REGULARIZER_WEIGHT * sum(regularizers.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(angles.grad).all():
            raise WaveletError("basis optimization produced non-finite gradient")
        optimizer.step()
        trace.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "normalized_l1": float(sparse.detach().cpu()),
                "sum_loss": float(regularizers["sum"].detach().cpu()),
                "highpass_zero_dc_loss": float(regularizers["highpass_zero_dc"].detach().cpu()),
                "even_shift_orthogonality_loss": float(
                    regularizers["even_shift_orthogonality"].detach().cpu()
                ),
                "free_angles": [float(value) for value in angles.detach().cpu().tolist()],
            }
        )
    frozen = lattice_lowpass(angles.detach()).cpu().float()
    return frozen, {
        "seed": BASIS_SEED,
        "steps": BASIS_OPT_STEPS,
        "batch_size": BASIS_BATCH_SIZE,
        "learning_rate": BASIS_LEARNING_RATE,
        "weight_decay": 0.0,
        "angle_and_admissibility_precision": "float64",
        "coefficient_objective_precision": "float32",
        "basis_optimizer_input_storage_precision": (
            "float32 after the pinned bf16 VAE encode"
        ),
        "initial_free_angles": [float(value) for value in initial.tolist()],
        "final_free_angles": [float(value) for value in angles.detach().cpu().tolist()],
        "trace": trace,
        "objective": "normalized terminal coefficient L1 plus paper sum/highpass/even-shift losses",
        "residual_predictability_or_outcomes_used": False,
        "threshold_gates_enabled": False,
        "clean_tensor_scope": "exact two future VAE latent frames [16,2,24,120]",
    }


def _synthetic_wavelet_contract() -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(BASIS_SEED + 1)
    value = torch.randn((2, *LATENT_SHAPE), generator=generator, dtype=torch.float32)
    learned = lattice_lowpass(torch.tensor([0.2, -0.1, 0.3]))
    receipts = {}
    decompositions = {
        "HAAR": haar_decompose(value, history_frames=HISTORY_LATENT_FRAMES),
        "QMF": qmf_decompose(
            value, lowpass=learned, history_frames=HISTORY_LATENT_FRAMES
        ),
    }
    for basis, bands in decompositions.items():
        receipts[basis] = partition_contract(
            value, bands, history_frames=HISTORY_LATENT_FRAMES
        )
    # View isolation: changing view 0 cannot change either other view.
    mutated = value.clone()
    mutated[..., :VIEW_WIDTH] += 7.0
    changed = {
        "HAAR": haar_decompose(mutated, history_frames=HISTORY_LATENT_FRAMES),
        "QMF": qmf_decompose(
            mutated, lowpass=learned, history_frames=HISTORY_LATENT_FRAMES
        ),
    }
    for basis in decompositions:
        for band in BASE_BANDS:
            if not torch.equal(
                decompositions[basis][band][..., VIEW_WIDTH:],
                changed[basis][band][..., VIEW_WIDTH:],
            ):
                raise WaveletError("wavelet transform crosses a camera-view boundary")
    return {
        "partition_contracts": receipts,
        "haar_basis": basis_admissibility(haar_lowpass()),
        "nontrivial_qmf_basis": basis_admissibility(learned),
        "latent_shape": list(LATENT_SHAPE),
        "future_shape_per_view": [LATENT_SHAPE[0], 2, LATENT_SHAPE[2], VIEW_WIDTH],
        "view_isolation_bit_exact": True,
        "spatial_lowpass_rank_per_view_frame_channel": (LATENT_SHAPE[2] // 2) * (VIEW_WIDTH // 2),
        "same_boundary_phase_nominal_support_and_rank": True,
        "temporal_basis": "exact length-two Haar; never learned",
    }


def _validate_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 512:
        raise WaveletError("canonical train manifest must contain 512 rows")
    episodes: set[str] = set()
    clips: set[str] = set()
    for index, row in enumerate(rows):
        episode = row.get("episode_dir")
        clip = row.get("clip_id")
        if (
            row.get("split") != "train"
            or int(row.get("auxiliary_index", -1)) != index
            or not isinstance(episode, str)
            or not isinstance(clip, str)
            or episode in episodes
            or clip in clips
        ):
            raise WaveletError(f"canonical manifest identity differs at row {index}")
        episodes.add(episode)
        clips.add(clip)
    ranges = {
        "excluded_history": list(EXCLUDED_RANGE),
        "nested_fit": list(FIT_RANGE),
        "calibration": list(CAL_RANGE),
        "prior_inspected_exploratory_development": list(DEV_RANGE),
        "forbidden_prior_consumed_reserve": list(RESERVE_RANGE),
        "excluded_constructor_probe": list(STRUCTURAL_EXCLUDED_RANGE),
    }
    covered = [index for start, stop in ranges.values() for index in range(start, stop)]
    if sorted(covered) != list(range(512)) or len(set(covered)) != 512:
        raise WaveletError("registered row ranges are not an exact partition")
    seed_sets = [set(FIT_NOISE_SEEDS), set(CAL_NOISE_SEEDS), set(DEV_NOISE_SEEDS)]
    if any(seed_sets[i] & seed_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise WaveletError("phase noise seeds overlap")
    prior_seeds = set(range(20260820, 20260920))
    if set().union(*seed_sets, {BASIS_SEED, PROJECTION_SEED, BOOTSTRAP_SEED}) & prior_seeds:
        raise WaveletError("exploratory seeds overlap prior ladder/frontier seeds")
    return {
        "ranges": ranges,
        "fit_doses": list(DOSES),
        "fit_noise_seeds": list(FIT_NOISE_SEEDS),
        "basis_seed": BASIS_SEED,
        "calibration_noise_seeds": [],
        "development_noise_seeds": list(DEV_NOISE_SEEDS),
        "all_episode_and_clip_identities_unique": True,
    }


def _validated_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    return frontier._validated_record(record, label)


def _validated_cache_record(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    return frontier._validated_cache_record(record, label)


def _prepare_registration(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    repo = ladder.canonical_directory(args.repo_root, "repository")
    observed_commit = ladder.git_output(repo, "rev-parse", "HEAD")
    if observed_commit != args.expected_source_commit or len(observed_commit) != 40:
        raise WaveletError("source commit differs")
    if ladder.git_output(repo, "status", "--porcelain", "--untracked-files=all"):
        raise WaveletError("source repository must be clean")
    output = Path(args.output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise WaveletError("fresh output already exists")

    parent_path = ladder.canonical_file(args.parent_registration, "parent ladder registration")
    runtime_path = ladder.canonical_file(args.runtime_record, "B200 runtime record")
    runtime = ladder.read_json(runtime_path)
    devices = runtime.get("gpus", {}).get("devices", [])
    if (
        runtime.get("gpus", {}).get("count") != 1
        or len(devices) != 1
        or "B200" not in str(devices[0].get("name", "")).upper()
        or devices[0].get("capability") != [10, 0]
    ):
        raise WaveletError("runtime receipt is not a one-B200 verification")
    parent = ladder.read_json(parent_path)
    if parent.get("schema") != ladder.SCHEMA or not ladder.identity_valid(parent) or "VPM" not in parent.get("lineage", {}):
        raise WaveletError("parent ladder registration is invalid")
    if args.vpm_snapshot_sha256 != parent["inputs"]["vpm_snapshot"].get("sha256"):
        raise WaveletError("VPM snapshot digest is not parent-bound")
    required_inputs = (
        "train_manifest",
        "train_cache_metadata",
        "vpm_resolved_config",
        "vpm_snapshot",
        "vpm_arm_manifest",
        "vpm_stage_manifest",
        "vpm_stage_outcome",
    )
    inputs = {key: _validated_record(parent["inputs"][key], key) for key in required_inputs}
    manifest_rows = ladder.read_jsonl(Path(inputs["train_manifest"]["path"]))
    partitions = _validate_manifest(manifest_rows)
    arrays = {
        key: _validated_cache_record(record, f"cache {key}")
        for key, record in parent["cache_arrays"].items()
    }
    if set(arrays) != {"target", "rgb", "actions"}:
        raise WaveletError("parent cache array inventory differs")
    vpm_spec = parent["lineage"]["VPM"].get("arm_spec", {})
    if vpm_spec.get("parameter_matched_control") is not True or vpm_spec.get("condition_mode") != "off":
        raise WaveletError("registered parent is not the VPM frontier")

    output.mkdir(parents=True, mode=0o700)
    output = output.resolve(strict=True)
    registration = ladder.identity_payload(
        {
            "schema": SCHEMA,
            "created_at_utc": ladder._now(),
            "status": "registered_before_fit_or_prior_dev_outcome_open",
            "repo_root": str(repo),
            "source_commit": observed_commit,
            "output_dir": str(output),
            "parent_registration": ladder.file_record(parent_path),
            "parent_registration_identity_sha256": parent["identity_sha256"],
            "runtime_verification": ladder.file_record(runtime_path),
            "inputs": inputs,
            "cache_arrays": arrays,
            "lineage": {"VPM": parent["lineage"]["VPM"]},
            "primary_source": {
                "title": "Frequency-Forcing: From Scaling-as-Time to Soft Frequency Guidance",
                "local_path_at_protocol_freeze": (
                    "/mnt/data1/ldu/research/Dual Video Diffusion/literature/"
                    "arxiv_2604.20902/main.tex"
                ),
                "sha256": FREQUENCY_FORCING_SOURCE_SHA256,
                "joint_generator_training_epochs_reported": 400,
                "provenance_scope": (
                    "protocol-freeze literature provenance only; not a runtime scientific input"
                ),
                "runtime_hash_recomputed": False,
                "source_content_bundled_in_repository": False,
            },
            "partitions": partitions,
            "frozen_design": {
                "bands": list(BANDS),
                "grouping_per_basis": {
                    "ST_COARSE": "T S_basis r",
                    "TEMPORAL_CHANGE": "(I-T) S_basis r",
                    "SPATIOTEMPORAL_DETAIL": "(I-S_basis) r",
                },
                "basis_prefit": {
                    "filter_length": BASIS_FILTER_LENGTH,
                    "free_lattice_angles": BASIS_FREE_ANGLES,
                    "parameterization": "four-rotation paraunitary lattice; total angle=-pi/4",
                    "objective": "paper-style L1 terminal sparsity plus sum/highpass/even-shift losses",
                    "optimizer": "Adam",
                    "seed": BASIS_SEED,
                    "steps": BASIS_OPT_STEPS,
                    "batch_size": BASIS_BATCH_SIZE,
                    "learning_rate": BASIS_LEARNING_RATE,
                    "weight_decay": 0.0,
                    "tensor_scope": "only two clean future VAE latent frames [16,2,24,120]",
                    "training_rows": list(FIT_RANGE),
                    "residuals_hidden_states_actions_or_outcomes_enter_objective_allowed": False,
                    "ordinary_dataset_arrays_loaded": ["rgb", "actions"],
                    "actions_enter_basis_objective": False,
                    "threshold_gates_enabled": False,
                    "terminal_band_gate_retention": 1.0,
                    "freeze_point": "fixed final update 512 before any residual-head target construction",
                },
                "basis_matching": {
                    "spatial_boundary": "periodic within each 40-column camera view",
                    "phase": "stride-two origin zero",
                    "nominal_support": BASIS_FILTER_LENGTH,
                    "packet_levels": 1,
                    "spatial_lowpass_rank_per_view_frame_channel": (LATENT_SHAPE[2] // 2) * (VIEW_WIDTH // 2),
                    "temporal_basis": "exact length-two Haar; not learned",
                    "haar_filter": [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                "arms": list(ARMS),
                "doses": list(DOSES),
                "capacity_rungs": list(CAPACITY_RUNGS),
                "ridge_lambdas": list(RIDGE_LAMBDAS),
                "projection_seed": PROJECTION_SEED,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "token_sample_count_per_clip_noise": TOKEN_SAMPLE_COUNT,
                "head_selection": "none; capacity 1024 and ridge 1.0 frozen from completed fixed-Haar calibration",
                "synthetic_wavelet_contract": _synthetic_wavelet_contract(),
                "thresholds": {
                    "max_abs_reconstruction_error": HAAR_MAX_ABS_TOLERANCE,
                    "relative_reconstruction_energy": HAAR_RELATIVE_ENERGY_TOLERANCE,
                    "normalized_pairwise_inner_product": HAAR_ORTHOGONALITY_TOLERANCE,
                    "postprojection_offband_relative_energy": LEAKAGE_TOLERANCE,
                    "basis_admissibility": BASIS_ADMISSIBILITY_TOLERANCE,
                    "basis_matrix_orthogonality": BASIS_MATRIX_ORTHOGONALITY_TOLERANCE,
                    "basis_reconstruction": BASIS_RECONSTRUCTION_TOLERANCE,
                    "basis_relative_reconstruction_energy": BASIS_RELATIVE_RECONSTRUCTION_ENERGY_TOLERANCE,
                    "basis_energy_ratio": BASIS_ENERGY_RATIO_TOLERANCE,
                    "basis_terminal_min_energy_fraction": BASIS_TERMINAL_MIN_ENERGY_FRACTION,
                    "basis_filter_haar_max_absolute_cosine": BASIS_FILTER_HAAR_MAX_COSINE,
                    "heldout_normalized_l1_improvement_percent": BASIS_SPARSITY_IMPROVEMENT_PERCENT,
                    "grouped_band_energy_fraction": [
                        GROUPED_BAND_MIN_ENERGY_FRACTION,
                        GROUPED_BAND_MAX_ENERGY_FRACTION,
                    ],
                },
            },
            "selected_manifest_rows": {
                role: [
                    {
                        "clip_index": index,
                        "clip_id": manifest_rows[index]["clip_id"],
                        "episode_dir": manifest_rows[index]["episode_dir"],
                    }
                    for index in range(start, stop)
                ]
                for role, (start, stop) in {
                    "fit": FIT_RANGE,
                    "calibration": CAL_RANGE,
                    "exploratory_development": DEV_RANGE,
                }.items()
            },
            "access_contract": {
                "vjepa_target_array_allowed": False,
                "teacher_calls": 0,
                "basis_prefit_uses_only_clean_training_vae_latents": True,
                "basis_prefit_actions_loaded_by_dataset": True,
                "basis_prefit_actions_enter_objective": False,
                "basis_prefit_future_latent_frames": 2,
                "basis_prefit_history_latent_frames": 0,
                "basis_frozen_before_residual_targets": True,
                "future_validity_enabled": False,
                "future_validity_max_retries": 0,
                "fresh_reserve_480_510_opened": False,
                "constructor_probe_511_opened": False,
                "validation_opened": False,
                "protected_test_opened": False,
                "development_opened_during_registration": False,
                "wandb_enabled": False,
            },
            "barred_range_provenance": {
                "rows_480_510": (
                    "already consumed by the completed direct-residual frontier; "
                    "barred and not opened by this qualification"
                ),
                "legacy_machine_field_semantics": (
                    "fresh_reserve_480_510_opened=false means only that this "
                    "qualification did not open those rows; it is not a global "
                    "freshness claim"
                ),
            },
            "claim_boundary": (
                "post-selection exploratory reuse of prior-inspected ABC-train rows "
                "416--479; one frozen VPM parent and linear token-local heads; no "
                "reserve, validation, protected-test, or paper claim"
            ),
        }
    )
    ladder.exclusive_json(output / "registration.json", registration)
    return output, registration


def _load_registration(output: Path) -> dict[str, Any]:
    value = ladder.read_json(output / "registration.json")
    if value.get("schema") != SCHEMA or not ladder.identity_valid(value):
        raise WaveletError("registration identity/schema differs")
    if Path(value["output_dir"]).resolve(strict=True) != output:
        raise WaveletError("registration output path differs")
    repo = ladder.canonical_directory(value["repo_root"], "registered repository")
    if (
        ladder.git_output(repo, "rev-parse", "HEAD") != value["source_commit"]
        or ladder.git_output(repo, "status", "--porcelain", "--untracked-files=all")
    ):
        raise WaveletError("registered source state changed")
    for key, label in (("runtime_verification", "runtime receipt"), ("parent_registration", "parent registration")):
        record = value.get(key, {})
        observed = _validated_record(record, label)
        if observed != {name: record[name] for name in ("path", "bytes", "sha256")}:
            raise WaveletError(f"registered {label} changed")
    return value


def _resolve_input(registration: Mapping[str, Any], key: str) -> Path:
    return Path(_validated_record(registration["inputs"][key], key)["path"])


class _IndexAuditedDataset:
    def __init__(self, dataset: Any, allowed: Sequence[int]) -> None:
        self.dataset = dataset
        self.allowed = frozenset(int(index) for index in allowed)
        self.access_counts = {index: 0 for index in self.allowed}
        if not self.allowed:
            raise WaveletError("dataset phase has no allowed rows")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        index = int(index)
        if index not in self.allowed:
            raise WaveletError(f"dataset phase attempted forbidden row {index}")
        self.access_counts[index] += 1
        return self.dataset[index]

    def assert_exact_accesses(self, expected_per_row: int) -> None:
        if any(count != expected_per_row for count in self.access_counts.values()):
            raise WaveletError(f"dataset row-access inventory differs: {self.access_counts}")


def _phase_dataset(config: Any, registration: Mapping[str, Any], index_range: tuple[int, int]) -> _IndexAuditedDataset:
    start, stop = index_range
    dataset = ladder._no_auxiliary_dataset(
        config,
        registration,
        validation_sample_indices=(start, stop - 1),
    )
    return _IndexAuditedDataset(dataset, range(start, stop))


def _future_rows(
    projected: Any,
    target_tokens: Mapping[str, Any],
    positions: Any,
    indexes: Sequence[int],
    noise_seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    feature_rows = []
    target_rows = {arm: [] for arm in ARMS}
    for local, clip_index in enumerate(indexes):
        selected = ladder.deterministic_token_subsample(
            positions,
            clip_index=int(clip_index),
            noise_seed=int(noise_seed),
            count=TOKEN_SAMPLE_COUNT,
        )
        feature_rows.append(projected[local, selected])
        for arm in ARMS:
            target_rows[arm].append(target_tokens[arm][local, selected])
    return torch.cat(feature_rows), {
        arm: torch.cat(values) for arm, values in target_rows.items()
    }


def _project_hidden_fp32(hidden: Any, projection: Any) -> Any:
    import torch

    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return hidden.float() @ projection.to(device=hidden.device, dtype=torch.float32)


def _difference_fp32(first: Any, second: Any) -> Any:
    import torch

    with torch.autocast(device_type=first.device.type, enabled=False):
        return first.float() - second.float()


def _sum_fp32(first: Any, second: Any) -> Any:
    import torch

    with torch.autocast(device_type=first.device.type, enabled=False):
        return first.float() + second.float()


def _update_sufficient_statistics_fp32(
    stats: dict[str, Any], features: Any, targets: Mapping[str, Any]
) -> None:
    import torch

    with torch.autocast(device_type=features.device.type, enabled=False):
        ladder.update_sufficient_statistics(
            stats,
            features.float(),
            {name: value.float() for name, value in targets.items()},
        )


def _predict_ridge_fp32(features: Any, head: Mapping[str, Any]) -> Any:
    import torch

    with torch.autocast(device_type=features.device.type, enabled=False):
        return ladder.predict_ridge_head(features.float(), head).float()


def _target_tokens(
    direct: Any,
    *,
    learned_lowpass: Any,
    patch_size: Sequence[int],
    history_frames: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    shuffled = torch.roll(direct, shifts=-1, dims=0)
    decompositions = {
        "HAAR": {
            "aligned": haar_decompose(direct, history_frames=history_frames),
            "shuffled": haar_decompose(shuffled, history_frames=history_frames),
        },
        "QMF": {
            "aligned": qmf_decompose(
                direct, lowpass=learned_lowpass, history_frames=history_frames
            ),
            "shuffled": qmf_decompose(
                shuffled, lowpass=learned_lowpass, history_frames=history_frames
            ),
        },
    }
    receipts = {
        basis: {
            alignment: partition_contract(
                direct if alignment == "aligned" else shuffled,
                bands,
                history_frames=history_frames,
            )
            for alignment, bands in aligned_and_shuffled.items()
        }
        for basis, aligned_and_shuffled in decompositions.items()
    }
    full = _future_only(direct, history_frames)
    shuffled_full = _future_only(shuffled, history_frames)
    tensors = {
        "FULL_DIRECT_ALIGNED": full,
        "FULL_DIRECT_SHUFFLED": shuffled_full,
        **{
            f"{basis}_{band}_{alignment.upper()}": value
            for basis, aligned_and_shuffled in decompositions.items()
            for alignment, bands in aligned_and_shuffled.items()
            for band, value in bands.items()
        },
    }
    tokenized = {arm: ladder.patchify_video(value, patch_size) for arm, value in tensors.items()}
    tokenized["ZERO"] = torch.zeros_like(tokenized["FULL_DIRECT_ALIGNED"])
    return tokenized, receipts


def _fit_phase(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise WaveletError("CUDA is required")
    _configure_shared_helpers()
    output, registration = _prepare_registration(args)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    model, config = ladder._load_model(
        _resolve_input(registration, "vpm_resolved_config"),
        _resolve_input(registration, "vpm_snapshot"),
        device,
    )
    target_path = Path(registration["cache_arrays"]["target"]["path"])
    expected_arrays = {
        Path(registration["cache_arrays"]["rgb"]["path"]).resolve(strict=True),
        Path(registration["cache_arrays"]["actions"]["path"]).resolve(strict=True),
    }

    # Phase A: encode only ordinary clean training videos. No VPM forward,
    # residual, hidden state, or outcome exists before the basis is frozen.
    clean_fit = []
    with ladder.NumpyTargetOpenGuard(target_path) as guard:
        dataset = _phase_dataset(config, registration, FIT_RANGE)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(FIT_RANGE[0], FIT_RANGE[1], 2):
                indexes = (start, start + 1)
                batch = ladder._batch_samples(dataset, indexes, device)
                video_clean = model._encode_clip(batch["rgb"])
                _reference, history_frames = model._history_reference(
                    batch["rgb"], video_clean.shape
                )
                future = video_clean.float()[:, :, int(history_frames) :]
                if tuple(future.shape[1:]) != (LATENT_SHAPE[0], 2, LATENT_SHAPE[2], LATENT_SHAPE[3]):
                    raise WaveletError("basis prefit did not isolate exactly two future VAE tokens")
                clean_fit.append(future.cpu())
        opened = {path.resolve(strict=True) for path in guard.opened}
    dataset.assert_exact_accesses(1)
    if opened != expected_arrays:
        raise WaveletError(f"basis-prefit input graph differs: {sorted(opened)}")
    clean_fit_cpu = torch.cat(clean_fit, dim=0)
    del dataset, clean_fit
    learned_lowpass, basis_optimization = fit_unsupervised_basis(clean_fit_cpu, device)
    del clean_fit_cpu
    gc.collect()

    # Phase B: frozen-basis clean-latent receipts only. Calibration rows never
    # select, update, or replace the final-step filter.
    cal_rows: list[dict[str, Any]] = []
    cal_registered = {
        int(sample["clip_index"]): sample
        for sample in registration["selected_manifest_rows"]["calibration"]
    }
    with ladder.NumpyTargetOpenGuard(target_path) as guard:
        dataset = _phase_dataset(config, registration, CAL_RANGE)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(CAL_RANGE[0], CAL_RANGE[1], 2):
                indexes = (start, start + 1)
                batch = ladder._batch_samples(dataset, indexes, device)
                video_clean = model._encode_clip(batch["rgb"])
                _reference, history_frames = model._history_reference(
                    batch["rgb"], video_clean.shape
                )
                future = video_clean.float()[:, :, int(history_frames) :]
                batch_metrics = {}
                for basis, lowpass in (
                    ("HAAR", haar_lowpass(device=device, dtype=torch.float32)),
                    ("QMF", learned_lowpass.to(device=device)),
                ):
                    metrics = _per_sample_sparsity(future, lowpass)
                    if set(metrics) != set(SPARSITY_METRICS):
                        raise WaveletError("basis-calibration metric schema differs")
                    batch_metrics[basis] = metrics
                for local, clip_index in enumerate(indexes):
                    identity = cal_registered[clip_index]
                    cal_rows.append(
                        ladder.identity_payload(
                            {
                                "schema": BASIS_CAL_ROW_SCHEMA,
                                "clip_index": clip_index,
                                "clip_id": identity["clip_id"],
                                "episode_dir": identity["episode_dir"],
                                "basis_was_frozen": True,
                                "history_frames_used": 0,
                                "future_frames_used": 2,
                                "vpm_forwards": 0,
                                "residuals_hidden_states_or_outcomes_used": False,
                                "actions_loaded_by_dataset": True,
                                "actions_entered_basis_objective": False,
                                "metrics": {
                                    basis: {
                                        name: float(values[name][local].detach().cpu())
                                        for name in SPARSITY_METRICS
                                    }
                                    for basis, values in batch_metrics.items()
                                },
                                "tensor_sha256": {
                                    "clean_future_vae_latent": ladder._safe_tensor_sha256(
                                        future[local : local + 1]
                                    )
                                },
                                "prior_dev_opened": False,
                                "fresh_reserve_480_510_opened": False,
                                "validation_opened": False,
                                "protected_test_opened": False,
                            }
                        )
                    )
        opened = {path.resolve(strict=True) for path in guard.opened}
    dataset.assert_exact_accesses(1)
    if opened != expected_arrays:
        raise WaveletError(f"basis-calibration input graph differs: {sorted(opened)}")
    del dataset
    cal_rows_path = output / "basis_calibration_rows.jsonl"
    ladder.exclusive_jsonl(cal_rows_path, cal_rows)
    sealed_cal_rows = ladder.read_jsonl(cal_rows_path)
    basis_weights = {
        "schema": BASIS_WEIGHT_SCHEMA,
        "source_commit": registration["source_commit"],
        "registration_identity_sha256": registration["identity_sha256"],
        "learned_lowpass": learned_lowpass,
        "haar_lowpass": haar_lowpass(),
        "threshold_gates_enabled": False,
        "terminal_band_gate_retention": torch.ones(4),
    }
    buffer = io.BytesIO()
    torch.save(basis_weights, buffer)
    ladder.exclusive_bytes(output / "learned_basis.pt", buffer.getvalue())
    basis_evidence = recompute_basis_qualification(
        rows=sealed_cal_rows,
        registered_samples=registration["selected_manifest_rows"]["calibration"],
        learned_lowpass=learned_lowpass,
    )
    basis_qualification = basis_evidence["qualification_gates"]
    basis_receipt = ladder.identity_payload(
        {
            "schema": BASIS_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "primary_source_protocol_freeze_sha256": FREQUENCY_FORCING_SOURCE_SHA256,
            "weights": ladder.file_record(output / "learned_basis.pt"),
            "calibration_rows": ladder.file_record(cal_rows_path),
            "optimization": basis_optimization,
            "training_access": {
                "clip_indices": list(FIT_RANGE),
                "exact_dataset_accesses_per_row": 1,
                "tensor_scope": "two future VAE latent frames [16,2,24,120]",
                "history_frames_used": 0,
                "vpm_forwards": 0,
                "residuals_hidden_states_or_outcomes_used": False,
                "dataset_arrays_loaded": ["rgb", "actions"],
                "actions_loaded_by_dataset": True,
                "actions_entered_basis_objective": False,
                "vjepa_target_array_opened": False,
            },
            "calibration_access": {
                "clip_indices": list(CAL_RANGE),
                "exact_dataset_accesses_per_row": 1,
                "basis_was_updated": False,
                "dataset_arrays_loaded": ["rgb", "actions"],
                "actions_loaded_by_dataset": True,
                "actions_entered_basis_objective": False,
            },
            **basis_evidence,
            "threshold_gates_enabled": False,
            "terminal_band_gate_retention": [1.0, 1.0, 1.0, 1.0],
            "basis_frozen_before_any_residual_target": True,
            "prior_dev_opened": False,
            "fresh_reserve_480_510_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
        }
    )
    ladder.exclusive_json(output / "basis.json", basis_receipt)

    # Phase C: only after the basis artifact is immutable do we construct VPM
    # residual targets and fit equal-capacity heads.
    projection_cpu = ladder.make_projection(
        int(model.forward_model.transformer.dim), max(CAPACITY_RUNGS), seed=PROJECTION_SEED
    )
    projection = projection_cpu.to(device=device)
    stats_by_dose: dict[int, dict[str, Any]] = {}
    grid = patch_size = None
    patch_dim = None
    fit_calls = 0
    fit_partitions = {"HAAR": _new_partition_summary(), "QMF": _new_partition_summary()}
    with ladder.NumpyTargetOpenGuard(target_path) as guard:
        dataset = _phase_dataset(config, registration, FIT_RANGE)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for noise_seed in FIT_NOISE_SEEDS:
                for start in range(FIT_RANGE[0], FIT_RANGE[1], 2):
                    indexes = (start, start + 1)
                    batch = ladder._batch_samples(dataset, indexes, device)
                    prepared = ladder._model_inputs(
                        model, batch, noise_seed=noise_seed, nfe=1, need_clean_auxiliary=False
                    )
                    off_velocity, hidden = ladder._off_prediction_and_hidden(model, prepared)
                    fit_calls += 1
                    observed = ladder._grid_contract(model, off_velocity, hidden)
                    if grid is None:
                        grid, patch_size, patch_dim = observed
                        stats_by_dose = {
                            dose: ladder._new_sufficient_statistics(max(CAPACITY_RUNGS), patch_dim, device)
                            for dose in DOSES
                        }
                    elif observed != (grid, patch_size, patch_dim):
                        raise WaveletError("Wan grid changed during fit")
                    clean_velocity = _difference_fp32(
                        prepared["initial_video"], prepared["video_clean"]
                    )
                    direct = _difference_fp32(clean_velocity, off_velocity)
                    targets, receipts = _target_tokens(
                        direct,
                        learned_lowpass=learned_lowpass.to(device=device),
                        patch_size=patch_size,
                        history_frames=prepared["history_frames"],
                    )
                    for basis in ("HAAR", "QMF"):
                        for alignment in ("aligned", "shuffled"):
                            _update_partition_summary(
                                fit_partitions[basis], receipts[basis][alignment]
                            )
                    projected = _project_hidden_fp32(hidden, projection)
                    positions = ladder.future_token_positions(
                        grid=grid,
                        patch_size=patch_size,
                        history_frames=prepared["history_frames"],
                        device=device,
                    )
                    x, y = _future_rows(projected, targets, positions, indexes, noise_seed)
                    for dose in DOSES:
                        if start < FIT_RANGE[0] + dose:
                            _update_sufficient_statistics_fp32(stats_by_dose[dose], x, y)
                    del batch, prepared, hidden, projected, clean_velocity, direct, targets, x, y
        opened = {path.resolve(strict=True) for path in guard.opened}
    dataset.assert_exact_accesses(len(FIT_NOISE_SEEDS))
    if opened != expected_arrays:
        raise WaveletError(f"fit input graph differs: {sorted(opened)}")
    if grid is None or patch_size is None or patch_dim is None:
        raise WaveletError("fit produced no sufficient statistics")
    for dose, stats in stats_by_dose.items():
        expected = dose * len(FIT_NOISE_SEEDS) * TOKEN_SAMPLE_COUNT
        if int(stats["count"]) != expected:
            raise WaveletError(f"fit token count differs at dose {dose}")
    del dataset
    gc.collect()
    selected = {
        "capacity": CAPACITY_RUNGS[0],
        "ridge_lambda": RIDGE_LAMBDAS[0],
        "selection": "prospectively inherited from completed fixed-Haar full-dose calibration",
    }
    heads = {
        str(dose): {
            arm: ladder.fit_ridge_head(
                stats_by_dose[dose],
                arm=arm,
                capacity=int(selected["capacity"]),
                ridge_lambda=float(selected["ridge_lambda"]),
            )
            for arm in ARMS
        }
        for dose in DOSES
    }
    counts = {
        int(head["parameter_count"])
        for dose_heads in heads.values()
        for head in dose_heads.values()
    }
    if len(counts) != 1:
        raise WaveletError("all dose/target heads must have equal capacity")
    if any(
        int(torch.count_nonzero(heads[str(dose)]["ZERO"][name])) != 0
        for dose in DOSES
        for name in ("weight", "mean_y")
    ):
        raise WaveletError("ZERO head is not exactly zero")
    weights = {
        "schema": WEIGHT_SCHEMA,
        "source_commit": registration["source_commit"],
        "registration_identity_sha256": registration["identity_sha256"],
        "projection_seed": PROJECTION_SEED,
        "projection": projection_cpu,
        "basis_identity_sha256": basis_receipt["identity_sha256"],
        "learned_lowpass": learned_lowpass,
        "grid": list(grid),
        "patch_size": list(patch_size),
        "patch_dim": int(patch_dim),
        "heads": heads,
        "selected_capacity": int(selected["capacity"]),
        "selected_ridge_lambda": float(selected["ridge_lambda"]),
    }
    buffer = io.BytesIO()
    torch.save(weights, buffer)
    ladder.exclusive_bytes(output / "adapter_weights.pt", buffer.getvalue())
    fit = ladder.identity_payload(
        {
            "schema": FIT_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "weights": ladder.file_record(output / "adapter_weights.pt"),
            "basis": ladder.file_record(output / "basis.json"),
            "basis_identity_sha256": basis_receipt["identity_sha256"],
            "optimization": {
                "clip_indices": list(FIT_RANGE),
                "fit_doses": list(DOSES),
                "dataset_array_validation_rows": [FIT_RANGE[0], FIT_RANGE[1] - 1],
                "exact_dataset_accesses_per_row": len(FIT_NOISE_SEEDS),
                "noise_seeds": list(FIT_NOISE_SEEDS),
                "sampled_future_tokens_by_dose": {
                    str(dose): int(stats_by_dose[dose]["count"]) for dose in DOSES
                },
                "off_wan_invocations": fit_calls,
                "teacher_calls": 0,
                "vjepa_target_array_opened": False,
                "basis_frozen_before_first_residual_target": True,
                "partition_contracts": {
                    basis: _finalize_partition_summary(summary)
                    for basis, summary in fit_partitions.items()
                },
            },
            "basis_calibration": {
                "clip_indices": list(CAL_RANGE),
                "dataset_array_validation_rows": [CAL_RANGE[0], CAL_RANGE[1] - 1],
                "exact_dataset_accesses_per_row": 1,
                "noise_seeds": [],
                "off_wan_invocations": 0,
                "teacher_calls": 0,
                "vjepa_target_array_opened": False,
                "basis_was_updated": False,
                "qualification_gates": basis_qualification,
            },
            "selected": selected,
            "head_parameter_count": counts.pop(),
            "all_heads_equal_parameter_count": True,
            "zero_head_exact_zero": True,
            "prior_dev_opened": False,
            "fresh_reserve_480_510_opened": False,
            "constructor_probe_511_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
        }
    )
    ladder.exclusive_json(output / "fit.json", fit)
    print(
        json.dumps(
            {
                "basis_identity_sha256": basis_receipt["identity_sha256"],
                "basis_qualified": basis_qualification["all_passed"],
                "fit_identity_sha256": fit["identity_sha256"],
                "selected": selected,
            },
            sort_keys=True,
        )
    )
    return 0


def _validated_basis(
    output: Path, registration: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    import torch

    basis_path = ladder.canonical_file(output / "basis.json", "basis receipt")
    basis = ladder.read_json(basis_path)
    training_access = basis.get("training_access", {})
    calibration_access = basis.get("calibration_access", {})
    if (
        basis.get("schema") != BASIS_SCHEMA
        or not ladder.identity_valid(basis)
        or basis.get("registration_identity_sha256") != registration["identity_sha256"]
        or basis.get("basis_frozen_before_any_residual_target") is not True
        or basis.get("threshold_gates_enabled") is not False
        or basis.get("terminal_band_gate_retention") != [1.0, 1.0, 1.0, 1.0]
        or basis.get("primary_source_protocol_freeze_sha256")
        != FREQUENCY_FORCING_SOURCE_SHA256
        or basis.get("prior_dev_opened") is not False
        or basis.get("fresh_reserve_480_510_opened") is not False
        or basis.get("validation_opened") is not False
        or basis.get("protected_test_opened") is not False
        or training_access.get("clip_indices") != list(FIT_RANGE)
        or int(training_access.get("exact_dataset_accesses_per_row", -1)) != 1
        or training_access.get("tensor_scope")
        != "two future VAE latent frames [16,2,24,120]"
        or int(training_access.get("history_frames_used", -1)) != 0
        or int(training_access.get("vpm_forwards", -1)) != 0
        or training_access.get("residuals_hidden_states_or_outcomes_used") is not False
        or training_access.get("dataset_arrays_loaded") != ["rgb", "actions"]
        or training_access.get("actions_loaded_by_dataset") is not True
        or training_access.get("actions_entered_basis_objective") is not False
        or training_access.get("vjepa_target_array_opened") is not False
        or calibration_access.get("clip_indices") != list(CAL_RANGE)
        or int(calibration_access.get("exact_dataset_accesses_per_row", -1)) != 1
        or calibration_access.get("basis_was_updated") is not False
        or calibration_access.get("dataset_arrays_loaded") != ["rgb", "actions"]
        or calibration_access.get("actions_loaded_by_dataset") is not True
        or calibration_access.get("actions_entered_basis_objective") is not False
    ):
        raise WaveletError("sealed basis receipt differs")
    weights_record = basis.get("weights", {})
    weights_path = ladder.canonical_file(weights_record.get("path", ""), "basis weights")
    rows_record = basis.get("calibration_rows", {})
    rows_path = ladder.canonical_file(
        rows_record.get("path", ""), "basis calibration rows"
    )
    for path, record, label in (
        (weights_path, weights_record, "basis weights"),
        (rows_path, rows_record, "basis calibration rows"),
    ):
        if (
            path.parent != output
            or path.stat().st_size != int(record.get("bytes", -1))
            or ladder.sha256_file(path) != record.get("sha256")
        ):
            raise WaveletError(f"{label} artifact differs")
    basis_weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    if (
        basis_weights.get("schema") != BASIS_WEIGHT_SCHEMA
        or basis_weights.get("source_commit") != registration["source_commit"]
        or basis_weights.get("registration_identity_sha256") != registration["identity_sha256"]
        or basis_weights.get("threshold_gates_enabled") is not False
        or not all(
            isinstance(basis_weights.get(name), torch.Tensor)
            and tuple(basis_weights[name].shape) == (BASIS_FILTER_LENGTH,)
            for name in ("learned_lowpass", "haar_lowpass")
        )
        or not torch.equal(basis_weights.get("haar_lowpass"), haar_lowpass())
        or not isinstance(basis_weights.get("terminal_band_gate_retention"), torch.Tensor)
        or tuple(basis_weights["terminal_band_gate_retention"].shape) != (4,)
        or not torch.equal(
            basis_weights["terminal_band_gate_retention"], torch.ones(4)
        )
    ):
        raise WaveletError("basis filter tensor inventory differs")
    rows = ladder.read_jsonl(rows_path)
    recomputed = recompute_basis_qualification(
        rows=rows,
        registered_samples=registration["selected_manifest_rows"]["calibration"],
        learned_lowpass=basis_weights["learned_lowpass"],
    )
    for key in (
        "heldout_clean_receipt",
        "learned_admissibility",
        "haar_admissibility",
        "learned_filter_absolute_cosine_with_haar",
        "qualification_gates",
        "calibration_rows_replayed",
    ):
        if basis.get(key) != recomputed[key]:
            raise WaveletError(f"basis replay differs for {key}")
    return basis, basis_weights, rows, recomputed


def _load_weights(
    output: Path, registration: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    fit = ladder.read_json(output / "fit.json")
    if fit.get("schema") != FIT_SCHEMA or not ladder.identity_valid(fit):
        raise WaveletError("fit identity/schema differs")
    if fit.get("registration_identity_sha256") != registration["identity_sha256"]:
        raise WaveletError("fit registration link differs")
    basis_record = fit.get("basis", {})
    basis_path = ladder.canonical_file(basis_record.get("path", ""), "basis receipt")
    if (
        basis_path != output / "basis.json"
        or basis_path.stat().st_size != int(basis_record.get("bytes", -1))
        or ladder.sha256_file(basis_path) != basis_record.get("sha256")
    ):
        raise WaveletError("fit basis receipt link differs")
    basis, basis_weights, _basis_rows, recomputed_basis = _validated_basis(
        output, registration
    )
    if (
        basis.get("identity_sha256") != fit.get("basis_identity_sha256")
        or fit.get("basis_calibration", {}).get("qualification_gates")
        != recomputed_basis["qualification_gates"]
    ):
        raise WaveletError("fit basis qualification link differs")
    record = fit.get("weights", {})
    path = ladder.canonical_file(record.get("path", ""), "adapter weights")
    if (
        path.parent != output
        or path.stat().st_size != int(record.get("bytes", -1))
        or ladder.sha256_file(path) != record.get("sha256")
    ):
        raise WaveletError("adapter weight artifact differs")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    selected = fit.get("selected", {})
    capacity = int(selected.get("capacity", -1))
    ridge_lambda = float(selected.get("ridge_lambda", float("nan")))
    heads = weights.get("heads", {})
    fit_access = fit.get("optimization", {})
    basis_calibration = fit.get("basis_calibration", {})
    expected_fit_calls = (
        ((FIT_RANGE[1] - FIT_RANGE[0] + 1) // 2) * len(FIT_NOISE_SEEDS)
    )
    expected_fit_tokens = (
        (FIT_RANGE[1] - FIT_RANGE[0])
        * len(FIT_NOISE_SEEDS)
        * TOKEN_SAMPLE_COUNT
    )
    if (
        weights.get("schema") != WEIGHT_SCHEMA
        or weights.get("source_commit") != registration["source_commit"]
        or weights.get("registration_identity_sha256") != registration["identity_sha256"]
        or set(heads) != {str(dose) for dose in DOSES}
        or any(set(dose_heads) != set(ARMS) for dose_heads in heads.values())
        or weights.get("projection_seed") != PROJECTION_SEED
        or int(weights.get("selected_capacity", -1)) != capacity
        or float(weights.get("selected_ridge_lambda", float("nan"))) != ridge_lambda
        or capacity not in CAPACITY_RUNGS
        or ridge_lambda not in RIDGE_LAMBDAS
        or fit_access.get("clip_indices") != list(FIT_RANGE)
        or fit_access.get("fit_doses") != list(DOSES)
        or fit_access.get("dataset_array_validation_rows")
        != [FIT_RANGE[0], FIT_RANGE[1] - 1]
        or int(fit_access.get("exact_dataset_accesses_per_row", -1))
        != len(FIT_NOISE_SEEDS)
        or fit_access.get("noise_seeds") != list(FIT_NOISE_SEEDS)
        or fit_access.get("sampled_future_tokens_by_dose")
        != {str(dose): expected_fit_tokens for dose in DOSES}
        or int(fit_access.get("off_wan_invocations", -1)) != expected_fit_calls
        or int(fit_access.get("teacher_calls", -1)) != 0
        or fit_access.get("vjepa_target_array_opened") is not False
        or fit_access.get("basis_frozen_before_first_residual_target") is not True
        or basis_calibration.get("clip_indices") != list(CAL_RANGE)
        or basis_calibration.get("dataset_array_validation_rows")
        != [CAL_RANGE[0], CAL_RANGE[1] - 1]
        or int(basis_calibration.get("exact_dataset_accesses_per_row", -1)) != 1
        or basis_calibration.get("noise_seeds") != []
        or int(basis_calibration.get("off_wan_invocations", -1)) != 0
        or int(basis_calibration.get("teacher_calls", -1)) != 0
        or basis_calibration.get("vjepa_target_array_opened") is not False
        or fit.get("zero_head_exact_zero") is not True
        or fit.get("prior_dev_opened") is not False
        or fit.get("fresh_reserve_480_510_opened") is not False
        or fit.get("constructor_probe_511_opened") is not False
        or fit.get("validation_opened") is not False
        or fit.get("protected_test_opened") is not False
        or set(fit_access.get("partition_contracts", {})) != {"HAAR", "QMF"}
        or any(
            receipt.get("passed") is not True
            or int(receipt.get("checked_batches", -1)) != 2 * expected_fit_calls
            for receipt in fit_access.get("partition_contracts", {}).values()
        )
        or basis_calibration.get("basis_was_updated") is not False
        or weights.get("basis_identity_sha256") != basis["identity_sha256"]
        or basis_weights.get("schema") != BASIS_WEIGHT_SCHEMA
        or basis_weights.get("registration_identity_sha256") != registration["identity_sha256"]
        or basis_weights.get("threshold_gates_enabled") is not False
        or not torch.equal(weights.get("learned_lowpass"), basis_weights.get("learned_lowpass"))
    ):
        raise WaveletError("sealed fit/weight contract differs")
    parameter_counts = set()
    for dose_heads in heads.values():
        for arm, head in dose_heads.items():
            if (
                head.get("arm") != arm
                or int(head.get("capacity", -1)) != capacity
                or float(head.get("ridge_lambda", float("nan"))) != ridge_lambda
            ):
                raise WaveletError("sealed head metadata differs")
            parameter_counts.add(int(head.get("parameter_count", -1)))
    if len(parameter_counts) != 1 or parameter_counts != {int(fit["head_parameter_count"])}:
        raise WaveletError("sealed heads are not parameter matched")
    return fit, dict(weights)


def _event_ms(start: Any, end: Any) -> float:
    end.synchronize()
    return float(start.elapsed_time(end))


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or np.any(array < 0):
        raise WaveletError("latency trace is empty, negative, or non-finite")
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _project_named_band(
    value: Any, *, target: str, learned_lowpass: Any, history_frames: int
) -> Any:
    if target.startswith("HAAR_"):
        base = target.removeprefix("HAAR_")
        return haar_decompose(value, history_frames=history_frames)[base]
    if target.startswith("QMF_"):
        base = target.removeprefix("QMF_")
        return qmf_decompose(
            value,
            lowpass=learned_lowpass.to(device=value.device),
            history_frames=history_frames,
        )[base]
    raise WaveletError(f"unknown named band {target}")


def _prediction_for_endpoint(
    *,
    endpoint: str,
    hidden: Any,
    projection: Any,
    weights: Mapping[str, Any],
    grid: Sequence[int],
    patch_size: Sequence[int],
    channels: int,
    history_frames: int,
) -> tuple[Any, Any]:
    """Return the deployable correction and unprojected head output."""

    if endpoint == "ZERO":
        dose = DOSES[-1]
        arm = "ZERO"
    else:
        prefix, arm = endpoint.split("_", 1)
        dose = int(prefix[1:])
    projected_hidden = _project_hidden_fp32(hidden, projection)
    tokens = _predict_ridge_fp32(
        projected_hidden, weights["heads"][str(dose)][arm]
    ).clone()
    positions = ladder.future_token_positions(
        grid=grid,
        patch_size=patch_size,
        history_frames=history_frames,
        device=hidden.device,
    )
    tokens[:, : int(positions[0])] = 0
    raw = ladder.unpatchify_tokens(
        tokens,
        grid=grid,
        patch_size=patch_size,
        channels=channels,
    ).float()
    raw[:, :, :history_frames] = 0
    target = arm.rsplit("_", 1)[0] if arm != "ZERO" else "ZERO"
    if target in BANDS:
        correction = _project_named_band(
            raw,
            target=target,
            learned_lowpass=weights["learned_lowpass"],
            history_frames=history_frames,
        )
    else:
        correction = raw
    return correction, raw


def _projection_leakage_fp32(
    raw: Any,
    correction: Any,
    *,
    endpoint: str,
    learned_lowpass: Any,
    history_frames: int,
) -> tuple[Any, Any]:
    target = next((band for band in BANDS if band in endpoint), None)
    if target is None:
        projected_twice = correction
    else:
        projected_twice = _project_named_band(
            correction,
            target=target,
            learned_lowpass=learned_lowpass,
            history_frames=history_frames,
        )
    reduce = tuple(range(1, raw.ndim))
    raw_energy = raw.square().sum(dim=reduce).clamp_min(1e-30)
    projected_energy = correction.square().sum(dim=reduce).clamp_min(1e-30)
    pre_leakage = (raw - correction).square().sum(dim=reduce) / raw_energy
    post_leakage = (correction - projected_twice).square().sum(dim=reduce) / projected_energy
    return pre_leakage, post_leakage


def _projection_leakage(
    raw: Any,
    correction: Any,
    *,
    endpoint: str,
    learned_lowpass: Any,
    history_frames: int,
) -> tuple[Any, Any]:
    import torch

    with torch.autocast(device_type=raw.device.type, enabled=False):
        return _projection_leakage_fp32(
            raw.float(),
            correction.float(),
            endpoint=endpoint,
            learned_lowpass=learned_lowpass.float(),
            history_frames=history_frames,
        )


def _per_sample_band_metrics_fp32(
    prediction: Any,
    target: Any,
    *,
    history_frames: int,
    pre_leakage: Any,
    post_leakage: Any,
) -> dict[str, Any]:
    import torch

    prediction = prediction.float()[:, :, history_frames:]
    target = target.float()[:, :, history_frames:]
    reduce = tuple(range(1, prediction.ndim))
    error = (prediction - target).square().sum(dim=reduce)
    energy = target.square().sum(dim=reduce).clamp_min(1e-30)
    count = math.prod(target.shape[1:])
    return {
        "target_r2": 1.0 - error / energy,
        "target_cosine": torch.nn.functional.cosine_similarity(
            prediction.flatten(1), target.flatten(1), dim=1, eps=1e-12
        ),
        "target_energy": energy / count,
        "target_rms": (energy / count).sqrt(),
        "prediction_energy": prediction.square().sum(dim=reduce) / count,
        "correction_rms": (prediction.square().sum(dim=reduce) / count).sqrt(),
        "preprojection_offband_relative_energy": pre_leakage,
        "postprojection_offband_relative_energy": post_leakage,
    }


def _per_sample_band_metrics(
    prediction: Any,
    target: Any,
    *,
    history_frames: int,
    pre_leakage: Any,
    post_leakage: Any,
) -> dict[str, Any]:
    import torch

    with torch.autocast(device_type=prediction.device.type, enabled=False):
        return _per_sample_band_metrics_fp32(
            prediction.float(),
            target.float(),
            history_frames=history_frames,
            pre_leakage=pre_leakage.float(),
            post_leakage=post_leakage.float(),
        )


def _standard_future_metrics_fp32(**kwargs: Any) -> dict[str, Any]:
    import torch

    velocity = kwargs["velocity"]
    with torch.autocast(device_type=velocity.device.type, enabled=False):
        converted = dict(kwargs)
        for name in (
            "velocity",
            "base_velocity",
            "direct_residual",
            "final",
            "clean",
        ):
            converted[name] = converted[name].float()
        return ladder._per_sample_future_metrics(**converted)


def _one_step_final_fp32(
    initial: Any, velocity: Any, reference: Any, history_frames: int
) -> Any:
    """Integrate the one-step flow in FP32 even inside model bf16 autocast."""
    import torch

    with torch.autocast(device_type=initial.device.type, enabled=False):
        return ladder._one_step_final(
            initial.float(), velocity.float(), reference.float(), history_frames
        )


def _evaluate_phase(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise WaveletError("CUDA is required")
    _configure_shared_helpers()
    output = ladder.canonical_directory(args.output_dir, "output directory")
    registration = _load_registration(output)
    fit, weights = _load_weights(output, registration)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    model, config = ladder._load_model(
        _resolve_input(registration, "vpm_resolved_config"),
        _resolve_input(registration, "vpm_snapshot"),
        device,
    )
    projection = weights["projection"].to(device=device)
    grid = tuple(int(value) for value in weights["grid"])
    patch_size = tuple(int(value) for value in weights["patch_size"])
    manifest = ladder.read_jsonl(_resolve_input(registration, "train_manifest"))
    target_path = Path(registration["cache_arrays"]["target"]["path"])
    expected_arrays = {
        Path(registration["cache_arrays"]["rgb"]["path"]).resolve(strict=True),
        Path(registration["cache_arrays"]["actions"]["path"]).resolve(strict=True),
    }
    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    adapter_timings = {endpoint: [] for endpoint in ENDPOINTS if endpoint != "VPM_OFF"}
    integration_timings = {endpoint: [] for endpoint in ENDPOINTS}
    decoder_timings = {endpoint: [] for endpoint in ENDPOINTS}
    preparation_timings: list[float] = []
    wan_timings: list[float] = []
    full_primary_timings: list[float] = []
    wan_invocations = 0
    dev_partitions = {"HAAR": _new_partition_summary(), "QMF": _new_partition_summary()}
    primary = endpoint_name(256, "FULL_DIRECT", "ALIGNED")

    with ladder.NumpyTargetOpenGuard(target_path) as guard:
        dataset = _phase_dataset(config, registration, DEV_RANGE)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for noise_seed in DEV_NOISE_SEEDS:
                for start in range(DEV_RANGE[0], DEV_RANGE[1], 2):
                    indexes = tuple(range(start, min(start + 2, DEV_RANGE[1])))
                    batch = ladder._batch_samples(dataset, indexes, device)
                    torch.cuda.synchronize()
                    full_started = time.perf_counter()
                    preparation_started = time.perf_counter()
                    try:
                        prepared = frontier._serving_inputs(model, batch, noise_seed=noise_seed)
                    except ladder.LadderError as exc:
                        raise WaveletError(str(exc)) from exc
                    torch.cuda.synchronize()
                    preparation_timings.append(1000.0 * (time.perf_counter() - preparation_started))
                    before = torch.cuda.Event(enable_timing=True)
                    after = torch.cuda.Event(enable_timing=True)
                    before.record()
                    off_velocity, hidden = ladder._off_prediction_and_hidden(model, prepared)
                    after.record()
                    wan_timings.append(_event_ms(before, after))
                    wan_invocations += 1
                    observed_grid, observed_patch, _ = ladder._grid_contract(model, off_velocity, hidden)
                    if observed_grid != grid or observed_patch != patch_size:
                        raise WaveletError("development Wan grid differs from fit")
                    if tuple(off_velocity.shape[1:]) != LATENT_SHAPE:
                        raise WaveletError("development VPM latent shape differs")

                    residuals: dict[str, Any] = {}
                    pre_leakage: dict[str, Any] = {}
                    post_leakage: dict[str, Any] = {}
                    velocities = {"VPM_OFF": off_velocity.float()}

                    def predict_endpoint(endpoint: str, *, audit_leakage: bool = True) -> Any:
                        before = torch.cuda.Event(enable_timing=True)
                        after = torch.cuda.Event(enable_timing=True)
                        before.record()
                        correction, raw_correction = _prediction_for_endpoint(
                            endpoint=endpoint,
                            hidden=hidden,
                            projection=projection,
                            weights=weights,
                            grid=grid,
                            patch_size=patch_size,
                            channels=off_velocity.shape[1],
                            history_frames=prepared["history_frames"],
                        )
                        after.record()
                        adapter_timings[endpoint].append(_event_ms(before, after))
                        residuals[endpoint] = correction
                        velocities[endpoint] = _sum_fp32(off_velocity, correction)
                        if audit_leakage:
                            pre, post = _projection_leakage(
                                raw_correction,
                                correction,
                                endpoint=endpoint,
                                learned_lowpass=weights["learned_lowpass"],
                                history_frames=prepared["history_frames"],
                            )
                            pre_leakage[endpoint] = pre
                            post_leakage[endpoint] = post
                        return raw_correction

                    finals: dict[str, Any] = {}
                    decoded_by_endpoint: dict[str, Any] = {}

                    def integrate_decode(endpoint: str) -> None:
                        before = torch.cuda.Event(enable_timing=True)
                        after = torch.cuda.Event(enable_timing=True)
                        before.record()
                        final = _one_step_final_fp32(
                            prepared["initial_video"],
                            velocities[endpoint],
                            prepared["reference"],
                            prepared["history_frames"],
                        )
                        after.record()
                        integration_timings[endpoint].append(_event_ms(before, after))
                        before = torch.cuda.Event(enable_timing=True)
                        after = torch.cuda.Event(enable_timing=True)
                        before.record()
                        decoded_full = model.rgb_tokenizer.decode_temporal(
                            final.to(batch["rgb"].dtype),
                            out_hw=(batch["rgb"].shape[-2], batch["rgb"].shape[-1]),
                        )
                        decoded = ladder._to_uint8_video(decoded_full[:, :, -model.num_future_frames :])
                        after.record()
                        decoder_timings[endpoint].append(_event_ms(before, after))
                        ladder._validate_decoded_horizon(
                            model_future_frames=model.num_future_frames,
                            decoded_frames=decoded_full.shape[2],
                            raw_frames=batch["rgb"].shape[1],
                            endpoint=endpoint,
                        )
                        finals[endpoint] = final
                        decoded_by_endpoint[endpoint] = decoded

                    # Materialize only the frozen primary endpoint before closing
                    # its synchronized end-to-end timer. Other control adapters
                    # must not contaminate this serving measurement.
                    primary_raw = predict_endpoint(primary, audit_leakage=False)
                    integrate_decode(primary)
                    torch.cuda.synchronize()
                    full_primary_timings.append(1000.0 * (time.perf_counter() - full_started))
                    primary_pre, primary_post = _projection_leakage(
                        primary_raw,
                        residuals[primary],
                        endpoint=primary,
                        learned_lowpass=weights["learned_lowpass"],
                        history_frames=prepared["history_frames"],
                    )
                    pre_leakage[primary] = primary_pre
                    post_leakage[primary] = primary_post
                    for endpoint in ENDPOINTS:
                        if endpoint not in ("VPM_OFF", primary):
                            predict_endpoint(endpoint)
                    zeros = torch.zeros(
                        (len(indexes),), device=device, dtype=torch.float32
                    )
                    residuals["VPM_OFF"] = torch.zeros_like(off_velocity, dtype=torch.float32)
                    pre_leakage["VPM_OFF"] = zeros
                    post_leakage["VPM_OFF"] = zeros
                    if int(torch.count_nonzero(residuals["ZERO"]).detach().cpu()) != 0:
                        raise WaveletError("ZERO correction is not exact zero")
                    for endpoint in ENDPOINTS:
                        if endpoint != primary:
                            integrate_decode(endpoint)
                    if not torch.equal(finals["VPM_OFF"], finals["ZERO"]):
                        raise WaveletError("ZERO final differs bitwise from VPM_OFF")

                    # Only now may the evaluator encode and inspect clean future video.
                    video_clean = model._encode_clip(batch["rgb"]).to(batch["rgb"].dtype)
                    if video_clean.shape != prepared["initial_video"].shape:
                        raise WaveletError("development clean-video grid differs")
                    video_target = _difference_fp32(
                        prepared["initial_video"], video_clean
                    )
                    direct_target = _difference_fp32(video_target, off_velocity)
                    target_bands = {
                        "HAAR": haar_decompose(
                            direct_target, history_frames=prepared["history_frames"]
                        ),
                        "QMF": qmf_decompose(
                            direct_target,
                            lowpass=weights["learned_lowpass"].to(device=device),
                            history_frames=prepared["history_frames"],
                        ),
                    }
                    for basis, bands in target_bands.items():
                        _update_partition_summary(
                            dev_partitions[basis],
                            partition_contract(
                                direct_target,
                                bands,
                                history_frames=prepared["history_frames"],
                            ),
                        )
                    full_target = _future_only(direct_target, prepared["history_frames"])
                    raw_video = batch["rgb"].permute(0, 2, 1, 3, 4)
                    raw_target_exact = raw_video[:, :, -model.num_future_frames :]
                    raw_history_exact = raw_video[
                        :, :, -(model.num_future_frames + 1) : -model.num_future_frames
                    ]
                    raw_target = ladder._to_uint8_video(raw_target_exact)
                    raw_history = ladder._to_uint8_video(raw_history_exact)
                    decoded_clean = model.rgb_tokenizer.decode_temporal(
                        video_clean,
                        out_hw=(batch["rgb"].shape[-2], batch["rgb"].shape[-1]),
                    )
                    vae_target = ladder._to_uint8_video(decoded_clean[:, :, -model.num_future_frames :])
                    vae_history = ladder._to_uint8_video(
                        decoded_clean[:, :, -(model.num_future_frames + 1) : -model.num_future_frames]
                    )

                    metrics_by_endpoint = {}
                    for endpoint in ENDPOINTS:
                        standard = _standard_future_metrics_fp32(
                            velocity=velocities[endpoint],
                            base_velocity=off_velocity,
                            direct_residual=direct_target,
                            final=finals[endpoint],
                            clean=video_clean,
                            decoded=decoded_by_endpoint[endpoint],
                            raw_target=raw_target,
                            raw_history=raw_history,
                            vae_target=vae_target,
                            vae_history=vae_history,
                            history_frames=prepared["history_frames"],
                        )
                        if endpoint in ("VPM_OFF", "ZERO") or "FULL_DIRECT" in endpoint:
                            named_target = full_target
                        else:
                            named = next(band for band in BANDS if band in endpoint)
                            basis, base = named.split("_", 1)
                            named_target = target_bands[basis][base]
                        band_metrics = _per_sample_band_metrics(
                            residuals[endpoint],
                            named_target,
                            history_frames=prepared["history_frames"],
                            pre_leakage=pre_leakage[endpoint],
                            post_leakage=post_leakage[endpoint],
                        )
                        metrics_by_endpoint[endpoint] = {**standard, **band_metrics}

                    common_hashes = {
                        "initial_video": prepared["initial_video"],
                        "actions": batch["actions"],
                        "morphology_index": batch["morphology_index"],
                        "action_control": prepared["z_control"],
                        "auxiliary_noise": prepared["initial_auxiliary"],
                        "raw_history_input": batch["rgb"][:, : model.num_history_frames],
                        "history_reference": prepared["reference"],
                        "latent_flow_target": video_target[:, :, prepared["history_frames"] :],
                        "raw_future_target": raw_target_exact,
                        "raw_history_boundary": raw_history_exact,
                        "scoring_future_uint8": raw_target,
                        "scoring_history_uint8": raw_history,
                    }
                    for endpoint in ENDPOINTS:
                        metrics = metrics_by_endpoint[endpoint]
                        for local, clip_index in enumerate(indexes):
                            rows.append(
                                ladder.identity_payload(
                                    {
                                        "schema": ROW_SCHEMA,
                                        "endpoint": endpoint,
                                        "clip_index": clip_index,
                                        "clip_id": manifest[clip_index]["clip_id"],
                                        "episode_dir": manifest[clip_index]["episode_dir"],
                                        "noise_seed": noise_seed,
                                        "wan_calls": 1,
                                        "teacher_calls": 0,
                                        "vjepa_target_array_opened": False,
                                        "future_target_entered_correction": False,
                                        "metrics": {
                                            key: float(value[local].detach().cpu())
                                            for key, value in metrics.items()
                                        },
                                        "tensor_sha256": {
                                            **{
                                                name: ladder._safe_tensor_sha256(value[local : local + 1])
                                                for name, value in common_hashes.items()
                                            },
                                            "correction": ladder._safe_tensor_sha256(
                                                residuals[endpoint][local : local + 1]
                                            ),
                                            "final_video": ladder._safe_tensor_sha256(
                                                finals[endpoint][local : local + 1]
                                            ),
                                        },
                                        "scoring_target_constructed_after_all_endpoints": True,
                                        "post_selection_exploratory_dev_opened": True,
                                        "fresh_reserve_480_510_opened": False,
                                        "constructor_probe_511_opened": False,
                                        "validation_opened": False,
                                        "protected_test_opened": False,
                                    }
                                )
                            )
                    timing_rows.append(
                        ladder.identity_payload(
                            {
                                "schema": TIMING_SCHEMA,
                                "batch_ordinal": len(timing_rows),
                                "noise_seed": noise_seed,
                                "clip_indices": list(indexes),
                                "batch_size": len(indexes),
                                "latency_ms": {
                                    "causal_preparation": preparation_timings[-1],
                                    "wan": wan_timings[-1],
                                    "adapter": {
                                        endpoint: adapter_timings[endpoint][-1]
                                        for endpoint in adapter_timings
                                    },
                                    "euler": {
                                        endpoint: integration_timings[endpoint][-1]
                                        for endpoint in ENDPOINTS
                                    },
                                    "decoder": {
                                        endpoint: decoder_timings[endpoint][-1]
                                        for endpoint in ENDPOINTS
                                    },
                                    "full_primary_endpoint": full_primary_timings[-1],
                                },
                            }
                        )
                    )
                    del (
                        batch,
                        prepared,
                        hidden,
                        velocities,
                        residuals,
                        finals,
                        decoded_by_endpoint,
                        video_clean,
                        video_target,
                        direct_target,
                        target_bands,
                    )
        opened = {path.resolve(strict=True) for path in guard.opened}
    dataset.assert_exact_accesses(len(DEV_NOISE_SEEDS))
    if opened != expected_arrays:
        raise WaveletError(f"development input graph differs: {sorted(opened)}")
    expected_rows = (DEV_RANGE[1] - DEV_RANGE[0]) * len(DEV_NOISE_SEEDS) * len(ENDPOINTS)
    expected_calls = ((DEV_RANGE[1] - DEV_RANGE[0] + 1) // 2) * len(DEV_NOISE_SEEDS)
    if len(rows) != expected_rows or wan_invocations != expected_calls:
        raise WaveletError("development row/Wan accounting differs")
    rows_path = output / "development_rows.jsonl"
    timing_path = output / "timing_rows.jsonl"
    ladder.exclusive_jsonl(rows_path, rows)
    ladder.exclusive_jsonl(timing_path, timing_rows)
    receipt = ladder.identity_payload(
        {
            "schema": ENDPOINT_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "fit_identity_sha256": fit["identity_sha256"],
            "rows": ladder.file_record(rows_path),
            "timing_rows": ladder.file_record(timing_path),
            "row_count": len(rows),
            "timing_row_count": len(timing_rows),
            "development_clip_range": list(DEV_RANGE),
            "development_noise_seeds": list(DEV_NOISE_SEEDS),
            "dataset_array_validation_rows": [DEV_RANGE[0], DEV_RANGE[1] - 1],
            "exact_dataset_accesses_per_row": len(DEV_NOISE_SEEDS),
            "shared_wan_invocations": wan_invocations,
            "wan_sample_calls": (DEV_RANGE[1] - DEV_RANGE[0]) * len(DEV_NOISE_SEEDS),
            "teacher_calls": 0,
            "vjepa_target_array_opened": False,
            "future_target_entered_correction": False,
            "scoring_target_constructed_after_all_endpoints": True,
            "zero_equals_off_bit_exact": True,
            "partition_contracts": {
                basis: _finalize_partition_summary(summary)
                for basis, summary in dev_partitions.items()
            },
            "basis_identity_sha256": weights["basis_identity_sha256"],
            "adapter_latency_ms_per_batch": {
                endpoint: _latency_summary(values) for endpoint, values in adapter_timings.items()
            },
            "wan_latency_ms_per_batch": _latency_summary(wan_timings),
            "causal_preparation_latency_ms_per_batch": _latency_summary(preparation_timings),
            "euler_latency_ms_per_batch": {
                endpoint: _latency_summary(values) for endpoint, values in integration_timings.items()
            },
            "decoder_latency_ms_per_batch": {
                endpoint: _latency_summary(values) for endpoint, values in decoder_timings.items()
            },
            "full_primary_endpoint": primary,
            "full_primary_endpoint_latency_ms_per_batch": _latency_summary(full_primary_timings),
            "latency_scope": (
                "primary synchronized wall time from resident observed RGB/actions through "
                "causal preparation, one shared Wan call, adapter, Euler, and decode; clean "
                "future target encoding, scoring, and post-projection leakage audit excluded"
            ),
            "post_selection_exploratory_dev_opened": True,
            "fresh_reserve_480_510_opened": False,
            "constructor_probe_511_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
        }
    )
    ladder.exclusive_json(output / "endpoint_complete.json", receipt)
    print(json.dumps({"endpoint_identity_sha256": receipt["identity_sha256"]}, sort_keys=True))
    return 0


def _bootstrap_relative(control: Any, candidate: Any, *, seed: int) -> dict[str, Any]:
    import numpy as np

    control = np.asarray(control, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    expected_shape = (DEV_RANGE[1] - DEV_RANGE[0], len(DEV_NOISE_SEEDS))
    if (
        control.shape != expected_shape
        or candidate.shape != expected_shape
        or not np.isfinite(control).all()
        or not np.isfinite(candidate).all()
        or control.mean() <= 0
    ):
        raise WaveletError("paired bootstrap input differs")
    point = 100.0 * (control.mean() - candidate.mean()) / control.mean()
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, control.shape[0], size=(BOOTSTRAP_REPLICATES, control.shape[0]))
    control_draw = control[indexes].mean(axis=(1, 2))
    candidate_draw = candidate[indexes].mean(axis=(1, 2))
    effects = 100.0 * (control_draw - candidate_draw) / control_draw
    return {
        "control_mean": float(control.mean()),
        "candidate_mean": float(candidate.mean()),
        "relative_improvement_percent": float(point),
        "paired_episode_bootstrap_95_percent": [
            float(value) for value in np.quantile(effects, (0.025, 0.975))
        ],
        "favorable_episode_fraction": float(
            np.mean(candidate.mean(axis=1) < control.mean(axis=1))
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    registered_samples: Sequence[Mapping[str, Any]],
    partition_contract_receipts: Mapping[str, Any],
    basis_qualification: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    registered = {
        int(sample.get("clip_index", -1)): (sample.get("clip_id"), sample.get("episode_dir"))
        for sample in registered_samples
    }
    if (
        set(registered) != set(range(*DEV_RANGE))
        or len(registered) != len(registered_samples)
        or any(not isinstance(clip, str) or not isinstance(episode, str) for clip, episode in registered.values())
    ):
        raise WaveletError("registered exploratory-development identity inventory differs")
    expected = {
        (endpoint, clip, seed)
        for endpoint in ENDPOINTS
        for clip in range(*DEV_RANGE)
        for seed in DEV_NOISE_SEEDS
    }
    inventory: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    required_metrics = {*ERROR_METRICS, *BAND_METRICS}
    paired_hashes = {
        "initial_video",
        "actions",
        "morphology_index",
        "action_control",
        "auxiliary_noise",
        "raw_history_input",
        "history_reference",
        "latent_flow_target",
        "raw_future_target",
        "raw_history_boundary",
        "scoring_future_uint8",
        "scoring_history_uint8",
    }
    required_hashes = {*paired_hashes, "correction", "final_video"}
    for row in rows:
        if row.get("schema") != ROW_SCHEMA or not ladder.identity_valid(row):
            raise WaveletError("development row identity/schema differs")
        key = (
            str(row.get("endpoint")),
            int(row.get("clip_index", -1)),
            int(row.get("noise_seed", -1)),
        )
        if key in inventory:
            raise WaveletError(f"duplicate development row: {key}")
        if (
            key not in expected
            or (row.get("clip_id"), row.get("episode_dir")) != registered.get(key[1])
            or int(row.get("wan_calls", -1)) != 1
            or int(row.get("teacher_calls", -1)) != 0
            or row.get("vjepa_target_array_opened") is not False
            or row.get("future_target_entered_correction") is not False
            or row.get("scoring_target_constructed_after_all_endpoints") is not True
            or row.get("post_selection_exploratory_dev_opened") is not True
            or row.get("fresh_reserve_480_510_opened") is not False
            or row.get("constructor_probe_511_opened") is not False
            or row.get("validation_opened") is not False
            or row.get("protected_test_opened") is not False
        ):
            raise WaveletError("development row violates identity/access/serving contract")
        metrics = row.get("metrics")
        hashes = row.get("tensor_sha256")
        if (
            not isinstance(metrics, Mapping)
            or not required_metrics.issubset(metrics)
            or any(not math.isfinite(float(metrics[name])) for name in required_metrics)
            or not isinstance(hashes, Mapping)
            or not required_hashes.issubset(hashes)
            or any(not isinstance(hashes[name], str) or len(hashes[name]) != 64 for name in required_hashes)
        ):
            raise WaveletError("development row metric/hash payload differs")
        inventory[key] = row
    if set(inventory) != expected:
        raise WaveletError("development row inventory differs")
    for clip in range(*DEV_RANGE):
        for seed in DEV_NOISE_SEEDS:
            for field in paired_hashes:
                if len({inventory[(endpoint, clip, seed)]["tensor_sha256"][field] for endpoint in ENDPOINTS}) != 1:
                    raise WaveletError(f"paired {field} differs for {(clip, seed)}")
            off = inventory[("VPM_OFF", clip, seed)]
            zero = inventory[("ZERO", clip, seed)]
            if (
                off["tensor_sha256"]["final_video"] != zero["tensor_sha256"]["final_video"]
                or off["tensor_sha256"]["correction"] != zero["tensor_sha256"]["correction"]
                or off["metrics"] != zero["metrics"]
            ):
                raise WaveletError("ZERO is not bit-exact to VPM_OFF")
    expected_batches = ((DEV_RANGE[1] - DEV_RANGE[0] + 1) // 2) * len(DEV_NOISE_SEEDS)
    if set(partition_contract_receipts) != {"HAAR", "QMF"}:
        raise WaveletError("development partition receipt inventory differs")
    for basis, receipt in partition_contract_receipts.items():
        if (
            receipt.get("passed") is not True
            or int(receipt.get("checked_batches", 0)) != expected_batches
            or float(receipt.get("max_abs_reconstruction_error", float("inf")))
            > HAAR_MAX_ABS_TOLERANCE
            or float(receipt.get("max_relative_reconstruction_energy", float("inf")))
            > HAAR_RELATIVE_ENERGY_TOLERANCE
            or float(receipt.get("max_abs_normalized_pairwise_inner_product", float("inf")))
            > HAAR_ORTHOGONALITY_TOLERANCE
            or receipt.get("nontrivial_grouped_band_energy") is not True
        ):
            raise WaveletError(f"development {basis} partition receipt differs")
    if set(basis_qualification) != {
        "admissibility_reconstruction_energy_pass",
        "heldout_coefficient_energy_pass",
        "nontrivial_filter_change_pass",
        "heldout_normalized_l1_sparsity_pass",
        "nontrivial_terminal_energy_pass",
        "threshold_retention_pass",
        "basis_frozen_before_any_residual_target",
        "all_passed",
    }:
        raise WaveletError("basis qualification gate inventory differs")

    clips = list(range(*DEV_RANGE))
    seeds = list(DEV_NOISE_SEEDS)

    def values(endpoint: str, metric: str) -> Any:
        return np.asarray(
            [
                [float(inventory[(endpoint, clip, seed)]["metrics"][metric]) for seed in seeds]
                for clip in clips
            ],
            dtype=np.float64,
        )

    aggregates = {
        endpoint: {
            metric: {
                "mean": float(values(endpoint, metric).mean()),
                "std": float(values(endpoint, metric).std(ddof=1)),
                "min": float(values(endpoint, metric).min()),
                "max": float(values(endpoint, metric).max()),
            }
            for metric in (*ERROR_METRICS, *BAND_METRICS)
        }
        for endpoint in ENDPOINTS
    }
    comparisons: dict[str, Any] = {}
    comparison_index = 0
    for dose in DOSES:
        full = endpoint_name(dose, "FULL_DIRECT", "ALIGNED")
        for target in TARGETS:
            aligned = endpoint_name(dose, target, "ALIGNED")
            shuffled = endpoint_name(dose, target, "SHUFFLED")
            controls = {"VPM_OFF": "VPM_OFF", "MATCHED_SHUFFLED": shuffled}
            if target != "FULL_DIRECT":
                controls["MATCHED_FULL_DIRECT"] = full
            if target.startswith("QMF_"):
                controls["MATCHED_HAAR"] = endpoint_name(
                    dose, f"HAAR_{target.removeprefix('QMF_')}", "ALIGNED"
                )
            for control_label, control in controls.items():
                label = f"{aligned}_vs_{control_label}"
                comparisons[label] = {
                    metric: _bootstrap_relative(
                        values(control, metric),
                        values(aligned, metric),
                        seed=BOOTSTRAP_SEED + 100 * comparison_index + metric_index,
                    )
                    for metric_index, metric in enumerate(ERROR_METRICS)
                }
                comparison_index += 1

    def effect(candidate: str, control_label: str, metric: str) -> Mapping[str, Any]:
        return comparisons[f"{candidate}_vs_{control_label}"][metric]

    band_gates = {}
    any_special = False
    for band in BANDS:
        dose_gates = {}
        for dose in DOSES:
            candidate = endpoint_name(dose, band, "ALIGNED")
            attribution_metrics = (
                "corrected_velocity_mse",
                "decoded_mse_unit_range",
                "decoded_temporal_difference_mse_unit_range",
            )
            attribution = all(
                effect(candidate, "MATCHED_SHUFFLED", metric)["relative_improvement_percent"] >= 3.0
                and effect(candidate, "MATCHED_SHUFFLED", metric)["paired_episode_bootstrap_95_percent"][0] > 0
                for metric in attribution_metrics
            ) and all(
                effect(candidate, "MATCHED_SHUFFLED", metric)["favorable_episode_fraction"] >= 0.60
                for metric in ("decoded_mse_unit_range", "decoded_temporal_difference_mse_unit_range")
            )
            off_quality = all(
                effect(candidate, "VPM_OFF", metric)["relative_improvement_percent"] >= 3.0
                and effect(candidate, "VPM_OFF", metric)["paired_episode_bootstrap_95_percent"][0] > 0
                for metric in ("decoded_mse_unit_range", "decoded_temporal_difference_mse_unit_range")
            )
            latent = effect(candidate, "VPM_OFF", "video_future_nmse")
            latent_guardrail = (
                latent["relative_improvement_percent"] >= 0.0
                and latent["paired_episode_bootstrap_95_percent"][0] > -1.0
            )
            matched_full_quality = all(
                effect(candidate, "MATCHED_FULL_DIRECT", metric)["relative_improvement_percent"] >= 1.0
                and effect(candidate, "MATCHED_FULL_DIRECT", metric)["paired_episode_bootstrap_95_percent"][0] > 0
                for metric in ("decoded_mse_unit_range", "decoded_temporal_difference_mse_unit_range")
            )
            matched_haar_quality = (
                band.startswith("QMF_")
                and all(
                    effect(candidate, "MATCHED_HAAR", metric)["relative_improvement_percent"] >= 1.0
                    and effect(candidate, "MATCHED_HAAR", metric)["paired_episode_bootstrap_95_percent"][0] > 0
                    for metric in (
                        "decoded_mse_unit_range",
                        "decoded_temporal_difference_mse_unit_range",
                    )
                )
            )
            predictable = (
                aggregates[candidate]["target_r2"]["mean"] > 0
                and aggregates[candidate]["target_cosine"]["mean"] > 0
            )
            leakage = aggregates[candidate]["postprojection_offband_relative_energy"]["max"] <= LEAKAGE_TOLERANCE
            passed = (
                band.startswith("QMF_")
                and basis_qualification["all_passed"] is True
                and attribution
                and off_quality
                and latent_guardrail
                and matched_full_quality
                and matched_haar_quality
                and predictable
                and leakage
            )
            dose_gates[str(dose)] = {
                "aligned_beats_matched_shuffled": attribution,
                "aligned_beats_vpm_off": off_quality,
                "latent_guardrail": latent_guardrail,
                "matched_full_direct_quality_superiority": matched_full_quality,
                "matched_rank_haar_quality_superiority": matched_haar_quality,
                "basis_qualification_pass": basis_qualification["all_passed"] is True,
                "positive_band_r2_and_cosine": predictable,
                "postprojection_leakage_pass": leakage,
                "all_passed": passed,
            }
            any_special = any_special or passed
        band_gates[band] = {
            "doses": dose_gates,
            "any_dose_passed": any(value["all_passed"] for value in dose_gates.values()),
        }

    # Predeclared heterogeneity receipt only. Quartiles are computed separately
    # from each VPM-off metric and cannot select an arm or satisfy a gate.
    off_error_quartiles = {}
    for metric in ERROR_METRICS:
        off = values("VPM_OFF", metric)
        order = np.argsort(off.mean(axis=1), kind="stable")
        quartiles = []
        for ordinal, selected_indexes in enumerate(np.array_split(order, 4), start=1):
            selected_clips = [clips[int(index)] for index in selected_indexes]
            endpoint_effects = {}
            for band in BANDS:
                aligned_name = endpoint_name(256, band, "ALIGNED")
                shuffled_name = endpoint_name(256, band, "SHUFFLED")
                aligned_values = values(aligned_name, metric)[selected_indexes]
                shuffled_values = values(shuffled_name, metric)[selected_indexes]
                off_values = off[selected_indexes]

                def relative(control: Any, candidate: Any) -> float:
                    return float(100.0 * (control.mean() - candidate.mean()) / control.mean())

                endpoint_effects[band] = {
                    "aligned_vs_off_percent": relative(off_values, aligned_values),
                    "shuffled_vs_off_percent": relative(off_values, shuffled_values),
                    "aligned_vs_shuffled_percent": relative(shuffled_values, aligned_values),
                }
            quartiles.append(
                {
                    "quartile": ordinal,
                    "clip_indices": selected_clips,
                    "vpm_off_mean": float(off[selected_indexes].mean()),
                    "effects": endpoint_effects,
                }
            )
        off_error_quartiles[metric] = quartiles
    global_gates = {
        "zero_equals_off_bit_exact": True,
        "one_wan_call_zero_teacher_feature_calls": True,
        "haar_and_qmf_partition_reconstruction_orthogonality_pass": True,
        "basis_qualification_pass": basis_qualification["all_passed"] is True,
        "post_selection_dev_only": True,
        "prior_consumed_rows_480_510_unopened_by_this_run": True,
        "constructor_probe_511_unopened": True,
        "validation_and_protected_test_unopened": True,
    }
    passed = any_special and all(global_gates.values())
    if passed:
        decision = "EXPLORATORY_LEARNED_QMF_SPECIAL"
    elif basis_qualification["all_passed"] is True:
        decision = "EXPLORATORY_BASIS_ADAPTS_BUT_NO_CAUSAL_ADVANTAGE"
    else:
        decision = "EXPLORATORY_NO_LEARNED_QMF_QUALIFICATION"
    return ladder.identity_payload(
        {
            "schema": ANALYSIS_SCHEMA,
            "created_at_utc": ladder._now(),
            "development_episode_count": len(clips),
            "noise_seeds_per_episode": len(seeds),
            "aggregates": aggregates,
            "comparisons": comparisons,
            "off_error_quartile_heterogeneity_not_for_selection": off_error_quartiles,
            "basis_qualification": dict(basis_qualification),
            "band_gates": band_gates,
            "global_gates": {**global_gates, "all_passed": passed},
            "decision": decision,
            "claim_boundary": (
                "post-selection exploratory reuse of prior-inspected ABC-train rows 416--479; "
                "one frozen VPM parent and linear token-local heads; no reserve, validation, "
                "protected-test, generalization, or paper claim"
            ),
        }
    )


def _validated_endpoint(
    output: Path,
    registration: Mapping[str, Any],
    fit: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    endpoint = ladder.read_json(output / "endpoint_complete.json")
    expected_rows = (DEV_RANGE[1] - DEV_RANGE[0]) * len(DEV_NOISE_SEEDS) * len(ENDPOINTS)
    expected_batches = ((DEV_RANGE[1] - DEV_RANGE[0] + 1) // 2) * len(DEV_NOISE_SEEDS)
    if (
        endpoint.get("schema") != ENDPOINT_SCHEMA
        or not ladder.identity_valid(endpoint)
        or endpoint.get("registration_identity_sha256") != registration["identity_sha256"]
        or endpoint.get("fit_identity_sha256") != fit["identity_sha256"]
        or endpoint.get("basis_identity_sha256") != fit.get("basis_identity_sha256")
        or int(endpoint.get("row_count", -1)) != expected_rows
        or int(endpoint.get("timing_row_count", -1)) != expected_batches
        or endpoint.get("development_clip_range") != list(DEV_RANGE)
        or endpoint.get("dataset_array_validation_rows") != [DEV_RANGE[0], DEV_RANGE[1] - 1]
        or int(endpoint.get("exact_dataset_accesses_per_row", -1)) != len(DEV_NOISE_SEEDS)
        or int(endpoint.get("shared_wan_invocations", -1)) != expected_batches
        or int(endpoint.get("wan_sample_calls", -1))
        != (DEV_RANGE[1] - DEV_RANGE[0]) * len(DEV_NOISE_SEEDS)
        or endpoint.get("development_noise_seeds") != list(DEV_NOISE_SEEDS)
        or int(endpoint.get("teacher_calls", -1)) != 0
        or endpoint.get("vjepa_target_array_opened") is not False
        or endpoint.get("future_target_entered_correction") is not False
        or endpoint.get("scoring_target_constructed_after_all_endpoints") is not True
        or endpoint.get("zero_equals_off_bit_exact") is not True
        or endpoint.get("post_selection_exploratory_dev_opened") is not True
        or endpoint.get("fresh_reserve_480_510_opened") is not False
        or endpoint.get("constructor_probe_511_opened") is not False
        or endpoint.get("validation_opened") is not False
        or endpoint.get("protected_test_opened") is not False
    ):
        raise WaveletError("endpoint receipt contract differs")
    paths = []
    for key, label in (("rows", "development rows"), ("timing_rows", "timing rows")):
        record = endpoint.get(key, {})
        path = ladder.canonical_file(record.get("path", ""), label)
        if (
            path.parent != output
            or path.stat().st_size != int(record.get("bytes", -1))
            or ladder.sha256_file(path) != record.get("sha256")
        ):
            raise WaveletError(f"{label} artifact differs")
        paths.append(path)
    timing = ladder.read_jsonl(paths[1])
    expected_timing = [
        (seed, list(range(start, min(start + 2, DEV_RANGE[1]))))
        for seed in DEV_NOISE_SEEDS
        for start in range(DEV_RANGE[0], DEV_RANGE[1], 2)
    ]
    if len(timing) != len(expected_timing):
        raise WaveletError("timing row count differs")
    for ordinal, (row, (seed, clips)) in enumerate(zip(timing, expected_timing, strict=True)):
        latency = row.get("latency_ms", {})
        if (
            row.get("schema") != TIMING_SCHEMA
            or not ladder.identity_valid(row)
            or int(row.get("batch_ordinal", -1)) != ordinal
            or int(row.get("noise_seed", -1)) != seed
            or row.get("clip_indices") != clips
            or int(row.get("batch_size", -1)) != len(clips)
            or set(latency.get("adapter", {})) != set(ENDPOINTS) - {"VPM_OFF"}
            or set(latency.get("euler", {})) != set(ENDPOINTS)
            or set(latency.get("decoder", {})) != set(ENDPOINTS)
            or any(
                not math.isfinite(float(value)) or float(value) < 0
                for group in (latency["adapter"], latency["euler"], latency["decoder"])
                for value in group.values()
            )
            or any(
                not math.isfinite(float(latency.get(name, float("nan"))))
                or float(latency[name]) < 0
                for name in (
                    "causal_preparation",
                    "wan",
                    "full_primary_endpoint",
                )
            )
        ):
            raise WaveletError("timing row inventory differs")
    observed_timing_summaries = {
        "adapter_latency_ms_per_batch": {
            name: _latency_summary(
                [float(row["latency_ms"]["adapter"][name]) for row in timing]
            )
            for name in set(ENDPOINTS) - {"VPM_OFF"}
        },
        "wan_latency_ms_per_batch": _latency_summary(
            [float(row["latency_ms"]["wan"]) for row in timing]
        ),
        "causal_preparation_latency_ms_per_batch": _latency_summary(
            [float(row["latency_ms"]["causal_preparation"]) for row in timing]
        ),
        "euler_latency_ms_per_batch": {
            name: _latency_summary(
                [float(row["latency_ms"]["euler"][name]) for row in timing]
            )
            for name in ENDPOINTS
        },
        "decoder_latency_ms_per_batch": {
            name: _latency_summary(
                [float(row["latency_ms"]["decoder"][name]) for row in timing]
            )
            for name in ENDPOINTS
        },
        "full_primary_endpoint_latency_ms_per_batch": _latency_summary(
            [float(row["latency_ms"]["full_primary_endpoint"]) for row in timing]
        ),
    }
    if (
        endpoint.get("full_primary_endpoint")
        != endpoint_name(256, "FULL_DIRECT", "ALIGNED")
        or any(endpoint.get(name) != value for name, value in observed_timing_summaries.items())
    ):
        raise WaveletError("endpoint timing summary replay differs")
    return endpoint, paths[0], paths[1]


def _analyze_phase(args: argparse.Namespace) -> int:
    output = ladder.canonical_directory(args.output_dir, "output directory")
    registration = _load_registration(output)
    fit, _weights = _load_weights(output, registration)
    endpoint, rows_path, _timing_path = _validated_endpoint(output, registration, fit)
    rows = ladder.read_jsonl(rows_path)
    _basis, _basis_weights, _basis_rows, recomputed_basis = _validated_basis(
        output, registration
    )
    analysis = analyze_rows(
        rows,
        registered_samples=registration["selected_manifest_rows"]["exploratory_development"],
        partition_contract_receipts=endpoint["partition_contracts"],
        basis_qualification=recomputed_basis["qualification_gates"],
    )
    analysis = ladder.identity_payload(
        {
            **analysis,
            "registration_identity_sha256": registration["identity_sha256"],
            "fit_identity_sha256": fit["identity_sha256"],
            "endpoint_identity_sha256": endpoint["identity_sha256"],
        }
    )
    ladder.exclusive_json(output / "analysis.json", analysis)
    complete = ladder.identity_payload(
        {
            "schema": COMPLETE_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "fit_identity_sha256": fit["identity_sha256"],
            "endpoint_identity_sha256": endpoint["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "analysis": ladder.file_record(output / "analysis.json"),
            "decision": analysis["decision"],
            "development_row_count": len(rows),
            "post_selection_exploratory_dev_opened": True,
            "fresh_reserve_480_510_opened": False,
            "constructor_probe_511_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
        }
    )
    ladder.exclusive_json(output / "run_complete.json", complete)
    print(json.dumps({"analysis_identity_sha256": analysis["identity_sha256"], "decision": analysis["decision"]}, sort_keys=True))
    return 0


def _audit_phase(args: argparse.Namespace) -> int:
    output = ladder.canonical_directory(args.output_dir, "output directory")
    registration = _load_registration(output)
    fit, _weights = _load_weights(output, registration)
    endpoint, rows_path, timing_path = _validated_endpoint(output, registration, fit)
    analysis = ladder.read_json(output / "analysis.json")
    complete = ladder.read_json(output / "run_complete.json")
    if (
        analysis.get("schema") != ANALYSIS_SCHEMA
        or not ladder.identity_valid(analysis)
        or complete.get("schema") != COMPLETE_SCHEMA
        or not ladder.identity_valid(complete)
        or complete.get("decision") != analysis.get("decision")
        or complete.get("analysis_identity_sha256") != analysis.get("identity_sha256")
        or complete.get("fresh_reserve_480_510_opened") is not False
        or complete.get("constructor_probe_511_opened") is not False
        or complete.get("validation_opened") is not False
        or complete.get("protected_test_opened") is not False
    ):
        raise WaveletError("completion/analysis identity contract differs")
    analysis_record = complete.get("analysis", {})
    analysis_path = ladder.canonical_file(analysis_record.get("path", ""), "analysis")
    if (
        analysis_path != output / "analysis.json"
        or analysis_path.stat().st_size != int(analysis_record.get("bytes", -1))
        or ladder.sha256_file(analysis_path) != analysis_record.get("sha256")
    ):
        raise WaveletError("analysis artifact receipt differs")
    rows = ladder.read_jsonl(rows_path)
    _basis, _basis_weights, _basis_rows, recomputed_basis = _validated_basis(
        output, registration
    )
    recomputed = analyze_rows(
        rows,
        registered_samples=registration["selected_manifest_rows"]["exploratory_development"],
        partition_contract_receipts=endpoint["partition_contracts"],
        basis_qualification=recomputed_basis["qualification_gates"],
    )
    for key in (
        "aggregates",
        "comparisons",
        "off_error_quartile_heterogeneity_not_for_selection",
        "basis_qualification",
        "band_gates",
        "global_gates",
        "decision",
    ):
        if recomputed[key] != analysis.get(key):
            raise WaveletError(f"audited {key} differs")
    artifacts = (
        "registration.json",
        "basis.json",
        "basis_calibration_rows.jsonl",
        "learned_basis.pt",
        "fit.json",
        "adapter_weights.pt",
        "development_rows.jsonl",
        "timing_rows.jsonl",
        "endpoint_complete.json",
        "analysis.json",
        "run_complete.json",
    )
    audit = ladder.identity_payload(
        {
            "schema": AUDIT_SCHEMA,
            "created_at_utc": ladder._now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "completion_identity_sha256": complete["identity_sha256"],
            "decision": analysis["decision"],
            "artifact_sha256": {
                name: ladder.sha256_file(ladder.canonical_file(output / name, name))
                for name in artifacts
            },
            "row_inventory_recomputed": True,
            "paired_stored_tensor_hash_equality_revalidated": True,
            "timing_row_inventory_and_summary_recomputed": True,
            "zero_noop_stored_hash_and_metric_equality_revalidated": True,
            "partition_contract_aggregate_receipts_revalidated": True,
            "basis_qualification_recomputed_from_sealed_receipt": True,
            "basis_calibration_row_identity_inventory_and_bootstrap_recomputed": True,
            "role_scoped_row_access_receipts_validated": True,
            "post_selection_exploratory_dev_opened": True,
            "fresh_reserve_480_510_opened": False,
            "constructor_probe_511_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
            "status": "PASS",
        }
    )
    ladder.exclusive_json(output / "audit.json", audit)
    print(json.dumps({"audit_identity_sha256": audit["identity_sha256"], "decision": analysis["decision"], "status": "PASS"}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--repo-root", required=True)
    fit.add_argument("--expected-source-commit", required=True)
    fit.add_argument("--parent-registration", required=True)
    fit.add_argument("--vpm-snapshot-sha256", required=True)
    fit.add_argument("--runtime-record", required=True)
    fit.add_argument("--output-dir", required=True)
    for command in ("evaluate", "analyze", "audit"):
        commands.add_parser(command).add_argument("--output-dir", required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "fit":
        return _fit_phase(args)
    if args.command == "evaluate":
        return _evaluate_phase(args)
    if args.command == "analyze":
        return _analyze_phase(args)
    if args.command == "audit":
        return _audit_phase(args)
    raise WaveletError("unsupported command")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WaveletError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
