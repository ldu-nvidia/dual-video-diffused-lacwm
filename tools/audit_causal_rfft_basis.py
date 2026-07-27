#!/usr/bin/env python3
"""Audit the exact linear algebra of ``PerViewCausalRFFT``.

This is a CPU-only, read-only analysis tool.  It establishes the relationship
between the production length-four RFFT state and a same-shape causal
time-domain packing.  Optional old-pilot ``tf_clean`` tensors are read without
modification.  The report is created outside the input artifact directories
with exclusive-create semantics.

For the LACWM diffusion clock, ``sigma=1`` is noise and ``sigma=0`` is clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.modeling.dual_diffusion.time_frequency import PerViewCausalRFFT


SCHEMA_VERSION = 1
SIGMA_CONVENTION = "sigma=1 is noise; sigma=0 is clean"


class AuditError(RuntimeError):
    """Raised when an input would make the audit ambiguous or unsafe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    return result.stdout.strip()


def packing_labels(window_size: int) -> list[str]:
    """Return the exact real-channel order used by ``PerViewCausalRFFT``."""
    if window_size < 2:
        raise ValueError("window_size must be at least two")
    labels = ["X0.real"]
    for frequency in range(1, window_size // 2 + 1):
        labels.append(f"X{frequency}.real")
        is_nyquist = window_size % 2 == 0 and frequency == window_size // 2
        if not is_nyquist:
            labels.append(f"X{frequency}.imag")
    if len(labels) != window_size:
        raise RuntimeError("RFFT packing must have exactly window_size real values")
    return labels


def packed_rfft_matrix(
    window_size: int = 4, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Return ``A`` such that production packed coefficients satisfy ``y=A@x``."""
    basis = torch.eye(window_size, dtype=dtype)
    columns = []
    for input_index in range(window_size):
        spectrum = torch.fft.rfft(basis[input_index], norm="ortho")
        parts = [spectrum[0].real]
        for frequency in range(1, spectrum.shape[0]):
            parts.append(spectrum[frequency].real)
            is_nyquist = (
                window_size % 2 == 0 and frequency == window_size // 2
            )
            if not is_nyquist:
                parts.append(spectrum[frequency].imag)
        columns.append(torch.stack(parts))
    return torch.stack(columns, dim=1)


def parseval_channel_scales(
    window_size: int = 4, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Scales that make the nonredundant real RFFT packing orthonormal."""
    scales = []
    for label in packing_labels(window_size):
        frequency = int(label[1 : label.index(".")])
        is_endpoint = frequency == 0 or (
            window_size % 2 == 0 and frequency == window_size // 2
        )
        scales.append(1.0 if is_endpoint else math.sqrt(2.0))
    return torch.tensor(scales, dtype=dtype)


def original_causal_operator(
    matrix: torch.Tensor, temporal_bins: int = 4
) -> torch.Tensor:
    """Map the original ``1+(Fp-1)*N`` frames to all ``Fp*N`` coefficients."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if temporal_bins < 1:
        raise ValueError("temporal_bins must be positive")
    window_size = matrix.shape[0]
    input_frames = 1 + (temporal_bins - 1) * window_size
    operator = matrix.new_zeros((temporal_bins * window_size, input_frames))
    operator[:window_size, 0] = matrix @ matrix.new_ones(window_size)
    for temporal_bin in range(1, temporal_bins):
        row_start = temporal_bin * window_size
        column_start = 1 + (temporal_bin - 1) * window_size
        operator[
            row_start : row_start + window_size,
            column_start : column_start + window_size,
        ] = matrix
    return operator


def causal_time_domain_packing(
    transform: PerViewCausalRFFT, video: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack causal windows into the production TF tensor shape before the RFFT.

    Returns:
        ``(packed_windows, pooled_video)``.  Both use width-stacked views.
        ``packed_windows`` is ``[B,C*N,Fp,H,W_total]`` and has channel order
        ``[channel, within_window_time]``, exactly matching the coefficient
        tensor's ``[channel, packed_RFFT_component]`` order.
    """
    if video.ndim != 5:
        raise ValueError("video must have shape [B,T,C,H,W]")
    batch, frames, channels, _, _ = video.shape
    window_size = transform.window_size
    if (frames - 1) % window_size:
        raise ValueError(f"frames must satisfy T=1+k*{window_size}")

    pooled = transform._split_and_pool(video)
    _, num_views, _, _, height, view_width = pooled.shape
    temporal_bins = 1 + (frames - 1) // window_size
    anchor = pooled[:, :, :1].expand(-1, -1, window_size, -1, -1, -1)
    tail = pooled[:, :, 1:].reshape(
        batch,
        num_views,
        temporal_bins - 1,
        window_size,
        channels,
        height,
        view_width,
    )
    windows = torch.cat([anchor.unsqueeze(2), tail], dim=2)
    packed_views = windows.permute(0, 1, 4, 3, 2, 5, 6).reshape(
        batch,
        num_views,
        channels * window_size,
        temporal_bins,
        height,
        view_width,
    )
    packed = packed_views.permute(0, 2, 3, 4, 1, 5).reshape(
        batch,
        channels * window_size,
        temporal_bins,
        height,
        num_views * view_width,
    )
    pooled_video = pooled.permute(0, 2, 3, 4, 1, 5).reshape(
        batch,
        frames,
        channels,
        height,
        num_views * view_width,
    )
    return packed, pooled_video


def apply_component_matrix(
    packed: torch.Tensor, matrix: torch.Tensor, *, num_views: int
) -> torch.Tensor:
    """Apply an ``N x N`` matrix to each packed channel/window/view location."""
    if packed.ndim != 5:
        raise ValueError("packed tensor must have shape [B,C*N,Fp,H,W]")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    batch, packed_channels, temporal_bins, height, total_width = packed.shape
    window_size = matrix.shape[1]
    if packed_channels % window_size:
        raise ValueError("packed channels must be divisible by matrix width")
    if total_width % num_views:
        raise ValueError("packed width must be divisible by num_views")
    channels = packed_channels // window_size
    view_width = total_width // num_views
    views = packed.reshape(
        batch,
        packed_channels,
        temporal_bins,
        height,
        num_views,
        view_width,
    ).permute(0, 4, 1, 2, 3, 5)
    components = views.reshape(
        batch,
        num_views,
        channels,
        window_size,
        temporal_bins,
        height,
        view_width,
    )
    transformed = torch.einsum(
        "on,bvcnfhw->bvcofhw",
        matrix.to(device=packed.device, dtype=packed.dtype),
        components,
    )
    transformed_views = transformed.reshape(
        batch,
        num_views,
        channels * matrix.shape[0],
        temporal_bins,
        height,
        view_width,
    )
    return transformed_views.permute(0, 2, 3, 4, 1, 5).reshape(
        batch,
        channels * matrix.shape[0],
        temporal_bins,
        height,
        total_width,
    )


def _float_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1)]


def _matrix_list(tensor: torch.Tensor) -> list[list[float]]:
    return [
        [float(value) for value in row]
        for row in tensor.detach().cpu().tolist()
    ]


def _component_statistics(
    tensor: torch.Tensor, *, window_size: int
) -> dict[str, Any]:
    batch, packed_channels, temporal_bins, height, width = tensor.shape
    if packed_channels % window_size:
        raise AuditError("tf_clean channels are not divisible by window_size")
    channels = packed_channels // window_size
    grouped = tensor.reshape(
        batch, channels, window_size, temporal_bins, height, width
    ).movedim(2, 0)
    flattened = grouped.reshape(window_size, -1)
    by_bin = grouped.permute(0, 3, 1, 2, 4, 5).reshape(
        window_size, temporal_bins, -1
    )
    return {
        "labels": packing_labels(window_size),
        "sample_count_per_component": int(flattened.shape[1]),
        "mean": _float_list(flattened.mean(dim=1)),
        "population_variance": _float_list(flattened.var(dim=1, unbiased=False)),
        "mean_square": _float_list(flattened.square().mean(dim=1)),
        "rms": _float_list(flattened.square().mean(dim=1).sqrt()),
        "mean_square_by_temporal_bin": [
            _float_list(row.square().mean(dim=1)) for row in by_bin
        ],
        "rms_by_temporal_bin": [
            _float_list(row.square().mean(dim=1).sqrt()) for row in by_bin
        ],
    }


def audit_tf_clean_artifacts(
    artifact_paths: Sequence[Path],
    *,
    matrix: torch.Tensor,
    num_views: int,
) -> dict[str, Any] | None:
    if not artifact_paths:
        return None
    tensors = []
    provenance = []
    expected_shape = None
    for raw_path in artifact_paths:
        path = raw_path.expanduser().resolve(strict=True)
        if not path.is_file():
            raise AuditError(f"artifact is not a regular file: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            if "tf_clean" not in handle.keys():
                raise AuditError(f"artifact lacks tf_clean: {path}")
            tensor = handle.get_tensor("tf_clean")
        if tensor.ndim != 5 or not tensor.is_floating_point():
            raise AuditError(f"invalid tf_clean tensor in {path}: {tensor.shape}")
        if expected_shape is None:
            expected_shape = tuple(tensor.shape[1:])
        elif tuple(tensor.shape[1:]) != expected_shape:
            raise AuditError("all tf_clean tensors must have the same non-batch shape")
        tensors.append(tensor.to(torch.float64))
        provenance.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "tf_clean_shape": list(tensor.shape),
                "tf_clean_dtype": str(tensor.dtype),
            }
        )

    coefficients = torch.cat(tensors, dim=0)
    window_size = matrix.shape[0]
    inverse = torch.linalg.inv(matrix)
    time_packed = apply_component_matrix(
        coefficients, inverse, num_views=num_views
    )
    reprojected = apply_component_matrix(
        time_packed, matrix, num_views=num_views
    )
    scales = parseval_channel_scales(
        window_size, dtype=coefficients.dtype
    ).reshape(1, 1, window_size, 1, 1, 1)
    grouped = coefficients.reshape(
        coefficients.shape[0],
        coefficients.shape[1] // window_size,
        window_size,
        *coefficients.shape[2:],
    )
    weighted = grouped * scales
    coefficient_energy = coefficients.square().sum()
    weighted_energy = weighted.square().sum()
    time_energy = time_packed.square().sum()
    anchor_non_dc = grouped[:, :, 1:, 0]
    future_statistics = _component_statistics(
        coefficients[:, :, 2:], window_size=window_size
    )
    future_mean_square = torch.tensor(
        future_statistics["mean_square"], dtype=torch.float64
    )

    return {
        "source": "old-pilot tf_clean tensors; inputs opened read-only",
        "artifacts": provenance,
        "combined_shape": list(coefficients.shape),
        "component_statistics": _component_statistics(
            coefficients, window_size=window_size
        ),
        "future_temporal_bins": [2, 3],
        "future_component_statistics": future_statistics,
        "future_dc_mean_square_over_each_dynamic_component": _float_list(
            future_mean_square[0] / future_mean_square[1:]
        ),
        "future_standard_gaussian_noise_variance_over_clean_mean_square": (
            _float_list(future_mean_square.reciprocal())
        ),
        "round_trip_coefficient_max_abs_error": float(
            (reprojected - coefficients).abs().max()
        ),
        "unweighted_coefficient_energy_over_time_packed_energy": float(
            coefficient_energy / time_energy
        ),
        "sqrt2_weighted_coefficient_energy_over_time_packed_energy": float(
            weighted_energy / time_energy
        ),
        "anchor_non_dc_max_abs": float(anchor_non_dc.abs().max()),
        "clean_coefficient_mean_square_over_standard_gaussian_noise_variance": (
            float(coefficients.square().mean())
        ),
        "interpretation": (
            "The clean-state channel statistics are descriptive of this fixed "
            "eight-rank pilot probe, not a training-split whitening estimate."
        ),
    }


def synthetic_audit(
    *, matrix: torch.Tensor, seed: int, num_views: int
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    video = torch.randn(
        2, 13, 3, 4, 12, dtype=torch.float64, generator=generator
    )
    transform = PerViewCausalRFFT(
        num_views=num_views,
        output_size=(4, 12),
        window_size=matrix.shape[0],
        pad_multiple=None,
        normalization="none",
    )
    coefficients = transform(video)
    time_packed, pooled_video = causal_time_domain_packing(transform, video)
    projected = apply_component_matrix(
        time_packed, matrix, num_views=num_views
    )
    reconstructed = transform.inverse(coefficients)

    isolated = torch.zeros_like(video)
    input_view_width = isolated.shape[-1] // num_views
    isolated[..., :input_view_width] = torch.randn(
        isolated[..., :input_view_width].shape,
        dtype=isolated.dtype,
        generator=generator,
    )
    isolated_coefficients = transform(isolated)
    output_view_width = isolated_coefficients.shape[-1] // num_views
    view_l1 = [
        float(
            isolated_coefficients[
                ..., view * output_view_width : (view + 1) * output_view_width
            ]
            .abs()
            .sum()
        )
        for view in range(num_views)
    ]

    white = torch.randn(
        200_000, matrix.shape[1], dtype=matrix.dtype, generator=generator
    )
    white_coefficients = white @ matrix.T
    empirical_variances = white_coefficients.var(dim=0, unbiased=False)
    return {
        "seed": seed,
        "video_shape": list(video.shape),
        "coefficient_shape": list(coefficients.shape),
        "same_shape_time_packing_shape": list(time_packed.shape),
        "production_vs_matrix_max_abs_error": float(
            (coefficients - projected).abs().max()
        ),
        "inverse_vs_spatially_pooled_video_max_abs_error": float(
            (reconstructed - pooled_video).abs().max()
        ),
        "view_isolation_output_l1_by_view": view_l1,
        "view_isolation_exact": bool(
            view_l1[0] > 0.0 and all(value == 0.0 for value in view_l1[1:])
        ),
        "iid_unit_variance_time_input": {
            "draws": int(white.shape[0]),
            "empirical_output_population_variance": _float_list(
                empirical_variances
            ),
            "theoretical_output_population_variance": _float_list(
                (matrix @ matrix.T).diag()
            ),
        },
    }


def build_report(
    *,
    artifact_paths: Sequence[Path] = (),
    seed: int = 20260726,
    num_views: int = 3,
    window_size: int = 4,
) -> dict[str, Any]:
    if window_size != 4:
        raise AuditError(
            "this audited claim set is pinned to the implemented production "
            "length-four representation"
        )
    matrix = packed_rfft_matrix(window_size)
    scales = parseval_channel_scales(window_size)
    weighted_matrix = torch.diag(scales) @ matrix
    singular_values = torch.linalg.svdvals(matrix)
    original_operator = original_causal_operator(matrix, temporal_bins=4)
    original_singular_values = torch.linalg.svdvals(original_operator)
    inverse_noise_covariance = torch.linalg.inv(matrix) @ torch.linalg.inv(matrix).T

    implementation = REPO_ROOT / "robot_wm/modeling/dual_diffusion/time_frequency.py"
    tool = Path(__file__).resolve()
    status = _git_value("status", "--porcelain")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "per_view_causal_rfft_basis_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "representation": "PerViewCausalRFFT",
            "window_size": window_size,
            "temporal_bins": 4,
            "original_frames": 13,
            "num_views": num_views,
            "normalization": "none",
            "clock_convention": SIGMA_CONVENTION,
        },
        "source_provenance": {
            "repository_root": str(REPO_ROOT),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "git_dirty": bool(status),
            "implementation_path": str(implementation),
            "implementation_sha256": _sha256_file(implementation),
            "audit_tool_path": str(tool),
            "audit_tool_sha256": _sha256_file(tool),
        },
        "exact_relationship": {
            "time_domain_packing": (
                "For each input channel, view, spatial location, and causal "
                "temporal bin, x=[x0,x1,x2,x3]. The output has the same shape "
                "with that within-window axis replaced by packed RFFT channels."
            ),
            "coefficient_order": packing_labels(window_size),
            "formula": (
                "y=A x, y=[Re(X0),Re(X1),Im(X1),Re(X2)], with an "
                "orthonormal complex RFFT and no sqrt(2) packing factors"
            ),
            "matrix_A": _matrix_list(matrix),
            "closed_form_A": (
                "(1/2)*[[1,1,1,1],[1,0,-1,0],"
                "[0,-1,0,1],[1,-1,1,-1]]"
            ),
            "inverse_exists": True,
            "consequence": (
                "At the spatially pooled resolution, every four-frame tail "
                "window and its packed RFFT contain exactly the same information."
            ),
        },
        "linear_algebra": {
            "packed_window_singular_values_descending": _float_list(
                singular_values
            ),
            "packed_window_condition_number_2": float(
                singular_values.max() / singular_values.min()
            ),
            "packed_window_rank": int(torch.linalg.matrix_rank(matrix)),
            "sqrt2_parseval_scales": _float_list(scales),
            "weighted_matrix_gram": _matrix_list(
                weighted_matrix.T @ weighted_matrix
            ),
            "weighted_matrix_condition_number_2": float(
                torch.linalg.cond(weighted_matrix)
            ),
            "original_13_frame_to_16_coefficient_shape": list(
                original_operator.shape
            ),
            "original_operator_rank": int(
                torch.linalg.matrix_rank(original_operator)
            ),
            "original_operator_singular_values_descending": _float_list(
                original_singular_values
            ),
            "original_operator_nonzero_condition_number_2": float(
                original_singular_values.max() / original_singular_values.min()
            ),
            "anchor_expansion": (
                "The first frame is repeated four times, so its coefficient "
                "bin is [2*x0,0,0,0]. This creates three structural zero output "
                "directions but does not lose the anchor."
            ),
        },
        "energy_and_noise": {
            "exact_unweighted_energy_formula": (
                "||y||^2=|X0|^2+|Re X1|^2+|Im X1|^2+|X2|^2"
            ),
            "exact_time_energy_formula": (
                "||x||^2=|X0|^2+2*(|Re X1|^2+|Im X1|^2)+|X2|^2"
            ),
            "unweighted_energy_ratio_bounds": [0.5, 1.0],
            "iid_unit_variance_time_input_output_covariance": _matrix_list(
                matrix @ matrix.T
            ),
            "iid_unit_variance_time_input_expected_mean_output_variance": float(
                (matrix @ matrix.T).diag().mean()
            ),
            "iid_unit_variance_packed_coefficient_noise_implied_time_covariance": (
                _matrix_list(inverse_noise_covariance)
            ),
            "interpretation": (
                "Without sqrt(2), the real and imaginary k=1 channels have "
                "half the variance/energy of DC and Nyquist under white "
                "time-domain input. Conversely, iid N(0,1) diffusion noise in "
                "packed coefficient space is anisotropic after inversion to time."
            ),
        },
        "synthetic_verification": synthetic_audit(
            matrix=matrix, seed=seed, num_views=num_views
        ),
        "artifact_verification": audit_tf_clean_artifacts(
            artifact_paths, matrix=matrix, num_views=num_views
        ),
        "claims": {
            "permitted": [
                (
                    "The length-four packed RFFT is an invertible linear "
                    "reparameterization of each spatially pooled four-frame "
                    "tail window and retains phase through real/imaginary parts."
                ),
                (
                    "The implementation is causal by nonoverlapping bins and "
                    "keeps width-stacked camera views isolated."
                ),
                (
                    "Any measured gain may be described as a gain from this "
                    "representation, schedule, and fusion system."
                ),
            ],
            "forbidden_without_further_controls": [
                (
                    "It does not add information or a pretrained semantic prior "
                    "relative to the same spatially pooled time-domain packing."
                ),
                (
                    "A gain cannot be attributed specifically to frequency "
                    "coordinates without a same-shape time-domain packed control."
                ),
                (
                    "The raw packing is not an orthonormal change of basis, so "
                    "iid coefficient noise is not equivalent to iid time-domain "
                    "noise and channel SNR is not uniform."
                ),
                (
                    "Round-trip exactness applies only after per-view spatial "
                    "pooling; it does not reconstruct the original full-resolution video."
                ),
                (
                    "Invertibility alone does not imply easier, faster, or "
                    "higher-fidelity denoising."
                ),
            ],
            "minimum_next_representation_ablation": [
                "current raw packed RFFT",
                "sqrt(2)-Parseval-scaled packed RFFT",
                (
                    "same-shape causal time-domain four-frame packing, with "
                    "noise transformed to preserve the intended matched corruption"
                ),
            ],
        },
    }
    payload["identity_sha256"] = _canonical_sha256(payload)
    return payload


def _exclusive_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_report(path: Path, payload: dict[str, Any]) -> tuple[str, Path]:
    output = path.expanduser().resolve(strict=False)
    content = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _exclusive_write(output, content)
    report_sha256 = hashlib.sha256(content).hexdigest()
    sidecar = output.with_name(output.name + ".sha256")
    _exclusive_write(
        sidecar, f"{report_sha256}  {output.name}\n".encode("ascii")
    )
    return report_sha256, sidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=[],
        help="read-only safetensors artifact containing tf_clean; repeatable",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--num-views", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve(strict=False)
    for artifact in args.artifact:
        artifact_parent = artifact.expanduser().resolve(strict=True).parent
        if artifact_parent == output.parent or artifact_parent in output.parents:
            raise AuditError(
                "output must be outside every immutable input artifact directory"
            )
    payload = build_report(
        artifact_paths=args.artifact,
        seed=args.seed,
        num_views=args.num_views,
    )
    report_sha256, sidecar = write_report(output, payload)
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(output),
                "report_sha256": report_sha256,
                "sha256_sidecar": str(sidecar),
                "identity_sha256": payload["identity_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
