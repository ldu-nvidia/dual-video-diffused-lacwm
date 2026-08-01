"""Publication-gate video quality metrics for the Video Latent Forcing POC.

The library uses an immutable clip as the statistical unit and reserves the
name ``r3d18_frechet`` for a Frechet distance over pinned torchvision R3D-18
Kinetics-400 average-pool features.  It is not interchangeable with metrics
based on I3D or any other feature distribution.

Canonical videos have shape ``[B,3,8,64,112]`` and values in ``[-1,1]``.
Production extractors never download weights: every required file must already
exist, pass its pinned digest check, and is recorded with its full SHA-256.
Tests and dry runs can inject callable fake extractors into the functional API.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


LPIPS_ALEX_FRAME_METRIC = "lpips_alex_frame"
LPIPS_ALEX_TEMPORAL_DIFFERENCE_METRIC = "lpips_alex_temporal_difference"
R3D18_FRECHET_METRIC = "r3d18_frechet"

VIDEO_SHAPE = (3, 8, 64, 112)
R3D18_FEATURE_DIM = 512
LPIPS_PACKAGE_VERSION = "0.1.4"
TORCHVISION_PACKAGE_VERSION = "0.22.1"
LPIPS_LINEAR_FILENAME = "alex.pth"
LPIPS_LINEAR_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
ALEXNET_FILENAME = "alexnet-owt-7be5be79.pth"
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
ALEXNET_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
R3D18_FILENAME = "r3d_18-b3b3357e.pth"
R3D18_SHA256 = "b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d"
R3D18_URL = "https://download.pytorch.org/models/r3d_18-b3b3357e.pth"

R3D18_MEAN = (0.43216, 0.394666, 0.37645)
R3D18_STD = (0.22803, 0.22145, 0.216989)


class QualityMetricError(RuntimeError):
    """A quality input, dependency, weight, or paired sample is invalid."""


class PerceptualExtractor(Protocol):
    def __call__(self, reference: Tensor, candidate: Tensor) -> Tensor: ...


class VideoFeatureExtractor(Protocol):
    def __call__(self, video: Tensor) -> Tensor: ...


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _required_weight(
    path: str | Path,
    *,
    role: str,
    expected_sha256: str | None = None,
    expected_prefix: str | None = None,
    url: str,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise QualityMetricError(
            f"required {role} weight is unavailable: {resolved}; "
            f"download the pinned file out of band from {url}"
        )
    actual = sha256_file(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise QualityMetricError(
            f"{role} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    if expected_prefix is not None and not actual.startswith(expected_prefix):
        raise QualityMetricError(
            f"{role} SHA-256 prefix mismatch: expected {expected_prefix}, got {actual}"
        )
    return {
        "role": role,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": actual,
        "expected_sha256": expected_sha256,
        "expected_sha256_prefix": expected_prefix,
        "source_url": url,
    }


def _base_package_version(distribution: str) -> str:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise QualityMetricError(
            f"required package {distribution} is unavailable"
        ) from exc
    return version.split("+", 1)[0]


def _require_package_version(distribution: str, expected: str) -> str:
    actual = _base_package_version(distribution)
    if actual != expected:
        raise QualityMetricError(
            f"{distribution} version mismatch: expected {expected}, got {actual}"
        )
    return actual


def preprocessing_specification() -> dict[str, Any]:
    """Return the canonical, hashable preprocessing contract."""
    specification: dict[str, Any] = {
        "input": {
            "layout": "B,C,T,H,W",
            "shape_without_batch": list(VIDEO_SHAPE),
            "dtype": "floating-point",
            "range": [-1.0, 1.0],
            "range_policy": "fail-outside-range; no clipping",
        },
        "phase1_lowres_integration": {
            "input_layout": "B,C,T,H,W or C,T,H,W",
            "input_shape_without_batch": [3, 8, 32, 56],
            "operation": "bilinear upsample to 64x112",
            "align_corners": False,
            "antialias": False,
            "range_change": "none",
            "required_before_quality_metrics": True,
        },
        "explicit_candidate_clamp": {
            "operation": "torch.clamp(x,-1,1)",
            "default_metric_behavior": "not automatic; raw out-of-range input fails",
            "required_audit": "clipped count/fraction and input extrema",
        },
        LPIPS_ALEX_FRAME_METRIC: {
            "network": "LPIPS v0.1 AlexNet",
            "unit": "frame then mean over 8 frames per example",
            "backend_input_range": [-1.0, 1.0],
            "backend_normalize_argument": False,
        },
        LPIPS_ALEX_TEMPORAL_DIFFERENCE_METRIC: {
            "difference": "delta[t]=(frame[t+1]-frame[t])/2",
            "normalization_reason": "maps differences from [-2,2] to [-1,1]",
            "unit": "transition then mean over 7 transitions per example",
            "backend_normalize_argument": False,
        },
        "r3d18_avgpool": {
            "weights": "torchvision R3D_18_Weights.KINETICS400_V1",
            "input_conversion": "(x+1)/2 from [-1,1] to [0,1]",
            "input_permutation": "B,C,T,H,W -> B,T,C,H,W before preset",
            "resize_hw": [128, 171],
            "resize_interpolation": "bilinear",
            "resize_antialias": False,
            "center_crop_hw": [112, 112],
            "mean": list(R3D18_MEAN),
            "std": list(R3D18_STD),
            "temporal_resampling": "none; exactly 8 input frames",
            "feature_layer": "avgpool followed by flatten",
            "feature_dimension": R3D18_FEATURE_DIM,
        },
        R3D18_FRECHET_METRIC: {
            "feature": "r3d18_avgpool",
            "statistics_dtype": "float64",
            "covariance": "unbiased sample covariance (N-1 denominator)",
            "matrix_square_root": "symmetric eigendecomposition with roundoff clipping",
        },
    }
    specification["sha256"] = hashlib.sha256(
        _canonical_json(specification).encode("utf-8")
    ).hexdigest()
    return specification


def _validate_video_structure(
    video: Tensor,
    *,
    name: str,
    expected_shape: tuple[int, int, int, int] = VIDEO_SHAPE,
) -> None:
    if not isinstance(video, Tensor):
        raise QualityMetricError(f"{name} must be a torch.Tensor")
    if video.ndim != 5 or tuple(video.shape[1:]) != expected_shape:
        raise QualityMetricError(
            f"{name} must have shape [B,{','.join(map(str, expected_shape))}], got {tuple(video.shape)}"
        )
    if not video.is_floating_point():
        raise QualityMetricError(f"{name} must be floating-point")
    if not bool(torch.isfinite(video).all()):
        raise QualityMetricError(f"{name} contains non-finite values")


def _validate_video(video: Tensor, *, name: str) -> None:
    _validate_video_structure(video, name=name)
    minimum = float(video.detach().amin().cpu())
    maximum = float(video.detach().amax().cpu())
    if minimum < -1.000001 or maximum > 1.000001:
        raise QualityMetricError(
            f"{name} must be in [-1,1] without clipping, got [{minimum},{maximum}]"
        )


def upsample_lowres_video_for_quality(video: Tensor) -> Tensor:
    """Bilinearly map phase-1 ``[3,8,32,56]`` RGB to metric resolution.

    Batched and unbatched tensors are accepted.  Values remain in ``[-1,1]``;
    there is no temporal resampling, clamping, or data-dependent operation.
    """
    unbatched = isinstance(video, Tensor) and video.ndim == 4
    if unbatched:
        video = video.unsqueeze(0)
    _validate_video_structure(
        video,
        name="lowres_video",
        expected_shape=(3, 8, 32, 56),
    )
    minimum = float(video.detach().amin().cpu())
    maximum = float(video.detach().amax().cpu())
    if minimum < -1.000001 or maximum > 1.000001:
        raise QualityMetricError("lowres_video must be in [-1,1] before upsampling")
    batch = int(video.shape[0])
    frames = video.permute(0, 2, 1, 3, 4).reshape(batch * 8, 3, 32, 56)
    frames = F.interpolate(
        frames,
        size=(64, 112),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    result = frames.reshape(batch, 8, 3, 64, 112).permute(0, 2, 1, 3, 4)
    return result.squeeze(0) if unbatched else result


def clamp_video_for_quality(video: Tensor) -> tuple[Tensor, dict[str, Any]]:
    """Explicitly clamp a candidate and return a publication audit record.

    Quality functions never invoke this helper implicitly.  A caller choosing
    to clamp must save the returned clipped fraction and provenance.
    """
    _validate_video_structure(video, name="candidate")
    detached = video.detach()
    below = int((detached < -1.0).sum().cpu())
    above = int((detached > 1.0).sum().cpu())
    total = int(detached.numel())
    clamped = video.clamp(-1.0, 1.0)
    specification = {
        "operation": "torch.clamp(x,-1,1)",
        "input_layout": "B,C,T,H,W",
        "input_shape_without_batch": list(VIDEO_SHAPE),
        "deterministic": True,
    }
    audit: dict[str, Any] = {
        "clipped_below_count": below,
        "clipped_above_count": above,
        "clipped_count": below + above,
        "total_value_count": total,
        "clipped_fraction": float((below + above) / total),
        "input_min": float(detached.amin().cpu()),
        "input_max": float(detached.amax().cpu()),
        "output_min": float(clamped.detach().amin().cpu()),
        "output_max": float(clamped.detach().amax().cpu()),
        "provenance": specification,
    }
    audit["provenance_sha256"] = hashlib.sha256(
        _canonical_json(specification).encode("utf-8")
    ).hexdigest()
    return clamped, audit


def _validate_pair(reference: Tensor, candidate: Tensor) -> int:
    _validate_video(reference, name="reference")
    _validate_video(candidate, name="candidate")
    if reference.shape != candidate.shape:
        raise QualityMetricError("reference and candidate video shapes differ")
    return int(reference.shape[0])


def _perceptual_scores(
    extractor: PerceptualExtractor,
    reference: Tensor,
    candidate: Tensor,
    *,
    expected: int,
) -> Tensor:
    with torch.inference_mode():
        scores = extractor(reference, candidate)
    if not isinstance(scores, Tensor) or scores.numel() != expected:
        shape = tuple(scores.shape) if isinstance(scores, Tensor) else type(scores).__name__
        raise QualityMetricError(
            f"perceptual extractor must return one score per image ({expected}), got {shape}"
        )
    scores = scores.reshape(expected)
    if not bool(torch.isfinite(scores).all()):
        raise QualityMetricError("perceptual extractor returned non-finite values")
    return scores


def lpips_alex_per_frame_per_example(
    reference: Tensor,
    candidate: Tensor,
    extractor: PerceptualExtractor,
) -> Tensor:
    """Return frozen LPIPS-Alex averaged over frames, shape ``[B]``."""
    batch = _validate_pair(reference, candidate)
    ref_frames = reference.permute(0, 2, 1, 3, 4).reshape(batch * 8, 3, 64, 112)
    gen_frames = candidate.permute(0, 2, 1, 3, 4).reshape(batch * 8, 3, 64, 112)
    scores = _perceptual_scores(
        extractor, ref_frames, gen_frames, expected=batch * 8
    )
    return scores.reshape(batch, 8).mean(dim=1)


def temporal_difference_lpips_per_example(
    reference: Tensor,
    candidate: Tensor,
    extractor: PerceptualExtractor,
) -> Tensor:
    """Return LPIPS over explicitly normalized temporal differences.

    For canonical frames in ``[-1,1]``, each difference lies in ``[-2,2]``.
    Dividing by two maps it back to the LPIPS input range ``[-1,1]``.  No
    clipping or data-dependent normalization is used.
    """
    batch = _validate_pair(reference, candidate)
    ref_delta = (reference[:, :, 1:] - reference[:, :, :-1]) * 0.5
    gen_delta = (candidate[:, :, 1:] - candidate[:, :, :-1]) * 0.5
    ref_delta = ref_delta.permute(0, 2, 1, 3, 4).reshape(batch * 7, 3, 64, 112)
    gen_delta = gen_delta.permute(0, 2, 1, 3, 4).reshape(batch * 7, 3, 64, 112)
    scores = _perceptual_scores(
        extractor, ref_delta, gen_delta, expected=batch * 7
    )
    return scores.reshape(batch, 7).mean(dim=1)


class FrozenLPIPSAlex(nn.Module):
    """LPIPS 0.1.4 / AlexNet with explicit local, fully hashed weights."""

    def __init__(
        self,
        *,
        linear_weight_path: str | Path,
        alexnet_weight_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        package_version = _require_package_version("lpips", LPIPS_PACKAGE_VERSION)
        linear_record = _required_weight(
            linear_weight_path,
            role="LPIPS-Alex v0.1 linear calibration",
            expected_sha256=LPIPS_LINEAR_SHA256,
            url="https://github.com/richzhang/PerceptualSimilarity/tree/master/lpips/weights/v0.1",
        )
        alexnet_record = _required_weight(
            alexnet_weight_path,
            role="ImageNet AlexNet backbone",
            expected_sha256=ALEXNET_SHA256,
            url=ALEXNET_URL,
        )
        try:
            import lpips
            from torchvision.models import alexnet
        except ImportError as exc:  # pragma: no cover - version check normally catches
            raise QualityMetricError("lpips and torchvision are required") from exc

        model = lpips.LPIPS(
            net="alex",
            version="0.1",
            lpips=True,
            pretrained=True,
            pnet_rand=True,
            model_path=str(Path(linear_weight_path).expanduser().resolve()),
            eval_mode=True,
            verbose=False,
        )
        backbone = alexnet(weights=None)
        try:
            state = torch.load(
                Path(alexnet_weight_path).expanduser().resolve(),
                map_location="cpu",
                weights_only=True,
            )
            backbone.load_state_dict(state, strict=True)
        except Exception as exc:
            raise QualityMetricError(f"cannot load pinned AlexNet backbone: {exc}") from exc
        features = list(backbone.features.children())
        model.net.slice1 = nn.Sequential(*features[0:2])
        model.net.slice2 = nn.Sequential(*features[2:5])
        model.net.slice3 = nn.Sequential(*features[5:8])
        model.net.slice4 = nn.Sequential(*features[8:10])
        model.net.slice5 = nn.Sequential(*features[10:12])
        model.eval()
        model.requires_grad_(False)
        self.model = model.to(device)
        self.device = torch.device(device)
        self.provenance = {
            "extractor": "FrozenLPIPSAlex",
            "package": {"name": "lpips", "version": package_version},
            "weights": [linear_record, alexnet_record],
            "preprocessing": preprocessing_specification(),
        }

    def forward(self, reference: Tensor, candidate: Tensor) -> Tensor:
        if reference.ndim != 4 or reference.shape[1] != 3 or reference.shape != candidate.shape:
            raise QualityMetricError("LPIPS images must have matching [N,3,H,W] shapes")
        reference = reference.to(self.device, dtype=torch.float32)
        candidate = candidate.to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            # Inputs already use the documented [-1,1] range.
            return self.model(reference, candidate, normalize=False).reshape(reference.shape[0])


class FrozenR3D18AvgPool(nn.Module):
    """Pinned torchvision Kinetics-400 R3D-18 average-pool extractor."""

    def __init__(
        self,
        *,
        weight_path: str | Path,
        device: str | torch.device = "cpu",
        expected_sha256: str,
    ) -> None:
        super().__init__()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256.lower()
        ):
            raise QualityMetricError(
                "R3D-18 publication extraction requires a full 64-character expected SHA-256"
            )
        if expected_sha256.lower() != R3D18_SHA256:
            raise QualityMetricError(
                "R3D-18 expected SHA-256 differs from the preregistered official weight"
            )
        package_version = _require_package_version(
            "torchvision", TORCHVISION_PACKAGE_VERSION
        )
        weight_record = _required_weight(
            weight_path,
            role="torchvision R3D-18 Kinetics-400 V1",
            expected_sha256=expected_sha256,
            expected_prefix=None,
            url=R3D18_URL,
        )
        try:
            from torchvision.models.video import R3D_18_Weights, r3d_18
        except ImportError as exc:  # pragma: no cover
            raise QualityMetricError("torchvision video models are unavailable") from exc
        weights = R3D_18_Weights.KINETICS400_V1
        if weights.url != R3D18_URL:
            raise QualityMetricError(
                f"torchvision KINETICS400_V1 URL changed unexpectedly: {weights.url}"
            )
        model = r3d_18(weights=None, progress=False)
        try:
            state = torch.load(
                Path(weight_path).expanduser().resolve(),
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise QualityMetricError(f"cannot load pinned R3D-18 weights: {exc}") from exc
        model.fc = nn.Identity()
        model.eval()
        model.requires_grad_(False)
        self.model = model.to(device)
        self.transform = weights.transforms()
        self.device = torch.device(device)
        self.provenance = {
            "extractor": "FrozenR3D18AvgPool",
            "package": {"name": "torchvision", "version": package_version},
            "weights_enum": "R3D_18_Weights.KINETICS400_V1",
            "weights": [weight_record],
            "preprocessing": preprocessing_specification(),
        }

    def forward(self, video: Tensor) -> Tensor:
        _validate_video(video, name="video")
        video = video.to(dtype=torch.float32)
        # torchvision's pinned VideoClassification preset flattens leading
        # dimensions with ``view``.  The required B,T,C,H,W permutation is
        # generally strided, so materialize it before crossing that boundary.
        preset_input = (
            video.add(1.0).mul(0.5).permute(0, 2, 1, 3, 4).contiguous()
        )
        transformed = self.transform(preset_input).to(self.device)
        with torch.inference_mode():
            features = self.model(transformed)
        if tuple(features.shape) != (video.shape[0], R3D18_FEATURE_DIM):
            raise QualityMetricError(
                f"R3D-18 avgpool returned {tuple(features.shape)}, expected "
                f"[{video.shape[0]},{R3D18_FEATURE_DIM}]"
            )
        return features


def r3d18_avgpool_features(video: Tensor, extractor: VideoFeatureExtractor) -> Tensor:
    """Extract one pinned 512-D video feature per example."""
    _validate_video(video, name="video")
    with torch.inference_mode():
        features = extractor(video)
    if not isinstance(features, Tensor) or tuple(features.shape) != (
        video.shape[0],
        R3D18_FEATURE_DIM,
    ):
        shape = tuple(features.shape) if isinstance(features, Tensor) else type(features).__name__
        raise QualityMetricError(
            f"R3D-18 extractor must return [B,{R3D18_FEATURE_DIM}], got {shape}"
        )
    if not bool(torch.isfinite(features).all()):
        raise QualityMetricError("R3D-18 extractor returned non-finite features")
    return features


def _feature_matrix(features: Tensor | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(features, Tensor):
        array = features.detach().cpu().numpy()
    else:
        array = np.asarray(features)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise QualityMetricError(f"{name} features must be [N,D] with N>=2")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise QualityMetricError(f"{name} features contain non-finite values")
    return array


def _sample_covariance(features: np.ndarray) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / float(features.shape[0] - 1)


def _symmetric_psd_square_root(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(matrix)
    tolerance = np.finfo(np.float64).eps * max(1.0, float(np.abs(values).max())) * matrix.shape[0]
    if float(values.min()) < -tolerance:
        raise QualityMetricError(
            f"covariance is not positive semidefinite (minimum eigenvalue {values.min()})"
        )
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def r3d18_frechet(
    real_features: Tensor | np.ndarray,
    generated_features: Tensor | np.ndarray,
) -> float:
    """Stable float64 Frechet distance over R3D-18 avgpool features."""
    real = _feature_matrix(real_features, name="real")
    generated = _feature_matrix(generated_features, name="generated")
    if real.shape[1] != generated.shape[1]:
        raise QualityMetricError("real and generated feature dimensions differ")
    mean_delta = real.mean(axis=0) - generated.mean(axis=0)
    cov_real = _sample_covariance(real)
    cov_generated = _sample_covariance(generated)
    sqrt_real = _symmetric_psd_square_root(cov_real)
    middle = sqrt_real @ cov_generated @ sqrt_real
    middle = (middle + middle.T) * 0.5
    middle_values = np.linalg.eigvalsh(middle)
    tolerance = (
        np.finfo(np.float64).eps
        * max(1.0, float(np.abs(middle_values).max()))
        * middle.shape[0]
    )
    if float(middle_values.min()) < -tolerance:
        raise QualityMetricError("covariance product has a materially negative eigenvalue")
    trace_root = float(np.sqrt(np.clip(middle_values, 0.0, None)).sum())
    distance = float(
        mean_delta @ mean_delta
        + np.trace(cov_real)
        + np.trace(cov_generated)
        - 2.0 * trace_root
    )
    # Singular high-dimensional sample covariance estimates can accumulate
    # small negative cancellation error even in float64 (including comparing a
    # matrix with itself).  The tolerance is relative to the trace scale and
    # remains far below any reportable publication effect.
    roundoff = 1e-6 * max(
        1.0,
        float(mean_delta @ mean_delta + np.trace(cov_real) + np.trace(cov_generated)),
    )
    if distance < -roundoff:
        raise QualityMetricError(f"Frechet distance became negative: {distance}")
    return max(0.0, distance)


def _keyed_values(
    values: Mapping[str, float] | Sequence[tuple[str, float]],
    *,
    name: str,
) -> dict[str, float]:
    items = list(values.items()) if isinstance(values, Mapping) else list(values)
    result: dict[str, float] = {}
    for clip_id, raw_value in items:
        if not isinstance(clip_id, str) or not clip_id:
            raise QualityMetricError(f"{name} has an invalid clip ID")
        if clip_id in result:
            raise QualityMetricError(f"{name} contains duplicate clip ID {clip_id}")
        value = float(raw_value)
        if not np.isfinite(value):
            raise QualityMetricError(f"{name} contains a non-finite value for {clip_id}")
        result[clip_id] = value
    if not result:
        raise QualityMetricError(f"{name} is empty")
    return result


def paired_bootstrap_mean_difference(
    reference: Mapping[str, float] | Sequence[tuple[str, float]],
    candidate: Mapping[str, float] | Sequence[tuple[str, float]],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260801,
) -> dict[str, Any]:
    """Paired percentile bootstrap of candidate-minus-reference by clip ID."""
    if samples < 100:
        raise QualityMetricError("bootstrap samples must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise QualityMetricError("bootstrap confidence must lie in (0,1)")
    reference_by_id = _keyed_values(reference, name="reference")
    candidate_by_id = _keyed_values(candidate, name="candidate")
    if reference_by_id.keys() != candidate_by_id.keys():
        missing_candidate = sorted(reference_by_id.keys() - candidate_by_id.keys())
        missing_reference = sorted(candidate_by_id.keys() - reference_by_id.keys())
        raise QualityMetricError(
            "paired clip IDs differ: "
            f"missing_candidate={missing_candidate[:5]} missing_reference={missing_reference[:5]}"
        )
    clip_ids = sorted(reference_by_id)
    differences = np.asarray(
        [candidate_by_id[clip_id] - reference_by_id[clip_id] for clip_id in clip_ids],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    max_index_elements = 4_000_000
    chunk = max(1, min(samples, max_index_elements // len(clip_ids)))
    cursor = 0
    while cursor < samples:
        count = min(chunk, samples - cursor)
        indexes = generator.integers(
            0, len(clip_ids), size=(count, len(clip_ids)), endpoint=False
        )
        bootstrap[cursor : cursor + count] = differences[indexes].mean(axis=1)
        cursor += count
    tail = (1.0 - confidence) * 0.5
    low, high = np.quantile(bootstrap, [tail, 1.0 - tail], method="linear")
    identity_sha256 = hashlib.sha256(
        _canonical_json({"clip_ids": clip_ids}).encode("utf-8")
    ).hexdigest()
    return {
        "effect": "candidate_minus_reference",
        "estimate": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": float(confidence),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
        "paired_count": len(clip_ids),
        "clip_id_sha256": identity_sha256,
    }


@dataclass(frozen=True)
class QualityBatch:
    """CPU tensors suitable for distributed object gather and offline merging."""

    clip_ids: tuple[str, ...]
    lpips_alex_frame: Tensor
    lpips_alex_temporal_difference: Tensor
    r3d18_real_features: Tensor
    r3d18_generated_features: Tensor

    def __post_init__(self) -> None:
        count = len(self.clip_ids)
        if count == 0 or len(set(self.clip_ids)) != count or any(not value for value in self.clip_ids):
            raise QualityMetricError("QualityBatch clip IDs must be non-empty and unique")
        expected = {
            LPIPS_ALEX_FRAME_METRIC: (count,),
            LPIPS_ALEX_TEMPORAL_DIFFERENCE_METRIC: (count,),
            "r3d18_real_features": (count, R3D18_FEATURE_DIM),
            "r3d18_generated_features": (count, R3D18_FEATURE_DIM),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise QualityMetricError(f"QualityBatch {name} must be finite {shape}")


def evaluate_quality_batch(
    reference: Tensor,
    candidate: Tensor,
    clip_ids: Sequence[str],
    *,
    perceptual_extractor: PerceptualExtractor,
    video_feature_extractor: VideoFeatureExtractor,
) -> QualityBatch:
    """Compute per-example scores/features online and move results to CPU."""
    batch = _validate_pair(reference, candidate)
    ids = tuple(str(value) for value in clip_ids)
    if len(ids) != batch:
        raise QualityMetricError("clip ID count does not match batch size")
    frame = lpips_alex_per_frame_per_example(
        reference, candidate, perceptual_extractor
    )
    temporal = temporal_difference_lpips_per_example(
        reference, candidate, perceptual_extractor
    )
    real_features = r3d18_avgpool_features(reference, video_feature_extractor)
    generated_features = r3d18_avgpool_features(candidate, video_feature_extractor)
    return QualityBatch(
        clip_ids=ids,
        lpips_alex_frame=frame.detach().to(device="cpu", dtype=torch.float64),
        lpips_alex_temporal_difference=temporal.detach().to(
            device="cpu", dtype=torch.float64
        ),
        r3d18_real_features=real_features.detach().to(
            device="cpu", dtype=torch.float64
        ),
        r3d18_generated_features=generated_features.detach().to(
            device="cpu", dtype=torch.float64
        ),
    )


def merge_quality_batches(batches: Sequence[QualityBatch]) -> QualityBatch:
    """Merge gathered rank/batch objects and sort them by immutable clip ID."""
    if not batches:
        raise QualityMetricError("no quality batches were provided")
    rows: list[tuple[str, Tensor, Tensor, Tensor, Tensor]] = []
    for batch in batches:
        for index, clip_id in enumerate(batch.clip_ids):
            rows.append(
                (
                    clip_id,
                    batch.lpips_alex_frame[index],
                    batch.lpips_alex_temporal_difference[index],
                    batch.r3d18_real_features[index],
                    batch.r3d18_generated_features[index],
                )
            )
    ids = [row[0] for row in rows]
    if len(ids) != len(set(ids)):
        raise QualityMetricError("gathered quality batches contain duplicate clip IDs")
    rows.sort(key=lambda row: row[0])
    return QualityBatch(
        clip_ids=tuple(row[0] for row in rows),
        lpips_alex_frame=torch.stack([row[1] for row in rows]),
        lpips_alex_temporal_difference=torch.stack([row[2] for row in rows]),
        r3d18_real_features=torch.stack([row[3] for row in rows]),
        r3d18_generated_features=torch.stack([row[4] for row in rows]),
    )


def quality_summary(batch: QualityBatch) -> dict[str, float]:
    """Aggregate a complete gathered evaluation population."""
    return {
        LPIPS_ALEX_FRAME_METRIC: float(batch.lpips_alex_frame.mean()),
        LPIPS_ALEX_TEMPORAL_DIFFERENCE_METRIC: float(
            batch.lpips_alex_temporal_difference.mean()
        ),
        R3D18_FRECHET_METRIC: r3d18_frechet(
            batch.r3d18_real_features, batch.r3d18_generated_features
        ),
    }


def quality_metric_provenance(
    perceptual_extractor: Any,
    video_feature_extractor: Any,
) -> dict[str, Any]:
    """Collect production extractor provenance or fail closed."""
    perceptual = getattr(perceptual_extractor, "provenance", None)
    video = getattr(video_feature_extractor, "provenance", None)
    if not isinstance(perceptual, Mapping) or not isinstance(video, Mapping):
        raise QualityMetricError(
            "publication metrics require extractor provenance with actual weight hashes"
        )
    payload = {
        "metrics": [
            LPIPS_ALEX_FRAME_METRIC,
            LPIPS_ALEX_TEMPORAL_DIFFERENCE_METRIC,
            R3D18_FRECHET_METRIC,
        ],
        "perceptual_extractor": dict(perceptual),
        "video_feature_extractor": dict(video),
        "preprocessing": preprocessing_specification(),
    }
    payload["sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload
