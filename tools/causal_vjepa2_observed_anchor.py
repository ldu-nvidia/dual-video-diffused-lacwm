#!/usr/bin/env python3
"""Observed-history anchors and anchored-increment targets for video forcing.

This module implements the preregistered ``AINC-OFF`` representation without
changing the semantic predictor architecture.  A frozen V-JEPA 2.1 encoder
maps only the five observed RGB frames to an anchor ``A``.  The diffused state
is the normalized sequence of increments from that anchor to the eight cached
prefix-causal future semantic states.

The deployable path is deliberately split in two:

* :func:`extract_observed_anchor` accepts ``history`` but has no future input;
* :func:`decode_normalized_increment_prediction` combines a generated state
  with that observed-only anchor after the diffusion trajectory is complete.

The CLI materializes immutable train/validation anchor caches and fits the
single train-only increment normalization.  Protected-test manifests are not
accepted by any command in this file.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.datasets.droid.causal_vjepa2 import (  # noqa: E402
    FROZEN_SPLIT_EPISODE_IDS_SHA256,
    LAST_TEMPORAL_TOKEN,
    POOL_KERNEL,
    POOLED_TOKEN_GRID,
    TARGET_CHANNELS,
    TEACHER_FRAMES,
    TEACHER_SIZE,
    VJEPA2_CHECKPOINT_SHA256,
    VJEPA2_SOURCE_COMMIT,
    manifest_order_sha256,
)
from robot_wm.datasets.droid.video_latent_forcing import (  # noqa: E402
    FUTURE_SIZE,
    HISTORY_SIZE,
    read_clip_manifest,
    sha256_file,
)
from robot_wm.modeling.dual_diffusion.vjepa2_target import (  # noqa: E402
    VJEPA2_1_PATCH_SIZE,
    VJEPA2_1_SOURCE_DIM,
    VJEPA2_1_TUBELET_SIZE,
    PCAWhiteningStats,
    load_vjepa2_1_vit_base_encoder,
    prepare_vjepa2_1_views,
)
from tools import causal_vjepa2_screen as screen  # noqa: E402
from tools import video_latent_forcing_poc as vlf  # noqa: E402
from tools.causal_vjepa2_cache_bridge import (  # noqa: E402
    ProducerAttestedCausalDataset,
    construct_producer_attested_dataset,
)

ANCHOR_CACHE_SCHEMA = "causal-vjepa2-observed-anchor-cache-v1"
NORMALIZATION_SCHEMA = "causal-vjepa2-observed-increment-normalization-v1"
ANCHOR_TARGET_KIND = "observed-prefix-vjepa2-pca48-whitened-v1"
INCREMENT_TARGET_KIND = "observed-anchor-normalized-increments-v1"
ANCHOR_FILE = "anchors.fp16.npy"
ANCHOR_SHAPE = (TARGET_CHANNELS, *POOLED_TOKEN_GRID)
INCREMENT_SHAPE = screen.TARGET_SHAPE
TEMPORAL_AXIS = screen.TEMPORAL_AXIS
CHANNEL_AXIS = screen.TARGET_CHANNEL_AXIS
STD_FLOOR = 1e-6
ROUNDTRIP_TOLERANCE = 2e-5
PREREGISTRATION_COMMIT = "3cbeaac52b50233d3d3299da6fe2d15ce982a9f8"
ORIGINAL_PREREGISTRATION_SHA256 = (
    "4955e28af8bbae17c18a41abac83534a832726104ddbdacc991e948a53c3f82e"
)
PREREGISTRATION_SHA256 = (
    "58431b551011709122eb9e5b7681e7bf9468949223d8bb3c307a119a4323e36a"
)
PREREGISTRATION_REPO_PATH = (
    "docs/experiments/VIDEO_LATENT_FORCING_OBSERVED_ANCHOR_PROTOCOL.md"
)
PREREGISTRATION_PATH = REPO_ROOT / PREREGISTRATION_REPO_PATH

# [h0]^11 || [h0,h1,h2,h3,h4].  Keeping this explicit makes the final
# tubelet pair (h3,h4) auditable without interpreting concatenation prose.
OBSERVED_PREFIX_FRAME_MAP = (0,) * 12 + (1, 2, 3, 4)
OBSERVED_PREFIX_FINAL_PAIR = (3, 4)
ALLOWED_SPLITS = ("train", "val")
EXPECTED_SPLIT_CLIPS = {
    "train": screen.FROZEN_TRAIN_CLIPS,
    "val": screen.FROZEN_VALIDATION_CLIPS,
}
FROZEN_CACHE_WORLD_SIZE = 8
EXPECTED_INCREMENT_ELEMENTS_PER_CHANNEL = (
    screen.FROZEN_TRAIN_CLIPS
    * FUTURE_SIZE
    * ANCHOR_SHAPE[1]
    * ANCHOR_SHAPE[2]
)


class ObservedAnchorError(RuntimeError):
    """An observed-anchor scientific, cache, or provenance contract failed."""


def preregistration_record() -> dict[str, Any]:
    """Bind every artifact command to the frozen pre-metric protocol bytes."""
    if sha256_file(PREREGISTRATION_PATH) != PREREGISTRATION_SHA256:
        raise ObservedAnchorError("observed-anchor preregistration bytes changed")
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                PREREGISTRATION_COMMIT,
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        committed = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "show",
                f"{PREREGISTRATION_COMMIT}:{PREREGISTRATION_REPO_PATH}",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ObservedAnchorError(
            "cannot attest observed-anchor preregistration"
        ) from exc
    if hashlib.sha256(committed).hexdigest() != ORIGINAL_PREREGISTRATION_SHA256:
        raise ObservedAnchorError("original preregistered Git object changed")
    return {
        "commit": PREREGISTRATION_COMMIT,
        "original_file_sha256": ORIGINAL_PREREGISTRATION_SHA256,
        "file": vlf.file_record(PREREGISTRATION_PATH),
        "commit_is_ancestor": True,
        "frozen_before_candidate_metrics": True,
        "prospective_mathematical_correction_before_any_artifact": True,
        "prospective_matched_control_recovery_before_any_artifact": True,
        "running_temporal_doe_used_for_contingency_only": True,
        "continuation_local_abs_required": True,
        "correction_reason": (
            "time-constant anchors cancel from temporal differences; a later AINC "
            "commit requires its own same-commit absolute control"
        ),
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_history(history: Tensor) -> None:
    if history.ndim != 5 or tuple(history.shape[1:3]) != (3, HISTORY_SIZE):
        raise ValueError(
            f"history must have shape [B,3,{HISTORY_SIZE},H,W], got {tuple(history.shape)}"
        )
    if not history.is_floating_point() or not bool(torch.isfinite(history).all()):
        raise ValueError("history must be finite floating-point RGB")
    if bool((history < -1.0).any()) or bool((history > 1.0).any()):
        raise ValueError("history must use the [-1,1] RGB convention")


def build_observed_prefix(history: Tensor) -> Tensor:
    """Return ``[B,3,16,H,W]`` equal to ``[h0]^11 || history``.

    There is intentionally no ``future`` parameter.  The explicit frame map
    contains twelve occurrences of index zero because the five-frame suffix
    itself begins with ``h0``.
    """
    _validate_history(history)
    indices = torch.tensor(
        OBSERVED_PREFIX_FRAME_MAP, device=history.device, dtype=torch.long
    )
    prefix = history.index_select(2, indices)
    if prefix.shape[2] != TEACHER_FRAMES:
        raise AssertionError("observed prefix did not produce exactly 16 frames")
    return prefix


def prepare_observed_anchor_teacher_input(
    history: Tensor,
    *,
    teacher_size: tuple[int, int] = TEACHER_SIZE,
) -> Tensor:
    """Normalize one observed-only prefix to ``[B,3,16,Ht,Wt]``."""
    prefix = build_observed_prefix(history)
    batch, channels, frames, height, width = prefix.shape
    time_first = prefix.permute(0, 2, 1, 3, 4).contiguous()
    prepared = prepare_vjepa2_1_views(
        time_first,
        num_views=1,
        frame_map=tuple(range(TEACHER_FRAMES)),
        expected_view_size=(height, width),
        padded_view_size=(height, width),
        teacher_size=teacher_size,
    )
    expected = (batch, channels, frames, *teacher_size)
    if tuple(prepared.shape) != expected:
        raise AssertionError(
            f"observed teacher input has shape {tuple(prepared.shape)}, expected {expected}"
        )
    return prepared


def select_and_pool_observed_anchor(
    tokens: Tensor,
    *,
    batch_size: int,
    teacher_size: tuple[int, int] = TEACHER_SIZE,
    patch_size: int = VJEPA2_1_PATCH_SIZE,
    tubelet_size: int = VJEPA2_1_TUBELET_SIZE,
    pool_kernel: tuple[int, int] = POOL_KERNEL,
    expected_source_dim: int | None = VJEPA2_1_SOURCE_DIM,
) -> Tensor:
    """Select tubelet seven and return pooled ``[B,8,14,D]`` tokens."""
    if tokens.ndim != 3 or tokens.shape[0] != batch_size:
        raise ValueError("V-JEPA output must have shape [B,N,D]")
    if teacher_size[0] % patch_size or teacher_size[1] % patch_size:
        raise ValueError("teacher dimensions must divide by the patch size")
    temporal_tokens = TEACHER_FRAMES // tubelet_size
    token_height = teacher_size[0] // patch_size
    token_width = teacher_size[1] // patch_size
    expected_tokens = temporal_tokens * token_height * token_width
    if tokens.shape[1] != expected_tokens:
        raise ValueError(
            f"expected {expected_tokens} dense tokens, got {tokens.shape[1]}"
        )
    if expected_source_dim is not None and tokens.shape[-1] != expected_source_dim:
        raise ValueError(
            f"expected token width {expected_source_dim}, got {tokens.shape[-1]}"
        )
    if LAST_TEMPORAL_TOKEN != temporal_tokens - 1:
        raise ValueError("observed anchor must select the final temporal tubelet")
    if token_height % pool_kernel[0] or token_width % pool_kernel[1]:
        raise ValueError("spatial token grid must divide by the pooling kernel")

    grid = tokens.reshape(
        batch_size,
        temporal_tokens,
        token_height,
        token_width,
        tokens.shape[-1],
    )[:, LAST_TEMPORAL_TOKEN]
    channels_first = grid.permute(0, 3, 1, 2).contiguous()
    pooled = F.avg_pool2d(channels_first, kernel_size=pool_kernel, stride=pool_kernel)
    return pooled.permute(0, 2, 3, 1).contiguous()


def project_observed_anchor_tokens(
    tokens: Tensor,
    stats: PCAWhiteningStats,
    *,
    quantize_online: bool = True,
) -> Tensor:
    """Apply the frozen PCA and return ``[B,48,8,14]`` float32.

    Production online extraction passes through float16 exactly once before
    returning to float32, matching the immutable cache storage contract.
    """
    if tokens.ndim != 4 or tuple(tokens.shape[1:3]) != POOLED_TOKEN_GRID:
        raise ValueError("observed pooled tokens must be [B,8,14,D]")
    if stats.input_dim != VJEPA2_1_SOURCE_DIM or stats.output_dim != TARGET_CHANNELS:
        raise ValueError("PCA statistics must project exactly 768 -> 48")
    projected = stats.project(tokens).permute(0, 3, 1, 2).contiguous()
    if tuple(projected.shape[1:]) != ANCHOR_SHAPE or not bool(
        torch.isfinite(projected).all()
    ):
        raise FloatingPointError("projected observed anchor is malformed")
    projected = projected.float()
    result = projected.to(torch.float16).float() if quantize_online else projected
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("float16-quantized observed anchor is non-finite")
    return result


@torch.inference_mode()
def extract_observed_anchor(
    history: Tensor,
    *,
    encoder: nn.Module,
    stats: PCAWhiteningStats,
    encoder_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Extract a deployable anchor using observed RGB only.

    The absence of a future/target parameter is a leakage boundary and is
    covered by tests.  Returned values have undergone the cache-equivalent
    float16 -> float32 round trip.
    """
    prepared = prepare_observed_anchor_teacher_input(history)
    encoder.eval()
    with torch.autocast(
        device_type=prepared.device.type,
        dtype=encoder_dtype,
        enabled=prepared.device.type == "cuda",
    ):
        tokens = encoder(prepared, training=False)
    if not isinstance(tokens, Tensor):
        raise TypeError("V-JEPA encoder must return one dense token tensor")
    pooled = select_and_pool_observed_anchor(tokens, batch_size=history.shape[0])
    return project_observed_anchor_tokens(pooled, stats, quantize_online=True)


def _validate_semantic_and_anchor(semantic: Tensor, anchor: Tensor) -> None:
    if semantic.ndim != 5 or tuple(semantic.shape[1:]) != INCREMENT_SHAPE:
        raise ValueError(f"semantic must have shape [B,{INCREMENT_SHAPE}]")
    if anchor.ndim != 4 or tuple(anchor.shape[1:]) != ANCHOR_SHAPE:
        raise ValueError(f"anchor must have shape [B,{ANCHOR_SHAPE}]")
    if semantic.shape[0] != anchor.shape[0]:
        raise ValueError("semantic and anchor batches differ")
    if semantic.device != anchor.device:
        raise ValueError("semantic and anchor must share a device")
    if not semantic.is_floating_point() or not anchor.is_floating_point():
        raise TypeError("semantic and anchor tensors must be floating point")
    if not bool(torch.isfinite(semantic).all()) or not bool(
        torch.isfinite(anchor).all()
    ):
        raise ValueError("semantic and anchor tensors must be finite")


def anchored_increments(semantic: Tensor, anchor: Tensor) -> Tensor:
    """Return ``D0=S0-A`` and ``Dj=Sj-S[j-1]`` in the input dtype."""
    _validate_semantic_and_anchor(semantic, anchor)
    dtype = torch.promote_types(semantic.dtype, anchor.dtype)
    semantic = semantic.to(dtype=dtype)
    anchor = anchor.to(dtype=dtype)
    result = torch.empty_like(semantic)
    result[:, :, 0] = semantic[:, :, 0] - anchor
    result[:, :, 1:] = semantic[:, :, 1:] - semantic[:, :, :-1]
    return result


def decode_anchored_increments(increments: Tensor, anchor: Tensor) -> Tensor:
    """Deterministically invert :func:`anchored_increments`."""
    _validate_semantic_and_anchor(increments, anchor)
    dtype = torch.promote_types(increments.dtype, anchor.dtype)
    increments = increments.to(dtype=dtype)
    anchor = anchor.to(dtype=dtype)
    return anchor.unsqueeze(TEMPORAL_AXIS) + increments.cumsum(dim=TEMPORAL_AXIS)


def _channel_view(value: Tensor, reference: Tensor) -> Tensor:
    if tuple(value.shape) != (TARGET_CHANNELS,):
        raise ValueError(f"channel statistic must have shape [{TARGET_CHANNELS}]")
    return value.to(device=reference.device, dtype=reference.dtype).reshape(
        1, TARGET_CHANNELS, 1, 1, 1
    )


@dataclass(frozen=True)
class ObservedIncrementNormalization:
    """One per-channel train-only normalization for all eight increments."""

    mean: Tensor
    std: Tensor
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("mean", "std"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or tuple(value.shape) != (
                TARGET_CHANNELS,
            ):
                raise ValueError(f"{name} must be a [{TARGET_CHANNELS}] tensor")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite floating point")
        if bool((self.std <= 0).any()):
            raise ValueError("increment standard deviations must be positive")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("normalization provenance must be a mapping")

    @classmethod
    def identity(cls) -> ObservedIncrementNormalization:
        return cls(
            torch.zeros(TARGET_CHANNELS, dtype=torch.float32),
            torch.ones(TARGET_CHANNELS, dtype=torch.float32),
            {"identity": True},
        )

    def encode(self, increments: Tensor) -> Tensor:
        if increments.ndim != 5 or tuple(increments.shape[1:]) != INCREMENT_SHAPE:
            raise ValueError(f"increments must have shape [B,{INCREMENT_SHAPE}]")
        increments = increments.float()
        return (increments - _channel_view(self.mean, increments)) / _channel_view(
            self.std, increments
        )

    def decode(self, normalized: Tensor) -> Tensor:
        if normalized.ndim != 5 or tuple(normalized.shape[1:]) != INCREMENT_SHAPE:
            raise ValueError(f"normalized state must have shape [B,{INCREMENT_SHAPE}]")
        normalized = normalized.float()
        return normalized * _channel_view(self.std, normalized) + _channel_view(
            self.mean, normalized
        )


def encode_normalized_increment_target(
    semantic: Tensor,
    anchor: Tensor,
    normalization: ObservedIncrementNormalization,
) -> Tensor:
    """Construct the training state ``Q`` in float32."""
    return normalization.encode(anchored_increments(semantic.float(), anchor.float()))


def decode_normalized_increment_prediction(
    normalized: Tensor,
    anchor: Tensor,
    normalization: ObservedIncrementNormalization,
) -> Tensor:
    """Decode generated ``Q`` to absolute semantic states in float32."""
    return decode_anchored_increments(
        normalization.decode(normalized), anchor.float()
    ).float()


def anchor_static_control(
    anchor: Tensor,
    normalization: ObservedIncrementNormalization,
) -> tuple[Tensor, Tensor]:
    """Return exact zero increments as ``(Q, decoded S)``."""
    if anchor.ndim != 4 or tuple(anchor.shape[1:]) != ANCHOR_SHAPE:
        raise ValueError(f"anchor must have shape [B,{ANCHOR_SHAPE}]")
    zeros = torch.zeros(
        anchor.shape[0], *INCREMENT_SHAPE, device=anchor.device, dtype=torch.float32
    )
    normalized = normalization.encode(zeros)
    # Construct the decoded control directly so it is exactly static even when
    # ``(-mean / std) * std + mean`` has a float32 rounding residual.
    decoded = (
        anchor.float()
        .unsqueeze(TEMPORAL_AXIS)
        .expand(-1, -1, FUTURE_SIZE, -1, -1)
        .clone()
    )
    return normalized, decoded


def mean_increment_control(
    anchor: Tensor,
    normalization: ObservedIncrementNormalization,
) -> tuple[Tensor, Tensor]:
    """Return ``Q=0`` and the decoded train-mean increment trajectory."""
    if anchor.ndim != 4 or tuple(anchor.shape[1:]) != ANCHOR_SHAPE:
        raise ValueError(f"anchor must have shape [B,{ANCHOR_SHAPE}]")
    normalized = torch.zeros(
        anchor.shape[0], *INCREMENT_SHAPE, device=anchor.device, dtype=torch.float32
    )
    decoded = decode_normalized_increment_prediction(normalized, anchor, normalization)
    return normalized, decoded


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservedAnchorError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ObservedAnchorError(f"{label} must be a JSON mapping")
    return value


def _split_manifest(
    path: str | Path, *, expected_split: str | None = None
) -> tuple[Path, list[dict[str, Any]], str]:
    manifest = Path(path).expanduser().resolve()
    rows = read_clip_manifest(manifest, expected_split=expected_split)
    splits = {str(row["split"]) for row in rows}
    if len(splits) != 1:
        raise ObservedAnchorError("one anchor artifact cannot mix dataset splits")
    split = splits.pop()
    if split not in ALLOWED_SPLITS:
        raise ObservedAnchorError(
            "observed-anchor artifacts permit train/validation only"
        )
    if len(rows) != EXPECTED_SPLIT_CLIPS[split]:
        raise ObservedAnchorError(
            f"{split} manifest must contain exactly {EXPECTED_SPLIT_CLIPS[split]} clips"
        )
    if any(bool(row.get("protected")) for row in rows):
        raise ObservedAnchorError(
            "protected rows are forbidden in observed-anchor artifacts"
        )
    return manifest, rows, split


def _resolve_anchor_paths(
    anchor_cache_root: str | Path, split: str
) -> tuple[Path, Path]:
    root = Path(anchor_cache_root).expanduser().resolve()
    split_root = root / split
    return split_root / "metadata.json", split_root / ANCHOR_FILE


def validate_anchor_cache(
    *,
    manifest_path: str | Path,
    anchor_cache_root: str | Path,
    expected_split: str | None = None,
) -> tuple[dict[str, Any], np.memmap]:
    """Validate and open a complete train/validation anchor cache read-only."""
    manifest, rows, split = _split_manifest(
        manifest_path, expected_split=expected_split
    )
    metadata_path, anchor_path = _resolve_anchor_paths(anchor_cache_root, split)
    metadata = _read_json(metadata_path, label="observed-anchor cache metadata")
    expected_shape = [len(rows), *ANCHOR_SHAPE]
    artifact_identity = metadata.get("artifact_identity")
    if (
        metadata.get("schema") != ANCHOR_CACHE_SCHEMA
        or metadata.get("status") != "complete"
        or metadata.get("complete") is not True
        or metadata.get("target_kind") != ANCHOR_TARGET_KIND
        or metadata.get("split") != split
        or metadata.get("clips") != len(rows)
        or metadata.get("manifest_sha256") != sha256_file(manifest)
        or metadata.get("manifest_order_sha256") != manifest_order_sha256(rows)
        or metadata.get("episode_ids_sha256") != FROZEN_SPLIT_EPISODE_IDS_SHA256[split]
        or metadata.get("split_episode_ids_sha256")
        != {name: FROZEN_SPLIT_EPISODE_IDS_SHA256[name] for name in ALLOWED_SPLITS}
        or metadata.get("train_validation_episode_disjoint") is not True
        or metadata.get("anchor_shape") != expected_shape
        or metadata.get("per_clip_anchor_shape") != list(ANCHOR_SHAPE)
        or metadata.get("anchor_dtype") != "float16"
        or metadata.get("prefix_frame_map") != list(OBSERVED_PREFIX_FRAME_MAP)
        or metadata.get("final_tubelet_history_pair")
        != list(OBSERVED_PREFIX_FINAL_PAIR)
        or metadata.get("future_tensor_read_inside_anchor_extractor") is not False
        or metadata.get("protected_test_accessed") is not False
        or metadata.get("test_rows_extracted") != 0
        or metadata.get("allowed_splits") != list(ALLOWED_SPLITS)
        or metadata.get("online_output_dtype") != "float32_after_float16_roundtrip"
        or metadata.get("source_commit") != VJEPA2_SOURCE_COMMIT
        or metadata.get("checkpoint_sha256") != VJEPA2_CHECKPOINT_SHA256
        or metadata.get("pca_sha256") != metadata.get(
            "semantic_cache_pca_sha256"
        )
        or metadata.get("teacher_size") != list(TEACHER_SIZE)
        or metadata.get("teacher_frames") != TEACHER_FRAMES
        or metadata.get("last_temporal_token") != LAST_TEMPORAL_TOKEN
        or metadata.get("pool_kernel") != list(POOL_KERNEL)
        or metadata.get("pooled_token_grid") != list(POOLED_TOKEN_GRID)
        or metadata.get("world_size") != FROZEN_CACHE_WORLD_SIZE
        or not isinstance(artifact_identity, Mapping)
        or metadata.get("cache_id") != _canonical_sha256(artifact_identity)
        or any(metadata.get(key) != value for key, value in artifact_identity.items())
    ):
        raise ObservedAnchorError("observed-anchor cache contract is malformed")
    implementation = artifact_identity.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ObservedAnchorError("anchor cache lacks implementation provenance")
    dependencies = implementation.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise ObservedAnchorError("anchor cache lacks implementation file bindings")
    recorded_entrypoint = dependencies.get("entrypoint")
    if not isinstance(recorded_entrypoint, Mapping) or recorded_entrypoint.get(
        "sha256"
    ) != sha256_file(__file__):
        raise ObservedAnchorError(
            "active observed-anchor implementation differs from cache producer"
        )
    try:
        verified_files = vlf._verify_embedded_file_records(  # noqa: SLF001
            dependencies, label="observed-anchor implementation"
        )
    except vlf.PocError as exc:
        raise ObservedAnchorError(str(exc)) from exc
    if verified_files < 5:
        raise ObservedAnchorError("anchor cache implementation evidence is incomplete")
    pca_record = metadata.get("pca_file")
    semantic_record = metadata.get("semantic_cache_metadata")
    if (
        not isinstance(pca_record, Mapping)
        or not isinstance(semantic_record, Mapping)
        or dict(pca_record)
        != vlf.file_record(Path(str(pca_record.get("path", ""))))
        or dict(semantic_record)
        != vlf.file_record(Path(str(semantic_record.get("path", ""))))
        or pca_record.get("sha256") != artifact_identity.get("pca_sha256")
        or semantic_record.get("sha256")
        != artifact_identity.get("semantic_cache_metadata_sha256")
    ):
        raise ObservedAnchorError("anchor cache PCA/semantic evidence changed")
    record = metadata.get("anchor_file")
    if (
        not isinstance(record, Mapping)
        or Path(str(record.get("path", ""))).expanduser().resolve() != anchor_path
        or record.get("sha256") != sha256_file(anchor_path)
        or int(record.get("bytes", -1)) != anchor_path.stat().st_size
    ):
        raise ObservedAnchorError("observed-anchor array differs from metadata")
    anchors = np.load(anchor_path, mmap_mode="r", allow_pickle=False)
    if (
        tuple(anchors.shape) != tuple(expected_shape)
        or anchors.dtype != np.float16
        or anchors.flags.writeable
    ):
        raise ObservedAnchorError("observed-anchor array shape/dtype/access differs")
    agreement = metadata.get("cached_online_agreement")
    if not isinstance(agreement, Mapping):
        raise ObservedAnchorError("cache lacks frozen cached/online agreement evidence")
    index = int(agreement.get("manifest_index", -1))
    if not 0 <= index < len(rows):
        raise ObservedAnchorError("cached/online agreement index is invalid")
    row_digest = hashlib.sha256(
        np.ascontiguousarray(anchors[index]).tobytes()
    ).hexdigest()
    if (
        agreement.get("split") != split
        or agreement.get("clip_id") != str(rows[index]["clip_id"])
        or agreement.get("cached_float16_sha256") != row_digest
        or agreement.get("online_float16_sha256") != row_digest
        or float(agreement.get("max_abs_error", -1.0)) != 0.0
    ):
        raise ObservedAnchorError("cached/online agreement evidence is inconsistent")
    return metadata, anchors


class ObservedAnchorDataset(Dataset):
    """Producer-attested future semantics paired with observed-only anchors."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        semantic_cache_root: str | Path,
        anchor_cache_root: str | Path,
    ) -> None:
        manifest, rows, split = _split_manifest(manifest_path)
        self.base: ProducerAttestedCausalDataset = construct_producer_attested_dataset(
            manifest, data_root, semantic_cache_root
        )
        metadata, anchors = validate_anchor_cache(
            manifest_path=manifest,
            anchor_cache_root=anchor_cache_root,
            expected_split=split,
        )
        if len(self.base) != len(rows) or len(anchors) != len(rows):
            raise ObservedAnchorError("semantic and anchor populations differ")
        semantic_metadata_path = (
            Path(semantic_cache_root).expanduser().resolve() / split / "metadata.json"
        )
        if metadata.get("semantic_cache_metadata_sha256") != sha256_file(
            semantic_metadata_path
        ) or metadata.get("semantic_cache_id") != self.base.cache_metadata.get(
            "cache_id"
        ):
            raise ObservedAnchorError("anchor cache is bound to another semantic cache")
        self.rows = rows
        self.split = split
        self.anchor_cache_metadata = metadata
        self._anchors = anchors

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        anchor = torch.from_numpy(np.array(self._anchors[index], copy=True))
        if anchor.dtype != torch.float16 or tuple(anchor.shape) != ANCHOR_SHAPE:
            raise ObservedAnchorError("observed-anchor row is malformed")
        sample["observed_anchor"] = anchor
        sample["observed_anchor_cache_id"] = str(self.anchor_cache_metadata["cache_id"])
        return sample


def construct_observed_anchor_dataset(
    manifest_path: str | Path,
    data_root: str | Path,
    semantic_cache_root: str | Path,
    anchor_cache_root: str | Path,
) -> ObservedAnchorDataset:
    dataset = ObservedAnchorDataset(
        manifest_path,
        data_root,
        semantic_cache_root,
        anchor_cache_root,
    )
    if len(dataset) < 1:
        raise ObservedAnchorError("observed-anchor dataset is empty")
    return dataset


def _load_bound_pca(
    semantic_metadata: Mapping[str, Any], metadata_path: Path
) -> tuple[PCAWhiteningStats, Path, dict[str, Any]]:
    pca_value = semantic_metadata.get("pca_file")
    if not isinstance(pca_value, str):
        raise ObservedAnchorError("semantic cache does not bind a PCA file")
    path = Path(pca_value)
    if not path.is_absolute():
        path = metadata_path.parent / path
    path = path.expanduser().resolve()
    if not path.is_file() or sha256_file(path) != semantic_metadata.get("pca_sha256"):
        raise ObservedAnchorError("bound PCA artifact is missing or changed")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        raise ObservedAnchorError("cannot safely load producer-attested PCA") from exc
    if not isinstance(payload, Mapping):
        raise ObservedAnchorError("PCA payload is not a mapping")
    stats = PCAWhiteningStats.from_payload(payload)
    if stats.input_dim != VJEPA2_1_SOURCE_DIM or stats.output_dim != TARGET_CHANNELS:
        raise ObservedAnchorError("PCA does not implement the frozen 768 -> 48 map")
    return stats, path, dict(payload)


def _source_record() -> dict[str, Any]:
    source = vlf.git_record()
    if source.get("dirty") is not False:
        raise ObservedAnchorError("artifact commands require a clean committed source")
    dependencies = {
        "entrypoint": vlf.file_record(__file__),
        "cache_bridge": vlf.file_record(
            inspect.getsourcefile(ProducerAttestedCausalDataset)
        ),
        "causal_dataset": vlf.file_record(
            REPO_ROOT / "robot_wm" / "datasets" / "droid" / "causal_vjepa2.py"
        ),
        "vjepa_target": vlf.file_record(
            REPO_ROOT / "robot_wm" / "modeling" / "dual_diffusion" / "vjepa2_target.py"
        ),
        "preregistration": vlf.file_record(PREREGISTRATION_PATH),
    }
    return {
        **source,
        "dependencies": dependencies,
        "preregistration": preregistration_record(),
    }


def _row_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _anchor_cache_identity(
    *,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    manifest: Path,
    semantic_metadata: Mapping[str, Any],
    semantic_metadata_path: Path,
    source: Mapping[str, Any],
    teacher_evidence: Mapping[str, Any],
    world_size: int,
) -> dict[str, Any]:
    return {
        "schema": ANCHOR_CACHE_SCHEMA,
        "target_kind": ANCHOR_TARGET_KIND,
        "split": split,
        "clips": len(rows),
        "manifest_sha256": sha256_file(manifest),
        "manifest_order_sha256": manifest_order_sha256(rows),
        "episode_ids_sha256": FROZEN_SPLIT_EPISODE_IDS_SHA256[split],
        "split_episode_ids_sha256": {
            name: FROZEN_SPLIT_EPISODE_IDS_SHA256[name] for name in ALLOWED_SPLITS
        },
        "train_validation_episode_disjoint": True,
        "semantic_cache_id": semantic_metadata.get("cache_id"),
        "semantic_cache_metadata_sha256": sha256_file(semantic_metadata_path),
        "pca_sha256": semantic_metadata.get("pca_sha256"),
        "semantic_cache_pca_sha256": semantic_metadata.get("pca_sha256"),
        "source_commit": VJEPA2_SOURCE_COMMIT,
        "checkpoint_sha256": VJEPA2_CHECKPOINT_SHA256,
        "teacher_evidence": dict(teacher_evidence),
        "teacher_size": list(TEACHER_SIZE),
        "teacher_frames": TEACHER_FRAMES,
        "last_temporal_token": LAST_TEMPORAL_TOKEN,
        "pool_kernel": list(POOL_KERNEL),
        "pooled_token_grid": list(POOLED_TOKEN_GRID),
        "prefix_frame_map": list(OBSERVED_PREFIX_FRAME_MAP),
        "final_tubelet_history_pair": list(OBSERVED_PREFIX_FINAL_PAIR),
        "future_tensor_read_inside_anchor_extractor": False,
        "anchor_shape": [len(rows), *ANCHOR_SHAPE],
        "per_clip_anchor_shape": list(ANCHOR_SHAPE),
        "anchor_dtype": "float16",
        "online_output_dtype": "float32_after_float16_roundtrip",
        "world_size": world_size,
        "encoder_dtype": "bfloat16",
        "encoder_batch_size_per_rank": 1,
        "implementation": dict(source),
        "allowed_splits": list(ALLOWED_SPLITS),
        "protected_test_accessed": False,
        "test_rows_extracted": 0,
    }


def build_cache_command(args: argparse.Namespace) -> int:
    """Materialize a distributed immutable history-only anchor cache."""
    context = vlf.initialize_distributed()
    try:
        if context.world_size != FROZEN_CACHE_WORLD_SIZE:
            raise ObservedAnchorError(
                "production anchor extraction requires exactly eight ranks"
            )
        if args.batch_size != 1:
            raise ObservedAnchorError(
                "production anchor extraction fixes batch size to one"
            )
        manifest, rows, split = _split_manifest(args.manifest)
        dataset = construct_producer_attested_dataset(
            manifest, args.data_root, args.semantic_cache_root
        )
        semantic_metadata_path = (
            Path(args.semantic_cache_root).expanduser().resolve()
            / split
            / "metadata.json"
        )
        semantic_metadata = dict(dataset.cache_metadata)
        stats, pca_path, _ = _load_bound_pca(semantic_metadata, semantic_metadata_path)
        source = _source_record()
        vjepa_source = Path(args.vjepa_source).expanduser().resolve()
        vjepa_checkpoint = Path(args.vjepa_checkpoint).expanduser().resolve()
        teacher_evidence = {
            "source_root": str(vjepa_source),
            "source_commit": VJEPA2_SOURCE_COMMIT,
            "checkpoint": vlf.file_record(vjepa_checkpoint),
            "source_archive_sha256": semantic_metadata.get("source_archive_sha256"),
            "source_license": semantic_metadata.get("source_license"),
            "producer_checkpoint_evidence": semantic_metadata.get(
                "checkpoint_evidence"
            ),
        }
        identity = _anchor_cache_identity(
            split=split,
            rows=rows,
            manifest=manifest,
            semantic_metadata=semantic_metadata,
            semantic_metadata_path=semantic_metadata_path,
            source=source,
            teacher_evidence=teacher_evidence,
            world_size=context.world_size,
        )
        cache_id = _canonical_sha256(identity)
        root = vlf.approved_artifact_path(args.output_root)
        split_root = root / split
        metadata_path = split_root / "metadata.json"
        final_path = split_root / ANCHOR_FILE
        work_root = split_root / ".anchor-work"
        if metadata_path.is_file():
            if not args.resume:
                raise ObservedAnchorError(
                    f"immutable anchor output already exists: {split_root}"
                )
            validate_anchor_cache(
                manifest_path=manifest,
                anchor_cache_root=root,
                expected_split=split,
            )
            context.barrier()
            return 0
        if context.is_primary:
            if split_root.exists() and not args.resume:
                raise ObservedAnchorError(
                    f"immutable anchor output already exists: {split_root}"
                )
            identity_path = work_root / "identity.json"
            expected_work = {"cache_id": cache_id, "identity": identity}
            if split_root.exists():
                if not work_root.is_dir() or not identity_path.is_file():
                    raise ObservedAnchorError(
                        "resume output lacks an exact anchor work identity"
                    )
                if (
                    _read_json(identity_path, label="anchor work identity")
                    != expected_work
                ):
                    raise ObservedAnchorError("anchor resume identity differs")
            else:
                work_root.mkdir(parents=True, exist_ok=False)
                vlf.atomic_write_json(
                    identity_path,
                    expected_work,
                    exclusive=True,
                )
        context.barrier()

        assigned = list(range(context.rank, len(rows), context.world_size))
        shard_path = work_root / f"rank-{context.rank:05d}.fp16.npy"
        sidecar_path = work_root / f"rank-{context.rank:05d}.json"
        shard_record: dict[str, Any]
        if args.resume and sidecar_path.is_file():
            shard_record = _read_json(
                sidecar_path, label="completed anchor rank sidecar"
            )
            record = shard_record.get("file")
            if (
                shard_record.get("cache_id") != cache_id
                or shard_record.get("rank") != context.rank
                or shard_record.get("world_size") != context.world_size
                or shard_record.get("assigned_indices") != assigned
                or not isinstance(record, Mapping)
                or dict(record) != vlf.file_record(shard_path)
            ):
                raise ObservedAnchorError("completed anchor rank shard cannot resume")
        else:
            if sidecar_path.exists():
                raise ObservedAnchorError("anchor rank sidecar collision")
            encoder = load_vjepa2_1_vit_base_encoder(
                source_path=args.vjepa_source,
                checkpoint_path=args.vjepa_checkpoint,
                expected_source_commit=VJEPA2_SOURCE_COMMIT,
                expected_checkpoint_sha256=VJEPA2_CHECKPOINT_SHA256,
            ).to(context.device)
            shard = np.lib.format.open_memmap(
                shard_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(assigned), *ANCHOR_SHAPE),
            )
            clip_ids: list[str] = []
            row_hashes: list[str] = []
            for local_index, manifest_index in enumerate(assigned):
                sample = dataset.base[manifest_index]
                history = (
                    sample["history"]
                    .unsqueeze(0)
                    .to(device=context.device, dtype=torch.float32)
                )
                extracted = extract_observed_anchor(
                    history,
                    encoder=encoder,
                    stats=stats,
                    encoder_dtype=torch.bfloat16,
                )
                value = extracted[0].half().cpu().numpy()
                shard[local_index] = value
                clip_ids.append(str(rows[manifest_index]["clip_id"]))
                row_hashes.append(_row_sha256(value))
            shard.flush()
            shard_record = {
                "cache_id": cache_id,
                "rank": context.rank,
                "world_size": context.world_size,
                "assigned_indices": assigned,
                "clip_ids": clip_ids,
                "row_sha256": row_hashes,
                "file": vlf.file_record(shard_path),
            }
            vlf.atomic_write_json(
                sidecar_path,
                shard_record,
                exclusive=True,
            )
        context.barrier()

        if context.is_primary:
            temporary = split_root / f".{ANCHOR_FILE}.partial"
            merged = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float16,
                shape=(len(rows), *ANCHOR_SHAPE),
            )
            covered: set[int] = set()
            shard_records: list[dict[str, Any]] = []
            online_sha_at_zero: str | None = None
            for rank in range(context.world_size):
                sidecar = _read_json(
                    work_root / f"rank-{rank:05d}.json", label="anchor rank sidecar"
                )
                expected_indices = list(range(rank, len(rows), context.world_size))
                if (
                    sidecar.get("cache_id") != cache_id
                    or sidecar.get("rank") != rank
                    or sidecar.get("world_size") != context.world_size
                    or sidecar.get("assigned_indices") != expected_indices
                ):
                    raise ObservedAnchorError("anchor rank sidecar identity differs")
                record = sidecar.get("file")
                if not isinstance(record, Mapping):
                    raise ObservedAnchorError("anchor rank sidecar lacks file evidence")
                path = Path(str(record.get("path", ""))).resolve()
                if dict(record) != vlf.file_record(path):
                    raise ObservedAnchorError("anchor rank shard changed before merge")
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                hashes = sidecar.get("row_sha256")
                if (
                    tuple(values.shape) != (len(expected_indices), *ANCHOR_SHAPE)
                    or values.dtype != np.float16
                    or not isinstance(hashes, list)
                    or len(hashes) != len(expected_indices)
                ):
                    raise ObservedAnchorError("anchor rank shard is malformed")
                for local_index, manifest_index in enumerate(expected_indices):
                    if manifest_index in covered:
                        raise ObservedAnchorError("anchor rank shards overlap")
                    actual_hash = _row_sha256(values[local_index])
                    if actual_hash != hashes[local_index]:
                        raise ObservedAnchorError(
                            "anchor rank row changed before merge"
                        )
                    merged[manifest_index] = values[local_index]
                    covered.add(manifest_index)
                    if manifest_index == 0:
                        online_sha_at_zero = actual_hash
                shard_records.append(dict(sidecar))
            if covered != set(range(len(rows))) or online_sha_at_zero is None:
                raise ObservedAnchorError(
                    "anchor rank shards do not cover the population"
                )
            merged.flush()
            os.replace(temporary, final_path)
            cached = np.load(final_path, mmap_mode="r", allow_pickle=False)
            cached_sha_at_zero = _row_sha256(cached[0])
            if cached_sha_at_zero != online_sha_at_zero:
                raise ObservedAnchorError("cached and online anchor bytes differ")
            metadata = {
                **identity,
                "artifact_identity": identity,
                "status": "complete",
                "complete": True,
                "cache_id": cache_id,
                "pca_file": vlf.file_record(pca_path),
                "semantic_cache_metadata": vlf.file_record(semantic_metadata_path),
                "anchor_file": vlf.file_record(final_path),
                "rank_shards": [
                    {
                        "rank": int(item["rank"]),
                        "rows": len(item["assigned_indices"]),
                        "sidecar": vlf.file_record(
                            work_root / f"rank-{int(item['rank']):05d}.json"
                        ),
                        "data_file": dict(item["file"]),
                    }
                    for item in shard_records
                ],
                "cached_online_agreement": {
                    "split": split,
                    "manifest_index": 0,
                    "clip_id": str(rows[0]["clip_id"]),
                    "online_float16_sha256": online_sha_at_zero,
                    "cached_float16_sha256": cached_sha_at_zero,
                    "max_abs_error": 0.0,
                },
                "cache_producer_attestation": dataset.producer_attestation,
            }
            vlf.atomic_write_json(metadata_path, metadata, exclusive=True)
            validate_anchor_cache(
                manifest_path=manifest,
                anchor_cache_root=root,
                expected_split=split,
            )
        context.barrier()
        return 0
    finally:
        vlf.close_distributed(context)


def _normalization_payload_vectors(payload: Mapping[str, Any], name: str) -> Tensor:
    value = payload.get(name)
    if not isinstance(value, list) or len(value) != TARGET_CHANNELS:
        raise ObservedAnchorError(
            f"normalization {name} must contain {TARGET_CHANNELS} numbers"
        )
    tensor = torch.tensor(value, dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        raise ObservedAnchorError(f"normalization {name} is non-finite")
    return tensor


def load_increment_normalization(
    path: str | Path,
    *,
    expected_train_manifest_sha256: str | None = None,
    expected_semantic_cache_metadata_sha256: str | None = None,
    expected_anchor_cache_metadata_sha256: str | None = None,
) -> tuple[ObservedIncrementNormalization, dict[str, Any]]:
    """Load one immutable normalization fitted on all 64k train clips."""
    resolved = Path(path).expanduser().resolve()
    payload = _read_json(resolved, label="observed-increment normalization")
    if (
        payload.get("schema") != NORMALIZATION_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("complete") is not True
        or payload.get("target_kind") != INCREMENT_TARGET_KIND
        or payload.get("split") != "train"
        or payload.get("clips") != screen.FROZEN_TRAIN_CLIPS
        or payload.get("target_shape") != list(INCREMENT_SHAPE)
        or payload.get("source_storage_dtype") != "float16"
        or payload.get("statistics_compute_dtype") != "float64"
        or payload.get("encode_decode_dtype") != "float32"
        or payload.get("std_floor") != STD_FLOOR
        or payload.get("statistics_source")
        != "frozen_train_semantic_and_observed_anchor_caches_only"
        or payload.get("population_elements_per_channel")
        != EXPECTED_INCREMENT_ELEMENTS_PER_CHANNEL
        or payload.get("channel_axis") != CHANNEL_AXIS
        or payload.get("temporal_axis") != TEMPORAL_AXIS
        or payload.get("increment_definition")
        != "D0=S0-A; Dj=Sj-S[j-1] for j=1..7"
        or payload.get("declared_roundtrip_max_abs_tolerance")
        != ROUNDTRIP_TOLERANCE
        or isinstance(payload.get("roundtrip_max_abs_error"), bool)
        or not isinstance(payload.get("roundtrip_max_abs_error"), (int, float))
        or not np.isfinite(float(payload["roundtrip_max_abs_error"]))
        or float(payload["roundtrip_max_abs_error"]) > ROUNDTRIP_TOLERANCE
        or payload.get("protected_test_accessed") is not False
        or payload.get("test_rows_used") != 0
    ):
        raise ObservedAnchorError("observed-increment normalization is malformed")
    for key, expected in (
        ("train_manifest_sha256", expected_train_manifest_sha256),
        (
            "semantic_cache_metadata_sha256",
            expected_semantic_cache_metadata_sha256,
        ),
        ("anchor_cache_metadata_sha256", expected_anchor_cache_metadata_sha256),
    ):
        actual = payload.get(key)
        if not isinstance(actual, str) or not screen.HEX64.fullmatch(actual):
            raise ObservedAnchorError(f"normalization lacks a valid {key}")
        if expected is not None and actual != expected:
            raise ObservedAnchorError(f"normalization {key} differs from data")
    implementation = payload.get("implementation")
    if not isinstance(implementation, Mapping) or implementation.get("dirty") is not False:
        raise ObservedAnchorError("normalization lacks clean implementation provenance")
    dependencies = implementation.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise ObservedAnchorError("normalization lacks implementation file bindings")
    try:
        verified_files = vlf._verify_embedded_file_records(  # noqa: SLF001
            dependencies, label="observed-increment normalization implementation"
        )
    except vlf.PocError as exc:
        raise ObservedAnchorError(str(exc)) from exc
    if verified_files < 5:
        raise ObservedAnchorError("normalization implementation evidence is incomplete")
    evidence_bindings = (
        ("train_manifest", "train_manifest_sha256"),
        ("semantic_cache_metadata", "semantic_cache_metadata_sha256"),
        ("anchor_cache_metadata", "anchor_cache_metadata_sha256"),
    )
    for record_name, digest_name in evidence_bindings:
        record = payload.get(record_name)
        if (
            not isinstance(record, Mapping)
            or dict(record) != vlf.file_record(Path(str(record.get("path", ""))))
            or record.get("sha256") != payload.get(digest_name)
        ):
            raise ObservedAnchorError(
                f"normalization {record_name} evidence is missing or changed"
            )
    normalization = ObservedIncrementNormalization(
        mean=_normalization_payload_vectors(payload, "increment_mean"),
        std=_normalization_payload_vectors(payload, "increment_std"),
        provenance=payload,
    )
    return normalization, {
        **vlf.file_record(resolved),
        "payload_sha256": _canonical_sha256(payload),
    }


def fit_normalization_command(args: argparse.Namespace) -> int:
    """Fit channel moments in float64 from float16 train-cache values."""
    output = vlf.approved_artifact_path(args.output)
    if output.exists():
        raise ObservedAnchorError(f"immutable normalization exists: {output}")
    manifest, rows, split = _split_manifest(args.train_manifest, expected_split="train")
    if split != "train" or len(rows) != screen.FROZEN_TRAIN_CLIPS:
        raise ObservedAnchorError(
            "normalization requires the complete train population"
        )
    dataset = construct_observed_anchor_dataset(
        manifest,
        args.data_root,
        args.semantic_cache_root,
        args.anchor_cache_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    sum_ = torch.zeros(TARGET_CHANNELS, dtype=torch.float64)
    square_sum = torch.zeros_like(sum_)
    count = 0
    first_semantic: Tensor | None = None
    first_anchor: Tensor | None = None
    for batch in loader:
        semantic_source = batch["auxiliary_target"]
        anchor_source = batch["observed_anchor"]
        if (
            semantic_source.dtype != torch.float16
            or anchor_source.dtype != torch.float16
        ):
            raise ObservedAnchorError("normalization source values must remain float16")
        increments = anchored_increments(
            semantic_source.to(torch.float64), anchor_source.to(torch.float64)
        )
        sum_ += increments.sum(dim=(0, 2, 3, 4))
        square_sum += increments.square().sum(dim=(0, 2, 3, 4))
        count += increments.shape[0] * FUTURE_SIZE * ANCHOR_SHAPE[1] * ANCHOR_SHAPE[2]
        if first_semantic is None:
            first_semantic = semantic_source[:1].float()
            first_anchor = anchor_source[:1].float()
    expected_count = len(dataset) * FUTURE_SIZE * ANCHOR_SHAPE[1] * ANCHOR_SHAPE[2]
    if count != expected_count or first_semantic is None or first_anchor is None:
        raise ObservedAnchorError("normalization did not consume every train increment")
    mean64 = sum_ / count
    variance64 = (square_sum / count - mean64.square()).clamp_min(0.0)
    std64 = variance64.sqrt().clamp_min(STD_FLOOR)
    normalization = ObservedIncrementNormalization(
        mean64.float(), std64.float(), {"pending": True}
    )
    target = encode_normalized_increment_target(
        first_semantic, first_anchor, normalization
    )
    recovered = decode_normalized_increment_prediction(
        target, first_anchor, normalization
    )
    roundtrip = float((recovered - first_semantic).abs().max())
    if roundtrip > ROUNDTRIP_TOLERANCE:
        raise ObservedAnchorError(
            f"float32 normalization round trip {roundtrip} exceeds tolerance"
        )
    semantic_metadata_path = (
        Path(args.semantic_cache_root).expanduser().resolve()
        / "train"
        / "metadata.json"
    )
    anchor_metadata_path, _ = _resolve_anchor_paths(args.anchor_cache_root, "train")
    payload = {
        "schema": NORMALIZATION_SCHEMA,
        "status": "complete",
        "complete": True,
        "target_kind": INCREMENT_TARGET_KIND,
        "split": "train",
        "clips": len(dataset),
        "target_shape": list(INCREMENT_SHAPE),
        "channel_axis": CHANNEL_AXIS,
        "temporal_axis": TEMPORAL_AXIS,
        "source_storage_dtype": "float16",
        "statistics_compute_dtype": "float64",
        "encode_decode_dtype": "float32",
        "statistics_source": ("frozen_train_semantic_and_observed_anchor_caches_only"),
        "increment_definition": "D0=S0-A; Dj=Sj-S[j-1] for j=1..7",
        "population_elements_per_channel": count,
        "increment_mean": mean64.tolist(),
        "increment_std": std64.tolist(),
        "std_floor": STD_FLOOR,
        "roundtrip_sample_manifest_index": 0,
        "roundtrip_max_abs_error": roundtrip,
        "declared_roundtrip_max_abs_tolerance": ROUNDTRIP_TOLERANCE,
        "train_manifest_sha256": sha256_file(manifest),
        "semantic_cache_metadata_sha256": sha256_file(semantic_metadata_path),
        "anchor_cache_metadata_sha256": sha256_file(anchor_metadata_path),
        "train_manifest": vlf.file_record(manifest),
        "semantic_cache_metadata": vlf.file_record(semantic_metadata_path),
        "anchor_cache_metadata": vlf.file_record(anchor_metadata_path),
        "implementation": _source_record(),
        "protected_test_accessed": False,
        "test_rows_used": 0,
    }
    vlf.atomic_write_json(output, payload, exclusive=True)
    load_increment_normalization(
        output,
        expected_train_manifest_sha256=sha256_file(manifest),
        expected_semantic_cache_metadata_sha256=sha256_file(semantic_metadata_path),
        expected_anchor_cache_metadata_sha256=sha256_file(anchor_metadata_path),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-cache")
    build.add_argument("--manifest", required=True)
    build.add_argument("--data-root", required=True)
    build.add_argument("--semantic-cache-root", required=True)
    build.add_argument("--vjepa-source", required=True)
    build.add_argument("--vjepa-checkpoint", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--batch-size", type=int, default=1)
    build.add_argument("--resume", action="store_true")

    fit = commands.add_parser("fit-normalization")
    fit.add_argument("--output", required=True)
    fit.add_argument("--train-manifest", required=True)
    fit.add_argument("--data-root", required=True)
    fit.add_argument("--semantic-cache-root", required=True)
    fit.add_argument("--anchor-cache-root", required=True)
    fit.add_argument("--batch-size", type=int, default=64)
    fit.add_argument("--workers", type=int, default=4)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "build-cache" and args.batch_size != 1:
        raise ObservedAnchorError("anchor cache extraction batch size is frozen at one")
    if args.command == "fit-normalization" and (
        args.batch_size < 1 or args.workers < 0
    ):
        raise ObservedAnchorError("normalization batch/workers arguments are invalid")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        if args.command == "build-cache":
            return build_cache_command(args)
        return fit_normalization_command(args)
    except (
        ObservedAnchorError,
        screen.ScreenError,
        vlf.PocError,
        ValueError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"Observed-anchor error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
