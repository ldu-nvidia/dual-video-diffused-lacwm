#!/usr/bin/env python3
"""Fit train-only V-JEPA 2.1 PCA statistics and build offline target caches.

This is deliberately an *offline* tool.  It runs the frozen, pinned V-JEPA
teacher in a separate Python >=3.11 environment and writes compact tensors
that the LACWM trainer memory-maps.  The trainer neither imports V-JEPA nor
calls a teacher at training or inference time.

Pipeline
--------
1. ``validate-inputs`` verifies the exact official source commit and an
   explicitly supplied checkpoint SHA-256 (as well as the expected byte size).
2. ``fit-pca`` extracts each camera independently, aligns V-JEPA tubelets to
   the four Wan temporal bins, and fits a 64-channel PCA whitening transform
   using *only* a manifest whose rows are marked ``split=train``.
3. ``extract`` applies those frozen statistics and writes a resumable FP16
   ``[N, 64, 4, 24, 120]`` target cache together with the exact quantized RGB
   ``[N,13,3,180,960]`` and explicit actions ``[N,13,5,23]`` consumed by the
   trainer. Metadata stays ``complete=false`` until all arrays are finite,
   flushed, and content-hashed.
4. ``validate-pca`` and ``validate-cache`` perform read-only integrity checks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_vjepa2_clip_manifests import (  # noqa: E402
    ACTION_SPAN,
    CAMERAS,
    CHUNK_SIZE,
    FRAME_OFFSETS,
    SAMPLE_SIZE,
    ManifestError,
    sha256_file,
    validate_clip_rows,
)
from robot_wm.modeling.dual_diffusion.vjepa2_target import (  # noqa: E402
    PCAWhiteningStats,
    VJEPA2_1_CHECKPOINT_BYTES,
    VJEPA2_1_MODEL_NAME,
    VJEPA2_1_RELEASE_COMMIT,
    VJEPA2_1_SOURCE_DIM,
    align_vjepa2_1_tokens,
    extract_vjepa2_1_target,
    load_vjepa2_1_vit_base_encoder,
    prepare_vjepa2_1_views,
    validate_vjepa_source,
)


PCA_FORMAT_VERSION = 1
CACHE_FORMAT_VERSION = 1
TARGET_CHANNELS = 64
TARGET_SHAPE = (TARGET_CHANNELS, 4, 24, 120)
TOKENS_PER_CLIP = 3 * 4 * 24 * 40
RGB_SIZE = (180, 320)
RGB_CACHE_SHAPE = (SAMPLE_SIZE, 3, RGB_SIZE[0], RGB_SIZE[1] * len(CAMERAS))
ACTION_CACHE_SHAPE = (SAMPLE_SIZE, CHUNK_SIZE, 23)
VJEPA2_1_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/vjepa2/"
    "vjepa2_1_vitb_dist_vitG_384.pt"
)


class ExtractionError(RuntimeError):
    """An offline extraction or artifact-integrity check failed."""


def _validate_sha256(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64:
        raise ExtractionError(f"{label} must be a full 64-character SHA-256")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ExtractionError(f"{label} is not hexadecimal") from exc
    return normalized


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _pca_runtime_metadata() -> dict[str, str]:
    """Return weights-only-safe scalar metadata for the PCA artifact."""
    # PyTorch 2.6 exposes ``torch.__version__`` as a ``TorchVersion`` (a
    # ``str`` subclass with a non-allowlisted pickle global).  Persist a plain
    # built-in string so our mandatory ``weights_only=True`` validation can
    # reload the artifact without weakening the safe-load boundary.
    return {"torch_version": str(torch.__version__)}


def validate_teacher_inputs(
    *,
    source_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Strictly verify the pinned tracked source and official checkpoint."""
    source = validate_vjepa_source(
        source_path, expected_commit=VJEPA2_1_RELEASE_COMMIT
    )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ExtractionError(
            "V-JEPA source has tracked modifications; use a clean checkout of "
            f"{VJEPA2_1_RELEASE_COMMIT}"
        )

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise ExtractionError(f"V-JEPA checkpoint is missing: {checkpoint}")
    if checkpoint.stat().st_size != VJEPA2_1_CHECKPOINT_BYTES:
        raise ExtractionError(
            "V-JEPA checkpoint byte-size mismatch: expected "
            f"{VJEPA2_1_CHECKPOINT_BYTES}, found {checkpoint.stat().st_size}"
        )
    expected_digest = _validate_sha256(
        checkpoint_sha256, label="checkpoint SHA-256"
    )
    actual_digest = sha256_file(checkpoint)
    if actual_digest != expected_digest:
        raise ExtractionError(
            "V-JEPA checkpoint SHA-256 mismatch: "
            f"expected {expected_digest}, found {actual_digest}"
        )
    return {
        "model_name": VJEPA2_1_MODEL_NAME,
        "source_path": str(source),
        "source_commit": VJEPA2_1_RELEASE_COMMIT,
        "checkpoint_path": str(checkpoint),
        "checkpoint_url": VJEPA2_1_CHECKPOINT_URL,
        "checkpoint_bytes": VJEPA2_1_CHECKPOINT_BYTES,
        "checkpoint_sha256": actual_digest,
    }


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExtractionError("CUDA was requested but is not available")
    if device.type not in {"cpu", "cuda"}:
        raise ExtractionError("only cpu and cuda devices are supported")
    return device


def _encoder_dtype(value: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[value]


def _load_teacher(
    *,
    provenance: Mapping[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    encoder = load_vjepa2_1_vit_base_encoder(
        source_path=provenance["source_path"],
        checkpoint_path=provenance["checkpoint_path"],
        expected_source_commit=provenance["source_commit"],
        expected_checkpoint_sha256=provenance["checkpoint_sha256"],
    )
    encoder.to(device=device)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def _read_camera_frames(
    episode_dir: Path,
    camera: str,
    frame_indices: Sequence[int],
) -> torch.Tensor:
    try:
        import decord
    except ImportError as exc:
        raise ExtractionError("decord is required for target extraction") from exc

    video_path = episode_dir / f"{camera}.mp4"
    try:
        reader = decord.VideoReader(str(video_path))
    except Exception as exc:
        raise ExtractionError(f"cannot open ABC video {video_path}: {exc}") from exc
    if max(frame_indices) >= len(reader):
        raise ExtractionError(
            f"clip frame {max(frame_indices)} exceeds {video_path} "
            f"with {len(reader)} frames"
        )
    try:
        array = reader.get_batch(list(frame_indices)).asnumpy()
    except Exception as exc:
        raise ExtractionError(f"cannot decode {video_path}: {exc}") from exc
    if array.ndim != 4 or array.shape[0] != SAMPLE_SIZE or array.shape[-1] != 3:
        raise ExtractionError(
            f"unexpected decoded RGB shape for {video_path}: {array.shape}"
        )
    video = torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2)
    video = F.interpolate(
        video.float(),
        size=RGB_SIZE,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return video.div_(127.5).sub_(1.0)


def load_rgb_clip(row: Mapping[str, Any]) -> torch.Tensor:
    """Decode one manifest row to ``[13,3,180,960]`` in ``[-1,1]``."""
    start = int(row["start"])
    expected = [start + offset for offset in FRAME_OFFSETS]
    frame_indices = [int(value) for value in row["frame_indices"]]
    if frame_indices != expected:
        raise ExtractionError(
            f"clip {row.get('clip_id')} does not use the required 13x5 frame map"
        )
    episode = Path(row["episode_dir"]).expanduser().resolve()
    views = [
        _read_camera_frames(episode, camera, frame_indices) for camera in CAMERAS
    ]
    return torch.cat(views, dim=-1)


def load_action_clip(row: Mapping[str, Any]) -> np.ndarray:
    """Load the exact explicit-action tensor ``[13,5,23]`` for one clip."""
    episode = Path(row["episode_dir"]).expanduser().resolve()
    state_path = episode / "states.npz"
    try:
        with np.load(state_path, allow_pickle=False) as state:
            joint = np.asarray(state["joint_actions"], dtype=np.float32)
            gripper = np.asarray(state["gripper_actions"], dtype=np.float32)
    except Exception as exc:
        raise ExtractionError(f"cannot read ABC actions {state_path}: {exc}") from exc
    if joint.ndim != 2 or gripper.ndim != 2 or len(joint) != len(gripper):
        raise ExtractionError(f"malformed ABC action arrays in {state_path}")
    actions = np.concatenate(
        [
            joint,
            gripper,
            np.zeros((len(joint), 9), dtype=np.float32),
        ],
        axis=1,
    )
    start = int(row["start"])
    selected = actions[start : start + ACTION_SPAN]
    if selected.shape != (ACTION_SPAN, ACTION_CACHE_SHAPE[-1]):
        raise ExtractionError(
            f"clip actions exceed {state_path}: got {selected.shape}"
        )
    return np.ascontiguousarray(selected.reshape(ACTION_CACHE_SHAPE))


def quantize_cached_rgb(video: torch.Tensor) -> torch.Tensor:
    """Make teacher input exactly equal to the immutable FP16 trainer cache."""
    if tuple(video.shape[1:]) != RGB_CACHE_SHAPE:
        raise ExtractionError(
            f"unexpected RGB cache shape: {tuple(video.shape)}"
        )
    quantized = video.to(torch.float16)
    if not torch.isfinite(quantized).all():
        raise ExtractionError("quantized RGB cache contains non-finite values")
    return quantized


def _batches(
    rows: Sequence[Mapping[str, Any]],
    *,
    start: int,
    batch_size: int,
) -> Iterator[tuple[int, Sequence[Mapping[str, Any]]]]:
    if batch_size <= 0:
        raise ExtractionError("batch-size must be positive")
    for offset in range(start, len(rows), batch_size):
        yield offset, rows[offset : offset + batch_size]


@torch.inference_mode()
def _extract_aligned_tokens(
    video: torch.Tensor,
    *,
    encoder: torch.nn.Module,
    encoder_dtype: torch.dtype,
) -> torch.Tensor:
    """Return independent-view dense features ``[B,3,4,24,40,768]``."""
    prepared = prepare_vjepa2_1_views(video, num_views=len(CAMERAS))
    autocast_enabled = prepared.device.type == "cuda"
    with torch.autocast(
        device_type=prepared.device.type,
        dtype=encoder_dtype,
        enabled=autocast_enabled,
    ):
        tokens = encoder(prepared, training=False)
    if not isinstance(tokens, torch.Tensor):
        raise ExtractionError("V-JEPA encoder did not return one dense tensor")
    aligned = align_vjepa2_1_tokens(
        tokens,
        batch_size=video.shape[0],
        num_views=len(CAMERAS),
    )
    if tuple(aligned.shape[1:]) != (3, 4, 24, 40, VJEPA2_1_SOURCE_DIM):
        raise ExtractionError(f"unexpected aligned V-JEPA shape: {aligned.shape}")
    if not torch.isfinite(aligned).all():
        raise ExtractionError("V-JEPA emitted non-finite dense features")
    return aligned


def _clip_sample_indices(
    *,
    clip_id: str,
    seed: int,
    count: int,
) -> torch.Tensor:
    if count < 0 or count > TOKENS_PER_CLIP:
        raise ExtractionError("invalid per-clip PCA sample count")
    digest = hashlib.sha256(
        f"vjepa2.1-pca64\0{seed}\0{clip_id}".encode("utf-8")
    ).digest()
    rng = np.random.Generator(
        np.random.PCG64(int.from_bytes(digest[:16], "big"))
    )
    indexes = rng.choice(TOKENS_PER_CLIP, size=count, replace=False)
    indexes.sort()
    return torch.from_numpy(indexes.astype(np.int64, copy=False))


def _pca_metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def fit_train_pca(
    *,
    train_manifest: Path,
    output_path: Path,
    provenance: Mapping[str, Any],
    device: torch.device,
    pca_device: torch.device,
    encoder_dtype: torch.dtype,
    batch_size: int,
    max_tokens: int,
    max_clips: int | None,
    seed: int,
    pca_oversample: int,
    pca_iterations: int,
    whitening_eps: float,
    overwrite: bool,
) -> dict[str, Any]:
    train_manifest = train_manifest.expanduser().resolve()
    rows = validate_clip_rows(train_manifest, expected_split="train")
    if max_clips is not None:
        if max_clips <= 0:
            raise ExtractionError("max-clips must be positive")
        rows = rows[:max_clips]
    if not rows:
        raise ExtractionError("no training clips selected for PCA")
    if max_tokens <= TARGET_CHANNELS:
        raise ExtractionError(
            f"max-tokens must exceed the {TARGET_CHANNELS} output channels"
        )
    if whitening_eps <= 0:
        raise ExtractionError("whitening-eps must be positive")
    if pca_oversample < 0 or pca_iterations < 1:
        raise ExtractionError("invalid PCA oversampling or iteration count")

    output_path = output_path.expanduser().resolve()
    companion = _pca_metadata_path(output_path)
    if (output_path.exists() or companion.exists()) and not overwrite:
        raise ExtractionError(
            f"PCA artifact already exists: {output_path}; pass --overwrite "
            "only for an intentional rebuild"
        )

    budget = min(int(max_tokens), len(rows) * TOKENS_PER_CLIP)
    base_quota, extra = divmod(budget, len(rows))
    quotas = [
        base_quota + (1 if index < extra else 0) for index in range(len(rows))
    ]
    if max(quotas) > TOKENS_PER_CLIP:
        raise AssertionError("internal PCA token budget error")

    encoder = _load_teacher(provenance=provenance, device=device)
    samples: list[torch.Tensor] = []
    sampled = 0
    processed = 0
    for offset, batch_rows in _batches(rows, start=0, batch_size=batch_size):
        video = quantize_cached_rgb(
            torch.stack([load_rgb_clip(row) for row in batch_rows])
        ).to(device=device, dtype=torch.float32, non_blocking=True)
        aligned = _extract_aligned_tokens(
            video, encoder=encoder, encoder_dtype=encoder_dtype
        )
        for local_index, row in enumerate(batch_rows):
            quota = quotas[offset + local_index]
            indexes = _clip_sample_indices(
                clip_id=str(row["clip_id"]), seed=seed, count=quota
            ).to(device=aligned.device)
            flat = aligned[local_index].reshape(
                TOKENS_PER_CLIP, VJEPA2_1_SOURCE_DIM
            )
            samples.append(
                flat.index_select(0, indexes).float().cpu().contiguous()
            )
            sampled += quota
        processed += len(batch_rows)
        print(
            f"PCA extraction: {processed}/{len(rows)} clips, "
            f"{sampled}/{budget} tokens",
            flush=True,
        )

    del video, aligned
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    matrix = torch.cat(samples, dim=0)
    del samples
    if matrix.shape != (budget, VJEPA2_1_SOURCE_DIM):
        raise ExtractionError(f"unexpected PCA sample matrix shape: {matrix.shape}")
    matrix = matrix.to(device=pca_device, dtype=torch.float32)
    mean = matrix.mean(dim=0)
    matrix.sub_(mean)
    q = min(
        VJEPA2_1_SOURCE_DIM,
        matrix.shape[0] - 1,
        TARGET_CHANNELS + pca_oversample,
    )
    if q < TARGET_CHANNELS:
        raise ExtractionError("not enough PCA samples for 64 components")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _, singular_values, right_vectors = torch.pca_lowrank(
        matrix,
        q=q,
        center=False,
        niter=pca_iterations,
    )
    components = right_vectors[:, :TARGET_CHANNELS].transpose(0, 1).contiguous()
    eigenvalues = singular_values[:TARGET_CHANNELS].square().div(matrix.shape[0] - 1)

    # Canonicalize the arbitrary PCA signs for stable downstream artifacts.
    pivots = components.abs().argmax(dim=1, keepdim=True)
    signs = components.gather(1, pivots).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    components.mul_(signs)
    stats = PCAWhiteningStats(
        mean=mean.detach().cpu(),
        components=components.detach().cpu(),
        eigenvalues=eigenvalues.detach().cpu(),
        eps=whitening_eps,
    )
    payload = stats.to_payload()
    payload.update(
        {
            "artifact_type": "vjepa2.1-pca64-whitening",
            "model_name": VJEPA2_1_MODEL_NAME,
            "source_commit": provenance["source_commit"],
            "checkpoint_url": provenance["checkpoint_url"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "checkpoint_bytes": provenance["checkpoint_bytes"],
            "train_manifest_sha256": sha256_file(train_manifest),
            "train_manifest_clip_count": len(
                validate_clip_rows(train_manifest, expected_split="train")
            ),
            "pca_clip_count": len(rows),
            "sampled_token_count": int(matrix.shape[0]),
            "source_dimension": VJEPA2_1_SOURCE_DIM,
            "target_channels": TARGET_CHANNELS,
            "sample_size": SAMPLE_SIZE,
            "chunk_size": CHUNK_SIZE,
            "action_span": ACTION_SPAN,
            "frame_offsets": list(FRAME_OFFSETS),
            "camera_order": list(CAMERAS),
            "aligned_shape_per_clip": [3, 4, 24, 40, VJEPA2_1_SOURCE_DIM],
            "pca_seed": int(seed),
            "pca_rank_computed": int(q),
            "pca_iterations": int(pca_iterations),
            **_pca_runtime_metadata(),
        }
    )
    _atomic_torch_save(output_path, payload)
    artifact_digest = sha256_file(output_path)
    summary = {
        "format_version": PCA_FORMAT_VERSION,
        "artifact_type": payload["artifact_type"],
        "artifact_file": output_path.name,
        "artifact_sha256": artifact_digest,
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": payload["train_manifest_sha256"],
        "source_commit": payload["source_commit"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "input_dimension": stats.input_dim,
        "output_dimension": stats.output_dim,
        "pca_clip_count": payload["pca_clip_count"],
        "sampled_token_count": payload["sampled_token_count"],
        "whitening_eps": stats.eps,
    }
    _atomic_write_json(companion, summary)
    return summary


def load_and_validate_pca(
    *,
    pca_path: Path,
    train_manifest: Path,
    expected_source_commit: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> tuple[PCAWhiteningStats, dict[str, Any], str]:
    pca_path = pca_path.expanduser().resolve()
    train_manifest = train_manifest.expanduser().resolve()
    if not pca_path.is_file():
        raise ExtractionError(f"PCA artifact is missing: {pca_path}")
    validate_clip_rows(train_manifest, expected_split="train")
    try:
        payload = torch.load(
            pca_path, map_location="cpu", weights_only=True, mmap=True
        )
    except Exception as exc:
        raise ExtractionError(f"cannot read PCA artifact {pca_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExtractionError("PCA artifact payload must be a mapping")
    stats = PCAWhiteningStats.from_payload(payload)
    if stats.input_dim != VJEPA2_1_SOURCE_DIM:
        raise ExtractionError(
            f"PCA input dimension is {stats.input_dim}, expected {VJEPA2_1_SOURCE_DIM}"
        )
    if stats.output_dim != TARGET_CHANNELS:
        raise ExtractionError(
            f"PCA output dimension is {stats.output_dim}, expected {TARGET_CHANNELS}"
        )
    if payload.get("artifact_type") != "vjepa2.1-pca64-whitening":
        raise ExtractionError("unexpected PCA artifact type")
    if payload.get("model_name") != VJEPA2_1_MODEL_NAME:
        raise ExtractionError("PCA artifact model identity mismatch")
    if payload.get("source_commit") != VJEPA2_1_RELEASE_COMMIT:
        raise ExtractionError("PCA artifact was not fit with the pinned source")
    if payload.get("train_manifest_sha256") != sha256_file(train_manifest):
        raise ExtractionError("PCA artifact does not match the training manifest")
    if payload.get("frame_offsets") != list(FRAME_OFFSETS):
        raise ExtractionError("PCA artifact frame map mismatch")
    if payload.get("camera_order") != list(CAMERAS):
        raise ExtractionError("PCA artifact camera order mismatch")
    if (
        int(payload.get("sample_size", -1)) != SAMPLE_SIZE
        or int(payload.get("chunk_size", -1)) != CHUNK_SIZE
        or int(payload.get("action_span", -1)) != ACTION_SPAN
    ):
        raise ExtractionError("PCA artifact clip schema mismatch")
    if int(payload.get("source_dimension", -1)) != VJEPA2_1_SOURCE_DIM:
        raise ExtractionError("PCA artifact source-dimension metadata mismatch")
    if int(payload.get("target_channels", -1)) != TARGET_CHANNELS:
        raise ExtractionError("PCA artifact channel metadata mismatch")
    if payload.get("checkpoint_url") != VJEPA2_1_CHECKPOINT_URL:
        raise ExtractionError("PCA artifact checkpoint URL mismatch")
    if int(payload.get("sampled_token_count", -1)) <= TARGET_CHANNELS:
        raise ExtractionError("PCA artifact records too few sampled tokens")
    if int(payload.get("pca_clip_count", -1)) <= 0:
        raise ExtractionError("PCA artifact records no training clips")
    _validate_sha256(
        str(payload.get("checkpoint_sha256", "")),
        label="PCA checkpoint SHA-256",
    )
    if expected_source_commit is not None and payload.get(
        "source_commit"
    ) != expected_source_commit:
        raise ExtractionError("PCA/source provenance mismatch")
    if expected_checkpoint_sha256 is not None and payload.get(
        "checkpoint_sha256"
    ) != expected_checkpoint_sha256:
        raise ExtractionError("PCA/checkpoint provenance mismatch")

    artifact_digest = sha256_file(pca_path)
    companion = _pca_metadata_path(pca_path)
    if not companion.is_file():
        raise ExtractionError(f"PCA companion metadata is missing: {companion}")
    with companion.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if int(metadata.get("format_version", -1)) != PCA_FORMAT_VERSION:
        raise ExtractionError("unsupported PCA companion format")
    if metadata.get("artifact_sha256") != artifact_digest:
        raise ExtractionError("PCA companion SHA-256 mismatch")
    if metadata.get("train_manifest_sha256") != sha256_file(train_manifest):
        raise ExtractionError("PCA companion training-manifest mismatch")
    if (
        int(metadata.get("input_dimension", -1)) != stats.input_dim
        or int(metadata.get("output_dimension", -1)) != stats.output_dim
    ):
        raise ExtractionError("PCA companion dimension mismatch")
    return stats, payload, artifact_digest


def _new_cache_metadata(
    *,
    rows: Sequence[Mapping[str, Any]],
    split: str,
    clip_manifest: Path,
    train_manifest: Path,
    pca_path: Path,
    pca_digest: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    target_shape = [len(rows), *TARGET_SHAPE]
    identity = {
        "schema": "vjepa2.1-wan-grid-cache-v1",
        "split": split,
        "clip_manifest_sha256": sha256_file(clip_manifest),
        "train_manifest_sha256": sha256_file(train_manifest),
        "pca_sha256": pca_digest,
        "source_commit": provenance["source_commit"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "target_shape": target_shape,
        "rgb_shape": [len(rows), *RGB_CACHE_SHAPE],
        "actions_shape": [len(rows), *ACTION_CACHE_SHAPE],
    }
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "artifact_type": "vjepa2.1-wan-grid-cache",
        "cache_id": hashlib.sha256(
            _canonical_json(identity).encode("utf-8")
        ).hexdigest(),
        "complete": False,
        "written_rows": 0,
        "split": split,
        "clip_count": len(rows),
        "clip_manifest": str(clip_manifest),
        "clip_manifest_sha256": identity["clip_manifest_sha256"],
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": identity["train_manifest_sha256"],
        "pca_file": str(pca_path),
        "pca_sha256": pca_digest,
        "provenance": {
            # Names consumed by tools/vjepa2_controlled_study.py.
            "vjepa_source_commit": provenance["source_commit"],
            "vjepa_checkpoint_sha256": provenance["checkpoint_sha256"],
            "pca_stats_sha256": pca_digest,
            # Additional inputs needed to reproduce the projection.
            "train_manifest_sha256": identity["train_manifest_sha256"],
            "clip_manifest_sha256": identity["clip_manifest_sha256"],
        },
        "model_name": VJEPA2_1_MODEL_NAME,
        "source_commit": provenance["source_commit"],
        "checkpoint_url": provenance["checkpoint_url"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "checkpoint_bytes": provenance["checkpoint_bytes"],
        "sample_size": SAMPLE_SIZE,
        "chunk_size": CHUNK_SIZE,
        "action_span": ACTION_SPAN,
        "frame_offsets": list(FRAME_OFFSETS),
        "camera_order": list(CAMERAS),
        "target_file": "targets.fp16.npy",
        "target_dtype": "float16",
        "target_shape": target_shape,
        "target_sha256": None,
        # These immutable arrays are the actual trainer inputs.  Caching them
        # removes any dependency on mutable source MP4/state files after
        # extraction and guarantees exact RGB/action alignment with the frozen
        # V-JEPA target.
        "rgb_file": "rgb.fp16.npy",
        "rgb_dtype": "float16",
        "rgb_shape": identity["rgb_shape"],
        "rgb_sha256": None,
        "actions_file": "actions.float32.npy",
        "actions_dtype": "float32",
        "actions_shape": identity["actions_shape"],
        "actions_sha256": None,
    }


def _assert_resume_metadata(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    mutable = {
        "complete",
        "written_rows",
        "target_sha256",
        "rgb_sha256",
        "actions_sha256",
    }
    for key, expected_value in expected.items():
        if key in mutable:
            continue
        if actual.get(key) != expected_value:
            raise ExtractionError(
                f"existing cache metadata differs at {key!r}; choose a new "
                "cache directory or pass --overwrite intentionally"
            )
    written = int(actual.get("written_rows", -1))
    if written < 0 or written > int(expected["clip_count"]):
        raise ExtractionError("existing cache has an invalid written_rows value")


def _validate_target_array(
    path: Path,
    *,
    shape: Sequence[int],
    finite_chunk_rows: int,
) -> np.memmap:
    if not path.is_file():
        raise ExtractionError(f"cached target array is missing: {path}")
    try:
        target = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise ExtractionError(f"cannot memory-map target array {path}: {exc}") from exc
    if tuple(target.shape) != tuple(int(value) for value in shape):
        raise ExtractionError(
            f"target array shape mismatch: {target.shape} versus {tuple(shape)}"
        )
    if target.dtype != np.float16:
        raise ExtractionError(f"target array dtype must be float16, got {target.dtype}")
    if finite_chunk_rows <= 0:
        raise ExtractionError("finite-check-rows must be positive")
    for start in range(0, len(target), finite_chunk_rows):
        if not np.isfinite(
            np.asarray(target[start : start + finite_chunk_rows])
        ).all():
            raise ExtractionError(
                f"target array has non-finite values near row {start}"
            )
    return target


def _validate_rgb_array(
    path: Path,
    *,
    shape: Sequence[int],
    finite_chunk_rows: int,
) -> np.memmap:
    if not path.is_file():
        raise ExtractionError(f"cached RGB array is missing: {path}")
    try:
        rgb = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise ExtractionError(f"cannot memory-map RGB cache {path}: {exc}") from exc
    if tuple(rgb.shape) != tuple(int(value) for value in shape):
        raise ExtractionError(
            f"RGB cache shape mismatch: {rgb.shape} versus {tuple(shape)}"
        )
    if rgb.dtype != np.float16:
        raise ExtractionError(f"RGB cache dtype must be float16, got {rgb.dtype}")
    if finite_chunk_rows <= 0:
        raise ExtractionError("finite-check-rows must be positive")
    for start in range(0, len(rgb), finite_chunk_rows):
        chunk = np.asarray(rgb[start : start + finite_chunk_rows])
        if not np.isfinite(chunk).all():
            raise ExtractionError(f"RGB cache is non-finite near row {start}")
        if chunk.min() < -1.0 or chunk.max() > 1.0:
            raise ExtractionError(f"RGB cache is outside [-1,1] near row {start}")
    return rgb


def _validate_actions_array(
    path: Path,
    *,
    shape: Sequence[int],
    finite_chunk_rows: int,
) -> np.memmap:
    if not path.is_file():
        raise ExtractionError(f"cached action array is missing: {path}")
    try:
        actions = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise ExtractionError(
            f"cannot memory-map action cache {path}: {exc}"
        ) from exc
    if tuple(actions.shape) != tuple(int(value) for value in shape):
        raise ExtractionError(
            f"action cache shape mismatch: {actions.shape} versus {tuple(shape)}"
        )
    if actions.dtype != np.float32:
        raise ExtractionError(
            f"action cache dtype must be float32, got {actions.dtype}"
        )
    if finite_chunk_rows <= 0:
        raise ExtractionError("finite-check-rows must be positive")
    for start in range(0, len(actions), finite_chunk_rows):
        if not np.isfinite(
            np.asarray(actions[start : start + finite_chunk_rows])
        ).all():
            raise ExtractionError(
                f"action cache is non-finite near row {start}"
            )
    return actions


def _publish_initialized_cache_directory(
    cache_dir: Path,
    metadata: Mapping[str, Any],
) -> None:
    """Atomically publish a complete zero-row cache initialization.

    A fresh cache used to create its three large arrays directly in
    ``cache_dir`` and write metadata last.  A process failure in that narrow
    window left arrays without a trustworthy ``written_rows`` boundary, so a
    no-overwrite retry had to fail closed.  Build the arrays and metadata in a
    unique sibling directory instead.  The directory rename is the publication
    point: before it, the requested cache does not exist; after it, all four
    files exist and metadata records ``written_rows=0``.

    Failed unpublished staging directories are deliberately retained for
    forensic/manual cleanup.  A retry uses another unique sibling and does not
    trust or overwrite the abandoned bytes.
    """
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        raise ExtractionError(
            f"cannot initialize an existing cache directory: {cache_dir}"
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{cache_dir.name}.initializing.",
            dir=cache_dir.parent,
        )
    )
    target = rgb = actions = None
    try:
        target = np.lib.format.open_memmap(
            staging / str(metadata["target_file"]),
            mode="w+",
            dtype=np.float16,
            shape=tuple(int(value) for value in metadata["target_shape"]),
        )
        rgb = np.lib.format.open_memmap(
            staging / str(metadata["rgb_file"]),
            mode="w+",
            dtype=np.float16,
            shape=tuple(int(value) for value in metadata["rgb_shape"]),
        )
        actions = np.lib.format.open_memmap(
            staging / str(metadata["actions_file"]),
            mode="w+",
            dtype=np.float32,
            shape=tuple(int(value) for value in metadata["actions_shape"]),
        )
        target.flush()
        rgb.flush()
        actions.flush()
        del target, rgb, actions
        target = rgb = actions = None
        _atomic_write_json(staging / "metadata.json", metadata)
        if cache_dir.exists():
            raise ExtractionError(
                f"cache directory appeared during initialization: {cache_dir}"
            )
        os.rename(staging, cache_dir)
    finally:
        # Do not recursively remove an unpublished directory: retaining it
        # makes a partial initialization auditable and avoids ever deleting an
        # operator-owned path.  The unique name cannot be mistaken for the
        # requested cache on a retry.
        del target, rgb, actions


def extract_cache(
    *,
    split: str,
    clip_manifest: Path,
    train_manifest: Path,
    pca_path: Path,
    cache_dir: Path,
    provenance: Mapping[str, Any],
    device: torch.device,
    encoder_dtype: torch.dtype,
    batch_size: int,
    finite_check_rows: int,
    overwrite: bool,
) -> dict[str, Any]:
    clip_manifest = clip_manifest.expanduser().resolve()
    train_manifest = train_manifest.expanduser().resolve()
    pca_path = pca_path.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    rows = validate_clip_rows(clip_manifest, expected_split=split)
    stats, _, pca_digest = load_and_validate_pca(
        pca_path=pca_path,
        train_manifest=train_manifest,
        expected_source_commit=provenance["source_commit"],
        expected_checkpoint_sha256=provenance["checkpoint_sha256"],
    )
    expected = _new_cache_metadata(
        rows=rows,
        split=split,
        clip_manifest=clip_manifest,
        train_manifest=train_manifest,
        pca_path=pca_path,
        pca_digest=pca_digest,
        provenance=provenance,
    )
    metadata_path = cache_dir / "metadata.json"
    target_path = cache_dir / expected["target_file"]
    rgb_path = cache_dir / expected["rgb_file"]
    actions_path = cache_dir / expected["actions_file"]

    if overwrite:
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = np.lib.format.open_memmap(
            target_path,
            mode="w+",
            dtype=np.float16,
            shape=tuple(expected["target_shape"]),
        )
        rgb_cache = np.lib.format.open_memmap(
            rgb_path,
            mode="w+",
            dtype=np.float16,
            shape=tuple(expected["rgb_shape"]),
        )
        actions_cache = np.lib.format.open_memmap(
            actions_path,
            mode="w+",
            dtype=np.float32,
            shape=tuple(expected["actions_shape"]),
        )
        metadata = dict(expected)
        _atomic_write_json(metadata_path, metadata)
    elif cache_dir.exists():
        if not cache_dir.is_dir():
            raise ExtractionError(f"cache path is not a directory: {cache_dir}")
        if not metadata_path.exists():
            raise ExtractionError(
                "cache directory exists without metadata; choose a new cache "
                "directory or pass --overwrite intentionally"
            )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        _assert_resume_metadata(metadata, expected)
        if bool(metadata.get("complete", False)):
            return validate_cache(
                cache_metadata=metadata_path,
                clip_manifest=clip_manifest,
                train_manifest=train_manifest,
                pca_path=pca_path,
                provenance=provenance,
                finite_check_rows=finite_check_rows,
            )
        if not target_path.is_file():
            raise ExtractionError("incomplete cache metadata exists without target array")
        if not rgb_path.is_file() or not actions_path.is_file():
            raise ExtractionError(
                "incomplete cache metadata exists without RGB/action arrays"
            )
        target = np.load(target_path, mmap_mode="r+", allow_pickle=False)
        rgb_cache = np.load(rgb_path, mmap_mode="r+", allow_pickle=False)
        actions_cache = np.load(
            actions_path, mmap_mode="r+", allow_pickle=False
        )
        if (
            tuple(target.shape) != tuple(expected["target_shape"])
            or target.dtype != np.float16
        ):
            raise ExtractionError("incomplete target array shape or dtype mismatch")
        if (
            tuple(rgb_cache.shape) != tuple(expected["rgb_shape"])
            or rgb_cache.dtype != np.float16
        ):
            raise ExtractionError("incomplete RGB array shape or dtype mismatch")
        if (
            tuple(actions_cache.shape) != tuple(expected["actions_shape"])
            or actions_cache.dtype != np.float32
        ):
            raise ExtractionError("incomplete action array shape or dtype mismatch")
    else:
        metadata = dict(expected)
        _publish_initialized_cache_directory(cache_dir, metadata)
        target = np.load(target_path, mmap_mode="r+", allow_pickle=False)
        rgb_cache = np.load(rgb_path, mmap_mode="r+", allow_pickle=False)
        actions_cache = np.load(
            actions_path, mmap_mode="r+", allow_pickle=False
        )

    start_row = int(metadata["written_rows"])
    if start_row < len(rows):
        encoder = _load_teacher(provenance=provenance, device=device)
        for offset, batch_rows in _batches(
            rows, start=start_row, batch_size=batch_size
        ):
            cached_video = quantize_cached_rgb(
                torch.stack([load_rgb_clip(row) for row in batch_rows])
            )
            batch_rgb = cached_video.numpy()
            batch_actions = np.stack(
                [load_action_clip(row) for row in batch_rows]
            )
            if not np.isfinite(batch_actions).all():
                raise ExtractionError(
                    f"non-finite cached actions at row {offset}"
                )
            rgb_cache[offset : offset + len(batch_rows)] = batch_rgb
            actions_cache[offset : offset + len(batch_rows)] = batch_actions
            rgb_cache.flush()
            actions_cache.flush()
            video = cached_video.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            auxiliary = extract_vjepa2_1_target(
                video,
                encoder=encoder,
                stats=stats,
                num_views=len(CAMERAS),
                encoder_dtype=encoder_dtype,
            )
            expected_batch_shape = (len(batch_rows), *TARGET_SHAPE)
            if tuple(auxiliary.shape) != expected_batch_shape:
                raise ExtractionError(
                    f"unexpected projected target shape: {auxiliary.shape}"
                )
            batch_array = auxiliary.detach().cpu().numpy()
            if batch_array.dtype != np.float16:
                batch_array = batch_array.astype(np.float16)
            if not np.isfinite(batch_array).all():
                raise ExtractionError(f"non-finite projected target at row {offset}")
            target[offset : offset + len(batch_rows)] = batch_array
            target.flush()
            metadata["written_rows"] = offset + len(batch_rows)
            _atomic_write_json(metadata_path, metadata)
            print(
                f"cache extraction: {metadata['written_rows']}/{len(rows)} rows",
                flush=True,
            )
        del encoder
    target.flush()
    rgb_cache.flush()
    actions_cache.flush()
    del target
    del rgb_cache
    del actions_cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    verified = _validate_target_array(
        target_path,
        shape=expected["target_shape"],
        finite_chunk_rows=finite_check_rows,
    )
    del verified
    verified_rgb = _validate_rgb_array(
        rgb_path,
        shape=expected["rgb_shape"],
        finite_chunk_rows=finite_check_rows,
    )
    del verified_rgb
    verified_actions = _validate_actions_array(
        actions_path,
        shape=expected["actions_shape"],
        finite_chunk_rows=finite_check_rows,
    )
    del verified_actions
    metadata["target_sha256"] = sha256_file(target_path)
    metadata["rgb_sha256"] = sha256_file(rgb_path)
    metadata["actions_sha256"] = sha256_file(actions_path)
    metadata["written_rows"] = len(rows)
    metadata["complete"] = True
    _atomic_write_json(metadata_path, metadata)
    return validate_cache(
        cache_metadata=metadata_path,
        clip_manifest=clip_manifest,
        train_manifest=train_manifest,
        pca_path=pca_path,
        provenance=provenance,
        finite_check_rows=finite_check_rows,
    )


def validate_cache(
    *,
    cache_metadata: Path,
    clip_manifest: Path,
    train_manifest: Path,
    pca_path: Path,
    provenance: Mapping[str, Any] | None,
    finite_check_rows: int,
) -> dict[str, Any]:
    """Read-only validation of every cache dependency and target row."""
    cache_metadata = cache_metadata.expanduser().resolve()
    clip_manifest = clip_manifest.expanduser().resolve()
    train_manifest = train_manifest.expanduser().resolve()
    pca_path = pca_path.expanduser().resolve()
    if not cache_metadata.is_file():
        raise ExtractionError(f"cache metadata is missing: {cache_metadata}")
    with cache_metadata.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if int(metadata.get("format_version", -1)) != CACHE_FORMAT_VERSION:
        raise ExtractionError("unsupported target-cache format")
    if metadata.get("artifact_type") != "vjepa2.1-wan-grid-cache":
        raise ExtractionError("unexpected target-cache artifact type")
    if not bool(metadata.get("complete", False)):
        raise ExtractionError("target cache is not marked complete")
    split = str(metadata.get("split"))
    if split not in {"train", "val", "test"}:
        raise ExtractionError(f"target cache has unknown split {split!r}")
    rows = validate_clip_rows(clip_manifest, expected_split=split)
    if int(metadata.get("clip_count", -1)) != len(rows):
        raise ExtractionError("cache clip_count mismatch")
    if metadata.get("clip_manifest_sha256") != sha256_file(clip_manifest):
        raise ExtractionError("cache/clip-manifest SHA-256 mismatch")
    if metadata.get("train_manifest_sha256") != sha256_file(train_manifest):
        raise ExtractionError("cache/training-manifest SHA-256 mismatch")
    _, _, pca_digest = load_and_validate_pca(
        pca_path=pca_path,
        train_manifest=train_manifest,
        expected_source_commit=(
            provenance["source_commit"] if provenance is not None else None
        ),
        expected_checkpoint_sha256=(
            provenance["checkpoint_sha256"] if provenance is not None else None
        ),
    )
    if metadata.get("pca_sha256") != pca_digest:
        raise ExtractionError("cache/PCA SHA-256 mismatch")
    recorded_provenance = metadata.get("provenance")
    if not isinstance(recorded_provenance, dict):
        raise ExtractionError("cache extraction provenance is missing")
    expected_recorded_provenance = {
        "vjepa_source_commit": metadata.get("source_commit"),
        "vjepa_checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "pca_stats_sha256": metadata.get("pca_sha256"),
        "train_manifest_sha256": metadata.get("train_manifest_sha256"),
        "clip_manifest_sha256": metadata.get("clip_manifest_sha256"),
    }
    if recorded_provenance != expected_recorded_provenance:
        raise ExtractionError("cache extraction provenance mismatch")
    if provenance is not None:
        if metadata.get("source_commit") != provenance["source_commit"]:
            raise ExtractionError("cache/source provenance mismatch")
        if metadata.get("checkpoint_sha256") != provenance["checkpoint_sha256"]:
            raise ExtractionError("cache/checkpoint provenance mismatch")
    if metadata.get("model_name") != VJEPA2_1_MODEL_NAME:
        raise ExtractionError("cache model identity mismatch")
    if metadata.get("source_commit") != VJEPA2_1_RELEASE_COMMIT:
        raise ExtractionError("cache was not built from the pinned source")
    if (
        metadata.get("frame_offsets") != list(FRAME_OFFSETS)
        or metadata.get("camera_order") != list(CAMERAS)
    ):
        raise ExtractionError("cache frame map or camera order mismatch")
    if (
        int(metadata.get("sample_size", -1)) != SAMPLE_SIZE
        or int(metadata.get("chunk_size", -1)) != CHUNK_SIZE
        or int(metadata.get("action_span", -1)) != ACTION_SPAN
    ):
        raise ExtractionError("cache clip schema mismatch")
    if metadata.get("target_dtype") != "float16":
        raise ExtractionError("cache target_dtype metadata mismatch")
    if metadata.get("target_file") != "targets.fp16.npy":
        raise ExtractionError("cache target filename mismatch")
    if (
        metadata.get("rgb_file") != "rgb.fp16.npy"
        or metadata.get("rgb_dtype") != "float16"
    ):
        raise ExtractionError("cache RGB file/dtype metadata mismatch")
    if (
        metadata.get("actions_file") != "actions.float32.npy"
        or metadata.get("actions_dtype") != "float32"
    ):
        raise ExtractionError("cache action file/dtype metadata mismatch")
    _validate_sha256(
        str(metadata.get("checkpoint_sha256", "")),
        label="cache checkpoint SHA-256",
    )
    expected_shape = [len(rows), *TARGET_SHAPE]
    if metadata.get("target_shape") != expected_shape:
        raise ExtractionError("cache metadata target shape mismatch")
    expected_rgb_shape = [len(rows), *RGB_CACHE_SHAPE]
    expected_actions_shape = [len(rows), *ACTION_CACHE_SHAPE]
    if metadata.get("rgb_shape") != expected_rgb_shape:
        raise ExtractionError("cache metadata RGB shape mismatch")
    if metadata.get("actions_shape") != expected_actions_shape:
        raise ExtractionError("cache metadata action shape mismatch")
    if int(metadata.get("written_rows", -1)) != len(rows):
        raise ExtractionError("complete cache written_rows mismatch")
    identity = {
        "schema": "vjepa2.1-wan-grid-cache-v1",
        "split": split,
        "clip_manifest_sha256": metadata["clip_manifest_sha256"],
        "train_manifest_sha256": metadata["train_manifest_sha256"],
        "pca_sha256": metadata["pca_sha256"],
        "source_commit": metadata["source_commit"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "target_shape": expected_shape,
        "rgb_shape": expected_rgb_shape,
        "actions_shape": expected_actions_shape,
    }
    expected_cache_id = hashlib.sha256(
        _canonical_json(identity).encode("utf-8")
    ).hexdigest()
    if metadata.get("cache_id") != expected_cache_id:
        raise ExtractionError("cache identity digest mismatch")
    target_path = Path(metadata["target_file"])
    if not target_path.is_absolute():
        target_path = cache_metadata.parent / target_path
    target_path = target_path.resolve()
    target = _validate_target_array(
        target_path,
        shape=expected_shape,
        finite_chunk_rows=finite_check_rows,
    )
    del target
    actual_target_digest = sha256_file(target_path)
    if metadata.get("target_sha256") != actual_target_digest:
        raise ExtractionError("target-array SHA-256 mismatch")
    rgb_path = Path(metadata["rgb_file"])
    if not rgb_path.is_absolute():
        rgb_path = cache_metadata.parent / rgb_path
    rgb_path = rgb_path.resolve()
    rgb = _validate_rgb_array(
        rgb_path,
        shape=expected_rgb_shape,
        finite_chunk_rows=finite_check_rows,
    )
    del rgb
    actual_rgb_digest = sha256_file(rgb_path)
    if metadata.get("rgb_sha256") != actual_rgb_digest:
        raise ExtractionError("RGB-cache SHA-256 mismatch")
    actions_path = Path(metadata["actions_file"])
    if not actions_path.is_absolute():
        actions_path = cache_metadata.parent / actions_path
    actions_path = actions_path.resolve()
    actions = _validate_actions_array(
        actions_path,
        shape=expected_actions_shape,
        finite_chunk_rows=finite_check_rows,
    )
    del actions
    actual_actions_digest = sha256_file(actions_path)
    if metadata.get("actions_sha256") != actual_actions_digest:
        raise ExtractionError("action-cache SHA-256 mismatch")
    return metadata


def _add_teacher_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="full independently recorded SHA-256 of the official checkpoint",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_inputs = subparsers.add_parser(
        "validate-inputs", help="verify pinned source and checkpoint provenance"
    )
    _add_teacher_arguments(validate_inputs)

    fit = subparsers.add_parser(
        "fit-pca", help="fit frozen PCA64 whitening on training clips only"
    )
    _add_teacher_arguments(fit)
    fit.add_argument("--train-manifest", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--pca-device", default="cuda")
    fit.add_argument(
        "--encoder-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    fit.add_argument("--batch-size", type=int, default=1)
    fit.add_argument("--max-tokens", type=int, default=250_000)
    fit.add_argument("--max-clips", type=int)
    fit.add_argument("--seed", type=int, default=20260729)
    fit.add_argument("--pca-oversample", type=int, default=16)
    fit.add_argument("--pca-iterations", type=int, default=4)
    fit.add_argument("--whitening-eps", type=float, default=1e-6)
    fit.add_argument("--overwrite", action="store_true")

    validate_pca = subparsers.add_parser(
        "validate-pca", help="read-only validation of a PCA artifact"
    )
    validate_pca.add_argument("--pca", type=Path, required=True)
    validate_pca.add_argument("--train-manifest", type=Path, required=True)

    extract = subparsers.add_parser(
        "extract", help="create or resume one split's FP16 target cache"
    )
    _add_teacher_arguments(extract)
    extract.add_argument("--split", choices=("train", "val", "test"), required=True)
    extract.add_argument("--clip-manifest", type=Path, required=True)
    extract.add_argument("--train-manifest", type=Path, required=True)
    extract.add_argument("--pca", type=Path, required=True)
    extract.add_argument("--cache-dir", type=Path, required=True)
    extract.add_argument("--device", default="cuda")
    extract.add_argument(
        "--encoder-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    extract.add_argument("--batch-size", type=int, default=1)
    extract.add_argument("--finite-check-rows", type=int, default=8)
    extract.add_argument("--overwrite", action="store_true")

    validate_cache_parser = subparsers.add_parser(
        "validate-cache", help="read-only validation of a completed target cache"
    )
    validate_cache_parser.add_argument(
        "--cache-metadata", type=Path, required=True
    )
    validate_cache_parser.add_argument("--clip-manifest", type=Path, required=True)
    validate_cache_parser.add_argument("--train-manifest", type=Path, required=True)
    validate_cache_parser.add_argument("--pca", type=Path, required=True)
    validate_cache_parser.add_argument("--finite-check-rows", type=int, default=8)
    _add_teacher_arguments(validate_cache_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-pca":
            stats, payload, digest = load_and_validate_pca(
                pca_path=args.pca,
                train_manifest=args.train_manifest,
            )
            print(
                "PCA validation passed: "
                f"sha256={digest} shape={stats.output_dim}x{stats.input_dim} "
                f"tokens={payload['sampled_token_count']}"
            )
            return 0

        provenance = validate_teacher_inputs(
            source_path=args.source_path,
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
        )
        if args.command == "validate-inputs":
            print(
                "V-JEPA inputs validated: "
                f"commit={provenance['source_commit']} "
                f"checkpoint_sha256={provenance['checkpoint_sha256']}"
            )
        elif args.command == "fit-pca":
            summary = fit_train_pca(
                train_manifest=args.train_manifest,
                output_path=args.output,
                provenance=provenance,
                device=_resolve_device(args.device),
                pca_device=_resolve_device(args.pca_device),
                encoder_dtype=_encoder_dtype(args.encoder_dtype),
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                max_clips=args.max_clips,
                seed=args.seed,
                pca_oversample=args.pca_oversample,
                pca_iterations=args.pca_iterations,
                whitening_eps=args.whitening_eps,
                overwrite=args.overwrite,
            )
            print(f"PCA fit complete: {json.dumps(summary, sort_keys=True)}")
        elif args.command == "extract":
            metadata = extract_cache(
                split=args.split,
                clip_manifest=args.clip_manifest,
                train_manifest=args.train_manifest,
                pca_path=args.pca,
                cache_dir=args.cache_dir,
                provenance=provenance,
                device=_resolve_device(args.device),
                encoder_dtype=_encoder_dtype(args.encoder_dtype),
                batch_size=args.batch_size,
                finite_check_rows=args.finite_check_rows,
                overwrite=args.overwrite,
            )
            print(
                "cache extraction complete: "
                f"id={metadata['cache_id']} "
                f"shape={metadata['target_shape']} "
                f"sha256={metadata['target_sha256']}"
            )
        else:
            metadata = validate_cache(
                cache_metadata=args.cache_metadata,
                clip_manifest=args.clip_manifest,
                train_manifest=args.train_manifest,
                pca_path=args.pca,
                provenance=provenance,
                finite_check_rows=args.finite_check_rows,
            )
            print(
                "cache validation passed: "
                f"id={metadata['cache_id']} sha256={metadata['target_sha256']}"
            )
    except (
        ExtractionError,
        ManifestError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        ImportError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"V-JEPA extraction error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
