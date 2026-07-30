"""Offline V-JEPA 2.1 targets aligned to the LACWM/Wan latent grid.

The frozen V-JEPA teacher is intentionally not part of the trainable world
model.  A separate extraction process uses the helpers in this module and
writes compact, whitened targets.  LACWM training reads those cached targets,
so autonomous sampling never imports or invokes V-JEPA.

Input clips use the repository convention ``[B, 13, 3, H, 3*W_view]``.  Each
camera is processed independently.  The temporal map

``[0, 0, 0, 0, 1, ..., 12]``

turns the 13 RGB frames into 16 teacher frames.  V-JEPA's two-frame tubelets
then produce eight temporal tokens.  Averaging adjacent tubelet tokens aligns
them with Wan's four temporal bins:

``frame 0; frames 1--4; frames 5--8; frames 9--12``.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


VJEPA2_1_RELEASE_COMMIT = "45d025f636dfc58fc2426905fc4a1ab755b1c3e5"
VJEPA2_1_MODEL_NAME = "vjepa2_1_vit_base_384"
VJEPA2_1_CHECKPOINT_KEY = "ema_encoder"
VJEPA2_1_CHECKPOINT_BYTES = 1_664_223_428
VJEPA2_1_SOURCE_DIM = 768
VJEPA2_1_PATCH_SIZE = 16
VJEPA2_1_TUBELET_SIZE = 2
VJEPA2_1_FRAME_MAP = (0, 0, 0, 0, *range(1, 13))
VJEPA2_1_IMAGE_MEAN = (0.485, 0.456, 0.406)
VJEPA2_1_IMAGE_STD = (0.229, 0.224, 0.225)


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a checkpoint at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit(path: str | Path) -> str:
    """Read the pinned source identity and fail if ``path`` is not a Git tree."""
    result = subprocess.run(
        ["git", "-C", str(Path(path).resolve()), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_vjepa_source(
    source_path: str | Path,
    *,
    expected_commit: str = VJEPA2_1_RELEASE_COMMIT,
) -> Path:
    source = Path(source_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"V-JEPA source directory is missing: {source}")
    actual = repository_commit(source)
    if actual != expected_commit:
        raise RuntimeError(
            "V-JEPA source commit mismatch: "
            f"expected {expected_commit}, found {actual}"
        )
    return source


def _clean_encoder_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    cleaned = {}
    for raw_key, value in state.items():
        key = raw_key
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def load_vjepa2_1_vit_base_encoder(
    *,
    source_path: str | Path,
    checkpoint_path: str | Path,
    expected_source_commit: str = VJEPA2_1_RELEASE_COMMIT,
    expected_checkpoint_sha256: str | None = None,
) -> nn.Module:
    """Instantiate only the official encoder and strictly load ``ema_encoder``.

    The upstream hub entry currently creates an unused predictor and relies on
    a development download URL.  Direct construction is both smaller and
    deterministic.  The returned teacher is frozen and in evaluation mode.
    """
    source = validate_vjepa_source(
        source_path, expected_commit=expected_source_commit
    )
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"V-JEPA checkpoint is missing: {checkpoint}")
    size = checkpoint.stat().st_size
    if size != VJEPA2_1_CHECKPOINT_BYTES:
        raise RuntimeError(
            "V-JEPA checkpoint size mismatch: "
            f"expected {VJEPA2_1_CHECKPOINT_BYTES}, found {size}"
        )
    if expected_checkpoint_sha256 is not None:
        actual_sha256 = sha256_file(checkpoint)
        if actual_sha256 != expected_checkpoint_sha256:
            raise RuntimeError(
                "V-JEPA checkpoint SHA-256 mismatch: "
                f"expected {expected_checkpoint_sha256}, found {actual_sha256}"
            )

    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module(
        "app.vjepa_2_1.models.vision_transformer"
    )
    module_path = Path(module.__file__).resolve()
    if source not in module_path.parents:
        raise RuntimeError(
            f"imported V-JEPA module from unexpected source: {module_path}"
        )
    encoder = module.vit_base(
        patch_size=VJEPA2_1_PATCH_SIZE,
        img_size=(384, 384),
        num_frames=16,
        tubelet_size=VJEPA2_1_TUBELET_SIZE,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        modality_embedding=True,
        n_output_distillation=1,
    )
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    if VJEPA2_1_CHECKPOINT_KEY not in payload:
        raise RuntimeError(
            f"checkpoint lacks {VJEPA2_1_CHECKPOINT_KEY!r}"
        )
    state = _clean_encoder_state_dict(payload[VJEPA2_1_CHECKPOINT_KEY])
    incompatible = encoder.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict V-JEPA load failed: {incompatible}")
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def prepare_vjepa2_1_views(
    video: Tensor,
    *,
    num_views: int = 3,
    frame_map: Sequence[int] = VJEPA2_1_FRAME_MAP,
    expected_view_size: tuple[int, int] | None = (180, 320),
    padded_view_size: tuple[int, int] = (192, 320),
    teacher_size: tuple[int, int] = (384, 640),
    pad_value: float = -1.0,
) -> Tensor:
    """Return normalized independent teacher clips ``[B*V,3,16,384,640]``."""
    if video.ndim != 5:
        raise ValueError("video must have shape [B,T,C,H,W_total]")
    if not video.is_floating_point():
        raise TypeError("video must be floating point")
    if not torch.isfinite(video).all():
        raise FloatingPointError("video contains non-finite values")
    if torch.any(video < -1.0) or torch.any(video > 1.0):
        raise ValueError("video must use the LACWM [-1,1] RGB convention")
    batch, frames, channels, height, total_width = video.shape
    if channels != 3:
        raise ValueError(f"V-JEPA expects RGB input, received {channels} channels")
    if num_views < 1 or total_width % num_views:
        raise ValueError("video width must divide exactly into camera views")
    if not frame_map:
        raise ValueError("frame_map cannot be empty")
    if min(frame_map) < 0 or max(frame_map) >= frames:
        raise ValueError("frame_map indexes outside the input clip")
    if len(frame_map) % VJEPA2_1_TUBELET_SIZE:
        raise ValueError("teacher frame count must divide by the tubelet size")

    view_width = total_width // num_views
    if expected_view_size is not None and (height, view_width) != tuple(
        expected_view_size
    ):
        raise ValueError(
            "unexpected source camera size: "
            f"expected {tuple(expected_view_size)}, got {(height, view_width)}"
        )
    pad_height = padded_view_size[0] - height
    pad_width = padded_view_size[1] - view_width
    if pad_height < 0 or pad_width < 0:
        raise ValueError(
            "padded_view_size cannot crop the source camera view"
        )

    views = video.reshape(
        batch, frames, channels, height, num_views, view_width
    ).permute(0, 4, 1, 2, 3, 5)
    indices = torch.as_tensor(frame_map, device=video.device, dtype=torch.long)
    views = views.index_select(2, indices)
    flat = views.reshape(
        batch * num_views * len(frame_map),
        channels,
        height,
        view_width,
    )
    flat = F.pad(
        flat,
        (0, pad_width, 0, pad_height),
        mode="constant",
        value=pad_value,
    )
    flat = F.interpolate(
        flat,
        size=teacher_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    flat = (flat.float() + 1.0).mul_(0.5).clamp_(0.0, 1.0)
    mean = flat.new_tensor(VJEPA2_1_IMAGE_MEAN).reshape(1, 3, 1, 1)
    std = flat.new_tensor(VJEPA2_1_IMAGE_STD).reshape(1, 3, 1, 1)
    flat = (flat - mean) / std
    return flat.reshape(
        batch,
        num_views,
        len(frame_map),
        channels,
        *teacher_size,
    ).permute(0, 1, 3, 2, 4, 5).reshape(
        batch * num_views,
        channels,
        len(frame_map),
        *teacher_size,
    )


def align_vjepa2_1_tokens(
    tokens: Tensor,
    *,
    batch_size: int,
    num_views: int = 3,
    teacher_frames: int = 16,
    teacher_size: tuple[int, int] = (384, 640),
    output_frames: int = 4,
    patch_size: int = VJEPA2_1_PATCH_SIZE,
    tubelet_size: int = VJEPA2_1_TUBELET_SIZE,
) -> Tensor:
    """Reshape and pair-pool dense tokens to ``[B,V,4,24,40,D]``."""
    if tokens.ndim != 3:
        raise ValueError("V-JEPA output must have shape [B*V,N,D]")
    if tokens.shape[0] != batch_size * num_views:
        raise ValueError("V-JEPA batch does not match B*num_views")
    temporal_tokens = teacher_frames // tubelet_size
    height_tokens = teacher_size[0] // patch_size
    width_tokens = teacher_size[1] // patch_size
    expected_tokens = temporal_tokens * height_tokens * width_tokens
    if tokens.shape[1] != expected_tokens:
        raise ValueError(
            f"expected {expected_tokens} V-JEPA tokens, got {tokens.shape[1]}"
        )
    if temporal_tokens % output_frames:
        raise ValueError("temporal teacher tokens cannot pool to output_frames")
    pool = temporal_tokens // output_frames
    grid = tokens.reshape(
        batch_size,
        num_views,
        temporal_tokens,
        height_tokens,
        width_tokens,
        tokens.shape[-1],
    )
    return grid.reshape(
        batch_size,
        num_views,
        output_frames,
        pool,
        height_tokens,
        width_tokens,
        tokens.shape[-1],
    ).mean(dim=3)


@dataclass(frozen=True)
class PCAWhiteningStats:
    """Fixed training-split channel projection for V-JEPA dense tokens."""

    mean: Tensor
    components: Tensor
    eigenvalues: Tensor
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.mean.ndim != 1:
            raise ValueError("PCA mean must have shape [D]")
        if self.components.ndim != 2:
            raise ValueError("PCA components must have shape [C,D]")
        if self.eigenvalues.ndim != 1:
            raise ValueError("PCA eigenvalues must have shape [C]")
        if self.components.shape[1] != self.mean.shape[0]:
            raise ValueError("PCA component source dimension is inconsistent")
        if self.components.shape[0] != self.eigenvalues.shape[0]:
            raise ValueError("PCA component count is inconsistent")
        if self.eps <= 0:
            raise ValueError("PCA epsilon must be positive")
        if not (
            torch.isfinite(self.mean).all()
            and torch.isfinite(self.components).all()
            and torch.isfinite(self.eigenvalues).all()
        ):
            raise ValueError("PCA statistics must be finite")
        if torch.any(self.eigenvalues < 0):
            raise ValueError("PCA eigenvalues cannot be negative")

    @property
    def input_dim(self) -> int:
        return self.mean.numel()

    @property
    def output_dim(self) -> int:
        return self.eigenvalues.numel()

    def project(self, tokens: Tensor) -> Tensor:
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"PCA expected {self.input_dim} channels, got {tokens.shape[-1]}"
            )
        mean = self.mean.to(device=tokens.device, dtype=torch.float32)
        components = self.components.to(device=tokens.device, dtype=torch.float32)
        eigenvalues = self.eigenvalues.to(
            device=tokens.device, dtype=torch.float32
        )
        centered = tokens.float() - mean
        projected = centered @ components.transpose(0, 1)
        return projected / eigenvalues.clamp_min(self.eps).sqrt()

    def to_payload(self) -> dict:
        return {
            "format_version": 1,
            "mean": self.mean.detach().cpu().float(),
            "components": self.components.detach().cpu().float(),
            "eigenvalues": self.eigenvalues.detach().cpu().float(),
            "eps": float(self.eps),
        }

    @classmethod
    def from_payload(cls, payload: Mapping) -> "PCAWhiteningStats":
        if int(payload.get("format_version", -1)) != 1:
            raise ValueError("unsupported PCA statistics format")
        return cls(
            mean=payload["mean"].detach().cpu().float(),
            components=payload["components"].detach().cpu().float(),
            eigenvalues=payload["eigenvalues"].detach().cpu().float(),
            eps=float(payload.get("eps", 1e-6)),
        )


def project_and_stack_views(
    aligned_tokens: Tensor,
    stats: PCAWhiteningStats,
    *,
    output_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Project ``[B,V,F,H,W,D]`` and width-stack to ``[B,C,F,H,V*W]``."""
    if aligned_tokens.ndim != 6:
        raise ValueError("aligned tokens must have shape [B,V,F,H,W,D]")
    projected = stats.project(aligned_tokens)
    if not torch.isfinite(projected).all():
        raise FloatingPointError("projected V-JEPA target is non-finite")
    batch, views, frames, height, width, channels = projected.shape
    return projected.permute(0, 5, 2, 3, 1, 4).reshape(
        batch, channels, frames, height, views * width
    ).to(dtype=output_dtype)


@torch.inference_mode()
def extract_vjepa2_1_target(
    video: Tensor,
    *,
    encoder: nn.Module,
    stats: PCAWhiteningStats,
    num_views: int = 3,
    encoder_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Run the frozen teacher and return a compact Wan-grid target."""
    prepared = prepare_vjepa2_1_views(video, num_views=num_views)
    encoder.eval()
    device_type = prepared.device.type
    autocast_enabled = device_type == "cuda"
    with torch.autocast(
        device_type=device_type,
        dtype=encoder_dtype,
        enabled=autocast_enabled,
    ):
        tokens = encoder(prepared, training=False)
    if not isinstance(tokens, Tensor):
        raise TypeError("V-JEPA encoder must return one dense token tensor")
    aligned = align_vjepa2_1_tokens(
        tokens, batch_size=video.shape[0], num_views=num_views
    )
    return project_and_stack_views(aligned, stats)
