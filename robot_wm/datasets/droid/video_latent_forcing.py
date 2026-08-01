"""Immutable one-view DROID clips for the Video Latent Forcing POC.

This module is deliberately independent of the production LACWM transform.  It
never stacks camera views and it preserves the native 7-D DROID action stream.
The temporal contract for one clip is

* history RGB: offsets ``[0, 2, 4, 6, 8]``;
* future RGB: offsets ``[10, 12, ..., 24]``; and
* future controls: native actions at offsets ``[8, 9, ..., 23]``.

An action at native time ``t`` is interpreted as controlling the transition
from frame ``t`` to frame ``t + 1``.  Consequently the 16 actions cover every
native transition from the final history frame through the final future frame.
RGB tensors are returned in ``[-1, 1]``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


SCHEMA = "droid-video-latent-forcing-poc-v1"
SPLIT_RANK_PREFIX = "video-latent-forcing-poc-v1"
CAMERA = "exterior_image_1_left"
MIN_EPISODE_FRAMES = 66
HISTORY_SIZE = 5
FUTURE_SIZE = 8
FRAME_STRIDE = 2
HISTORY_OFFSETS = tuple(range(0, HISTORY_SIZE * FRAME_STRIDE, FRAME_STRIDE))
FUTURE_OFFSETS = tuple(
    range(HISTORY_SIZE * FRAME_STRIDE, (HISTORY_SIZE + FUTURE_SIZE) * FRAME_STRIDE, FRAME_STRIDE)
)
FRAME_OFFSETS = HISTORY_OFFSETS + FUTURE_OFFSETS
ACTION_OFFSETS = tuple(range(HISTORY_OFFSETS[-1], FUTURE_OFFSETS[-1]))
RGB_SIZE = (64, 112)
LOWRES_SIZE = (32, 56)
LOWRES_PATCH_SIZE = (4, 4)
SCRATCHPAD_SHAPE = (48, 8, 8, 14)
ELIGIBLE_INVENTORY_SHA256 = (
    "3bc6f2c06abe74f1a60ddc4f9a44ce734fb8fa85f9ec94ac99e7bcc954993651"
)
SPLIT_EPISODE_IDS_SHA256 = {
    "train": "cea3449f2a7cf3e9251b1fb1859f4fd8c3717a4ce29a344f2f48c65452e3ac12",
    "val": "58f1a863a7be8f273212030c902c568b32ed75df5aa79993a8aa5c1a7a0252e6",
    "test": "aa36a731cde260ddb94875a0678435b339be18ebeaa4584b507cdbaeac71ba11",
}


class DroidVideoLatentForcingError(RuntimeError):
    """A fixed DROID clip or its provenance failed validation."""


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def droid_paths(data_root: str | Path, episode_index: int) -> tuple[Path, Path]:
    """Return the parquet and the *only* camera path allowed by this study."""
    root = Path(data_root).expanduser().resolve()
    chunk = int(episode_index) // 1000
    parquet = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    video = (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / f"observation.images.{CAMERA}"
        / f"episode_{episode_index:06d}.mp4"
    )
    return parquet, video


def clip_indices(start: int, trajectory_length: int) -> dict[str, tuple[int, ...]]:
    """Construct a complete, unpadded clip and check both temporal endpoints."""
    start = int(start)
    trajectory_length = int(trajectory_length)
    if start < 0:
        raise DroidVideoLatentForcingError("clip start must be non-negative")
    if trajectory_length < MIN_EPISODE_FRAMES:
        raise DroidVideoLatentForcingError(
            f"episode has {trajectory_length} frames; at least {MIN_EPISODE_FRAMES} are required"
        )
    history = tuple(start + offset for offset in HISTORY_OFFSETS)
    future = tuple(start + offset for offset in FUTURE_OFFSETS)
    actions = tuple(start + offset for offset in ACTION_OFFSETS)
    if future[-1] >= trajectory_length:
        raise DroidVideoLatentForcingError(
            f"future endpoint {future[-1]} exceeds final frame {trajectory_length - 1}"
        )
    if actions[-1] + 1 > future[-1]:
        raise DroidVideoLatentForcingError("action endpoint does not terminate at the final target frame")
    return {"history": history, "future": future, "actions": actions}


def valid_start_count(trajectory_length: int) -> int:
    """Number of starts whose final target frame remains inside the episode."""
    if trajectory_length < MIN_EPISODE_FRAMES:
        return 0
    return max(0, int(trajectory_length) - FUTURE_OFFSETS[-1])


def deterministic_starts(
    episode_index: int,
    trajectory_length: int,
    *,
    seed: int,
    count: int,
) -> list[int]:
    """Choose deterministic starts without replacement or global RNG state."""
    available = valid_start_count(trajectory_length)
    if available <= 0:
        return []
    if count <= 0:
        raise DroidVideoLatentForcingError("clips per episode must be positive")
    digest = hashlib.sha256(
        f"{SCHEMA}\0clip-start\0{seed}\0{episode_index}".encode("utf-8")
    ).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    selected = generator.choice(available, size=min(count, available), replace=False)
    return [int(value) for value in selected]


def patchify_lowres_rgb(lowres: torch.Tensor) -> torch.Tensor:
    """Pixel-unshuffle low-resolution RGB to ``[...,48,8,8,14]`` exactly.

    Accepted shapes are ``[3,8,32,56]`` and ``[B,3,8,32,56]``.  The temporal
    axis is never transformed or mixed with a spatial/view boundary.
    """
    unbatched = lowres.ndim == 4
    if unbatched:
        lowres = lowres.unsqueeze(0)
    if lowres.ndim != 5 or tuple(lowres.shape[1:]) != (3, 8, 32, 56):
        raise ValueError(f"expected [B,3,8,32,56], got {tuple(lowres.shape)}")
    batch, channels, time, height, width = lowres.shape
    patch_h, patch_w = LOWRES_PATCH_SIZE
    grid_h, grid_w = height // patch_h, width // patch_w
    scratchpad = (
        lowres.reshape(batch, channels, time, grid_h, patch_h, grid_w, patch_w)
        .permute(0, 1, 4, 6, 2, 3, 5)
        .reshape(batch, channels * patch_h * patch_w, time, grid_h, grid_w)
    )
    return scratchpad.squeeze(0) if unbatched else scratchpad


def unpatchify_lowres_rgb(scratchpad: torch.Tensor) -> torch.Tensor:
    """Exact inverse of :func:`patchify_lowres_rgb`."""
    unbatched = scratchpad.ndim == 4
    if unbatched:
        scratchpad = scratchpad.unsqueeze(0)
    if scratchpad.ndim != 5 or tuple(scratchpad.shape[1:]) != SCRATCHPAD_SHAPE:
        raise ValueError(f"expected [B,{','.join(map(str, SCRATCHPAD_SHAPE))}], got {tuple(scratchpad.shape)}")
    batch, _, time, grid_h, grid_w = scratchpad.shape
    patch_h, patch_w = LOWRES_PATCH_SIZE
    lowres = (
        scratchpad.reshape(batch, 3, patch_h, patch_w, time, grid_h, grid_w)
        .permute(0, 1, 4, 5, 2, 6, 3)
        .reshape(batch, 3, time, grid_h * patch_h, grid_w * patch_w)
    )
    return lowres.squeeze(0) if unbatched else lowres


def future_to_lowres_scratchpad(future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize future RGB and return ``(scratchpad, lowres_rgb)``.

    ``future`` is ``[3,8,64,112]`` or batched.  ``lowres_rgb`` is returned so
    tests and evaluations can apply the exact unpatchify identity without
    pretending the lossy resize itself is invertible.
    """
    unbatched = future.ndim == 4
    if unbatched:
        future = future.unsqueeze(0)
    if future.ndim != 5 or tuple(future.shape[1:]) != (3, 8, 64, 112):
        raise ValueError(f"expected [B,3,8,64,112], got {tuple(future.shape)}")
    batch = future.shape[0]
    frames = future.permute(0, 2, 1, 3, 4).reshape(batch * 8, 3, 64, 112)
    frames = F.interpolate(frames, size=LOWRES_SIZE, mode="area")
    lowres = frames.reshape(batch, 8, 3, *LOWRES_SIZE).permute(0, 2, 1, 3, 4)
    scratchpad = patchify_lowres_rgb(lowres)
    if unbatched:
        return scratchpad.squeeze(0), lowres.squeeze(0)
    return scratchpad, lowres


def episode_rank(episode_index: int) -> str:
    """Frozen protocol rank: SHA256('video-latent-forcing-poc-v1:<ID>')."""
    return hashlib.sha256(
        f"{SPLIT_RANK_PREFIX}:{int(episode_index)}".encode("utf-8")
    ).hexdigest()


def assign_episode_splits(
    episode_lengths: Mapping[int, int],
    *,
    counts: Mapping[str, int],
) -> dict[str, list[tuple[int, int]]]:
    """Hash-rank eligible episodes into exact, episode-disjoint splits."""
    expected = ("train", "val", "test")
    if set(counts) != set(expected):
        raise DroidVideoLatentForcingError(f"split counts must have exactly these keys: {expected}")
    for split in expected:
        if int(counts[split]) < 0:
            raise DroidVideoLatentForcingError("split counts must be non-negative")
    eligible = [
        (int(index), int(length))
        for index, length in episode_lengths.items()
        if int(length) >= MIN_EPISODE_FRAMES
    ]
    if len({index for index, _ in eligible}) != len(eligible):
        raise DroidVideoLatentForcingError("eligible episode inventory contains duplicate IDs")
    needed = sum(int(counts[split]) for split in expected)
    if len(eligible) < needed:
        raise DroidVideoLatentForcingError(
            f"need {needed} eligible episodes but found only {len(eligible)}"
        )
    ranked = sorted(eligible, key=lambda row: (episode_rank(row[0]), row[0]))[:needed]
    result: dict[str, list[tuple[int, int]]] = {}
    cursor = 0
    for split in expected:
        count = int(counts[split])
        result[split] = ranked[cursor : cursor + count]
        cursor += count
    return result


def make_clip_row(
    *,
    split: str,
    episode_index: int,
    trajectory_length: int,
    start: int,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise DroidVideoLatentForcingError(f"invalid split: {split}")
    indices = clip_indices(start, trajectory_length)
    identity = {
        "schema": SCHEMA,
        "split": split,
        "episode_index": int(episode_index),
        "start": int(start),
        "camera": CAMERA,
    }
    return {
        **identity,
        "clip_id": hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest(),
        "trajectory_length": int(trajectory_length),
        "history_indices": list(indices["history"]),
        "future_indices": list(indices["future"]),
        "action_indices": list(indices["actions"]),
        "protected": split == "test",
    }


def validate_clip_row(row: Mapping[str, Any], *, expected_split: str | None = None) -> None:
    if row.get("schema") != SCHEMA:
        raise DroidVideoLatentForcingError("clip row schema mismatch")
    split = row.get("split")
    if split not in {"train", "val", "test"} or (expected_split and split != expected_split):
        raise DroidVideoLatentForcingError("clip row split mismatch")
    if row.get("camera") != CAMERA:
        raise DroidVideoLatentForcingError(
            f"clip row camera must be exactly {CAMERA}; cross-view input is forbidden"
        )
    expected = make_clip_row(
        split=str(split),
        episode_index=int(row["episode_index"]),
        trajectory_length=int(row["trajectory_length"]),
        start=int(row["start"]),
    )
    for key in (
        "clip_id",
        "history_indices",
        "future_indices",
        "action_indices",
        "protected",
    ):
        if row.get(key) != expected[key]:
            raise DroidVideoLatentForcingError(f"clip row has invalid {key}")
    if split == "test" and row.get("protected") is not True:
        raise DroidVideoLatentForcingError("test rows must remain protected")
    cache_keys = {"cache_relpath", "cache_sha256"}.intersection(row)
    if cache_keys and cache_keys != {"cache_relpath", "cache_sha256"}:
        raise DroidVideoLatentForcingError("cache path and checksum must appear together")
    if split == "test" and cache_keys:
        raise DroidVideoLatentForcingError("protected test rows may contain identifiers only, not caches")


def read_clip_manifest(path: str | Path, *, expected_split: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                validate_clip_row(row, expected_split=expected_split)
            except Exception as exc:
                raise DroidVideoLatentForcingError(f"invalid manifest row {line_number}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise DroidVideoLatentForcingError(f"clip manifest is empty: {path}")
    if len({row["clip_id"] for row in rows}) != len(rows):
        raise DroidVideoLatentForcingError("clip manifest contains duplicate clip IDs")
    return rows


def _decode_selected_frames(path: Path, indices: Sequence[int]) -> np.ndarray:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - dependency exercised in production
        raise DroidVideoLatentForcingError("PyAV is required to decode DROID clips") from exc
    wanted = {int(index): position for position, index in enumerate(indices)}
    frames: list[np.ndarray | None] = [None] * len(indices)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            position = wanted.get(frame_index)
            if position is not None:
                frames[position] = frame.to_ndarray(format="rgb24")
            if frame_index >= indices[-1] and all(value is not None for value in frames):
                break
    if any(value is None for value in frames):
        missing = [indices[i] for i, value in enumerate(frames) if value is None]
        raise DroidVideoLatentForcingError(f"video {path} is missing selected frames {missing}")
    return np.stack(frames)  # type: ignore[arg-type]


def _resize_rgb_uint8(frames: np.ndarray) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise DroidVideoLatentForcingError(f"expected decoded [T,H,W,3], got {frames.shape}")
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    resized = F.interpolate(
        tensor,
        size=RGB_SIZE,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    # Immutable caches store resized RGB as uint8. Canonicalize live decoding
    # through the identical quantization so cached train/validation clips and
    # identifier-only protected-test clips have exactly the same preprocessing.
    return resized.add(1.0).mul(127.5).round().clamp_(0, 255).div_(127.5).sub_(1.0)


def _read_native_actions(parquet_path: Path, indices: Sequence[int]) -> torch.Tensor:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise DroidVideoLatentForcingError("pandas with parquet support is required") from exc
    frame = pd.read_parquet(parquet_path, columns=["action"])
    values = np.stack(frame["action"].to_numpy()).astype(np.float32, copy=False)
    if values.ndim != 2 or values.shape[1] != 7:
        raise DroidVideoLatentForcingError(
            f"DROID action must have explicit width 7, got {values.shape} in {parquet_path}"
        )
    if indices[-1] >= len(values):
        raise DroidVideoLatentForcingError("action endpoint exceeds the parquet trajectory")
    return torch.from_numpy(values[np.asarray(indices)])


def _masks() -> dict[str, torch.Tensor]:
    return {
        "history_mask": torch.ones(HISTORY_SIZE, dtype=torch.bool),
        "future_mask": torch.ones(FUTURE_SIZE, dtype=torch.bool),
        "action_mask": torch.ones(len(ACTION_OFFSETS), dtype=torch.bool),
    }


def _validate_sample_shapes(sample: Mapping[str, torch.Tensor]) -> None:
    expected = {
        "history": (3, 5, 64, 112),
        "future": (3, 8, 64, 112),
        "actions": (16, 7),
        "lowres_scratchpad": SCRATCHPAD_SHAPE,
        "lowres_rgb": (3, 8, 32, 56),
        "history_mask": (5,),
        "future_mask": (8,),
        "action_mask": (16,),
    }
    for key, shape in expected.items():
        if tuple(sample[key].shape) != shape:
            raise DroidVideoLatentForcingError(
                f"sample {key} has shape {tuple(sample[key].shape)}, expected {shape}"
            )
    for key in ("history_mask", "future_mask", "action_mask"):
        if sample[key].dtype != torch.bool or not bool(sample[key].all()):
            raise DroidVideoLatentForcingError(f"fixed clips must have a fully-valid {key}")


class DroidVideoLatentForcingDataset(Dataset):
    """Replay an immutable single-camera clip manifest.

    Protected test rows are fail-closed.  Final evaluation must pass both
    ``allow_protected_test=True`` and a non-empty audit purpose, making an
    accidental validation-time test read difficult and visible in the config.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        *,
        allow_protected_test: bool = False,
        protected_test_purpose: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        # A cache is immutable and content-addressed by its manifest entry.
        # Verify it on first use in each dataloader process, not on every epoch.
        self._verified_cache_paths: set[Path] = set()
        self.rows = read_clip_manifest(self.manifest_path)
        splits = {row["split"] for row in self.rows}
        if len(splits) != 1:
            raise DroidVideoLatentForcingError("one dataset cannot mix manifest splits")
        self.split = splits.pop()
        if self.split == "test" and not (
            allow_protected_test and protected_test_purpose and protected_test_purpose.strip()
        ):
            raise DroidVideoLatentForcingError(
                "protected test access is locked; final evaluation requires an explicit audit purpose"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _source_sample(self, row: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parquet, video = droid_paths(self.data_root, int(row["episode_index"]))
        if not parquet.is_file() or not video.is_file():
            raise DroidVideoLatentForcingError(
                f"missing one-view DROID assets for episode {row['episode_index']}"
            )
        frame_indices = row["history_indices"] + row["future_indices"]
        frames = _resize_rgb_uint8(_decode_selected_frames(video, frame_indices))
        history = frames[:HISTORY_SIZE].permute(1, 0, 2, 3).contiguous()
        future = frames[HISTORY_SIZE:].permute(1, 0, 2, 3).contiguous()
        actions = _read_native_actions(parquet, row["action_indices"])
        return history, future, actions

    def _cached_sample(self, row: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = (self.manifest_path.parent / row["cache_relpath"]).resolve()
        if not path.is_relative_to(self.manifest_path.parent):
            raise DroidVideoLatentForcingError("cache path escapes the manifest artifact root")
        if path not in self._verified_cache_paths:
            if sha256_file(path) != row["cache_sha256"]:
                raise DroidVideoLatentForcingError(f"cache checksum mismatch: {path}")
            self._verified_cache_paths.add(path)
        with np.load(path, allow_pickle=False) as payload:
            history_u8 = payload["history_uint8"]
            future_u8 = payload["future_uint8"]
            actions_np = payload["actions"]
        if history_u8.shape != (3, 5, 64, 112) or future_u8.shape != (3, 8, 64, 112):
            raise DroidVideoLatentForcingError("cached RGB shape mismatch")
        if history_u8.dtype != np.uint8 or future_u8.dtype != np.uint8:
            raise DroidVideoLatentForcingError("cached RGB must remain uint8")
        if actions_np.shape != (16, 7) or actions_np.dtype != np.float32:
            raise DroidVideoLatentForcingError("cached actions must be float32 [16,7]")
        history = torch.from_numpy(history_u8.copy()).float().div_(127.5).sub_(1.0)
        future = torch.from_numpy(future_u8.copy()).float().div_(127.5).sub_(1.0)
        actions = torch.from_numpy(actions_np.copy())
        return history, future, actions

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if "cache_relpath" in row:
            history, future, actions = self._cached_sample(row)
        else:
            history, future, actions = self._source_sample(row)
        scratchpad, lowres = future_to_lowres_scratchpad(future)
        sample: dict[str, Any] = {
            "history": history,
            "future": future,
            "actions": actions,
            "lowres_scratchpad": scratchpad,
            "lowres_rgb": lowres,
            **_masks(),
            "clip_id": row["clip_id"],
            "episode_index": int(row["episode_index"]),
            "camera": CAMERA,
            "split": self.split,
        }
        _validate_sample_shapes(sample)
        return sample


def rows_episode_ids(rows: Iterable[Mapping[str, Any]]) -> set[int]:
    """Small public helper used by leakage audits."""
    return {int(row["episode_index"]) for row in rows}
