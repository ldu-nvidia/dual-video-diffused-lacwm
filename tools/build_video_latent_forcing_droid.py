#!/usr/bin/env python3
"""Build immutable one-view DROID manifests and optional RGB/action caches.

The tool reads only parquet metadata while assigning splits.  It never decodes
protected test video.  Optional caches are allowed for train/validation only
and must be written outside the Git repository under approved Lustre or
``/mnt/data*`` storage.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.datasets.droid.video_latent_forcing import (  # noqa: E402
    CAMERA,
    FUTURE_SIZE,
    HISTORY_SIZE,
    MIN_EPISODE_FRAMES,
    SCHEMA,
    DroidVideoLatentForcingError,
    _read_native_actions,
    _resize_rgb_uint8,
    assign_episode_splits,
    canonical_json,
    deterministic_starts,
    droid_paths,
    make_clip_row,
    sha256_file,
)


DEFAULT_COUNTS = {"train": 8000, "val": 890, "test": 890}
DEFAULT_CLIPS_PER_EPISODE = {"train": 8, "val": 1, "test": 1}
CACHE_FORMAT = "numpy-npz-uncompressed-v1"
DEFAULT_CACHE_WORKERS = 32
APPROVED_ARTIFACT_ROOTS = (Path("/lustre"), Path("/mnt/data1"), Path("/mnt/data2"))


def git_source_record() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
        "builder_tool_sha256": sha256_file(Path(__file__).resolve()),
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_artifact_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    repo = REPO_ROOT.resolve()
    containing_git_root = next(
        (parent for parent in (output, *output.parents) if (parent / ".git").exists()),
        None,
    )
    if not any(_is_relative_to(output, root.resolve()) for root in APPROVED_ARTIFACT_ROOTS):
        raise DroidVideoLatentForcingError(
            f"artifact output must be under /lustre, /mnt/data1, or /mnt/data2: {output}"
        )
    if _is_relative_to(output, repo) or containing_git_root is not None:
        raise DroidVideoLatentForcingError(
            f"artifacts may not be written inside a Git repository: {output}"
        )
    return output


def discover_episode_lengths(data_root: str | Path) -> dict[int, int]:
    """Inventory usable one-view episodes without decoding RGB content."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DroidVideoLatentForcingError("pyarrow is required to inspect DROID parquet metadata") from exc
    root = Path(data_root).expanduser().resolve()
    parquets = sorted(glob.glob(str(root / "data" / "chunk-*" / "episode_*.parquet")))
    lengths: dict[int, int] = {}
    for parquet_raw in parquets:
        parquet = Path(parquet_raw)
        try:
            episode_index = int(parquet.stem.split("_")[1])
        except (IndexError, ValueError) as exc:
            raise DroidVideoLatentForcingError(f"invalid DROID parquet name: {parquet}") from exc
        expected_parquet, video = droid_paths(root, episode_index)
        if parquet.resolve() != expected_parquet or not video.is_file() or video.stat().st_size <= 0:
            continue
        length = int(pq.ParquetFile(parquet).metadata.num_rows)
        if length >= MIN_EPISODE_FRAMES:
            lengths[episode_index] = length
    return lengths


def _manifest_payload(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _float_rgb_to_uint8(tensor) -> np.ndarray:
    return (
        tensor.add(1.0)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .to(dtype=torch.uint8)
        .cpu()
        .numpy()
    )


def _configure_cache_runtime() -> None:
    """Bound shared CPU libraries before starting cache worker threads."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    # Cache workers overlap independent video/parquet I/O.  Torch must not
    # create an additional CPU pool per resize operation.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before its first
        # parallel operation.  Dedicated CLI runs take the successful path;
        # embedded callers may already have initialized the pool.
        pass
    try:
        import pyarrow

        pyarrow.set_cpu_count(1)
        pyarrow.set_io_thread_count(1)
    except (ImportError, AttributeError):
        # Discovery already reports a clear dependency error when pyarrow is
        # required.  Keeping this helper import-safe makes mocked tests small.
        pass


def _decode_cache_frames(path: Path, indices: Sequence[int]) -> np.ndarray:
    """Decode selected RGB frames with one FFmpeg codec thread per worker."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover - exercised in production
        raise DroidVideoLatentForcingError("PyAV is required to decode DROID clips") from exc
    wanted = {int(index): position for position, index in enumerate(indices)}
    frames: list[np.ndarray | None] = [None] * len(indices)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for frame_index, frame in enumerate(container.decode(stream)):
            position = wanted.get(frame_index)
            if position is not None:
                frames[position] = frame.to_ndarray(format="rgb24")
            if frame_index >= indices[-1] and all(value is not None for value in frames):
                break
    if any(value is None for value in frames):
        missing = [indices[index] for index, value in enumerate(frames) if value is None]
        raise DroidVideoLatentForcingError(f"video {path} is missing selected frames {missing}")
    return np.stack(frames)  # type: ignore[arg-type]


def _write_cache_file(
    *,
    row: Mapping[str, Any],
    stage_root: Path,
    history: torch.Tensor,
    future: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, Any]:
    """Write one low-overhead, per-clip immutable cache and checksum it."""
    if row["split"] == "test":
        raise DroidVideoLatentForcingError("protected test clips must never be cached")
    relative = Path("cache") / str(row["split"]) / f"{row['clip_id']}.npz"
    target = stage_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        # ZIP_STORED avoids spending cache-build CPU recompressing already
        # compressed source video.  The enclosing artifact is atomically
        # published only after every file has been closed and checksummed.
        np.savez(
            handle,
            history_uint8=_float_rgb_to_uint8(history),
            future_uint8=_float_rgb_to_uint8(future),
            actions=actions.cpu().numpy().astype(np.float32, copy=False),
        )
        handle.flush()
    cached = dict(row)
    cached["cache_relpath"] = str(relative)
    cached["cache_sha256"] = sha256_file(target)
    cached["cache_format"] = CACHE_FORMAT
    return cached


def _cache_episode_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_root: Path,
    stage_root: Path,
) -> list[dict[str, Any]]:
    """Cache all selected clips from one episode with one video/parquet read."""
    if not rows:
        return []
    episode_ids = {int(row["episode_index"]) for row in rows}
    splits = {str(row["split"]) for row in rows}
    if len(episode_ids) != 1 or len(splits) != 1:
        raise DroidVideoLatentForcingError(
            "episode cache batches must contain exactly one episode and one split"
        )
    split = next(iter(splits))
    if split == "test":
        raise DroidVideoLatentForcingError("protected test clips must never be cached")
    episode_index = next(iter(episode_ids))
    parquet, video = droid_paths(source_root, episode_index)
    if not parquet.is_file() or not video.is_file():
        raise DroidVideoLatentForcingError(
            f"missing one-view DROID assets for episode {episode_index}"
        )

    frame_indices = sorted(
        {
            int(index)
            for row in rows
            for index in row["history_indices"] + row["future_indices"]
        }
    )
    action_indices = sorted(
        {int(index) for row in rows for index in row["action_indices"]}
    )
    decoded = _resize_rgb_uint8(_decode_cache_frames(video, frame_indices))
    episode_actions = _read_native_actions(parquet, action_indices)
    frame_position = {frame_index: position for position, frame_index in enumerate(frame_indices)}
    action_position = {
        action_index: position for position, action_index in enumerate(action_indices)
    }

    cached_rows: list[dict[str, Any]] = []
    for row in rows:
        selected_frames = decoded[
            [
                frame_position[int(index)]
                for index in row["history_indices"] + row["future_indices"]
            ]
        ]
        history = selected_frames[:HISTORY_SIZE].permute(1, 0, 2, 3).contiguous()
        future = selected_frames[HISTORY_SIZE:].permute(1, 0, 2, 3).contiguous()
        actions = episode_actions[
            [action_position[int(index)] for index in row["action_indices"]]
        ]
        if history.shape[1] != HISTORY_SIZE or future.shape[1] != FUTURE_SIZE:
            raise DroidVideoLatentForcingError("episode cache selection shape mismatch")
        cached_rows.append(
            _write_cache_file(
                row=row,
                stage_root=stage_root,
                history=history,
                future=future,
                actions=actions,
            )
        )
    return cached_rows


def _cache_rows_by_episode(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_root: Path,
    stage_root: Path,
    cache_workers: int = DEFAULT_CACHE_WORKERS,
) -> list[dict[str, Any]]:
    """Group rows without changing manifest order, then cache per episode."""
    if cache_workers < 1:
        raise DroidVideoLatentForcingError("cache_workers must be at least 1")
    positions_by_episode: dict[int, list[int]] = {}
    for position, row in enumerate(rows):
        positions_by_episode.setdefault(int(row["episode_index"]), []).append(position)
    cached_by_position: list[dict[str, Any] | None] = [None] * len(rows)
    position_groups = list(positions_by_episode.values())

    def cache_group(positions: list[int]) -> list[dict[str, Any]]:
        return _cache_episode_rows(
            rows=[rows[position] for position in positions],
            source_root=source_root,
            stage_root=stage_root,
        )

    effective_workers = min(cache_workers, len(position_groups))
    if effective_workers <= 1:
        cached_groups = map(cache_group, position_groups)
    else:
        executor = ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix="droid-cache",
        )
        # executor.map yields in input order even when episode completion order
        # differs, so manifest row order remains bit-for-bit deterministic.
        cached_groups = executor.map(cache_group, position_groups)
    try:
        for positions, cached_group in zip(position_groups, cached_groups, strict=True):
            for position, cached in zip(positions, cached_group, strict=True):
                cached_by_position[position] = cached
    finally:
        if effective_workers > 1:
            executor.shutdown(wait=True, cancel_futures=True)
    if any(row is None for row in cached_by_position):
        raise DroidVideoLatentForcingError("internal cache grouping left an unwritten row")
    return [row for row in cached_by_position if row is not None]


def build_artifact(
    *,
    data_root: str | Path,
    output_root: str | Path,
    episode_lengths: Mapping[int, int] | None = None,
    counts: Mapping[str, int] = DEFAULT_COUNTS,
    seed: int = 20260801,
    clips_per_episode: Mapping[str, int] = DEFAULT_CLIPS_PER_EPISODE,
    cache_splits: Sequence[str] = (),
    cache_workers: int = DEFAULT_CACHE_WORKERS,
) -> dict[str, Any]:
    """Publish a complete artifact directory atomically and never overwrite it."""
    source = Path(data_root).expanduser().resolve()
    output = validate_artifact_output(output_root)
    if output.exists():
        raise DroidVideoLatentForcingError(f"immutable artifact output already exists: {output}")
    invalid_cache = set(cache_splits) - {"train", "val"}
    if invalid_cache:
        raise DroidVideoLatentForcingError(
            f"only train/val may be cached; forbidden splits: {sorted(invalid_cache)}"
        )
    if cache_workers < 1:
        raise DroidVideoLatentForcingError("cache_workers must be at least 1")
    if episode_lengths is None:
        episode_lengths = discover_episode_lengths(source)
    if set(clips_per_episode) != {"train", "val", "test"}:
        raise DroidVideoLatentForcingError(
            "clips_per_episode must have exactly train, val, and test keys"
        )
    if any(int(value) <= 0 for value in clips_per_episode.values()):
        raise DroidVideoLatentForcingError("all clips-per-episode counts must be positive")
    assignments = assign_episode_splits(episode_lengths, counts=counts)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split, episodes in assignments.items():
        rows: list[dict[str, Any]] = []
        for episode_index, trajectory_length in episodes:
            starts = deterministic_starts(
                episode_index,
                trajectory_length,
                seed=seed,
                count=int(clips_per_episode[split]),
            )
            rows.extend(
                make_clip_row(
                    split=split,
                    episode_index=episode_index,
                    trajectory_length=trajectory_length,
                    start=start,
                )
                for start in starts
            )
        split_rows[split] = rows

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        if cache_splits:
            _configure_cache_runtime()
        for split in cache_splits:
            split_rows[split] = _cache_rows_by_episode(
                rows=split_rows[split],
                source_root=source,
                stage_root=stage,
                cache_workers=cache_workers,
            )

        manifest_records: dict[str, dict[str, Any]] = {}
        for split, rows in split_rows.items():
            filename = "test.identifiers.jsonl" if split == "test" else f"{split}.jsonl"
            path = stage / filename
            _write_text(path, _manifest_payload(rows))
            manifest_records[split] = {
                "path": filename,
                "sha256": sha256_file(path),
                "clip_count": len(rows),
                "episode_count": len(assignments[split]),
                "protected": split == "test",
                "cached": split in cache_splits,
                "cache_format": CACHE_FORMAT if split in cache_splits else None,
                "cache_workers_effective": (
                    min(cache_workers, len(assignments[split]))
                    if split in cache_splits
                    else 0
                ),
            }

        inventory_payload = canonical_json(
            {
                str(index): int(length)
                for index, length in sorted(episode_lengths.items())
                if int(length) >= MIN_EPISODE_FRAMES
            }
        )
        provenance = {
            "schema": SCHEMA,
            "complete": True,
            "builder_source": git_source_record(),
            "source_root": str(source),
            "camera": CAMERA,
            "camera_count": 1,
            "split_rank_expression": "sha256('video-latent-forcing-poc-v1:<episode_id>')",
            "clip_start_seed": int(seed),
            "clips_per_episode": {
                split: int(clips_per_episode[split]) for split in ("train", "val", "test")
            },
            "minimum_episode_frames": MIN_EPISODE_FRAMES,
            "eligible_episode_count": sum(
                int(length) >= MIN_EPISODE_FRAMES for length in episode_lengths.values()
            ),
            "eligible_inventory_sha256": hashlib.sha256(inventory_payload.encode("utf-8")).hexdigest(),
            "split_counts": {key: int(value) for key, value in counts.items()},
            "split_episode_ids_sha256": {
                split: hashlib.sha256(
                    json.dumps(
                        [int(index) for index, _ in assignments[split]],
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                for split in ("train", "val", "test")
            },
            "manifests": manifest_records,
            "protected_test_policy": "identifier-only; no video decode or cache during construction",
            "cache": {
                "format": CACHE_FORMAT,
                "layout": "one checksummed NPZ per clip",
                "compression": "ZIP_STORED",
                "construction_io": "one video decode and one parquet action read per episode",
                "workers_requested": int(cache_workers),
                "worker_runtime": {
                    "torch_intraop_threads": 1,
                    "torch_interop_threads_requested": 1,
                    "ffmpeg_codec_threads": 1,
                    "pyarrow_cpu_threads": 1,
                    "pyarrow_io_threads": 1,
                },
            },
        }
        _write_text(stage / "provenance.json", json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        os.replace(stage, output)
        return provenance
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--train-clips-per-episode", type=int, default=8)
    parser.add_argument("--val-clips-per-episode", type=int, default=1)
    parser.add_argument("--test-clips-per-episode", type=int, default=1)
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=DEFAULT_CACHE_WORKERS,
        help="episode-parallel cache workers; each uses one Torch and FFmpeg CPU thread",
    )
    parser.add_argument("--train-episodes", type=int, default=8000)
    parser.add_argument("--val-episodes", type=int, default=890)
    parser.add_argument("--test-episodes", type=int, default=890)
    parser.add_argument(
        "--cache-split",
        action="append",
        default=[],
        choices=("train", "val"),
        help="optional; repeat to cache train and/or validation (never test)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provenance = build_artifact(
        data_root=args.data_root,
        output_root=args.output_root,
        counts={
            "train": args.train_episodes,
            "val": args.val_episodes,
            "test": args.test_episodes,
        },
        seed=args.seed,
        clips_per_episode={
            "train": args.train_clips_per_episode,
            "val": args.val_clips_per_episode,
            "test": args.test_clips_per_episode,
        },
        cache_splits=args.cache_split,
        cache_workers=args.cache_workers,
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
