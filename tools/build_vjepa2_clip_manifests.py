#!/usr/bin/env python3
"""Build and validate deterministic, episode-disjoint ABC clip manifests.

The V-JEPA controlled study must replay the exact same RGB clips in every arm.
This tool converts an existing ABC preprocessing-success manifest (one absolute
episode directory per line) into immutable JSONL clip manifests.  It never
modifies the source dataset.

Every clip contains the 13 explicit RGB frame indices

    start + [0, 5, 10, ..., 60]

and reserves the full 65-step action span consumed by ``ABCTransform``.  Split
assignment is deterministic and episode-disjoint: episodes are ordered by a
SHA-256 rank derived from the seed, then consumed by train, validation, and
test in that order.  An episode is never reused by a later split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORMAT_VERSION = 1
SCHEMA = "abc-vjepa2.1-fixed-clips-v1"
CAMERAS = ("top", "left_wrist", "right_wrist")
STATE_KEYS = (
    "joint_states",
    "joint_actions",
    "gripper_states",
    "gripper_actions",
)
SAMPLE_SIZE = 13
CHUNK_SIZE = 5
ACTION_SPAN = SAMPLE_SIZE * CHUNK_SIZE
FRAME_OFFSETS = tuple(range(0, ACTION_SPAN, CHUNK_SIZE))
SPLITS = ("train", "val", "test")


class ManifestError(RuntimeError):
    """A deterministic clip manifest failed a structural or asset check."""


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, payload: str) -> None:
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
            handle.write(payload)
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


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_episode_manifest(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ManifestError(f"episode manifest is missing: {path}")
    episodes: list[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            value = raw.strip()
            if not value:
                continue
            if len(value.split()) != 1:
                raise ManifestError(
                    f"{path}:{line_number}: expected one whitespace-free path"
                )
            episode = Path(value)
            if not episode.is_absolute():
                raise ManifestError(
                    f"{path}:{line_number}: episode path is not absolute: {episode}"
                )
            episodes.append(episode.resolve())
    if not episodes:
        raise ManifestError(f"episode manifest is empty: {path}")
    if len(episodes) != len(set(episodes)):
        raise ManifestError(f"episode manifest contains duplicate paths: {path}")
    return episodes


def _video_lengths(episode: Path) -> dict[str, int]:
    try:
        import decord
    except ImportError as exc:
        raise ManifestError(
            "decord is required to validate ABC video lengths"
        ) from exc

    lengths: dict[str, int] = {}
    for camera in CAMERAS:
        video_path = episode / f"{camera}.mp4"
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise ManifestError(f"missing or empty ABC video: {video_path}")
        try:
            lengths[camera] = len(decord.VideoReader(str(video_path)))
        except Exception as exc:
            raise ManifestError(f"cannot open ABC video {video_path}: {exc}") from exc
    return lengths


def inspect_episode(episode: Path) -> int:
    """Return the common usable trajectory length after strict asset checks."""
    if not episode.is_dir():
        raise ManifestError(f"ABC episode directory is missing: {episode}")
    state_path = episode / "states.npz"
    if not state_path.is_file() or state_path.stat().st_size <= 0:
        raise ManifestError(f"missing or empty ABC states: {state_path}")
    try:
        with np.load(state_path, allow_pickle=False) as state:
            missing = [key for key in STATE_KEYS if key not in state]
            if missing:
                raise ManifestError(
                    f"{state_path} lacks required arrays: {', '.join(missing)}"
                )
            state_lengths = {key: int(len(state[key])) for key in STATE_KEYS}
    except ManifestError:
        raise
    except Exception as exc:
        raise ManifestError(f"cannot read ABC states {state_path}: {exc}") from exc

    lengths = {**state_lengths, **_video_lengths(episode)}
    common_length = min(lengths.values())
    if common_length < ACTION_SPAN:
        raise ManifestError(
            f"ABC episode is too short for a {ACTION_SPAN}-step clip: "
            f"{episode} lengths={lengths}"
        )
    return common_length


def _episode_rank(episode: Path, seed: int) -> str:
    return _sha256_text(f"{SCHEMA}\0episode-rank\0{seed}\0{episode}")


def _coprime_step(modulus: int, initial: int) -> int:
    if modulus <= 1:
        return 0
    step = initial % modulus
    if step == 0:
        step = 1
    while math.gcd(step, modulus) != 1:
        step = (step + 1) % modulus
        if step == 0:
            step = 1
    return step


def deterministic_starts(
    episode: Path,
    *,
    trajectory_length: int,
    seed: int,
    limit: int,
) -> list[int]:
    """Choose a deterministic no-replacement subset without global RNG state."""
    if limit <= 0:
        raise ManifestError("clips-per-episode must be positive")
    valid_count = trajectory_length - ACTION_SPAN + 1
    if valid_count <= 0:
        return []
    digest = hashlib.sha256(
        f"{SCHEMA}\0starts\0{seed}\0{episode}".encode("utf-8")
    ).digest()
    offset = int.from_bytes(digest[:8], "big") % valid_count
    step = _coprime_step(valid_count, int.from_bytes(digest[8:16], "big"))
    count = min(limit, valid_count)
    if valid_count == 1:
        return [0]
    return [(offset + index * step) % valid_count for index in range(count)]


def _clip_row(
    *,
    split: str,
    episode: Path,
    start: int,
    auxiliary_index: int,
) -> dict[str, Any]:
    frame_indices = [start + offset for offset in FRAME_OFFSETS]
    identity = {
        "schema": SCHEMA,
        "split": split,
        "episode_dir": str(episode),
        "start": start,
        "frame_indices": frame_indices,
    }
    return {
        "clip_id": _sha256_text(_canonical_json(identity)),
        "split": split,
        "episode_dir": str(episode),
        "start": start,
        "frame_indices": frame_indices,
        "sample_size": SAMPLE_SIZE,
        "chunk_size": CHUNK_SIZE,
        "action_span": ACTION_SPAN,
        "auxiliary_index": auxiliary_index,
    }


def _jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(_canonical_json(row) + "\n" for row in rows)


def build_manifests(
    *,
    episode_manifest: Path,
    output_dir: Path,
    counts: Mapping[str, int],
    seed: int,
    clips_per_episode: int,
    overwrite: bool,
) -> dict[str, Any]:
    for split in SPLITS:
        if int(counts[split]) <= 0:
            raise ManifestError(f"{split} clip count must be positive")
    if clips_per_episode <= 0:
        raise ManifestError("clips-per-episode must be positive")

    episode_manifest = episode_manifest.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    metadata_path = output_dir / "manifest_metadata.json"
    manifest_paths = {split: output_dir / f"{split}.jsonl" for split in SPLITS}
    existing = [path for path in (*manifest_paths.values(), metadata_path) if path.exists()]
    if existing and not overwrite:
        raise ManifestError(
            "output already exists; pass --overwrite only for an intentional "
            f"deterministic rebuild: {existing[0]}"
        )

    episodes = sorted(
        _read_episode_manifest(episode_manifest),
        key=lambda episode: (_episode_rank(episode, seed), str(episode)),
    )
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLITS
    }
    episode_cursor = 0
    for split in SPLITS:
        requested = int(counts[split])
        while len(rows_by_split[split]) < requested:
            if episode_cursor >= len(episodes):
                raise ManifestError(
                    f"not enough episode-disjoint ABC clips for {split}: "
                    f"built {len(rows_by_split[split])}, requested {requested}"
                )
            episode = episodes[episode_cursor]
            episode_cursor += 1
            trajectory_length = inspect_episode(episode)
            starts = deterministic_starts(
                episode,
                trajectory_length=trajectory_length,
                seed=seed,
                limit=clips_per_episode,
            )
            remaining = requested - len(rows_by_split[split])
            for start in starts[:remaining]:
                rows_by_split[split].append(
                    _clip_row(
                        split=split,
                        episode=episode,
                        start=start,
                        auxiliary_index=len(rows_by_split[split]),
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_metadata: dict[str, Any] = {}
    for split, path in manifest_paths.items():
        _atomic_write_text(path, _jsonl_payload(rows_by_split[split]))
        split_metadata[split] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "clip_count": len(rows_by_split[split]),
            "episode_count": len(
                {row["episode_dir"] for row in rows_by_split[split]}
            ),
        }
    metadata = {
        "format_version": FORMAT_VERSION,
        "schema": SCHEMA,
        "seed": int(seed),
        "sample_size": SAMPLE_SIZE,
        "chunk_size": CHUNK_SIZE,
        "action_span": ACTION_SPAN,
        "frame_offsets": list(FRAME_OFFSETS),
        "clips_per_episode": int(clips_per_episode),
        "episode_manifest": str(episode_manifest),
        "episode_manifest_sha256": sha256_file(episode_manifest),
        "episode_count_consumed": episode_cursor,
        "splits": split_metadata,
    }
    _atomic_write_json(metadata_path, metadata)
    validate_manifests(metadata_path, verify_assets=False)
    return metadata


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ManifestError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def validate_clip_rows(
    path: str | Path,
    *,
    expected_split: str | None = None,
) -> list[dict[str, Any]]:
    """Validate one JSONL file and return its ordered descriptors."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ManifestError(f"clip manifest is missing: {path}")
    rows = _read_jsonl(path)
    if not rows:
        raise ManifestError(f"clip manifest is empty: {path}")
    required = {
        "clip_id",
        "split",
        "episode_dir",
        "start",
        "frame_indices",
        "sample_size",
        "chunk_size",
        "action_span",
        "auxiliary_index",
    }
    clip_ids: set[str] = set()
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ManifestError(f"{path}: row {index} lacks {sorted(missing)}")
        split = str(row["split"])
        if expected_split is not None and split != expected_split:
            raise ManifestError(
                f"{path}: row {index} has split={split!r}, expected {expected_split!r}"
            )
        if split not in SPLITS:
            raise ManifestError(f"{path}: row {index} has unknown split {split!r}")
        if int(row["auxiliary_index"]) != index:
            raise ManifestError(f"{path}: auxiliary indexes must be dense and ordered")
        start = int(row["start"])
        if start < 0:
            raise ManifestError(f"{path}: row {index} has a negative start")
        expected_frames = [start + offset for offset in FRAME_OFFSETS]
        if [int(value) for value in row["frame_indices"]] != expected_frames:
            raise ManifestError(
                f"{path}: row {index} does not contain the required 13x5 frame map"
            )
        if (
            int(row["sample_size"]) != SAMPLE_SIZE
            or int(row["chunk_size"]) != CHUNK_SIZE
            or int(row["action_span"]) != ACTION_SPAN
        ):
            raise ManifestError(f"{path}: row {index} has an incompatible clip schema")
        expected_identity = {
            "schema": SCHEMA,
            "split": split,
            "episode_dir": str(Path(row["episode_dir"]).resolve()),
            "start": start,
            "frame_indices": expected_frames,
        }
        expected_id = _sha256_text(_canonical_json(expected_identity))
        if row["clip_id"] != expected_id:
            raise ManifestError(f"{path}: row {index} clip_id is not canonical")
        if expected_id in clip_ids:
            raise ManifestError(f"{path}: duplicate clip_id {expected_id}")
        clip_ids.add(expected_id)
    return rows


def validate_manifests(
    metadata_path: str | Path,
    *,
    verify_assets: bool,
) -> dict[str, Any]:
    """Validate hashes, split isolation, row schema, and optionally raw assets."""
    metadata_path = Path(metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise ManifestError(f"manifest metadata is missing: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if int(metadata.get("format_version", -1)) != FORMAT_VERSION:
        raise ManifestError("unsupported clip-manifest metadata format")
    if metadata.get("schema") != SCHEMA:
        raise ManifestError("unexpected clip-manifest schema")
    if metadata.get("frame_offsets") != list(FRAME_OFFSETS):
        raise ManifestError("metadata does not use the required 13x5 frame map")
    episode_manifest = Path(metadata["episode_manifest"]).expanduser().resolve()
    if sha256_file(episode_manifest) != metadata.get("episode_manifest_sha256"):
        raise ManifestError("source episode-manifest SHA-256 mismatch")

    split_episodes: dict[str, set[str]] = {}
    all_clip_ids: set[str] = set()
    inspected: dict[str, int] = {}
    for split in SPLITS:
        descriptor = metadata.get("splits", {}).get(split)
        if not isinstance(descriptor, dict):
            raise ManifestError(f"metadata lacks split {split!r}")
        path = metadata_path.parent / descriptor["file"]
        if sha256_file(path) != descriptor.get("sha256"):
            raise ManifestError(f"{split} manifest SHA-256 mismatch")
        rows = validate_clip_rows(path, expected_split=split)
        if len(rows) != int(descriptor.get("clip_count", -1)):
            raise ManifestError(f"{split} manifest count mismatch")
        split_episodes[split] = {str(Path(row["episode_dir"]).resolve()) for row in rows}
        if len(split_episodes[split]) != int(descriptor.get("episode_count", -1)):
            raise ManifestError(f"{split} episode count mismatch")
        for row in rows:
            if row["clip_id"] in all_clip_ids:
                raise ManifestError(f"clip reused across splits: {row['clip_id']}")
            all_clip_ids.add(row["clip_id"])
            if verify_assets:
                episode = str(Path(row["episode_dir"]).resolve())
                if episode not in inspected:
                    inspected[episode] = inspect_episode(Path(episode))
                length = inspected[episode]
                if int(row["start"]) + ACTION_SPAN > length:
                    raise ManifestError(
                        f"clip exceeds the current episode assets: {row['clip_id']}"
                    )

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = split_episodes[left] & split_episodes[right]
            if overlap:
                raise ManifestError(
                    f"{left}/{right} are not episode-disjoint: {sorted(overlap)[:3]}"
                )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build immutable split manifests")
    build.add_argument("--episode-manifest", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--train-clips", type=int, required=True)
    build.add_argument("--val-clips", type=int, required=True)
    build.add_argument("--test-clips", type=int, required=True)
    build.add_argument("--seed", type=int, default=20260729)
    build.add_argument("--clips-per-episode", type=int, default=1)
    build.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser(
        "validate", help="validate hashes, isolation, and optional raw assets"
    )
    validate.add_argument("--metadata", type=Path, required=True)
    validate.add_argument("--verify-assets", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            metadata = build_manifests(
                episode_manifest=args.episode_manifest,
                output_dir=args.output_dir,
                counts={
                    "train": args.train_clips,
                    "val": args.val_clips,
                    "test": args.test_clips,
                },
                seed=args.seed,
                clips_per_episode=args.clips_per_episode,
                overwrite=args.overwrite,
            )
            summary = {
                split: metadata["splits"][split]["clip_count"] for split in SPLITS
            }
            print(
                f"wrote deterministic episode-disjoint manifests to "
                f"{args.output_dir}: {summary}"
            )
        else:
            metadata = validate_manifests(
                args.metadata, verify_assets=args.verify_assets
            )
            summary = {
                split: metadata["splits"][split]["clip_count"] for split in SPLITS
            }
            print(f"manifest validation passed: {summary}")
    except (ManifestError, OSError, ValueError, KeyError) as exc:
        print(f"V-JEPA clip manifest error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
