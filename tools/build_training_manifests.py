#!/usr/bin/env python3
"""Build capped training manifests atomically from complete local assets.

This helper performs no downloads and never edits dataset payloads.  It writes a
manifest only after every candidate used by that manifest passes structural
checks and the requested finite count is available.  ABC is deliberately built
only from an explicit preprocessing-success manifest; it never discovers
``episode_*`` directories, because a partially populated directory is not proof
of preprocessing success.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence


ABC_FILES = ("states.npz", "top.mp4", "left_wrist.mp4", "right_wrist.mp4")
AGIBOT_VIDEOS = ("head_color.mp4", "hand_left_color.mp4", "hand_right_color.mp4")
AGIBOT_CAMERA_JSONS = (
    "head_extrinsic_params_aligned.json",
    "hand_left_extrinsic_params_aligned.json",
    "hand_right_extrinsic_params_aligned.json",
)


class ManifestError(RuntimeError):
    pass


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ManifestError(f"missing file: {path}")
    if path.stat().st_size <= 0:
        raise ManifestError(f"empty file: {path}")


def _require_dir(path: Path) -> None:
    if not path.is_dir():
        raise ManifestError(f"missing directory: {path}")


def _require_count(items: Sequence[str], limit: int, label: str) -> list[str]:
    if limit <= 0:
        raise ManifestError("limit must be a positive finite integer")
    if len(items) < limit:
        raise ManifestError(f"{label}: found {len(items):,} complete entries, need {limit:,}")
    return list(items[:limit])


def _atomic_write(path: Path, lines: Iterable[str]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{line.rstrip()}\n" for line in lines)
    if not payload.strip():
        raise ManifestError(f"refusing to write an empty manifest: {path}")
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def build_egodex(
    root: Path,
    output: Path,
    limit: int,
    include_roots: Sequence[Path] | None = None,
) -> int:
    root = root.expanduser().resolve()
    _require_dir(root)
    scan_roots = [path.expanduser().resolve() for path in include_roots] if include_roots else [root]
    for scan_root in scan_roots:
        _require_dir(scan_root)
        if not scan_root.is_relative_to(root):
            raise ManifestError(f"EgoDex include root escapes dataset root: {scan_root}")
    hdf5_paths = sorted(path for scan_root in scan_roots for path in scan_root.rglob("*.hdf5"))
    if not hdf5_paths:
        raise ManifestError(f"EgoDex: found no .hdf5 episodes under {root}")
    complete: list[str] = []
    for hdf5_path in hdf5_paths:
        _require_file(hdf5_path)
        _require_file(hdf5_path.with_suffix(".mp4"))
        complete.append(str(hdf5_path.resolve()))
    selected = _require_count(complete, limit, "EgoDex")
    _atomic_write(output, selected)
    return len(selected)


def _read_absolute_path_manifest(path: Path) -> list[Path]:
    _require_file(path)
    entries: list[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            value = raw.strip()
            if not value:
                continue
            if len(value.split()) != 1:
                raise ManifestError(f"{path}:{line_number}: expected one whitespace-free path")
            entry = Path(value)
            if not entry.is_absolute():
                raise ManifestError(f"{path}:{line_number}: path is not absolute: {entry}")
            entries.append(entry)
    if not entries:
        raise ManifestError(f"manifest has no entries: {path}")
    if len(set(entries)) != len(entries):
        raise ManifestError(f"manifest contains duplicate entries: {path}")
    return entries


def build_abc(success_manifest: Path, output: Path, limit: int) -> int:
    success_manifest = success_manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    if success_manifest == output:
        raise ManifestError(
            "ABC success manifest and runtime manifest must be different files; "
            "refusing to overwrite preprocessing provenance"
        )
    entries = _read_absolute_path_manifest(success_manifest)
    complete: list[str] = []
    for episode_dir in entries:
        _require_dir(episode_dir)
        for filename in ABC_FILES:
            _require_file(episode_dir / filename)
        complete.append(str(episode_dir.resolve()))
    selected = _require_count(complete, limit, "ABC")
    _atomic_write(output, selected)
    return len(selected)


def _agibot_episode_files(root: Path, task: str, episode: str) -> tuple[Path, ...]:
    video_dir = root / "observations" / task / episode / "videos"
    camera_dir = root / "parameters" / task / episode / "parameters" / "camera"
    return (
        root / "proprio_stats" / task / episode / "proprio_stats.h5",
        *(video_dir / filename for filename in AGIBOT_VIDEOS),
        *(camera_dir / filename for filename in AGIBOT_CAMERA_JSONS),
    )


def build_agibot(root: Path, output: Path, limit: int, dataset_id: str = "scr") -> int:
    root = root.expanduser().resolve()
    observations = root / "observations"
    _require_dir(observations)
    episode_dirs = sorted(
        path
        for task_dir in observations.iterdir()
        if task_dir.is_dir()
        for path in task_dir.iterdir()
        if path.is_dir()
    )
    if not episode_dirs:
        raise ManifestError(f"AgiBot: found no extracted episode directories under {observations}")
    rows: list[str] = []
    for episode_dir in episode_dirs:
        task, episode = episode_dir.parent.name, episode_dir.name
        for path in _agibot_episode_files(root, task, episode):
            _require_file(path)
        rows.append(f"{task},{episode},{dataset_id}")
    selected = _require_count(rows, limit, "AgiBot")
    _atomic_write(output, ["task_id,episode_id,dataset", *selected])
    return len(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    egodex = subparsers.add_parser("egodex")
    egodex.add_argument("--root", type=Path, required=True)
    egodex.add_argument("--output", type=Path, required=True)
    egodex.add_argument("--limit", type=int, required=True)
    egodex.add_argument("--include-root", type=Path, action="append")

    abc = subparsers.add_parser("abc")
    abc.add_argument("--success-manifest", type=Path, required=True)
    abc.add_argument("--output", type=Path, required=True)
    abc.add_argument("--limit", type=int, required=True)

    agibot = subparsers.add_parser("agibot")
    agibot.add_argument("--root", type=Path, required=True)
    agibot.add_argument("--output", type=Path, required=True)
    agibot.add_argument("--limit", type=int, required=True)
    agibot.add_argument("--dataset-id", default="scr", choices=("alpha", "beta", "viscam", "scr"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "egodex":
            count = build_egodex(args.root, args.output, args.limit, args.include_root)
        elif args.command == "abc":
            count = build_abc(args.success_manifest, args.output, args.limit)
        else:
            count = build_agibot(args.root, args.output, args.limit, args.dataset_id)
    except (ManifestError, OSError) as exc:
        print(f"manifest build failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {count:,} {args.command} episodes to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
