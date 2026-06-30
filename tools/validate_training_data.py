#!/usr/bin/env python3
"""Strict, read-only preflight for the four-dataset Wan-DiT training mix.

The validator mirrors the paths and fields consumed by the active loaders without
instantiating a PyTorch dataset.  It never creates, edits, extracts, or deletes
data.  By default it:

* checks every manifest entry (or every discovered DROID parquet) for complete
  files and views;
* deeply checks every episode that the active 10k-per-dataset loader caps select;
* checks array fields/shapes, minimum trajectory length, and video readability;
* enforces the documented active counts (10k DROID/EgoDex/ABC, 5,671 AgiBot).

Use ``--validate-all`` to deeply inspect entries beyond the active caps, or
``--files-only`` for a quick layout pass.  The latter is intentionally not a
substitute for the default preflight before a training launch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional, Sequence


DATASET_NAMES = ("droid", "egodex", "agibot", "abc")
DEFAULT_CAPS = {name: 10_000 for name in DATASET_NAMES}
DEFAULT_EXPECTED = {"droid": 10_000, "egodex": 10_000, "agibot": 5_671, "abc": 10_000}
MIN_TIMESTEPS = 13 * 5 + 1  # 65 sampled low-level steps + the loader's future margin
MAX_FINDING_EXAMPLES = 50

DROID_CAMERAS = (
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
)
AGIBOT_CAMERAS = ("head_color", "hand_left_color", "hand_right_color")
ABC_CAMERAS = ("top", "left_wrist", "right_wrist")
AGIBOT_CAMERA_JSONS = (
    "head_extrinsic_params_aligned.json",
    "hand_left_extrinsic_params_aligned.json",
    "hand_right_extrinsic_params_aligned.json",
)

# Kept explicit so this utility does not import the training stack merely to
# validate HDF5 keys.  These names mirror robot_wm/datasets/egodex/dataset.py.
LEFT_HAND_JOINTS = (
    "leftThumbKnuckle", "leftThumbIntermediateBase", "leftThumbIntermediateTip", "leftThumbTip",
    "leftIndexFingerKnuckle", "leftIndexFingerIntermediateBase", "leftIndexFingerIntermediateTip", "leftIndexFingerTip",
    "leftMiddleFingerKnuckle", "leftMiddleFingerIntermediateBase", "leftMiddleFingerIntermediateTip", "leftMiddleFingerTip",
    "leftRingFingerKnuckle", "leftRingFingerIntermediateBase", "leftRingFingerIntermediateTip", "leftRingFingerTip",
    "leftLittleFingerKnuckle", "leftLittleFingerIntermediateBase", "leftLittleFingerIntermediateTip", "leftLittleFingerTip",
)
RIGHT_HAND_JOINTS = tuple(name.replace("left", "right", 1) for name in LEFT_HAND_JOINTS)
EGODEX_TRANSFORMS = ("camera", "leftHand", "rightHand") + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS


@dataclass(frozen=True)
class Finding:
    severity: str
    message: str
    path: Optional[str] = None


@dataclass
class DatasetReport:
    name: str
    source: str
    cap: Optional[int]
    expected: int
    discovered: int = 0
    complete: int = 0
    selected: int = 0
    active_complete: int = 0
    checked: int = 0
    error_count: int = 0
    warning_count: int = 0
    findings: list[Finding] = field(default_factory=list)
    fingerprint: dict[str, Any] = field(default_factory=dict)

    def add(self, severity: str, message: str, path: Optional[Path | str] = None) -> None:
        finding = Finding(severity, message, str(path) if path is not None else None)
        if severity == "error":
            self.error_count += 1
        else:
            self.warning_count += 1
        if len(self.findings) < MAX_FINDING_EXAMPLES:
            self.findings.append(finding)

    def error(self, message: str, path: Optional[Path | str] = None) -> None:
        self.add("error", message, path)

    def warning(self, message: str, path: Optional[Path | str] = None) -> None:
        self.add("warning", message, path)

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["passed"] = self.passed
        return output


def parse_cap(value: str) -> Optional[int]:
    if value.lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cap must be a positive integer or 'all'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("cap must be positive")
    return parsed


def _selected(items: Sequence[Any], cap: Optional[int]) -> list[Any]:
    return list(items if cap is None else items[:cap])


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stat_fingerprint(paths: Iterable[Path], manifest: Path | None = None) -> dict[str, Any]:
    """Fingerprint the exact selected file set without hashing multi-TB payloads."""
    digest = hashlib.sha256()
    unique_paths = sorted({path.expanduser().resolve(strict=False) for path in paths}, key=str)
    missing = []
    for path in unique_paths:
        try:
            stat = path.stat()
            record = f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        except OSError as exc:
            missing.append(str(path))
            record = f"{path}\0MISSING\0{type(exc).__name__}\n"
        digest.update(record.encode("utf-8"))
    result: dict[str, Any] = {
        "algorithm": "sha256(canonical_path,NUL,size,NUL,mtime_ns)",
        "file_count": len(unique_paths),
        "digest": digest.hexdigest(),
        "missing": missing,
    }
    if manifest is not None:
        result["manifest"] = str(manifest.resolve(strict=False))
        result["manifest_sha256"] = _sha256_file(manifest) if manifest.is_file() else None
    return result


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _check_cap(
    report: DatasetReport,
    available: int,
    active_complete: Optional[int] = None,
) -> None:
    effective = available if report.cap is None else min(available, report.cap)
    report.selected = effective
    report.active_complete = effective if active_complete is None else active_complete
    if report.cap is not None and report.expected > report.cap:
        report.error(
            f"expected count {report.expected:,} exceeds loader cap {report.cap:,}; "
            "the configured run cannot reach its target count"
        )
    if effective < report.expected:
        report.error(
            f"only {effective:,} episodes are selectable, below expected {report.expected:,}"
        )
    if report.active_complete < report.expected and report.active_complete != effective:
        report.error(
            f"only {report.active_complete:,} structurally complete episodes occur in the "
            f"active prefix, below expected {report.expected:,}"
        )
    if report.cap is not None and report.cap > available:
        report.warning(
            f"loader cap {report.cap:,} exceeds available count {available:,}; "
            f"effective count is {available:,}"
        )
    if report.cap is not None and available > report.cap:
        report.warning(
            f"{available - report.cap:,} entries lie beyond the loader cap and will not train"
        )


def _file_error(path: Path) -> Optional[str]:
    try:
        st = path.stat()
    except FileNotFoundError:
        return "missing file"
    except OSError as exc:
        return f"cannot stat file: {exc}"
    if not path.is_file():
        return "not a regular file"
    if st.st_size <= 0:
        return "empty file"
    return None


def _directory_error(path: Path) -> Optional[str]:
    try:
        return None if path.is_dir() else "missing directory"
    except OSError as exc:
        return f"cannot inspect directory: {exc}"


def _required_files(paths: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        message = _file_error(path)
        if message:
            findings.append(Finding("error", message, str(path)))
    return findings


def _load_dependency(module_name: str, report: DatasetReport) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # import failures can include shared-library errors
        report.error(
            f"required content-check dependency '{module_name}' is unavailable: {exc}"
        )
        return None


def _merge_findings(report: DatasetReport, findings: Iterable[Finding]) -> None:
    for finding in findings:
        report.add(finding.severity, finding.message, finding.path)


def _parallel_deep_check(
    items: Sequence[Any],
    check: Callable[[Any], list[Finding]],
    report: DatasetReport,
    workers: int,
) -> None:
    if not items:
        return

    def safe_check(item: Any) -> list[Finding]:
        try:
            return check(item)
        except Exception as exc:
            return [Finding("error", f"unexpected validator exception: {exc}", str(item))]

    if workers == 1:
        results = map(safe_check, items)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(safe_check, items)
    try:
        for findings in results:
            report.checked += 1
            _merge_findings(report, findings)
    finally:
        if workers != 1:
            executor.shutdown(wait=True)


def _probe_video(path: Path, av: Any, required_frames: Optional[int] = None) -> list[Finding]:
    findings: list[Finding] = []
    try:
        container = av.open(str(path), mode="r")
        try:
            streams = list(container.streams.video)
            if not streams:
                return [Finding("error", "contains no video stream", str(path))]
            stream = streams[0]
            width = int(getattr(stream, "width", 0) or 0)
            height = int(getattr(stream, "height", 0) or 0)
            if width <= 0 or height <= 0:
                findings.append(Finding("error", "video has invalid dimensions", str(path)))
            # Decode one frame: an MP4 header can be valid while its media payload is not.
            try:
                first = next(container.decode(stream))
                if first.width <= 0 or first.height <= 0:
                    findings.append(Finding("error", "decoded frame has invalid dimensions", str(path)))
            except StopIteration:
                findings.append(Finding("error", "video contains no decodable frames", str(path)))
            frame_count = int(getattr(stream, "frames", 0) or 0)
            if required_frames is not None:
                if frame_count > 0 and frame_count != required_frames:
                    findings.append(
                        Finding(
                            "error",
                            f"video reports {frame_count} frames; trajectory has {required_frames}",
                            str(path),
                        )
                    )
                elif frame_count <= 0:
                    findings.append(
                        Finding(
                            "error",
                            "container does not report frame count; cannot prove trajectory alignment",
                            str(path),
                        )
                    )
                if frame_count > 0:
                    duration = int(getattr(stream, "duration", 0) or 0)
                    if duration <= 0:
                        findings.append(
                            Finding("error", "video stream has no seekable duration", str(path))
                        )
                    else:
                        try:
                            container.seek(max(0, duration - 1), stream=stream, backward=True)
                            tail = None
                            for tail in container.decode(stream):
                                pass
                            if tail is None or tail.width <= 0 or tail.height <= 0:
                                findings.append(
                                    Finding("error", "last video segment is not decodable", str(path))
                                )
                        except Exception as exc:
                            findings.append(
                                Finding("error", f"cannot seek/decode video tail: {exc}", str(path))
                            )
        finally:
            container.close()
    except Exception as exc:
        findings.append(Finding("error", f"cannot open/decode video: {exc}", str(path)))
    return findings


def _check_shape(
    path: Path,
    key: str,
    shape: Sequence[int],
    tail: Sequence[int],
    expected_t: Optional[int] = None,
) -> list[Finding]:
    findings: list[Finding] = []
    shape_tuple = tuple(int(x) for x in shape)
    if len(shape_tuple) != len(tail) + 1 or shape_tuple[1:] != tuple(tail):
        findings.append(
            Finding("error", f"field {key!r} has shape {shape_tuple}, expected [T,{','.join(map(str, tail))}]", str(path))
        )
    elif expected_t is not None and shape_tuple[0] != expected_t:
        findings.append(
            Finding("error", f"field {key!r} length {shape_tuple[0]} != trajectory length {expected_t}", str(path))
        )
    return findings


def _read_path_manifest(path: Path, report: DatasetReport) -> list[Path]:
    message = _file_error(path)
    if message:
        report.error(message, path)
        return []
    entries: list[Path] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                line = raw.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) != 1:
                    report.error(
                        f"line {line_number} must contain exactly one path without whitespace",
                        path,
                    )
                    continue
                entry = Path(fields[0])
                if not entry.is_absolute():
                    report.error(f"line {line_number} is not an absolute path: {entry}", path)
                entries.append(entry)
    except OSError as exc:
        report.error(f"cannot read manifest: {exc}", path)
        return []
    if not entries:
        report.error("manifest contains no episode entries", path)
    seen: set[Path] = set()
    for entry in entries:
        if entry in seen:
            report.error(f"duplicate manifest entry: {entry}", path)
        seen.add(entry)
    return entries


@dataclass(frozen=True)
class AgibotEntry:
    task_id: str
    episode_id: str
    dataset_id: str


def _read_agibot_manifest(path: Path, report: DatasetReport) -> list[AgibotEntry]:
    message = _file_error(path)
    if message:
        report.error(message, path)
        return []
    entries: list[AgibotEntry] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw_lines = [line.strip() for line in handle if line.strip()]
    except OSError as exc:
        report.error(f"cannot read manifest: {exc}", path)
        return []
    if not raw_lines:
        report.error("manifest is empty", path)
        return []
    if raw_lines[0].replace(" ", "") != "task_id,episode_id,dataset":
        report.error("first line must be task_id,episode_id,dataset", path)
    for line_number, line in enumerate(raw_lines[1:], 2):
        fields = line.split(",")
        if len(fields) != 3 or any(not value.strip() for value in fields):
            report.error(f"line {line_number} must contain exactly three comma-separated fields", path)
            continue
        task, episode, dataset_id = (value.strip() for value in fields)
        for label, value in (("task_id", task), ("episode_id", episode)):
            if value in {".", ".."} or "/" in value or "\\" in value:
                report.error(
                    f"line {line_number} has unsafe {label} {value!r}", path
                )
        if dataset_id not in {"alpha", "beta", "viscam", "scr"}:
            report.error(f"line {line_number} has unsupported dataset id {dataset_id!r}", path)
        entries.append(AgibotEntry(task, episode, dataset_id))
    if not entries:
        report.error("manifest contains no episode entries", path)
    seen: set[AgibotEntry] = set()
    for entry in entries:
        if entry in seen:
            report.error(f"duplicate manifest entry: {entry}", path)
        seen.add(entry)
    return entries


@dataclass(frozen=True)
class DroidEntry:
    episode: int
    parquet: Path
    videos: tuple[Path, ...]


def _droid_entry(root: Path, parquet: Path) -> tuple[Optional[DroidEntry], list[Finding]]:
    findings: list[Finding] = []
    match = re.fullmatch(r"episode_(\d{6})\.parquet", parquet.name)
    if not match:
        return None, [Finding("error", "unexpected parquet filename", str(parquet))]
    episode = int(match.group(1))
    chunk = episode // 1_000
    expected_parent = root / "data" / f"chunk-{chunk:03d}"
    if parquet.parent != expected_parent:
        findings.append(
            Finding("error", f"episode id maps to {expected_parent}, but parquet is elsewhere", str(parquet))
        )
    videos = tuple(
        root / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{camera}" / f"episode_{episode:06d}.mp4"
        for camera in DROID_CAMERAS
    )
    findings.extend(_required_files((parquet, *videos)))
    return DroidEntry(episode, parquet, videos), findings


def _deep_droid(entry: DroidEntry, pq: Any, av: Any, min_timesteps: int) -> list[Finding]:
    findings: list[Finding] = []
    trajectory_length: Optional[int] = None
    try:
        parquet_file = pq.ParquetFile(str(entry.parquet))
        names = set(parquet_file.schema_arrow.names)
        required = {"observation.state", "action"}
        missing = sorted(required - names)
        if missing:
            findings.append(Finding("error", f"missing parquet columns: {missing}", str(entry.parquet)))
        trajectory_length = int(parquet_file.metadata.num_rows)
        if trajectory_length < min_timesteps:
            findings.append(
                Finding("error", f"trajectory has {trajectory_length} rows; need at least {min_timesteps}", str(entry.parquet))
            )
        if not missing and trajectory_length:
            table = parquet_file.read_row_group(0, columns=["observation.state", "action"]).slice(0, 1)
            expected_dims = {"observation.state": 8, "action": 7}
            for key, dim in expected_dims.items():
                value = table[key][0].as_py()
                if value is None or len(value) != dim:
                    findings.append(
                        Finding("error", f"column {key!r} first value must have length {dim}", str(entry.parquet))
                    )
            for batch in parquet_file.iter_batches(
                batch_size=2048, columns=["observation.state", "action"]
            ):
                for key, dim in expected_dims.items():
                    column = batch.column(batch.schema.get_field_index(key))
                    for value in column.to_pylist():
                        if value is None or len(value) != dim:
                            findings.append(
                                Finding("error", f"column {key!r} contains a malformed row", str(entry.parquet))
                            )
                            break
                        if not all(math.isfinite(float(item)) for item in value):
                            findings.append(
                                Finding("error", f"column {key!r} contains NaN/Inf", str(entry.parquet))
                            )
                            break
    except Exception as exc:
        findings.append(Finding("error", f"cannot inspect parquet: {exc}", str(entry.parquet)))
    for video in entry.videos:
        findings.extend(_probe_video(video, av, trajectory_length))
    return findings


def validate_droid(args: argparse.Namespace) -> DatasetReport:
    root = args.droid_root
    report = DatasetReport("droid", str(root), args.droid_cap, args.droid_expected)
    data_dir = root / "data"
    message = _directory_error(data_dir)
    if message:
        report.error(message, data_dir)
        return report
    parquets = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    report.discovered = len(parquets)
    if not parquets:
        report.error("no data/chunk-*/episode_*.parquet files found", data_dir)
        return report
    entries: list[DroidEntry] = []
    seen_episodes: set[int] = set()
    for parquet in parquets:
        entry, findings = _droid_entry(root, parquet)
        _merge_findings(report, findings)
        if entry is None or any(item.severity == "error" for item in findings):
            continue
        if entry.episode in seen_episodes:
            report.error(f"duplicate episode id {entry.episode}", parquet)
            continue
        seen_episodes.add(entry.episode)
        entries.append(entry)
    report.complete = len(entries)
    _check_cap(report, len(entries))
    selected = _selected(entries, report.cap)
    report.fingerprint = stat_fingerprint(
        path for entry in selected for path in (entry.parquet, *entry.videos)
    )
    if args.files_only:
        report.warning("content checks skipped by --files-only")
        return report
    pq = _load_dependency("pyarrow.parquet", report)
    av = _load_dependency("av", report)
    if pq is None or av is None:
        return report
    deep_entries = entries if args.validate_all else selected
    _parallel_deep_check(
        deep_entries,
        lambda item: _deep_droid(item, pq, av, args.min_timesteps),
        report,
        args.workers,
    )
    return report


def _deep_egodex(path: Path, h5py: Any, av: Any, min_timesteps: int) -> list[Finding]:
    findings: list[Finding] = []
    trajectory_length: Optional[int] = None
    import numpy as np
    try:
        with h5py.File(path, "r") as handle:
            if "transforms" not in handle:
                findings.append(Finding("error", "missing HDF5 group 'transforms'", str(path)))
            else:
                group = handle["transforms"]
                missing = [key for key in EGODEX_TRANSFORMS if key not in group]
                if missing:
                    findings.append(Finding("error", f"missing transform datasets: {missing}", str(path)))
                lengths: dict[str, int] = {}
                for key in EGODEX_TRANSFORMS:
                    if key not in group:
                        continue
                    shape = tuple(group[key].shape)
                    findings.extend(_check_shape(path, f"transforms/{key}", shape, (4, 4)))
                    if len(shape) == 3:
                        lengths[key] = int(shape[0])
                        if not np.isfinite(group[key][...]).all():
                            findings.append(
                                Finding("error", f"transforms/{key} contains NaN/Inf", str(path))
                            )
                if lengths:
                    trajectory_length = lengths.get("camera", next(iter(lengths.values())))
                    for key, length in lengths.items():
                        if length != trajectory_length:
                            findings.append(
                                Finding("error", f"transforms/{key} length {length} != {trajectory_length}", str(path))
                            )
                    if trajectory_length < min_timesteps:
                        findings.append(
                            Finding("error", f"trajectory has {trajectory_length} poses; need at least {min_timesteps}", str(path))
                        )
    except Exception as exc:
        findings.append(Finding("error", f"cannot inspect HDF5: {exc}", str(path)))
    findings.extend(_probe_video(path.with_suffix(".mp4"), av, trajectory_length))
    return findings


def validate_egodex(args: argparse.Namespace) -> DatasetReport:
    manifest = args.egodex_manifest
    report = DatasetReport("egodex", str(manifest), args.egodex_cap, args.egodex_expected)
    entries = _read_path_manifest(manifest, report)
    allowed_root = manifest.parent.resolve()
    for path in entries:
        if not path.resolve(strict=False).is_relative_to(allowed_root):
            report.error(f"manifest entry escapes EgoDex root {allowed_root}", path)
    report.discovered = len(entries)
    complete: list[Path] = []
    for path in entries:
        findings = _required_files((path, path.with_suffix(".mp4")))
        _merge_findings(report, findings)
        if not findings:
            complete.append(path)
    report.complete = len(complete)
    selected = _selected(entries, report.cap)
    report.fingerprint = stat_fingerprint(
        (item for path in selected for item in (path, path.with_suffix(".mp4"))),
        manifest,
    )
    complete_set = set(complete)
    _check_cap(report, len(entries), sum(path in complete_set for path in selected))
    if args.files_only:
        report.warning("content checks skipped by --files-only")
        return report
    h5py = _load_dependency("h5py", report)
    av = _load_dependency("av", report)
    if h5py is None or av is None:
        return report
    deep_entries = complete if args.validate_all else [path for path in selected if path in complete_set]
    _parallel_deep_check(
        deep_entries,
        lambda item: _deep_egodex(item, h5py, av, args.min_timesteps),
        report,
        args.workers,
    )
    return report


AGIBOT_H5_FIELDS: dict[str, tuple[int, ...]] = {
    "timestamp": (),
    "state/effector/position": (2,),
    "state/end/orientation": (2, 4),
    "state/end/position": (2, 3),
    "state/head/position": (2,),
    "state/joint/position": (14,),
    "state/robot/orientation": (4,),
    "state/robot/position": (3,),
    "state/waist/position": (2,),
    "action/effector/position": (2,),
    "action/end/orientation": (2, 4),
    "action/end/position": (2, 3),
    "action/head/position": (2,),
    "action/joint/position": (14,),
    "action/waist/position": (2,),
}


@dataclass(frozen=True)
class AgibotPaths:
    entry: AgibotEntry
    root: Path
    proprio: Path
    videos: tuple[Path, ...]
    camera_jsons: tuple[Path, ...]


def _agibot_paths(entry: AgibotEntry, roots: dict[str, Path]) -> AgibotPaths:
    root = roots[entry.dataset_id]
    task, episode = entry.task_id, entry.episode_id
    proprio = root / "proprio_stats" / task / episode / "proprio_stats.h5"
    videos = tuple(root / "observations" / task / episode / "videos" / f"{cam}.mp4" for cam in AGIBOT_CAMERAS)
    camera_dir = root / "parameters" / task / episode / "parameters" / "camera"
    camera_jsons = tuple(camera_dir / name for name in AGIBOT_CAMERA_JSONS)
    return AgibotPaths(entry, root, proprio, videos, camera_jsons)


def _deep_agibot(paths: AgibotPaths, h5py: Any, av: Any, min_timesteps: int) -> list[Finding]:
    findings: list[Finding] = []
    trajectory_length: Optional[int] = None
    import numpy as np
    try:
        with h5py.File(paths.proprio, "r") as handle:
            missing = [key for key in AGIBOT_H5_FIELDS if key not in handle]
            if missing:
                findings.append(Finding("error", f"missing HDF5 fields: {missing}", str(paths.proprio)))
            if "timestamp" in handle:
                timestamp_shape = tuple(handle["timestamp"].shape)
                if len(timestamp_shape) != 1:
                    findings.append(Finding("error", f"timestamp has shape {timestamp_shape}, expected [T]", str(paths.proprio)))
                else:
                    trajectory_length = int(timestamp_shape[0])
                    if trajectory_length < min_timesteps:
                        findings.append(
                            Finding("error", f"trajectory has {trajectory_length} timestamps; need at least {min_timesteps}", str(paths.proprio))
                        )
            for key, tail in AGIBOT_H5_FIELDS.items():
                if key == "timestamp" or key not in handle:
                    continue
                findings.extend(_check_shape(paths.proprio, key, handle[key].shape, tail, trajectory_length))
                if not np.isfinite(handle[key][...]).all():
                    findings.append(Finding("error", f"field {key!r} contains NaN/Inf", str(paths.proprio)))
    except Exception as exc:
        findings.append(Finding("error", f"cannot inspect HDF5: {exc}", str(paths.proprio)))

    for json_path in paths.camera_jsons:
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
            if not isinstance(records, list) or not records:
                findings.append(Finding("error", "camera JSON must be a nonempty list", str(json_path)))
                continue
            if trajectory_length is not None and len(records) != trajectory_length:
                findings.append(
                    Finding("error", f"camera JSON length {len(records)} != trajectory length {trajectory_length}", str(json_path))
                )
            for index in range(len(records)):
                record = records[index]
                extrinsic = record.get("extrinsic") if isinstance(record, dict) else None
                rotation = extrinsic.get("rotation_matrix") if isinstance(extrinsic, dict) else None
                translation = extrinsic.get("translation_vector") if isinstance(extrinsic, dict) else None
                if (
                    not isinstance(rotation, list)
                    or len(rotation) != 3
                    or any(not isinstance(row, list) or len(row) != 3 for row in rotation)
                    or not isinstance(translation, list)
                    or len(translation) != 3
                ):
                    findings.append(
                        Finding("error", f"record {index} has invalid extrinsic matrix/vector", str(json_path))
                    )
        except Exception as exc:
            findings.append(Finding("error", f"cannot inspect camera JSON: {exc}", str(json_path)))
    for video in paths.videos:
        findings.extend(_probe_video(video, av, trajectory_length))
    return findings


def validate_agibot(args: argparse.Namespace) -> DatasetReport:
    manifest = args.agibot_manifest
    report = DatasetReport("agibot", str(manifest), args.agibot_cap, args.agibot_expected)
    entries = _read_agibot_manifest(manifest, report)
    report.discovered = len(entries)
    roots = {
        "alpha": args.agibot_alpha_root,
        "beta": args.agibot_beta_root,
        "viscam": args.agibot_viscam_root,
        "scr": args.agibot_root,
    }
    all_paths = [_agibot_paths(entry, roots) for entry in entries]
    complete: list[AgibotPaths] = []
    for paths in all_paths:
        findings = _required_files((paths.proprio, *paths.videos, *paths.camera_jsons))
        _merge_findings(report, findings)
        if not findings:
            complete.append(paths)
    report.complete = len(complete)
    selected = _selected(all_paths, report.cap)
    report.fingerprint = stat_fingerprint(
        (
            item
            for paths in selected
            for item in (paths.proprio, *paths.videos, *paths.camera_jsons)
        ),
        manifest,
    )
    complete_set = set(complete)
    _check_cap(report, len(entries), sum(paths in complete_set for paths in selected))
    if args.files_only:
        report.warning("content checks skipped by --files-only")
        return report
    h5py = _load_dependency("h5py", report)
    av = _load_dependency("av", report)
    if h5py is None or av is None:
        return report
    deep_entries = complete if args.validate_all else [paths for paths in selected if paths in complete_set]
    _parallel_deep_check(
        deep_entries,
        lambda item: _deep_agibot(item, h5py, av, args.min_timesteps),
        report,
        args.workers,
    )
    return report


def _deep_abc(path: Path, np: Any, av: Any, min_timesteps: int) -> list[Finding]:
    state_path = path / "states.npz"
    findings: list[Finding] = []
    trajectory_length: Optional[int] = None
    expected_shapes = {
        "joint_states": (12,),
        "joint_actions": (12,),
        "gripper_states": (2,),
        "gripper_actions": (2,),
    }
    try:
        with np.load(state_path, allow_pickle=True) as archive:
            required = set(expected_shapes) | {"frame_ts", "instruction"}
            missing = sorted(required - set(archive.files))
            if missing:
                findings.append(Finding("error", f"missing NPZ arrays: {missing}", str(state_path)))
            if "frame_ts" in archive:
                frame_ts = archive["frame_ts"]
                if frame_ts.ndim != 1:
                    findings.append(Finding("error", f"frame_ts has shape {frame_ts.shape}, expected [T]", str(state_path)))
                else:
                    trajectory_length = int(frame_ts.shape[0])
                    if trajectory_length < min_timesteps:
                        findings.append(
                            Finding("error", f"trajectory has {trajectory_length} timestamps; need at least {min_timesteps}", str(state_path))
                        )
            for key, tail in expected_shapes.items():
                if key not in archive:
                    continue
                array = archive[key]
                findings.extend(_check_shape(state_path, key, array.shape, tail, trajectory_length))
                if not np.issubdtype(array.dtype, np.number):
                    findings.append(Finding("error", f"array {key!r} is not numeric", str(state_path)))
                elif not np.isfinite(array).all():
                    findings.append(Finding("error", f"array {key!r} contains NaN/Inf", str(state_path)))
    except Exception as exc:
        findings.append(Finding("error", f"cannot inspect states.npz: {exc}", str(state_path)))
    for camera in ABC_CAMERAS:
        findings.extend(_probe_video(path / f"{camera}.mp4", av, trajectory_length))
    return findings


def validate_abc(args: argparse.Namespace) -> DatasetReport:
    manifest = args.abc_manifest
    report = DatasetReport("abc", str(manifest), args.abc_cap, args.abc_expected)
    entries = _read_path_manifest(manifest, report)
    allowed_root = (args.data_root / "abc_pp").resolve()
    for path in entries:
        if not path.resolve(strict=False).is_relative_to(allowed_root):
            report.error(f"manifest entry escapes ABC root {allowed_root}", path)
    report.discovered = len(entries)
    complete: list[Path] = []
    for path in entries:
        directory_message = _directory_error(path)
        findings = [Finding("error", directory_message, str(path))] if directory_message else []
        findings.extend(_required_files((path / "states.npz", *(path / f"{cam}.mp4" for cam in ABC_CAMERAS))))
        _merge_findings(report, findings)
        if not findings:
            complete.append(path)
    report.complete = len(complete)
    selected = _selected(entries, report.cap)
    report.fingerprint = stat_fingerprint(
        (
            item
            for path in selected
            for item in (path / "states.npz", *(path / f"{cam}.mp4" for cam in ABC_CAMERAS))
        ),
        manifest,
    )
    complete_set = set(complete)
    _check_cap(report, len(entries), sum(path in complete_set for path in selected))
    if args.files_only:
        report.warning("content checks skipped by --files-only")
        return report
    np = _load_dependency("numpy", report)
    av = _load_dependency("av", report)
    if np is None or av is None:
        return report
    deep_entries = complete if args.validate_all else [path for path in selected if path in complete_set]
    _parallel_deep_check(
        deep_entries,
        lambda item: _deep_abc(item, np, av, args.min_timesteps),
        report,
        args.workers,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(os.environ.get("LACWM_DATA", "/scr/ravenh/lacwm_data")).resolve()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=default_root, help="LACWM_DATA root")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
        help="datasets to validate (default: all four)",
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--min-timesteps", type=int, default=MIN_TIMESTEPS)
    parser.add_argument("--validate-all", action="store_true", help="deep-check entries beyond active caps too")
    parser.add_argument("--files-only", action="store_true", help="check manifests/files only; skip field and video decoding")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable JSON report")

    for name in DATASET_NAMES:
        parser.add_argument(f"--{name}-cap", type=parse_cap, default=DEFAULT_CAPS[name], metavar="N|all")
        parser.add_argument(f"--{name}-expected", type=int, default=DEFAULT_EXPECTED[name], metavar="N")

    parser.add_argument("--droid-root", type=Path)
    parser.add_argument("--egodex-manifest", type=Path)
    parser.add_argument("--agibot-manifest", type=Path)
    parser.add_argument("--abc-manifest", type=Path)
    parser.add_argument("--agibot-root", type=Path)
    parser.add_argument("--agibot-alpha-root", type=Path)
    parser.add_argument("--agibot-beta-root", type=Path)
    parser.add_argument("--agibot-viscam-root", type=Path)
    return parser


def _normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    args.data_root = args.data_root.expanduser().resolve()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.min_timesteps <= 0:
        parser.error("--min-timesteps must be positive")
    for name in DATASET_NAMES:
        if getattr(args, f"{name}_expected") < 0:
            parser.error(f"--{name}-expected cannot be negative")

    args.droid_root = (args.droid_root or args.data_root / "droid_lerobot").expanduser().resolve()
    args.egodex_manifest = (args.egodex_manifest or args.data_root / "egodex_cdn" / "manifest.csv").expanduser().resolve()
    args.agibot_manifest = (args.agibot_manifest or args.data_root / "agibot" / "manifest.csv").expanduser().resolve()
    args.abc_manifest = (args.abc_manifest or args.data_root / "abc_pp" / "manifest.txt").expanduser().resolve()
    args.agibot_root = (args.agibot_root or args.data_root / "agibot").expanduser().resolve()
    args.agibot_alpha_root = (
        args.agibot_alpha_root
        or Path(os.environ.get("AGIBOT_ALPHA_ROOT", str(args.agibot_root)))
    ).expanduser().resolve()
    args.agibot_beta_root = (
        args.agibot_beta_root
        or Path(os.environ.get("AGIBOT_BETA_ROOT", str(args.agibot_root)))
    ).expanduser().resolve()
    args.agibot_viscam_root = (
        args.agibot_viscam_root or args.data_root / "agibot_combined"
    ).expanduser().resolve()
    return args


def _print_human(reports: Sequence[DatasetReport], files_only: bool) -> None:
    print("LACWM training-data preflight (read-only)")
    print(f"mode: {'FILES ONLY' if files_only else 'STRICT CONTENT'}")
    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        cap = "all" if report.cap is None else f"{report.cap:,}"
        print(
            f"[{status}] {report.name:7s} discovered={report.discovered:,} "
            f"complete={report.complete:,} selected={report.selected:,} "
            f"active_complete={report.active_complete:,} "
            f"checked={report.checked:,} cap={cap} expected={report.expected:,} "
            f"errors={report.error_count:,} warnings={report.warning_count:,}"
        )
        print(f"       source: {report.source}")
        for finding in report.findings:
            location = f" ({finding.path})" if finding.path else ""
            print(f"       {finding.severity.upper()}: {finding.message}{location}")
        hidden = report.error_count + report.warning_count - len(report.findings)
        if hidden > 0:
            print(f"       ... {hidden:,} additional findings omitted")
    total_errors = sum(report.error_count for report in reports)
    total_warnings = sum(report.warning_count for report in reports)
    print(f"result: {'PASS' if total_errors == 0 else 'FAIL'} ({total_errors:,} errors, {total_warnings:,} warnings)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = _normalize_args(parser.parse_args(argv), parser)
    validators = {
        "droid": validate_droid,
        "egodex": validate_egodex,
        "agibot": validate_agibot,
        "abc": validate_abc,
    }
    reports: list[DatasetReport] = []
    for name in args.datasets:
        try:
            reports.append(validators[name](args))
        except Exception as exc:
            report = DatasetReport(
                name,
                str(args.data_root),
                getattr(args, f"{name}_cap"),
                getattr(args, f"{name}_expected"),
            )
            report.error(f"unexpected top-level validator exception: {exc}")
            reports.append(report)
    total_errors = sum(report.error_count for report in reports)
    if args.json:
        generated_at = datetime.now(timezone.utc).isoformat()
        invocation = {
            "data_root": str(args.data_root),
            "datasets": list(args.datasets),
            "min_timesteps": args.min_timesteps,
            "validate_all": args.validate_all,
            "files_only": args.files_only,
            "caps": {name: getattr(args, f"{name}_cap") for name in DATASET_NAMES},
            "expected": {name: getattr(args, f"{name}_expected") for name in DATASET_NAMES},
            "sources": {
                "droid": str(args.droid_root),
                "egodex": str(args.egodex_manifest),
                "agibot": str(args.agibot_manifest),
                "abc": str(args.abc_manifest),
            },
            "agibot_roots": {
                "scr": str(args.agibot_root),
                "alpha": str(args.agibot_alpha_root),
                "beta": str(args.agibot_beta_root),
                "viscam": str(args.agibot_viscam_root),
            },
        }
        print(
            json.dumps(
                {
                    "schema_version": 2,
                    "generated_at_utc": generated_at,
                    "read_only": True,
                    "passed": total_errors == 0,
                    "git_commit": _git_output("rev-parse", "HEAD"),
                    "git_status": _git_output("status", "--porcelain"),
                    "validator_sha256": _sha256_file(Path(__file__).resolve()),
                    "invocation": invocation,
                    "reports": [report.to_dict() for report in reports],
                },
                indent=2,
            )
        )
    else:
        _print_human(reports, args.files_only)
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
