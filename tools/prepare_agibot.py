#!/usr/bin/env python3
"""Prepare an extracted official AgiBot corpus for production training.

This command never invents missing robot or camera motion.  It optionally
extracts an explicit, checksummed archive plan, qualifies complete episodes,
and atomically publishes success/runtime manifests plus a provenance report.
Episodes with incomplete or malformed payloads are reported and excluded.

The default mode is read-only.  Pass ``--execute`` to extract archives or write
manifests.  The small upstream ``sample_dataset.tar`` is intentionally rejected:
it lacks the production motion streams required by this repository.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Optional, Sequence


OFFICIAL_REPO = "agibot-world/AgiBotWorld-Alpha"
OFFICIAL_REVISION = "128665c9e0244c45d1cbe5c13f5a4706afd24f27"
SECTIONS = ("observations", "parameters", "proprio_stats")
VIDEOS = ("head_color.mp4", "hand_left_color.mp4", "hand_right_color.mp4")
ALIGNED_EXTRINSICS = (
    "head_extrinsic_params_aligned.json",
    "hand_left_extrinsic_params_aligned.json",
    "hand_right_extrinsic_params_aligned.json",
)
H5_FIELDS: dict[str, tuple[int, ...]] = {
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
QUALIFICATION_MARKERS = (
    "QUALIFICATION_ONLY_DO_NOT_TRAIN.md",
    "qualification_provenance.json",
    ".qualification-only",
    ".qualification_only",
    "QUALIFICATION_ONLY",
    "qualification-only.marker",
    "qualification_only.marker",
)
QUALIFICATION_SIGNALS = (
    "qualification_only",
    "qualification only",
    "static_extrinsic_repetition",
    "static extrinsic repetition",
    "synthesized_identity_base_pose",
    "synthesized identity base pose",
    "synthesized as identity",
    "identity base pose",
    "synthesized by repeating",
    "repeating a single static extrinsic",
    "repeated camera",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class PreparationError(RuntimeError):
    """A fail-closed preparation error."""


@dataclass(frozen=True, order=True)
class Episode:
    task_id: str
    episode_id: str


@dataclass(frozen=True)
class Archive:
    section: str
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class Rejection:
    task_id: str
    episode_id: str
    reasons: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(handle: Any, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
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


def _json_payload(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def qualification_markers(root: Path) -> list[Path]:
    """Return explicit qualification markers next to or above ``agibot/``."""
    locations = (root, root.parent)
    detected = {
        directory / name
        for directory in locations
        for name in QUALIFICATION_MARKERS
        if (directory / name).is_file()
    }
    for directory in locations:
        for path in directory.glob("*provenance*.json"):
            if not path.is_file() or path in detected:
                continue
            try:
                serialized = json.dumps(
                    json.loads(path.read_text(encoding="utf-8")), sort_keys=True
                ).lower()
            except (OSError, json.JSONDecodeError):
                continue
            if "agibot" in serialized and any(
                signal in serialized for signal in QUALIFICATION_SIGNALS
            ):
                detected.add(path)
    return sorted(detected, key=str)


def parse_archive_plan(path: Path) -> list[Archive]:
    """Parse ``section hf/relative/path.tar sha256`` records."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise PreparationError(f"archive plan is missing or empty: {path}")
    archives: list[Archive] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            value = raw.strip()
            if not value or value.startswith("#"):
                continue
            fields = value.split()
            if len(fields) != 3:
                raise PreparationError(
                    f"{path}:{line_number}: expected '<section> <relative.tar> <sha256>'"
                )
            section, relative, digest = fields
            digest = digest.lower()
            pure = PurePosixPath(relative)
            if section not in SECTIONS:
                raise PreparationError(f"{path}:{line_number}: invalid section {section!r}")
            if pure.is_absolute() or ".." in pure.parts or pure.suffix != ".tar":
                raise PreparationError(f"{path}:{line_number}: unsafe/non-tar path {relative!r}")
            if not SHA256_RE.fullmatch(digest):
                raise PreparationError(f"{path}:{line_number}: invalid SHA-256")
            if pure.parts[0] != section:
                raise PreparationError(
                    f"{path}:{line_number}: path {relative!r} is outside section {section!r}"
                )
            archives.append(Archive(section, pure.as_posix(), digest))
    if not archives:
        raise PreparationError(f"archive plan contains no archives: {path}")
    if len({item.relative_path for item in archives}) != len(archives):
        raise PreparationError(f"archive plan contains duplicate paths: {path}")
    if not {item.section for item in archives}.issuperset(SECTIONS):
        missing = sorted(set(SECTIONS) - {item.section for item in archives})
        raise PreparationError(f"archive plan omits required sections: {missing}")
    return archives


def verify_official_archive_plan(archives: Sequence[Archive]) -> list[dict[str, Any]]:
    """Bind planned paths/hashes to the pinned official Hugging Face revision."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise PreparationError(
            "huggingface-hub is required to verify the official AgiBot archive inventory"
        ) from exc
    paths = [item.relative_path for item in archives]
    try:
        entries = HfApi().get_paths_info(
            OFFICIAL_REPO,
            paths,
            repo_type="dataset",
            revision=OFFICIAL_REVISION,
            token=True,
            expand=True,
        )
    except Exception as exc:
        raise PreparationError(f"cannot verify pinned official archive inventory: {exc}") from exc
    by_path = {getattr(entry, "path", None): entry for entry in entries}
    verified: list[dict[str, Any]] = []
    for item in archives:
        entry = by_path.get(item.relative_path)
        lfs = getattr(entry, "lfs", None) if entry is not None else None
        upstream_hash = getattr(lfs, "sha256", None)
        size = int(getattr(entry, "size", 0) or 0) if entry is not None else 0
        if entry is None or upstream_hash != item.sha256 or size <= 0:
            raise PreparationError(
                f"archive plan does not match pinned official inventory: {item.relative_path}"
            )
        verified.append(
            {
                "section": item.section,
                "path": item.relative_path,
                "sha256": item.sha256,
                "size": size,
            }
        )
    return verified


def parse_episode_plan(
    path: Path, expected_dataset_id: Optional[str] = None
) -> list[Episode]:
    """Read an exact, ordered ``task_id,episode_id[,dataset]`` CSV plan."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise PreparationError(f"episode plan is missing or empty: {path}")
    episodes: list[Episode] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        try:
            first = [value.strip() for value in next(reader)]
        except StopIteration as exc:
            raise PreparationError(f"episode plan is empty: {path}") from exc
        expected_headers = (
            ["task_id", "episode_id"],
            ["task_id", "episode_id", "dataset"],
        )
        if first not in expected_headers:
            raise PreparationError(
                f"invalid episode-plan header {first}; expected task_id,episode_id[,dataset]"
            )
        width = len(first)
        for line_number, raw in enumerate(reader, 2):
            values = [value.strip() for value in raw]
            if len(values) != width or any(not value for value in values):
                raise PreparationError(f"invalid episode-plan row {line_number}: {raw}")
            if any(
                value in {".", ".."} or not SAFE_ID_RE.fullmatch(value)
                for value in values[:2]
            ):
                raise PreparationError(
                    f"unsafe task/episode identifier at episode-plan row {line_number}: {values[:2]}"
                )
            if width == 3 and expected_dataset_id is not None and values[2] != expected_dataset_id:
                raise PreparationError(
                    f"episode-plan row {line_number} dataset {values[2]!r} does not match "
                    f"--dataset-id {expected_dataset_id!r}"
                )
            episodes.append(Episode(values[0], values[1]))
    if not episodes:
        raise PreparationError(f"episode plan contains no episodes: {path}")
    if len(set(episodes)) != len(episodes):
        raise PreparationError(f"episode plan contains duplicate task/episode pairs: {path}")
    return episodes


def _safe_member_name(member: tarfile.TarInfo) -> Optional[PurePosixPath]:
    raw = member.name[2:] if member.name.startswith("./") else member.name
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts:
        raise PreparationError(f"unsafe tar member: {member.name!r}")
    if member.isdir():
        return None
    if not member.isfile():
        raise PreparationError(f"unsupported non-regular tar member: {member.name!r}")
    if len(pure.parts) < 3:
        raise PreparationError(f"unexpected AgiBot tar member: {member.name!r}")
    return pure


def _extract_archive(
    archive: Path,
    archive_handle: Any,
    section_root: Path,
    containment_root: Path,
    execute: bool,
    required_payloads: Optional[set[Path]] = None,
    covered_payloads: Optional[set[Path]] = None,
    required_hashes: Optional[dict[Path, str]] = None,
) -> dict[str, int]:
    """Safely and resumably extract one official uncompressed tar."""
    files = written = reused = 0
    containment_root = containment_root.resolve()
    section_root_resolved = section_root.resolve(strict=False)
    if not section_root_resolved.is_relative_to(containment_root):
        raise PreparationError(f"section destination escapes output root: {section_root}")
    for parent in (section_root, *section_root.parents):
        if parent == containment_root.parent:
            break
        if parent.is_symlink():
            raise PreparationError(f"refusing symlinked extraction directory: {parent}")
    archive_handle.seek(0)
    with tarfile.open(fileobj=archive_handle, mode="r:") as bundle:
        for member in bundle:
            relative = _safe_member_name(member)
            if relative is None:
                continue
            files += 1
            destination = section_root.joinpath(*relative.parts)
            resolved = destination.resolve(strict=False)
            if not resolved.is_relative_to(containment_root):
                raise PreparationError(f"tar member escapes destination: {member.name!r}")
            if (
                required_payloads is not None
                and covered_payloads is not None
                and resolved in required_payloads
            ):
                covered_payloads.add(resolved)
            is_required = required_payloads is not None and resolved in required_payloads
            if destination.is_symlink():
                raise PreparationError(f"refusing existing symlink destination: {destination}")
            if destination.is_file() and destination.stat().st_size == member.size:
                if not execute:
                    continue
                existing_digest = sha256_file(destination)
                existing_source = bundle.extractfile(member)
                if existing_source is None:
                    raise PreparationError(f"cannot read tar member: {member.name!r}")
                with existing_source:
                    source_digest = _sha256_stream(existing_source)
                if existing_digest == source_digest:
                    if is_required and required_hashes is not None:
                        required_hashes[resolved] = source_digest
                    reused += 1
                    continue
            if not execute:
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise PreparationError(f"cannot read tar member: {member.name!r}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            current = destination.parent
            while current != containment_root.parent:
                if current.is_symlink():
                    raise PreparationError(f"refusing symlinked extraction directory: {current}")
                if current == containment_root:
                    break
                current = current.parent
            temporary: Optional[str] = None
            try:
                with source:
                    digest = hashlib.sha256() if is_required else None
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        dir=destination.parent,
                        prefix=f".{destination.name}.",
                        suffix=".partial",
                        delete=False,
                    ) as output:
                        temporary = output.name
                        while chunk := source.read(8 * 1024 * 1024):
                            output.write(chunk)
                            if digest is not None:
                                digest.update(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                temporary_path = Path(temporary)
                if temporary_path.stat().st_size != member.size:
                    raise PreparationError(
                        f"short extraction for {member.name!r}: {temporary_path.stat().st_size} != {member.size}"
                    )
                os.replace(temporary_path, destination)
                temporary = None
                if digest is not None and required_hashes is not None:
                    required_hashes[resolved] = digest.hexdigest()
                written += 1
            finally:
                if temporary is not None:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
    if files == 0:
        raise PreparationError(f"archive has no regular payloads: {archive}")
    return {"files": files, "written": written, "reused": reused}


def extract_plan(
    archive_root: Path,
    output_root: Path,
    archives: Sequence[Archive],
    execute: bool,
    required_payloads: Optional[set[Path]] = None,
    required_hashes: Optional[dict[Path, str]] = None,
) -> list[dict[str, Any]]:
    """Verify every planned archive, then extract it under its section."""
    results: list[dict[str, Any]] = []
    archive_root = archive_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    normalized_required = (
        {path.expanduser().resolve(strict=False) for path in required_payloads}
        if required_payloads is not None
        else None
    )
    covered_payloads: set[Path] = set()
    with contextlib.ExitStack() as stack:
        verified: list[tuple[Archive, Path, str, Any]] = []
        for item in archives:
            path = archive_root.joinpath(*PurePosixPath(item.relative_path).parts).resolve()
            if not path.is_relative_to(archive_root):
                raise PreparationError(f"planned archive escapes archive root: {path}")
            if path.is_symlink() or not path.is_file():
                raise PreparationError(f"planned archive is missing or symlinked: {path}")
            handle = stack.enter_context(path.open("rb"))
            actual = _sha256_stream(handle)
            if actual != item.sha256:
                raise PreparationError(
                    f"archive SHA-256 mismatch for {path}: {actual} != {item.sha256}"
                )
            verified.append((item, path, actual, handle))

        # Keep every verified descriptor open through extraction. Reopening a
        # path here would permit an archive-replacement race after its hash was
        # approved. Do not mutate output until all planned archives pass.
        for item, path, actual, handle in verified:
            relative = PurePosixPath(item.relative_path)
            # Observation archives are stored as observations/<task>/<range>.tar
            # and contain only <episode>/... members. Preserve any directory
            # components between the declared section and archive filename. The
            # global parameter/proprio archives live directly under their section,
            # so their prefix is empty and their members already include task IDs.
            archive_prefix = relative.parts[1:-1]
            section_root = output_root.joinpath(item.section, *archive_prefix)
            stats = _extract_archive(
                path,
                handle,
                section_root,
                output_root,
                execute,
                normalized_required,
                covered_payloads,
                required_hashes,
            )
            results.append(
                {
                    "section": item.section,
                    "archive": item.relative_path,
                    "destination_prefix": "/".join(archive_prefix),
                    "sha256": actual,
                    **stats,
                }
            )

        # Detect in-place mutation while extraction was running before the
        # staging tree can be published.
        for item, path, _actual, handle in verified:
            handle.seek(0)
            final_hash = _sha256_stream(handle)
            if final_hash != item.sha256:
                raise PreparationError(
                    f"archive changed during extraction: {path}: {final_hash} != {item.sha256}"
                )
    if normalized_required is not None:
        missing = sorted(normalized_required - covered_payloads, key=str)
        if missing:
            raise PreparationError(
                f"reviewed archives do not contain {len(missing)} required planned payloads; "
                f"first entries: {[str(path) for path in missing[:5]]}"
            )
        if execute and required_hashes is not None:
            missing_hashes = sorted(normalized_required - set(required_hashes), key=str)
            if missing_hashes:
                raise PreparationError(
                    f"failed to hash {len(missing_hashes)} required planned payloads"
                )
        for result in results:
            result["planned_payload_count"] = len(normalized_required)
        if results:
            results[0]["covered_planned_payload_count"] = len(covered_payloads)
    return results


def _episode_dirs(root: Path, section: str) -> set[Episode]:
    section_root = root / section
    if not section_root.is_dir():
        return set()
    return {
        Episode(task_dir.name, episode_dir.name)
        for task_dir in section_root.iterdir()
        if task_dir.is_dir() and not task_dir.is_symlink()
        for episode_dir in task_dir.iterdir()
        if episode_dir.is_dir() and not episode_dir.is_symlink()
    }


def discover_episodes(root: Path) -> list[Episode]:
    candidates: set[Episode] = set()
    for section in SECTIONS:
        candidates.update(_episode_dirs(root, section))
    return sorted(candidates)


def episode_paths(root: Path, episode: Episode) -> tuple[Path, tuple[Path, ...], tuple[Path, ...]]:
    task, ep = episode.task_id, episode.episode_id
    proprio = root / "proprio_stats" / task / ep / "proprio_stats.h5"
    video_dir = root / "observations" / task / ep / "videos"
    camera_dir = root / "parameters" / task / ep / "parameters" / "camera"
    return (
        proprio,
        tuple(video_dir / name for name in VIDEOS),
        tuple(camera_dir / name for name in ALIGNED_EXTRINSICS),
    )


def required_episode_payloads(root: Path, episodes: Sequence[Episode]) -> set[Path]:
    return {
        path.resolve(strict=False)
        for episode in episodes
        for path in (
            episode_paths(root, episode)[0],
            *episode_paths(root, episode)[1],
            *episode_paths(root, episode)[2],
        )
    }


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_rotation(rotation: Any, tolerance: float = 1e-3) -> bool:
    if not isinstance(rotation, list) or len(rotation) != 3:
        return False
    if any(not isinstance(row, list) or len(row) != 3 for row in rotation):
        return False
    if any(not _finite(value) for row in rotation for value in row):
        return False
    # Dependency-free SO(3) checks.
    rows = [[float(value) for value in row] for row in rotation]
    for i in range(3):
        for j in range(3):
            dot = sum(rows[i][k] * rows[j][k] for k in range(3))
            expected = 1.0 if i == j else 0.0
            if abs(dot - expected) > tolerance:
                return False
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    return determinant > 0.0 and abs(determinant - 1.0) <= tolerance * 2


def _validate_h5(path: Path, min_timesteps: int) -> tuple[Optional[int], list[str]]:
    errors: list[str] = []
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise PreparationError("h5py and numpy are required for AgiBot preparation") from exc
    try:
        with h5py.File(path, "r") as handle:
            missing = [key for key in H5_FIELDS if key not in handle]
            if missing:
                return None, [f"missing HDF5 fields: {missing}"]
            timestamps = np.asarray(handle["timestamp"])
            if timestamps.ndim != 1:
                return None, [f"timestamp shape {timestamps.shape} is not [T]"]
            timesteps = int(timestamps.shape[0])
            if timesteps < min_timesteps:
                errors.append(f"trajectory has {timesteps} timesteps; need >= {min_timesteps}")
            if not np.isfinite(timestamps).all():
                errors.append("timestamp contains NaN/Inf")
            elif timesteps > 1 and not np.all(timestamps[1:] > timestamps[:-1]):
                errors.append("timestamps are not strictly increasing")
            for key, tail in H5_FIELDS.items():
                dataset = handle[key]
                expected = (timesteps, *tail)
                if tuple(dataset.shape) != expected:
                    errors.append(f"{key} shape {tuple(dataset.shape)} != {expected}")
                    continue
                if not np.isfinite(dataset[...]).all():
                    errors.append(f"{key} contains NaN/Inf")

            for key in ("state/end/orientation", "action/end/orientation"):
                values = np.asarray(handle[key], dtype=np.float64)
                norms = np.linalg.norm(values, axis=-1)
                if not np.all(np.abs(norms - 1.0) <= 1e-3):
                    errors.append(f"{key} contains non-unit/zero quaternions")

            base_position = np.asarray(handle["state/robot/position"], dtype=np.float64)
            base_orientation = np.asarray(handle["state/robot/orientation"], dtype=np.float64)
            base_norms = np.linalg.norm(base_orientation, axis=-1)
            zero_sentinel = base_norms <= 1e-8
            unit_quaternion = np.abs(base_norms - 1.0) <= 1e-3
            if not np.all(zero_sentinel | unit_quaternion):
                errors.append(
                    "state/robot/orientation contains quaternions that are neither "
                    "unit length nor the official all-zero fixed-base sentinel"
                )
            if np.any(zero_sentinel) and not np.all(zero_sentinel):
                errors.append("state/robot/orientation mixes zero sentinels and quaternions")

            position_static = bool(
                timesteps <= 1
                or np.all(np.abs(base_position - base_position[:1]) <= 1e-7)
            )
            orientation_static = bool(
                timesteps <= 1
                or np.all(np.abs(base_orientation - base_orientation[:1]) <= 1e-7)
            )
            velocity_nonzero = False
            if "action/robot/velocity" in handle:
                velocity = np.asarray(handle["action/robot/velocity"])
                if tuple(velocity.shape) != (timesteps, 2):
                    errors.append(
                        f"action/robot/velocity shape {tuple(velocity.shape)} != {(timesteps, 2)}"
                    )
                elif not np.isfinite(velocity).all():
                    errors.append("action/robot/velocity contains NaN/Inf")
                else:
                    velocity_nonzero = bool(np.any(np.abs(velocity) > 1e-6))
            if velocity_nonzero and position_static and orientation_static:
                errors.append(
                    "nonzero commanded base velocity is inconsistent with a static base-pose stream"
                )
            if np.all(zero_sentinel) and (not position_static or velocity_nonzero):
                errors.append(
                    "all-zero robot-orientation sentinel is valid only for a stationary fixed base"
                )
            return timesteps, errors
    except Exception as exc:
        return None, [f"cannot inspect HDF5: {exc}"]


def _validate_camera_json(path: Path, timesteps: int) -> list[str]:
    errors: list[str] = []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"cannot read aligned camera JSON: {exc}"]
    if not isinstance(records, list):
        return ["aligned camera JSON is not a list"]
    if len(records) != timesteps:
        return [f"aligned camera length {len(records)} != trajectory length {timesteps}"]
    for index, record in enumerate(records):
        extrinsic = record.get("extrinsic") if isinstance(record, dict) else None
        rotation = extrinsic.get("rotation_matrix") if isinstance(extrinsic, dict) else None
        translation = extrinsic.get("translation_vector") if isinstance(extrinsic, dict) else None
        if not _valid_rotation(rotation):
            errors.append(f"record {index} has a non-SO(3) rotation")
            break
        if not isinstance(translation, list) or len(translation) != 3 or not all(_finite(v) for v in translation):
            errors.append(f"record {index} has an invalid translation")
            break
    return errors


def _validate_video(path: Path, timesteps: int) -> list[str]:
    try:
        import av
    except ImportError as exc:
        raise PreparationError("PyAV is required for AgiBot video qualification") from exc
    errors: list[str] = []
    try:
        # Strict error recognition turns decoder-detected corruption into a hard
        # failure.  A first/last-frame probe is insufficient: damaged packets
        # in a middle GOP can otherwise survive qualification and fail later
        # in a training worker.
        container = av.open(str(path), mode="r")
        try:
            streams = list(container.streams.video)
            if not streams:
                return ["contains no video stream"]
            stream = streams[0]
            stream.codec_context.options = {
                **stream.codec_context.options,
                "err_detect": "crccheck+bitstream+buffer+explode",
            }
            if int(getattr(stream, "width", 0) or 0) <= 0 or int(
                getattr(stream, "height", 0) or 0
            ) <= 0:
                errors.append("video has invalid dimensions")
            frame_count = int(getattr(stream, "frames", 0) or 0)
            if frame_count > 0 and frame_count != timesteps:
                errors.append(
                    f"video reports {frame_count} frames; trajectory has {timesteps}"
                )
            decoded_frames = 0
            reported_corruption = False
            for frame in container.decode(stream):
                if frame.width <= 0 or frame.height <= 0:
                    errors.append(f"decoded frame {decoded_frames} has invalid dimensions")
                if bool(getattr(frame, "is_corrupt", False)) and not reported_corruption:
                    errors.append(f"decoded frame {decoded_frames} is marked corrupt")
                    reported_corruption = True
                decoded_frames += 1
            if decoded_frames == 0:
                errors.append("video contains no decodable frames")
            elif decoded_frames != timesteps:
                errors.append(
                    f"video decodes to {decoded_frames} frames; trajectory has {timesteps}"
                )
        finally:
            container.close()
    except Exception as exc:
        errors.append(f"cannot open/decode video: {exc}")
    return errors


def qualify_episode(root: Path, episode: Episode, min_timesteps: int) -> list[str]:
    proprio, videos, camera_jsons = episode_paths(root, episode)
    required = (proprio, *videos, *camera_jsons)
    missing = [str(path.relative_to(root)) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        return [f"missing/empty required payloads: {missing}"]
    timesteps, errors = _validate_h5(proprio, min_timesteps)
    if timesteps is None:
        return errors
    for path in camera_jsons:
        errors.extend(f"{path.name}: {message}" for message in _validate_camera_json(path, timesteps))
    for path in videos:
        errors.extend(f"{path.name}: {message}" for message in _validate_video(path, timesteps))
    return errors


def _manifest_payload(episodes: Iterable[Episode], dataset_id: str) -> str:
    lines = ["task_id,episode_id,dataset"]
    lines.extend(f"{item.task_id},{item.episode_id},{dataset_id}" for item in episodes)
    return "\n".join(lines) + "\n"


def _payload_hash_manifest(root: Path, hashes: dict[Path, str]) -> str:
    rows: list[tuple[str, str]] = []
    root = root.resolve()
    for path, digest in hashes.items():
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise PreparationError(f"hashed payload escapes staging root: {path}") from exc
        if not SHA256_RE.fullmatch(digest):
            raise PreparationError(f"invalid payload SHA-256 for {path}")
        rows.append((relative, digest))
    if not rows:
        raise PreparationError("refusing to publish an empty payload hash manifest")
    return "".join(f"{digest}  {relative}\n" for relative, digest in sorted(rows))


def prepare(
    root: Path,
    limit: int,
    dataset_id: str,
    min_timesteps: int,
    manifest: Path,
    success_manifest: Path,
    report_path: Path,
    execute: bool,
    validate_all: bool,
    archive_results: Optional[list[dict[str, Any]]] = None,
    source_revision: str = OFFICIAL_REVISION,
    episode_plan: Optional[Sequence[Episode]] = None,
    archive_verified: bool = False,
    archive_plan_sha256: Optional[str] = None,
    published_root: Optional[Path] = None,
    official_inventory: Optional[list[dict[str, Any]]] = None,
    payload_manifest_path: Optional[Path] = None,
    payload_manifest_sha256: Optional[str] = None,
    payload_count: int = 0,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    published_root = (published_root or root).expanduser().resolve()
    if limit <= 0 or min_timesteps <= 0:
        raise PreparationError("limit and min-timesteps must be positive integers")
    if execute and (
        not archive_verified
        or not official_inventory
        or payload_manifest_path is None
        or not payload_manifest_path.is_file()
        or payload_manifest_sha256 != sha256_file(payload_manifest_path)
        or payload_count < limit * 7
    ):
        raise PreparationError(
            "publishing a production manifest requires the pinned official archive inventory, "
            "verified/hash-bound payloads, and a clean staging root"
        )
    markers = qualification_markers(root)
    if markers:
        raise PreparationError(
            "production preparation refuses qualification-only data: "
            + ", ".join(str(path) for path in markers)
        )
    discovered = discover_episodes(root)
    candidates = list(episode_plan) if episode_plan is not None else discovered
    if not candidates:
        raise PreparationError(f"no extracted AgiBot episodes found under {root}")
    accepted: list[Episode] = []
    rejected: list[Rejection] = []
    checked = 0
    for episode in candidates:
        reasons = qualify_episode(root, episode, min_timesteps)
        checked += 1
        if reasons:
            rejected.append(Rejection(episode.task_id, episode.episode_id, tuple(reasons)))
        else:
            accepted.append(episode)
        if not validate_all and len(accepted) >= limit:
            break
    selected = accepted[:limit]
    success_payload = _manifest_payload(accepted, dataset_id) if accepted else ""
    runtime_payload = _manifest_payload(selected, dataset_id) if selected else ""
    try:
        published_manifest = published_root / manifest.resolve().relative_to(root)
        published_success_manifest = published_root / success_manifest.resolve().relative_to(root)
        published_report = published_root / report_path.resolve().relative_to(root)
        published_payload_manifest = (
            published_root / payload_manifest_path.resolve().relative_to(root)
            if payload_manifest_path is not None
            else None
        )
    except ValueError as exc:
        raise PreparationError(
            "manifest and success-manifest must be contained by the prepared root"
        ) from exc
    report: dict[str, Any] = {
        "schema_version": 1,
        "preparer_sha256": sha256_file(Path(__file__).resolve()),
        "generated_at": _utc_now(),
        "profile": "production" if archive_verified else "structural_preview",
        "source": {"repo_id": OFFICIAL_REPO, "revision": source_revision},
        "root": str(published_root),
        "limit": limit,
        "min_timesteps": min_timesteps,
        "discovered_count": len(discovered),
        "candidate_count": len(candidates),
        "episode_plan_supplied": episode_plan is not None,
        "checked_count": checked,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": [asdict(item) for item in accepted],
        "rejected": [asdict(item) for item in rejected],
        "archive_results": archive_results or [],
        "archive_verified": archive_verified,
        "official_inventory_verified": bool(official_inventory),
        "official_inventory": official_inventory or [],
        "payload_manifest": (
            str(published_payload_manifest) if published_payload_manifest is not None else None
        ),
        "payload_manifest_sha256": payload_manifest_sha256,
        "payload_count": payload_count,
        "archive_plan_sha256": archive_plan_sha256,
        "manifest": str(published_manifest),
        "success_manifest": str(published_success_manifest),
        "report": str(published_report),
        "manifest_sha256": _sha256_text(runtime_payload) if runtime_payload else None,
        "success_manifest_sha256": _sha256_text(success_payload) if success_payload else None,
        "selected_count": len(selected),
        "synthetic_camera_motion": False if archive_verified else None,
        "synthetic_base_pose": False if archive_verified else None,
        "passed": len(accepted) >= limit,
        "executed": execute,
    }
    if len(accepted) < limit:
        if execute:
            _atomic_write(report_path, _json_payload(report))
        examples = [
            f"{item.task_id}/{item.episode_id}: {'; '.join(item.reasons)}"
            for item in rejected[:3]
        ]
        raise PreparationError(
            f"only {len(accepted):,} production-valid episodes found; need {limit:,}; "
            f"report={report_path}; examples={examples}"
        )
    if execute:
        _atomic_write(success_manifest, success_payload)
        _atomic_write(manifest, runtime_payload)
        _atomic_write(report_path, _json_payload(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="canonical output/extracted agibot root")
    parser.add_argument("--limit", type=int, required=True, help="finite number of valid runtime episodes")
    parser.add_argument("--dataset-id", default="scr", choices=("alpha", "beta", "viscam", "scr"))
    parser.add_argument("--min-timesteps", type=int, default=66)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--success-manifest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--archive-plan", type=Path, help="checksummed archive plan to verify/extract")
    parser.add_argument("--archive-root", type=Path, help="root containing archive-plan relative paths")
    parser.add_argument(
        "--episode-plan",
        type=Path,
        help="exact task_id,episode_id[,dataset] CSV selection; avoids selecting unrelated episodes from coarse archives",
    )
    parser.add_argument("--source-revision", default=OFFICIAL_REVISION)
    parser.add_argument("--validate-all", action="store_true")
    parser.add_argument("--execute", action="store_true", help="permit extraction and atomic manifest/report writes")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    manifest = (args.manifest or root / "manifest.csv").expanduser().resolve()
    success_manifest = (args.success_manifest or root / "manifest.success.csv").expanduser().resolve()
    report = (args.report or root / "preparation_report.json").expanduser().resolve()
    if len({manifest, success_manifest, report}) != 3:
        print("AgiBot preparation failed: manifest/success/report paths must be distinct", file=sys.stderr)
        return 1
    try:
        episode_plan = (
            parse_episode_plan(
                args.episode_plan.expanduser().resolve(), args.dataset_id
            )
            if args.episode_plan
            else None
        )
        if args.execute:
            output_paths = {
                "manifest": manifest,
                "success manifest": success_manifest,
                "preparation report": report,
            }
            outside = [
                f"{name}={path}"
                for name, path in output_paths.items()
                if not path.is_relative_to(root)
            ]
            if outside:
                raise PreparationError(
                    "production output paths must be inside the dataset root: "
                    + ", ".join(outside)
                )
            canonical_report = (root / "preparation_report.json").resolve(strict=False)
            if report != canonical_report:
                raise PreparationError(
                    f"production report must use canonical path {canonical_report}"
                )
            reserved = {
                (root / "payloads.sha256").resolve(strict=False),
                (root / ".agibot_archive_plan.sha256").resolve(strict=False),
            }
            if episode_plan is not None:
                reserved.update(required_episode_payloads(root, episode_plan))
            collisions = sorted(
                {manifest, success_manifest, report}.intersection(reserved), key=str
            )
            if collisions:
                raise PreparationError(
                    "manifest/success/report paths collide with reserved lineage or "
                    f"episode payloads: {[str(path) for path in collisions]}"
                )
        if args.archive_plan is None:
            if args.archive_root is not None:
                raise PreparationError("--archive-root requires --archive-plan")
            if args.execute:
                raise PreparationError(
                    "--execute requires --archive-plan and --archive-root; an existing "
                    "tree can be structurally previewed but cannot be certified as production"
                )
            result = prepare(
                root=root,
                limit=args.limit,
                dataset_id=args.dataset_id,
                min_timesteps=args.min_timesteps,
                manifest=manifest,
                success_manifest=success_manifest,
                report_path=report,
                execute=False,
                validate_all=args.validate_all,
                source_revision=args.source_revision,
                episode_plan=episode_plan,
            )
        else:
            if args.archive_root is None:
                raise PreparationError("--archive-plan requires --archive-root")
            if episode_plan is None:
                raise PreparationError(
                    "archive preparation requires an exact --episode-plan before extraction"
                )
            if len(episode_plan) != args.limit:
                raise PreparationError(
                    f"episode plan has {len(episode_plan):,} entries; expected exactly --limit={args.limit:,}"
                )
            if args.source_revision != OFFICIAL_REVISION:
                raise PreparationError(
                    f"archive preparation is pinned to source revision {OFFICIAL_REVISION}"
                )
            archive_plan_path = args.archive_plan.expanduser().resolve()
            archives = parse_archive_plan(archive_plan_path)
            official_inventory = verify_official_archive_plan(archives)
            archive_plan_digest = sha256_file(archive_plan_path)

            if not args.execute:
                archive_results = extract_plan(
                    args.archive_root,
                    root,
                    archives,
                    execute=False,
                    required_payloads=required_episode_payloads(root, episode_plan),
                )
                preview: dict[str, Any] = {
                    "schema_version": 1,
                    "profile": "archive_plan_preview",
                    "executed": False,
                    "archive_plan_sha256": archive_plan_digest,
                    "official_inventory": official_inventory,
                    "planned_episode_count": len(episode_plan),
                    "archive_results": archive_results,
                    "note": "all archive hashes and members passed; --execute is required for clean staged publication",
                }
                if root.is_dir():
                    preview["existing_tree"] = prepare(
                        root=root,
                        limit=args.limit,
                        dataset_id=args.dataset_id,
                        min_timesteps=args.min_timesteps,
                        manifest=manifest,
                        success_manifest=success_manifest,
                        report_path=report,
                        execute=False,
                        validate_all=args.validate_all,
                        source_revision=args.source_revision,
                        episode_plan=episode_plan,
                    )
                print(_json_payload(preview), end="")
                return 0

            if root.exists():
                raise PreparationError(
                    f"refusing to overlay production data onto existing root: {root}; "
                    "use a new root or preserve the existing curated manifest"
                )
            if root.parent.is_symlink():
                raise PreparationError(f"refusing symlinked output parent: {root.parent}")
            root.parent.mkdir(parents=True, exist_ok=True)
            staging = root.parent / f".{root.name}.preparing-{archive_plan_digest[:12]}"
            lineage = staging / ".agibot_archive_plan.sha256"
            if staging.exists():
                if not staging.is_dir() or staging.is_symlink():
                    raise PreparationError(f"invalid preparation staging path: {staging}")
                if not lineage.is_file() or lineage.read_text(encoding="utf-8").strip() != archive_plan_digest:
                    raise PreparationError(
                        f"staging directory is not bound to this archive plan: {staging}"
                    )
            else:
                staging.mkdir(mode=0o700)
                _atomic_write(lineage, archive_plan_digest + "\n")

            def staged_path(final_path: Path) -> Path:
                try:
                    relative = final_path.relative_to(root)
                except ValueError as exc:
                    raise PreparationError(
                        f"publication path must be inside final root {root}: {final_path}"
                    ) from exc
                return staging / relative

            payload_hashes: dict[Path, str] = {}
            archive_results = extract_plan(
                args.archive_root,
                staging,
                archives,
                execute=True,
                required_payloads=required_episode_payloads(staging, episode_plan),
                required_hashes=payload_hashes,
            )
            payload_manifest_path = staging / "payloads.sha256"
            _atomic_write(
                payload_manifest_path,
                _payload_hash_manifest(staging, payload_hashes),
            )
            payload_manifest_sha256 = sha256_file(payload_manifest_path)
            result = prepare(
                root=staging,
                published_root=root,
                limit=args.limit,
                dataset_id=args.dataset_id,
                min_timesteps=args.min_timesteps,
                manifest=staged_path(manifest),
                success_manifest=staged_path(success_manifest),
                report_path=staged_path(report),
                execute=True,
                validate_all=args.validate_all,
                archive_results=archive_results,
                source_revision=args.source_revision,
                episode_plan=episode_plan,
                archive_verified=True,
                archive_plan_sha256=archive_plan_digest,
                official_inventory=official_inventory,
                payload_manifest_path=payload_manifest_path,
                payload_manifest_sha256=payload_manifest_sha256,
                payload_count=len(payload_hashes),
            )
            os.replace(staging, root)
    except (PreparationError, OSError, tarfile.TarError) as exc:
        print(f"AgiBot preparation failed: {exc}", file=sys.stderr)
        return 1
    print(_json_payload(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
