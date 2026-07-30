#!/usr/bin/env python3
"""Construct and register a fresh V-JEPA NFE-frontier lockbox.

The original controlled study's test split has already been inspected and
cannot support a new confirmatory claim.  This tool deterministically takes
the next 128 valid episodes from the study's pinned source episode population
after excluding every episode in the original train, validation, and test
manifests.  It writes a fresh clip manifest, then registers the separately
extracted cache only after checking every file hash and provenance field.

The clip rows retain ``split=test`` because the existing offline V-JEPA cache
extractor accepts that physical cache label.  Downstream evidence uses the
semantic split ``lockbox`` and binds this registration identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import build_vjepa2_clip_manifests as clips  # noqa: E402


SCHEMA_VERSION = 1
KIND_CONSTRUCTION = "vjepa2_nfe_frontier_lockbox_construction"
KIND_REGISTRATION = "vjepa2_nfe_frontier_lockbox_registration"
LOCKBOX_CLIPS = 128
LOCKBOX_SEED = 20260730
ATTESTATION_TEXT = (
    "I attest that this newly constructed lockbox has never been scored, "
    "previewed, or used for model or NFE selection."
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
INFERENCE_CRITICAL_PATHS = (
    "projects/latent_action_models/lam",
    "robot_wm/modeling",
    "robot_wm/datasets",
    "tools/env/videox_shim",
)


class LockboxError(RuntimeError):
    """Raised when a lockbox is selectable, reused, or provenance-unsafe."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: Mapping[str, Any]) -> bool:
    identity = payload.get("identity_sha256")
    if not isinstance(identity, str) or SHA256_RE.fullmatch(identity) is None:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest() == identity


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise LockboxError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
    ):
        raise LockboxError(
            f"{label} must be a non-empty non-symlink file: {path}"
        )
    return path.resolve(strict=True)


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise LockboxError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise LockboxError(f"{label} must be a non-symlink directory: {path}")
    return path.resolve(strict=True)


def canonical_episode(value: Any, label: str) -> Path:
    """Reject lexical and physical aliases before episode comparisons."""

    if not isinstance(value, str) or not value:
        raise LockboxError(f"{label} must be a non-empty string")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise LockboxError(f"{label} is not a canonical absolute path")
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise LockboxError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise LockboxError(f"{label} must be a non-symlink directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise LockboxError(f"{label} uses a physical alias: {path} -> {resolved}")
    return resolved


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LockboxError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise LockboxError(f"{label} must contain one JSON object")
    return payload


def _read_rows(path: Path, *, expected_count: int) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LockboxError(f"invalid clip manifest: {path}") from exc
    if len(rows) != expected_count or any(not isinstance(row, dict) for row in rows):
        raise LockboxError(
            f"clip manifest must contain exactly {expected_count} objects"
        )
    return rows


def _record(path: Path, *, rehash: bool = True) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path) if rehash else "",
        "bytes": path.stat().st_size,
    }


def _exclusive(path: Path, content: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise LockboxError(f"refusing to overwrite lockbox output: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _read_study(path: Path) -> dict[str, Any]:
    study = read_json(path, "study manifest")
    if (
        not identity_valid(study)
        or study.get("kind") != "vjepa2_controlled_video_diffusion_study"
    ):
        raise LockboxError("study manifest identity/kind is invalid")
    return study


def _validated_record(record: Any, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise LockboxError(f"{label} record is missing")
    path = _file(str(record.get("path", "")), label)
    if (
        record.get("sha256") != sha256_file(path)
        or record.get("bytes", path.stat().st_size) != path.stat().st_size
    ):
        raise LockboxError(f"{label} path/hash/bytes differ")
    return path


def _episode_digest(episodes: set[str]) -> str:
    return hashlib.sha256(canonical_json(sorted(episodes))).hexdigest()


def _clip_digest(clip_ids: set[str]) -> str:
    return hashlib.sha256(canonical_json(sorted(clip_ids))).hexdigest()


def _validate_manifest_rows(
    path: Path,
    *,
    expected_count: int,
    expected_split: str,
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    rows = _read_rows(path, expected_count=expected_count)
    episodes: set[str] = set()
    clip_ids: set[str] = set()
    for index, row in enumerate(rows):
        start = row.get("start")
        frame_indices = row.get("frame_indices")
        episode = str(
            canonical_episode(
                row.get("episode_dir"), f"clip {index} episode directory"
            )
        )
        if (
            row.get("split") != expected_split
            or isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or row.get("auxiliary_index") != index
            or row.get("sample_size") != clips.SAMPLE_SIZE
            or row.get("chunk_size") != clips.CHUNK_SIZE
            or row.get("action_span") != clips.ACTION_SPAN
            or frame_indices != [start + value for value in clips.FRAME_OFFSETS]
        ):
            raise LockboxError(f"clip {index} violates the canonical 13x5 schema")
        identity = {
            "schema": clips.SCHEMA,
            "split": expected_split,
            "episode_dir": episode,
            "start": start,
            "frame_indices": frame_indices,
        }
        clip_id = hashlib.sha256(canonical_json(identity)).hexdigest()
        if row.get("clip_id") != clip_id:
            raise LockboxError(f"clip {index} identity is not canonical")
        if episode in episodes or clip_id in clip_ids:
            raise LockboxError("lockbox must contain one unique episode per clip")
        episodes.add(episode)
        clip_ids.add(clip_id)
    return rows, episodes, clip_ids


def study_population(study: Mapping[str, Any]) -> dict[str, Any]:
    splits = study.get("inputs", {}).get("splits", {})
    expected = {"train": ("train", 512), "validation": ("val", 64), "test": ("test", 128)}
    result: dict[str, Any] = {}
    for name, (physical_split, count) in expected.items():
        record = splits.get(name, {}).get("clip_manifest")
        path = _validated_record(record, f"original {name} manifest")
        _rows, episodes, clip_ids = _validate_manifest_rows(
            path, expected_count=count, expected_split=physical_split
        )
        result[name] = {
            "manifest": {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "entries": count,
            },
            "episode_count": len(episodes),
            "episode_identity_sha256": _episode_digest(episodes),
            "clip_identity_sha256": _clip_digest(clip_ids),
            "_episodes": episodes,
            "_clips": clip_ids,
        }
    names = tuple(result)
    overlaps: dict[str, int] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            count = len(result[left]["_episodes"] & result[right]["_episodes"])
            overlaps[f"{left}__{right}"] = count
            if count:
                raise LockboxError(f"original {left}/{right} episodes overlap")
    result["pairwise_episode_overlap_counts"] = overlaps
    return result


def _source_population(study: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    record = study.get("inputs", {}).get("cache_build", {}).get("episode_manifest")
    path = _validated_record(record, "source episode manifest")
    episodes = clips._read_episode_manifest(path)
    return path, {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "episode_count": len(episodes),
    }


def _derive_rows(
    study: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original = study_population(study)
    source_path, source_record = _source_population(study)
    excluded = set().union(
        *(original[name]["_episodes"] for name in ("train", "validation", "test"))
    )
    ranked = sorted(
        clips._read_episode_manifest(source_path),
        key=lambda episode: (clips._episode_rank(episode, seed), str(episode)),
    )
    rows: list[dict[str, Any]] = []
    examined = 0
    for episode_path in ranked:
        episode = str(canonical_episode(str(episode_path), "eligible source episode"))
        if episode in excluded:
            continue
        examined += 1
        length = clips.inspect_episode(episode_path)
        starts = clips.deterministic_starts(
            episode_path,
            trajectory_length=length,
            seed=seed,
            limit=1,
        )
        if len(starts) != 1:
            raise LockboxError(f"eligible episode produced no clip: {episode}")
        rows.append(
            clips._clip_row(
                split="test",
                episode=episode_path,
                start=starts[0],
                auxiliary_index=len(rows),
            )
        )
        if len(rows) == LOCKBOX_CLIPS:
            break
    if len(rows) != LOCKBOX_CLIPS:
        raise LockboxError(
            f"only {len(rows)} unused eligible episodes; need {LOCKBOX_CLIPS}"
        )
    selected_episodes = {str(row["episode_dir"]) for row in rows}
    selected_clips = {str(row["clip_id"]) for row in rows}
    public_original = {
        name: {
            key: value
            for key, value in original[name].items()
            if not key.startswith("_")
        }
        for name in ("train", "validation", "test")
    }
    overlap = {
        name: len(selected_episodes & original[name]["_episodes"])
        for name in ("train", "validation", "test")
    }
    if any(overlap.values()):
        raise LockboxError("derived lockbox overlaps an original episode split")
    return rows, {
        "source_episode_manifest": source_record,
        "seed": seed,
        "ranking": (
            "ascending SHA256(abc-vjepa2.1-fixed-clips-v1\\0episode-rank"
            "\\0seed\\0canonical_episode_path), then path"
        ),
        "selection": (
            "exclude all original train/validation/test episodes; take the "
            "next 128 valid ranked episodes; one deterministic clip each"
        ),
        "eligible_unused_episodes_examined": examined,
        "original_population": public_original,
        "original_pairwise_episode_overlap_counts": original[
            "pairwise_episode_overlap_counts"
        ],
        "lockbox_episode_count": len(selected_episodes),
        "lockbox_episode_identity_sha256": _episode_digest(selected_episodes),
        "lockbox_clip_identity_sha256": _clip_digest(selected_clips),
        "lockbox_original_episode_overlap_counts": overlap,
    }


def build_manifest(
    *,
    study_path: Path,
    output_dir: Path,
    seed: int = LOCKBOX_SEED,
    construction_commit: str,
    inference_compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    study_path = _file(study_path, "study manifest")
    study = _read_study(study_path)
    if seed != LOCKBOX_SEED:
        raise LockboxError(f"confirmatory lockbox seed must be {LOCKBOX_SEED}")
    output_dir = output_dir.expanduser()
    if output_dir.exists():
        raise LockboxError(f"fresh lockbox output already exists: {output_dir}")
    parent = _directory(output_dir.parent, "lockbox output parent")
    output_dir = parent / output_dir.name
    output_dir.mkdir(mode=0o700)
    manifest_path = output_dir / "lockbox.jsonl"
    construction_path = output_dir / "lockbox_construction.json"
    rows, audit = _derive_rows(study, seed=seed)
    manifest_bytes = b"".join(canonical_json(row) + b"\n" for row in rows)
    _exclusive(manifest_path, manifest_bytes)
    construction = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_CONSTRUCTION,
            "study": {
                "path": str(study_path),
                "sha256": sha256_file(study_path),
                "bytes": study_path.stat().st_size,
                "identity_sha256": study["identity_sha256"],
            },
            "training_git_commit": study["inputs"]["repository"]["git_commit"],
            "construction_git_commit": construction_commit,
            "inference_code_compatibility": dict(inference_compatibility),
            "physical_cache_split": "test",
            "semantic_evaluation_split": "lockbox",
            "clip_manifest": {
                "path": str(manifest_path),
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "bytes": len(manifest_bytes),
                "entries": LOCKBOX_CLIPS,
            },
            "construction_audit": audit,
        }
    )
    _exclusive(construction_path, canonical_json(construction) + b"\n")
    return construction


def _metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata[key]
    provenance = metadata.get("provenance")
    return provenance.get(key) if isinstance(provenance, Mapping) else None


def _cache_registration(
    *,
    metadata_path: Path,
    construction: Mapping[str, Any],
    study: Mapping[str, Any],
    rehash_arrays: bool,
) -> dict[str, Any]:
    metadata_path = _file(metadata_path, "lockbox cache metadata")
    metadata = read_json(metadata_path, "lockbox cache metadata")
    manifest_record = construction["clip_manifest"]
    train_sha = study["inputs"]["splits"]["train"]["clip_manifest"]["sha256"]
    vjepa = study["inputs"]["vjepa"]
    pca_sha = vjepa["pca_stats"]["sha256"]
    checkpoint_sha = vjepa["checkpoint"]["sha256"]
    source_commit = vjepa["source"]["commit"]
    expected = {
        "format_version": 1,
        "artifact_type": "vjepa2.1-wan-grid-cache",
        "complete": True,
        "split": "test",
        "clip_count": LOCKBOX_CLIPS,
        "clip_manifest_sha256": manifest_record["sha256"],
        "train_manifest_sha256": train_sha,
        "pca_sha256": pca_sha,
        "checkpoint_sha256": checkpoint_sha,
        "source_commit": source_commit,
        "model_name": vjepa["model_name"],
        "checkpoint_bytes": vjepa["checkpoint"]["bytes"],
        "sample_size": 13,
        "chunk_size": 5,
        "action_span": 65,
        "frame_offsets": list(range(0, 65, 5)),
        "camera_order": ["top", "left_wrist", "right_wrist"],
        "target_shape": [LOCKBOX_CLIPS, 64, 4, 24, 120],
        "target_dtype": "float16",
        "rgb_shape": [LOCKBOX_CLIPS, 13, 3, 180, 960],
        "rgb_dtype": "float16",
        "actions_shape": [LOCKBOX_CLIPS, 13, 5, 23],
        "actions_dtype": "float32",
    }
    for key, wanted in expected.items():
        actual = _metadata_value(metadata, key)
        if actual != wanted:
            raise LockboxError(
                f"lockbox cache {key} differs: {actual!r} != {wanted!r}"
            )
    if Path(str(metadata.get("clip_manifest", ""))).resolve() != Path(
        str(manifest_record["path"])
    ):
        raise LockboxError("lockbox cache points to another manifest path")
    cache_id = metadata.get("cache_id")
    if not isinstance(cache_id, str) or SHA256_RE.fullmatch(cache_id) is None:
        raise LockboxError("lockbox cache_id is not a full SHA-256")
    arrays: dict[str, Any] = {}
    for name in ("target", "rgb", "actions"):
        value = metadata.get(f"{name}_file")
        if not isinstance(value, str) or not value:
            raise LockboxError(f"lockbox cache lacks {name}_file")
        path = Path(value)
        if not path.is_absolute():
            path = metadata_path.parent / path
        path = _file(path, f"lockbox {name} array")
        recorded = metadata.get(f"{name}_sha256")
        if (
            not isinstance(recorded, str)
            or SHA256_RE.fullmatch(recorded) is None
            or (rehash_arrays and sha256_file(path) != recorded)
        ):
            raise LockboxError(f"lockbox {name} array SHA-256 differs")
        arrays[name] = {
            "path": str(path),
            "sha256": recorded,
            "bytes": path.stat().st_size,
            "full_sha256_verified": rehash_arrays,
            "shape": expected[f"{name}_shape"],
            "dtype": expected[f"{name}_dtype"],
        }
    return {
        "metadata": {
            "path": str(metadata_path),
            "sha256": sha256_file(metadata_path),
            "bytes": metadata_path.stat().st_size,
        },
        "cache_id": cache_id,
        "clip_count": LOCKBOX_CLIPS,
        "train_manifest_sha256": train_sha,
        "pca_sha256": pca_sha,
        "vjepa_checkpoint_sha256": checkpoint_sha,
        "vjepa_checkpoint_bytes": vjepa["checkpoint"]["bytes"],
        "vjepa_source_commit": source_commit,
        "vjepa_model_name": vjepa["model_name"],
        "arrays": arrays,
    }


def build_registration(
    *,
    construction_path: Path,
    cache_metadata_path: Path,
    study_path: Path,
    registration_commit: str,
    attested_never_scored: bool,
    inference_compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    if not attested_never_scored:
        raise LockboxError(
            "confirmatory registration requires the never-scored attestation"
        )
    construction_path = _file(construction_path, "lockbox construction")
    construction = read_json(construction_path, "lockbox construction")
    study_path = _file(study_path, "study manifest")
    study = _read_study(study_path)
    if (
        not identity_valid(construction)
        or construction.get("kind") != KIND_CONSTRUCTION
        or construction.get("study", {}).get("identity_sha256")
        != study["identity_sha256"]
        or construction.get("training_git_commit")
        != study["inputs"]["repository"]["git_commit"]
        or construction.get("physical_cache_split") != "test"
        or construction.get("semantic_evaluation_split") != "lockbox"
        or construction.get("construction_git_commit") != registration_commit
        or construction.get("inference_code_compatibility")
        != dict(inference_compatibility)
    ):
        raise LockboxError("lockbox construction identity/provenance differs")
    rows, audit = _derive_rows(study, seed=LOCKBOX_SEED)
    manifest_path = _validated_record(
        construction.get("clip_manifest"), "lockbox manifest"
    )
    expected_bytes = b"".join(canonical_json(row) + b"\n" for row in rows)
    if manifest_path.read_bytes() != expected_bytes:
        raise LockboxError("lockbox is not the deterministic next eligible sample")
    if construction.get("construction_audit") != audit:
        raise LockboxError("lockbox construction isolation audit differs")
    cache = _cache_registration(
        metadata_path=cache_metadata_path,
        construction=construction,
        study=study,
        rehash_arrays=True,
    )
    training_commit = study["inputs"]["repository"]["git_commit"]
    if (
        COMMIT_RE.fullmatch(str(registration_commit)) is None
        or COMMIT_RE.fullmatch(str(training_commit)) is None
    ):
        raise LockboxError("registration/training commit is invalid")
    return identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_REGISTRATION,
            "study_identity_sha256": study["identity_sha256"],
            "training_git_commit": training_commit,
            "registration_git_commit": registration_commit,
            "inference_code_compatibility": dict(inference_compatibility),
            "construction": {
                "path": str(construction_path),
                "sha256": sha256_file(construction_path),
                "bytes": construction_path.stat().st_size,
                "identity_sha256": construction["identity_sha256"],
            },
            "manifest": dict(construction["clip_manifest"]),
            "cache": cache,
            "episode_isolation": audit,
            "operator_attestation": {
                "lockbox_never_scored_before_registration": True,
                "text": ATTESTATION_TEXT,
            },
            "selection_must_bind_registration_before_evaluation": True,
        }
    )


def validate_registration(
    payload: Mapping[str, Any],
    *,
    study: Mapping[str, Any],
    rehash_arrays: bool,
    verify_construction: bool = True,
) -> dict[str, Any]:
    if (
        not identity_valid(payload)
        or payload.get("kind") != KIND_REGISTRATION
        or payload.get("study_identity_sha256") != study.get("identity_sha256")
        or payload.get("training_git_commit")
        != study.get("inputs", {}).get("repository", {}).get("git_commit")
        or COMMIT_RE.fullmatch(
            str(payload.get("registration_git_commit", ""))
        )
        is None
        or payload.get("inference_code_compatibility", {}).get(
            "training_commit_is_ancestor"
        )
        is not True
        or payload.get("inference_code_compatibility", {}).get(
            "inference_critical_paths_unchanged"
        )
        is not True
        or payload.get("operator_attestation", {}).get(
            "lockbox_never_scored_before_registration"
        )
        is not True
        or payload.get("operator_attestation", {}).get("text")
        != ATTESTATION_TEXT
        or payload.get("selection_must_bind_registration_before_evaluation")
        is not True
    ):
        raise LockboxError("lockbox registration identity/provenance is invalid")
    construction_path = _validated_record(
        payload.get("construction"), "lockbox construction"
    )
    construction = read_json(construction_path, "lockbox construction")
    if (
        not identity_valid(construction)
        or construction.get("identity_sha256")
        != payload["construction"]["identity_sha256"]
        or construction.get("clip_manifest") != payload.get("manifest")
        or construction.get("construction_git_commit")
        != payload.get("registration_git_commit")
        or construction.get("inference_code_compatibility")
        != payload.get("inference_code_compatibility")
    ):
        raise LockboxError("registered construction identity differs")
    manifest_path = _validated_record(payload.get("manifest"), "lockbox manifest")
    _rows, episodes, clip_ids = _validate_manifest_rows(
        manifest_path,
        expected_count=LOCKBOX_CLIPS,
        expected_split="test",
    )
    audit = payload.get("episode_isolation", {})
    if (
        audit.get("lockbox_episode_count") != LOCKBOX_CLIPS
        or audit.get("lockbox_episode_identity_sha256")
        != _episode_digest(episodes)
        or audit.get("lockbox_clip_identity_sha256") != _clip_digest(clip_ids)
        or any(
            audit.get("lockbox_original_episode_overlap_counts", {}).get(name)
            != 0
            for name in ("train", "validation", "test")
        )
    ):
        raise LockboxError("registered episode-isolation proof differs")
    if verify_construction:
        expected_rows, expected_audit = _derive_rows(
            study, seed=LOCKBOX_SEED
        )
        expected_bytes = b"".join(
            canonical_json(row) + b"\n" for row in expected_rows
        )
        if (
            manifest_path.read_bytes() != expected_bytes
            or audit != expected_audit
            or construction.get("construction_audit") != expected_audit
        ):
            raise LockboxError(
                "lockbox is not the deterministic next unused episode sample"
            )
    rebuilt_cache = _cache_registration(
        metadata_path=_validated_record(
            payload.get("cache", {}).get("metadata"),
            "lockbox cache metadata",
        ),
        construction=construction,
        study=study,
        rehash_arrays=rehash_arrays,
    )
    recorded_cache = payload.get("cache")
    for key in (
        "metadata",
        "cache_id",
        "clip_count",
        "train_manifest_sha256",
        "pca_sha256",
        "vjepa_checkpoint_sha256",
        "vjepa_checkpoint_bytes",
        "vjepa_source_commit",
        "vjepa_model_name",
    ):
        if rebuilt_cache[key] != recorded_cache.get(key):
            raise LockboxError(f"registered lockbox cache {key} differs")
    for name, rebuilt in rebuilt_cache["arrays"].items():
        recorded = recorded_cache.get("arrays", {}).get(name, {})
        for key in ("path", "sha256", "bytes", "shape", "dtype"):
            if rebuilt[key] != recorded.get(key):
                raise LockboxError(f"registered lockbox {name} {key} differs")
    return {
        "identity_sha256": payload["identity_sha256"],
        "manifest_path": str(manifest_path),
        "cache_metadata_path": rebuilt_cache["metadata"]["path"],
        "cache_arrays": rebuilt_cache["arrays"],
        "full_array_hashes_rechecked": rehash_arrays,
        "deterministic_construction_reverified": verify_construction,
        "episode_isolation_verified": True,
    }


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise LockboxError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _assert_clean_commit(repo: Path, commit: str) -> None:
    if COMMIT_RE.fullmatch(commit) is None or _git(repo, "rev-parse", "HEAD") != commit:
        raise LockboxError("repository HEAD differs from registration commit")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise LockboxError("repository must be clean")


def _git_compatibility(
    repo: Path, *, training_commit: str, tool_commit: str
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            training_commit,
            tool_commit,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise LockboxError(
            "registration commit is not a descendant of the training commit"
        )
    paths = {}
    for path in INFERENCE_CRITICAL_PATHS:
        training_object = _git(repo, "rev-parse", f"{training_commit}:{path}")
        tool_object = _git(repo, "rev-parse", f"{tool_commit}:{path}")
        if training_object != tool_object:
            raise LockboxError(
                f"inference-critical code changed after training: {path}"
            )
        paths[path] = {
            "training_object_sha": training_object,
            "tool_object_sha": tool_object,
            "unchanged": True,
        }
    return {
        "training_commit_is_ancestor": True,
        "inference_critical_paths_unchanged": True,
        "paths": paths,
    }


def _command_build(args: argparse.Namespace) -> int:
    repo = _directory(args.repo_root, "repository root")
    _assert_clean_commit(repo, args.registration_commit)
    study_path = _file(args.study_manifest, "study manifest")
    study = _read_study(study_path)
    compatibility = _git_compatibility(
        repo,
        training_commit=study["inputs"]["repository"]["git_commit"],
        tool_commit=args.registration_commit,
    )
    payload = build_manifest(
        study_path=study_path,
        output_dir=Path(args.output_dir),
        construction_commit=args.registration_commit,
        inference_compatibility=compatibility,
    )
    print(json.dumps({"status": "constructed", "identity": payload["identity_sha256"]}))
    return 0


def _command_register(args: argparse.Namespace) -> int:
    repo = _directory(args.repo_root, "repository root")
    _assert_clean_commit(repo, args.registration_commit)
    study_path = _file(args.study_manifest, "study manifest")
    study = _read_study(study_path)
    compatibility = _git_compatibility(
        repo,
        training_commit=study["inputs"]["repository"]["git_commit"],
        tool_commit=args.registration_commit,
    )
    payload = build_registration(
        construction_path=Path(args.construction),
        cache_metadata_path=Path(args.cache_metadata),
        study_path=study_path,
        registration_commit=args.registration_commit,
        attested_never_scored=args.attest_never_scored,
        inference_compatibility=compatibility,
    )
    output = Path(args.output).expanduser()
    _directory(output.parent, "registration output parent")
    _exclusive(output, canonical_json(payload) + b"\n")
    print(json.dumps({"status": "registered", "identity": payload["identity_sha256"]}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--repo-root", required=True)
    build.add_argument("--registration-commit", required=True)
    build.add_argument("--study-manifest", required=True)
    build.add_argument("--output-dir", required=True)
    build.set_defaults(handler=_command_build)
    register = commands.add_parser("register")
    register.add_argument("--repo-root", required=True)
    register.add_argument("--registration-commit", required=True)
    register.add_argument("--study-manifest", required=True)
    register.add_argument("--construction", required=True)
    register.add_argument("--cache-metadata", required=True)
    register.add_argument("--attest-never-scored", action="store_true")
    register.add_argument("--output", required=True)
    register.set_defaults(handler=_command_register)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (LockboxError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
