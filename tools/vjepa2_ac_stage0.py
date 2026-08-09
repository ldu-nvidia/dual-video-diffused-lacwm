#!/usr/bin/env python3
"""Auditable Stage-0 qualification of the official V-JEPA 2-AC predictor.

This tool answers a deliberately narrower question than video integration:
does the released action-conditioned predictor retain sample-specific action
signal under its *native DROID contract*, and can that signal be rolled out
without future RGB or future measured state?

Only frozen DROID ``train`` rows are accepted.  The protected test manifest is
never opened.  A registration is durably written before any RGB is decoded or
model forward is executed.  The official checkpoint is loaded manually from a
caller-supplied, SHA-256-pinned path because the pinned upstream checkout has a
testing-only localhost URL in ``src/hub/backbones.py``.

Clock note: this is a deterministic JEPA predictor, not a diffusion schedule;
there is no sigma clock in this Stage-0 probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA_VERSION = 1
KIND_PREFIX = "vjepa2_ac_native_droid_stage0"
SOURCE_COMMIT = "45d025f636dfc58fc2426905fc4a1ab755b1c3e5"
UPSTREAM_FILE_SHA256 = {
    "ac_predictor": "5a520ad92b3f78b231aa153a5bbd9f800f880024974109a04d19aed6a0900643",
    "hub_backbones": "391cdde1e9a1da47cb8094bbea5fbbe8acac0135b27e82f1a6ab19c0b39cc692",
    "droid_loader": "25f3fe88d6d29af0460155f5e0f2b2d05e005f6201a2090bf351fc3daee1d549",
    "droid_train": "f480c51b80425c88b6cd617e3c5386d3c09c8179810fd5e0e7cb7e5f5ecab61a",
    "droid_config": "25f8d23f80dd24c2c9c410f3690edd966f4e9725809912d525dc925e030eb101",
    "droid_transforms": "777f217aedf2171194dc89f875ec6549fc48dbd5a4e15c617006f5aac35aecf9",
}
OFFICIAL_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
)
OFFICIAL_CHECKPOINT_BYTES = 11_760_743_310
OFFICIAL_CHECKPOINT_SHA256 = (
    "0b5e3c4bf77a473cd8c61d32fbd87b28cdbba043fb3b8267f3b8bcfb1d5b9e6b"
)
OFFICIAL_CHECKPOINT_ETAG = "12c945a28cc4f447e728fd2ba810607c-1402"
OFFICIAL_CHECKPOINT_LAST_MODIFIED = "Fri, 30 May 2025 00:00:12 GMT"
EXPECTED_ENCODER_MISSING_KEYS: frozenset[str] = frozenset()
EXPECTED_ENCODER_UNEXPECTED_KEYS: frozenset[str] = frozenset()

DROID_MANIFEST_SHA256 = (
    "cc10bccece1ac0e20abacf30ee0db60339145ec54ab2e28af977ded21e02f27e"
)
DROID_CAMERA = "exterior_image_1_left"
DROID_NATIVE_FPS = 15
VJEPA_AC_FPS = 4
FRAME_STEP = math.ceil(DROID_NATIVE_FPS / VJEPA_AC_FPS)
DROID_RESIZE_SCALE = (1.777, 1.777)
DROID_RESIZE_ASPECT_RATIO = (0.75, 1.35)
FRAMES_PER_CLIP = 8
FRAME_OFFSETS = tuple(index * FRAME_STEP for index in range(FRAMES_PER_CLIP))
ACTION_STEPS = FRAMES_PER_CLIP - 1
ACTION_DIM = 7
TOKENS_PER_FRAME = 16 * 16
DEFAULT_SEED = 20260809
DEFAULT_BOOTSTRAP_SEED = 20260810
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
PRIMARY_HORIZONS = (1, 2)
CONTROL_NAMES = (
    "aligned",
    "zero",
    "episode_shuffled",
    "time_shifted_plus1",
    "time_shifted_minus1",
    "logged_raw",
)
PRIMARY_CONTROLS = (
    "zero",
    "episode_shuffled",
    "time_shifted_plus1",
    "time_shifted_minus1",
)
GATE_MIN_RELATIVE_PERCENT = 5.0


class QualificationError(RuntimeError):
    """A frozen input, causal contract, or artifact failed closed."""


@dataclass(frozen=True)
class NativeClip:
    """One exact native-DROID clip before tensor/model materialization."""

    clip_id: str
    episode_index: int
    start: int
    trajectory_length: int
    frame_indices: tuple[int, ...]
    parquet: Path
    video: Path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def identity_sha256(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("identity_sha256", None)
    return sha256_bytes(canonical_json(body).encode("utf-8"))


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["identity_sha256"] = identity_sha256(result)
    return result


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_record(path: str | Path, *, hash_payload: bool = True) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise QualificationError(f"required file is missing: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
    }
    if hash_payload:
        result["sha256"] = sha256_file(resolved)
    return result


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QualificationError(f"cannot identify source checkout {repo}: {exc}") from exc


def git_status(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QualificationError(f"cannot audit source checkout {repo}: {exc}") from exc


def validate_source(source_dir: Path) -> dict[str, Any]:
    source = source_dir.expanduser().resolve()
    if git_commit(source) != SOURCE_COMMIT:
        raise QualificationError("official V-JEPA source commit differs")
    if git_status(source):
        raise QualificationError("official V-JEPA source checkout is dirty")
    required = {
        "ac_predictor": source / "src/models/ac_predictor.py",
        "hub_backbones": source / "src/hub/backbones.py",
        "droid_loader": source / "app/vjepa_droid/droid.py",
        "droid_train": source / "app/vjepa_droid/train.py",
        "droid_config": source / "configs/train/vitg16/droid-256px-8f.yaml",
        "droid_transforms": source / "app/vjepa_droid/transforms.py",
    }
    records = {name: file_record(path) for name, path in required.items()}
    for name, record in records.items():
        if record["sha256"] != UPSTREAM_FILE_SHA256[name]:
            raise QualificationError(f"pinned upstream file digest changed: {name}")
    hub_text = required["hub_backbones"].read_text()
    if 'VJEPA_BASE_URL = "http://localhost:8300"' not in hub_text:
        raise QualificationError("pinned upstream localhost-loader audit changed")
    train_text = required["droid_train"].read_text()
    if "action_embed_dim=7" not in train_text or "auto_steps" not in train_text:
        raise QualificationError("pinned DROID action/training contract changed")
    return {
        "root": str(source),
        "commit": SOURCE_COMMIT,
        "clean": True,
        "files": records,
        "manual_checkpoint_loading_required": True,
        "reason": (
            "pinned src/hub/backbones.py activates a localhost testing URL; "
            "pretrained=False plus strict manual state loading is required"
        ),
    }


def validate_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    checkpoint = path.expanduser().resolve()
    if len(expected_sha256) != 64:
        raise QualificationError("expected checkpoint SHA-256 must contain 64 hex characters")
    try:
        int(expected_sha256, 16)
    except ValueError as exc:
        raise QualificationError("expected checkpoint SHA-256 is not hexadecimal") from exc
    if expected_sha256.lower() != OFFICIAL_CHECKPOINT_SHA256:
        raise QualificationError(
            "checkpoint SHA-256 argument differs from the preregistered official digest"
        )
    record = file_record(checkpoint)
    if record["bytes"] != OFFICIAL_CHECKPOINT_BYTES:
        raise QualificationError(
            f"checkpoint bytes {record['bytes']} != {OFFICIAL_CHECKPOINT_BYTES}"
        )
    if record["sha256"] != expected_sha256.lower():
        raise QualificationError("checkpoint SHA-256 differs from the pinned argument")
    return {
        **record,
        "official_url": OFFICIAL_CHECKPOINT_URL,
        "http_head": {
            "content_length": OFFICIAL_CHECKPOINT_BYTES,
            "etag": OFFICIAL_CHECKPOINT_ETAG,
            "last_modified": OFFICIAL_CHECKPOINT_LAST_MODIFIED,
        },
    }


def lerobot_state8_to_vjepa_pose7(state: np.ndarray) -> np.ndarray:
    """Convert the observed LeRobot payload to official AC pose order.

    The published LeRobot metadata labels columns 3:7 as ``rx,ry,rz,rw``.
    Direct payload inspection contradicts that label: columns 3:6 are Euler
    angles (values wrap near +/-pi), column 6 is constant zero padding, and
    column 7 is the varying gripper coordinate.  The existing LACWM loader's
    comment states the same layout but accidentally reads column 6 as gripper.
    This adapter is explicit so neither metadata ambiguity nor that loader bug
    silently changes the official ``[xyz,euler_xyz,gripper]`` contract.
    """
    array = np.asarray(state, dtype=np.float32)
    if array.ndim < 1 or array.shape[-1] != 8:
        raise ValueError(f"expected trailing DROID state dimension 8, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("DROID state contains non-finite values")
    if np.max(np.abs(array[..., 6])) > 1e-6:
        raise ValueError("DROID state padding column 6 is unexpectedly nonzero")
    gripper = array[..., 7:8]
    if np.min(gripper) < -1e-5 or np.max(gripper) > 1.00001:
        raise ValueError("DROID state gripper column is outside [0,1]")
    return np.concatenate((array[..., :6], gripper), axis=-1)


def poses_to_diffs(poses: np.ndarray) -> np.ndarray:
    """Exact official DROID pose-difference convention for ``[T,7]``."""
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM or len(values) < 2:
        raise ValueError(f"expected [T>=2,{ACTION_DIM}] poses, got {values.shape}")
    xyz_diff = values[1:, :3] - values[:-1, :3]
    rotations = Rotation.from_euler("xyz", values[:, 3:6], degrees=False).as_matrix()
    relative = rotations[1:] @ np.swapaxes(rotations[:-1], 1, 2)
    angle_diff = Rotation.from_matrix(relative).as_euler("xyz", degrees=False)
    gripper_diff = values[1:, 6:7] - values[:-1, 6:7]
    return np.concatenate((xyz_diff, angle_diff, gripper_diff), axis=1).astype(np.float32)


def integrate_pose(pose: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Apply the official notebook's left-multiplied Cartesian action."""
    p = np.asarray(pose, dtype=np.float64)
    a = np.asarray(action, dtype=np.float64)
    if p.shape != (ACTION_DIM,) or a.shape != (ACTION_DIM,):
        raise ValueError("pose and action must both have shape [7]")
    xyz = p[:3] + a[:3]
    old_rotation = Rotation.from_euler("xyz", p[3:6], degrees=False).as_matrix()
    delta_rotation = Rotation.from_euler("xyz", a[3:6], degrees=False).as_matrix()
    angles = Rotation.from_matrix(delta_rotation @ old_rotation).as_euler(
        "xyz", degrees=False
    )
    gripper = np.clip(p[6:7] + a[6:7], 0.0, 1.0)
    return np.concatenate((xyz, angles, gripper)).astype(np.float32)


def integrate_actions(initial_pose: np.ndarray, actions: np.ndarray) -> np.ndarray:
    controls = np.asarray(actions, dtype=np.float32)
    if controls.ndim != 2 or controls.shape[1] != ACTION_DIM:
        raise ValueError(f"expected [T,{ACTION_DIM}] actions, got {controls.shape}")
    trajectory = [np.asarray(initial_pose, dtype=np.float32)]
    for action in controls:
        trajectory.append(integrate_pose(trajectory[-1], action))
    return np.stack(trajectory)


def nonwrapping_shift(actions: np.ndarray, offset: int) -> np.ndarray:
    """Shift controls without borrowing a cyclic endpoint from the same clip."""
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM:
        raise ValueError(f"expected [T,{ACTION_DIM}] actions, got {values.shape}")
    if offset == 0 or abs(offset) >= len(values):
        raise ValueError("shift offset must be nonzero and smaller than the sequence")
    result = np.zeros_like(values)
    if offset > 0:
        result[:-offset] = values[offset:]
    else:
        result[-offset:] = values[:offset]
    return result


def build_action_controls(
    aligned: np.ndarray,
    donor: np.ndarray,
    logged_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    aligned = np.asarray(aligned, dtype=np.float32)
    donor = np.asarray(donor, dtype=np.float32)
    logged_raw = np.asarray(logged_raw, dtype=np.float32)
    expected = (ACTION_STEPS, ACTION_DIM)
    for name, values in (
        ("aligned", aligned),
        ("donor", donor),
        ("logged_raw", logged_raw),
    ):
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"{name} controls must be finite {expected}, got {values.shape}")
    return {
        "aligned": aligned.copy(),
        "zero": np.zeros_like(aligned),
        "episode_shuffled": donor.copy(),
        "time_shifted_plus1": nonwrapping_shift(aligned, 1),
        "time_shifted_minus1": nonwrapping_shift(aligned, -1),
        "logged_raw": logged_raw.copy(),
    }


def metric_dict(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("prediction/target shapes must be equal and nonempty")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("prediction/target contains non-finite values")
    error = pred - truth
    mse = float(np.mean(error**2))
    target_ms = float(np.mean(truth**2))
    flat_pred = pred.reshape(-1, pred.shape[-1])
    flat_truth = truth.reshape(-1, truth.shape[-1])
    denominator = np.linalg.norm(flat_pred, axis=1) * np.linalg.norm(flat_truth, axis=1)
    cosine = np.divide(
        np.sum(flat_pred * flat_truth, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )
    return {
        "l1": float(np.mean(np.abs(error))),
        "mse": mse,
        "nmse": float(mse / max(target_ms, 1e-12)),
        "token_cosine": float(np.mean(cosine)),
    }


def official_fallback_crop(height: int, width: int) -> tuple[int, int, int, int]:
    """The upstream deterministic fallback for the released DROID config.

    Scale 1.777 makes every random proposal larger than a 180x320 frame, so
    upstream ``_get_param_spatial_crop`` exhausts its attempts and uses this
    central, aspect-ratio-capped crop.
    """
    if height <= 0 or width <= 0:
        raise ValueError("crop dimensions must be positive")
    input_ratio = float(width) / float(height)
    if input_ratio < min(DROID_RESIZE_ASPECT_RATIO):
        crop_width = width
        crop_height = int(round(crop_width / min(DROID_RESIZE_ASPECT_RATIO)))
    elif input_ratio > max(DROID_RESIZE_ASPECT_RATIO):
        crop_height = height
        crop_width = int(round(crop_height * max(DROID_RESIZE_ASPECT_RATIO)))
    else:
        crop_width, crop_height = width, height
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return top, left, crop_height, crop_width


def paired_bootstrap_relative_improvement(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.ndim != 1 or left.shape != right.shape or len(left) < 2:
        raise ValueError("paired vectors must be equal one-dimensional arrays of length >=2")
    if samples <= 0 or np.any(right <= 0):
        raise ValueError("bootstrap samples and reference losses must be positive")
    point = 100.0 * (1.0 - float(np.mean(left)) / float(np.mean(right)))
    generator = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for offset in range(0, samples, 1000):
        count = min(1000, samples - offset)
        indices = generator.integers(0, len(left), size=(count, len(left)))
        candidate_mean = left[indices].mean(axis=1)
        reference_mean = right[indices].mean(axis=1)
        values[offset : offset + count] = 100.0 * (
            1.0 - candidate_mean / reference_mean
        )
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "point_percent": point,
        "ci95_low_percent": float(low),
        "ci95_high_percent": float(high),
    }


def _episode_rank(episode_index: int, seed: int) -> str:
    return sha256_bytes(
        f"{KIND_PREFIX}:cohort:{seed}:{int(episode_index)}".encode("utf-8")
    )


def _clip_rank(clip_id: str, seed: int) -> str:
    return sha256_bytes(f"{KIND_PREFIX}:clip:{seed}:{clip_id}".encode("utf-8"))


def droid_paths(data_root: Path, episode_index: int) -> tuple[Path, Path]:
    chunk = int(episode_index) // 1000
    parquet = (
        data_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    )
    video = (
        data_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / f"observation.images.{DROID_CAMERA}"
        / f"episode_{episode_index:06d}.mp4"
    )
    return parquet, video


def select_native_clips(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
    count: int,
    seed: int,
) -> list[NativeClip]:
    if count <= 1:
        raise QualificationError("native cohort count must exceed one")
    candidates: dict[int, list[Mapping[str, Any]]] = {}
    seen_clip_ids: set[str] = set()
    for index, row in enumerate(manifest_rows):
        if row.get("split") != "train" or row.get("protected") is not False:
            raise QualificationError(f"manifest row {index} is not an unprotected train row")
        if row.get("camera") != DROID_CAMERA:
            raise QualificationError(f"manifest row {index} camera differs")
        clip_id = str(row.get("clip_id", ""))
        if len(clip_id) != 64 or clip_id in seen_clip_ids:
            raise QualificationError("manifest clip IDs are malformed or duplicated")
        seen_clip_ids.add(clip_id)
        episode = int(row["episode_index"])
        start = int(row["start"])
        trajectory_length = int(row["trajectory_length"])
        if start < 0 or start + FRAME_OFFSETS[-1] >= trajectory_length:
            continue
        candidates.setdefault(episode, []).append(row)
    if len(candidates) < count:
        raise QualificationError(
            f"only {len(candidates)} episodes support the native clip, need {count}"
        )
    selected_episodes = sorted(candidates, key=lambda item: (_episode_rank(item, seed), item))[
        :count
    ]
    result: list[NativeClip] = []
    for episode in selected_episodes:
        row = min(
            candidates[episode],
            key=lambda item: (_clip_rank(str(item["clip_id"]), seed), str(item["clip_id"])),
        )
        start = int(row["start"])
        indices = tuple(start + offset for offset in FRAME_OFFSETS)
        parquet, video = droid_paths(data_root, episode)
        if not parquet.is_file() or not video.is_file():
            raise QualificationError(f"DROID payload is missing for episode {episode}")
        result.append(
            NativeClip(
                clip_id=str(row["clip_id"]),
                episode_index=episode,
                start=start,
                trajectory_length=int(row["trajectory_length"]),
                frame_indices=indices,
                parquet=parquet.resolve(),
                video=video.resolve(),
            )
        )
    if len({item.episode_index for item in result}) != count:
        raise QualificationError("native cohort is not episode-disjoint")
    return result


def _clip_payload(clip: NativeClip) -> dict[str, Any]:
    return {
        "clip_id": clip.clip_id,
        "episode_index": clip.episode_index,
        "start": clip.start,
        "trajectory_length": clip.trajectory_length,
        "frame_indices": list(clip.frame_indices),
        "parquet": str(clip.parquet),
        "video": str(clip.video),
    }


def build_registration(args: argparse.Namespace) -> tuple[dict[str, Any], list[NativeClip]]:
    source = validate_source(args.source_dir)
    checkpoint = validate_checkpoint(args.checkpoint, args.expected_checkpoint_sha256)
    manifest_record = file_record(args.train_manifest)
    if manifest_record["sha256"] != DROID_MANIFEST_SHA256:
        raise QualificationError("DROID train manifest SHA-256 differs")
    rows = read_jsonl(args.train_manifest)
    cohort = select_native_clips(
        rows,
        data_root=args.data_root.expanduser().resolve(),
        count=args.clip_count,
        seed=args.seed,
    )
    info_path = args.data_root.expanduser().resolve() / "meta/info.json"
    info = read_json(info_path)
    features = info.get("features", {})
    state_names = features.get("observation.state", {}).get("names")
    if info.get("fps") != DROID_NATIVE_FPS or state_names is None:
        raise QualificationError("DROID metadata FPS/state schema changed")
    camera_metadata = features.get(f"observation.images.{DROID_CAMERA}", {})
    camera_video_metadata = camera_metadata.get("info", {})
    if (
        camera_metadata.get("shape") != [180, 320, 3]
        or camera_video_metadata.get("video.codec") != "av1"
        or camera_video_metadata.get("video.fps") != float(DROID_NATIVE_FPS)
    ):
        raise QualificationError("DROID camera geometry/codec metadata changed")
    implementation = Path(__file__).resolve()
    implementation_repo = implementation.parents[1]
    if git_status(implementation_repo):
        raise QualificationError("qualification implementation repository is dirty")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{KIND_PREFIX}_registration",
        "status": "registered_before_rgb_decode_or_model_forward",
        "created_at_utc": now_utc(),
        "protected_test_accessed": False,
        "source": source,
        "checkpoint": checkpoint,
        "implementation": {
            **file_record(implementation),
            "repo_commit": git_commit(implementation_repo),
            "repo_clean": True,
        },
        "data": {
            "root": str(args.data_root.expanduser().resolve()),
            "train_manifest": manifest_record,
            "metadata": file_record(info_path),
            "metadata_state_names": state_names,
            "camera_metadata": camera_metadata,
            "decoder": {
                "backend": "PyAV",
                "output_pixel_format": "rgb24",
                "selection": "sequential display-order frame index",
                "reason": (
                    "the immutable LeRobot payload is AV1 and the installed "
                    "Decord 0.6 build cannot open its video stream"
                ),
            },
            "observed_payload_adapter": (
                "state[:3]=xyz,state[3:6]=Euler,state[6]=zero padding,"
                "state[7]=gripper; metadata quaternion-style names are rejected as literal"
            ),
            "protected_test_manifest_opened": False,
        },
        "cohort": {
            "seed": args.seed,
            "count": len(cohort),
            "episode_disjoint": True,
            "rows": [_clip_payload(item) for item in cohort],
            "identity_sha256": sha256_bytes(
                canonical_json({"rows": [_clip_payload(item) for item in cohort]}).encode()
            ),
        },
        "protocol": {
            "question": (
                "does the official AC checkpoint provide action-specific future-video "
                "embeddings under its native DROID contract, including an inference-causal rollout?"
            ),
            "camera": DROID_CAMERA,
            "native_fps": DROID_NATIVE_FPS,
            "checkpoint_fps": VJEPA_AC_FPS,
            "frame_step": FRAME_STEP,
            "frame_offsets": list(FRAME_OFFSETS),
            "frames_per_clip": FRAMES_PER_CLIP,
            "official_spatial_transform": {
                "random_resize_scale": list(DROID_RESIZE_SCALE),
                "random_resize_aspect_ratio": list(DROID_RESIZE_ASPECT_RATIO),
                "observed_180x320_fallback_crop_tlhw": list(
                    official_fallback_crop(180, 320)
                ),
                "output_hw": [256, 256],
            },
            "state": "[xyz,euler_xyz,gripper], measured only at initial time for causal rollout",
            "aligned_action": (
                "official measured-pose difference; future measured poses are target-only "
                "outside the nondeployable teacher-forced positive control"
            ),
            "controls": list(CONTROL_NAMES),
            "episode_shuffled_donor": "next cohort row cyclically; always a distinct episode",
            "time_shift": "one native AC step, zero-filled without wraparound",
            "logged_raw": "DROID logged 7-D action diagnostic; not assumed to equal pose delta",
            "teacher_forced": {
                "purpose": "native checkpoint positive control",
                "future_measured_state_used": True,
                "future_target_embeddings_used_as_context": True,
                "deployable": False,
            },
            "causal_autoregressive": {
                "initial_rgb_embedding": True,
                "initial_measured_state": True,
                "future_state": "integrated from the same candidate action sequence",
                "future_rgb_or_embedding_input": False,
                "deployable": True,
            },
            "primary_horizons": list(PRIMARY_HORIZONS),
            "reason": "official config trains auto_steps=2; longer rollout is extrapolation",
            "maximum_horizon": args.max_horizon,
            "metrics": ["l1", "mse", "nmse", "token_cosine"],
            "gate": {
                "loss": "l1",
                "minimum_relative_improvement_percent": GATE_MIN_RELATIVE_PERCENT,
                "paired_bootstrap_ci95_low_strictly_positive": True,
                "required_controls": list(PRIMARY_CONTROLS),
                "required_modes": ["teacher_forced", "causal_autoregressive"],
                "required_horizons": list(PRIMARY_HORIZONS),
                "native_gate_is_necessary_not_sufficient_for_abc_or_wan": True,
            },
            "post_native_gates_before_abc": [
                "pretrained AC must beat persistence, a raw-action baseline, and a train-from-scratch matched predictor",
                "ABC joint commands must be converted by audited FK into the same Cartesian action/state contract",
                "aligned ABC actions must beat zero, episode-shuffled, and time-shifted controls",
                "predictor and total auxiliary latency must fit the rollout budget",
            ],
            "wan_integration_allowed": False,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    return seal(payload), cohort


def validate_registration(registration: Mapping[str, Any]) -> None:
    if registration.get("identity_sha256") != identity_sha256(registration):
        raise QualificationError("registration identity digest differs")
    if registration.get("kind") != f"{KIND_PREFIX}_registration":
        raise QualificationError("registration kind differs")
    if registration.get("protected_test_accessed") is not False:
        raise QualificationError("registration protected-test flag is not false")
    for section, key in (
        ("checkpoint", "checkpoint"),
        ("implementation", "implementation"),
    ):
        record = registration[section]
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise QualificationError(f"registered {key} payload changed")


def _decode_rgb_frames_pyav(
    path: Path,
    indices: Sequence[int],
    *,
    expected_frame_count: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    try:
        import av
    except ImportError as exc:
        raise QualificationError("PyAV is required to decode immutable AV1 RGB") from exc
    requested = tuple(int(index) for index in indices)
    if (
        len(requested) == 0
        or len(set(requested)) != len(requested)
        or tuple(sorted(requested)) != requested
        or requested[0] < 0
        or requested[-1] >= expected_frame_count
    ):
        raise QualificationError("requested RGB frame indices are invalid")
    positions = {frame_index: offset for offset, frame_index in enumerate(requested)}
    selected: list[np.ndarray | None] = [None] * len(requested)
    try:
        with av.open(str(path), mode="r") as container:
            if len(container.streams.video) != 1:
                raise QualificationError("native DROID MP4 must have exactly one video stream")
            stream = container.streams.video[0]
            if stream.average_rate is None:
                raise QualificationError("native DROID MP4 has no average frame rate")
            fps = float(stream.average_rate)
            codec_name = str(stream.codec_context.name)
            declared_frames = int(stream.frames or 0)
            decoded_count = 0
            for frame in container.decode(stream):
                if decoded_count in positions:
                    selected[positions[decoded_count]] = frame.to_ndarray(format="rgb24")
                decoded_count += 1
    except QualificationError:
        raise
    except Exception as exc:
        raise QualificationError(f"cannot decode native DROID AV1 video {path}: {exc}") from exc
    if decoded_count != expected_frame_count:
        raise QualificationError(
            f"decoded video length {decoded_count} != {expected_frame_count}"
        )
    if any(frame is None for frame in selected):
        raise QualificationError("one or more requested RGB frames were not decoded")
    frames = np.stack(selected)
    return frames, fps, {
        "backend": "PyAV",
        "backend_version": str(av.__version__),
        "codec": codec_name,
        "output_pixel_format": "rgb24",
        "declared_frame_count": declared_frames,
        "decoded_frame_count": decoded_count,
        "selected_frame_indices": list(requested),
    }


def _load_native_arrays(clip: NativeClip) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise QualificationError("pandas and pyarrow are required") from exc
    dataframe = pd.read_parquet(
        clip.parquet, columns=["observation.state", "action"]
    )
    state8 = np.stack(dataframe["observation.state"].to_numpy()).astype(np.float32)
    logged = np.stack(dataframe["action"].to_numpy()).astype(np.float32)
    if len(state8) != clip.trajectory_length or len(logged) != clip.trajectory_length:
        raise QualificationError("parquet trajectory length differs from the manifest")
    indices = np.asarray(clip.frame_indices, dtype=np.int64)
    pose7 = lerobot_state8_to_vjepa_pose7(state8[indices])
    aligned = poses_to_diffs(pose7)
    logged_selected = logged[indices[:-1]]
    frames, video_fps, decoder_audit = _decode_rgb_frames_pyav(
        clip.video,
        clip.frame_indices,
        expected_frame_count=clip.trajectory_length,
    )
    if abs(video_fps - DROID_NATIVE_FPS) > 1e-6:
        raise QualificationError("video FPS differs from the frozen contract")
    expected_rgb_shape = (FRAMES_PER_CLIP, 180, 320, 3)
    if frames.shape != expected_rgb_shape:
        raise QualificationError(
            "decoded native RGB shape differs from the 180x320 geometry on which "
            f"the registered official fallback crop depends: {frames.shape} != "
            f"{expected_rgb_shape}"
        )
    return {
        "frames": frames,
        "poses": pose7,
        "aligned_actions": aligned,
        "logged_actions": logged_selected,
        "video_fps": video_fps,
        "decoder_audit": decoder_audit,
        "state_schema_audit": {
            "trajectory_rows": int(len(state8)),
            "padding_col6_abs_max": float(np.max(np.abs(state8[:, 6]))),
            "gripper_col7_min": float(np.min(state8[:, 7])),
            "gripper_col7_max": float(np.max(state8[:, 7])),
            "euler_cols3_5_min": state8[:, 3:6].min(axis=0).tolist(),
            "euler_cols3_5_max": state8[:, 3:6].max(axis=0).tolist(),
            "metadata_quaternion_interpretation_valid": False,
            "reason": (
                "column 6 is identically zero while columns 3:6 contain angle-scale "
                "values including +/-pi; state7 uses columns 0:6 plus gripper column 7"
            ),
        },
    }


def _strip_checkpoint_keys(state: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("module.", "backbone."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        result[name] = value
    return result


def _load_model(
    source_dir: Path,
    checkpoint_path: Path,
    *,
    device: str,
    dtype_name: str,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    source = str(source_dir.expanduser().resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from app.vjepa_droid.transforms import make_transforms
    from src.hub.backbones import vjepa2_ac_vit_giant

    # Preserve the released hub's default num_frames=64.  This is causal-mask
    # capacity (32 tubelet groups), not the sampled clip length.  Setting it to
    # 8 would allocate only four groups and cannot represent the seven
    # teacher-forced context groups used by the official 8-frame DROID recipe.
    encoder, predictor = vjepa2_ac_vit_giant(pretrained=False)
    required_mask_tokens = ACTION_STEPS * (TOKENS_PER_FRAME + 2)
    if (
        getattr(predictor, "attn_mask", None) is None
        or predictor.attn_mask.shape[0] < required_mask_tokens
        or predictor.attn_mask.shape[1] < required_mask_tokens
    ):
        raise QualificationError("released predictor causal mask cannot cover seven frame groups")
    load_start = time.perf_counter()
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True, mmap=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or not {
        "encoder",
        "predictor",
    }.issubset(checkpoint):
        raise QualificationError("official checkpoint lacks encoder/predictor state")
    encoder_message = encoder.load_state_dict(
        _strip_checkpoint_keys(checkpoint["encoder"]), strict=False
    )
    predictor_message = predictor.load_state_dict(
        _strip_checkpoint_keys(checkpoint["predictor"]), strict=True
    )
    del checkpoint
    if predictor_message.missing_keys or predictor_message.unexpected_keys:
        raise QualificationError("strict predictor checkpoint load was incomplete")
    encoder_missing = set(encoder_message.missing_keys)
    encoder_unexpected = set(encoder_message.unexpected_keys)
    if encoder_missing != EXPECTED_ENCODER_MISSING_KEYS:
        raise QualificationError(
            "encoder checkpoint missing-key contract changed: "
            f"observed={sorted(encoder_missing)}, "
            f"expected={sorted(EXPECTED_ENCODER_MISSING_KEYS)}"
        )
    if encoder_unexpected != EXPECTED_ENCODER_UNEXPECTED_KEYS:
        raise QualificationError(
            "encoder checkpoint unexpected-key contract changed: "
            f"observed={sorted(encoder_unexpected)}, "
            f"expected={sorted(EXPECTED_ENCODER_UNEXPECTED_KEYS)}"
        )
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise QualificationError("production qualification requires CUDA")
    # Match app/vjepa_droid/train.py: parameters remain FP32 while the forward
    # executes under BF16 autocast.  Casting parameter storage itself to BF16
    # changes residual/normalization numerics and is not the released recipe.
    encoder = encoder.eval().to(device=torch_device, dtype=torch.float32)
    predictor = predictor.eval().to(device=torch_device, dtype=torch.float32)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=DROID_RESIZE_ASPECT_RATIO,
        random_resize_scale=DROID_RESIZE_SCALE,
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=256,
    )
    torch.cuda.synchronize(torch_device)
    device_index = (
        torch_device.index
        if torch_device.index is not None
        else torch.cuda.current_device()
    )
    device_properties = torch.cuda.get_device_properties(device_index)
    return encoder, predictor, transform, {
        "device": str(torch_device),
        "gpu_name": str(device_properties.name),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(device_index)),
        "gpu_total_memory_bytes": int(device_properties.total_memory),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "parameter_storage_dtype": "float32",
        "forward_autocast_dtype": dtype_name,
        "load_seconds": time.perf_counter() - load_start,
        "encoder_missing_keys": list(encoder_message.missing_keys),
        "encoder_unexpected_keys": list(encoder_message.unexpected_keys),
        "expected_encoder_missing_keys": sorted(EXPECTED_ENCODER_MISSING_KEYS),
        "expected_encoder_unexpected_keys": sorted(
            EXPECTED_ENCODER_UNEXPECTED_KEYS
        ),
        "predictor_strict": True,
        "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "predictor_parameters": sum(parameter.numel() for parameter in predictor.parameters()),
        "hub_num_frames": 64,
        "causal_mask_shape": list(predictor.attn_mask.shape),
    }


def _encode_frames(
    frames: np.ndarray,
    *,
    encoder: Any,
    transform: Any,
    device: str,
    dtype_name: str,
) -> tuple[Any, float]:
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(0)
    clip = transform(frames).unsqueeze(0)
    batch, channels, temporal, height, width = clip.shape
    if (batch, temporal) != (1, FRAMES_PER_CLIP):
        raise QualificationError("official transform changed temporal geometry")
    model_input = (
        clip.permute(0, 2, 1, 3, 4)
        .flatten(0, 1)
        .unsqueeze(2)
        .repeat(1, 1, 2, 1, 1)
    )
    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]
    model_input = model_input.to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=compute_dtype,
        enabled=dtype_name == "bfloat16",
    ):
        encoded = encoder(model_input)
        encoded = encoded.view(1, FRAMES_PER_CLIP, -1, encoded.size(-1))
        encoded = functional.layer_norm(encoded, (encoded.size(-1),))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if encoded.shape[2] != TOKENS_PER_FRAME:
        raise QualificationError(f"encoder token geometry differs: {tuple(encoded.shape)}")
    return encoded, elapsed


def _predict_teacher_forced(
    encoded: Any,
    controls: Mapping[str, np.ndarray],
    measured_poses: np.ndarray,
    *,
    predictor: Any,
    device: str,
    dtype_name: str,
) -> tuple[dict[str, Any], float]:
    import torch
    import torch.nn.functional as functional

    names = list(CONTROL_NAMES)
    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]
    context = encoded[:, :-1].flatten(1, 2).repeat(len(names), 1, 1)
    action_tensor = torch.from_numpy(np.stack([controls[name] for name in names])).to(
        device=device, dtype=torch.float32
    )
    state_tensor = torch.from_numpy(
        np.repeat(measured_poses[None, :-1], len(names), axis=0)
    ).to(device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=compute_dtype,
        enabled=dtype_name == "bfloat16",
    ):
        predicted = predictor(context, action_tensor, state_tensor)
        predicted = functional.layer_norm(predicted, (predicted.size(-1),))
        predicted = predicted.view(
            len(names), ACTION_STEPS, TOKENS_PER_FRAME, predicted.size(-1)
        )
    torch.cuda.synchronize()
    return {name: predicted[index] for index, name in enumerate(names)}, time.perf_counter() - started


def _predict_causal_autoregressive(
    encoded: Any,
    controls: Mapping[str, np.ndarray],
    initial_pose: np.ndarray,
    *,
    predictor: Any,
    device: str,
    dtype_name: str,
    max_horizon: int,
) -> tuple[dict[str, Any], list[float]]:
    import torch
    import torch.nn.functional as functional

    names = list(CONTROL_NAMES)
    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]
    action_arrays = np.stack([controls[name] for name in names])
    state_arrays = np.stack(
        [integrate_actions(initial_pose, controls[name])[:-1] for name in names]
    )
    context = encoded[:, :1].flatten(1, 2).repeat(len(names), 1, 1)
    outputs: list[Any] = []
    per_call_seconds: list[float] = []
    for step in range(max_horizon):
        action_tensor = torch.from_numpy(action_arrays[:, : step + 1]).to(
            device=device, dtype=torch.float32
        )
        state_tensor = torch.from_numpy(state_arrays[:, : step + 1]).to(
            device=device, dtype=torch.float32
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=compute_dtype,
            enabled=dtype_name == "bfloat16",
        ):
            predicted = predictor(context, action_tensor, state_tensor)
            next_tokens = predicted[:, -TOKENS_PER_FRAME:]
            next_tokens = functional.layer_norm(next_tokens, (next_tokens.size(-1),))
        torch.cuda.synchronize()
        per_call_seconds.append(time.perf_counter() - started)
        outputs.append(next_tokens)
        context = torch.cat((context, next_tokens), dim=1)
    stacked = torch.stack(outputs, dim=1)
    return {name: stacked[index] for index, name in enumerate(names)}, per_call_seconds


def _tensor_numpy(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _evaluate_clip(
    clip_index: int,
    clip: NativeClip,
    payload: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    encoder: Any,
    predictor: Any,
    transform: Any,
    device: str,
    dtype_name: str,
    max_horizon: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aligned = np.asarray(payload["aligned_actions"])
    controls = build_action_controls(
        aligned,
        np.asarray(donor["aligned_actions"]),
        np.asarray(payload["logged_actions"]),
    )
    poses = np.asarray(payload["poses"])
    encoded, encoder_seconds = _encode_frames(
        np.asarray(payload["frames"]),
        encoder=encoder,
        transform=transform,
        device=device,
        dtype_name=dtype_name,
    )
    teacher, teacher_seconds = _predict_teacher_forced(
        encoded,
        controls,
        poses,
        predictor=predictor,
        device=device,
        dtype_name=dtype_name,
    )
    causal, causal_seconds = _predict_causal_autoregressive(
        encoded,
        controls,
        poses[0],
        predictor=predictor,
        device=device,
        dtype_name=dtype_name,
        max_horizon=max_horizon,
    )
    target = _tensor_numpy(encoded[0, 1 : max_horizon + 1])
    persistence = np.repeat(_tensor_numpy(encoded[0, :1]), max_horizon, axis=0)
    rows: list[dict[str, Any]] = []
    for horizon in range(1, max_horizon + 1):
        for mode, predictions in (
            ("teacher_forced", teacher),
            ("causal_autoregressive", causal),
        ):
            for control in CONTROL_NAMES:
                prediction = _tensor_numpy(predictions[control][:horizon])
                rows.append(
                    {
                        "clip_index": clip_index,
                        "clip_id": clip.clip_id,
                        "episode_index": clip.episode_index,
                        "donor_episode_index": int(donor["episode_index"]),
                        "mode": mode,
                        "control": control,
                        "horizon": horizon,
                        "metrics": metric_dict(prediction, target[:horizon]),
                        "protected_test_accessed": False,
                    }
                )
        rows.append(
            {
                "clip_index": clip_index,
                "clip_id": clip.clip_id,
                "episode_index": clip.episode_index,
                "donor_episode_index": None,
                "mode": "causal_autoregressive",
                "control": "persistence",
                "horizon": horizon,
                "metrics": metric_dict(persistence[:horizon], target[:horizon]),
                "protected_test_accessed": False,
            }
        )
    integration_error = float(
        np.max(np.abs(integrate_actions(poses[0], aligned) - poses))
    )
    timing = {
        "clip_index": clip_index,
        "encoder_seconds": encoder_seconds,
        "teacher_forced_batch_seconds": teacher_seconds,
        "causal_batch_call_seconds": causal_seconds,
        "causal_batch_total_seconds": float(sum(causal_seconds)),
        "control_batch_size": len(CONTROL_NAMES),
        "aligned_pose_reintegration_max_abs": integration_error,
        "state_schema_audit": payload["state_schema_audit"],
        "decoder_audit": payload["decoder_audit"],
    }
    return rows, timing


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    timings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    indexed: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        indexed.setdefault(
            (str(row["mode"]), str(row["control"]), int(row["horizon"])), []
        ).append(row)
    aggregates: dict[str, Any] = {}
    for key, values in sorted(indexed.items()):
        mode, control, horizon = key
        aggregate_key = f"{mode}/{control}/h{horizon}"
        aggregates[aggregate_key] = {
            metric: float(np.mean([item["metrics"][metric] for item in values]))
            for metric in ("l1", "mse", "nmse", "token_cosine")
        }
    comparisons: dict[str, Any] = {}
    gates: list[bool] = []
    for mode in ("teacher_forced", "causal_autoregressive"):
        for horizon in PRIMARY_HORIZONS:
            aligned_rows = indexed[(mode, "aligned", horizon)]
            aligned = [item["metrics"]["l1"] for item in aligned_rows]
            for control in PRIMARY_CONTROLS:
                reference_rows = indexed[(mode, control, horizon)]
                reference = [item["metrics"]["l1"] for item in reference_rows]
                effect = paired_bootstrap_relative_improvement(aligned, reference)
                passed = (
                    effect["point_percent"] >= GATE_MIN_RELATIVE_PERCENT
                    and effect["ci95_low_percent"] > 0.0
                )
                effect["passed"] = passed
                comparisons[f"{mode}/aligned_vs_{control}/h{horizon}"] = effect
                gates.append(passed)
    latency = {
        "encoder_mean_ms": 1000.0 * float(np.mean([row["encoder_seconds"] for row in timings])),
        "encoder_p95_ms": 1000.0 * float(np.quantile([row["encoder_seconds"] for row in timings], 0.95)),
        "teacher_forced_control_batch_mean_ms": 1000.0
        * float(np.mean([row["teacher_forced_batch_seconds"] for row in timings])),
        "teacher_forced_control_batch_amortized_per_condition_mean_ms": 1000.0
        * float(np.mean([row["teacher_forced_batch_seconds"] for row in timings]))
        / len(CONTROL_NAMES),
        "causal_control_batch_total_mean_ms": 1000.0
        * float(np.mean([row["causal_batch_total_seconds"] for row in timings])),
        "causal_control_batch_total_amortized_per_condition_mean_ms": 1000.0
        * float(np.mean([row["causal_batch_total_seconds"] for row in timings]))
        / len(CONTROL_NAMES),
        "causal_per_call_mean_ms": 1000.0
        * float(
            np.mean(
                [
                    value
                    for row in timings
                    for value in row["causal_batch_call_seconds"]
                ]
            )
        ),
        "causal_per_call_amortized_per_condition_mean_ms": 1000.0
        * float(
            np.mean(
                [
                    value
                    for row in timings
                    for value in row["causal_batch_call_seconds"]
                ]
            )
        )
        / len(CONTROL_NAMES),
        "control_batch_size": len(CONTROL_NAMES),
    }
    max_reintegration_error = max(
        float(row["aligned_pose_reintegration_max_abs"]) for row in timings
    )
    schema_audit = {
        "sampled_episode_count": len(timings),
        "padding_col6_abs_max": max(
            float(row["state_schema_audit"]["padding_col6_abs_max"])
            for row in timings
        ),
        "gripper_col7_global_min": min(
            float(row["state_schema_audit"]["gripper_col7_min"])
            for row in timings
        ),
        "gripper_col7_global_max": max(
            float(row["state_schema_audit"]["gripper_col7_max"])
            for row in timings
        ),
        "metadata_quaternion_interpretation_valid": False,
    }
    decoder_audits = [row["decoder_audit"] for row in timings]
    decoder_audit = {
        "backend": "PyAV",
        "backend_versions": sorted(
            {str(row["backend_version"]) for row in decoder_audits}
        ),
        "codecs": sorted({str(row["codec"]) for row in decoder_audits}),
        "output_pixel_formats": sorted(
            {str(row["output_pixel_format"]) for row in decoder_audits}
        ),
        "all_declared_counts_match_decoded": all(
            int(row["declared_frame_count"]) == int(row["decoded_frame_count"])
            for row in decoder_audits
        ),
    }
    return {
        "aggregates": aggregates,
        "aligned_comparisons": comparisons,
        "native_action_gate_passed": bool(gates and all(gates)),
        "decision": (
            "NATIVE_ACTION_GATE_PASS"
            if gates and all(gates)
            else "STOP_NATIVE_ACTION_GATE_FAIL"
        ),
        "latency": latency,
        "aligned_pose_reintegration_max_abs": max_reintegration_error,
        "sampled_parquet_state_schema_audit": schema_audit,
        "sampled_video_decoder_audit": decoder_audit,
        "claim_boundary": (
            "a native DROID pass is necessary but does not qualify ABC, improve video, "
            "or authorize Wan integration; raw/scratch and ABC causal gates remain"
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    registration, cohort = build_registration(args)
    registration_path = output / "registration.json"
    write_json(registration_path, registration)
    validate_registration(registration)

    encoder, predictor, transform, model_load = _load_model(
        args.source_dir,
        args.checkpoint,
        device=args.device,
        dtype_name=args.dtype,
    )
    payloads: list[dict[str, Any]] = []
    for clip in cohort:
        loaded = dict(_load_native_arrays(clip))
        loaded["episode_index"] = clip.episode_index
        payloads.append(loaded)
    warmup_started = time.perf_counter()
    _evaluate_clip(
        -1,
        cohort[0],
        payloads[0],
        payloads[1],
        encoder=encoder,
        predictor=predictor,
        transform=transform,
        device=args.device,
        dtype_name=args.dtype,
        max_horizon=args.max_horizon,
    )
    import torch

    torch.cuda.synchronize(torch.device(args.device))
    warmup_seconds = time.perf_counter() - warmup_started
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    print("completed one excluded full-shape latency warmup", flush=True)
    rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for index, clip in enumerate(cohort):
        donor_index = (index + 1) % len(cohort)
        if cohort[donor_index].episode_index == clip.episode_index:
            raise QualificationError("episode-shuffled donor is not distinct")
        clip_rows, timing = _evaluate_clip(
            index,
            clip,
            payloads[index],
            payloads[donor_index],
            encoder=encoder,
            predictor=predictor,
            transform=transform,
            device=args.device,
            dtype_name=args.dtype,
            max_horizon=args.max_horizon,
        )
        rows.extend(clip_rows)
        timings.append(timing)
        print(f"evaluated {index + 1}/{len(cohort)} native DROID clips", flush=True)
    memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(args.device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(args.device)),
    }
    per_clip_path = output / "per_clip_metrics.jsonl"
    timings_path = output / "timings.jsonl"
    write_jsonl(per_clip_path, rows)
    write_jsonl(timings_path, timings)
    summary = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND_PREFIX}_summary",
            "created_at_utc": now_utc(),
            "registration_identity_sha256": registration["identity_sha256"],
            "clip_count": len(cohort),
            "row_count": len(rows),
            "model": model_load,
            "warmup": {
                "full_shape_excluded_from_latency_and_quality": True,
                "seconds": warmup_seconds,
            },
            "memory": memory,
            **_summarize(rows, timings),
            "protected_test_accessed": False,
            "wan_integration_authorized": False,
        }
    )
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    complete = seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": f"{KIND_PREFIX}_complete",
            "status": "completed",
            "created_at_utc": now_utc(),
            "registration_identity_sha256": registration["identity_sha256"],
            "summary_identity_sha256": summary["identity_sha256"],
            "decision": summary["decision"],
            "artifacts": {
                "registration": file_record(registration_path),
                "per_clip_metrics": file_record(per_clip_path),
                "timings": file_record(timings_path),
                "summary": file_record(summary_path),
            },
            "protected_test_accessed": False,
            "wan_integration_authorized": False,
        }
    )
    write_json(output / "run_complete.json", complete)
    return complete


def audit(output_dir: Path) -> dict[str, Any]:
    root = output_dir.expanduser().resolve()
    complete = read_json(root / "run_complete.json")
    if complete.get("identity_sha256") != identity_sha256(complete):
        raise QualificationError("completion identity differs")
    if complete.get("kind") != f"{KIND_PREFIX}_complete":
        raise QualificationError("completion kind differs")
    if complete.get("protected_test_accessed") is not False:
        raise QualificationError("completion protected-test flag differs")
    for name, record in complete.get("artifacts", {}).items():
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise QualificationError(f"completion artifact changed: {name}")
    registration = read_json(root / "registration.json")
    summary = read_json(root / "summary.json")
    validate_registration(registration)
    if summary.get("identity_sha256") != identity_sha256(summary):
        raise QualificationError("summary identity differs")
    if summary.get("registration_identity_sha256") != registration.get("identity_sha256"):
        raise QualificationError("summary/registration identity differs")
    rows = read_jsonl(root / "per_clip_metrics.jsonl")
    if len(rows) != int(summary["row_count"]):
        raise QualificationError("per-clip row count differs")
    if any(row.get("protected_test_accessed") is not False for row in rows):
        raise QualificationError("per-clip protected-test flag differs")
    return {
        "status": "passed",
        "decision": complete["decision"],
        "clip_count": summary["clip_count"],
        "row_count": len(rows),
        "registration_identity_sha256": registration["identity_sha256"],
        "summary_identity_sha256": summary["identity_sha256"],
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("evaluate")
    run.add_argument("--source-dir", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--expected-checkpoint-sha256", required=True)
    run.add_argument("--train-manifest", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--clip-count", type=int, default=32)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--max-horizon", type=int, choices=range(2, 8), default=2)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    check = commands.add_parser("audit")
    check.add_argument("--output-dir", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = evaluate(args) if args.command == "evaluate" else audit(args.output_dir)
    except (QualificationError, ValueError, OSError, KeyError) as exc:
        print(f"V-JEPA 2-AC qualification error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
