#!/usr/bin/env python3
"""Leakage-safe nominal YAM/D405 geometry probe for ABC train clips.

The probe has two deliberately separate stages:

``extract``
    Reads only explicitly selected rows from a train clip manifest, the matching
    preprocessed episode, and the matching raw MCAP calibration.  It writes a
    compact numeric bundle so rendering can happen on a workstation without
    copying a full episode or raw MCAP.

``evaluate``
    Loads train-only bundles, constructs a robot-only MuJoCo scene from the
    pinned official ``amazon-far/abc`` MJCF, renders the nominal D405 top view,
    and compares rendered silhouette edges against observed RGB edges.  A
    cyclically time-shifted pose is the negative control.

This is a geometry/calibration diagnostic, not a video-generation evaluation.
It never needs future held-out video, and it refuses validation/test bundles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


EXPECTED_ABC_COMMIT = "6bc6586721cf0c409ccee80f675a28de9b9b2f5e"
OFFICIAL_SCENE_RELATIVE = Path("assets/put_bottles/put_bottle.xml")
OFFICIAL_ASSET_RELATIVE = Path("assets/put_bottles/assets")
CAMERA_TYPE = "Intel RealSense D405"
GRIPPER_QPOS_MAX = 0.0475

# LACWM cache/state order is [left arm 6, right arm 6, left grip, right grip].
# Official ABC sim/policy order is [left arm 6, left grip, right arm 6, right grip].
CACHE14_TO_OFFICIAL14 = (0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cache14_to_official14(actions: np.ndarray) -> np.ndarray:
    """Permute LACWM's 14-D ABC action/state layout into official sim order."""

    array = np.asarray(actions)
    if array.shape[-1] != 14:
        raise ValueError(f"Expected trailing action dimension 14, got {array.shape}")
    return array[..., CACHE14_TO_OFFICIAL14]


def require_train_row(row: dict[str, Any]) -> None:
    split = row.get("split")
    if split != "train":
        raise ValueError(f"Protected split isolation: expected split='train', got {split!r}")
    episode_dir = str(row.get("episode_dir", ""))
    if not episode_dir:
        raise ValueError("Manifest row has no episode_dir")
    indices = row.get("frame_indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError("Manifest row has no non-empty frame_indices list")
    if any(not isinstance(index, int) or index < 0 for index in indices):
        raise ValueError("frame_indices must be non-negative integers")


def select_train_rows(
    manifest: Path,
    clip_ids: set[str],
    max_clips: int,
) -> list[dict[str, Any]]:
    if max_clips <= 0:
        raise ValueError("max_clips must be positive")
    selected: list[dict[str, Any]] = []
    with manifest.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                require_train_row(row)
            except ValueError as exc:
                raise ValueError(f"{manifest}:{line_number}: {exc}") from exc
            if clip_ids and row.get("clip_id") not in clip_ids:
                continue
            selected.append(row)
            if len(selected) >= max_clips and not clip_ids:
                break
    if clip_ids:
        observed = {str(row["clip_id"]) for row in selected}
        missing = sorted(clip_ids - observed)
        if missing:
            raise ValueError(f"Requested clip IDs are absent from train manifest: {missing}")
        selected = sorted(selected, key=lambda row: str(row["clip_id"]))
    if not selected:
        raise ValueError("No train rows selected")
    if len(selected) > max_clips:
        raise ValueError(f"Selected {len(selected)} clips, exceeding --max-clips={max_clips}")
    return selected


def resolve_raw_mcap(
    episode_dir: Path,
    preprocessed_root: Path,
    raw_root: Path,
) -> Path:
    episode = episode_dir.resolve()
    preprocessed = preprocessed_root.resolve()
    try:
        relative = episode.relative_to(preprocessed)
    except ValueError as exc:
        raise ValueError(f"Episode {episode} is outside preprocessed root {preprocessed}") from exc
    raw = raw_root.resolve() / relative / "episode.mcap"
    if not raw.is_file():
        raise FileNotFoundError(raw)
    return raw


def _decode_top_calibration(raw_mcap: Path) -> dict[str, Any]:
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError("extract requires mcap and mcap-protobuf-support") from exc

    metadata: dict[str, str] | None = None
    topics: set[str]
    with raw_mcap.open("rb") as handle:
        reader = make_reader(handle)
        summary = reader.get_summary()
        if summary is None:
            raise ValueError(f"MCAP has no summary: {raw_mcap}")
        topics = {channel.topic for channel in summary.channels.values()}
    with raw_mcap.open("rb") as handle:
        records = list(make_reader(handle).iter_metadata())
        if records:
            metadata = dict(records[0].metadata)
    if not metadata:
        raise ValueError(f"MCAP has no metadata record: {raw_mcap}")
    if metadata.get("top_camera_type") != CAMERA_TYPE:
        raise ValueError(
            f"Expected {CAMERA_TYPE!r}, got {metadata.get('top_camera_type')!r}: {raw_mcap}"
        )
    required_topics = {"/top-camera", "/top-camera-info"}
    if not required_topics.issubset(topics):
        raise ValueError(f"D405 MCAP lacks {sorted(required_topics - topics)}: {raw_mcap}")

    calibration = None
    with raw_mcap.open("rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _, channel, _, decoded in reader.iter_decoded_messages(
            topics={"/top-camera-info"}
        ):
            calibration = decoded
            break
    if calibration is None:
        raise ValueError(f"No decoded /top-camera-info message: {raw_mcap}")
    if len(calibration.K) != 9:
        raise ValueError(f"Expected 3x3 K, got {len(calibration.K)} values")
    return {
        "camera_type": metadata["top_camera_type"],
        "camera_width": int(calibration.width),
        "camera_height": int(calibration.height),
        "distortion_model": str(calibration.distortion_model),
        "frame_id": str(calibration.frame_id),
        "K": [float(value) for value in calibration.K],
        "D": [float(value) for value in calibration.D],
        "R": [float(value) for value in calibration.R],
        "P": [float(value) for value in calibration.P],
        "topics": sorted(topics),
        "episode_metadata": metadata,
    }


def _read_video_frames(video: Path, frame_indices: Sequence[int]) -> np.ndarray:
    requested = [int(index) for index in frame_indices]
    if requested != sorted(requested) or len(requested) != len(set(requested)):
        raise ValueError("frame_indices must be strictly increasing and unique")
    try:
        import av
    except ImportError:
        av = None
    if av is not None:
        targets = set(requested)
        decoded: dict[int, np.ndarray] = {}
        with av.open(str(video)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in targets:
                    decoded[index] = frame.to_ndarray(format="rgb24")
                if len(decoded) == len(targets):
                    break
        missing = [index for index in requested if index not in decoded]
        if missing:
            raise RuntimeError(f"Could not decode frames {missing} from {video} with PyAV")
        return np.stack([decoded[index] for index in requested], axis=0)

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError("extract requires opencv-python-headless") from exc

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frames = []
    try:
        for index in requested:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index} from {video}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return np.stack(frames, axis=0)


def _bundle_name(clip_id: str) -> str:
    if not clip_id or any(character not in "0123456789abcdef" for character in clip_id.lower()):
        raise ValueError(f"Unsafe/non-hex clip_id: {clip_id!r}")
    return clip_id


def extract_bundles(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.clip_manifest.resolve()
    rows = select_train_rows(manifest, set(args.clip_id), args.max_clips)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool_path = Path(__file__).resolve()
    tool_sha256 = sha256_file(tool_path)
    tool_commit = git_commit(tool_path.parents[1])
    artifacts = []
    for row in rows:
        clip_id = _bundle_name(str(row["clip_id"]))
        episode_dir = Path(row["episode_dir"]).resolve()
        raw_mcap = resolve_raw_mcap(episode_dir, args.preprocessed_root, args.raw_root)
        calibration = _decode_top_calibration(raw_mcap)
        state_path = episode_dir / "states.npz"
        video_path = episode_dir / "top.mp4"
        if not state_path.is_file() or not video_path.is_file():
            raise FileNotFoundError(f"Incomplete preprocessed episode: {episode_dir}")

        frame_indices = np.asarray(row["frame_indices"], dtype=np.int64)
        with np.load(state_path, allow_pickle=False) as state:
            max_index = int(frame_indices.max())
            if max_index >= len(state["joint_states"]):
                raise IndexError(f"Frame {max_index} exceeds states length in {state_path}")
            arrays = {
                "frame_indices": frame_indices,
                "rgb": _read_video_frames(video_path, frame_indices.tolist()),
                "joint_states": state["joint_states"][frame_indices].astype(np.float32),
                "joint_actions": state["joint_actions"][frame_indices].astype(np.float32),
                "gripper_states": state["gripper_states"][frame_indices].astype(np.float32),
                "gripper_actions": state["gripper_actions"][frame_indices].astype(np.float32),
                "frame_ts": state["frame_ts"][frame_indices].astype(np.int64),
                "K": np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3),
                "D": np.asarray(calibration["D"], dtype=np.float64),
            }
        height, width = arrays["rgb"].shape[1:3]
        if (width, height) != (
            calibration["camera_width"],
            calibration["camera_height"],
        ):
            raise ValueError(
                "Decoded top video and calibration resolution differ: "
                f"video={(width, height)} calibration="
                f"{(calibration['camera_width'], calibration['camera_height'])}"
            )

        npz_path = args.output_dir / f"{clip_id}.npz"
        np.savez_compressed(npz_path, **arrays)
        metadata = {
            "artifact_type": "abc-d405-nominal-geometry-bundle",
            "format_version": 1,
            "tool_sha256": tool_sha256,
            "tool_git_commit": tool_commit,
            "split": "train",
            "protected_test_accessed": False,
            "clip_id": clip_id,
            "clip_manifest": str(manifest),
            "clip_manifest_sha256": sha256_file(manifest),
            "episode_dir": str(episode_dir),
            "raw_mcap": str(raw_mcap),
            "frame_indices": frame_indices.tolist(),
            "camera": calibration,
            "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
            "npz": str(npz_path),
            "npz_sha256": sha256_file(npz_path),
            "source_files": {
                "states_npz_sha256": sha256_file(state_path),
                "top_mp4_sha256": sha256_file(video_path),
                "raw_mcap_sha256": sha256_file(raw_mcap),
            },
        }
        metadata["identity_sha256"] = sha256_json(metadata)
        metadata_path = args.output_dir / f"{clip_id}.json"
        atomic_json(metadata_path, metadata)
        artifacts.append(
            {
                "clip_id": clip_id,
                "npz": str(npz_path),
                "metadata": str(metadata_path),
                "identity_sha256": metadata["identity_sha256"],
            }
        )

    result = {
        "artifact_type": "abc-d405-nominal-geometry-extraction",
        "format_version": 1,
        "tool_sha256": tool_sha256,
        "tool_git_commit": tool_commit,
        "split": "train",
        "protected_test_accessed": False,
        "clip_count": len(artifacts),
        "artifacts": artifacts,
    }
    result["identity_sha256"] = sha256_json(result)
    atomic_json(args.output_dir / "extraction.json", result)
    return result


def build_robot_only_xml(scene_xml: Path, asset_root: Path) -> str:
    """Remove task objects/background while retaining official robot/cameras."""

    root = ET.parse(scene_xml).getroot()
    asset = root.find("asset")
    world = root.find("worldbody")
    compiler = root.find("compiler")
    if asset is None or world is None or compiler is None:
        raise ValueError("Official scene lacks compiler, asset, or worldbody")

    for child in list(asset):
        file_name = child.get("file", "")
        keep_mesh = child.tag == "mesh" and file_name.startswith("i2rt_yam/")
        keep_material = child.tag == "material" and child.get("name") in {"black", "white"}
        if not keep_mesh and not keep_material:
            asset.remove(child)

    keep_bodies = {"gate_collision", "left_arm", "right_arm"}
    for child in list(world):
        keep_light = child.tag == "light"
        keep_body = child.tag == "body" and child.get("name") in keep_bodies
        if not keep_light and not keep_body:
            world.remove(child)

    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    compiler.set("meshdir", str(asset_root.resolve()))
    compiler.attrib.pop("texturedir", None)
    return ET.tostring(root, encoding="unicode")


def validate_bundle_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("artifact_type") != "abc-d405-nominal-geometry-bundle":
        raise ValueError("Unsupported bundle artifact_type")
    if metadata.get("split") != "train":
        raise ValueError("Protected split isolation: only train bundles may be evaluated")
    if metadata.get("protected_test_accessed") is not False:
        raise ValueError("Bundle does not prove protected_test_accessed=false")
    camera = metadata.get("camera", {})
    if camera.get("camera_type") != CAMERA_TYPE:
        raise ValueError(f"Only {CAMERA_TYPE} bundles are supported")


@dataclass(frozen=True)
class LoadedBundle:
    metadata_path: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def load_bundles(bundle_dir: Path) -> list[LoadedBundle]:
    bundles = []
    for metadata_path in sorted(bundle_dir.glob("*.json")):
        if metadata_path.name == "extraction.json":
            continue
        metadata = json.loads(metadata_path.read_text())
        validate_bundle_metadata(metadata)
        npz_path = bundle_dir / f"{metadata['clip_id']}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        if sha256_file(npz_path) != metadata.get("npz_sha256"):
            raise ValueError(f"Bundle hash mismatch: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as payload:
            arrays = {name: payload[name].copy() for name in payload.files}
        required = {
            "rgb",
            "joint_states",
            "gripper_states",
            "K",
            "frame_indices",
            "frame_ts",
        }
        missing = required - arrays.keys()
        if missing:
            raise ValueError(f"Bundle {npz_path} lacks arrays: {sorted(missing)}")
        if arrays["rgb"].shape[0] != arrays["joint_states"].shape[0]:
            raise ValueError(f"RGB/state length mismatch in {npz_path}")
        bundles.append(LoadedBundle(metadata_path, metadata, arrays))
    if not bundles:
        raise ValueError(f"No bundle metadata found under {bundle_dir}")
    return bundles


def _joint_qpos_addresses(model: Any, mujoco: Any) -> dict[str, int]:
    names = [
        *(f"left_joint{index}" for index in range(1, 7)),
        "left_left_finger",
        "left_right_finger",
        *(f"right_joint{index}" for index in range(1, 7)),
        "right_left_finger",
        "right_right_finger",
    ]
    addresses = {}
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Official model is missing joint {name}")
        addresses[name] = int(model.jnt_qposadr[joint_id])
    return addresses


def set_observed_pose(
    model: Any,
    data: Any,
    mujoco: Any,
    addresses: dict[str, int],
    joint_state: np.ndarray,
    gripper_state: np.ndarray,
) -> None:
    joint_state = np.asarray(joint_state, dtype=np.float64)
    gripper_state = np.asarray(gripper_state, dtype=np.float64)
    if joint_state.shape != (12,) or gripper_state.shape != (2,):
        raise ValueError(
            f"Expected joint_state (12,) and gripper_state (2,), got "
            f"{joint_state.shape} and {gripper_state.shape}"
        )
    for offset in range(6):
        data.qpos[addresses[f"left_joint{offset + 1}"]] = joint_state[offset]
        data.qpos[addresses[f"right_joint{offset + 1}"]] = joint_state[offset + 6]
    left = float(np.clip(gripper_state[0], 0.0, 1.0) * GRIPPER_QPOS_MAX)
    right = float(np.clip(gripper_state[1], 0.0, 1.0) * GRIPPER_QPOS_MAX)
    data.qpos[addresses["left_left_finger"]] = left
    data.qpos[addresses["left_right_finger"]] = -left
    data.qpos[addresses["right_left_finger"]] = right
    data.qpos[addresses["right_right_finger"]] = -right
    mujoco.mj_forward(model, data)


def silhouette_boundary(mask: np.ndarray) -> np.ndarray:
    import cv2

    uint8 = np.asarray(mask, dtype=np.uint8) * 255
    return cv2.morphologyEx(uint8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def observed_edges(rgb: np.ndarray) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    median = float(np.median(blurred))
    lower = int(max(20, 0.66 * median))
    upper = int(max(lower + 20, min(240, 1.33 * median)))
    return cv2.Canny(blurred, lower, upper) > 0


def edge_alignment_metrics(render_mask: np.ndarray, image_edges: np.ndarray) -> dict[str, float]:
    import cv2

    boundary = silhouette_boundary(render_mask)
    count = int(boundary.sum())
    if count == 0:
        raise ValueError("Rendered robot silhouette has an empty boundary")
    inverse_edges = (~np.asarray(image_edges, dtype=bool)).astype(np.uint8) * 255
    distance = cv2.distanceTransform(inverse_edges, cv2.DIST_L2, 3)
    values = distance[boundary]
    return {
        "chamfer_px": float(values.mean()),
        "edge_support_3px": float(np.mean(values <= 3.0)),
        "edge_support_5px": float(np.mean(values <= 5.0)),
        "render_boundary_pixels": float(count),
        "render_mask_fraction": float(np.mean(render_mask)),
        "observed_edge_fraction": float(np.mean(image_edges)),
    }


def paired_bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    seed: int = 20260808,
    samples: int = 20_000,
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("differences must be a non-empty vector")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    boot = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _overlay(rgb: np.ndarray, mask: np.ndarray, boundary: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8).copy()
    tint = np.zeros_like(image)
    tint[...] = np.asarray(color, dtype=np.uint8)
    alpha = 0.18
    image[mask] = np.clip(
        (1.0 - alpha) * image[mask].astype(np.float32) + alpha * tint[mask].astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    image[boundary] = np.asarray(color, dtype=np.uint8)
    return image


def _write_overlay(
    path: Path,
    rgb: np.ndarray,
    aligned: np.ndarray,
    shifted: np.ndarray,
    label: str,
) -> None:
    import cv2

    aligned_image = _overlay(rgb, aligned, silhouette_boundary(aligned), (30, 230, 80))
    shifted_image = _overlay(rgb, shifted, silhouette_boundary(shifted), (235, 50, 180))
    cv2.putText(
        aligned_image,
        f"aligned nominal | {label}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 230, 80),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        shifted_image,
        "time-shifted control",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 50, 180),
        2,
        cv2.LINE_AA,
    )
    combined = np.concatenate([aligned_image, shifted_image], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write overlay: {path}")


def evaluate_bundles(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    try:
        import cv2  # noqa: F401
        import mujoco
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError("evaluate requires mujoco and opencv-python-headless") from exc

    official_root = args.official_abc_root.resolve()
    official_commit = git_commit(official_root)
    if official_commit != EXPECTED_ABC_COMMIT and not args.allow_unpinned_official:
        raise ValueError(
            f"Official ABC checkout must be {EXPECTED_ABC_COMMIT}, got {official_commit!r}"
        )
    scene_path = official_root / OFFICIAL_SCENE_RELATIVE
    asset_root = official_root / OFFICIAL_ASSET_RELATIVE
    if not scene_path.is_file() or not asset_root.is_dir():
        raise FileNotFoundError(f"Incomplete official ABC assets under {official_root}")

    bundles = load_bundles(args.bundle_dir.resolve())
    if args.pose_shift <= 0:
        raise ValueError("--pose-shift must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    robot_xml = build_robot_only_xml(scene_path, asset_root)
    xml_sha256 = hashlib.sha256(robot_xml.encode("utf-8")).hexdigest()
    model = mujoco.MjModel.from_xml_string(robot_xml)
    data = mujoco.MjData(model)
    addresses = _joint_qpos_addresses(model, mujoco)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    if camera_id < 0:
        raise ValueError("Official model has no top camera")

    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[2] = 1  # official visual geometry; hide collision geometry

    rows: list[dict[str, Any]] = []
    latency_ms: list[float] = []
    max_pose_overlay_paths: list[str] = []
    failure_overlay_paths: list[str] = []
    calibrations: list[dict[str, Any]] = []
    control_timings: list[dict[str, Any]] = []
    renderer = None
    renderer_shape = None
    try:
        for bundle in bundles:
            arrays = bundle.arrays
            rgb = arrays["rgb"]
            joint = arrays["joint_states"]
            gripper = arrays["gripper_states"]
            count, height, width, channels = rgb.shape
            if channels != 3 or count <= args.pose_shift:
                raise ValueError(f"Unexpected RGB shape or too-short clip: {rgb.shape}")
            if renderer is None:
                renderer = mujoco.Renderer(model, height=height, width=width)
                renderer.enable_segmentation_rendering()
                renderer_shape = (height, width)
            elif renderer_shape != (height, width):
                raise ValueError("All bundles must share one resolution")

            fy = float(arrays["K"][1, 1])
            model.cam_fovy[camera_id] = math.degrees(2.0 * math.atan(height / (2.0 * fy)))

            aligned_masks = []
            shifted_masks = []
            shifted_indices = [int((index + args.pose_shift) % count) for index in range(count)]
            nonwrap_count = count - args.pose_shift
            frame_shift = (
                arrays["frame_indices"][args.pose_shift:]
                - arrays["frame_indices"][:nonwrap_count]
            )
            timestamp_shift_ns = (
                arrays["frame_ts"][args.pose_shift:]
                - arrays["frame_ts"][:nonwrap_count]
            )
            control_timings.append(
                {
                    "clip_id": bundle.metadata["clip_id"],
                    "median_source_video_frame_shift": float(np.median(frame_shift)),
                    "median_timestamp_shift_seconds": float(
                        np.median(timestamp_shift_ns) / 1_000_000_000.0
                    ),
                    "cyclic_wrap_frame_count": args.pose_shift,
                    "nonwrap_frame_count": nonwrap_count,
                }
            )
            for index in range(count):
                for pose_index, target in (
                    (index, aligned_masks),
                    (shifted_indices[index], shifted_masks),
                ):
                    started = time.perf_counter()
                    set_observed_pose(
                        model,
                        data,
                        mujoco,
                        addresses,
                        joint[pose_index],
                        gripper[pose_index],
                    )
                    renderer.update_scene(data, camera="top", scene_option=option)
                    segmentation = renderer.render().copy()
                    latency_ms.append((time.perf_counter() - started) * 1000.0)
                    mask = (
                        (segmentation[..., 0] >= 0)
                        & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                    )
                    target.append(mask)

            calibrations.append(
                {
                    "clip_id": bundle.metadata["clip_id"],
                    "recorded_K": arrays["K"].tolist(),
                    "recorded_D": arrays["D"].tolist(),
                    "render_resolution_wh": [width, height],
                    "render_fovy_degrees": float(model.cam_fovy[camera_id]),
                    "nominal_camera_world_position": data.cam_xpos[camera_id].tolist(),
                    "nominal_camera_to_world_rotation": data.cam_xmat[camera_id]
                    .reshape(3, 3)
                    .tolist(),
                    "projection_note": (
                        "Recorded fy sets render fovy; recorded fx/cx/cy and distortion "
                        "are retained for audit but not applied by this feasibility probe."
                    ),
                }
            )

            frame_pose_distances = np.linalg.norm(joint - joint[shifted_indices], axis=1)
            overlay_index = int(np.argmax(frame_pose_distances))
            overlay_path = args.output_dir / f"overlay_{bundle.metadata['clip_id'][:12]}.png"
            _write_overlay(
                overlay_path,
                rgb[overlay_index],
                aligned_masks[overlay_index],
                shifted_masks[overlay_index],
                f"frame={int(arrays['frame_indices'][overlay_index])}",
            )
            max_pose_overlay_paths.append(str(overlay_path))

            bundle_rows = []
            for index in range(count):
                edges = observed_edges(rgb[index])
                aligned_metrics = edge_alignment_metrics(aligned_masks[index], edges)
                shifted_metrics = edge_alignment_metrics(shifted_masks[index], edges)
                row = {
                    "clip_id": bundle.metadata["clip_id"],
                    "bundle_identity_sha256": bundle.metadata["identity_sha256"],
                    "frame_ordinal": index,
                    "frame_index": int(arrays["frame_indices"][index]),
                    "shifted_frame_ordinal": shifted_indices[index],
                    "pose_l2": float(frame_pose_distances[index]),
                    "aligned": aligned_metrics,
                    "shifted": shifted_metrics,
                }
                rows.append(row)
                bundle_rows.append(row)

            worst_index = int(
                np.argmax([row["aligned"]["chamfer_px"] for row in bundle_rows])
            )
            failure_overlay_path = (
                args.output_dir / f"failure_worst_{bundle.metadata['clip_id'][:12]}.png"
            )
            _write_overlay(
                failure_overlay_path,
                rgb[worst_index],
                aligned_masks[worst_index],
                shifted_masks[worst_index],
                (
                    f"worst aligned frame={int(arrays['frame_indices'][worst_index])} "
                    f"chamfer={bundle_rows[worst_index]['aligned']['chamfer_px']:.1f}px"
                ),
            )
            failure_overlay_paths.append(str(failure_overlay_path))
    finally:
        if renderer is not None:
            renderer.close()

    row_path = args.output_dir / "rows.jsonl"
    with row_path.open("w") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")

    aligned_chamfer = np.asarray([row["aligned"]["chamfer_px"] for row in rows])
    shifted_chamfer = np.asarray([row["shifted"]["chamfer_px"] for row in rows])
    aligned_support = np.asarray([row["aligned"]["edge_support_3px"] for row in rows])
    shifted_support = np.asarray([row["shifted"]["edge_support_3px"] for row in rows])
    delta, lower, upper = paired_bootstrap_mean_ci(shifted_chamfer - aligned_chamfer)
    clip_counts = {
        clip_id: sum(1 for row in rows if row["clip_id"] == clip_id)
        for clip_id in {str(row["clip_id"]) for row in rows}
    }
    nonwrap_rows = [
        row
        for row in rows
        if row["frame_ordinal"] + args.pose_shift < clip_counts[str(row["clip_id"])]
    ]
    nonwrap_delta = np.asarray(
        [
            row["shifted"]["chamfer_px"] - row["aligned"]["chamfer_px"]
            for row in nonwrap_rows
        ]
    )
    nonwrap_support_delta = np.asarray(
        [
            row["aligned"]["edge_support_3px"]
            - row["shifted"]["edge_support_3px"]
            for row in nonwrap_rows
        ]
    )
    nonwrap_mean, nonwrap_lower, nonwrap_upper = paired_bootstrap_mean_ci(nonwrap_delta)
    clip_metrics = []
    clip_mean_chamfer_deltas = []
    for clip_id in sorted({str(row["clip_id"]) for row in rows}):
        clip_rows = [row for row in rows if row["clip_id"] == clip_id]
        clip_aligned = np.asarray(
            [row["aligned"]["chamfer_px"] for row in clip_rows]
        )
        clip_shifted = np.asarray(
            [row["shifted"]["chamfer_px"] for row in clip_rows]
        )
        clip_aligned_support = np.asarray(
            [row["aligned"]["edge_support_3px"] for row in clip_rows]
        )
        clip_shifted_support = np.asarray(
            [row["shifted"]["edge_support_3px"] for row in clip_rows]
        )
        clip_delta, clip_lower, clip_upper = paired_bootstrap_mean_ci(
            clip_shifted - clip_aligned
        )
        clip_mean_chamfer_deltas.append(clip_delta)
        clip_metrics.append(
            {
                "clip_id": clip_id,
                "frame_count": len(clip_rows),
                "aligned_chamfer_px_mean": float(clip_aligned.mean()),
                "shifted_chamfer_px_mean": float(clip_shifted.mean()),
                "paired_shifted_minus_aligned_chamfer_px_mean": clip_delta,
                "paired_frame_bootstrap_95_ci": [clip_lower, clip_upper],
                "aligned_edge_support_3px_mean": float(clip_aligned_support.mean()),
                "shifted_edge_support_3px_mean": float(clip_shifted_support.mean()),
                "edge_support_delta_percentage_points": float(
                    100.0 * (clip_aligned_support.mean() - clip_shifted_support.mean())
                ),
            }
        )
    _, clip_block_lower, clip_block_upper = paired_bootstrap_mean_ci(
        np.asarray(clip_mean_chamfer_deltas)
    )
    latency = np.asarray(latency_ms[2:] if len(latency_ms) > 2 else latency_ms)
    quality_gate = bool(lower > 0.0 and aligned_support.mean() > shifted_support.mean())
    tool_path = Path(__file__).resolve()
    preprocessor_path = (
        tool_path.parents[1]
        / "robot_wm/datasets/abc/preprocessing/abc_preprocess.py"
    )
    analysis = {
        "artifact_type": "abc-d405-nominal-geometry-probe",
        "format_version": 1,
        "tool_sha256": sha256_file(tool_path),
        "tool_git_commit": git_commit(tool_path.parents[1]),
        "status": "exploratory_pass" if quality_gate else "exploratory_fail",
        "split": "train",
        "protected_test_accessed": False,
        "future_video_conditioning_used": False,
        "clip_count": len(bundles),
        "frame_count": len(rows),
        "official_abc_commit": official_commit,
        "expected_official_abc_commit": EXPECTED_ABC_COMMIT,
        "official_scene": str(scene_path),
        "official_scene_sha256": sha256_file(scene_path),
        "robot_only_xml_sha256": xml_sha256,
        "camera": {
            "name": "top",
            "model": CAMERA_TYPE,
            "nominal_extrinsics": True,
            "per_episode_intrinsic_fovy": True,
            "principal_point_mode": "mujoco_centered_not_bundle_cx_cy",
            "distortion_applied": False,
            "official_source_lines": "assets/put_bottles/put_bottle.xml:195-204",
            "per_bundle_calibration": calibrations,
        },
        "pose_mapping": {
            "joint_states": "[left_joint1..6,right_joint1..6]",
            "gripper_states": "[left,right] normalized then +/-0.0475 finger qpos",
            "cache14_to_official14": list(CACHE14_TO_OFFICIAL14),
            "preprocessor_timing": {
                "source": "robot_wm/datasets/abc/preprocessing/abc_preprocess.py:86-92",
                "source_sha256": sha256_file(preprocessor_path),
                "behavior": (
                    "np.searchsorted timestamps are used directly, so a frame between "
                    "samples receives the next/ceiling state despite the nearest comment."
                ),
            },
        },
        "control": {
            "type": "cyclic_time_shift_within_same_train_clip",
            "pose_shift_frames": args.pose_shift,
            "timing_by_clip": control_timings,
            "nonwrap_sensitivity": {
                "frame_count": len(nonwrap_rows),
                "paired_shifted_minus_aligned_chamfer_px_mean": nonwrap_mean,
                "paired_frame_bootstrap_95_ci": [nonwrap_lower, nonwrap_upper],
                "edge_support_delta_percentage_points": float(
                    100.0 * nonwrap_support_delta.mean()
                ),
            },
        },
        "metrics": {
            "aligned_chamfer_px_mean": float(aligned_chamfer.mean()),
            "shifted_chamfer_px_mean": float(shifted_chamfer.mean()),
            "aligned_chamfer_improvement_pct": float(
                100.0 * (shifted_chamfer.mean() - aligned_chamfer.mean()) / shifted_chamfer.mean()
            ),
            "paired_shifted_minus_aligned_chamfer_px_mean": delta,
            "paired_bootstrap_95_ci": [lower, upper],
            "clip_block_bootstrap_95_ci_sensitivity": [
                clip_block_lower,
                clip_block_upper,
            ],
            "clips_with_positive_mean_chamfer_delta": int(
                np.sum(np.asarray(clip_mean_chamfer_deltas) > 0.0)
            ),
            "aligned_edge_support_3px_mean": float(aligned_support.mean()),
            "shifted_edge_support_3px_mean": float(shifted_support.mean()),
            "edge_support_delta_percentage_points": float(
                100.0 * (aligned_support.mean() - shifted_support.mean())
            ),
            "by_clip": clip_metrics,
        },
        "diagnostic_gate": {
            "definition": "chamfer paired-bootstrap lower bound > 0 and aligned 3px support > shifted",
            "pass": quality_gate,
        },
        "latency": {
            "render_pose_p50_ms": float(np.percentile(latency, 50)),
            "render_pose_p95_ms": float(np.percentile(latency, 95)),
            "render_pose_mean_ms": float(latency.mean()),
            "render_pose_fps_from_mean": float(1000.0 / latency.mean()),
            "sample_count_after_two_warmups": int(len(latency)),
        },
        "bundles": [bundle.metadata["identity_sha256"] for bundle in bundles],
        "rows": str(row_path),
        "rows_sha256": sha256_file(row_path),
        "overlays": {
            "maximum_pose_difference": max_pose_overlay_paths,
            "worst_aligned_failure_audit": failure_overlay_paths,
        },
        "limitations": [
            "Exploratory train-only calibration probe; no held-out quality claim.",
            "Observed RGB edges contain objects/background and are not robot segmentation ground truth.",
            "Official MJCF camera extrinsics are nominal and not per-episode calibration.",
            "MuJoCo rendering uses centered principal point and does not apply D405 distortion.",
            "Observed states are used only to validate geometry; deployable scaffolds must replay planned actions.",
            "ABC preprocessing uses next/ceiling timestamp resampling, not true nearest resampling; this probe cannot establish sub-frame temporal calibration.",
            "The primary four-clip-step control is about 20 source-video frames, not a subtle timing perturbation.",
        ],
    }
    analysis["identity_sha256"] = sha256_json(analysis)
    atomic_json(args.output_dir / "analysis.json", analysis)
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Create compact train-only D405 bundles")
    extract.add_argument("--clip-manifest", type=Path, required=True)
    extract.add_argument("--preprocessed-root", type=Path, required=True)
    extract.add_argument("--raw-root", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--clip-id", action="append", default=[])
    extract.add_argument("--max-clips", type=int, default=3)

    evaluate = subparsers.add_parser("evaluate", help="Render and compare aligned/time-shifted poses")
    evaluate.add_argument("--bundle-dir", type=Path, required=True)
    evaluate.add_argument("--official-abc-root", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--pose-shift", type=int, default=4)
    evaluate.add_argument("--mujoco-gl", choices=("egl", "glfw", "osmesa"), default="egl")
    evaluate.add_argument("--allow-unpinned-official", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "extract":
        result = extract_bundles(args)
    else:
        result = evaluate_bundles(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
