from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.vjepa2_ac_stage0 import (
    ACTION_DIM,
    ACTION_STEPS,
    FRAME_OFFSETS,
    QualificationError,
    build_action_controls,
    integrate_actions,
    lerobot_state8_to_vjepa_pose7,
    metric_dict,
    nonwrapping_shift,
    official_fallback_crop,
    paired_bootstrap_relative_improvement,
    poses_to_diffs,
    select_native_clips,
)


def test_lerobot_state_adapter_uses_gripper_col7_not_padding_col6() -> None:
    state = np.asarray(
        [
            [0.1, 0.2, 0.3, np.pi - 0.1, -0.2, 0.4, 0.0, 0.25],
            [0.2, 0.3, 0.4, -np.pi + 0.1, -0.1, 0.5, 0.0, 0.75],
        ],
        dtype=np.float32,
    )
    pose = lerobot_state8_to_vjepa_pose7(state)
    assert pose.shape == (2, 7)
    np.testing.assert_allclose(pose[:, :6], state[:, :6])
    np.testing.assert_allclose(pose[:, 6], state[:, 7])
    assert not np.array_equal(pose[:, 6], state[:, 6])


def test_lerobot_state_adapter_rejects_nonzero_padding_and_bad_gripper() -> None:
    state = np.zeros((2, 8), dtype=np.float32)
    state[:, 6] = 0.01
    with pytest.raises(ValueError, match="padding column 6"):
        lerobot_state8_to_vjepa_pose7(state)
    state[:, 6] = 0.0
    state[:, 7] = 1.2
    with pytest.raises(ValueError, match="outside"):
        lerobot_state8_to_vjepa_pose7(state)


def test_official_pose_difference_and_integration_round_trip() -> None:
    rng = np.random.default_rng(7)
    poses = np.zeros((8, ACTION_DIM), dtype=np.float32)
    poses[0] = [0.3, -0.1, 0.5, 3.0, -0.2, 0.1, 0.2]
    for index in range(1, len(poses)):
        delta_xyz = rng.normal(scale=0.01, size=3)
        delta_rot = Rotation.from_rotvec(rng.normal(scale=0.03, size=3))
        previous = Rotation.from_euler("xyz", poses[index - 1, 3:6])
        pose_rotation = delta_rot * previous
        poses[index, :3] = poses[index - 1, :3] + delta_xyz
        poses[index, 3:6] = pose_rotation.as_euler("xyz")
        poses[index, 6] = np.clip(poses[index - 1, 6] + 0.05, 0, 1)
    actions = poses_to_diffs(poses)
    reconstructed = integrate_actions(poses[0], actions)
    np.testing.assert_allclose(reconstructed[:, :3], poses[:, :3], atol=2e-6)
    np.testing.assert_allclose(reconstructed[:, 6], poses[:, 6], atol=2e-6)
    expected_rot = Rotation.from_euler("xyz", poses[:, 3:6]).as_matrix()
    actual_rot = Rotation.from_euler("xyz", reconstructed[:, 3:6]).as_matrix()
    np.testing.assert_allclose(actual_rot, expected_rot, atol=2e-6)


def test_action_controls_are_nonwrapping_and_donor_exact() -> None:
    aligned = np.arange(ACTION_STEPS * ACTION_DIM, dtype=np.float32).reshape(
        ACTION_STEPS, ACTION_DIM
    )
    donor = aligned + 1000
    logged = aligned - 1000
    controls = build_action_controls(aligned, donor, logged)
    np.testing.assert_array_equal(controls["episode_shuffled"], donor)
    np.testing.assert_array_equal(controls["logged_raw"], logged)
    np.testing.assert_array_equal(controls["time_shifted_plus1"][:-1], aligned[1:])
    np.testing.assert_array_equal(controls["time_shifted_plus1"][-1], 0)
    np.testing.assert_array_equal(controls["time_shifted_minus1"][1:], aligned[:-1])
    np.testing.assert_array_equal(controls["time_shifted_minus1"][0], 0)
    with pytest.raises(ValueError, match="nonzero"):
        nonwrapping_shift(aligned, 0)


def _manifest_row(root: Path, episode: int, start: int, clip_id: str) -> dict:
    chunk = episode // 1000
    parquet = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    video = (
        root
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.exterior_image_1_left"
        / f"episode_{episode:06d}.mp4"
    )
    parquet.parent.mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"parquet")
    video.write_bytes(b"video")
    return {
        "split": "train",
        "protected": False,
        "camera": "exterior_image_1_left",
        "clip_id": clip_id,
        "episode_index": episode,
        "start": start,
        "trajectory_length": 100,
    }


def test_native_cohort_is_deterministic_episode_disjoint_and_uses_four_frame_step(
    tmp_path: Path,
) -> None:
    rows = [
        _manifest_row(tmp_path, episode, 5 + offset, f"{episode * 100 + offset:064x}")
        for episode in range(10, 14)
        for offset in (0, 1)
    ]
    first = select_native_clips(rows, data_root=tmp_path, count=3, seed=123)
    second = select_native_clips(rows, data_root=tmp_path, count=3, seed=123)
    assert first == second
    assert len({clip.episode_index for clip in first}) == 3
    for clip in first:
        assert tuple(value - clip.start for value in clip.frame_indices) == FRAME_OFFSETS


def test_native_cohort_rejects_protected_or_validation_rows(tmp_path: Path) -> None:
    row = _manifest_row(tmp_path, 10, 5, "a" * 64)
    row["protected"] = True
    with pytest.raises(QualificationError, match="unprotected train"):
        select_native_clips([row], data_root=tmp_path, count=2, seed=1)


def test_metrics_and_bootstrap_keep_loss_sign() -> None:
    target = np.ones((2, 3, 4), dtype=np.float32)
    prediction = target + 0.1
    metrics = metric_dict(prediction, target)
    assert metrics["l1"] == pytest.approx(0.1)
    assert metrics["mse"] == pytest.approx(0.01)
    assert metrics["token_cosine"] == pytest.approx(1.0)
    effect = paired_bootstrap_relative_improvement(
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
        samples=500,
        seed=1,
    )
    assert effect["point_percent"] == pytest.approx(50.0)
    assert effect["ci95_low_percent"] == pytest.approx(50.0)


def test_official_droid_transform_fallback_is_centered_180_by_243() -> None:
    assert official_fallback_crop(180, 320) == (0, 38, 180, 243)
