import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from robot_wm.datasets.agibot.dataset import AgibotDataset
from robot_wm.datasets.agibot.transform import _get_actions


IDENTITY_ROTATION_6D = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _identity_quaternions(timesteps: int) -> np.ndarray:
    quaternions = np.zeros((timesteps, 2, 4), dtype=np.float32)
    quaternions[..., 3] = 1.0
    return quaternions


def _episode(camera_x: tuple[float, ...]) -> dict:
    timesteps = len(camera_x)
    quaternions = _identity_quaternions(timesteps)
    robot_quaternions = np.zeros((timesteps, 4), dtype=np.float32)
    robot_quaternions[:, 3] = 1.0

    action_positions = np.zeros((timesteps, 2, 3), dtype=np.float32)
    action_positions[:, 0, 0] = np.arange(timesteps, dtype=np.float32)
    action_positions[:, 1, 1] = np.arange(timesteps, dtype=np.float32)

    camera_params = []
    for x in camera_x:
        camera_params.append(
            {
                "extrinsic": {
                    "rotation_matrix": np.eye(3).tolist(),
                    "translation_vector": [x, 0.0, 0.0],
                }
            }
        )

    return {
        "proprio_stats": {
            "state_effector_position": np.full(
                (timesteps, 2), 34.0, dtype=np.float32
            ),
            "state_end_orientation": quaternions.copy(),
            "state_end_position": np.zeros(
                (timesteps, 2, 3), dtype=np.float32
            ),
            "state_robot_orientation": robot_quaternions,
            "state_robot_position": np.zeros((timesteps, 3), dtype=np.float32),
            "state_joint_positon": np.zeros((timesteps, 14), dtype=np.float32),
            "action_effector_position": np.zeros(
                (timesteps, 2), dtype=np.float32
            ),
            "action_end_orientation": quaternions.copy(),
            "action_end_position": action_positions,
        },
        "camera_params": {
            "head_extrinsic_params_aligned": camera_params,
        },
    }


class AgibotTransformTest(unittest.TestCase):
    def test_action_mode_uses_correctly_spelled_end_position(self):
        episode = _episode((0.0, 0.0, 0.0))

        actions = _get_actions(episode, action_type="action")

        self.assertEqual(actions.shape, (3, 34))
        np.testing.assert_allclose(
            actions[:, :3], episode["proprio_stats"]["action_end_position"][:, 0]
        )
        self.assertTrue(np.isfinite(actions).all())

    def test_delta_camera_uses_relative_motion(self):
        actions = _get_actions(
            _episode((0.0, 1.0, 3.0)),
            action_type="delta-state+camera+abs_finger",
        )

        self.assertEqual(actions.shape, (3, 43))
        np.testing.assert_allclose(
            actions[:, -9:-6],
            np.asarray(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            actions[:, -6:],
            np.tile(IDENTITY_ROTATION_6D, (3, 1)),
            atol=1e-6,
        )

    def test_static_camera_pose_is_zero_relative_motion(self):
        actions = _get_actions(
            _episode((5.0, 5.0, 5.0)),
            action_type="delta-state+camera+abs_finger",
        )
        camera_delta = actions[:, -9:]

        np.testing.assert_allclose(camera_delta[:, :3], np.zeros((3, 3)), atol=1e-6)
        # A no-rotation delta is the identity basis in the repository's 6D
        # rotation encoding, rather than six literal zeros.
        np.testing.assert_allclose(
            camera_delta[:, 3:],
            np.tile(IDENTITY_ROTATION_6D, (3, 1)),
            atol=1e-6,
        )

    def test_absolute_action_keeps_absolute_camera_pose(self):
        actions = _get_actions(
            _episode((0.0, 1.0, 3.0)),
            action_type="action+camera+abs_finger",
        )

        np.testing.assert_allclose(actions[:, -9], [0.0, 1.0, 3.0])

    def test_official_zero_quaternion_sentinel_requires_fixed_base(self):
        episode = _episode((0.0, 0.0, 0.0))
        episode["proprio_stats"]["state_robot_orientation"][:] = 0.0
        actions = _get_actions(episode, action_type="delta-state")
        self.assertTrue(np.isfinite(actions).all())

        episode["proprio_stats"]["state_robot_position"][1, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "fixed base"):
            _get_actions(episode, action_type="delta-state")


class AgibotDatasetTest(unittest.TestCase):
    @staticmethod
    def _write_manifest(root: Path, rows: list[tuple[str, str, str]]) -> Path:
        manifest = root / "manifest.csv"
        lines = ["task_id,episode_id,dataset"]
        lines.extend(",".join(row) for row in rows)
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return manifest

    def test_environment_roots_are_resolved_at_construction_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._write_manifest(Path(tmp), [("1", "2", "scr")])
            with mock.patch.dict(os.environ, {"LACWM_DATA": "/runtime/data"}):
                dataset = AgibotDataset(manifest=str(manifest))

        self.assertEqual(dataset.dataset_roots["scr"], "/runtime/data/agibot")
        self.assertEqual(
            dataset.dataset_roots["viscam"], "/runtime/data/agibot_combined"
        )

    def test_explicit_root_mapping_supports_custom_dataset_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_manifest(root, [("task", "episode", "lab")])
            dataset = AgibotDataset(
                manifest=str(manifest),
                dataset_roots={"lab": root / "custom-data"},
                max_retries=0,
            )
            episode = {"loaded": True}
            with mock.patch.object(
                dataset, "_get_sample_from_ids", return_value=episode
            ) as load:
                self.assertIs(dataset._get_sample(0), episode)

        load.assert_called_once_with(
            os.fspath(root / "custom-data"), "task", "episode"
        )

    def test_unknown_manifest_dataset_id_fails_during_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._write_manifest(
                Path(tmp), [("task", "episode", "unknown")]
            )

            with self.assertRaisesRegex(
                ValueError, "dataset ID 'unknown'.*row 2"
            ):
                AgibotDataset(manifest=str(manifest))

    def test_retry_is_bounded_aggregated_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [(f"task-{i}", f"episode-{i}", "scr") for i in range(3)]
            manifest = self._write_manifest(root, rows)
            dataset = AgibotDataset(
                manifest=str(manifest),
                dataset_roots={"scr": root},
                max_retries=2,
            )
            dataset._gen.manual_seed(23)
            expected_generator = torch.Generator()
            expected_generator.manual_seed(23)
            expected_indices = [0]
            expected_indices.extend(
                int(torch.randint(0, 3, (1,), generator=expected_generator).item())
                for _ in range(2)
            )

            def fail(_root, task_id, episode_id):
                raise FileNotFoundError(f"missing {task_id}/{episode_id}")

            with mock.patch.object(
                dataset, "_get_sample_from_ids", side_effect=fail
            ) as load, self.assertRaises(RuntimeError) as raised:
                dataset._get_sample(0)

        attempted_indices = [
            int(call.args[1].removeprefix("task-")) for call in load.call_args_list
        ]
        self.assertEqual(attempted_indices, expected_indices)
        self.assertEqual(load.call_count, 3)
        self.assertIsInstance(raised.exception.__cause__, FileNotFoundError)
        message = str(raised.exception)
        self.assertIn("after 3 attempts", message)
        self.assertIn("attempt 1/3", message)
        self.assertIn("attempt 3/3", message)
        self.assertIn("FileNotFoundError: missing", message)


if __name__ == "__main__":
    unittest.main()
