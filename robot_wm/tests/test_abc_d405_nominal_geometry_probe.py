import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.abc_d405_nominal_geometry_probe import (
    CACHE14_TO_OFFICIAL14,
    build_robot_only_xml,
    cache14_to_official14,
    edge_alignment_metrics,
    paired_bootstrap_mean_ci,
    require_train_row,
    resolve_raw_mcap,
    validate_bundle_metadata,
)


class ABCD405NominalGeometryProbeTest(unittest.TestCase):
    def test_cache_action_permutation_matches_official_order(self):
        cache = np.arange(14, dtype=np.float32)
        official = cache14_to_official14(cache)
        np.testing.assert_array_equal(official, cache[list(CACHE14_TO_OFFICIAL14)])
        np.testing.assert_array_equal(official, [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13])

    def test_non_train_rows_and_bundles_are_rejected(self):
        row = {"split": "val", "episode_dir": "/x", "frame_indices": [0]}
        with self.assertRaisesRegex(ValueError, "split='train'"):
            require_train_row(row)
        metadata = {
            "artifact_type": "abc-d405-nominal-geometry-bundle",
            "split": "test",
            "protected_test_accessed": False,
            "camera": {"camera_type": "Intel RealSense D405"},
        }
        with self.assertRaisesRegex(ValueError, "only train"):
            validate_bundle_metadata(metadata)

    def test_bundle_requires_explicit_no_test_proof(self):
        metadata = {
            "artifact_type": "abc-d405-nominal-geometry-bundle",
            "split": "train",
            "protected_test_accessed": True,
            "camera": {"camera_type": "Intel RealSense D405"},
        }
        with self.assertRaisesRegex(ValueError, "protected_test_accessed=false"):
            validate_bundle_metadata(metadata)

    def test_raw_path_is_derived_within_allowed_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preprocessed = root / "abc_pp"
            raw = root / "abc_raw" / "data" / "train"
            episode = preprocessed / "task" / "episode_1"
            raw_episode = raw / "task" / "episode_1"
            episode.mkdir(parents=True)
            raw_episode.mkdir(parents=True)
            (raw_episode / "episode.mcap").write_bytes(b"mcap")
            self.assertEqual(
                resolve_raw_mcap(episode, preprocessed, raw),
                raw_episode / "episode.mcap",
            )
            outside = root / "outside" / "episode_1"
            outside.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "outside preprocessed root"):
                resolve_raw_mcap(outside, preprocessed, raw)

    def test_robot_only_xml_removes_task_assets_and_bodies(self):
        xml = """<mujoco>
          <compiler meshdir="old" texturedir="old"/>
          <asset>
            <material name="black"/><material name="task"/>
            <mesh name="robot" file="i2rt_yam/assets/robot.stl"/>
            <mesh name="bottle" file="task/bottle.obj"/>
          </asset>
          <worldbody>
            <light/><body name="gate_collision"/><body name="left_arm"/>
            <body name="right_arm"/><body name="bottle_1"/>
          </worldbody>
          <keyframe><key name="home"/></keyframe>
        </mujoco>"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scene.xml"
            source.write_text(xml)
            sanitized = build_robot_only_xml(source, root / "assets")
        self.assertIn("i2rt_yam/assets/robot.stl", sanitized)
        self.assertNotIn("task/bottle.obj", sanitized)
        self.assertNotIn("bottle_1", sanitized)
        self.assertNotIn("keyframe", sanitized)
        self.assertNotIn("texturedir", sanitized)

    def test_edge_metric_prefers_exact_synthetic_alignment(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[16:48, 20:44] = True
        from tools.abc_d405_nominal_geometry_probe import silhouette_boundary

        image_edges = silhouette_boundary(mask)
        shifted = np.roll(mask, 10, axis=1)
        aligned_metrics = edge_alignment_metrics(mask, image_edges)
        shifted_metrics = edge_alignment_metrics(shifted, image_edges)
        self.assertLess(aligned_metrics["chamfer_px"], shifted_metrics["chamfer_px"])
        self.assertGreater(aligned_metrics["edge_support_3px"], shifted_metrics["edge_support_3px"])

    def test_bootstrap_is_deterministic_and_signed(self):
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        first = paired_bootstrap_mean_ci(values, samples=2_000)
        second = paired_bootstrap_mean_ci(values, samples=2_000)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 2.5)
        self.assertGreater(first[1], 0.0)


if __name__ == "__main__":
    unittest.main()
