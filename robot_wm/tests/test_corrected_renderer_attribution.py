import unittest

import numpy as np

from tools.corrected_renderer_attribution import (
    ARMS,
    PoseRender,
    combine_native_history_with_action,
    mask_iou,
    paired_difference,
    planned_joint_motion,
    rgb_robot_band_metrics,
    robot_flow_error,
    select_motion_stratified,
    symmetric_boundary_chamfer,
)
from tools.measure_corrected_renderer_predictor_latency import causal_predictor_features


class CorrectedRendererAttributionTest(unittest.TestCase):
    def test_motion_selector_is_deterministic_and_donors_stay_within_stratum(self):
        eligible = [
            {
                "manifest_index": 384 + index,
                "clip_id": f"{index:064x}",
                "episode_dir": f"/train/episode_{index}",
                "planned_joint_motion_rms": float(index),
                "protected_test_accessed": False,
            }
            for index in range(60)
        ]
        first = select_motion_stratified(eligible)
        second = select_motion_stratified(list(reversed(eligible)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        by_index = {item["manifest_index"]: item for item in first}
        for item in first:
            donor = by_index[item["donor_manifest_index"]]
            self.assertNotEqual(item["manifest_index"], donor["manifest_index"])
            self.assertNotEqual(item["episode_dir"], donor["episode_dir"])
            self.assertEqual(item["motion_stratum"], donor["motion_stratum"])

    def test_shuffled_features_keep_recipient_history_and_only_donate_action(self):
        history = np.asarray([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
        action = np.asarray([[1.0], [2.0], [3.0]])
        donor = np.asarray([1, 2, 0])
        shuffled = combine_native_history_with_action(history, action[donor])
        np.testing.assert_array_equal(shuffled[:, :2], history)
        np.testing.assert_array_equal(shuffled[:, 2:], action[donor])
        self.assertTrue(np.all(donor != np.arange(len(donor))))

    def test_planned_motion_ignores_gripper_and_padding(self):
        action = np.zeros((13, 5, 23), dtype=np.float32)
        action[4:12, -1, :12] = 2.0
        baseline = planned_joint_motion(action)
        self.assertEqual(baseline, 2.0)
        action[..., 12:] = 1000.0
        self.assertEqual(planned_joint_motion(action), baseline)

    def test_silhouette_and_fixed_region_rgb_metrics_prefer_alignment(self):
        oracle = np.zeros((64, 64), dtype=bool)
        oracle[20:44, 18:46] = True
        shifted = np.roll(oracle, 12, axis=1)
        from tools.abc_d405_nominal_geometry_probe import silhouette_boundary

        edges = silhouette_boundary(oracle)
        self.assertEqual(mask_iou(oracle, oracle), 1.0)
        self.assertLess(
            symmetric_boundary_chamfer(oracle, oracle),
            symmetric_boundary_chamfer(shifted, oracle),
        )
        aligned = rgb_robot_band_metrics(oracle, oracle, edges)
        control = rgb_robot_band_metrics(shifted, oracle, edges)
        self.assertLess(
            aligned["rgb_robot_band_chamfer_px"],
            control["rgb_robot_band_chamfer_px"],
        )
        self.assertGreater(
            aligned["rgb_robot_band_edge_support_3px"],
            control["rgb_robot_band_edge_support_3px"],
        )

    @staticmethod
    def _synthetic_render(translation_x: float) -> PoseRender:
        height = width = 64
        mask = np.zeros((height, width), dtype=bool)
        mask[20:44, 20:44] = True
        geom_id = np.full((height, width), -1, dtype=np.int32)
        geom_id[mask] = 0
        depth = np.full((height, width), 100.0, dtype=np.float32)
        depth[mask] = 1.0
        return PoseRender(
            mask=mask,
            geom_id=geom_id,
            depth=depth,
            geom_xpos=np.asarray([[translation_x, 0.0, 0.0]], dtype=np.float64),
            geom_xmat=np.eye(3, dtype=np.float64)[None],
            camera_xpos=np.zeros(3, dtype=np.float64),
            camera_xmat=np.eye(3, dtype=np.float64),
            fy=100.0,
            render_ms=0.0,
        )

    def test_robot_flow_transport_has_known_endpoint_error(self):
        source = self._synthetic_render(0.0)
        oracle_target = self._synthetic_render(0.10)
        candidate_target = self._synthetic_render(0.05)
        metrics = robot_flow_error(
            source,
            candidate_target,
            source,
            oracle_target,
            far_depth=10.0,
        )
        self.assertAlmostEqual(metrics["robot_flow_epe_px"], 5.0, places=5)
        self.assertAlmostEqual(metrics["oracle_flow_magnitude_px"], 10.0, places=5)
        self.assertLess(metrics["oracle_source_reprojection_rmse_px"], 1e-10)
        oracle_metrics = robot_flow_error(
            source,
            oracle_target,
            source,
            oracle_target,
            far_depth=10.0,
        )
        self.assertLess(oracle_metrics["robot_flow_epe_px"], 1e-10)
        below_threshold = robot_flow_error(
            source,
            candidate_target,
            source,
            oracle_target,
            far_depth=10.0,
            min_oracle_motion_px=1000.0,
        )
        self.assertIsNone(below_threshold["robot_flow_epe_px"])
        self.assertEqual(below_threshold["oracle_moving_flow_sample_count"], 0.0)
        self.assertFalse(below_threshold["flow_transition_scored"])

    def test_paired_difference_is_deterministic_and_respects_direction(self):
        reference = np.asarray([2.0, 3.0, 4.0, 5.0])
        candidate = reference - 1.0
        first = paired_difference(reference, candidate, seed_offset=9)
        second = paired_difference(reference, candidate, seed_offset=9)
        self.assertEqual(first, second)
        self.assertEqual(first["paired_favorable_difference_mean"], 1.0)
        self.assertGreater(first["paired_bootstrap_95_ci"][0], 0.0)
        higher = paired_difference(
            reference, reference + 1.0, seed_offset=10, higher_is_better=True
        )
        self.assertEqual(higher["paired_favorable_difference_mean"], 1.0)

    def test_all_five_registered_arms_are_present(self):
        self.assertEqual(
            set(ARMS),
            {
                "raw_command",
                "predicted_corrected",
                "hold_current",
                "episode_shuffled",
                "measured_oracle",
            },
        )

    def test_latency_replay_feature_builder_never_reads_future_state(self):
        states = np.full((13, 14), np.nan, dtype=np.float32)
        states[:5] = np.arange(5 * 14, dtype=np.float32).reshape(5, 14)
        actions = np.arange(13 * 5 * 14, dtype=np.float32).reshape(13, 5, 14)
        history, future = causal_predictor_features(states, actions)
        self.assertEqual(history.shape, (406,))
        self.assertEqual(future.shape, (8, 5, 14))
        self.assertTrue(np.isfinite(history).all())
        self.assertTrue(np.isfinite(future).all())


if __name__ == "__main__":
    unittest.main()
