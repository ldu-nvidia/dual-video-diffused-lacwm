import inspect
import unittest

import numpy as np

from tools.trajectory_consistent_renderer_gate import (
    ACTION_DIM,
    ARMS,
    BOOTSTRAP_SAMPLES,
    EXPECTED_D405_POOL,
    EXPECTED_UNTOUCHED_D405,
    PRIMARY_FLOW_LABELS,
    RidgeState,
    apply_holm,
    bootstrap_indices,
    hybrid_anchor_raw_delta,
    paired_effect,
    raw_trajectory,
    recurrent_base_features,
    rollout_recurrent,
    select_fresh_motion_stratified,
)


class TrajectoryConsistentRendererGateTest(unittest.TestCase):
    def test_fresh_selector_excludes_prior_and_keeps_donors_in_stratum(self):
        eligible = [
            {
                "manifest_index": 384 + index,
                "clip_id": f"{index:064x}",
                "episode_dir": f"/train/episode_{index}",
                "planned_joint_motion_rms": float(index),
                "protected_test_accessed": False,
            }
            for index in range(EXPECTED_D405_POOL)
        ]
        excluded = {item["clip_id"] for item in eligible[: EXPECTED_D405_POOL - EXPECTED_UNTOUCHED_D405]}
        first, untouched = select_fresh_motion_stratified(eligible, excluded)
        second, _ = select_fresh_motion_stratified(list(reversed(eligible)), excluded)
        self.assertEqual(first, second)
        self.assertEqual(len(untouched), EXPECTED_UNTOUCHED_D405)
        self.assertEqual(len(first), 24)
        self.assertFalse({item["clip_id"] for item in first} & excluded)
        by_index = {item["manifest_index"]: item for item in first}
        for item in first:
            donor = by_index[item["donor_manifest_index"]]
            self.assertNotEqual(item["clip_id"], donor["clip_id"])
            self.assertNotEqual(item["episode_dir"], donor["episode_dir"])
            self.assertEqual(item["motion_stratum"], donor["motion_stratum"])

    def test_zero_recurrent_correction_replays_raw_trajectory_without_future_target(self):
        n = 3
        history = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
        context = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
        actions = np.arange(n * 8 * 5 * ACTION_DIM, dtype=np.float32).reshape(
            n, 8, 5, ACTION_DIM
        ) / 1000.0
        q0 = np.arange(n * ACTION_DIM, dtype=np.float32).reshape(n, ACTION_DIM) / 10.0
        raw = raw_trajectory(q0, actions)
        feature_dim = recurrent_base_features(
            history,
            context,
            actions,
            raw,
            raw[:, 0],
            np.zeros_like(q0),
            0,
        ).shape[1]
        model = RidgeState(
            input_mean=np.zeros(feature_dim, dtype=np.float32),
            input_std=np.ones(feature_dim, dtype=np.float32),
            target_mean=np.zeros(ACTION_DIM, dtype=np.float32),
            target_std=np.ones(ACTION_DIM, dtype=np.float32),
            coef=np.zeros((ACTION_DIM, feature_dim), dtype=np.float32),
            intercept=np.zeros(ACTION_DIM, dtype=np.float32),
        )
        prediction = rollout_recurrent(model, history, context, actions, q0, 1.0)
        np.testing.assert_allclose(prediction, raw, atol=2e-6)
        self.assertNotIn("measured", inspect.signature(rollout_recurrent).parameters)
        self.assertNotIn("target", inspect.signature(rollout_recurrent).parameters)

    def test_hybrid_uses_absolute_source_and_native_raw_increment(self):
        raw = np.zeros((2, 9, ACTION_DIM), dtype=np.float32)
        absolute = np.zeros_like(raw)
        raw[:, :, :12] = np.arange(9, dtype=np.float32)[None, :, None]
        absolute[:, :, :12] = 10.0 + 2.0 * np.arange(9, dtype=np.float32)[None, :, None]
        source, target = hybrid_anchor_raw_delta(absolute, raw)
        np.testing.assert_array_equal(source, absolute[:, :-1])
        np.testing.assert_array_equal(target - source, np.diff(raw, axis=1))
        self.assertFalse(np.array_equal(target[:, :-1], source[:, 1:]))

    def test_common_bootstrap_and_holm_are_deterministic(self):
        indices = bootstrap_indices()
        self.assertEqual(indices.shape, (BOOTSTRAP_SAMPLES, 24))
        reference = np.linspace(2.0, 3.0, 24)
        candidate = reference - 1.0
        effects = {}
        samples = {}
        for label in PRIMARY_FLOW_LABELS:
            effects[label], samples[label] = paired_effect(reference, candidate, indices)
        first = apply_holm(effects, samples)
        second_effects = {}
        second_samples = {}
        for label in PRIMARY_FLOW_LABELS:
            second_effects[label], second_samples[label] = paired_effect(
                reference, candidate, indices
            )
        second = apply_holm(second_effects, second_samples)
        self.assertEqual(first, second)
        self.assertTrue(all(row["stepdown_rejected"] for row in first))
        self.assertTrue(all(row["one_sided_lower_bound"] > 0 for row in first))

    def test_all_frozen_arms_are_present(self):
        self.assertEqual(
            set(ARMS),
            {
                "raw_command",
                "raw_episode_shuffled",
                "absolute_ridge",
                "recurrent_delta",
                "recurrent_delta_shuffled",
                "hybrid_anchor_raw_delta",
                "hold_current",
                "measured_oracle",
            },
        )


if __name__ == "__main__":
    unittest.main()
