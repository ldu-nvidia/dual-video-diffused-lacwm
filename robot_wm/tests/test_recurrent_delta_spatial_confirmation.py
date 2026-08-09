from __future__ import annotations

import copy
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import recurrent_delta_spatial_confirmation as study


def _eligible_and_receipt():
    eligible = [
        {
            "manifest_index": 384 + index,
            "clip_id": f"clip-{index:03d}",
            "episode_dir": f"/episodes/{index:03d}",
            "planned_joint_motion_rms": float(index),
        }
        for index in range(study.EXPECTED_D405_POOL)
    ]
    gate0c = {item["clip_id"] for item in eligible[-24:]}
    fresh79 = sorted(
        [item for item in eligible if item["clip_id"] not in gate0c],
        key=lambda item: (item["planned_joint_motion_rms"], item["clip_id"]),
    )
    bins = np.array_split(np.asarray(fresh79, dtype=object), 3)
    fresh24 = {
        item["clip_id"] for values in bins for item in values.tolist()[:8]
    }
    renderer_unopened = study._renderer_unopened_census(eligible, gate0c, fresh24)
    collision_by_stratum = {0: 5, 1: 10, 2: 11}
    collision_ids = set()
    for stratum, count in collision_by_stratum.items():
        values = sorted(
            [item for item in renderer_unopened if item["motion_stratum"] == stratum],
            key=lambda item: item["clip_id"],
        )
        collision_ids.update(item["clip_id"] for item in values[:count])
    episodes = [
        {
            "clip_id": item["clip_id"],
            "manifest_index": item["manifest_index"],
            "fresh24_motion_stratum": item["motion_stratum"],
            "later_vpm_dev64_opened": item["clip_id"] in collision_ids,
        }
        for item in renderer_unopened
    ]
    receipt = {
        "kind": "recurrent_delta_spatial_confirmation_freshness_audit",
        "claim_boundary": {"outcome_payloads_inspected": False},
        "canonical_bindings": {
            "fresh24_registration_identity_sha256": study.PRIOR_REGISTRATION_IDENTITY
        },
        "counts": {
            "d405_eligible": 103,
            "gate0c_renderer_opened": 24,
            "fresh24_renderer_opened": 24,
            "gate0c_fresh24_overlap": 0,
            "renderer_outcome_unopened": 55,
            "renderer_outcome_unopened_by_fresh24_stratum": {
                "0": 19,
                "1": 18,
                "2": 18,
            },
            "later_vpm_dev64_opened_collision": 26,
            "strict_post_fresh_outcome_unopened": 29,
            "strict_post_fresh_outcome_unopened_by_fresh24_stratum": {
                "0": 14,
                "1": 8,
                "2": 7,
            },
        },
        "collision_audit": {
            "vpm_invertible_multirate": {
                "development_clip_range": [416, 480],
                "post_selection_exploratory_dev_opened": True,
                "matched_renderer_reserve_id_count": 26,
                "classification": "strict_freshness_collision",
                "registration_identity_sha256": "a" * 64,
                "endpoint_identity_sha256": "b" * 64,
                "audit_identity_sha256": "c" * 64,
                "run_complete_identity_sha256": "d" * 64,
            }
        },
        "episodes": episodes,
    }
    return eligible, gate0c, fresh24, receipt


def _aggregates(*, spatial_degradation: float = 0.1):
    rows = []
    for index in range(study.SCORE_CLIPS):
        clip = f"score-{index:02d}"
        values = {
            "raw_command": (10.0, 2.0, 7.0, 0.900),
            "absolute_ridge": (8.0, 2.0, 7.0, 0.900),
            "recurrent_delta": (
                5.0,
                2.0 + spatial_degradation,
                7.0 + spatial_degradation,
                0.898,
            ),
            "recurrent_delta_shuffled": (20.0, 20.0, 20.0, 0.2),
            "measured_oracle": (0.0, 0.0, 6.9, 1.0),
        }
        for arm, (flow, silhouette, rgb, iou) in values.items():
            rows.append(
                {
                    "clip_id": clip,
                    "motion_stratum": 0 if index < 19 else (1 if index < 37 else 2),
                    "arm": arm,
                    "metrics": {
                        "robot_flow_epe_px": flow,
                        "silhouette_boundary_chamfer_px": silhouette,
                        "rgb_robot_band_chamfer_px": rgb,
                        "silhouette_iou": iou,
                    },
                    "split": "train",
                    "validation_accessed": False,
                    "protected_test_accessed": False,
                }
            )
    return rows


class RecurrentDeltaSpatialConfirmationTest(unittest.TestCase):
    def test_selector_uses_all_55_renderer_unopened_and_discloses_overlap(self):
        eligible, gate0c, fresh24, receipt = _eligible_and_receipt()
        selected = study.select_remaining_census(eligible, gate0c, fresh24, receipt)
        self.assertEqual(len(selected), 55)
        self.assertEqual(
            tuple(sum(item["motion_stratum"] == value for item in selected) for value in range(3)),
            (19, 18, 18),
        )
        collided = {
            item["clip_id"]
            for item in receipt["episodes"]
            if item["later_vpm_dev64_opened"]
        }
        self.assertEqual(len({item["clip_id"] for item in selected} & collided), 26)
        by_index = {item["manifest_index"]: item for item in selected}
        for item in selected:
            donor = by_index[item["donor_manifest_index"]]
            self.assertNotEqual(item["clip_id"], donor["clip_id"])
            self.assertEqual(item["motion_stratum"], donor["motion_stratum"])

    def test_selector_rejects_tampered_collision_receipt(self):
        eligible, gate0c, fresh24, receipt = _eligible_and_receipt()
        tampered = copy.deepcopy(receipt)
        first_unopened = next(
            item for item in tampered["episodes"] if not item["later_vpm_dev64_opened"]
        )
        first_unopened["later_vpm_dev64_opened"] = True
        with self.assertRaises(study.ConfirmationError):
            study.select_remaining_census(eligible, gate0c, fresh24, tampered)

    def test_common_bootstrap_is_deterministic_and_stratified_19_18_18(self):
        strata = np.asarray([0] * 19 + [1] * 18 + [2] * 18, dtype=np.int8)
        first = study.stratified_bootstrap_indices(strata, samples=256, seed=77)
        second = study.stratified_bootstrap_indices(strata, samples=256, seed=77)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (256, 55))
        self.assertTrue(np.all(first[:, :19] < 19))
        self.assertTrue(np.all((first[:, 19:37] >= 19) & (first[:, 19:37] < 37)))
        self.assertTrue(np.all(first[:, 37:] >= 37))

    def test_two_holm_families_jointly_pass_clear_flow_and_spatial_ni(self):
        aggregates = _aggregates(spatial_degradation=0.1)
        clip_order = [f"score-{index:02d}" for index in range(55)]
        strata = np.asarray([0] * 19 + [1] * 18 + [2] * 18, dtype=np.int8)
        indices = study.stratified_bootstrap_indices(strata, samples=1000, seed=3)
        result = study.analyze_aggregates(aggregates, clip_order, indices)
        self.assertEqual(result["decision"], "GO_RECURRENT_DELTA_WAN_SCREEN")
        self.assertTrue(all(result["gates"]["flow_superiority"].values()))
        self.assertTrue(all(result["gates"]["spatial_noninferiority"].values()))
        self.assertEqual(result["holm_families"]["flow_superiority_3"]["family_size"], 3)
        self.assertEqual(
            result["holm_families"]["spatial_noninferiority_6"]["family_size"], 6
        )

    def test_spatial_ni_fails_when_degradation_exceeds_quarter_pixel(self):
        aggregates = _aggregates(spatial_degradation=0.4)
        clip_order = [f"score-{index:02d}" for index in range(55)]
        strata = np.asarray([0] * 19 + [1] * 18 + [2] * 18, dtype=np.int8)
        indices = study.stratified_bootstrap_indices(strata, samples=1000, seed=3)
        result = study.analyze_aggregates(aggregates, clip_order, indices)
        self.assertEqual(result["decision"], "STOP_RECURRENT_DELTA_GEOMETRY")
        label = "recurrent_delta_vs_absolute_ridge:silhouette_boundary_chamfer_px"
        self.assertFalse(result["gates"]["spatial_noninferiority"][label])

    def test_causal_prefix_extraction_is_invariant_to_future_state_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actions = np.arange(13 * 5 * 14, dtype=np.float32).reshape(13, 5, 14)
            frame_indices = list(range(0, 65, 5))
            prefix_joint = np.arange(5 * 12, dtype=np.float32).reshape(5, 12)
            prefix_gripper = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
            outputs = []
            records = []
            for variant in (0.0, 10000.0):
                episode = root / f"episode-{int(variant)}"
                episode.mkdir()
                joint_states = np.full((65, 12), variant, dtype=np.float32)
                grip_states = np.full((65, 2), variant, dtype=np.float32)
                joint_states[frame_indices[:5]] = prefix_joint
                grip_states[frame_indices[:5]] = prefix_gripper
                flat_actions = actions.reshape(65, 14)
                np.savez(
                    episode / "states.npz",
                    joint_states=joint_states,
                    gripper_states=grip_states,
                    joint_actions=flat_actions[:, :12],
                    gripper_actions=flat_actions[:, 12:],
                    frame_ts=np.arange(65, dtype=np.int64),
                )
                row = {
                    "clip_id": "same-clip",
                    "episode_dir": str(episode),
                    "frame_indices": frame_indices,
                    "start": 0,
                }
                history, future, q4, record = study._causal_clip_input(row, actions)
                outputs.append((history, future, q4))
                records.append(record)
            for left, right in zip(outputs[0], outputs[1], strict=True):
                np.testing.assert_array_equal(left, right)
            self.assertTrue(records[0]["score_state_npz_container_opened_for_causal_prefix"])
            self.assertFalse(records[0]["future_measured_state_indices_extracted"])
            self.assertFalse(records[0]["future_measured_state_used"])
            self.assertFalse(records[0]["future_rgb_opened"])

    def test_cli_has_no_validation_or_protected_test_input(self):
        parser = study.build_parser()
        text = parser.format_help()
        self.assertNotIn("validation", text)
        self.assertNotIn("protected", text)
        register = parser.parse_args(
            [
                "register",
                "--output",
                "/o",
                "--train-cache",
                "/c",
                "--train-manifest",
                "/m",
                "--preprocessed-root",
                "/p",
                "--raw-root",
                "/r",
                "--prior-output",
                "/prior",
                "--freshness-audit",
                "/fresh.json",
                "--source-commit",
                "a" * 40,
            ]
        )
        self.assertEqual(register.freshness_audit, Path("/fresh.json"))
        self.assertNotIn("measured", inspect.signature(study.build_closed_trajectories).parameters)


if __name__ == "__main__":
    unittest.main()
