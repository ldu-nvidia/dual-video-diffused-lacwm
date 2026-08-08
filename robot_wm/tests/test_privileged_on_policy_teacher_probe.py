import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "probe_privileged_on_policy_teacher.py"
)
SPEC = importlib.util.spec_from_file_location("privileged_teacher_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(probe)


def _unit_row(clip: int, step: int, student: float, aligned: float, shuffled: float):
    return probe.identity_payload(
        {
            "schema": probe.ROW_SCHEMA,
            "clip_index": clip,
            "clip_id": f"clip-{clip}",
            "episode_dir": f"episode-{clip}",
            "shuffled_donor_episode_dir": f"episode-{clip ^ 1}",
            "rollout_step": step,
            "state_is_student_visited": True,
            "video_phase_active": True,
            "clean_future_feature_teacher_only": True,
            "aligned_feature_differs_from_shuffled": True,
            "protected_test_opened": False,
            "velocity_mse": {
                "student_off": student,
                "teacher_aligned": aligned,
                "teacher_shuffled": shuffled,
            },
        }
    )


def _rollout_row(clip: int, source: str, value: float):
    return probe.identity_payload(
        {
            "schema": probe.ROLLOUT_ROW_SCHEMA,
            "clip_index": clip,
            "source": source,
            "oracle_leakage": source != "off",
            "deployable_evidence": False,
            "protected_test_opened": False,
            "metrics": {metric: value for metric in probe.METRICS},
        }
    )


class ProbeAnalysisTest(unittest.TestCase):
    def test_high_noise_profile_is_frozen_to_disjoint_nfe2_slice(self):
        contract = probe.probe_profile_contract(
            probe.HIGH_NOISE_NFE2_PROFILE, 2, 64
        )
        self.assertEqual(contract["expected_video_active_steps"], 1)
        self.assertEqual(contract["expected_active_video_sigmas"], [1.0])
        with self.assertRaisesRegex(probe.ProbeError, "indices 64--127"):
            probe.probe_profile_contract(probe.HIGH_NOISE_NFE2_PROFILE, 2, 0)
        with self.assertRaisesRegex(probe.ProbeError, "NFE=2"):
            probe.probe_profile_contract(probe.HIGH_NOISE_NFE2_PROFILE, 4, 64)

    def test_legacy_nfe4_artifact_remains_auditable(self):
        profile = probe.validate_recorded_profile(
            {"nfe": 4}, {"observed_video_active_steps": 2}
        )
        self.assertEqual(profile["profile"], probe.ORIGINAL_PROFILE)
        self.assertTrue(profile["legacy_a375870_artifact"])
        with self.assertRaisesRegex(probe.ProbeError, "video-active count"):
            probe.validate_recorded_profile(
                {"nfe": 4}, {"observed_video_active_steps": 1}
            )

    def test_identity_detects_mutation(self):
        payload = probe.identity_payload({"value": 1})
        self.assertTrue(probe.identity_valid(payload))
        payload["value"] = 2
        self.assertFalse(probe.identity_valid(payload))

    def test_known_favorable_teacher_passes_all_gates(self):
        units = [
            _unit_row(clip, step, student=10.0, aligned=8.0, shuffled=9.0)
            for clip in range(64, 72)
            for step in (2, 3)
        ]
        rollouts = []
        for clip in range(64, 72):
            rollouts.extend(
                (
                    _rollout_row(clip, "off", 10.0),
                    _rollout_row(clip, "oracle_matched", 8.0),
                    _rollout_row(clip, "oracle_shuffled", 9.0),
                )
            )
        analysis = probe.analyze_rows(
            units,
            rollouts,
            num_clips=8,
            active_steps=2,
            seed=7,
            replicates=1_000,
        )
        self.assertEqual(analysis["decision"], "ELIGIBLE_FOR_STUDENT_SCREEN")
        self.assertEqual(analysis["teacher_better_unit_fraction"], 1.0)
        self.assertAlmostEqual(
            analysis["velocity_effects"]["aligned_vs_student_off"][
                "relative_improvement_percent"
            ],
            20.0,
        )
        self.assertTrue(analysis["eligibility_gates"]["all_passed"])
        self.assertTrue(probe.identity_valid(analysis))

    def test_non_sample_specific_unit_fails_closed(self):
        units = [
            _unit_row(clip, step, student=10.0, aligned=8.0, shuffled=9.0)
            for clip in range(8)
            for step in (2, 3)
        ]
        units[0] = probe.identity_payload(
            {**units[0], "aligned_feature_differs_from_shuffled": False}
        )
        rollouts = [
            _rollout_row(clip, source, 10.0)
            for clip in range(8)
            for source in probe.SOURCES
        ]
        with self.assertRaisesRegex(probe.ProbeError, "sample-specific"):
            probe.analyze_rows(
                units,
                rollouts,
                num_clips=8,
                active_steps=2,
                seed=7,
                replicates=1_000,
            )


class ManifestContractTest(unittest.TestCase):
    def _write_manifest(self, path: Path, split: str = "train") -> None:
        with path.open("w", encoding="utf-8") as handle:
            for index in range(128):
                handle.write(
                    json.dumps(
                        {
                            "split": split if index == 0 else "train",
                            "auxiliary_index": index,
                            "clip_id": f"clip-{index}",
                            "episode_dir": f"episode-{index}",
                        }
                    )
                    + "\n"
                )

    def test_first_eight_are_unique_train_items(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.jsonl"
            self._write_manifest(manifest)
            selected = probe._manifest_selection(manifest, 8)
            self.assertEqual(
                [row["auxiliary_index"] for row in selected], list(range(8))
            )

    def test_disjoint_followup_slice_uses_indices_64_through_127(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.jsonl"
            self._write_manifest(manifest)
            selected = probe._manifest_selection(
                manifest,
                64,
                start_index=64,
                disjoint_from_index_range=(0, 64),
            )
            self.assertEqual(
                [row["auxiliary_index"] for row in selected], list(range(64, 128))
            )

    def test_followup_rejects_episode_overlap_with_parent_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.jsonl"
            self._write_manifest(manifest)
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            rows[64]["episode_dir"] = rows[0]["episode_dir"]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(probe.ProbeError, "overlap parent"):
                probe._manifest_selection(
                    manifest,
                    64,
                    start_index=64,
                    disjoint_from_index_range=(0, 64),
                )

    def test_nontrain_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.jsonl"
            self._write_manifest(manifest, split="val")
            with self.assertRaisesRegex(probe.ProbeError, "not train-scoped"):
                probe._manifest_selection(manifest, 8)


if __name__ == "__main__":
    unittest.main()
