from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import corrected_renderer_attribution as base
from tools import recurrent_delta_spatial_confirmation as study
from tools import recurrent_delta_strict_postrun_audit as strict


def _selected_and_manifest():
    manifest = [
        {
            "clip_id": f"unused-{index}",
            "episode_dir": f"/unused/{index}",
            "frame_indices": list(range(0, 65, 5)),
        }
        for index in range(study.MANIFEST_COUNT)
    ]
    selected = []
    for position in range(study.SCORE_CLIPS):
        manifest_index = study.SCORE_START + position
        clip_id = f"clip-{position:02d}"
        manifest[manifest_index] = {
            "clip_id": clip_id,
            "episode_dir": f"/episodes/{position:02d}",
            "frame_indices": list(range(100, 165, 5)),
        }
        selected.append(
            {
                "clip_id": clip_id,
                "score_order": position,
                "manifest_index": manifest_index,
                "motion_stratum": 0 if position < 19 else (1 if position < 37 else 2),
            }
        )
    return selected, manifest


def _metric_payload(*, oracle: bool = False) -> dict[str, object]:
    epe = 0.0 if oracle else 1.0
    silhouette = 0.0 if oracle else 1.0
    iou = 1.0 if oracle else 0.9
    count = 20.0
    return {
        "silhouette_boundary_chamfer_px": silhouette,
        "silhouette_iou": iou,
        "rgb_robot_band_chamfer_px": 2.0,
        "rgb_robot_band_edge_support_3px": 0.8,
        "candidate_boundary_in_oracle_band_fraction": 0.9,
        "fixed_robot_band_fraction": 0.2,
        "fixed_robot_band_rgb_edge_pixels": 100.0,
        "robot_flow_epe_px": epe,
        "robot_flow_epe_p95_px": epe,
        "robot_flow_epe_sum_px": epe * count,
        "oracle_flow_magnitude_px": 2.0,
        "candidate_projection_valid_fraction": 1.0,
        "oracle_source_reprojection_rmse_px": 0.0,
        "oracle_source_sample_count": 100.0,
        "oracle_moving_flow_sample_count": count,
        "oracle_target_in_frame_fraction": 1.0,
        "flow_transition_scored": True,
    }


def _metric_rows():
    selected, manifest = _selected_and_manifest()
    rows = []
    for item in selected:
        frames = manifest[item["manifest_index"]]["frame_indices"][4:13]
        for transition in range(1, 9):
            for arm in study.ARMS:
                rows.append(
                    {
                        "schema_version": 1,
                        "clip_id": item["clip_id"],
                        "score_order": item["score_order"],
                        "motion_stratum": item["motion_stratum"],
                        "transition_ordinal": transition,
                        "source_frame_index": frames[transition - 1],
                        "target_frame_index": frames[transition],
                        "arm": arm,
                        "metrics": _metric_payload(oracle=arm == "measured_oracle"),
                        "split": "train",
                        "target_access_after_closure": True,
                        "validation_accessed": False,
                        "protected_test_accessed": False,
                    }
                )
    return rows, selected, manifest


class StrictPostrunAuditTest(unittest.TestCase):
    def test_report_destination_must_be_external_and_new(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / strict.EXPECTED_RUN_BASENAME
            root.mkdir()
            with self.assertRaises(strict.StrictAuditError):
                strict.validate_report_destination(root, root / "audit.json")
            report = Path(temporary) / "strict_audits" / "audit.json"
            canonical, destination = strict.validate_report_destination(root, report)
            self.assertEqual(canonical, root.resolve())
            self.assertEqual(destination, report.resolve())
            report.parent.mkdir()
            report.write_text("occupied")
            with self.assertRaises(strict.StrictAuditError):
                strict.validate_report_destination(root, report)

    def test_exclusive_report_writer_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            strict.write_json_exclusive(report, {"first": True})
            self.assertEqual(json.loads(report.read_text()), {"first": True})
            with self.assertRaises(strict.StrictAuditError):
                strict.write_json_exclusive(report, {"second": True})
            self.assertEqual(json.loads(report.read_text()), {"first": True})

    def test_record_verifier_detects_post_registration_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / strict.EXPECTED_RUN_BASENAME
            root.mkdir()
            value = root / "input.bin"
            value.write_bytes(b"registered")
            record = base.file_record(value)
            value.write_bytes(b"changed!!!")
            verifier = strict.RecordVerifier(root)
            with self.assertRaises(strict.StrictAuditError):
                verifier.verify(record, "input")

    def test_metric_rows_require_exact_clip_arm_transition_cartesian_product(self):
        rows, selected, manifest = _metric_rows()
        flags = strict.validate_metric_rows(rows, selected, manifest)
        self.assertEqual(flags, len(rows) * 2)
        corrupted = copy.deepcopy(rows)
        corrupted[-1]["transition_ordinal"] = 7
        with self.assertRaisesRegex(strict.StrictAuditError, "duplicate metric row"):
            strict.validate_metric_rows(corrupted, selected, manifest)

    def test_metric_rows_bind_source_and_target_frame_indices(self):
        rows, selected, manifest = _metric_rows()
        rows[0]["target_frame_index"] += 1
        with self.assertRaisesRegex(strict.StrictAuditError, "frame-index binding"):
            strict.validate_metric_rows(rows, selected, manifest)

    def test_metric_flow_mean_sum_count_must_agree(self):
        metrics = _metric_payload()
        metrics["robot_flow_epe_sum_px"] = 999.0
        with self.assertRaisesRegex(strict.StrictAuditError, "mean/sum/count"):
            strict._validate_metrics(metrics, "metric")

    def test_bootstrap_rejects_nonfrozen_seed_before_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bootstrap.npz"
            strata = np.asarray([0] * 19 + [1] * 18 + [2] * 18, dtype=np.int8)
            np.savez_compressed(
                path,
                indices=np.zeros((1, study.SCORE_CLIPS), dtype=np.int16),
                strata=strata,
                seed=np.asarray(study.BOOTSTRAP_SEED + 1, dtype=np.int64),
                samples=np.asarray(1, dtype=np.int64),
            )
            with self.assertRaisesRegex(strict.StrictAuditError, "bootstrap seed"):
                strict.validate_bootstrap(path, strata)

    def test_array_comparison_checks_exact_inputs_and_tolerant_replay(self):
        expected = np.ones((2, 3), dtype=np.float32)
        actual = expected.copy()
        actual[0, 0] += 1e-6
        error = strict._array_error(expected, actual, "replay", exact=False)
        self.assertGreater(error, 0.0)
        with self.assertRaises(strict.StrictAuditError):
            strict._array_error(expected, actual, "input", exact=True)
        actual[0, 0] += 1e-3
        with self.assertRaises(strict.StrictAuditError):
            strict._array_error(expected, actual, "replay", exact=False)

    def test_validation_access_flag_is_fail_closed(self):
        value = {
            "validation_accessed": False,
            "nested": {"protected_test_accessed": False},
        }
        self.assertEqual(strict._require_false_access_flags(value), 2)
        value["validation_accessed"] = True
        with self.assertRaisesRegex(strict.StrictAuditError, "validation_accessed"):
            strict._require_false_access_flags(value)


if __name__ == "__main__":
    unittest.main()
