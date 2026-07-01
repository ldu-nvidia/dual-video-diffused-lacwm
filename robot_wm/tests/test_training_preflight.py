import hashlib
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import training_preflight


class TrainingPreflightGpuSafetyTest(unittest.TestCase):
    @staticmethod
    def _b200_gpus(memory_total_mib: int) -> list[dict[str, object]]:
        return [
            {
                "index": index,
                "uuid": f"GPU-test-{index}",
                "name": "NVIDIA B200",
                "memory_total_mib": memory_total_mib,
                "memory_free_mib": memory_total_mib - 100,
                "compute_mode": "Default",
            }
            for index in range(8)
        ]

    def test_failed_compute_process_query_is_not_treated_as_idle(self):
        failed = subprocess.CompletedProcess(
            ["nvidia-smi"], returncode=1, stdout="", stderr="NVML unavailable"
        )
        with mock.patch.object(training_preflight, "run_command", return_value=failed):
            apps, error = training_preflight.query_compute_apps()

        self.assertEqual(apps, [])
        self.assertIn("NVML unavailable", error)

    def test_smoke_fails_when_gpu_occupancy_cannot_be_queried(self):
        gpu = {
            "index": 0,
            "uuid": "GPU-test",
            "name": "NVIDIA B200",
            "memory_total_mib": 183000,
            "memory_free_mib": 180000,
            "compute_mode": "Default",
        }
        results = training_preflight.Results()
        with (
            mock.patch.object(
                training_preflight, "query_gpus", return_value=([gpu], "inventory")
            ),
            mock.patch.object(
                training_preflight,
                "query_compute_apps",
                return_value=([], "NVML unavailable"),
            ),
            mock.patch.object(
                training_preflight, "list_training_processes", return_value=[]
            ),
        ):
            training_preflight.check_gpus(results, [0], "smoke")

        self.assertFalse(results.passed)
        checks = {item.name: item for item in results.checks}
        self.assertFalse(checks["GPU compute-process query"].ok)
        self.assertFalse(checks["selected GPUs idle"].ok)

    def test_full_profile_accepts_80_gib_class_b200(self):
        results = training_preflight.Results()
        gpus = self._b200_gpus(81_920)
        with (
            mock.patch.object(
                training_preflight, "query_gpus", return_value=(gpus, "inventory")
            ),
            mock.patch.object(
                training_preflight, "query_compute_apps", return_value=([], None)
            ),
            mock.patch.object(
                training_preflight, "list_training_processes", return_value=[]
            ),
        ):
            training_preflight.check_gpus(
                results, list(range(8)), "full", min_gpu_memory_mib=78_000
            )

        checks = {item.name: item for item in results.checks}
        self.assertTrue(checks["B200 model enforcement"].ok)
        self.assertTrue(checks["B200 memory capacity"].ok)
        self.assertTrue(results.passed)

    def test_full_profile_enforces_requested_memory_threshold(self):
        results = training_preflight.Results()
        gpus = self._b200_gpus(81_920)
        with (
            mock.patch.object(
                training_preflight, "query_gpus", return_value=(gpus, "inventory")
            ),
            mock.patch.object(
                training_preflight, "query_compute_apps", return_value=([], None)
            ),
            mock.patch.object(
                training_preflight, "list_training_processes", return_value=[]
            ),
        ):
            training_preflight.check_gpus(
                results, list(range(8)), "full", min_gpu_memory_mib=90_000
            )

        checks = {item.name: item for item in results.checks}
        self.assertFalse(checks["B200 memory capacity"].ok)
        self.assertIn("minimum=90000 MiB", checks["B200 memory capacity"].detail)
        self.assertFalse(results.passed)

    def test_memory_profile_rejects_unsafe_floor(self):
        parser = training_preflight.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--min-gpu-memory-mib", "77999"])


class TrainingPreflightDatasetSelectionTest(unittest.TestCase):
    @staticmethod
    def _args() -> SimpleNamespace:
        return SimpleNamespace(
            datasets=["droid", "egodex"],
            data_validation_policy=training_preflight.STRICT_DATA_POLICY,
            max_data_report_age_hours=24.0,
            data_validation_workers=1,
            min_droid=10_000,
            min_egodex=10_000,
            min_agibot=5_671,
            min_abc=10_000,
        )

    def test_subset_report_requires_exact_names_sources_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            data_root.mkdir()
            report_path = root / "report.json"
            sources = {
                "droid": str(data_root / "droid_lerobot"),
                "egodex": str(data_root / "egodex_cdn" / "manifest.csv"),
            }
            fingerprints = {
                name: {"digest": name} for name in ("droid", "egodex")
            }
            payload = {
                "schema_version": 2,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "read_only": True,
                "passed": True,
                "git_commit": "commit",
                "git_status": "",
                "validator_sha256": hashlib.sha256(
                    (
                        training_preflight.REPO_ROOT
                        / "tools"
                        / "validate_training_data.py"
                    ).read_bytes()
                ).hexdigest(),
                "invocation": {
                    "data_root": str(data_root.resolve(strict=False)),
                    "datasets": ["droid", "egodex"],
                    "min_timesteps": 66,
                    "validate_all": False,
                    "files_only": False,
                    "caps": {"droid": 10_000, "egodex": 10_000},
                    "expected": {"droid": 10_000, "egodex": 10_000},
                    "sources": sources,
                },
                "reports": [
                    {
                        "name": name,
                        "source": sources[name],
                        "checked": 10_000,
                        "selected": 10_000,
                        "active_complete": 10_000,
                        "error_count": 0,
                        "fingerprint": fingerprints[name],
                    }
                    for name in ("droid", "egodex")
                ],
            }
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            commit_result = subprocess.CompletedProcess(
                ["git"], returncode=0, stdout="commit\n", stderr=""
            )

            results = training_preflight.Results()
            with (
                mock.patch.object(
                    training_preflight, "run_command", return_value=commit_result
                ),
                mock.patch.object(
                    training_preflight,
                    "_current_data_fingerprints",
                    return_value=fingerprints,
                ) as current,
            ):
                training_preflight.check_strict_data_report(
                    results,
                    report_path,
                    data_root,
                    Path("/tmp/python"),
                    self._args(),
                )
            self.assertTrue(results.passed)
            current.assert_called_once_with(data_root, ("droid", "egodex"))

            payload["reports"].append(
                {
                    "name": "abc",
                    "source": str(data_root / "abc_pp" / "manifest.txt"),
                    "checked": 10_000,
                    "selected": 10_000,
                    "active_complete": 10_000,
                    "error_count": 0,
                    "fingerprint": {"digest": "abc"},
                }
            )
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            results = training_preflight.Results()
            with mock.patch.object(
                training_preflight, "run_command", return_value=commit_result
            ):
                training_preflight.check_strict_data_report(
                    results,
                    report_path,
                    data_root,
                    Path("/tmp/python"),
                    self._args(),
                )
            self.assertFalse(results.passed)

    def test_fast_files_only_report_requires_exact_external_manifest_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "overlay"
            data_root.mkdir()
            report_path = root / "files-only.json"
            sources = {
                "droid": str(data_root / "droid_lerobot"),
                "egodex": str(data_root / "egodex_cdn" / "manifest.csv"),
            }
            fingerprints = {
                name: {"digest": name} for name in ("droid", "egodex")
            }
            payload = {
                "schema_version": 2,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "read_only": True,
                "passed": True,
                "git_commit": "commit",
                "git_status": "",
                "validator_sha256": hashlib.sha256(
                    (
                        training_preflight.REPO_ROOT
                        / "tools"
                        / "validate_training_data.py"
                    ).read_bytes()
                ).hexdigest(),
                "invocation": {
                    "data_root": str(data_root.resolve()),
                    "datasets": ["droid", "egodex"],
                    "min_timesteps": 66,
                    "validate_all": False,
                    "files_only": True,
                    "caps": {"droid": 10_000, "egodex": 10_000},
                    "expected": {"droid": 10_000, "egodex": 10_000},
                    "sources": sources,
                    "allowed_external_roots": {
                        "egodex": str((data_root.parent / "egodex_cdn").resolve()),
                        "abc": str((data_root.parent / "abc_pp").resolve()),
                    },
                },
                "reports": [
                    {
                        "name": name,
                        "source": sources[name],
                        "checked": 0,
                        "selected": 10_000,
                        "active_complete": 10_000,
                        "complete": 10_000,
                        "error_count": 0,
                        "fingerprint": fingerprints[name],
                    }
                    for name in ("droid", "egodex")
                ],
            }
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            args = self._args()
            args.data_validation_policy = training_preflight.FAST_DATA_POLICY
            commit_result = subprocess.CompletedProcess(
                ["git"], returncode=0, stdout="commit\n", stderr=""
            )

            results = training_preflight.Results()
            with (
                mock.patch.object(
                    training_preflight, "run_command", return_value=commit_result
                ),
                mock.patch.object(
                    training_preflight,
                    "_current_data_fingerprints",
                    return_value=fingerprints,
                ) as current,
            ):
                training_preflight.check_strict_data_report(
                    results,
                    report_path,
                    data_root,
                    Path("/tmp/python"),
                    args,
                )
            self.assertTrue(results.passed)
            current.assert_called_once_with(
                data_root,
                ("droid", "egodex"),
                agibot_profile="qualification",
            )

            payload["invocation"]["allowed_external_roots"]["egodex"] = str(
                root / "wrong"
            )
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            results = training_preflight.Results()
            with (
                mock.patch.object(
                    training_preflight, "run_command", return_value=commit_result
                ),
                mock.patch.object(
                    training_preflight,
                    "_current_data_fingerprints",
                    return_value=fingerprints,
                ),
            ):
                training_preflight.check_strict_data_report(
                    results,
                    report_path,
                    data_root,
                    Path("/tmp/python"),
                    args,
                )
            self.assertFalse(results.passed)

    def test_check_data_skips_unselected_datasets(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            results = training_preflight.Results()
            args = self._args()
            with (
                mock.patch.object(training_preflight, "check_droid") as droid,
                mock.patch.object(training_preflight, "check_egodex") as egodex,
                mock.patch.object(training_preflight, "check_agibot") as agibot,
                mock.patch.object(training_preflight, "check_abc") as abc,
            ):
                training_preflight.check_data(
                    results, data_root, "smoke", args, Path("/tmp/python")
                )
            droid.assert_called_once_with(results, data_root, 1)
            egodex.assert_called_once_with(results, data_root, 1)
            agibot.assert_not_called()
            abc.assert_not_called()

    def test_duplicate_dataset_names_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            training_preflight.normalized_dataset_names(["droid", "droid"])


if __name__ == "__main__":
    unittest.main()
