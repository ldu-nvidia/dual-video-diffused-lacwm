import subprocess
import unittest
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


if __name__ == "__main__":
    unittest.main()
