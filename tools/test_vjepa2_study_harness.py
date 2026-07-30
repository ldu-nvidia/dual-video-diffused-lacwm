"""Pure contract tests for the V-JEPA controlled-study launch tools."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import benchmark_vjepa2_inference as benchmark
import benchmark_vjepa2_paired_latency as paired_benchmark
import vjepa2_controlled_study as study


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "tools" / "slurm" / "vjepa2_controlled_study.sbatch"
SUBMIT = ROOT / "tools" / "slurm" / "submit_vjepa2_controlled_study.sh"
PAIRED_SBATCH = (
    ROOT / "tools" / "slurm" / "vjepa2_paired_latency.sbatch"
)


class VJEPA2StudyHarnessTest(unittest.TestCase):
    def test_five_arms_are_immutable_and_parameter_matched(self):
        self.assertEqual(
            [arm["code"] for arm in study.ARMS],
            ["V0", "VPM", "A1", "J0", "J1"],
        )
        dual = study.ARMS[1:]
        self.assertTrue(study.ARMS[1]["parameter_matched_control"])
        self.assertEqual(
            [arm["parameter_matched_control"] for arm in dual[1:]],
            [False, False, False],
        )
        self.assertTrue(all(arm["dual_enabled"] for arm in dual))

    def test_update_and_nfe_contracts_use_completed_update_semantics(self):
        self.assertEqual(
            study.COMPLETED_UPDATE_MILESTONES,
            (1, 50, 100, 200, 400, 800, 1000),
        )
        self.assertEqual(
            study.STAGE_ENDPOINTS,
            (1, 50, 100, 200, 400, 600, 800, 1000),
        )
        self.assertEqual(study.INFERENCE_NFE, (1, 2, 4, 6, 8, 12, 20))

    def test_batch_one_latency_excludes_shuffled_quality_control(self):
        self.assertEqual(benchmark.BATCH_SIZE, 1)
        self.assertEqual(benchmark.WARMUPS, 20)
        self.assertEqual(benchmark.REPETITIONS, 100)
        self.assertEqual(benchmark.ALLOWED_SOURCES, ("autonomous", "off"))
        self.assertNotIn("autonomous_shuffled", benchmark.ALLOWED_SOURCES)

    def test_final_latency_comparison_is_paired_and_counterbalanced(self):
        self.assertEqual(paired_benchmark.WARMUP_PAIRS, 20)
        self.assertEqual(paired_benchmark.TIMED_PAIRS, 100)
        self.assertEqual(
            paired_benchmark.ARM_SPECS["J1"]["nfe"], 4
        )
        self.assertEqual(
            paired_benchmark.ARM_SPECS["VPM"]["nfe"], 8
        )
        self.assertEqual(
            sum(
                paired_benchmark.counterbalanced_order(index)[0] == "J1"
                for index in range(100)
            ),
            50,
        )

    def test_timed_counter_contract_rejects_teacher_clean_and_artifacts(self):
        valid = {
            "wan_calls_by_source_nfe": {"autonomous:nfe_4": 4},
            "wan_calls_total": 4,
            "online_teacher_calls": 0,
            "auxiliary_clean_available": 0,
            "artifacts_collected": 0,
            "deployment_mode": 1,
        }
        normalized = benchmark._validated_sampler_counters(
            valid, nfe=4, artifacts_collected=0
        )
        self.assertEqual(normalized["wan_calls"], 4)
        for field in (
            "online_teacher_calls",
            "auxiliary_clean_available",
            "artifacts_collected",
        ):
            invalid = dict(valid)
            invalid[field] = 1
            with self.subTest(field=field):
                with self.assertRaises(benchmark.BenchmarkError):
                    benchmark._validated_sampler_counters(
                        invalid, nfe=4, artifacts_collected=0
                    )
        wrong_mode = dict(valid)
        wrong_mode["deployment_mode"] = 0
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark._validated_sampler_counters(
                wrong_mode, nfe=4, artifacts_collected=0
            )
        full_audit = dict(valid)
        full_audit["deployment_mode"] = 0
        normalized_full = benchmark._validated_sampler_counters(
            full_audit,
            nfe=4,
            artifacts_collected=0,
            deployment_mode=0,
        )
        self.assertEqual(normalized_full["deployment_mode"], 0)

    def test_stage_launcher_never_reapplies_warmstart_on_resume(self):
        source = SBATCH.read_text(encoding="utf-8")
        self.assertIn('if [[ "$STAGE_ENDPOINT" == "1" ]]', source)
        self.assertIn(
            'LOAD_PATH_OVERRIDE="trainer.config.load_path=null"', source
        )
        self.assertIn("dataset.img_augment=false", source)
        self.assertIn("val_dataset.img_augment=false", source)
        self.assertIn("viz_dataset.img_augment=false", source)
        self.assertNotIn("transform.wrist_mask_prob", source)

    def test_submit_preflight_binds_python_to_the_pinned_checkout(self):
        source = SUBMIT.read_text(encoding="utf-8")
        self.assertIn('source "$ACTIVATE"', source)
        self.assertIn('ROBOT_WM_ORIGIN="$(', source)
        self.assertIn('"$REPO_ROOT"/robot_wm/*', source)

    def test_v0_shell_contract_preserves_nullable_field_positions(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = study.command_arm_contract(
                SimpleNamespace(array_task_id=0, format="pipe")
            )
        self.assertEqual(status, 0)
        self.assertEqual(
            output.getvalue().rstrip("\n").split("|"),
            [
                "V0",
                "v0_original_explicit_action",
                "baseline",
                "false",
                "",
                "false",
                "false",
                "",
                "",
                "0.00",
                "false",
            ],
        )

    def test_stage_launcher_uses_non_whitespace_arm_delimiter(self):
        source = SBATCH.read_text(encoding="utf-8")
        self.assertIn("IFS='|' read -r", source)
        self.assertIn("--format pipe", source)
        self.assertNotIn("IFS=$'\\t' read -r", source)

    def test_submit_adds_dependent_one_gpu_paired_job(self):
        source = SUBMIT.read_text(encoding="utf-8")
        self.assertIn('--dependency="afterok:$PREVIOUS_JOB_ID"', source)
        self.assertIn("--gpus-per-node=1", source)
        self.assertIn("--paired-latency-job-id", source)
        self.assertIn('"$PAIRED_SBATCH_SCRIPT"', source)

    def test_submit_uses_selected_wall_time_for_stages_and_paired_job(self):
        source = SUBMIT.read_text(encoding="utf-8")
        self.assertIn('TIME_LIMIT="04:00:00"', source)
        self.assertNotIn("PAIRED_TIME_LIMIT", source)
        self.assertEqual(source.count('--time="$TIME_LIMIT"'), 2)

    def test_paired_job_runs_benchmark_then_final_analyzer(self):
        source = PAIRED_SBATCH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gpus-per-node=1", source)
        self.assertIn('"$PYTHON_BIN" "$BENCHMARK"', source)
        self.assertIn('"$PYTHON_BIN" "$ANALYZER"', source)
        self.assertLess(
            source.index('"$PYTHON_BIN" "$BENCHMARK"'),
            source.index('"$PYTHON_BIN" "$ANALYZER"'),
        )
        self.assertIn("--bootstrap-samples 10000", source)


if __name__ == "__main__":
    unittest.main()
