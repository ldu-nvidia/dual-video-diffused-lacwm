import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMIT = REPO_ROOT / "tools" / "slurm" / "submit_8xb200.sh"
SLOT = REPO_ROOT / "tools" / "slurm" / "train_8xb200.sbatch"
STATE = REPO_ROOT / "tools" / "slurm" / "validate_state.py"


def executable(path: Path, source: str) -> None:
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class SlurmResilienceTest(unittest.TestCase):
    def _training_args(self, root: Path) -> list[str]:
        smoke = root / "smoke.json"
        data = root / "data.json"
        smoke.write_text("{}\n", encoding="utf-8")
        data.write_text("{}\n", encoding="utf-8")
        return [
            "--variant",
            "latent",
            "--python",
            sys.executable,
            "--wan-dir",
            str(root / "wan"),
            "--videox-home",
            str(root / "videox"),
            "--data-root",
            str(root / "data"),
            "--run-root",
            str(root / "runs"),
            "--run-name",
            "resilient-test",
            "--smoke-report",
            str(smoke),
            "--data-validation-report",
            str(data),
            "--wandb-mode",
            "disabled",
            "--batch-size",
            "4",
            "--gradient-accumulation-steps",
            "4",
            "--min-gpu-memory-mib",
            "78000",
        ]

    def test_submit_dry_run_has_one_task_and_no_training_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [str(SUBMIT), *self._training_args(Path(temporary))],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--ntasks=1", result.stdout)
        self.assertIn("--gpus-per-node=8", result.stdout)
        self.assertIn("--signal=B:USR1@1200", result.stdout)
        self.assertIn("--max-requeues 12", result.stdout)
        self.assertIn("--batch-size 4", result.stdout)
        self.assertIn("--gradient-accumulation-steps 4", result.stdout)
        self.assertIn("--min-gpu-memory-mib 78000", result.stdout)
        self.assertNotIn("data_loader.batch_size", result.stdout)
        self.assertNotIn("trainer.config.max_iter", result.stdout)

    def test_submit_execute_accepts_parsable_job_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_sbatch = root / "sbatch"
            executable(fake_sbatch, """
                #!/usr/bin/env bash
                printf '4242;test-cluster\n'
            """)
            env = dict(os.environ, SBATCH_BIN=str(fake_sbatch))
            result = subprocess.run(
                [str(SUBMIT), *self._training_args(root), "--execute"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Submitted Slurm job: 4242;test-cluster", result.stdout)

    def test_submit_uses_explicit_variant_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._training_args(Path(temporary))
            args[args.index("latent")] = "explicit"
            for option in (
                "--batch-size",
                "--gradient-accumulation-steps",
                "--min-gpu-memory-mib",
            ):
                index = args.index(option)
                del args[index : index + 2]
            result = subprocess.run(
                [str(SUBMIT), *args],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--batch-size 2", result.stdout)
        self.assertIn("--gradient-accumulation-steps 1", result.stdout)
        self.assertIn("--min-gpu-memory-mib 78000", result.stdout)

    def test_submit_rejects_unsafe_memory_floor(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._training_args(Path(temporary))
            args[args.index("78000")] = "77999"
            result = subprocess.run(
                [str(SUBMIT), *args],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("must be at least 78000", result.stdout)

    def test_state_validator_rejects_wrong_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            digest = "d" * 64
            identity = run_dir / "run_identity.json"
            identity.write_text(json.dumps({"identity_sha256": digest}))
            snapshot = run_dir / "snapshot.pt"
            snapshot.write_bytes(b"checkpoint")
            ack = root / "ack.json"
            ack.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "checkpointed_for_reschedule",
                        "checkpoint_written": True,
                        "next_iter": 10,
                        "max_iter": 60_000,
                        "run_identity_sha256": digest,
                        "slurm_attempt_id": "12.0",
                        "snapshot": str(snapshot),
                    }
                )
            )
            common = [
                sys.executable,
                str(STATE),
                "ack",
                "--path",
                str(ack),
                "--identity",
                str(identity),
                "--run-dir",
                str(run_dir),
            ]
            good = subprocess.run(
                [*common, "--attempt-id", "12.0"], capture_output=True, text=True
            )
            bad = subprocess.run(
                [*common, "--attempt-id", "12.1"], capture_output=True, text=True
            )

        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("slurm_attempt_id", bad.stderr)

    def test_training_failure_does_not_requeue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            called = root / "scontrol-called"
            executable(fake_bin / "srun", "#!/usr/bin/env bash\nexit 7\n")
            executable(
                fake_bin / "scontrol",
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > \"$TEST_SCONTROL_CALLED\"\n",
            )
            env = dict(
                os.environ,
                PATH=f"{fake_bin}:{os.environ['PATH']}",
                SLURM_JOB_ID="123",
                SLURM_JOB_NUM_NODES="1",
                SLURM_NTASKS="1",
                SLURM_RESTART_COUNT="0",
                TEST_SCONTROL_CALLED=str(called),
            )
            result = subprocess.run(
                [str(SLOT), *self._training_args(root)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            scontrol_was_called = called.exists()

        self.assertEqual(result.returncode, 7, result.stdout)
        self.assertFalse(scontrol_was_called)
        self.assertIn("not requeueing", result.stdout)

    def test_usr1_waits_for_ack_then_requeues(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._training_args(root)
            run_dir = root / "runs" / "resilient-test"
            run_dir.mkdir(parents=True)
            digest = "e" * 64
            (run_dir / "run_identity.json").write_text(
                json.dumps({"identity_sha256": digest}), encoding="utf-8"
            )
            snapshot = run_dir / "snapshot.pt"
            snapshot.write_bytes(b"checkpoint")

            fake_bin = root / "bin"
            fake_bin.mkdir()
            called = root / "scontrol-called"
            executable(
                fake_bin / "srun",
                """
                #!/usr/bin/env bash
                kill -USR1 "$PPID"
                for _ in $(seq 1 100); do
                  [[ -f "$LACWM_CHECKPOINT_REQUEST_FILE" ]] && break
                  sleep 0.05
                done
                "$TEST_PYTHON" - "$LACWM_CHECKPOINT_ACK_FILE" <<'PY'
                import json, os, pathlib, sys
                pathlib.Path(sys.argv[1]).write_text(json.dumps({
                    "schema_version": 1,
                    "status": "checkpointed_for_reschedule",
                    "checkpoint_written": True,
                    "next_iter": 10,
                    "max_iter": 60000,
                    "run_identity_sha256": os.environ["TEST_IDENTITY_SHA"],
                    "slurm_attempt_id": os.environ["LACWM_SLURM_ATTEMPT_ID"],
                    "snapshot": os.environ["TEST_SNAPSHOT"],
                }))
                PY
                exit 0
                """,
            )
            executable(
                fake_bin / "scontrol",
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > \"$TEST_SCONTROL_CALLED\"\n",
            )
            env = dict(
                os.environ,
                PATH=f"{fake_bin}:{os.environ['PATH']}",
                SLURM_JOB_ID="321",
                SLURM_JOB_NUM_NODES="1",
                SLURM_NTASKS="1",
                SLURM_RESTART_COUNT="0",
                TEST_PYTHON=sys.executable,
                TEST_IDENTITY_SHA=digest,
                TEST_SNAPSHOT=str(snapshot),
                TEST_SCONTROL_CALLED=str(called),
            )
            result = subprocess.run(
                [str(SLOT), *args, "--max-requeues", "1"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
                check=False,
            )

            called_text = called.read_text(encoding="utf-8") if called.exists() else ""

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Checkpoint-and-stop requested", result.stdout)
        self.assertIn("scheduler continuation acknowledged", result.stdout)
        self.assertEqual(called_text.strip(), "requeue 321")


if __name__ == "__main__":
    unittest.main()
