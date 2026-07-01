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


def allowed_run_root_env(root: Path, **updates: str) -> dict[str, str]:
    env = dict(
        os.environ,
        LACWM_ALLOWED_RUN_ROOTS=str(root.resolve(strict=False)),
    )
    env.update(updates)
    return env


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
            root = Path(temporary)
            result = subprocess.run(
                [str(SUBMIT), *self._training_args(root)],
                env=allowed_run_root_env(root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--ntasks=1", result.stdout)
        self.assertIn("--ntasks-per-node=1", result.stdout)
        self.assertIn("--gpus-per-node=8", result.stdout)
        self.assertIn("--export=ALL", result.stdout)
        self.assertIn("--signal=B:USR1@1200", result.stdout)
        self.assertIn("--max-requeues 12", result.stdout)
        self.assertIn("--batch-size 4", result.stdout)
        self.assertIn("--gradient-accumulation-steps 4", result.stdout)
        self.assertIn("--min-gpu-memory-mib 78000", result.stdout)
        self.assertIn("--max-iter 60000", result.stdout)
        self.assertIn("--warmup-steps 2000", result.stdout)
        self.assertIn("--log-every 50", result.stdout)
        self.assertIn("--save-every 1000", result.stdout)
        self.assertIn("--val-every 1000", result.stdout)
        self.assertIn("--viz-every 1000", result.stdout)
        self.assertNotIn("data_loader.batch_size", result.stdout)
        self.assertNotIn("trainer.config.max_iter", result.stdout)

    def test_submit_propagates_explicit_fast_validation_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization = root / "authorization.json"
            mixed = root / "mixed.json"
            authorization.write_text("{}\n", encoding="utf-8")
            mixed.write_text("{}\n", encoding="utf-8")
            result = subprocess.run(
                [
                    str(SUBMIT),
                    *self._training_args(root),
                    "--data-validation-policy",
                    "files_only_user_waived_v1",
                    "--fast-training-authorization",
                    str(authorization),
                    "--mixed-loader-report",
                    str(mixed),
                ],
                env=allowed_run_root_env(root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "--data-validation-policy files_only_user_waived_v1", result.stdout
        )
        self.assertIn(
            f"--fast-training-authorization {authorization}", result.stdout
        )
        self.assertIn(f"--mixed-loader-report {mixed}", result.stdout)

    def test_submit_strict_policy_rejects_fast_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization = root / "authorization.json"
            authorization.write_text("{}\n", encoding="utf-8")
            result = subprocess.run(
                [
                    str(SUBMIT),
                    *self._training_args(root),
                    "--fast-training-authorization",
                    str(authorization),
                ],
                env=allowed_run_root_env(root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("strict policy forbids fast authorization", result.stdout)

    def test_submit_dry_run_supports_one_to_thirty_two_nodes(self):
        for nodes in (1, 4, 16, 32):
            with self.subTest(nodes=nodes), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                result = subprocess.run(
                    [
                        str(SUBMIT),
                        *self._training_args(root),
                        "--nodes",
                        str(nodes),
                    ],
                    env=allowed_run_root_env(root),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn(f"--nodes={nodes}", result.stdout)
            self.assertIn(f"--ntasks={nodes}", result.stdout)
            self.assertIn("--ntasks-per-node=1", result.stdout)
            self.assertIn("--gpus-per-node=8", result.stdout)
            self.assertIn(
                f"nodes={nodes} gpus_per_node=8 world_size={nodes * 8}",
                result.stdout,
            )

    def test_submit_rejects_node_counts_outside_supported_range(self):
        for nodes in ("0", "33", "many"):
            with self.subTest(nodes=nodes), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                result = subprocess.run(
                    [
                        str(SUBMIT),
                        *self._training_args(root),
                        "--nodes",
                        nodes,
                    ],
                    env=allowed_run_root_env(root),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn("--nodes must be between 1 and 32", result.stdout)

    def test_submit_propagates_scaled_training_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run(
                [
                    str(SUBMIT),
                    *self._training_args(root),
                    "--nodes",
                    "16",
                    "--max-iter",
                    "7500",
                    "--warmup-steps",
                    "250",
                    "--log-every",
                    "6",
                    "--save-every",
                    "125",
                    "--val-every",
                    "125",
                    "--viz-every",
                    "125",
                ],
                env=allowed_run_root_env(root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("nodes=16 gpus_per_node=8 world_size=128", result.stdout)
        self.assertIn("--max-iter 7500", result.stdout)
        self.assertIn("--warmup-steps 250", result.stdout)
        self.assertIn("--log-every 6", result.stdout)
        self.assertIn("--save-every 125", result.stdout)
        self.assertIn("--val-every 125", result.stdout)
        self.assertIn("--viz-every 125", result.stdout)

    def test_submit_rejects_warmup_not_below_max_iter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run(
                [
                    str(SUBMIT),
                    *self._training_args(root),
                    "--max-iter",
                    "100",
                    "--warmup-steps",
                    "100",
                ],
                env=allowed_run_root_env(root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("--warmup-steps must be less than --max-iter", result.stdout)

    def test_submit_execute_accepts_parsable_job_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_sbatch = root / "sbatch"
            executable(fake_sbatch, """
                #!/usr/bin/env bash
                printf '4242;test-cluster\n'
            """)
            env = allowed_run_root_env(root, SBATCH_BIN=str(fake_sbatch))
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
            root = Path(temporary)
            args = self._training_args(root)
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
                env=allowed_run_root_env(root),
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
            root = Path(temporary)
            args = self._training_args(root)
            args[args.index("78000")] = "77999"
            result = subprocess.run(
                [str(SUBMIT), *args],
                env=allowed_run_root_env(root),
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
            executable(fake_bin / "flock", "#!/usr/bin/env bash\nexit 0\n")
            executable(
                fake_bin / "scontrol",
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > \"$TEST_SCONTROL_CALLED\"\n",
            )
            env = allowed_run_root_env(
                root,
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
            executable(fake_bin / "flock", "#!/usr/bin/env bash\nexit 0\n")
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
                    "max_iter": int(os.environ["TEST_MAX_ITER"]),
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
            env = allowed_run_root_env(
                root,
                PATH=f"{fake_bin}:{os.environ['PATH']}",
                SLURM_JOB_ID="321",
                SLURM_JOB_NUM_NODES="1",
                SLURM_NTASKS="1",
                SLURM_RESTART_COUNT="0",
                TEST_PYTHON=sys.executable,
                TEST_IDENTITY_SHA=digest,
                TEST_SNAPSHOT=str(snapshot),
                TEST_MAX_ITER="7500",
                TEST_SCONTROL_CALLED=str(called),
            )
            result = subprocess.run(
                [
                    str(SLOT),
                    *args,
                    "--max-iter",
                    "7500",
                    "--warmup-steps",
                    "250",
                    "--max-requeues",
                    "1",
                ],
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

    def test_submit_rejects_run_root_outside_default_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ)
            env.pop("LACWM_ALLOWED_RUN_ROOTS", None)
            result = subprocess.run(
                [str(SUBMIT), *self._training_args(root)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("outside allowed roots", result.stdout)

    def test_submit_canonicalizes_run_root_before_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args = self._training_args(root)
            run_root_index = args.index("--run-root") + 1
            args[run_root_index] = str(root / "unused" / ".." / "runs")
            result = subprocess.run(
                [str(SUBMIT), *args],
                env=allowed_run_root_env(root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            f"Planned persistent run directory: {root / 'runs' / 'resilient-test'}",
            result.stdout,
        )
        self.assertIn(f"--run-root {root / 'runs'}", result.stdout)

    def test_submit_rejects_unbounded_requeue_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run(
                [
                    str(SUBMIT),
                    *self._training_args(root),
                    "--max-requeues",
                    "101",
                ],
                env=allowed_run_root_env(root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("must not exceed 100", result.stdout)

    def test_slot_rejects_restart_count_above_cap_before_side_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = allowed_run_root_env(
                root,
                SLURM_RESTART_COUNT="2",
            )
            result = subprocess.run(
                [
                    str(SLOT),
                    *self._training_args(root),
                    "--max-requeues",
                    "1",
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("exceeds configured requeue cap 1", result.stdout)
        self.assertFalse((root / "runs").exists())


if __name__ == "__main__":
    unittest.main()
