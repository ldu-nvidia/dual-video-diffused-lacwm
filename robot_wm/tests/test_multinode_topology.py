import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.distributed_topology import validate_rank_topology


REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_RUNNER = REPO_ROOT / "tools" / "slurm" / "torchrun_node.sh"
NCCL_PROBE = REPO_ROOT / "tools" / "env" / "nccl_probe.py"
LAUNCHER = REPO_ROOT / "tools" / "launch_8xb200.sh"


def topology(nodes: int, gpus_per_node: int = 8):
    return [
        {
            "hostname": f"b200-node-{rank // gpus_per_node}",
            "rank": rank,
            "local_rank": rank % gpus_per_node,
        }
        for rank in range(nodes * gpus_per_node)
    ]


class DistributedTopologyTest(unittest.TestCase):
    def test_launcher_preserves_cold_nccl_and_wandb_identity(self):
        source = LAUNCHER.read_text()
        self.assertIn('source "$TOOLS_DIR/env/activate_b200.sh"', source)
        self.assertIn('--timeout-seconds 1200', source)
        self.assertIn(
            'timeout --signal=TERM --kill-after=60s 1260s "${NCCL_COMMAND[@]}"',
            source,
        )
        self.assertIn('"+wandb.id=$WANDB_RUN_ID"', source)
        for override in (
            '"trainer.config.max_iter=$MAX_ITER"',
            '"lr_scheduler_factory.lr_lambda.warmup_steps=$WARMUP_STEPS"',
            '"trainer.config.logging.log_every=$LOG_EVERY"',
            '"trainer.config.saving.save_every=$SAVE_EVERY"',
            '"trainer.config.validation.val_every=$VAL_EVERY"',
            '"trainer.config.visualization.viz_every=$VIZ_EVERY"',
        ):
            self.assertIn(override, source)

    def test_nccl_probe_rejects_unrealistic_payload_sizes_before_cuda(self):
        for payload in ("0", "1025"):
            with self.subTest(payload=payload):
                result = subprocess.run(
                    [sys.executable, str(NCCL_PROBE), "--payload-mib", payload],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("payload MiB must be between 1 and 1024", result.stdout)

    def test_accepts_one_to_thirty_two_dense_eight_gpu_nodes(self):
        for nodes in (1, 4, 16, 32):
            with self.subTest(nodes=nodes):
                result = validate_rank_topology(
                    topology(nodes), expected_nodes=nodes, gpus_per_node=8
                )
                self.assertEqual(len(result), nodes)
                self.assertTrue(
                    all(local_ranks == list(range(8)) for local_ranks in result.values())
                )

    def test_rejects_duplicate_or_missing_local_rank(self):
        records = topology(2)
        records[-1]["local_rank"] = 6
        with self.assertRaisesRegex(RuntimeError, "local ranks"):
            validate_rank_topology(records, expected_nodes=2, gpus_per_node=8)

    def test_rejects_wrong_host_count(self):
        records = topology(2)
        for record in records:
            record["hostname"] = "one-host"
        with self.assertRaisesRegex(RuntimeError, "spans 1 host"):
            validate_rank_topology(records, expected_nodes=2, gpus_per_node=8)

    def test_node_runner_renders_multinode_torchrun(self):
        env = dict(os.environ, SLURM_NODEID="31", SLURM_PROCID="31")
        result = subprocess.run(
            [
                "bash",
                str(NODE_RUNNER),
                "--python",
                sys.executable,
                "--nnodes",
                "32",
                "--nproc-per-node",
                "8",
                "--master-addr",
                "b200-node-0",
                "--master-port",
                "29402",
                "--rdzv-id",
                "job-42-train",
                "--dry-run",
                "--",
                "train.py",
                "name=test",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--nnodes=32", result.stdout)
        self.assertIn("--nproc_per_node=8", result.stdout)
        self.assertIn("--node_rank=31", result.stdout)
        self.assertIn("--rdzv_backend=c10d", result.stdout)
        self.assertIn("--rdzv_endpoint=b200-node-0:29402", result.stdout)
        self.assertIn("--rdzv_id=job-42-train", result.stdout)
        self.assertNotIn("--standalone", result.stdout)

    def test_node_runner_defaults_local_runtime_and_cuda_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            runtime.mkdir(mode=0o700)
            fake_python = root / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "test -n \"$CUDA_CACHE_PATH\"\n"
                "test -d \"$CUDA_CACHE_PATH\"\n"
                "test \"$NCCL_NVLS_ENABLE\" = 0\n"
                "test -n \"$XDG_RUNTIME_DIR\"\n"
                "test -d \"$XDG_RUNTIME_DIR\"\n"
                "test -w \"$XDG_RUNTIME_DIR\"\n"
                "test \"$XDG_RUNTIME_DIR\" = \"$TMPDIR\"\n"
                "case \"$CUDA_CACHE_PATH\" in *4242*) ;; *) exit 1 ;; esac\n"
            )
            fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
            env = dict(
                os.environ,
                SLURM_NODEID="0",
                SLURM_PROCID="0",
                SLURM_LOCALID="0",
                SLURM_JOB_ID="4242",
                XDG_RUNTIME_DIR=str(runtime),
            )
            env.pop("CUDA_CACHE_PATH", None)
            env.pop("NCCL_NVLS_ENABLE", None)

            result = subprocess.run(
                [
                    "bash",
                    str(NODE_RUNNER),
                    "--python",
                    str(fake_python),
                    "--nnodes",
                    "1",
                    "--nproc-per-node",
                    "1",
                    "--master-addr",
                    "localhost",
                    "--master-port",
                    "29402",
                    "--rdzv-id",
                    "cache-default-test",
                    "--",
                    "probe.py",
                ],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout)

    def test_node_runner_rejects_out_of_range_rank(self):
        env = dict(os.environ, SLURM_NODEID="2")
        result = subprocess.run(
            [
                "bash",
                str(NODE_RUNNER),
                "--python",
                sys.executable,
                "--nnodes",
                "2",
                "--master-addr",
                "b200-node-0",
                "--master-port",
                "29400",
                "--rdzv-id",
                "test",
                "--dry-run",
                "--",
                "probe.py",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("outside nnodes=2", result.stdout)

    def test_node_runner_rejects_inconsistent_slurm_rank_metadata(self):
        env = dict(
            os.environ,
            SLURM_NODEID="1",
            SLURM_PROCID="0",
            SLURM_LOCALID="0",
        )
        result = subprocess.run(
            [
                "bash",
                str(NODE_RUNNER),
                "--python",
                sys.executable,
                "--nnodes",
                "2",
                "--master-addr",
                "b200-node-0",
                "--master-port",
                "29400",
                "--rdzv-id",
                "test",
                "--dry-run",
                "--",
                "probe.py",
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("SLURM_NODEID == SLURM_PROCID", result.stdout)


if __name__ == "__main__":
    unittest.main()
