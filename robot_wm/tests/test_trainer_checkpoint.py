import io
import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from robot_wm.utils.trainer import Trainer


class TrainerCheckpointTest(unittest.TestCase):
    def test_single_rank_state_does_not_use_object_collective(self):
        trainer = Trainer.__new__(Trainer)
        trainer.world_size = 1
        trainer.global_rank = 0
        trainer._capture_rng_state = lambda: {"torch_cpu": torch.tensor([1])}
        trainer.data_loader = mock.Mock()
        trainer.data_loader.state_dict.return_value = {"cursor": torch.tensor([2])}
        trainer.val_data_loaders = []
        trainer.viz_data_loaders = []

        with mock.patch(
            "robot_wm.utils.trainer.dist.is_initialized", return_value=True
        ), mock.patch("torch.distributed.all_gather_object") as gather:
            states = trainer._gather_rank_states()

        gather.assert_not_called()
        self.assertEqual(states[0]["data_loader"]["cursor"].item(), 2)

    def test_multi_rank_state_collective_gathers_serialized_bytes(self):
        trainer = Trainer.__new__(Trainer)
        trainer.world_size = 2
        trainer.global_rank = 0
        trainer._capture_rng_state = lambda: {"torch_cpu": torch.tensor([1])}
        trainer.data_loader = mock.Mock()
        trainer.data_loader.state_dict.return_value = {"cursor": torch.tensor([2])}
        trainer.val_data_loaders = []
        trainer.viz_data_loaders = []

        def gather(payloads, local_payload):
            self.assertIsInstance(local_payload, bytes)
            decoded = torch.load(
                io.BytesIO(local_payload), map_location="cpu", weights_only=True
            )
            self.assertEqual(decoded["data_loader"]["cursor"].item(), 2)
            payloads[:] = [local_payload, local_payload]

        with mock.patch(
            "robot_wm.utils.trainer.dist.is_initialized", return_value=True
        ), mock.patch("torch.distributed.all_gather_object", side_effect=gather):
            states = trainer._gather_rank_states()

        self.assertEqual(len(states), 2)
        self.assertEqual(states[1]["rng"]["torch_cpu"].item(), 1)

    def test_milestone_also_updates_live_snapshot(self):
        trainer = Trainer.__new__(Trainer)
        trainer.is_main_process = True
        trainer._curr_iter = 10_000
        trainer._gather_rank_states = lambda: [{"rank": 0}]
        trainer._build_snapshot = lambda rank_states: {
            "rank_states": rank_states,
            "value": torch.tensor([7]),
        }
        with tempfile.TemporaryDirectory() as temporary:
            trainer.save_path = Path(temporary) / "snapshot.pt"
            trainer._save_snapshot()
            live = trainer.save_path
            archive = trainer.save_path.with_suffix(".10000.pt")
            self.assertTrue(live.is_file())
            self.assertTrue(archive.is_file())
            self.assertEqual(
                torch.load(live, weights_only=True)["value"].item(), 7
            )

    def test_iteration_zero_does_not_duplicate_live_snapshot(self):
        trainer = Trainer.__new__(Trainer)
        trainer.is_main_process = True
        trainer._curr_iter = 0
        trainer.save_path = Path("snapshot.pt")
        trainer._gather_rank_states = lambda: [{"rank": 0}]
        trainer._build_snapshot = lambda rank_states: {"rank_states": rank_states}
        trainer._atomic_save_snapshot = mock.Mock()

        with mock.patch(
            "robot_wm.utils.trainer.dist.is_initialized", return_value=False
        ):
            trainer._save_snapshot()

        trainer._atomic_save_snapshot.assert_called_once_with(
            {"rank_states": [{"rank": 0}]}, trainer.save_path
        )

    def test_snapshot_carries_guarded_run_identity(self):
        trainer = Trainer.__new__(Trainer)
        trainer.model = mock.Mock()
        trainer.model.module.state_dict.return_value = {}
        trainer.optimizer = mock.Mock()
        trainer.optimizer.state_dict.return_value = {}
        trainer.lr_scheduler = mock.Mock()
        trainer.lr_scheduler.state_dict.return_value = {}
        trainer._curr_iter = 4
        trainer.metrics = mock.Mock(total_observations=8, best_val_loss=1.0)
        trainer.world_size = 8
        trainer.use_amp = False
        trainer.wandb_run_id = None
        trainer.run_identity_sha256 = "a" * 64
        trainer.gradient_accumulation_steps = 4

        snapshot = trainer._build_snapshot([{"rank": 0}])
        self.assertEqual(snapshot["run_identity_sha256"], "a" * 64)
        self.assertEqual(snapshot["gradient_accumulation_steps"], 4)

    def test_resume_rejects_checkpoint_from_another_run(self):
        trainer = Trainer.__new__(Trainer)
        trainer.local_rank = 0
        trainer.run_identity_sha256 = "a" * 64
        trainer.gradient_accumulation_steps = 1
        with tempfile.TemporaryDirectory() as temporary:
            trainer.save_path = Path(temporary) / "snapshot.pt"
            torch.save(
                {
                    "snapshot_schema_version": 3,
                    "run_identity_sha256": "b" * 64,
                },
                trainer.save_path,
            )
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                trainer._load_snapshot()

    def test_resume_rejects_gradient_accumulation_change(self):
        trainer = Trainer.__new__(Trainer)
        trainer.local_rank = 0
        trainer.run_identity_sha256 = None
        trainer.gradient_accumulation_steps = 4
        with tempfile.TemporaryDirectory() as temporary:
            trainer.save_path = Path(temporary) / "snapshot.pt"
            torch.save(
                {
                    "snapshot_schema_version": 3,
                    "gradient_accumulation_steps": 2,
                },
                trainer.save_path,
            )
            with self.assertRaisesRegex(
                RuntimeError, "gradient-accumulation mismatch"
            ):
                trainer._load_snapshot()

    def test_resume_rejects_world_size_before_mutating_training_state(self):
        world_sizes = (8, 16, 24, 32)
        for index, current_world_size in enumerate(world_sizes):
            saved_world_size = world_sizes[(index + 1) % len(world_sizes)]
            with self.subTest(
                current=current_world_size, saved=saved_world_size
            ):
                trainer = Trainer.__new__(Trainer)
                trainer.local_rank = 0
                trainer.global_rank = 0
                trainer.world_size = current_world_size
                trainer.run_identity_sha256 = None
                trainer.gradient_accumulation_steps = 4
                trainer.model = mock.Mock()
                trainer.optimizer = mock.Mock()
                trainer.lr_scheduler = mock.Mock()
                trainer.metrics = mock.Mock()
                trainer.use_amp = False
                with tempfile.TemporaryDirectory() as temporary:
                    trainer.save_path = Path(temporary) / "snapshot.pt"
                    torch.save(
                        {
                            "snapshot_schema_version": 3,
                            "gradient_accumulation_steps": 4,
                            "world_size": saved_world_size,
                            "rank_states": [
                                {"global_rank": rank}
                                for rank in range(saved_world_size)
                            ],
                            "model": {},
                            "optimizer": {},
                            "lr_scheduler": {},
                        },
                        trainer.save_path,
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, f"saved_world_size={saved_world_size}"
                    ):
                        trainer._load_snapshot()

                trainer.model.module.load_state_dict.assert_not_called()
                trainer.optimizer.load_state_dict.assert_not_called()
                trainer.lr_scheduler.load_state_dict.assert_not_called()

    def test_resume_rejects_invalid_rank_state_cardinality_and_order(self):
        for world_size in (8, 16, 24, 32):
            valid_states = [
                {"global_rank": rank} for rank in range(world_size)
            ]
            cases = {
                "missing": valid_states[:-1],
                "extra": [*valid_states, {"global_rank": world_size}],
                "reordered": [valid_states[1], valid_states[0], *valid_states[2:]],
            }
            for case_name, rank_states in cases.items():
                with self.subTest(world_size=world_size, case=case_name):
                    trainer = Trainer.__new__(Trainer)
                    trainer.local_rank = 0
                    trainer.global_rank = 0
                    trainer.world_size = world_size
                    trainer.run_identity_sha256 = None
                    trainer.gradient_accumulation_steps = 4
                    trainer.model = mock.Mock()
                    trainer.optimizer = mock.Mock()
                    trainer.lr_scheduler = mock.Mock()
                    trainer.metrics = mock.Mock()
                    trainer.use_amp = False
                    with tempfile.TemporaryDirectory() as temporary:
                        trainer.save_path = Path(temporary) / "snapshot.pt"
                        torch.save(
                            {
                                "snapshot_schema_version": 3,
                                "gradient_accumulation_steps": 4,
                                "world_size": world_size,
                                "rank_states": rank_states,
                                "model": {},
                                "optimizer": {},
                                "lr_scheduler": {},
                            },
                            trainer.save_path,
                        )
                        with self.assertRaisesRegex(
                            RuntimeError, "rank states|rank_states"
                        ):
                            trainer._load_snapshot()

                    trainer.model.module.load_state_dict.assert_not_called()
                    trainer.optimizer.load_state_dict.assert_not_called()
                    trainer.lr_scheduler.load_state_dict.assert_not_called()

    def test_signal_handler_only_records_checkpoint_intent(self):
        trainer = Trainer.__new__(Trainer)
        trainer._checkpoint_stop_requested = False
        trainer._checkpoint_stop_signal = None

        trainer._checkpoint_signal_handler(signal.SIGUSR1, None)

        self.assertTrue(trainer._checkpoint_stop_requested)
        self.assertEqual(trainer._checkpoint_stop_signal, int(signal.SIGUSR1))

    def test_request_file_is_polled_by_rank_zero(self):
        trainer = Trainer.__new__(Trainer)
        trainer._checkpoint_stop_requested = False
        trainer.is_main_process = True
        trainer.world_size = 1
        with tempfile.TemporaryDirectory() as temporary:
            trainer.checkpoint_request_path = Path(temporary) / "stop.request"
            trainer.checkpoint_request_path.touch()
            with mock.patch(
                "robot_wm.utils.trainer.dist.is_initialized", return_value=False
            ):
                self.assertTrue(trainer._checkpoint_stop_requested_across_ranks())

    def test_one_rank_request_is_reduced_across_all_ranks(self):
        trainer = Trainer.__new__(Trainer)
        trainer._checkpoint_stop_requested = False
        trainer.is_main_process = False
        trainer.checkpoint_request_path = None
        trainer.world_size = 2
        trainer.local_rank = 1

        def remote_rank_requests_stop(vote, op):
            self.assertEqual(op, torch.distributed.ReduceOp.MAX)
            self.assertEqual(vote.device.type, "cpu")
            vote.fill_(1)

        with mock.patch(
            "robot_wm.utils.trainer.dist.is_initialized", return_value=True
        ), mock.patch(
            "torch.distributed.get_backend", return_value="gloo"
        ), mock.patch(
            "torch.distributed.all_reduce", side_effect=remote_rank_requests_stop
        ):
            self.assertTrue(trainer._checkpoint_stop_requested_across_ranks())

    def test_checkpoint_ack_and_completion_markers_are_atomic_json(self):
        trainer = Trainer.__new__(Trainer)
        trainer.is_main_process = True
        trainer.max_iter = 60_000
        trainer.run_identity_sha256 = "c" * 64
        trainer._checkpoint_stop_signal = int(signal.SIGUSR1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer.save_path = root / "snapshot.pt"
            trainer.save_path.touch()
            trainer.checkpoint_ack_path = root / "control" / "attempt.ack.json"
            trainer.completion_path = root / "training_complete.json"

            with mock.patch.dict(
                "os.environ", {"LACWM_SLURM_ATTEMPT_ID": "42.3"}
            ):
                trainer._write_checkpoint_ack(checkpoint_written=True, next_iter=1234)
            trainer._write_completion_marker()

            ack = json.loads(trainer.checkpoint_ack_path.read_text())
            complete = json.loads(trainer.completion_path.read_text())
            self.assertEqual(ack["status"], "checkpointed_for_reschedule")
            self.assertEqual(ack["next_iter"], 1234)
            self.assertEqual(ack["slurm_attempt_id"], "42.3")
            self.assertEqual(complete["status"], "completed")
            self.assertEqual(complete["completed_updates"], 60_000)
            self.assertFalse((trainer.checkpoint_ack_path.parent / "attempt.ack.json.tmp").exists())

    def test_checkpoint_write_failure_is_raised_before_ack(self):
        trainer = Trainer.__new__(Trainer)
        trainer.is_main_process = True
        trainer.world_size = 1
        trainer._curr_iter = 4
        trainer._gather_rank_states = lambda: [{"rank": 0}]
        trainer._build_snapshot = lambda rank_states: {"rank_states": rank_states}
        trainer._atomic_save_snapshot = mock.Mock(side_effect=OSError("disk full"))
        with tempfile.TemporaryDirectory() as temporary:
            trainer.save_path = Path(temporary) / "snapshot.pt"
            with mock.patch(
                "robot_wm.utils.trainer.dist.is_initialized", return_value=False
            ):
                with self.assertRaisesRegex(RuntimeError, "disk full"):
                    trainer._save_snapshot()

    def test_reschedule_finishes_one_iteration_then_checkpoints(self):
        trainer = Trainer.__new__(Trainer)
        trainer.model = mock.Mock()
        trainer._start_iter = 1
        trainer._curr_iter = 0
        trainer.max_iter = 4
        trainer.log_every = 1000
        trainer.val_every = 1000
        trainer.viz_every = 1000
        trainer.save_every = 1000
        trainer.viz_data_loaders = None
        trainer._step = mock.Mock(return_value={"loss": 1.0})
        trainer._save_snapshot = mock.Mock()
        trainer._write_checkpoint_ack = mock.Mock()
        trainer._write_completion_marker = mock.Mock()
        trainer._install_checkpoint_signal_handlers = mock.Mock()
        trainer._restore_checkpoint_signal_handlers = mock.Mock()
        trainer._checkpoint_stop_requested_across_ranks = mock.Mock(
            side_effect=[False, True]
        )

        outcome = trainer.train()

        self.assertEqual(outcome, "rescheduled")
        trainer._step.assert_called_once_with()
        trainer._save_snapshot.assert_called_once_with()
        trainer._write_checkpoint_ack.assert_called_once_with(
            checkpoint_written=True, next_iter=2
        )
        trainer._write_completion_marker.assert_not_called()

    def test_periodic_cadence_and_final_checkpoint_are_preserved(self):
        trainer = Trainer.__new__(Trainer)
        trainer.model = mock.Mock()
        trainer._start_iter = 0
        trainer._curr_iter = 0
        trainer.max_iter = 2_502
        trainer.log_every = 10_000
        trainer.val_every = 10_000
        trainer.viz_every = 10_000
        trainer.save_every = 1_000
        trainer.viz_data_loaders = None
        trainer.optimizer = mock.Mock(param_groups=[{"lr": 1e-4}])
        trainer.metrics = mock.Mock()
        trainer.metrics.get_train_metrics.return_value = {}
        trainer.metrics.get_val_metrics.return_value = {}
        trainer._step = mock.Mock(return_value={"loss": 1.0})
        trainer._validate = mock.Mock(return_value={})
        trainer._log = mock.Mock()
        saved_at = []
        trainer._save_snapshot = lambda *args, **kwargs: saved_at.append(
            trainer._curr_iter
        )
        trainer._write_completion_marker = mock.Mock()
        trainer._install_checkpoint_signal_handlers = mock.Mock()
        trainer._restore_checkpoint_signal_handlers = mock.Mock()
        trainer._checkpoint_stop_requested_across_ranks = mock.Mock(
            return_value=False
        )

        outcome = trainer.train()

        self.assertEqual(outcome, "completed")
        self.assertEqual(saved_at, [0, 1_000, 2_000, 2_501])
        self.assertEqual(trainer._step.call_count, 2_502)
        trainer._write_completion_marker.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
