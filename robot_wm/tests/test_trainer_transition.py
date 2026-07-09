import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from robot_wm.utils.trainer import Trainer


class TrainerTransitionTest(unittest.TestCase):
    def _make_trainer(self, child_identity: str) -> Trainer:
        trainer = Trainer.__new__(Trainer)
        trainer.local_rank = 0
        trainer.global_rank = 0
        trainer.world_size = 2
        trainer.run_identity_sha256 = child_identity
        trainer.gradient_accumulation_steps = 2
        trainer.model = mock.Mock()
        trainer.optimizer = mock.Mock()
        trainer.lr_scheduler = mock.Mock()
        trainer.scaler = mock.Mock()
        trainer.use_amp = True
        trainer.metrics = SimpleNamespace(
            total_observations=0,
            best_val_loss=3.0,
            best_val_loss_old=2.0,
            samples_since_log=4,
            time_since_log=8.0,
        )
        trainer.metrics.reset_throughput_counters = lambda: (
            setattr(trainer.metrics, "samples_since_log", 0),
            setattr(trainer.metrics, "time_since_log", 0.0),
        )
        trainer.wandb_run_id = "parent-wandb"
        trainer._resume_rng_state = {"parent": True}
        trainer.resumed = True
        trainer.transitioned = False
        trainer.transition_parent = None
        return trainer

    def _write_parent_and_handoff(self, root: Path, identity: str):
        snapshot_path = root / "parent.pt"
        torch.save(
            {
                "snapshot_schema_version": 3,
                "run_identity_sha256": identity,
                "model": {"weight": torch.tensor([1.0])},
                "optimizer": {"state": "optimizer"},
                "lr_scheduler": {"state": "scheduler"},
                "scaler": {"scale": torch.tensor(16.0)},
                "_start_iter": 41,
                "_total_observations": 9_999,
                "best_val_loss": 0.1,
                "best_val_losses": {"best_val_loss_parent": 0.1},
                "wandb_run_id": "parent-wandb",
                "gradient_accumulation_steps": 2,
                "world_size": 2,
                "rank_states": [
                    {"global_rank": 0, "data_loader": {"cursor": 10}},
                    {"global_rank": 1, "data_loader": {"cursor": 20}},
                ],
            },
            snapshot_path,
        )
        snapshot_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        handoff_path = root / "handoff_complete.json"
        handoff_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "parent_snapshot": str(snapshot_path),
                    "parent_snapshot_sha256": snapshot_sha,
                    "parent_run_identity_sha256": identity,
                },
                sort_keys=True,
            )
        )
        return snapshot_path, handoff_path

    def _write_topology_migration(self, root: Path, identity: str):
        snapshot_path, handoff_path = self._write_parent_and_handoff(root, identity)
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        snapshot["world_size"] = 4
        snapshot["gradient_accumulation_steps"] = 2
        snapshot["rank_states"] = [
            {"global_rank": rank, "data_loader": {"cursor": rank}}
            for rank in range(4)
        ]
        torch.save(snapshot, snapshot_path)
        snapshot_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()

        identity_path = root / "run_identity.json"
        identity_path.write_text(
            json.dumps(
                {
                    "identity_sha256": identity,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 2,
                    "world_size": 4,
                    "effective_global_batch_size": 32,
                },
                sort_keys=True,
            )
        )
        identity_file_sha = hashlib.sha256(identity_path.read_bytes()).hexdigest()
        ack_path = root / "checkpoint-ack.json"
        ack_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "checkpoint_written": True,
                    "next_iter": 41,
                    "run_identity_sha256": identity,
                    "snapshot": str(snapshot_path.resolve()),
                },
                sort_keys=True,
            )
        )
        ack_sha = hashlib.sha256(ack_path.read_bytes()).hexdigest()
        handoff_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "status": "complete",
                    "transition_kind": "topology_migration_reset_rank_state",
                    "rank_local_state_policy": "reset",
                    "authorization_basis": "user authorized topology migration",
                    "parent_snapshot": str(snapshot_path.resolve()),
                    "parent_snapshot_sha256": snapshot_sha,
                    "parent_run_identity_sha256": identity,
                    "parent_run_identity": str(identity_path.resolve()),
                    "parent_run_identity_file_sha256": identity_file_sha,
                    "checkpoint_ack": str(ack_path.resolve()),
                    "checkpoint_ack_sha256": ack_sha,
                    "checkpoint_ack_next_iter": 41,
                    "parent_batch_size": 4,
                    "parent_gradient_accumulation_steps": 2,
                    "parent_world_size": 4,
                    "parent_effective_global_batch_size": 32,
                    "target_batch_size": 4,
                    "target_gradient_accumulation_steps": 4,
                    "target_world_size": 2,
                    "target_effective_global_batch_size": 32,
                },
                sort_keys=True,
            )
        )
        return snapshot_path, handoff_path

    def test_transition_preserves_training_progress_and_resets_dataset_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_identity = "a" * 64
            trainer = self._make_trainer("b" * 64)
            _, handoff = self._write_parent_and_handoff(root, parent_identity)

            trainer._load_transition_snapshot(handoff)

            trainer.model.module.load_state_dict.assert_called_once()
            trainer.optimizer.load_state_dict.assert_called_once_with(
                {"state": "optimizer"}
            )
            trainer.lr_scheduler.load_state_dict.assert_called_once_with(
                {"state": "scheduler"}
            )
            trainer.scaler.load_state_dict.assert_called_once()
            self.assertEqual(trainer._start_iter, 41)
            self.assertEqual(trainer.metrics.total_observations, 9_999)
            self.assertEqual(trainer.metrics.best_val_loss, float("inf"))
            self.assertFalse(hasattr(trainer.metrics, "best_val_loss_old"))
            self.assertEqual(trainer.metrics.samples_since_log, 0)
            self.assertIsNone(trainer._resume_rng_state)
            self.assertIsNone(trainer.wandb_run_id)
            self.assertFalse(trainer.resumed)
            self.assertTrue(trainer.transitioned)
            self.assertEqual(
                trainer.transition_parent["run_identity_sha256"], parent_identity
            )

    def test_hash_mismatch_fails_before_training_state_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = self._make_trainer("b" * 64)
            _, handoff = self._write_parent_and_handoff(root, "a" * 64)
            manifest = json.loads(handoff.read_text())
            manifest["parent_snapshot_sha256"] = "c" * 64
            handoff.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                trainer._load_transition_snapshot(handoff)

            trainer.model.module.load_state_dict.assert_not_called()
            trainer.optimizer.load_state_dict.assert_not_called()
            trainer.lr_scheduler.load_state_dict.assert_not_called()

    def test_topology_migration_loads_progress_and_resets_rank_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_identity = "a" * 64
            trainer = self._make_trainer("b" * 64)
            trainer.batch_size = 4
            trainer.world_size = 2
            trainer.gradient_accumulation_steps = 4
            _, handoff = self._write_topology_migration(root, parent_identity)

            trainer._load_transition_snapshot(handoff)

            trainer.model.module.load_state_dict.assert_called_once()
            trainer.optimizer.load_state_dict.assert_called_once()
            trainer.lr_scheduler.load_state_dict.assert_called_once()
            self.assertEqual(trainer._start_iter, 41)
            self.assertEqual(trainer.metrics.total_observations, 9_999)
            self.assertIsNone(trainer._resume_rng_state)
            self.assertIsNone(trainer.wandb_run_id)
            self.assertEqual(
                trainer.transition_parent["transition_kind"],
                "topology_migration_reset_rank_state",
            )
            self.assertEqual(
                trainer.transition_parent["effective_global_batch_size"], 32
            )

    def test_topology_migration_rejects_non_reset_policy_before_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = self._make_trainer("b" * 64)
            trainer.batch_size = 4
            trainer.world_size = 2
            trainer.gradient_accumulation_steps = 4
            _, handoff = self._write_topology_migration(root, "a" * 64)
            manifest = json.loads(handoff.read_text())
            manifest["rank_local_state_policy"] = "preserve"
            handoff.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(RuntimeError, "reset rank-local state"):
                trainer._load_transition_snapshot(handoff)
            trainer.model.module.load_state_dict.assert_not_called()
            trainer.optimizer.load_state_dict.assert_not_called()
            trainer.lr_scheduler.load_state_dict.assert_not_called()

    def test_transition_requires_a_distinct_child_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = "a" * 64
            trainer = self._make_trainer(identity)
            _, handoff = self._write_parent_and_handoff(root, identity)
            with self.assertRaisesRegex(RuntimeError, "new run identity"):
                trainer._load_transition_snapshot(handoff)


if __name__ == "__main__":
    unittest.main()
