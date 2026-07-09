import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.transition_checkpoint import _atomic_write_json, build_handoff


class TransitionCheckpointToolTest(unittest.TestCase):
    def _write_parent(self, root: Path, identity: str) -> Path:
        path = root / "snapshot.pt"
        torch.save(
            {
                "snapshot_schema_version": 3,
                "run_identity_sha256": identity,
                "model": {},
                "optimizer": {},
                "lr_scheduler": {},
                "_start_iter": 41,
                "_total_observations": 9_999,
                "gradient_accumulation_steps": 2,
                "world_size": 2,
                "rank_states": [
                    {"global_rank": 0},
                    {"global_rank": 1},
                ],
            },
            path,
        )
        return path

    def _write_identity(self, root: Path, identity: str) -> Path:
        path = root / "run_identity.json"
        path.write_text(
            json.dumps(
                {
                    "identity_sha256": identity,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 2,
                    "world_size": 2,
                    "effective_global_batch_size": 16,
                }
            )
        )
        return path

    def _write_ack(self, root: Path, snapshot: Path, identity: str) -> Path:
        path = root / "checkpoint-ack.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "checkpoint_written": True,
                    "next_iter": 41,
                    "run_identity_sha256": identity,
                    "snapshot": str(snapshot.resolve()),
                }
            )
        )
        return path

    def test_build_handoff_binds_snapshot_and_parent_identity(self):
        identity = "a" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = self._write_parent(root, identity)
            payload = build_handoff(
                parent_snapshot=snapshot,
                expected_parent_run_identity_sha256=identity,
            )

            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["parent_run_identity_sha256"], identity)
            self.assertEqual(payload["parent_next_iteration"], 41)
            self.assertEqual(payload["parent_total_observations"], 9_999)
            self.assertEqual(len(payload["parent_snapshot_sha256"]), 64)

    def test_parent_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._write_parent(Path(temporary), "a" * 64)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                build_handoff(
                    parent_snapshot=snapshot,
                    expected_parent_run_identity_sha256="b" * 64,
                )

    def test_topology_migration_binds_ack_and_preserves_global_batch(self):
        identity = "a" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = self._write_parent(root, identity)
            run_identity = self._write_identity(root, identity)
            checkpoint_ack = self._write_ack(root, snapshot, identity)
            payload = build_handoff(
                parent_snapshot=snapshot,
                expected_parent_run_identity_sha256=identity,
                parent_run_identity=run_identity,
                checkpoint_ack=checkpoint_ack,
                transition_kind="topology_migration_reset_rank_state",
                target_batch_size=4,
                target_gradient_accumulation_steps=4,
                target_world_size=1,
                authorization_basis="user authorized 8-node migration",
            )

            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["rank_local_state_policy"], "reset")
            self.assertEqual(payload["parent_effective_global_batch_size"], 16)
            self.assertEqual(payload["target_effective_global_batch_size"], 16)
            self.assertEqual(payload["checkpoint_ack_next_iter"], 41)
            self.assertEqual(len(payload["checkpoint_ack_sha256"]), 64)

            handoff = root / "topology-handoff.json"
            _atomic_write_json(handoff, payload)
            _atomic_write_json(handoff, {**payload, "created_at_utc": "later"})
            self.assertEqual(json.loads(handoff.read_text()), payload)

    def test_topology_migration_rejects_global_batch_change(self):
        identity = "a" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = self._write_parent(root, identity)
            run_identity = self._write_identity(root, identity)
            checkpoint_ack = self._write_ack(root, snapshot, identity)
            with self.assertRaisesRegex(ValueError, "effective global batch"):
                build_handoff(
                    parent_snapshot=snapshot,
                    expected_parent_run_identity_sha256=identity,
                    parent_run_identity=run_identity,
                    checkpoint_ack=checkpoint_ack,
                    transition_kind="topology_migration_reset_rank_state",
                    target_batch_size=4,
                    target_gradient_accumulation_steps=3,
                    target_world_size=1,
                    authorization_basis="user authorized migration",
                )

    def test_topology_migration_rejects_stale_checkpoint_ack(self):
        identity = "a" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = self._write_parent(root, identity)
            run_identity = self._write_identity(root, identity)
            checkpoint_ack = self._write_ack(root, snapshot, identity)
            ack = json.loads(checkpoint_ack.read_text())
            ack["next_iter"] = 40
            checkpoint_ack.write_text(json.dumps(ack))
            with self.assertRaisesRegex(ValueError, "next_iter"):
                build_handoff(
                    parent_snapshot=snapshot,
                    expected_parent_run_identity_sha256=identity,
                    parent_run_identity=run_identity,
                    checkpoint_ack=checkpoint_ack,
                    transition_kind="topology_migration_reset_rank_state",
                    target_batch_size=4,
                    target_gradient_accumulation_steps=4,
                    target_world_size=1,
                    authorization_basis="user authorized migration",
                )

    def test_existing_handoff_is_idempotent_but_not_replaceable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "handoff_complete.json"
            payload = {
                "schema_version": 1,
                "status": "complete",
                "parent_snapshot": "/snapshot.pt",
                "parent_snapshot_sha256": "a" * 64,
                "parent_run_identity_sha256": "b" * 64,
            }
            _atomic_write_json(path, payload)
            _atomic_write_json(path, {**payload, "created_at_utc": "later"})
            self.assertEqual(json.loads(path.read_text()), payload)
            with self.assertRaises(FileExistsError):
                _atomic_write_json(
                    path, {**payload, "parent_snapshot_sha256": "c" * 64}
                )


if __name__ == "__main__":
    unittest.main()
