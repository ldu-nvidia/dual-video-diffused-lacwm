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
