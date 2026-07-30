import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "vjepa2_phase_gate",
    ROOT / "tools" / "vjepa2_phase_gate.py",
)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gate)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded = nn.Linear(3, 2)
        self.reset = nn.Linear(2, 1)


class _ExplicitActionWarmstartModel(nn.Module):
    def __init__(self, *, extra_layer: bool = False):
        super().__init__()
        self.backbone = nn.Linear(3, 2)
        layers = [
            nn.Linear(4, 3),
            nn.SiLU(),
            nn.Linear(3, 3),
            nn.SiLU(),
            nn.Linear(3, 2),
        ]
        if extra_layer:
            layers.extend((nn.SiLU(), nn.Linear(2, 1)))
        self.action_encoder = nn.Module()
        self.action_encoder.net = nn.Sequential(*layers)


def _snapshot_without_action_encoder(
    model: nn.Module,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "model": {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if not name.startswith("action_encoder.")
        }
    }


class _FutureOnlyAuditModel:
    num_history_frames = 5
    num_future_frames = 8

    def __init__(self):
        self._artifacts = None
        self._last_sampling_counters = None
        self.full_clip_arguments = None

    def eval(self):
        return self

    @staticmethod
    def _counters(deployment_mode):
        return {
            "wan_calls_by_source_nfe": {"autonomous:nfe_1": 1},
            "wan_calls_total": 1,
            "online_teacher_calls": 0,
            "auxiliary_clean_available": 0,
            "artifacts_collected": 1,
            "deployment_mode": deployment_mode,
        }

    @staticmethod
    def _evidence():
        return {
            "video_initial_state": torch.arange(4, dtype=torch.float16),
            "tf_initial_state": torch.arange(5, dtype=torch.float16),
            "tf_initial_noise": torch.arange(5, dtype=torch.float16),
            "reference_latents": torch.arange(3, dtype=torch.float16),
        }

    def sample_future_deployable(
        self,
        history,
        actions,
        morphology,
        *,
        collect_artifacts,
        sample_ids,
    ):
        del actions, morphology, sample_ids
        assert history.shape[1] == 5
        assert collect_artifacts
        self._last_sampling_counters = self._counters(1)
        self._artifacts = self._evidence()
        return torch.zeros(
            history.shape[0],
            3,
            8,
            180,
            960,
            dtype=history.dtype,
        )

    def _sample_future(
        self,
        rgb,
        actions,
        morphology,
        *,
        auxiliary_target,
        collect_artifacts,
        deployment_mode,
        sample_ids,
    ):
        del actions, morphology, sample_ids
        self.full_clip_arguments = {
            "rgb_frames": rgb.shape[1],
            "auxiliary_target": auxiliary_target,
            "collect_artifacts": collect_artifacts,
            "deployment_mode": deployment_mode,
        }
        self._last_sampling_counters = self._counters(0)
        self._artifacts = self._evidence()
        prediction = torch.zeros(
            rgb.shape[0],
            3,
            13,
            180,
            960,
            dtype=rgb.dtype,
        )
        return prediction, prediction.clone()

    def pop_visualization_artifacts(self):
        artifacts = self._artifacts
        self._artifacts = None
        return artifacts


class VJEPA2PhaseGateTest(unittest.TestCase):
    def test_strict_warmstart_accepts_only_allowlisted_resets(self):
        source = _ToyModel()
        expected_weight = torch.full_like(source.loaded.weight, 7.0)
        expected_bias = torch.full_like(source.loaded.bias, -2.0)
        snapshot = {
            "model": {
                "loaded.weight": expected_weight,
                "loaded.bias": expected_bias,
                # A legacy checkpoint tensor under a declared reset prefix is
                # explicitly ignored rather than treated as an unknown key.
                "reset.legacy": torch.ones(1),
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.pt"
            torch.save(snapshot, path)
            result = gate.strict_warmstart_load(source, path, ("reset",))

        self.assertTrue(torch.equal(source.loaded.weight, expected_weight))
        self.assertTrue(torch.equal(source.loaded.bias, expected_bias))
        self.assertEqual(result["loaded_tensor_count"], 2)
        self.assertEqual(result["reset_model_keys"], ["reset.bias", "reset.weight"])
        self.assertEqual(result["excluded_checkpoint_key_count"], 1)

    def test_strict_warmstart_rejects_unallowlisted_missing_key(self):
        model = _ToyModel()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.pt"
            torch.save({"model": {"loaded.weight": model.loaded.weight}}, path)
            with self.assertRaisesRegex(gate.GateError, "disallowed_missing"):
                gate.strict_warmstart_load(model, path, ("reset",))

    def test_strict_warmstart_accepts_exact_action_encoder_missing_keys(self):
        model = _ExplicitActionWarmstartModel()
        snapshot = _snapshot_without_action_encoder(model)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.pt"
            torch.save(snapshot, path)
            result = gate.strict_warmstart_load(
                model,
                path,
                (),
                expected_missing_keys=gate.EXPECTED_WARMSTART_MISSING_KEYS,
            )

        expected = list(gate.EXPECTED_WARMSTART_MISSING_KEYS)
        self.assertEqual(result["expected_missing_keys"], expected)
        self.assertEqual(result["non_prefix_missing_keys"], expected)
        self.assertEqual(result["reset_model_keys"], [])
        self.assertEqual(result["loaded_tensor_count"], 2)

    def test_strict_warmstart_rejects_inexact_missing_key_declarations(self):
        model = _ExplicitActionWarmstartModel()
        snapshot = _snapshot_without_action_encoder(model)
        exact = gate.EXPECTED_WARMSTART_MISSING_KEYS
        declarations = {
            "incomplete": exact[:-1],
            "duplicate": exact + (exact[0],),
            "broad": ("action_encoder",),
            "non_model": exact + ("action_encoder.net.6.weight",),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.pt"
            torch.save(snapshot, path)
            for name, declaration in declarations.items():
                with self.subTest(name=name):
                    with self.assertRaises(gate.GateError):
                        gate.strict_warmstart_load(
                            model,
                            path,
                            (),
                            expected_missing_keys=declaration,
                        )

    def test_strict_warmstart_rejects_action_encoder_architecture_drift(self):
        model = _ExplicitActionWarmstartModel(extra_layer=True)
        snapshot = _snapshot_without_action_encoder(model)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.pt"
            torch.save(snapshot, path)
            with self.assertRaisesRegex(
                gate.GateError, "action_encoder.net.6"
            ):
                gate.strict_warmstart_load(
                    model,
                    path,
                    (),
                    expected_missing_keys=gate.EXPECTED_WARMSTART_MISSING_KEYS,
                )

    def test_strict_warmstart_still_rejects_action_encoder_checkpoint_drift(self):
        model = _ExplicitActionWarmstartModel()
        exact = gate.EXPECTED_WARMSTART_MISSING_KEYS
        cases = {}

        unexpected = _snapshot_without_action_encoder(model)
        unexpected["model"]["action_encoder.legacy.weight"] = torch.ones(1)
        cases["unexpected"] = unexpected

        shape_mismatch = _snapshot_without_action_encoder(model)
        shape_mismatch["model"][exact[0]] = torch.ones(17)
        cases["shape_mismatch"] = shape_mismatch

        with tempfile.TemporaryDirectory() as temporary:
            for name, snapshot in cases.items():
                with self.subTest(name=name):
                    path = Path(temporary) / f"{name}.pt"
                    torch.save(snapshot, path)
                    with self.assertRaises(gate.GateError):
                        gate.strict_warmstart_load(
                            model,
                            path,
                            (),
                            expected_missing_keys=exact,
                        )

    def test_batch_shape_contract(self):
        batch = {
            key: torch.empty(shape)
            for key, shape in gate.EXPECTED_BATCH_SHAPES.items()
        }
        batch["morphology_index"] = torch.tensor([9], dtype=torch.long)
        observed = gate.validate_batch_shapes(batch)
        self.assertEqual(observed["rgb"], [1, 13, 3, 180, 960])
        self.assertEqual(observed["morphology_index"], [1])
        batch["actions"] = torch.empty(1, 13, 5, 23)
        with self.assertRaisesRegex(gate.GateError, "actions shape differs"):
            gate.validate_batch_shapes(batch)

    def test_abc_morphology_is_integer_nine(self):
        batch = {
            key: torch.empty(shape)
            for key, shape in gate.EXPECTED_BATCH_SHAPES.items()
        }
        batch["morphology_index"] = torch.tensor([8], dtype=torch.long)
        with self.assertRaisesRegex(gate.GateError, "exactly 9"):
            gate.validate_batch_shapes(batch)
        batch["morphology_index"] = torch.tensor([9.0], dtype=torch.float32)
        with self.assertRaisesRegex(gate.GateError, "int64"):
            gate.validate_batch_shapes(batch)

    def test_shape_audit_clip_indices_must_be_unique(self):
        self.assertEqual(
            gate.validate_unique_clip_indices([7, 4, 9, 1, 8, 3, 2, 6], 8),
            [7, 4, 9, 1, 8, 3, 2, 6],
        )
        with self.assertRaisesRegex(gate.GateError, "not unique"):
            gate.validate_unique_clip_indices([1, 1, 2, 3, 4, 5, 6, 7], 8)

    def test_future_only_path_is_audited_against_full_clip_scoring(self):
        model = _FutureOnlyAuditModel()
        batch = {
            "rgb": torch.zeros(1, 13, 3, 180, 960),
            "actions": torch.zeros(1, 13, 5, 157),
            "clip_index": torch.tensor([11]),
            "morphology_index": torch.tensor([9]),
        }
        report = gate.future_free_nfe1(model, batch)
        comparison = report["ordinary_full_clip_audit"]
        self.assertTrue(comparison["generated_future_bitwise_equal"])
        self.assertTrue(comparison["same_initial_noise_and_reference"])
        self.assertEqual(model.full_clip_arguments["rgb_frames"], 13)
        self.assertIsNone(model.full_clip_arguments["auxiliary_target"])
        self.assertFalse(model.full_clip_arguments["deployment_mode"])

    def test_cache_complete_links_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "train.jsonl"
            pca = root / "train_pca64.pt"
            metadata_path = root / "metadata.json"
            complete_path = root / "complete.json"
            manifest.write_text('{"clip_id":"one"}\n', encoding="utf-8")
            pca.write_bytes(b"pinned-pca")
            hashes = {
                "cache_id": "1" * 64,
                "target_sha256": "2" * 64,
                "rgb_sha256": "3" * 64,
                "actions_sha256": "4" * 64,
            }
            metadata = {
                "complete": True,
                "clip_manifest": str(manifest.resolve()),
                "clip_manifest_sha256": gate.sha256_file(manifest),
                "clip_count": 512,
                "target_shape": [512, 64, 4, 24, 120],
                "rgb_shape": [512, 13, 3, 180, 960],
                "actions_shape": [512, 13, 5, 23],
                "checkpoint_sha256": "5" * 64,
                "source_commit": "6" * 40,
                **hashes,
            }
            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            complete = {
                "artifact_type": "vjepa2.1-immutable-cache-build",
                "format_version": 1,
                "pca": str(pca.resolve()),
                "pca_sha256": gate.sha256_file(pca),
                "splits": {
                    "train": {
                        "metadata": str(metadata_path.resolve()),
                        "metadata_sha256": gate.sha256_file(metadata_path),
                        "clip_count": 512,
                        **hashes,
                    }
                },
            }
            complete_path.write_text(json.dumps(complete), encoding="utf-8")

            result = gate.validate_cache_build_links(
                complete_path=complete_path.resolve(),
                train_manifest=manifest.resolve(),
                train_metadata=metadata_path.resolve(),
                pca_path=pca.resolve(),
            )
            self.assertEqual(result["cache_id"], "1" * 64)

            complete["pca_sha256"] = "0" * 64
            complete_path.write_text(json.dumps(complete), encoding="utf-8")
            with self.assertRaisesRegex(gate.GateError, "PCA digest"):
                gate.validate_cache_build_links(
                    complete_path=complete_path.resolve(),
                    train_manifest=manifest.resolve(),
                    train_metadata=metadata_path.resolve(),
                    pca_path=pca.resolve(),
                )

    def test_immutable_report_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            gate.exclusive_json(output, {"passed": True})
            with self.assertRaises(FileExistsError):
                gate.exclusive_json(output, {"passed": False})


if __name__ == "__main__":
    unittest.main()
