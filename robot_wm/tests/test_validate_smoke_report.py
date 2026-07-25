import json
import tempfile
import unittest
from pathlib import Path

from tools.validate_smoke_report import validate


class ValidateSmokeReportTest(unittest.TestCase):
    def _payload(self, root: Path):
        return {
            "schema_version": 1,
            "kind": "lacwm_gradient_smoke_real",
            "data_mode": "real",
            "status": "passed",
            "variant": "latent",
            "git_commit": "abc123",
            "git_status": "",
            "paths": {
                "wan_dir": str(root / "wan"),
                "videox_home": str(root / "videox"),
                "data_root": str(root / "data"),
            },
            "validation": {
                "trainable_parameters": 10,
                "gpu": {"max_memory_allocated_gib": 1.0},
                "steps": [
                    {"loss": 1.0, "morphology_index": morphology}
                    for morphology in (0, 2, 6, 9)
                ],
                "groups_ever_nonzero": {
                    "lora_": True,
                    "forward_model.action_to_control": True,
                    "action_pool": True,
                    "morphology_tokens": True,
                    "inverse_model": True,
                    "action_decoder": True,
                },
            },
        }

    def test_accepts_complete_real_gradient_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            report.write_text(json.dumps(self._payload(root)))
            payload = validate(
                report,
                variant="latent",
                git_commit="abc123",
                wan_dir=str(root / "wan"),
                videox_home=str(root / "videox"),
                data_root=str(root / "data"),
            )
            self.assertEqual(payload["status"], "passed")

    def test_rejects_missing_delayed_gradient(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self._payload(root)
            payload["validation"]["groups_ever_nonzero"]["action_pool"] = False
            report = root / "report.json"
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "action_pool"):
                validate(
                    report,
                    variant="latent",
                    git_commit="abc123",
                    wan_dir=str(root / "wan"),
                    videox_home=str(root / "videox"),
                    data_root=str(root / "data"),
                )

    def test_accepts_dual_abc_warmstart_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self._payload(root)
            payload["variant"] = "dual-with-ztf"
            payload["paths"]["warmstart_model"] = str(root / "snapshot.pt")
            payload["validation"]["steps"] = [
                {"loss": 1.0, "morphology_index": 9}
                for _ in range(4)
            ]
            dual_groups = {
                "action_encoder",
                "forward_model.tf_velocity_head",
                "forward_model.tf_velocity_head.linear",
                "forward_model.tf_velocity_head.norm",
                "forward_model.tf_clock_embedding",
                "forward_model.tf_clock_embedding.gate",
                "forward_model.tf_clock_embedding.net",
                "forward_model.tf_token_adapter",
                "forward_model.tf_token_adapter.gate",
                "forward_model.tf_token_adapter.projection",
                "forward_model.tf_token_adapter.norm",
            }
            all_groups = {
                "lora_",
                "forward_model.action_to_control",
                "action_pool",
                "morphology_tokens",
                *dual_groups,
            }
            payload["validation"]["groups_ever_nonzero"] = {
                group: True for group in all_groups
            }
            payload["validation"]["matched_trainable_tensors"] = {
                group: 1 for group in all_groups
            }
            payload["validation"]["condition_on_tf"] = True
            payload["validation"]["sigma_convention"] = (
                "sigma=1 is noise; sigma=0 is clean data"
            )
            payload["validation"]["dual_zero_init"] = {
                "tf_state_gate_abs_max": 0.0,
                "tf_clock_gate_abs_max": 0.0,
                "tf_head_weight_abs_max": 0.0,
                "tf_head_bias_abs_max": 0.0,
            }
            payload["validation"]["dual_video_noop"] = {
                "exact_video_velocity_equal": True,
                "max_abs_difference": 0.0,
                "production_baseline_exact_equal": True,
                "production_baseline_max_abs_difference": 0.0,
            }
            sha256 = "a" * 64
            payload["validation"]["warmstart"] = {
                "path": str(root / "snapshot.pt"),
                "sha256": sha256,
                "model_only": True,
                "unexpected_keys": [],
                "file_identity": {"size_bytes": 123},
            }
            report = root / "report.json"
            report.write_text(json.dumps(payload))
            validated = validate(
                report,
                variant="dual-with-ztf",
                git_commit="abc123",
                wan_dir=str(root / "wan"),
                videox_home=str(root / "videox"),
                data_root=str(root / "data"),
                warmstart_model=str(root / "snapshot.pt"),
                warmstart_sha256=sha256,
            )
            self.assertTrue(validated["validation"]["condition_on_tf"])

            payload["validation"]["dual_video_noop"][
                "production_baseline_exact_equal"
            ] = False
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "ordinary production Wan"):
                validate(
                    report,
                    variant="dual-with-ztf",
                    git_commit="abc123",
                    wan_dir=str(root / "wan"),
                    videox_home=str(root / "videox"),
                    data_root=str(root / "data"),
                    warmstart_model=str(root / "snapshot.pt"),
                    warmstart_sha256=sha256,
                )


if __name__ == "__main__":
    unittest.main()
