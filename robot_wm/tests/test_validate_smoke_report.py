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


if __name__ == "__main__":
    unittest.main()
