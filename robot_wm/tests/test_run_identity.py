import json
import tempfile
import unittest
from pathlib import Path
from tools.run_identity import main


class RunIdentityTest(unittest.TestCase):
    def test_create_and_validate_binds_data_and_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_report = root / "data.json"
            runtime_report = root / "runtime.json"
            smoke_report = root / "smoke.json"
            identity = root / "identity.json"
            data_report.write_text(
                json.dumps(
                    {
                        "validator_sha256": "validator",
                        "invocation": {"min_timesteps": 66},
                        "reports": [
                            {"name": name, "fingerprint": {"digest": name}}
                            for name in ("droid", "egodex", "agibot", "abc")
                        ],
                    }
                )
            )
            runtime_report.write_text(
                json.dumps(
                    {
                        "python": "3.10.20",
                        "packages": {"torch": "2.7.1+cu128"},
                        "distributions": {},
                        "videox_commit": "pin",
                        "videox_status": "clean",
                        "weights": {"null_prompt": {"sha256": "prompt"}},
                    }
                )
            )
            smoke_report.write_text(
                json.dumps(
                    {
                        "kind": "lacwm_gradient_smoke_real",
                        "variant": "latent",
                        "status": "passed",
                    }
                )
            )
            common = [
                "--identity", str(identity),
                "--variant", "latent",
                "--git-commit", "abc123",
                "--batch-size", "4",
                "--gradient-accumulation-steps", "4",
                "--min-gpu-memory-mib", "78000",
                "--run-name", "test",
                "--python", "/tmp/python",
                "--wan-dir", "/tmp/wan",
                "--videox-home", "/tmp/videox",
                "--data-root", "/tmp/data",
                "--run-root", str(root),
                "--run-dir", str(root / "run"),
                "--wandb-mode", "disabled",
                "--data-report", str(data_report),
                "--runtime-report", str(runtime_report),
                "--smoke-report", str(smoke_report),
            ]
            self.assertEqual(main(["create", *common]), 0)
            self.assertEqual(main(["validate", *common]), 0)
            identity_payload = json.loads(identity.read_text())
            self.assertEqual(len(identity_payload["identity_sha256"]), 64)
            self.assertEqual(identity_payload["schema_version"], 3)
            self.assertEqual(identity_payload["batch_size"], 4)
            self.assertEqual(identity_payload["gradient_accumulation_steps"], 4)
            self.assertEqual(identity_payload["effective_global_batch_size"], 128)
            self.assertEqual(
                identity_payload["gpu_profile"],
                {"model": "B200", "minimum_memory_mib": 78_000},
            )

            changed_accumulation = list(common)
            accumulation_index = changed_accumulation.index(
                "--gradient-accumulation-steps"
            ) + 1
            changed_accumulation[accumulation_index] = "2"
            with self.assertRaises(RuntimeError):
                main(["validate", *changed_accumulation])

            payload = json.loads(data_report.read_text())
            payload["reports"][0]["fingerprint"]["digest"] = "changed"
            data_report.write_text(json.dumps(payload))
            with self.assertRaises(RuntimeError):
                main(["validate", *common])


if __name__ == "__main__":
    unittest.main()
