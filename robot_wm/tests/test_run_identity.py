import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from tools.run_identity import main


class RunIdentityTest(unittest.TestCase):
    @staticmethod
    def _file_record(path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }

    def test_create_and_validate_binds_data_and_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_report = root / "data.json"
            runtime_report = root / "runtime.json"
            smoke_report = root / "smoke.json"
            identity = root / "identity.json"
            data_root = root / "data"
            sources = {
                "droid": str(data_root / "droid_lerobot"),
                "egodex": str(data_root / "egodex_cdn" / "manifest.csv"),
                "agibot": str(data_root / "agibot" / "manifest.csv"),
                "abc": str(data_root / "abc_pp" / "manifest.txt"),
            }
            data_report.write_text(
                json.dumps(
                    {
                        "validator_sha256": "validator",
                        "invocation": {
                            "datasets": ["droid", "egodex", "agibot", "abc"],
                            "sources": sources,
                            "min_timesteps": 66,
                        },
                        "reports": [
                            {
                                "name": name,
                                "source": sources[name],
                                "fingerprint": {"digest": name},
                            }
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
                "--dataset-stage", "all-four",
                "--variant", "latent",
                "--git-commit", "abc123",
                "--batch-size", "4",
                "--gradient-accumulation-steps", "4",
                "--world-size", "8",
                "--node-count", "1",
                "--gpus-per-node", "8",
                "--min-gpu-memory-mib", "78000",
                "--run-name", "test",
                "--python", "/tmp/python",
                "--wan-dir", "/tmp/wan",
                "--videox-home", "/tmp/videox",
                "--data-root", str(data_root),
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
            self.assertEqual(identity_payload["schema_version"], 6)
            self.assertEqual(identity_payload["dataset_stage"], "all-four")
            self.assertEqual(
                identity_payload["dataset_names"],
                ["droid", "egodex", "agibot", "abc"],
            )
            self.assertEqual(identity_payload["batch_size"], 4)
            self.assertEqual(identity_payload["gradient_accumulation_steps"], 4)
            self.assertEqual(identity_payload["effective_global_batch_size"], 128)
            self.assertEqual(identity_payload["world_size"], 8)
            self.assertEqual(identity_payload["node_count"], 1)
            self.assertEqual(identity_payload["gpus_per_node"], 8)
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

            changed_world_size = list(common)
            world_size_index = changed_world_size.index("--world-size") + 1
            changed_world_size[world_size_index] = "16"
            with self.assertRaises(RuntimeError):
                main(["validate", *changed_world_size])

            changed_stage = list(common)
            stage_index = changed_stage.index("--dataset-stage") + 1
            changed_stage[stage_index] = "replacement-stage"
            with self.assertRaises(RuntimeError):
                main(["validate", *changed_stage])

            payload = json.loads(data_report.read_text())
            payload["reports"][0]["fingerprint"]["digest"] = "changed"
            data_report.write_text(json.dumps(payload))
            with self.assertRaises(RuntimeError):
                main(["validate", *common])

    def test_fast_identity_binds_user_authorization_and_loader_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "fast-overlay"
            data_root.mkdir()
            data_report = root / "files-only.json"
            runtime_report = root / "runtime.json"
            smoke_report = root / "gradient.json"
            mixed_report = root / "mixed.json"
            loader_state = root / "mixed.loader_state.pt"
            authorization_path = root / "authorization.json"
            identity = root / "identity.json"
            commit = "a" * 40
            sources = {
                "droid": str(data_root / "droid_lerobot"),
                "egodex": str(data_root / "egodex_cdn" / "manifest.csv"),
                "agibot": str(data_root / "agibot" / "manifest.csv"),
                "abc": str(data_root / "abc_pp" / "manifest.txt"),
            }
            data_report.write_text(
                json.dumps(
                    {
                        "invocation": {
                            "data_root": str(data_root),
                            "datasets": ["droid", "egodex", "agibot", "abc"],
                            "sources": sources,
                            "files_only": True,
                            "allowed_external_roots": {
                                "egodex": str(data_root.parent / "egodex_cdn"),
                                "abc": str(data_root.parent / "abc_pp"),
                            },
                        },
                        "reports": [
                            {
                                "name": name,
                                "source": sources[name],
                                "fingerprint": {"digest": name},
                            }
                            for name in ("droid", "egodex", "agibot", "abc")
                        ],
                    }
                ),
                encoding="utf-8",
            )
            runtime_report.write_text("{}", encoding="utf-8")
            smoke_report.write_text(
                json.dumps(
                    {
                        "kind": "lacwm_gradient_smoke_real",
                        "variant": "latent",
                        "status": "passed",
                    }
                ),
                encoding="utf-8",
            )
            loader_state.write_bytes(b"exact loader state")
            state_hash = hashlib.sha256(loader_state.read_bytes()).hexdigest()
            mixed_report.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "lacwm_real_mixed_stateful_dataloader_smoke",
                        "status": "passed",
                        "git_commit": commit,
                        "git_status": "",
                        "requested_data_root": str(data_root),
                        "validation": {
                            "data": {
                                "root": str(data_root),
                                "source_order": [
                                    "Droid",
                                    "EgoDex",
                                    "Agibot",
                                    "ABC",
                                ],
                                "source_lengths": [10_000, 10_000, 5_671, 10_000],
                                "total_episodes": 35_671,
                            },
                            "mix": {
                                "batches_checked": 8,
                                "mixed_batches": 8,
                                "observed_source_counts": {
                                    "Droid": 10,
                                    "EgoDex": 10,
                                    "Agibot": 6,
                                    "ABC": 10,
                                },
                            },
                            "resume": {
                                "exact_continuation": True,
                                "reference_signature": "signature",
                                "restored_signature": "signature",
                                "state_path": str(loader_state),
                                "state_size": loader_state.stat().st_size,
                                "state_sha256": state_hash,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            waiver_path = data_root / "fast_validation_waiver.json"
            waiver_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "lacwm_user_authorized_fast_mixed_overlay",
                        "logical_read_skipped": True,
                        "strict_validated": False,
                        "training_authorized": False,
                        "selected_episodes": 5_671,
                        "required_payloads": 39_697,
                        "required_payload_bytes": 123_456,
                        "metadata_fingerprint_sha256": "b" * 64,
                    }
                ),
                encoding="utf-8",
            )
            authorization = {
                "schema_version": 1,
                "kind": "lacwm_fast_training_authorization",
                "policy": "files_only_user_waived_v1",
                "training_authorized": True,
                "authorization_scope": "one_branch_one_commit_one_fast_overlay",
                "authorized_by": "user",
                "authorization_basis": "explicit test authorization",
                "certificate_path": str(authorization_path),
                "branch": "lora",
                "expected_commit": commit,
                "data_root": str(data_root),
                "source_order": ["Droid", "EgoDex", "Agibot", "ABC"],
                "source_lengths": [10_000, 10_000, 5_671, 10_000],
                "total_episodes": 35_671,
                "inputs": {
                    "waiver": self._file_record(waiver_path),
                    "files_only_report": self._file_record(data_report),
                    "mixed_loader_report": self._file_record(mixed_report),
                    "gradient_report": self._file_record(smoke_report),
                },
                "agibot": {
                    "metadata_fingerprint_sha256": "b" * 64,
                    "required_payloads": 39_697,
                    "required_payload_bytes": 123_456,
                },
                "validation": {
                    "original_waiver": {
                        "logical_read_skipped": True,
                        "strict_validated": False,
                        "training_authorized": False,
                    },
                    "files_only": {"passed": True, "files_only": True},
                    "mixed_loader": {"passed": True},
                    "real_gradient": {"passed": True},
                },
            }
            authorization_path.write_text(
                json.dumps(authorization), encoding="utf-8"
            )
            authorization_path.chmod(0o440)
            args = [
                "--identity", str(identity),
                "--variant", "latent",
                "--git-commit", commit,
                "--batch-size", "4",
                "--gradient-accumulation-steps", "2",
                "--world-size", "128",
                "--node-count", "16",
                "--gpus-per-node", "8",
                "--min-gpu-memory-mib", "78000",
                "--run-name", "fast-lora",
                "--python", "/tmp/python",
                "--wan-dir", "/tmp/wan",
                "--videox-home", "/tmp/videox",
                "--data-root", str(data_root),
                "--run-root", str(root),
                "--run-dir", str(root / "run"),
                "--wandb-mode", "disabled",
                "--data-report", str(data_report),
                "--data-validation-policy", "files_only_user_waived_v1",
                "--fast-training-authorization", str(authorization_path),
                "--mixed-loader-report", str(mixed_report),
                "--runtime-report", str(runtime_report),
                "--smoke-report", str(smoke_report),
            ]

            self.assertEqual(main(["create", *args]), 0)
            self.assertEqual(main(["validate", *args]), 0)
            payload = json.loads(identity.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 6)
            self.assertEqual(
                payload["data"]["validation_policy"],
                "files_only_user_waived_v1",
            )
            self.assertEqual(
                payload["data"]["fast_evidence"]["authorization"]["sha256"],
                hashlib.sha256(authorization_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                payload["data"]["fast_evidence"]["mixed_loader"][
                    "state_sha256"
                ],
                state_hash,
            )

            mixed_report.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "input binding changed"):
                main(["validate", *args])

    def test_world_size_controls_effective_global_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_report = root / "data.json"
            runtime_report = root / "runtime.json"
            smoke_report = root / "smoke.json"
            data_root = root / "data"
            sources = {
                "droid": str(data_root / "droid_lerobot"),
                "egodex": str(data_root / "egodex_cdn" / "manifest.csv"),
                "agibot": str(data_root / "agibot" / "manifest.csv"),
                "abc": str(data_root / "abc_pp" / "manifest.txt"),
            }
            data_report.write_text(
                json.dumps(
                    {
                        "invocation": {
                            "datasets": ["droid", "egodex", "agibot", "abc"],
                            "sources": sources,
                        },
                        "reports": [
                            {
                                "name": name,
                                "source": sources[name],
                                "fingerprint": {"digest": name},
                            }
                            for name in ("droid", "egodex", "agibot", "abc")
                        ]
                    }
                )
            )
            runtime_report.write_text("{}")
            smoke_report.write_text(json.dumps({"variant": "latent"}))

            for world_size in (8, 16, 24, 32, 128, 256):
                identity = root / f"identity-{world_size}.json"
                args = [
                    "--identity", str(identity),
                    "--variant", "latent",
                    "--git-commit", "abc123",
                    "--batch-size", "2",
                    "--gradient-accumulation-steps", "4",
                    "--world-size", str(world_size),
                    "--node-count", str(world_size // 8),
                    "--gpus-per-node", "8",
                    "--min-gpu-memory-mib", "78000",
                    "--run-name", f"test-{world_size}",
                    "--python", "/tmp/python",
                    "--wan-dir", "/tmp/wan",
                    "--videox-home", "/tmp/videox",
                    "--data-root", str(data_root),
                    "--run-root", str(root),
                    "--run-dir", str(root / f"run-{world_size}"),
                    "--wandb-mode", "disabled",
                    "--data-report", str(data_report),
                    "--runtime-report", str(runtime_report),
                    "--smoke-report", str(smoke_report),
                ]
                self.assertEqual(main(["create", *args]), 0)
                payload = json.loads(identity.read_text())
                self.assertEqual(payload["world_size"], world_size)
                self.assertEqual(payload["node_count"], world_size // 8)
                self.assertEqual(payload["gpus_per_node"], 8)
                self.assertEqual(
                    payload["effective_global_batch_size"], 2 * 4 * world_size
                )

    def test_subset_identity_rejects_extra_report_or_wrong_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            sources = {
                "droid": str(data_root / "droid_lerobot"),
                "egodex": str(data_root / "egodex_cdn" / "manifest.csv"),
            }
            data_report = root / "data.json"
            runtime_report = root / "runtime.json"
            smoke_report = root / "smoke.json"
            identity = root / "identity.json"
            payload = {
                "invocation": {
                    "datasets": ["droid", "egodex"],
                    "sources": sources,
                },
                "reports": [
                    {
                        "name": name,
                        "source": sources[name],
                        "fingerprint": {"digest": name},
                    }
                    for name in ("droid", "egodex")
                ],
            }
            data_report.write_text(json.dumps(payload))
            runtime_report.write_text("{}")
            smoke_report.write_text(json.dumps({"variant": "latent"}))
            args = [
                "--identity", str(identity),
                "--dataset-stage", "ready-two",
                "--datasets", "droid", "egodex",
                "--variant", "latent",
                "--git-commit", "abc123",
                "--batch-size", "1",
                "--gradient-accumulation-steps", "1",
                "--world-size", "128",
                "--node-count", "16",
                "--gpus-per-node", "8",
                "--min-gpu-memory-mib", "78000",
                "--run-name", "subset",
                "--python", "/tmp/python",
                "--wan-dir", "/tmp/wan",
                "--videox-home", "/tmp/videox",
                "--data-root", str(data_root),
                "--run-root", str(root),
                "--run-dir", str(root / "run"),
                "--wandb-mode", "disabled",
                "--data-report", str(data_report),
                "--runtime-report", str(runtime_report),
                "--smoke-report", str(smoke_report),
            ]
            self.assertEqual(main(["create", *args]), 0)
            identity_payload = json.loads(identity.read_text())
            self.assertEqual(identity_payload["dataset_names"], ["droid", "egodex"])
            self.assertEqual(
                set(identity_payload["data"]["fingerprints"]),
                {"droid", "egodex"},
            )

            payload["invocation"]["sources"]["abc"] = str(root / "ignored-abc")
            data_report.write_text(json.dumps(payload))
            self.assertEqual(main(["validate", *args]), 0)

            payload["reports"].append(
                {
                    "name": "abc",
                    "source": str(data_root / "abc_pp" / "manifest.txt"),
                    "fingerprint": {"digest": "abc"},
                }
            )
            data_report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "do not exactly match"):
                main(["validate", *args])

            payload["reports"].pop()
            payload["reports"][0]["source"] = str(root / "wrong")
            data_report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "source"):
                main(["validate", *args])


if __name__ == "__main__":
    unittest.main()
