"""Registration/config identity tests for the prospective ACD-P0 pilot."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools import low_nfe_direct_baseline as pilot
from tools import adjacent_consistency_evaluate as evaluator


LUSTRE_ROOT = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/"
    "users/ldu/lacwm_train"
)


def _registration(tmp_path: Path) -> dict:
    resolved = {}
    for arm in pilot.ARMS:
        payload = {
            "name": arm.run_name,
            "trainer": {"config": {"dtype": "bfloat16", "amp_enabled": True}},
            "model": {
                "forward_model": {
                    "lora_rank": 64,
                    "lora_alpha": 128,
                    "lora_dropout": 0.05,
                },
                "adjacent_consistency": {"arm_mode": arm.arm_mode},
            },
        }
        content = pilot.yaml.safe_dump(payload, sort_keys=True).encode()
        path = tmp_path / f"{arm.code}.yaml"
        path.write_bytes(content)
        resolved[arm.code] = {
            "yaml": pilot.file_record(path),
            "semantic_sha256": pilot.semantic_config_sha256(payload),
            "fully_resolved": True,
            "normalization": "none; registered absolute output paths are bound",
        }
    python = Path("/usr/bin/python3").resolve(strict=True)
    core = {
        "schema_version": pilot.SCHEMA_VERSION,
        "kind": pilot.KIND_REGISTRATION,
        "fixed_protocol": pilot.fixed_protocol(),
        "arm_resolved_configs": resolved,
        "wandb": {"enabled": False, "writes": 0},
        "jobs_submitted": 0,
        "protected_test_accessed": False,
        "runtime": {
            "python": str(python),
            "python_target": pilot.file_record(python),
        },
    }
    registration = pilot.identity_payload(core)
    registration["registration_core_identity_sha256"] = registration[
        "identity_sha256"
    ]
    registration["arm_run_identity_sha256"] = {
        arm.code: pilot.arm_identity(registration, arm) for arm in pilot.ARMS
    }
    registration.pop("identity_sha256")
    return pilot.identity_payload(registration)


def test_endpoint_factorial_and_resource_arithmetic_are_frozen() -> None:
    assert len(pilot.ENDPOINTS) == 12
    assert len({endpoint.code for endpoint in pilot.ENDPOINTS}) == 12
    assert sum(endpoint.primary for endpoint in pilot.ENDPOINTS) == 5
    assert sum(endpoint.action_source == "episode_shuffled" for endpoint in pilot.ENDPOINTS) == 3
    plan = pilot.resource_plan()
    expected = 1 + 2 * 8 * 6 + 8 * 2
    assert plan["maximum_reserved_b200_hours"] == expected == 113
    assert plan["wandb"] is False
    assert plan["requeue"] is False


def test_registration_artifact_allowlist_is_exact() -> None:
    assert pilot.APPROVED_ARTIFACT_ROOTS == (
        Path("/mnt/data1"),
        Path("/mnt/data2"),
        LUSTRE_ROOT,
    )
    for path in (
        Path("/mnt/data1/acd"),
        Path("/mnt/data2/acd"),
        LUSTRE_ROOT / "artifacts/acd",
    ):
        assert pilot.approved_artifact_path(path, "output") == path
    for path in (
        Path("/lustre/acd"),
        LUSTRE_ROOT.parent / "other/acd",
        Path("/mnt/data1/../tmp/acd"),
    ):
        with pytest.raises(pilot.ACDPilotError, match="approved artifact root"):
            pilot.approved_artifact_path(path, "output")


@pytest.mark.parametrize(
    "path,value",
    (
        (("trainer", "config", "amp_enabled"), False),
        (("trainer", "config", "dtype"), "float16"),
        (("model", "forward_model", "lora_alpha"), 64),
        (("model", "forward_model", "lora_dropout"), 0.0),
    ),
)
def test_full_config_digest_binds_hidden_semantics(path, value) -> None:
    config = {
        "trainer": {"config": {"amp_enabled": True, "dtype": "bfloat16"}},
        "model": {"forward_model": {"lora_alpha": 128, "lora_dropout": 0.05}},
    }
    changed = copy.deepcopy(config)
    cursor = changed
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    assert pilot.semantic_config_sha256(config) != pilot.semantic_config_sha256(changed)


def test_registration_rederives_core_arm_and_resolved_config(tmp_path: Path) -> None:
    registration = _registration(tmp_path)
    path = tmp_path / "registration.json"
    pilot.exclusive_json(path, registration)
    assert pilot.validate_registration(path) == registration

    config_path = Path(
        registration["arm_resolved_configs"]["ACD-CONS"]["yaml"]["path"]
    )
    config_path.write_text(config_path.read_text() + "tampered: true\n")
    with pytest.raises(pilot.ACDPilotError, match="resolved config bytes"):
        pilot.validate_registration(path)


def test_registration_source_calls_actual_train_byte_binding() -> None:
    source = Path(pilot.__file__).read_text(encoding="utf-8")
    register_body = source[source.index("def command_register") : source.index("def validate_registration")]
    assert register_body.count("_bind_training_array(") == 2
    assert "actual_bytes_hashed_at_registration" in source
    assert "trainer.config.visualization.viz_path=" in source


def test_evaluator_rehashes_both_frozen_metric_weights(tmp_path: Path) -> None:
    linear = tmp_path / "lpips-linear.pth"
    alexnet = tmp_path / "alexnet.pth"
    linear.write_bytes(b"registered-linear-weights")
    alexnet.write_bytes(b"registered-alexnet-weights")
    registration = {
        "metrics": {
            "lpips_linear_weight": pilot.file_record(linear),
            "alexnet_weight": pilot.file_record(alexnet),
        }
    }
    assert evaluator._validate_metric_weight_records(registration) == registration["metrics"]

    alexnet.write_bytes(b"tampered-alexnet-weights!")
    with pytest.raises(evaluator.ACDEvaluationError, match="alexnet_weight bytes differ"):
        evaluator._validate_metric_weight_records(registration)


def test_evaluator_rejects_symlinked_metric_weights(tmp_path: Path) -> None:
    linear = tmp_path / "lpips-linear.pth"
    alexnet = tmp_path / "alexnet.pth"
    alias = tmp_path / "alexnet-alias.pth"
    linear.write_bytes(b"registered-linear-weights")
    alexnet.write_bytes(b"registered-alexnet-weights")
    alias.symlink_to(alexnet)
    registration = {
        "metrics": {
            "lpips_linear_weight": pilot.file_record(linear),
            "alexnet_weight": {
                **pilot.file_record(alexnet),
                "path": str(alias),
            },
        }
    }
    with pytest.raises(
        evaluator.ACDEvaluationError,
        match="alexnet_weight is not a canonical regular file",
    ):
        evaluator._validate_metric_weight_records(registration)
