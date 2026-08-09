"""Registration/config identity tests for the prospective ACD-P0 pilot."""

from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from tools import adjacent_consistency_evaluate as evaluator
from tools import adjacent_consistency_memory_smoke as memory_smoke
from tools import low_nfe_direct_baseline as pilot
from robot_wm.modeling.low_nfe.adjacent_consistency import (
    AdjacentConsistencyError,
    _expanded_mask,
)


LUSTRE_ROOT = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/"
    "users/ldu/lacwm_train"
)


def _production_loss_mask_model(monkeypatch):
    """Load the real mask method while stubbing only unavailable VideoX types."""

    for module_name, attribute in (
        ("robot_wm.modeling.networks.wan_forward_model", "WanForwardModel"),
        ("robot_wm.modeling.tokenizers.rgb.wan_vae", "WanVAETokenizer"),
    ):
        module = types.ModuleType(module_name)
        setattr(module, attribute, nn.Module)
        monkeypatch.setitem(sys.modules, module_name, module)
    path = (
        Path(pilot.__file__).resolve().parents[1]
        / "projects/latent_action_models/lam/latent_action_dit_model.py"
    )
    spec = importlib.util.spec_from_file_location(
        "acd_production_loss_mask_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    model = object.__new__(module.LatentActionDiTModel)
    nn.Module.__init__(model)
    model.num_views = 3
    model.rgb_tokenizer = types.SimpleNamespace(temporal_ratio=4)
    return model


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
        "parent": {
            **pilot.parent_model_state_lineage(),
            "model_state_hash_receipt": {
                **pilot.parent_model_state_lineage(),
                "state_tensors": 1,
                "state_values": 1,
            },
        },
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
    expected = 1 + 2 * 8 * 2 + 8 * 2
    assert plan["maximum_reserved_b200_hours"] == expected == 49
    assert plan["training_time_limit_hours_each"] == 2
    assert plan["expected_parallel_wall_hours"] == "4-6"
    assert plan["expected_serial_wall_hours"] == "6-8"
    assert plan["wandb"] is False
    assert plan["requeue"] is False


def test_parent_dual_hash_lineage_names_noninterchangeable_algorithms() -> None:
    lineage = pilot.parent_model_state_lineage()
    assert lineage == {
        "canonical_model_state_sha256": (
            "d1231b8bc13a2391a94f2ade8ff216de3fbe5e91e7242b35c39c60197fd897a0"
        ),
        "canonical_hash_algorithm": "snapshot_model_state_receipt_v1",
        "runtime_tensor_state_sha256": (
            "82ffa76e99574f5202831897411e4b22eb281c1e2512dc358238bef72d5a4d5a"
        ),
        "runtime_hash_algorithm": "tensor_state_sha256_v1",
    }
    assert (
        lineage["canonical_model_state_sha256"]
        != lineage["runtime_tensor_state_sha256"]
    )


def test_synthetic_memory_fixture_has_production_history_and_future_support(
    monkeypatch,
) -> None:
    def forbidden_dataset_load(*_args, **_kwargs):
        raise AssertionError("synthetic fixture attempted dataset/checkpoint access")

    monkeypatch.setattr(torch, "load", forbidden_dataset_load)
    rgb, temporal_mask = memory_smoke._synthetic_full_geometry_fixture(
        torch, device=torch.device("cpu")
    )
    assert tuple(rgb.shape) == (1, 13, 3, 180, 960)
    assert rgb.dtype == torch.float32
    assert bool(torch.isfinite(rgb).all())
    assert -1.0 <= float(rgb.min()) < float(rgb.max()) <= 1.0
    assert temporal_mask.dtype == torch.bool
    assert temporal_mask.tolist() == [[True] * 13]
    assert int(temporal_mask[:, :5].sum()) == 5
    assert int(temporal_mask[:, 5:].sum()) == 8

    views = (
        rgb.reshape(1, 13, 3, 180, 3, 320)
        .permute(0, 4, 1, 2, 3, 5)
        .reshape(1, 3, -1)
    )
    assert bool((views.std(dim=-1) > 1e-3).all())

    model = _production_loss_mask_model(monkeypatch)
    loss_mask = model._build_loss_mask(
        rgb, temporal_mask, (1, 16, 4, 24, 120)
    )
    assert tuple(loss_mask.shape) == (1, 1, 4, 1, 120)
    assert torch.equal(loss_mask[:, 0, :, 0, 0], torch.ones(1, 4))
    expanded = _expanded_mask(
        loss_mask[:, :, 2:], torch.zeros(1, 16, 2, 24, 120)
    )
    assert expanded.sum(dim=(1, 2, 3, 4)).tolist() == [92160.0]

    # The failed v1 fixture was constant black, so the same production method
    # excluded every view even though its temporal mask was correctly all true.
    black_loss_mask = model._build_loss_mask(
        torch.zeros_like(rgb), temporal_mask, (1, 16, 4, 24, 120)
    )
    with pytest.raises(AdjacentConsistencyError, match="nonempty future support"):
        _expanded_mask(
            black_loss_mask[:, :, 2:], torch.zeros(1, 16, 2, 24, 120)
        )


def test_smoke_trainer_and_evaluator_use_runtime_hash_for_loaded_state() -> None:
    root = Path(pilot.__file__).resolve().parents[1]
    smoke = (root / "tools/adjacent_consistency_memory_smoke.py").read_text()
    trainer = (root / "robot_wm/utils/adjacent_consistency_trainer.py").read_text()
    evaluate = (root / "tools/adjacent_consistency_evaluate.py").read_text()
    for source in (smoke, trainer):
        assert "require_model_state_hashes(" in source
        assert "PARENT_RUNTIME_TENSOR_STATE_SHA256" in source
    parent_branch = evaluate[
        evaluate.index('if endpoint.checkpoint == "parent"') :
        evaluate.index("else:", evaluate.index('if endpoint.checkpoint == "parent"'))
    ]
    assert "PARENT_RUNTIME_TENSOR_STATE_SHA256" in parent_branch
    assert "PARENT_CANONICAL_MODEL_STATE_SHA256" not in parent_branch


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
