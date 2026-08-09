from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage = _load("physics_flow_stage1_test", "tools/physics_flow_stage1.py")
snapshot_receipt = _load(
    "snapshot_model_state_receipt_test",
    "tools/snapshot_model_state_receipt.py",
)


def test_endpoint_grid_is_equal_call_raw_only() -> None:
    assert len(stage.ENDPOINTS) == 18
    assert {endpoint.arm for endpoint in stage.ENDPOINTS} == {"FLOW-OFF", "RAW-FLOW"}
    assert {
        endpoint.condition_source
        for endpoint in stage.ENDPOINTS
        if endpoint.arm == "RAW-FLOW"
    } == set(stage.RUNTIME_SOURCES)
    assert all(endpoint.nfe in (1, 2, 4) for endpoint in stage.ENDPOINTS)
    assert all(
        endpoint.condition_source == "off"
        for endpoint in stage.ENDPOINTS
        if endpoint.arm == "FLOW-OFF"
    )
    source = (ROOT / "tools/physics_flow_stage1.py").read_text()
    assert "recurrent_delta" not in source
    assert "hybrid_anchor_raw_delta" not in source
    assert stage.PARENT_SNAPSHOT_SHA256.startswith("de65e832")
    assert stage.PARENT_RUN_IDENTITY_SHA256.startswith("d79c3699")
    assert stage.PARENT_CANONICAL_MODEL_STATE_SHA256.startswith("d1231b8b")
    assert stage.LEGACY_REJECTED_SNAPSHOT_SHA256.startswith("f67c7bae")


def test_snapshot_receipt_hashes_scalar_and_records_lineage(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.pt"
    torch.save(
        {
            "model": {
                "scalar": torch.tensor(0.25),
                "matrix": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            },
            "snapshot_schema_version": 3,
            "run_identity_sha256": "a" * 64,
            "_start_iter": 1,
            "world_size": 1,
            "gradient_accumulation_steps": 1,
        },
        path,
    )
    receipt, tensors = snapshot_receipt.model_state_receipt(path)
    assert receipt["model_tensor_count"] == 2
    assert len(receipt["canonical_model_state_sha256"]) == 64
    assert set(tensors) == {"matrix", "scalar"}


def test_support_weighted_pool_and_bottom_padding() -> None:
    native = np.zeros((4, 180, 320), dtype=np.float32)
    native[0] = 0.25
    native[1] = -0.5
    native[2] = 1.0
    native[3] = 0.125
    pooled = stage.pool_transition_field(native)
    assert pooled.shape == (4, 24, 40)
    np.testing.assert_array_equal(pooled[0, :23], 0.25)
    np.testing.assert_array_equal(pooled[1, :23], -0.5)
    np.testing.assert_array_equal(pooled[3, :23], 0.125)
    np.testing.assert_array_equal(pooled[2, :22], 1.0)
    np.testing.assert_array_equal(pooled[2, 22], 0.5)
    np.testing.assert_array_equal(pooled[:, 23], 0.0)


def test_support_weighting_does_not_attenuate_motion() -> None:
    native = np.zeros((4, 180, 320), dtype=np.float32)
    native[0, 0:4, 0:4] = 0.4
    native[1, 0:4, 0:4] = -0.2
    native[2, 0:4, 0:4] = 1.0
    native[3, 0:4, 0:4] = 0.1
    pooled = stage.pool_transition_field(native)
    assert pooled[2, 0, 0] == 0.25
    assert pooled[0, 0, 0] == np.float32(0.4)
    assert pooled[1, 0, 0] == np.float32(-0.2)
    assert pooled[3, 0, 0] == np.float32(0.1)
    assert np.count_nonzero(pooled[:, 1:, :]) == 0


def test_compact_control_shift_is_nonwrapping() -> None:
    value = np.zeros((2, 8, 4, 24, 40), dtype=np.float16)
    for transition in range(8):
        value[:, transition, 2] = 1
        value[:, transition, 0] = transition / 10
    shifted = stage.nonwrapping_timeshift(value)
    np.testing.assert_array_equal(shifted[:, :-1], value[:, 1:])
    assert np.count_nonzero(shifted[:, -1]) == 0
    stage.validate_compact_flow(value, count=2)
    stage.validate_compact_flow(shifted, count=2)


def test_cluster_bootstrap_keeps_noise_seeds_inside_episode() -> None:
    candidate = []
    reference = []
    for clip in range(6):
        for seed in stage.NOISE_SEEDS:
            candidate.append(
                {
                    "clip_index": clip,
                    "noise_seed_id": seed,
                    "metrics": {"metric": 0.8 + clip / 1000 + seed / 10000},
                }
            )
            reference.append(
                {
                    "clip_index": clip,
                    "noise_seed_id": seed,
                    "metrics": {"metric": 1.0 + clip / 1000 + seed / 10000},
                }
            )
    effect = stage._paired_cluster_effect(candidate, reference, "metric")
    assert effect["paired_episode_clusters"] == 6
    assert effect["noise_seeds_per_cluster"] == 4
    assert effect["relative_improvement_percent"] > 19
    assert effect["paired_episode_cluster_bootstrap_95_ci_percent"][0] > 0


def test_protocol_and_launcher_have_causal_guards() -> None:
    protocol = (
        ROOT / "docs/experiments/PHYSICS_FLOW_WAN_SCREEN_PROTOCOL.md"
    ).read_text()
    launcher = (ROOT / "tools/slurm/physics_flow_stage1.sbatch").read_text()
    evaluator = (ROOT / "tools/physics_flow_stage1_evaluate.py").read_text()
    for token in (
        "episode_shuffled",
        "timeshift_plus_one",
        "hold_current",
        "future measured state",
        "ADVANCE_RAW_FLOW_SCAFFOLD",
        "04f5013b7161fbf91ed6116d25f7e6ec66afc661024236ad27564b1899cb94be",
        "MEASURED_GEOMETRY_ORACLE",
    ):
        assert token in protocol
    assert "--no-requeue" in launcher
    assert "compare-traces" in launcher
    assert "every endpoint for every registered" in evaluator
    assert evaluator.index("materialized_by_seed.append") < evaluator.index(
        "dataset.scoring_batch"
    )
    lpips_helper = (ROOT / "tools/physics_flow_lpips.py").read_text()
    assert "offline AlexNet checkpoint is absent" in lpips_helper
    assert "network access is forbidden" in lpips_helper
    assert "loaded_state_dict_sha256" in lpips_helper
