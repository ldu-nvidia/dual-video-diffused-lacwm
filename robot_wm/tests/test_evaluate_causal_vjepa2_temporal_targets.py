from __future__ import annotations

import inspect
import copy
import subprocess
from types import SimpleNamespace

import pytest
import torch

from tools import evaluate_causal_vjepa2_temporal_targets as evaluator


class _ZeroAuxiliaryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        noisy_video,
        noisy_auxiliary,
        t_video,
        t_auxiliary,
        history,
        actions,
        *,
        auxiliary_fusion_mask,
        predict_video,
    ):
        del t_video, t_auxiliary, history, actions
        assert auxiliary_fusion_mask is True
        assert predict_video is False
        self.calls += 1
        return SimpleNamespace(
            video_x=torch.zeros_like(noisy_video),
            auxiliary_x=torch.zeros_like(noisy_auxiliary),
        )


def test_generated_sampler_api_is_clean_data_free_float32_and_exact_call_count():
    names = set(inspect.signature(evaluator.sample_generated_temporal).parameters)
    assert not any(
        term in name for name in names for term in evaluator.FORBIDDEN_SAMPLER_TERMS
    )
    model = _ZeroAuxiliaryModel()
    history = torch.randn(2, 3, 5, 4, 4)
    actions = torch.randn(2, 16, 7)
    video_noise = torch.randn(2, 3, 8, 4, 4, dtype=torch.float32)
    representation_noise = torch.randn(2, 4, 2, 2, 2, dtype=torch.float32)
    result = evaluator.sample_generated_temporal(
        model,
        history,
        actions,
        video_noise=video_noise,
        representation_noise=representation_noise,
        steps=2,
        representation_mode="absolute",
        normalization=None,
    )
    assert model.calls == result.model_calls == 2
    assert result.representation_prediction.dtype == torch.float32
    assert result.semantic_prediction.dtype == torch.float32
    assert all(len(trace) == 2 for trace in result.call_input_sha256_by_example)


def test_training_source_compatibility_requires_exact_inference_git_objects():
    commit = subprocess.run(
        ["git", "-C", str(evaluator.REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = {"commit": commit, "branch": "test", "dirty": False}
    result = evaluator.training_source_compatibility(source, source)
    assert result["training_is_ancestor"] is True
    assert result["inference_critical_paths_unchanged"] is True
    assert set(result["paths"]) == set(evaluator.INFERENCE_CRITICAL_PATHS)


def test_generated_sampler_rejects_non_float32_trajectory_state():
    with pytest.raises(ValueError, match="float32"):
        evaluator.sample_generated_temporal(
            _ZeroAuxiliaryModel(),
            torch.randn(1, 3, 5, 2, 2),
            torch.randn(1, 16, 7),
            video_noise=torch.randn(1, 3, 8, 2, 2, dtype=torch.float16),
            representation_noise=torch.randn(1, 4, 2, 2, 2),
            steps=1,
            representation_mode="absolute",
            normalization=None,
        )


def test_sampler_input_hash_has_only_generated_inputs_and_representation_binding():
    record = evaluator._sampler_input_record(  # noqa: SLF001
        torch.randn(3, 5, 2, 2),
        torch.randn(16, 7),
        torch.randn(3, 8, 2, 2),
        torch.randn(4, 2, 2, 2),
        representation_mode="delta_pack",
        normalization_binding="a" * 64,
    )
    assert set(record) == {
        "history_sha256",
        "actions_sha256",
        "initial_video_noise_sha256",
        "initial_representation_noise_sha256",
        "representation_mode",
        "normalization_binding_sha256",
    }
    assert not any(term in record for term in evaluator.FORBIDDEN_SAMPLER_TERMS)
    assert len(evaluator.sampler_input_sha256(record)) == 64
    with pytest.raises(ValueError, match="malformed|leaks"):
        evaluator.sampler_input_sha256({**record, "future": "b" * 64})


def test_registered_action_permutations_are_bijective_fixed_point_free_and_episode_disjoint():
    rows = [
        {"clip_id": f"clip-{index:04d}", "episode_index": 10_000 + index}
        for index in range(890)
    ]
    records = evaluator.validate_action_permutations(rows)
    assert set(records) == set(evaluator.ACTION_CONTROLS)
    for offset in evaluator.ACTION_PERMUTATION_OFFSETS:
        permutation = evaluator.action_permutation_indices(len(rows), offset)
        assert len(set(permutation)) == len(rows)
        assert all(index != source for index, source in enumerate(permutation))
        assert records[evaluator.action_control(offset)]["episode_disjoint"] is True


@pytest.mark.parametrize("arm", evaluator.DOE_ARMS)
def test_arm_contract_accepts_only_the_preregistered_configuration(arm):
    expected = evaluator.ARM_CONTRACTS[arm]
    config = {
        "doe_arm": arm,
        "target_mode": expected["target_mode"],
        "normalization": {"path": "/immutable"} if expected["normalization"] else None,
        "loss": {
            "flow_weight": 1.0,
            "normalized_temporal_velocity_weight": expected["temporal_weight"],
            "action_shuffle_margin_weight": 0.0,
        },
        "self_rollin": {
            "probability": expected["rollin_probability"],
            "later_time_rule": "sampled_final_clock_fraction",
        },
    }
    assert evaluator._arm_contract(config) == (arm, expected["target_mode"])  # noqa: SLF001
    config["loss"]["flow_weight"] = 0.5
    with pytest.raises(evaluator.TemporalEvaluationError, match="implement"):
        evaluator._arm_contract(config)  # noqa: SLF001


def test_doe_common_identity_ignores_only_registered_mechanism_fields():
    base = {
        "doe_arm": "ABS",
        "promotion_status": "reference_not_promotable",
        "target_mode": "absolute",
        "normalization": None,
        "experiment_identity_sha256": "a" * 64,
        "world_size": 8,
        "micro_batch_size_per_rank": 32,
        "loss": {
            "flow_weight": 1.0,
            "normalized_temporal_velocity_weight": 0.0,
        },
        "self_rollin": {"probability": 0.0, "later_time_rule": "sampled"},
    }
    candidate = copy.deepcopy(base)
    candidate.update(
        {
            "doe_arm": "DELTA-R",
            "promotion_status": "primary_candidate",
            "target_mode": "delta_pack",
            "normalization": {"sha256": evaluator.FROZEN_D1_NORMALIZATION_SHA256},
            "experiment_identity_sha256": "b" * 64,
        }
    )
    candidate["self_rollin"]["probability"] = 0.5
    assert evaluator.training_doe_common_identity(
        base
    ) == evaluator.training_doe_common_identity(candidate)
    candidate["world_size"] = 4
    assert evaluator.training_doe_common_identity(
        base
    ) != evaluator.training_doe_common_identity(candidate)


def test_delta_metrics_are_primary_only_after_decode_and_keep_packed_diagnostic():
    normalization = evaluator.temporal.TemporalNormalization(
        anchor_mean=torch.zeros(evaluator.temporal.CHANNELS),
        anchor_std=torch.ones(evaluator.temporal.CHANNELS),
        delta_mean=torch.zeros(evaluator.temporal.CHANNELS),
        delta_std=torch.full((evaluator.temporal.CHANNELS,), 2.0),
        provenance={"test": True},
    )
    target = torch.randn(2, *evaluator.temporal.TARGET_SHAPE)
    representation = normalization.encode(target, "delta_pack")
    decoded = normalization.decode(representation, "delta_pack")
    metrics = evaluator._metric_bundle(  # noqa: SLF001
        decoded, target, representation, representation
    )
    assert float(metrics["semantic_nmse"].max()) < 1e-10
    assert float(metrics["packed_semantic_nmse"].max()) == 0.0
    assert float(metrics["semantic_token_cosine"].min()) > 0.99999
