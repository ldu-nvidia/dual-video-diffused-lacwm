"""Protocol tests for action-variation evaluation and fixed decision gates."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict

import pytest

from tools import action_variation_evaluate as evaluation
from tools import action_variation_screen as screen
from tools.analyze_action_variation import _effect_gate, paired_interaction_effect


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _registration() -> dict:
    return {
        "identity_sha256": "r" * 64,
        "tool_repository": {"git_commit": "c" * 40},
        "validation_descriptors": [
            {"clip_id": f"clip-{index}", "episode_dir": f"episode-{index}"}
            for index in range(64)
        ],
    }


def _row(index: int, endpoint: screen.Endpoint, arm: screen.Arm) -> dict:
    donor_index = evaluation._expected_action_donor_index(index)
    source = endpoint.action_source
    sampler_action = (
        _hash(f"actions:{index}")
        if source == "aligned"
        else _hash("all-zero-actions")
        if source == "zero"
        else _hash(f"actions:{donor_index}")
    )
    donor = (
        evaluation.VALIDATION_SAMPLE_ID_OFFSET + donor_index
        if source == "global_shuffled"
        else None
    )
    hashes = {
        "cached_rgb_input_sha256": _hash(f"rgb:{index}"),
        "sampler_history_rgb_sha256": _hash(f"history:{index}"),
        "cached_actions_input_sha256": _hash(f"actions:{index}"),
        "sampler_actions_sha256": sampler_action,
        "action_control_sha256": _hash(f"z:{index}:{source}"),
        "wan_action_control_probe_sha256": _hash(f"control:{index}:{source}"),
        "video_clean_scoring_sha256": _hash(f"clean:{index}"),
        "raw_ground_truth_sha256": _hash(f"gt:{index}"),
        "raw_history_last_sha256": _hash(f"last:{index}"),
        "video_initial_noise_sha256": _hash(f"video-noise:{index}"),
        "auxiliary_initial_noise_sha256": _hash(f"aux-noise:{index}"),
        "video_final_sha256": _hash(f"video:{index}:{endpoint.code}"),
        "decoded_final_sha256": _hash(f"decoded:{index}:{endpoint.code}"),
    }
    return screen.identity_payload(
        {
            "schema_version": 1,
            "kind": evaluation.KIND_ROW,
            "registration_identity_sha256": "r" * 64,
            "tool_git_commit": "c" * 40,
            "arm": asdict(arm),
            "arm_snapshot": {"sha256": "s" * 64},
            "evaluation_split": "validation",
            "protected_test_accessed": False,
            "clip_index": index,
            "clip_id": f"clip-{index}",
            "sampling_id": evaluation.VALIDATION_SAMPLE_ID_OFFSET + index,
            "endpoint": asdict(endpoint),
            "action_donor_sampling_id": donor,
            "clean_future_rgb_passed_to_sampler": False,
            "clean_video_latent_passed_to_sampler": False,
            "clean_auxiliary_passed_to_sampler": False,
            "target_cache_array_opened": False,
            "online_feature_or_teacher_call_count": 0,
            "scoring_constructed_after_all_sampling": True,
            "history_rgb_frames": 5,
            "future_rgb_frames": 8,
            "history_video_latent_tokens": 2,
            "future_video_latent_tokens": 2,
            "actual_transformer_call_count": endpoint.nfe,
            "declared_nfe": endpoint.nfe,
            "sampler_action_abs_max": 0.0 if source == "zero" else 1.0,
            "wan_action_control_probe": [float(index + offset) for offset in range(32)],
            "latency_ms_per_local_batch": {
                "prepare_history_and_action": 1.0,
                "wan_trajectory": 2.0,
                "decode": 3.0,
                "total": 6.0,
            },
            "evaluator_only_action_control_probe_ms": 0.25,
            "metrics": {
                "video_future_nmse": 1.0,
                "video_future_temporal_delta_nmse": 1.0,
                "decoded_mse_unit_range": 1.0,
                "decoded_psnr_db": 1.0,
                "decoded_temporal_difference_mse_unit_range": 1.0,
            },
            "tensor_sha256": hashes,
        }
    )


def _rows(arm: screen.Arm):
    return [
        _row(index, endpoint, arm)
        for index in range(64)
        for endpoint in screen.ENDPOINTS
    ]


def test_endpoint_grid_has_aligned_quality_and_zero_global_shuffle_attribution():
    assert [endpoint.code for endpoint in screen.ENDPOINTS] == [
        "aligned_nfe_1",
        "aligned_nfe_2",
        "aligned_nfe_4",
        "zero_nfe_1",
        "global_shuffled_nfe_1",
        "aligned_residual_masked_nfe_1",
    ]
    assert screen.fixed_protocol()["protected_test_access_allowed"] is False
    assert screen.fixed_protocol()["future_rgb_or_feature_allowed_at_sampling"] is False


def test_rows_require_exact_calls_no_future_inputs_and_episode_disjoint_donor():
    arm = screen.ARM_BY_CODE["AV-CONT"]
    rows = _rows(arm)
    evaluation._validate_rows(rows, arm, _registration())
    changed = dict(rows[0])
    changed["clean_future_rgb_passed_to_sampler"] = True
    changed.pop("identity_sha256")
    rows[0] = screen.identity_payload(changed)
    with pytest.raises(evaluation.ActionVariationEvaluationError, match="protocol"):
        evaluation._validate_rows(rows, arm, _registration())


def test_rows_reject_changed_paired_noise():
    arm = screen.ARM_BY_CODE["AV-DELTA"]
    rows = _rows(arm)
    changed = dict(rows[1])
    hashes = dict(changed["tensor_sha256"])
    hashes["video_initial_noise_sha256"] = _hash("different")
    changed["tensor_sha256"] = hashes
    changed.pop("identity_sha256")
    rows[1] = screen.identity_payload(changed)
    with pytest.raises(evaluation.ActionVariationEvaluationError, match="noise"):
        evaluation._validate_rows(rows, arm, _registration())


def test_decision_gate_requires_primary_bound_and_guardrails():
    passing = {
        "decoded_temporal_difference_mse_unit_range": {
            "relative_improvement": 0.02,
            "one_sided_simultaneous_lower_bound": {"low": 0.011},
        },
        "video_future_nmse": {
            "relative_improvement": 0.0,
            "one_sided_simultaneous_lower_bound": {"low": -0.005},
        },
        "decoded_mse_unit_range": {
            "relative_improvement": 0.0,
            "one_sided_simultaneous_lower_bound": {"low": -0.005},
        },
    }
    assert _effect_gate(passing, primary_minimum=0.01)["passed"]
    failing = {key: dict(value) for key, value in passing.items()}
    failing["decoded_temporal_difference_mse_unit_range"] = {
        "relative_improvement": 0.02,
        "one_sided_simultaneous_lower_bound": {"low": 0.009},
    }
    assert not _effect_gate(failing, primary_minimum=0.01)["passed"]


def test_difference_in_differences_rejects_generic_candidate_gain():
    control_aligned = [1.0] * 64
    control_diagnostic = [1.1] * 64
    candidate_aligned = [0.98] * 64
    # Same two-percent generic gain at both endpoints: candidate retains only
    # the inherited control action gap, so incremental specificity is zero.
    candidate_diagnostic = [1.08] * 64
    effect = paired_interaction_effect(
        candidate_aligned,
        candidate_diagnostic,
        control_aligned,
        control_diagnostic,
        label="generic-gain",
        contrast_count=6,
    )
    assert abs(effect["relative_improvement"]) < 1e-12
    assert effect["one_sided_simultaneous_lower_bound"]["low"] < 0.005


def test_cli_has_no_test_or_feature_target_argument():
    for parser in (screen._parser(), evaluation._parser()):
        actions = list(parser._actions)
        subparsers = [
            action
            for action in actions
            if isinstance(getattr(action, "choices", None), Mapping)
        ]
        for action in subparsers:
            for child in action.choices.values():
                actions.extend(child._actions)
        options = [option for action in actions for option in action.option_strings]
        assert not any(
            forbidden in option
            for option in options
            for forbidden in ("test", "lockbox", "future", "teacher", "feature")
        )
