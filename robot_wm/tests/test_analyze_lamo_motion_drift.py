"""Tests for the fixed paired LaMo validation decision rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import analyze_lamo_motion_drift as analysis
from tools import lamo_motion_drift_evaluate as evaluation


def _effect(point: float, low: float) -> dict:
    return {
        "relative_improvement": point,
        "one_sided_simultaneous_lower_bound": {"low": low},
    }


def test_fixed_family_and_gate_require_temporal_gain_and_guardrails():
    assert analysis.CONTRAST_COUNT == 9
    assert analysis.ONE_SIDED_CONFIDENCE == pytest.approx(1.0 - 0.05 / 9)
    effects = {
        analysis.PRIMARY_METRIC: _effect(0.02, 0.011),
        "video_future_nmse": _effect(0.0, -0.009),
        "decoded_mse_unit_range": _effect(0.01, -0.005),
    }
    assert analysis.endpoint_gate(effects)["passed"] is True
    effects[analysis.PRIMARY_METRIC] = _effect(0.02, 0.009)
    assert analysis.endpoint_gate(effects)["passed"] is False


def test_paired_effect_sign_is_positive_when_drift_arm_is_lower():
    result = analysis.paired_effect(
        [0.8] * evaluation.EXPECTED_VALIDATION_CLIPS,
        [1.0] * evaluation.EXPECTED_VALIDATION_CLIPS,
        label="unit",
        bootstrap_samples=500,
        confidence=0.95,
    )
    assert result["relative_improvement"] == pytest.approx(0.2)
    assert result["one_sided_simultaneous_lower_bound"]["low"] > 0


def _pair_row(noise: str) -> dict:
    fields = (
        "cached_rgb_input_sha256",
        "sampler_history_rgb_sha256",
        "cached_actions_input_sha256",
        "sampler_actions_sha256",
        "video_clean_scoring_sha256",
        "raw_ground_truth_sha256",
        "raw_history_last_sha256",
        "video_initial_noise_sha256",
        "tf_initial_noise_sha256",
    )
    hashes = {field: field for field in fields}
    hashes["video_initial_noise_sha256"] = noise
    return {
        "clip_id": "clip",
        "sampling_id": 2_000_000,
        "actual_transformer_call_count": 1,
        "tensor_sha256": hashes,
    }


def test_cross_arm_pairing_detects_noise_mismatch():
    key = (0, "autonomous_nfe_1")
    analysis._assert_cross_arm_pairing({key: _pair_row("same")}, {key: _pair_row("same")})
    with pytest.raises(analysis.MotionDriftAnalysisError, match="pairing differs"):
        analysis._assert_cross_arm_pairing(
            {key: _pair_row("candidate")}, {key: _pair_row("reference")}
        )


def _write_trace(path: Path, *, mismatch_update: int | None = None) -> dict:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "lamo_motion_drift_training_trace_header"}) + "\n")
        for update in range(200):
            metrics = {
                "iteration": update,
                "learning_rate": 1e-4 * (200 - update) / 200,
                "train_loss/loss": 1.0,
            }
            for field_index, field in enumerate(analysis.AUDIT_FIELDS):
                metrics[field] = float(update + field_index)
            for field_index, field in enumerate(analysis.AUDIT_HASH_FIELDS):
                metrics[field] = f"{update + field_index:064x}"
            if mismatch_update == update:
                metrics[analysis.AUDIT_FIELDS[0]] += 0.5
            handle.write(
                json.dumps(
                    {
                        "kind": "lamo_motion_drift_training_trace_event",
                        "total_observations": (update + 1) * 8,
                        "metrics": metrics,
                    }
                )
                + "\n"
            )
    return {
        "arm_artifacts": {
            "training_trace": {
                "trace": {"path": str(path), "sha256": evaluation._sha256(path)}
            }
        }
    }


def test_training_pair_requires_all_200_exact_audit_updates(tmp_path):
    candidate = _write_trace(tmp_path / "candidate.jsonl")
    reference = _write_trace(tmp_path / "reference.jsonl")
    assert analysis._assert_training_pair(candidate, reference)["updates"] == 200

    mismatched = _write_trace(tmp_path / "mismatch.jsonl", mismatch_update=37)
    with pytest.raises(analysis.MotionDriftAnalysisError, match="update 37"):
        analysis._assert_training_pair(mismatched, reference)
