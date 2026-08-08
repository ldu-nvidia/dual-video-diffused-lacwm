from __future__ import annotations

import pytest

from tools import analyze_vpm_phaselock_probe as analysis


def _effect(point: float, low: float) -> dict:
    return {
        "relative_improvement": point,
        "one_sided_simultaneous_lower_bound": {
            "confidence": analysis.ONE_SIDED_CONFIDENCE,
            "low": low,
        },
    }


def test_simultaneous_confidence_matches_twelve_fixed_contrasts() -> None:
    assert analysis.CONTRAST_COUNT == 12
    assert analysis.ONE_SIDED_CONFIDENCE == pytest.approx(1.0 - 0.05 / 12.0)


def test_contrast_gate_requires_temporal_benefit_and_both_guardrails() -> None:
    effects = {
        analysis.PRIMARY_METRIC: _effect(0.03, 0.02),
        "video_future_nmse": _effect(0.0, -0.005),
        "decoded_mse_unit_range": _effect(0.01, -0.004),
    }
    assert analysis.contrast_gate(effects)["passed"] is True
    effects[analysis.PRIMARY_METRIC] = _effect(0.03, 0.009)
    assert analysis.contrast_gate(effects)["passed"] is False
    effects[analysis.PRIMARY_METRIC] = _effect(0.03, 0.02)
    effects["decoded_mse_unit_range"] = _effect(-0.011, -0.012)
    assert analysis.contrast_gate(effects)["passed"] is False


def test_paired_effect_sign_is_positive_when_candidate_is_lower() -> None:
    result = analysis.paired_effect(
        [0.8, 0.9, 1.0, 0.7],
        [1.0, 1.1, 1.2, 0.9],
        label="unit-test",
        bootstrap_samples=500,
        one_sided_confidence=0.95,
    )
    assert result["relative_improvement"] > 0
    assert result["one_sided_simultaneous_lower_bound"]["low"] > 0
    assert result["bootstrap_label"] == "unit-test"
    assert result["bootstrap_derived_seed"] == analysis._derived_seed(  # noqa: SLF001
        "unit-test"
    )


def test_candidate_grid_has_three_predeclared_controls_each() -> None:
    assert analysis.CANDIDATES == ("k1_f2", "k1_f3", "k2_f2", "k2_f4")
    for candidate in analysis.CANDIDATES:
        aligned = analysis.probe.ENDPOINT_BY_CODE[f"phaselock_{candidate}_aligned"]
        assert f"phaselock_{candidate}_shuffled" in analysis.probe.ENDPOINT_BY_CODE
        assert (
            f"ordinary_b{aligned.total_transformer_calls}"
            in analysis.probe.ENDPOINT_BY_CODE
        )
    assert "ordinary_b1" in analysis.probe.ENDPOINT_BY_CODE
