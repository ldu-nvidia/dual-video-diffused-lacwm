"""Focused tests for the preregistered CAMP statistical gate."""

from __future__ import annotations

import numpy as np
import pytest

from tools import causal_motion_plan_analyze as analyze


def test_paired_relative_improvement_and_fixed_thresholds() -> None:
    reference = np.ones(64, dtype=np.float64)
    candidate = np.full(64, 0.8, dtype=np.float64)
    bootstrap = np.tile(np.arange(64), (100, 1))
    effect = analyze.paired_relative_improvement(
        candidate, reference, bootstrap_indexes=bootstrap
    )
    assert effect["relative_improvement"] == pytest.approx(0.2)
    assert effect["simultaneous_one_sided_lower_bound"] == pytest.approx(0.2)
    assert analyze._cell_pass(
        "decoded_temporal_difference_mse_unit_range", effect
    )
    assert analyze._cell_pass("video_future_nmse", effect)


def test_temporal_gate_requires_three_percent_point_and_one_percent_bound() -> None:
    assert not analyze._cell_pass(
        "decoded_temporal_difference_mse_unit_range",
        {
            "relative_improvement": 0.0299,
            "simultaneous_one_sided_lower_bound": 0.02,
        },
    )
    assert not analyze._cell_pass(
        "decoded_temporal_difference_mse_unit_range",
        {
            "relative_improvement": 0.04,
            "simultaneous_one_sided_lower_bound": 0.0099,
        },
    )


def test_paired_effect_rejects_zero_reference_energy() -> None:
    with pytest.raises(analyze.CAMPAnalysisError, match="invalid"):
        analyze.paired_relative_improvement(
            np.ones(2),
            np.zeros(2),
            bootstrap_indexes=np.zeros((10, 2), dtype=np.int64),
        )
