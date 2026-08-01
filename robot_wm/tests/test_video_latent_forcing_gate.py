from __future__ import annotations

import json

import numpy as np
import pytest

from tools import analyze_video_latent_forcing_poc as gate


def _constant_values(size: int) -> dict[str, dict[str, np.ndarray]]:
    def values(lpips: float, temporal: float, nmse: float, cosine: float):
        return {
            "lpips_alex_frame": np.full(size, lpips, dtype=np.float64),
            "temporal_difference_mse": np.full(size, temporal, dtype=np.float64),
            "auxiliary_nmse": np.full(size, nmse, dtype=np.float64),
            "auxiliary_cosine": np.full(size, cosine, dtype=np.float64),
        }

    return {
        "autonomous": values(0.4, 0.4, 0.2, 0.9),
        "off": values(0.8, 0.8, 0.2, 0.9),
        "shuffled": values(0.8, 0.8, 0.2, 0.9),
        "oracle_clean": values(0.2, 0.2, 0.2, 0.9),
        "context_shuffled": values(0.4, 0.4, 0.4, 0.7),
    }


def test_phase1_gate_cell_passes_only_complete_representation_screen(monkeypatch):
    monkeypatch.setattr(gate, "CLIPS", 8)
    monkeypatch.setattr(gate, "BOOTSTRAP_SAMPLES", 64)
    decision = gate._gate_cell(4, _constant_values(8))
    assert decision["passed"] is True
    assert decision["nfe_pair"] == [4, 0]
    assert {check["id"] for check in decision["checks"]} == {
        "auxiliary_nmse",
        "auxiliary_cosine",
        "lpips_alex_frame_vs_off",
        "lpips_alex_frame_vs_shuffled",
        "lpips_alex_frame_retained_utility",
        "temporal_difference_mse_vs_off",
        "temporal_difference_mse_vs_shuffled",
        "temporal_difference_mse_retained_utility",
        "context_causality_nmse",
        "context_causality_cosine",
    }


def test_nonpositive_oracle_utility_denominator_is_valid_json_failure(monkeypatch):
    monkeypatch.setattr(gate, "CLIPS", 8)
    monkeypatch.setattr(gate, "BOOTSTRAP_SAMPLES", 64)
    values = _constant_values(8)
    values["oracle_clean"]["lpips_alex_frame"][:] = 0.9
    decision = gate._gate_cell(4, values)
    retained = next(
        check
        for check in decision["checks"]
        if check["id"] == "lpips_alex_frame_retained_utility"
    )
    assert retained["passed"] is False
    assert retained["estimate"] is None
    assert decision["passed"] is False
    assert "NaN" not in json.dumps(decision, allow_nan=False)


def test_paired_bootstrap_is_label_deterministic_and_label_separated(monkeypatch):
    monkeypatch.setattr(gate, "CLIPS", 8)
    monkeypatch.setattr(gate, "BOOTSTRAP_SAMPLES", 64)
    first = np.arange(1, 9, dtype=np.float64)
    second = first + np.arange(2, 10, dtype=np.float64)
    one = gate.paired_bootstrap(
        first,
        second,
        label="registered-statistic",
        statistic=gate.relative_improvement,
    )
    two = gate.paired_bootstrap(
        first,
        second,
        label="registered-statistic",
        statistic=gate.relative_improvement,
    )
    other = gate.paired_bootstrap(
        first,
        second,
        label="different-statistic",
        statistic=gate.relative_improvement,
    )
    assert one == two
    assert one["seed"] != other["seed"]


def test_bootstrap_rejects_nonfinite_or_unpaired_inputs(monkeypatch):
    monkeypatch.setattr(gate, "CLIPS", 8)
    monkeypatch.setattr(gate, "BOOTSTRAP_SAMPLES", 8)
    with pytest.raises(gate.GateError, match="exactly 8 aligned clips"):
        gate.paired_bootstrap(
            np.ones(7),
            np.ones(7),
            label="bad-size",
            statistic=gate.relative_improvement,
        )
    with pytest.raises(gate.GateError, match="nonfinite point statistic"):
        gate.paired_bootstrap(
            np.ones(8),
            np.zeros(8),
            label="zero-reference",
            statistic=gate.relative_improvement,
        )


def test_output_validation_rejects_evidence_and_git_roots(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    with pytest.raises(gate.GateError, match="outside both evidence and Git roots"):
        gate.validate_output_path(evidence / "gate.json", evidence)
    with pytest.raises(gate.GateError, match="under /lustre"):
        gate.validate_output_path(tmp_path / "gate.json", evidence)
