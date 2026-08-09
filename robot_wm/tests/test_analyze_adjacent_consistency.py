"""Frozen gate and exact endpoint-pairing tests for ACD-P0."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from tools import analyze_adjacent_consistency as analysis
from tools import low_nfe_direct_baseline as pilot


def _registration() -> dict:
    return {
        "identity_sha256": "a" * 64,
        "tool_repository": {"git_commit": "b" * 40},
        "validation_descriptors": [
            {"clip_id": f"clip-{index}", "episode_dir": f"episode-{index}"}
            for index in range(64)
        ],
    }


def _metric_value(endpoint: str, metric: str, *, weak_actions: bool) -> float:
    if endpoint in {analysis.PRIMARY_CANDIDATE, analysis.ONLINE_CANDIDATE}:
        return 0.90 if "decoded" in metric else 0.995
    if endpoint == analysis.SHUFFLED_CANDIDATE:
        if metric in {
            "decoded_mse_unit_range",
            "decoded_temporal_difference_mse_unit_range",
        }:
            return 0.91 if weak_actions else 1.20
        return 1.20
    if endpoint == analysis.SHUFFLED_PARENT:
        return 1.20
    return 1.00


def _rows(*, weak_actions: bool = False) -> list[dict]:
    rows = []
    for endpoint in pilot.ENDPOINTS:
        for nfe in pilot.NFE_GRID:
            for noise_seed in pilot.NOISE_SEEDS:
                for clip_index in range(64):
                    metrics = {
                        metric: _metric_value(
                            endpoint.code, metric, weak_actions=weak_actions
                        )
                        for metric in analysis.METRICS
                    }
                    rows.append(
                        {
                            "endpoint": asdict(endpoint),
                            "nfe": nfe,
                            "noise_seed": noise_seed,
                            "clip_index": clip_index,
                            "metrics": metrics,
                            "batch_key": (
                                f"{endpoint.code}:{nfe}:{noise_seed}:{clip_index // 2}"
                            ),
                            "latency": {
                                "complete_sampler_seconds_per_batch": [
                                    1.02
                                    if endpoint.code == analysis.PRIMARY_CANDIDATE
                                    else 1.0
                                ]
                                * 4
                            },
                        }
                    )
    return rows


def _inventory() -> dict:
    return {
        "identity_sha256": "c" * 64,
        "observed_reserved_b200_hours": 20.0,
        "artifact_bytes_under_registered_root": 1 << 30,
    }


def test_frozen_gate_selects_lowest_passing_nfe(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "BOOTSTRAPS", 100)
    result = analysis.analyze(_registration(), _inventory(), _rows())
    assert result["decision"] == "GO_ACD"
    assert result["selected_nfe"] == 1
    assert result["passing_nfe"] == [1, 2, 4]
    assert result["comparisons"]["1"]["action_sensitivity_gate"] is True


def test_shuffled_action_guard_can_veto_quality_gain(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "BOOTSTRAPS", 100)
    result = analysis.analyze(
        _registration(), _inventory(), _rows(weak_actions=True)
    )
    assert result["decision"] == "NO_GO_ACD"
    assert all(
        row["control_quality_gate"] and not row["action_sensitivity_gate"]
        for row in result["comparisons"].values()
    )


def _paired_family() -> list[dict]:
    rows = []
    for endpoint in pilot.ENDPOINTS:
        shuffled = endpoint.action_source == "episode_shuffled"
        rows.append(
            {
                "endpoint": asdict(endpoint),
                "nfe": 1,
                "noise_seed": pilot.NOISE_SEEDS[0],
                "clip_index": 0,
                "action_donor_clip_index": 1 if shuffled else 0,
                "tensor_sha256": {
                    "history_rgb": "history",
                    "noise_stream": "noise",
                    "clean_latent_scoring": "latent",
                    "raw_full_rgb_scoring": "rgb",
                    "actions": "shuffled" if shuffled else "aligned",
                },
            }
        )
    return rows


def test_pairing_rejects_cross_endpoint_noise_or_same_episode_donor() -> None:
    registration = _registration()
    rows = _paired_family()
    analysis._validate_paired_rows(rows, registration)

    rows[0]["tensor_sha256"]["noise_stream"] = "different"
    with pytest.raises(analysis.ACDAnalysisError, match="paired noise_stream"):
        analysis._validate_paired_rows(rows, registration)

    rows = _paired_family()
    registration["validation_descriptors"][1]["episode_dir"] = "episode-0"
    with pytest.raises(analysis.ACDAnalysisError, match="donor episode"):
        analysis._validate_paired_rows(rows, registration)
