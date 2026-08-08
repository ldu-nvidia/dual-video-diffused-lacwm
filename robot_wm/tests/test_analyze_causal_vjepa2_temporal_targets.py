from __future__ import annotations

import numpy as np
import pytest

from tools import analyze_causal_vjepa2_temporal_targets as analyzer
from tools import evaluate_causal_vjepa2_temporal_targets as evaluator
from tools import video_latent_forcing_poc as vlf


def _metric_vectors(clips: int, ordinary: float, temporal: float, cosine: float):
    base = {
        "semantic_nmse": np.full(clips, ordinary, dtype=np.float64),
        "semantic_token_cosine": np.full(clips, cosine, dtype=np.float64),
        "temporal_difference_nmse": np.full(clips, temporal, dtype=np.float64),
        "temporal_difference_token_cosine": np.full(
            clips, cosine - 0.02, dtype=np.float64
        ),
        "retained_utility": np.full(clips, 1.0 - ordinary, dtype=np.float64),
        "temporal_retained_utility": np.full(
            clips, 1.0 - temporal, dtype=np.float64
        ),
    }
    return {**base, **{f"packed_{name}": value.copy() for name, value in base.items()}}


def _evaluations(clips: int = 8):
    clip_ids = tuple(f"clip-{index:03d}" for index in range(clips))
    manifest = {"sha256": "1" * 64, "clips": clips, "split": "val"}
    evaluations = {}
    for arm in analyzer.ARMS:
        cells = {}
        for nfe in evaluator.NFE_GRID:
            for control in evaluator.CONTROLS:
                ordinary, temporal, cosine = 0.62, 0.66, 0.65
                if arm == "ABS" and control == "autonomous":
                    ordinary, temporal, cosine = 0.60, 0.64, 0.66
                if arm == "DELTA-T" and nfe == 2:
                    if control == "autonomous":
                        ordinary, temporal, cosine = 0.35, 0.36, 0.82
                    elif control in {"donor_target", "context_shuffled"}:
                        ordinary, temporal, cosine = 0.60, 0.62, 0.61
                    elif control in evaluator.ACTION_CONTROLS:
                        ordinary, temporal, cosine = 0.48, 0.50, 0.70
                key = (nfe, control)
                cells[key] = analyzer.CellData(
                    arm=arm,
                    nfe=nfe,
                    control=control,
                    clip_ids=clip_ids,
                    metrics=_metric_vectors(clips, ordinary, temporal, cosine),
                    pairing_identity_sha256=f"pair-{nfe}-{control}",
                    generation_identity_sha256=f"generation-{arm}-{nfe}-{control}",
                )
        evaluations[arm] = analyzer.EvaluationData(
            arm=arm,
            target_mode=evaluator.ARM_CONTRACTS[arm]["target_mode"],
            normalization_binding_sha256=(arm.lower().encode().hex() + "0" * 64)[:64],
            summary_record={"path": f"/{arm}/summary.json", "sha256": "2" * 64},
            checkpoint_record={"path": f"/{arm}/checkpoint.pt", "sha256": "3" * 64},
            training_config_record={"path": f"/{arm}/config.json", "sha256": "4" * 64},
            training_doe_common_identity_sha256="7" * 64,
            calibration_receipt_identity_sha256=(arm.encode().hex() + "8" * 64)[:64],
            implementation_registration_identity_sha256="9" * 64,
            preregistration_identity_sha256="a" * 64,
            manifest_record=manifest,
            semantic_cache_identity_sha256="5" * 64,
            cache_producer_attestation_identity_sha256="6" * 64,
            cells=cells,
        )
    return evaluations


def test_augmented_manifest_file_record_is_verified_without_weakening_bare_records(
    tmp_path,
):
    manifest = tmp_path / "val.jsonl"
    manifest.write_text('{"clip_id":"clip-0"}\n', encoding="utf-8")
    bare = vlf.file_record(manifest)
    augmented = {**bare, "split": "val", "clips": 1}
    assert analyzer._verified_file_record(  # noqa: SLF001
        augmented,
        label="manifest",
        allow_augmented=True,
    ) == manifest.resolve()
    with pytest.raises(analyzer.TemporalAnalysisError):
        analyzer._verified_file_record(augmented, label="ordinary file")  # noqa: SLF001
    changed = {**augmented, "sha256": "0" * 64}
    with pytest.raises(analyzer.TemporalAnalysisError):
        analyzer._verified_file_record(  # noqa: SLF001
            changed,
            label="manifest",
            allow_augmented=True,
        )


def test_paired_relative_effect_is_ratio_of_means_not_mean_of_ratios():
    candidate = np.asarray([1.0, 9.0])
    control = np.asarray([2.0, 10.0])
    indices = analyzer.common_bootstrap_indices(2, samples=500, seed=19)
    effect = analyzer.paired_effect(
        candidate,
        control,
        indices,
        kind="relative_lower",
        confidence=0.95,
    )
    assert effect["point_estimate"] == pytest.approx(1.0 / 6.0)
    assert effect["point_estimate"] != pytest.approx(
        np.mean((control - candidate) / control)
    )


def test_one_common_bootstrap_matrix_is_reproducible_and_clip_addressed():
    first = analyzer.common_bootstrap_indices(11, samples=333, seed=20260807)
    second = analyzer.common_bootstrap_indices(11, samples=333, seed=20260807)
    third = analyzer.common_bootstrap_indices(11, samples=333, seed=20260808)
    assert first.dtype == np.int32
    assert first.shape == (333, 11)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_composite_gate_selects_at_most_one_pair_with_frozen_tie_break(monkeypatch):
    monkeypatch.setattr(
        analyzer.screen,
        "_source_record",
        lambda: {"commit": "a" * 40, "branch": "test", "dirty": False},
    )
    result = analyzer.build_analysis(
        _evaluations(),
        bootstrap_samples=500,
        bootstrap_seed=20260807,
        confidence=analyzer.CELLWISE_CONFIDENCE,
        expected_clips=8,
    )
    assert result["selection_count"] == 1
    assert result["selected_cell"]["arm"] == "DELTA-T"
    assert result["selected_cell"]["nfe"] == 2
    selected_cell = next(
        cell
        for cell in result["candidate_cells"]
        if cell["arm"] == "DELTA-T" and cell["nfe"] == 2
    )
    assert selected_cell["composite_gate_passed"] is True
    assert all(
        item["promotion_criterion"] is False
        for item in selected_cell["factual_action_attribution"].values()
    )
    assert result["protocol_frozen"] is False
    assert result["protected_test_accessed"] is False


def test_five_percent_gate_requires_lower_bound_not_only_point():
    indices = analyzer.common_bootstrap_indices(8, samples=500, seed=7)
    # The mean point improvement exceeds 5%, but heterogeneous paired clips
    # make the one-sided bound fall below it.
    control = np.ones(8)
    candidate = np.asarray([0.70, 0.70, 0.70, 0.70, 1.10, 1.10, 1.10, 1.10])
    effect = analyzer.paired_effect(
        candidate,
        control,
        indices,
        kind="relative_lower",
        confidence=analyzer.CELLWISE_CONFIDENCE,
    )
    assert effect["point_estimate"] == pytest.approx(0.10)
    assert effect["one_sided_lower_bound"] < 0.05
    assert analyzer._effect_pass(effect, 0.05) is False  # noqa: SLF001


def test_selection_tie_break_prioritizes_nfe_then_temporal_then_nmse_then_arm():
    cells = []
    for arm in analyzer.CANDIDATE_ARMS:
        cells.append(
            {
                "arm": arm,
                "nfe": 2,
                "composite_gate_passed": True,
                "autonomous_means": {
                    "temporal_difference_nmse": 0.4,
                    "semantic_nmse": 0.4,
                    "semantic_token_cosine": 0.8,
                },
            }
        )
    cells.append(
        {
            "arm": "DELTA-R",
            "nfe": 1,
            "composite_gate_passed": True,
            "autonomous_means": {
                "temporal_difference_nmse": 0.49,
                "semantic_nmse": 0.49,
                "semantic_token_cosine": 0.71,
            },
        }
    )
    assert analyzer.select_cell(cells) == {
        "arm": "DELTA-R",
        "nfe": 1,
        "autonomous_semantic_nmse": 0.49,
        "autonomous_semantic_token_cosine": 0.71,
        "autonomous_temporal_difference_nmse": 0.49,
    }
