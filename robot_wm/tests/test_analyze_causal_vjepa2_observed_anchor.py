from __future__ import annotations

from types import MappingProxyType

import numpy as np

from tools import analyze_causal_vjepa2_observed_anchor as analysis
from tools import analyze_causal_vjepa2_temporal_targets as temporal_analysis
from tools import causal_vjepa2_observed_anchor_screen as ainc


def _metrics(
    clips: int,
    *,
    semantic_nmse: float,
    temporal_nmse: float,
    semantic_cosine: float = 0.8,
) -> dict[str, np.ndarray]:
    return {
        "semantic_nmse": np.full(clips, semantic_nmse, dtype=np.float64),
        "semantic_token_cosine": np.full(clips, semantic_cosine, dtype=np.float64),
        "temporal_difference_nmse": np.full(clips, temporal_nmse, dtype=np.float64),
        "temporal_difference_token_cosine": np.full(clips, 0.7, dtype=np.float64),
        "increment_nmse": np.full(clips, 0.4, dtype=np.float64),
        "increment_token_cosine": np.full(clips, 0.6, dtype=np.float64),
    }


def _ainc_evaluation(*, proxy: bool) -> analysis.EvaluationData:
    clips = 64
    clip_ids = tuple(f"clip-{index:03d}" for index in range(clips))
    cells = {}
    for nfe in analysis.SELECTION_NFE:
        for control in ainc.CONTROLS:
            semantic_nmse = 0.30 if nfe == 1 else 0.60
            temporal_nmse = 0.30 if nfe == 1 else 0.60
            semantic_cosine = 0.80 if nfe == 1 else 0.60
            if control in analysis.TEMPORAL_CONTROLS:
                temporal_nmse = 0.40 if nfe == 1 else 0.70
            if control == "anchor_decode_shuffled":
                semantic_nmse = 0.40 if nfe == 1 else 0.70
                semantic_cosine = 0.60
                # The corrected gate deliberately does not ask this invariant
                # additive-anchor control for a temporal improvement.
                temporal_nmse = 0.30 if nfe == 1 else 0.60
            cells[(nfe, control)] = analysis.CellData(
                nfe=nfe,
                control=control,
                clip_ids=clip_ids,
                metrics=MappingProxyType(
                    _metrics(
                        clips,
                        semantic_nmse=semantic_nmse,
                        temporal_nmse=temporal_nmse,
                        semantic_cosine=semantic_cosine,
                    )
                ),
                pairing_identity_sha256="a" * 64,
                generation_identity_sha256="b" * 64,
                increment_identity_sha256="c" * 64,
            )
    condition = {
        "proxy_validity_only": proxy,
        "semantic_screen_promotion_eligible": not proxy,
    }
    return analysis.EvaluationData(
        summary_record={"path": "/ainc", "sha256": "d" * 64, "bytes": 1},
        checkpoint_record={},
        training_config_record={},
        training_config={},
        execution_condition=condition,
        manifest_record={},
        cells=MappingProxyType(cells),
    )


def _abs_evaluation() -> temporal_analysis.EvaluationData:
    clips = 64
    clip_ids = tuple(f"clip-{index:03d}" for index in range(clips))
    cells = {}
    for nfe in analysis.SELECTION_NFE:
        metrics = _metrics(clips, semantic_nmse=0.40, temporal_nmse=0.40)
        # Temporal analysis cells contain its packed diagnostics too, but this
        # gate consumes only the four decoded semantic metrics above.
        cells[(nfe, "autonomous")] = temporal_analysis.CellData(
            arm="ABS",
            nfe=nfe,
            control="autonomous",
            clip_ids=clip_ids,
            metrics=MappingProxyType(metrics),
            pairing_identity_sha256="a" * 64,
            generation_identity_sha256="e" * 64,
        )
    return temporal_analysis.EvaluationData(
        arm="ABS",
        target_mode="absolute",
        normalization_binding_sha256="f" * 64,
        summary_record={"path": "/abs", "sha256": "1" * 64, "bytes": 1},
        checkpoint_record={},
        training_config_record={},
        training_doe_common_identity_sha256="2" * 64,
        calibration_receipt_identity_sha256="3" * 64,
        implementation_registration_identity_sha256="4" * 64,
        preregistration_identity_sha256="5" * 64,
        manifest_record={},
        semantic_cache_identity_sha256="6" * 64,
        cache_producer_attestation_identity_sha256="7" * 64,
        cells=MappingProxyType(cells),
    )


def test_corrected_common_bootstrap_gate_selects_only_nfe_one(monkeypatch) -> None:
    monkeypatch.setattr(ainc, "_source_record", lambda: {"commit": "a" * 40, "dirty": False})
    result = analysis.build_analysis(
        _ainc_evaluation(proxy=False),
        _abs_evaluation(),
        bootstrap_samples=200,
    )

    assert result["status"] == "one_candidate_selected"
    assert result["selection_count"] == 1
    assert result["selected_cell"]["nfe"] == 1
    assert result["bootstrap"]["bonferroni_candidate_cells"] == 3
    assert result["comparison_design"]["control_display_name"] == "C-ABS"
    assert result["comparison_design"]["same_clean_commit_required"] is True
    assert (
        result["comparison_design"]["external_temporal_abs_numeric_baseline_allowed"]
        is False
    )
    nfe_one = result["candidate_cells"][0]
    assert nfe_one["passed"] is True
    assert nfe_one["effects"]["temporal_nmse_vs_anchor_static"][
        "one_sided_lower_bound"
    ] >= 0.05
    assert "temporal_nmse_vs_anchor_decode_shuffled" not in nfe_one["effects"]
    assert nfe_one["effects"]["semantic_cosine_vs_anchor_decode_shuffled"][
        "one_sided_lower_bound"
    ] > 0.0


def test_proxy_mode_remains_nonpromotable_even_when_numeric_gate_passes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ainc, "_source_record", lambda: {"commit": "a" * 40, "dirty": False})
    result = analysis.build_analysis(
        _ainc_evaluation(proxy=True),
        _abs_evaluation(),
        bootstrap_samples=200,
    )
    selection = analysis._selection_record(  # noqa: SLF001
        result, {"path": "/analysis", "sha256": "8" * 64, "bytes": 1}
    )

    assert result["status"] == "proxy_nonpromotable"
    assert result["passing_cells_before_contingency"] == 1
    assert result["selected_cell"] is None
    assert selection["status"] == "frozen_proxy_no_selection"
    assert selection["selection_count"] == 0
    assert selection["lockbox_may_open"] is False


def test_familywise_confidence_is_corrected_over_three_nfe_cells() -> None:
    assert analysis.CELLWISE_CONFIDENCE == 1.0 - 0.05 / 3
    assert analysis.SELECTION_NFE == (1, 2, 4)
