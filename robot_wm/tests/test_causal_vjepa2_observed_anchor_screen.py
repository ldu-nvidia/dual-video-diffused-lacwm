from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools import causal_vjepa2_observed_anchor as observed
from tools import causal_vjepa2_observed_anchor_screen as ainc


def _normalization() -> observed.ObservedIncrementNormalization:
    return observed.ObservedIncrementNormalization(
        mean=torch.linspace(-0.05, 0.05, 48),
        std=torch.linspace(0.5, 1.5, 48),
        provenance={"split": "train"},
    )


class _ZeroCleanModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.forward_kwargs: dict[str, object] = {}

    def forward(
        self,
        noisy_video: torch.Tensor,
        noisy_auxiliary: torch.Tensor,
        t_video: torch.Tensor,
        t_auxiliary: torch.Tensor,
        history: torch.Tensor,
        actions: torch.Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        del t_video, t_auxiliary, history, actions
        self.forward_kwargs = dict(kwargs)
        return SimpleNamespace(
            video_x=torch.zeros_like(noisy_video),
            auxiliary_x=torch.zeros_like(noisy_auxiliary) + self.scale,
        )


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "history": torch.zeros(batch_size, 3, 5, 64, 112),
        "future": torch.zeros(batch_size, 3, 8, 64, 112),
        "actions": torch.zeros(batch_size, 16, 7),
        "auxiliary_target": torch.randn(batch_size, 48, 8, 8, 14),
        "observed_anchor": torch.randn(batch_size, 48, 8, 14),
    }


def test_action_offsets_are_fixed_and_episode_disjoint() -> None:
    rows = [
        {"clip_id": f"clip-{index}", "episode_index": index} for index in range(890)
    ]
    for offset in ainc.ACTION_OFFSETS:
        permutation = ainc.action_permutation_indices(rows, offset)
        assert len(permutation) == 890
        assert permutation[0] == offset
        assert permutation[-1] == offset - 1
        assert all(
            rows[index]["episode_index"] != rows[source]["episode_index"]
            for index, source in enumerate(permutation)
        )


def test_action_permutation_rejects_episode_overlap() -> None:
    rows = [
        {"clip_id": f"clip-{index}", "episode_index": index} for index in range(890)
    ]
    rows[1]["episode_index"] = rows[0]["episode_index"]
    with pytest.raises(
        ainc.ObservedAnchorScreenError, match="not clip/episode disjoint"
    ):
        ainc.action_permutation_indices(rows, 1)


def test_execution_condition_marks_cheap_probe_nonpromotable() -> None:
    condition = ainc._execution_condition(  # noqa: SLF001
        SimpleNamespace(
            execution_mode="cheap-proxy-validity",
            temporal_selection_record=None,
        )
    )
    assert condition["proxy_validity_only"] is True
    assert condition["semantic_screen_promotion_eligible"] is False
    assert condition["video_quality_claim_eligible"] is False
    assert condition["protected_test_eligible"] is False


def test_execution_condition_accepts_only_frozen_temporal_no_pass(
    tmp_path: Path,
) -> None:
    source = {
        "commit": ainc.TEMPORAL_AUTHORIZATION_COMMIT,
        "branch": "temporal-doe",
        "dirty": False,
    }
    input_evaluations = {}
    for arm in ainc.TEMPORAL_ARMS:
        summary = tmp_path / f"{arm}-summary.json"
        summary.write_text(json.dumps({"arm": arm}), encoding="utf-8")
        input_evaluations[arm] = {
            "summary": ainc.vlf.file_record(summary),
            "target_mode": "absolute",
        }
    candidate_cells = [
        {"arm": arm, "nfe": nfe, "composite_gate_passed": False}
        for arm in ainc.TEMPORAL_CANDIDATE_ARMS
        for nfe in ainc.TEMPORAL_SELECTION_NFE
    ]
    analysis_unsigned = {
        "schema": ainc.TEMPORAL_ANALYSIS_SCHEMA,
        "status": "no_candidate_passed",
        "source": source,
        "selected_cell": None,
        "selection_count": 0,
        "split": "val",
        "paired_clips": ainc.screen.FROZEN_VALIDATION_CLIPS,
        "input_evaluations": input_evaluations,
        "candidate_cells": candidate_cells,
        "bootstrap": {
            "samples": 10_000,
            "seed": 20260807,
            "bonferroni_candidate_cells": len(candidate_cells),
            "common_indices_across_all_metrics_cells_controls": True,
        },
        "development_selection_split": True,
        "protected_test_accessed": False,
        "protected_test_cache_opened": False,
        "protocol_frozen": True,
    }
    analysis = {
        **analysis_unsigned,
        "identity_sha256": ainc.screen.sha256_json(analysis_unsigned),
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    analysis_record = ainc.vlf.file_record(analysis_path)
    selection_unsigned = {
        "schema": ainc.TEMPORAL_SELECTION_SCHEMA,
        "status": "frozen_no_selection",
        "development_analysis": analysis_record,
        "development_analysis_identity_sha256": analysis["identity_sha256"],
        "selected_cell": None,
        "selection_count": 0,
        "selection_split": "val",
        "selection_used_protected_test": False,
        "protected_test_accessed": False,
        "lockbox_may_open": False,
        "input_evaluations": input_evaluations,
    }
    selection = {
        **selection_unsigned,
        "identity_sha256": ainc.screen.sha256_json(selection_unsigned),
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    condition = ainc._execution_condition(  # noqa: SLF001
        SimpleNamespace(
            execution_mode="post-temporal-no-pass",
            temporal_selection_record=str(selection_path),
        )
    )

    assert condition["temporal_primary_no_pass_attested"] is True
    assert condition["semantic_screen_promotion_eligible"] is True
    assert condition["video_quality_claim_eligible"] is False
    assert condition["protected_test_eligible"] is False
    assert condition["comparison_reference"] == "continuation_local_C-ABS_required"
    assert condition["external_temporal_abs_numeric_baseline_allowed"] is False
    assert condition["temporal_no_pass_is_authorization_only"] is True

    wrong_unsigned = {
        **analysis_unsigned,
        "source": {**source, "commit": "0" * 40},
    }
    wrong_analysis = {
        **wrong_unsigned,
        "identity_sha256": ainc.screen.sha256_json(wrong_unsigned),
    }
    analysis_path.write_text(json.dumps(wrong_analysis), encoding="utf-8")
    wrong_analysis_record = ainc.vlf.file_record(analysis_path)
    wrong_selection_unsigned = {
        **selection_unsigned,
        "development_analysis": wrong_analysis_record,
        "development_analysis_identity_sha256": wrong_analysis["identity_sha256"],
    }
    wrong_selection = {
        **wrong_selection_unsigned,
        "identity_sha256": ainc.screen.sha256_json(wrong_selection_unsigned),
    }
    selection_path.write_text(json.dumps(wrong_selection), encoding="utf-8")
    with pytest.raises(
        ainc.ObservedAnchorScreenError,
        match="does not prove a frozen validation-only no-pass result",
    ):
        ainc._execution_condition(  # noqa: SLF001
            SimpleNamespace(
                execution_mode="post-temporal-no-pass",
                temporal_selection_record=str(selection_path),
            )
        )


def test_training_step_diffuses_q_without_passing_anchor_to_model() -> None:
    torch.manual_seed(12)
    model = _ZeroCleanModel()
    weighted, telemetry = ainc.ainc_training_step(model, _batch(), _normalization())

    assert weighted.ndim == 0 and torch.isfinite(weighted)
    assert telemetry["auxiliary_loss"].ndim == 0
    assert model.forward_kwargs["auxiliary_fusion_mask"] is True
    assert model.forward_kwargs["predict_video"] is False
    assert "observed_anchor" not in model.forward_kwargs


def test_deployable_sampler_has_no_clean_future_api_and_decodes_anchor_skip() -> None:
    parameters = inspect.signature(ainc.sample_anchored_increments).parameters
    assert not {"future", "semantic", "target", "clean", "oracle"}.intersection(
        parameters
    )

    model = _ZeroCleanModel()
    normalization = _normalization()
    history = torch.zeros(1, 3, 5, 64, 112)
    actions = torch.zeros(1, 16, 7)
    anchor = torch.randn(1, 48, 8, 14)
    video_noise = torch.randn(1, 3, 8, 64, 112)
    increment_noise = torch.randn(1, 48, 8, 8, 14)
    sample = ainc.sample_anchored_increments(
        model,
        history,
        actions,
        anchor,
        video_noise=video_noise,
        increment_noise=increment_noise,
        steps=1,
        normalization=normalization,
    )
    _, expected = observed.mean_increment_control(anchor, normalization)

    assert sample.model_calls == 1
    assert torch.equal(sample.normalized_prediction, torch.zeros_like(increment_noise))
    assert torch.allclose(sample.semantic_prediction, expected, atol=1e-6, rtol=0)


def test_increment_metrics_are_per_example_and_identity_is_perfect() -> None:
    target = torch.randn(3, 48, 8, 8, 14)
    perfect = ainc.increment_metrics(target, target)
    zero = ainc.increment_metrics(torch.zeros_like(target), target)

    assert perfect["increment_nmse"].shape == (3,)
    assert torch.equal(perfect["increment_nmse"], torch.zeros(3))
    assert torch.allclose(perfect["increment_token_cosine"], torch.ones(3))
    assert torch.allclose(zero["increment_nmse"], torch.ones(3))


def test_minimal_screen_control_and_nfe_contract_is_frozen() -> None:
    assert ainc.NFE_GRID == (1, 2, 4)
    assert ainc.ACTION_OFFSETS == (1, 17, 101)
    assert ainc.EVALUATION_SEED == ainc.screen.FROZEN_EVALUATION_SEED
    assert ainc.CONTROLS == (
        "autonomous",
        "anchor_static",
        "mean_increment",
        "donor_target",
        "context_shuffled",
        "history_shuffled",
        "actions_offset_1",
        "actions_offset_17",
        "actions_offset_101",
        "anchor_decode_shuffled",
        "zero",
        "oracle_clean",
    )
    assert ainc.DOE_ARM == "AINC-OFF"
