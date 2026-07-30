import math

import pytest
import torch

from tools import analyze_vjepa2_controlled_study as analysis


def test_metric_definitions_use_independent_history_lengths():
    video_target = torch.ones(1, 2, 3, 1, 1)
    video_prediction = video_target.clone()
    video_prediction[:, :, 0] = 100
    assert (
        analysis._nmse(
            video_prediction,
            video_target,
            history=1,
            key="video",
        )
        == 0.0
    )

    auxiliary_target = torch.ones(1, 2, 2, 1, 1)
    auxiliary_prediction = -auxiliary_target
    assert analysis._nmse(
        auxiliary_prediction,
        auxiliary_target,
        history=0,
        key="auxiliary",
    ) == pytest.approx(4.0)
    assert analysis._cosine(
        auxiliary_prediction,
        auxiliary_target,
        history=0,
        key="auxiliary",
    ) == pytest.approx(-1.0)


def test_decoded_metrics_are_in_unit_range_and_temporal():
    target = torch.zeros(1, 3, 2, 1, 1, dtype=torch.uint8)
    prediction = target.clone()
    prediction[:, :, 1] = 255
    metrics = analysis._decoded_metrics(prediction, target)
    assert metrics["decoded_mse_unit_range"] == pytest.approx(0.5)
    assert metrics[
        "decoded_temporal_difference_mse_unit_range"
    ] == pytest.approx(1.0)
    assert metrics["decoded_psnr_db"] == pytest.approx(
        10.0 * math.log10(2.0)
    )


@pytest.mark.parametrize(
    ("direction", "left", "reference", "expected"),
    [
        ("lower", [0.8, 1.6], [1.0, 2.0], 0.2),
        ("higher", [1.2, 2.4], [1.0, 2.0], 0.2),
    ],
)
def test_paired_effect_positive_always_means_left_is_better(
    direction, left, reference, expected
):
    result = analysis._paired_effect(
        left,
        reference,
        direction=direction,
        bootstrap_samples=100,
        confidence=0.95,
        seed=7,
        label=direction,
    )
    assert result["relative_improvement"] == pytest.approx(expected)
    assert result["favorable_unit_fraction"] == 1.0
    assert result["bootstrap_ci"]["low"] > 0


def test_time_to_threshold_distinguishes_first_from_sustained():
    curve = {
        1: 1.2,
        50: 0.9,
        100: 1.1,
        200: 0.8,
        400: 0.7,
        800: 0.6,
        1000: 0.5,
    }
    wall = {update: update * 2 for update in analysis.COMPLETED_UPDATES}
    result = analysis._time_to_threshold(
        curve,
        threshold=1.0,
        direction="lower",
        cumulative_wall=wall,
    )
    assert result["first_completed_update"] == 50
    assert result["first_sustained_completed_update"] == 200
    assert result[
        "cumulative_stage_wall_seconds_at_first_sustained"
    ] == 400


def test_vjepa_inventory_requires_every_source_nfe_and_counter():
    keys = analysis._required_keys()
    for source in analysis.SOURCES:
        infix = analysis._source_infix(source)
        for nfe in analysis.NFE_STEPS:
            assert f"video_final{infix}_nfe_{nfe}" in keys
            assert f"tf_final{infix}_nfe_{nfe}" in keys
            assert f"decoded_future{infix}_nfe_{nfe}" in keys
            assert f"wan_call_count{infix}_nfe_{nfe}" in keys
    assert "online_teacher_call_count" in keys
    assert "auxiliary_history_latent_frames" in keys


def test_latency_sources_exclude_shuffled_and_oracles():
    assert analysis.LATENCY_SOURCES == ("autonomous", "off")
    assert analysis.LATENCY_RE.fullmatch(
        "source_autonomous_nfe_4.json"
    )
    assert analysis.LATENCY_RE.fullmatch("source_off_nfe_8.json")
    assert not analysis.LATENCY_RE.fullmatch(
        "source_autonomous_shuffled_nfe_4.json"
    )
    assert not analysis.LATENCY_RE.fullmatch(
        "source_oracle_matched_nfe_4.json"
    )


def test_scientific_quality_grid_is_128_clip_protocol():
    assert analysis.EXPECTED_TEST_CLIPS == 128
    assert len(analysis._quality_grid(800)) == 8
    assert len(analysis._quality_grid(1000)) == 35
    assert ("oracle_matched", 4) in analysis._quality_grid(800)
    assert ("oracle_matched", 8) not in analysis._quality_grid(800)


def test_oracle_gap_closure_uses_paired_clip_bootstrap():
    result = analysis._paired_gap_closure(
        off=[1.0] * 128,
        autonomous=[0.8] * 128,
        oracle=[0.5] * 128,
        bootstrap_samples=100,
        confidence=0.95,
        seed=11,
        label="gap",
    )
    assert result["n_paired_units"] == 128
    assert result["gap_closure_fraction"] == pytest.approx(0.4)
    assert result["bootstrap_ci"]["low"] > 0


def test_paired_latency_bootstrap_preserves_execution_order_strata():
    orders = [
        ["J1", "VPM"] if index % 2 == 0 else ["VPM", "J1"]
        for index in range(analysis.PAIRED_TIMED_PAIRS)
    ]
    j1 = [8.0 if index % 2 == 0 else 9.0 for index in range(100)]
    vpm = [12.0 if index % 2 == 0 else 13.0 for index in range(100)]
    result = analysis._counterbalanced_paired_latency_effect(
        j1,
        vpm,
        orders,
        bootstrap_samples=200,
        confidence=0.95,
        seed=19,
        label="paired-latency-test",
    )
    assert result["execution_order_strata"] == {
        "J1_first": 50,
        "VPM_first": 50,
    }
    assert result["mean_favorable_difference_ms"] == pytest.approx(4.0)
    assert result["relative_improvement"] == pytest.approx(4.0 / 12.5)
    assert result["bootstrap_ci"]["low"] > 0


def test_paired_latency_bootstrap_rejects_unbalanced_order():
    with pytest.raises(analysis.StudyValidationError):
        analysis._counterbalanced_paired_latency_effect(
            [8.0] * 100,
            [12.0] * 100,
            [["J1", "VPM"]] * 100,
            bootstrap_samples=100,
            confidence=0.95,
            seed=3,
            label="unbalanced",
        )
