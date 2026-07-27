import hashlib
import json
from pathlib import Path

import pytest

from tools.analyze_causal_training_logs import (
    SIGMA_CONVENTION,
    ArmInput,
    TrainingLogAnalysisError,
    analyze,
)

EXPOSURE_SUFFIXES = (
    "condition/raw_state_rms",
    "condition/state_residual_rms",
    "condition/clock_residual_rms",
    "condition/combined_rms",
    "condition/native_patch_embedding_rms",
    "condition/state_to_native_ratio",
    "condition/combined_to_native_ratio",
)


def _losses(
    *,
    video_flow_loss: float,
    video_nmse: float,
    tf_clock: float = 0.4,
    video_clock: float = 0.6,
    exposure_offset: float = 0.0,
) -> dict[str, float]:
    prefix = "MultiDatasetABC_0/"
    losses = {
        prefix + "clock/tf_sigma_mean": tf_clock,
        prefix + "clock/video_sigma_mean": video_clock,
        prefix + "video_flow_loss": video_flow_loss,
        prefix + "teacher_forced/video_x0_nmse": video_nmse,
        prefix + "teacher_forced/video_x0_nmse/sigma_0.25": (0.25 * video_nmse),
        prefix + "teacher_forced/video_x0_nmse/sigma_0.90": (2.0 * video_nmse),
    }
    exposure_values = {
        "condition/raw_state_rms": 2.0,
        "condition/state_residual_rms": 0.1,
        "condition/clock_residual_rms": 0.02,
        "condition/combined_rms": 0.12,
        "condition/native_patch_embedding_rms": 1.5,
        "condition/state_to_native_ratio": 0.1 / 1.5,
        "condition/combined_to_native_ratio": 0.12 / 1.5,
    }
    losses.update(
        {
            prefix + suffix: value + exposure_offset
            for suffix, value in exposure_values.items()
        }
    )
    return losses


def _validation_line(iteration: int, losses: dict[str, float]) -> str:
    return (
        f" 12345     INFO robot_wm.u Validation completed for iteration "
        f"{iteration}, losses: {losses!r}\n"
    )


def _write_log(
    path: Path,
    *,
    iterations: tuple[int, ...],
    flow_values: tuple[float, ...],
    nmse_values: tuple[float, ...],
    duplicate: bool = True,
    clock_overrides: dict[int, tuple[float, float]] | None = None,
    extra_lines: tuple[str, ...] = (),
) -> None:
    lines = list(extra_lines)
    for index, iteration in enumerate(iterations):
        tf_clock, video_clock = (0.4, 0.6)
        if clock_overrides and iteration in clock_overrides:
            tf_clock, video_clock = clock_overrides[iteration]
        losses = _losses(
            video_flow_loss=flow_values[index],
            video_nmse=nmse_values[index],
            tf_clock=tf_clock,
            video_clock=video_clock,
            exposure_offset=0.001 * index,
        )
        lines.append(_validation_line(iteration, losses))
        if duplicate:
            lines.append(_validation_line(iteration, losses))
    path.write_text("".join(lines), encoding="utf-8")


def _fixture_arms(
    tmp_path: Path,
    *,
    shuffled_iterations: tuple[int, ...] = (0, 10, 20),
    shuffled_clocks: dict[int, tuple[float, float]] | None = None,
) -> tuple[list[ArmInput], Path, Path]:
    matched_root = tmp_path / "matched"
    shuffled_root = tmp_path / "shuffled"
    matched_root.mkdir()
    shuffled_root.mkdir()
    matched_log = matched_root / "train.log"
    shuffled_log = shuffled_root / "train.log"
    _write_log(
        matched_log,
        iterations=(0, 10, 20),
        flow_values=(1.0, 0.3, 0.2),
        nmse_values=(2.0, 0.8, 0.4),
        extra_lines=(
            "ABCDataset sample 7 failed (decode error); retrying another\n",
            (
                "Rejected sample with no model-supervised future pixels; "
                "retrying global index 19. diagnostic={}\n"
            ),
            "future-validity retries exhausted before batching\n",
            "non-finite guard enabled\n",
            "NaN/Inf check enabled\n",
            "error_if_nonfinite: true\n",
            " 999 ERROR trainer non-finite loss detected\n",
            "training scalar loss=nan\n",
        ),
    )
    shuffled_flow = {0: 1.0, 10: 0.7, 20: 0.4}
    shuffled_nmse = {0: 2.0, 10: 1.2, 20: 0.8}
    _write_log(
        shuffled_log,
        iterations=shuffled_iterations,
        flow_values=tuple(shuffled_flow[item] for item in shuffled_iterations),
        nmse_values=tuple(shuffled_nmse[item] for item in shuffled_iterations),
        clock_overrides=shuffled_clocks,
    )
    arms = [
        ArmInput(
            name="matched_s003",
            condition_mode="matched",
            state_scale="0.030",
            root=matched_root,
            logs=(matched_log,),
        ),
        ArmInput(
            name="shuffled_s003",
            condition_mode="shuffled",
            state_scale=0.03,
            root=shuffled_root,
            logs=(shuffled_log,),
        ),
    ]
    return arms, matched_log, shuffled_log


def _canonical_payload_sha256(payload: dict) -> str:
    value = dict(payload)
    expected = value.pop("payload_identity_sha256")
    actual = hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    return actual


def test_analyzes_deduplicated_curves_exposure_and_markers(tmp_path: Path):
    arms, matched_log, _ = _fixture_arms(tmp_path)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    output = analysis_dir / "training_telemetry.json"

    payload = analyze(arms, output=output)

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert payload["schema_version"] == 1
    assert payload["sigma_convention"] == SIGMA_CONVENTION
    _canonical_payload_sha256(payload)

    matched = payload["provenance"]["arms"]["matched_s003"]
    assert matched["state_scale"] == "0.03"
    assert matched["validation"]["raw_occurrence_count"] == 6
    assert matched["validation"]["deduplicated_iteration_count"] == 3
    assert matched["validation"]["identical_duplicate_occurrence_count"] == 3
    assert matched["validation"]["iterations"] == [0, 10, 20]
    assert (
        matched["logs"][0]["sha256_of_snapshotted_bytes"]
        == hashlib.sha256(matched_log.read_bytes()).hexdigest()
    )
    assert not matched["logs"][0]["source_changed_during_snapshot"]

    warnings = matched["operational_markers"]["warnings"]["total_by_pattern"]
    assert warnings["abc_sample_decode_or_transform_retry"] == 1
    assert warnings["future_validity_retry"] == 1
    assert warnings["future_validity_exhausted"] == 1
    nonfinite = matched["operational_markers"]["nonfinite"]
    assert nonfinite["unique_matching_line_count"] == 2
    assert nonfinite["ignored_configuration_matches"] == {
        "error_if_nonfinite_setting": 1,
        "guard_or_check_enabled": 1,
        "nan_inf_check_enabled": 1,
    }
    assert matched["operational_markers"]["fatal"]["total_matches"] == 1
    assert matched["operational_markers"]["fatal_or_nonfinite_present"]

    for suffix in EXPOSURE_SUFFIXES:
        summary = matched["curves"]["exposure"][suffix]["summary"]
        assert summary["count"] == 3
        assert summary["trapezoid_auc"] is not None

    comparison = payload["same_scale_matched_vs_shuffled"]["0.03"]
    assert comparison["validation_iteration_grid"] == [0, 10, 20]
    assert comparison["validation_grid_exact_match"]
    assert all(
        item["exact_match"] for item in comparison["clock_mean_exact_match"].values()
    )
    assert set(comparison["metric_comparisons"]) == {
        "video_flow_loss",
        "teacher_forced/video_x0_nmse",
        "teacher_forced/video_x0_nmse/sigma_0.25",
        "teacher_forced/video_x0_nmse/sigma_0.90",
    }
    flow = comparison["metric_comparisons"]["video_flow_loss"]
    assert flow["matched"]["final_value"] == pytest.approx(0.2)
    assert flow["shuffled"]["final_value"] == pytest.approx(0.4)
    assert flow["matched"]["trapezoid_auc"] == pytest.approx(9.0)
    assert flow["shuffled"]["trapezoid_auc"] == pytest.approx(14.0)
    assert flow["final_matched_minus_shuffled"] == pytest.approx(-0.2)
    threshold = flow["time_to_reference_final"]
    assert threshold["threshold"] == pytest.approx(0.4)
    assert threshold["matched_first_iteration"] == 10
    assert threshold["shuffled_first_iteration"] == 20
    assert threshold["matched_lead_updates"] == 10


def test_duplicate_rank_disagreement_fails_before_output(tmp_path: Path):
    arms, matched_log, _ = _fixture_arms(tmp_path)
    conflicting = _losses(video_flow_loss=9.0, video_nmse=2.0)
    with matched_log.open("a", encoding="utf-8") as handle:
        handle.write(_validation_line(0, conflicting))
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    output = analysis_dir / "report.json"

    with pytest.raises(
        TrainingLogAnalysisError,
        match="duplicate rank validation records disagree",
    ):
        analyze(arms, output=output)
    assert not output.exists()


def test_same_scale_iteration_grid_must_match(tmp_path: Path):
    arms, _, _ = _fixture_arms(
        tmp_path,
        shuffled_iterations=(0, 20),
    )
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    output = analysis_dir / "report.json"

    with pytest.raises(
        TrainingLogAnalysisError,
        match="different validation iteration grids",
    ):
        analyze(arms, output=output)
    assert not output.exists()


def test_same_scale_clock_means_must_match(tmp_path: Path):
    arms, _, _ = _fixture_arms(
        tmp_path,
        shuffled_clocks={10: (0.401, 0.6)},
    )
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    output = analysis_dir / "report.json"

    with pytest.raises(
        TrainingLogAnalysisError,
        match="different clock/tf_sigma_mean values",
    ):
        analyze(arms, output=output)
    assert not output.exists()


def test_output_must_be_fresh_and_outside_arm_roots(tmp_path: Path):
    arms, _, _ = _fixture_arms(tmp_path)
    inside_output = Path(arms[0].root) / "report.json"
    with pytest.raises(
        TrainingLogAnalysisError,
        match="outside every read-only arm root",
    ):
        analyze(arms, output=inside_output)
    assert not inside_output.exists()

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    output = analysis_dir / "report.json"
    analyze(arms, output=output)
    first_bytes = output.read_bytes()
    with pytest.raises(
        TrainingLogAnalysisError,
        match="output path already exists",
    ):
        analyze(arms, output=output)
    assert output.read_bytes() == first_bytes


def test_nonfinite_validation_scalar_is_rejected(tmp_path: Path):
    arms, matched_log, _ = _fixture_arms(tmp_path)
    with matched_log.open("a", encoding="utf-8") as handle:
        handle.write(
            "Validation completed for iteration 30, losses: "
            "{'MultiDatasetABC_0/video_flow_loss': 1e309}\n"
        )
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    output = analysis_dir / "report.json"

    with pytest.raises(
        TrainingLogAnalysisError,
        match="is non-finite",
    ):
        analyze(arms, output=output)
    assert not output.exists()
