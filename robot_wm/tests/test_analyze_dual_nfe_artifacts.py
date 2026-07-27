import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools.analyze_dual_nfe_artifacts import (
    ArtifactValidationError,
    EVALUATION_CONDITION_SOURCE_NAMES,
    NFE_STEPS,
    SIGMA_CONVENTION,
    analyze,
)

SOURCE_CODES = {
    name: code for code, name in EVALUATION_CONDITION_SOURCE_NAMES.items()
}
ALL_CONDITION_SOURCES = (
    "autonomous",
    "off",
    "oracle_matched",
    "oracle_shuffled",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensors(
    rank: int,
    *,
    arm_gain: float,
    clean_offset: float = 0.0,
    evaluation_seed: int = 20_260_726,
    condition_sources: tuple[str, ...] = ("autonomous",),
    oracle_leakage_flag: int = 1,
):
    video_clean = (
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        .reshape(1, 1, 3, 1, 1)
        .add(clean_offset)
    )
    tf_clean = (
        torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
        .reshape(1, 1, 3, 1, 1)
        .add(clean_offset)
    )
    ground_truth = torch.tensor(
        [
            [
                [[[20 + rank, 30 + rank]], [[40, 50]], [[60, 70]]],
                [[[30 + rank, 40 + rank]], [[50, 60]], [[70, 80]]],
            ]
        ],
        dtype=torch.uint8,
    ).permute(0, 2, 1, 3, 4).contiguous()
    generator = torch.Generator().manual_seed(evaluation_seed + rank)
    video_initial = torch.randn(
        video_clean.shape, dtype=video_clean.dtype, generator=generator
    )
    tf_initial_noise = torch.randn(
        tf_clean.shape, dtype=tf_clean.dtype, generator=generator
    )
    tf_initial = tf_initial_noise.clone()
    tf_initial[:, :, :1] = tf_clean[:, :, :1]
    tensors = {
        "video_clean": video_clean,
        "tf_clean": tf_clean,
        "video_initial_state": video_initial,
        "tf_initial_state": tf_initial,
        "tf_initial_noise": tf_initial_noise,
        "ground_truth_future_uint8": ground_truth,
        "history_latent_frames": torch.tensor([1], dtype=torch.int64),
        "condition_on_tf": torch.tensor([int(arm_gain < 1.0)], dtype=torch.int64),
        "condition_mode_code": torch.tensor(
            [int(arm_gain < 1.0)], dtype=torch.int64
        ),
        "evaluation_noise_seed": torch.tensor([evaluation_seed], dtype=torch.int64),
        "evaluation_nfe_steps": torch.tensor(NFE_STEPS, dtype=torch.int64),
        "evaluation_condition_source_codes": torch.tensor(
            [SOURCE_CODES[source] for source in condition_sources],
            dtype=torch.int64,
        ),
        "oracle_sources_are_leakage": torch.tensor(
            [oracle_leakage_flag], dtype=torch.int64
        ),
    }
    source_gains = {
        "autonomous": arm_gain,
        "off": 1.5,
        "oracle_matched": 0.2,
        "oracle_shuffled": 0.8,
    }
    for source in condition_sources:
        infix = "" if source == "autonomous" else f"_{source}"
        for nfe in NFE_STEPS:
            source_gain = source_gains[source]
            latent_error = source_gain * (0.40 / nfe + 0.01 * rank)
            pixel_error = max(1, round(source_gain * (12 / nfe + rank)))
            frame_error = torch.tensor(
                [pixel_error, 2 * pixel_error], dtype=torch.int16
            ).reshape(1, 1, 2, 1, 1)
            tensors[f"video_final{infix}_nfe_{nfe}"] = (
                video_clean + latent_error
            )
            tensors[f"tf_final{infix}_nfe_{nfe}"] = (
                tf_clean + 2.0 * latent_error
            )
            tensors[f"decoded_future{infix}_nfe_{nfe}"] = torch.clamp(
                ground_truth.to(torch.int16) + frame_error, 0, 255
            ).to(torch.uint8)
    return tensors


def _write_artifact(
    arm_root: Path,
    *,
    rank: int,
    arm_gain: float,
    clean_offset: float = 0.0,
    evaluation_seed: int = 20_260_726,
    condition_sources: tuple[str, ...] = ("autonomous",),
    oracle_leakage_flag: int = 1,
    truncate_video_nfe_4: bool = False,
    drop_key: str | None = None,
) -> Path:
    dataset = "ABC_0"
    folder = arm_root / dataset
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"latent_trajectory_rank_{rank}.safetensors"
    tensors = _tensors(
        rank,
        arm_gain=arm_gain,
        clean_offset=clean_offset,
        evaluation_seed=evaluation_seed,
        condition_sources=condition_sources,
        oracle_leakage_flag=oracle_leakage_flag,
    )
    if truncate_video_nfe_4:
        tensors["video_final_nfe_4"] = tensors["video_final_nfe_4"][:, :, :-1]
    if drop_key is not None:
        tensors.pop(drop_key)
    save_file(
        tensors,
        str(path),
        metadata={
            "iteration": "99",
            "dataset": dataset,
            "sigma_convention": SIGMA_CONVENTION,
        },
    )
    sidecar = {
        "iteration": 99,
        "dataset": dataset,
        "global_rank": rank,
        "sigma_convention": SIGMA_CONVENTION,
        "tensors": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in tensors.items()
        },
        "safetensors_sha256": _sha256(path),
    }
    path.with_suffix(".json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    return path


def _matched_arms(
    root: Path,
    *,
    condition_sources: tuple[str, ...] = ("autonomous",),
) -> dict[str, Path]:
    arms = {
        "zero": root / "runs" / "zero" / "visualization" / "iter_99",
        "correct": root / "runs" / "correct" / "visualization" / "iter_99",
    }
    for rank in range(2):
        _write_artifact(
            arms["zero"],
            rank=rank,
            arm_gain=1.0,
            condition_sources=condition_sources,
        )
        _write_artifact(
            arms["correct"],
            rank=rank,
            arm_gain=0.5,
            condition_sources=condition_sources,
        )
    return arms


def test_analyzes_matched_ranks_and_reports_favorable_paired_deltas(tmp_path):
    arms = _matched_arms(tmp_path)
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()
    output = output_dir / "result.json"

    payload = analyze(
        arms,
        baseline="zero",
        output=output,
        bootstrap_samples=500,
        bootstrap_seed=123,
    )

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert payload["provenance"]["paired_unit_count"] == 2
    assert payload["bootstrap"]["seed"] == 123
    delta = payload["aggregate"]["paired_deltas"]["correct"]["nfe"]["4"]
    assert delta["video_future_nmse"]["mean"] < 0
    assert delta["tf_future_nmse"]["mean"] < 0
    assert delta["decoded_mse_unit_range"]["mean"] < 0
    assert delta["decoded_psnr_db"]["mean"] > 0
    assert (
        delta["decoded_temporal_difference_mse_unit_range"]["mean"] < 0
    )
    assert delta["video_future_nmse"]["favorable_fraction"] == 1.0
    assert (
        payload["per_paired_unit"][0]["arms"]["correct"]["metrics"]["1"][
            "video_future_nmse"
        ]
        > payload["per_paired_unit"][0]["arms"]["correct"]["metrics"]["8"][
            "video_future_nmse"
        ]
    )


def test_reports_oracle_leakage_source_metrics_and_within_arm_deltas(tmp_path):
    arms = _matched_arms(
        tmp_path, condition_sources=ALL_CONDITION_SOURCES
    )
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()

    payload = analyze(
        arms,
        baseline="zero",
        output=output_dir / "result.json",
        bootstrap_samples=500,
        bootstrap_seed=123,
    )

    assert payload["oracle_diagnostics"] == {
        "oracle_sources_are_leakage": True,
        "deployable_evidence": False,
        "interpretation": (
            "oracle_matched and oracle_shuffled consume hidden-future TF "
            "content and are leakage-only mechanism diagnostics; they "
            "cannot establish causal or deployable generation quality"
        ),
    }
    within_sources = payload["aggregate"]["within_arm_condition_sources"][
        "correct"
    ]
    assert tuple(within_sources["declared_sources"]) == ALL_CONDITION_SOURCES
    assert not within_sources["sources"]["off"]["oracle_leakage"]
    assert within_sources["sources"]["oracle_matched"]["oracle_leakage"]
    comparisons = payload["aggregate"]["within_arm_source_deltas"]["correct"]
    assert set(comparisons) == {
        "oracle_matched_minus_off",
        "oracle_matched_minus_oracle_shuffled",
    }
    for comparison in comparisons.values():
        assert comparison["oracle_leakage"]
        assert not comparison["deployable_evidence"]
        delta = comparison["nfe"]["4"]
        assert delta["video_future_nmse"]["mean"] < 0
        assert delta["tf_future_nmse"]["mean"] < 0
        assert delta["decoded_mse_unit_range"]["mean"] < 0
        assert delta["decoded_psnr_db"]["mean"] > 0
        assert (
            delta["decoded_temporal_difference_mse_unit_range"]["mean"] < 0
        )
    per_unit_sources = payload["per_paired_unit"][0]["arms"]["correct"][
        "condition_source_metrics"
    ]
    assert set(per_unit_sources) == set(ALL_CONDITION_SOURCES)
    assert per_unit_sources["oracle_shuffled"]["oracle_leakage"]


def test_rejects_incomplete_declared_oracle_source_keys(tmp_path):
    arms = _matched_arms(
        tmp_path, condition_sources=ALL_CONDITION_SOURCES
    )
    _write_artifact(
        arms["correct"],
        rank=1,
        arm_gain=0.5,
        condition_sources=ALL_CONDITION_SOURCES,
        drop_key="decoded_future_oracle_matched_nfe_4",
    )
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()

    with pytest.raises(ArtifactValidationError, match="inventory is incomplete"):
        analyze(
            arms,
            baseline="zero",
            output=output_dir / "result.json",
            bootstrap_samples=100,
        )


def test_rejects_oracle_artifacts_not_declared_as_leakage(tmp_path):
    arms = _matched_arms(
        tmp_path, condition_sources=ALL_CONDITION_SOURCES
    )
    _write_artifact(
        arms["correct"],
        rank=1,
        arm_gain=0.5,
        condition_sources=ALL_CONDITION_SOURCES,
        oracle_leakage_flag=0,
    )
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()

    with pytest.raises(
        ArtifactValidationError, match="oracle_sources_are_leakage"
    ):
        analyze(
            arms,
            baseline="zero",
            output=output_dir / "result.json",
            bootstrap_samples=100,
        )


def test_bootstrap_is_reproducible_and_output_is_exclusive(tmp_path):
    arms = _matched_arms(tmp_path)
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()
    first_path = output_dir / "first.json"
    second_path = output_dir / "second.json"
    first = analyze(
        arms,
        baseline="zero",
        output=first_path,
        bootstrap_samples=200,
        bootstrap_seed=456,
    )
    second = analyze(
        arms,
        baseline="zero",
        output=second_path,
        bootstrap_samples=200,
        bootstrap_seed=456,
    )
    assert first["aggregate"] == second["aggregate"]

    with pytest.raises(ArtifactValidationError, match="already exists"):
        analyze(
            arms,
            baseline="zero",
            output=first_path,
            bootstrap_samples=200,
        )


def test_rejects_clean_data_mismatch_between_arms(tmp_path):
    arms = _matched_arms(tmp_path)
    _write_artifact(
        arms["correct"], rank=1, arm_gain=0.5, clean_offset=0.25
    )
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()

    with pytest.raises(ArtifactValidationError, match="paired provenance"):
        analyze(
            arms,
            baseline="zero",
            output=output_dir / "result.json",
            bootstrap_samples=100,
        )


def test_rejects_evaluation_seed_and_initial_state_mismatch(tmp_path):
    arms = _matched_arms(tmp_path)
    _write_artifact(
        arms["correct"],
        rank=1,
        arm_gain=0.5,
        evaluation_seed=20_260_727,
    )
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()

    with pytest.raises(ArtifactValidationError, match="provenance mismatch"):
        analyze(
            arms,
            baseline="zero",
            output=output_dir / "result.json",
            bootstrap_samples=100,
        )


def test_rejects_missing_required_key_and_does_not_create_output(tmp_path):
    arms = _matched_arms(tmp_path)
    _write_artifact(
        arms["correct"],
        rank=1,
        arm_gain=0.5,
        drop_key="decoded_future_nfe_4",
    )
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()
    output = output_dir / "result.json"

    with pytest.raises(ArtifactValidationError, match="missing required tensor keys"):
        analyze(
            arms,
            baseline="zero",
            output=output,
            bootstrap_samples=100,
        )
    assert not output.exists()


def test_rejects_latent_shape_mismatch(tmp_path):
    arms = _matched_arms(tmp_path)
    _write_artifact(
        arms["correct"],
        rank=1,
        arm_gain=0.5,
        truncate_video_nfe_4=True,
    )
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()

    with pytest.raises(ArtifactValidationError, match="video_final_nfe_4 shape"):
        analyze(
            arms,
            baseline="zero",
            output=output_dir / "result.json",
            bootstrap_samples=100,
        )


def test_rejects_tampered_sidecar_and_output_inside_arm(tmp_path):
    arms = _matched_arms(tmp_path)
    artifact = (
        arms["correct"] / "ABC_0" / "latent_trajectory_rank_1.safetensors"
    )
    sidecar_path = artifact.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["safetensors_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()

    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        analyze(
            arms,
            baseline="zero",
            output=output_dir / "result.json",
            bootstrap_samples=100,
        )

    clean_root = tmp_path / "clean"
    clean_arms = _matched_arms(clean_root)
    with pytest.raises(ArtifactValidationError, match="outside every"):
        analyze(
            clean_arms,
            baseline="zero",
            output=clean_arms["zero"] / "result.json",
            bootstrap_samples=100,
        )
