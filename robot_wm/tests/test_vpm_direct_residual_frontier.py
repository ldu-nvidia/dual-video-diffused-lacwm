import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "vpm_direct_residual_frontier.py"
)
SPEC = importlib.util.spec_from_file_location("vpm_direct_residual_frontier", SCRIPT)
frontier = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(frontier)


def _digest(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def _row(endpoint: str, clip: int, seed: int, error: float):
    common = f"{clip}-{seed}"
    return frontier.ladder.identity_payload(
        {
            "schema": frontier.ROW_SCHEMA,
            "endpoint": endpoint,
            "clip_index": clip,
            "clip_id": f"clip-{clip}",
            "episode_dir": f"episode-{clip}",
            "noise_seed": seed,
            "wan_calls": 1,
            "teacher_calls": 0,
            "vjepa_target_array_opened": False,
            "future_target_entered_correction": False,
            "fresh_reserve_outcome_opened": True,
            "validation_opened": False,
            "protected_test_opened": False,
            "metrics": {
                "residual_r2": 0.4 if endpoint == "DIRECT_ALIGNED" else 0.1,
                "residual_cosine": 0.6 if endpoint == "DIRECT_ALIGNED" else 0.2,
                **{metric: error for metric in frontier.ERROR_METRICS},
            },
            "tensor_sha256": {
                "initial_video": _digest(f"initial-{common}"),
                "actions": _digest(f"actions-{common}"),
                "raw_future_target": _digest(f"target-{common}"),
                "raw_history_boundary": _digest(f"history-{common}"),
                "final_video": _digest(
                    f"{'off' if endpoint in {'VPM_OFF', 'ZERO'} else endpoint}-{common}"
                ),
            },
        }
    )


def _inventory(aligned: float = 0.88, shuffled: float = 1.01):
    rows = []
    values = {
        "VPM_OFF": 1.0,
        "ZERO": 1.0,
        "DIRECT_ALIGNED": aligned,
        "DIRECT_SHUFFLED": shuffled,
    }
    for clip in range(*frontier.OUTCOME_RANGE):
        for seed in frontier.OUTCOME_NOISE_SEEDS:
            for endpoint in frontier.ENDPOINTS:
                rows.append(_row(endpoint, clip, seed, values[endpoint]))
    return rows


def test_partition_contract_reserves_only_the_fresh_final_32_episodes():
    rows = [
        {
            "split": "train",
            "auxiliary_index": index,
            "clip_id": f"clip-{index}",
            "episode_dir": f"episode-{index}",
        }
        for index in range(512)
    ]
    result = frontier._validate_manifest(rows)
    assert result["ranges"]["fit"] == [128, 384]
    assert result["ranges"]["fresh_reserve_outcome"] == [480, 512]
    rows[500] = {**rows[500], "episode_dir": "episode-200"}
    with pytest.raises(frontier.FrontierError, match="identity differs"):
        frontier._validate_manifest(rows)


def test_synthetic_sample_specific_improvement_passes_every_frozen_gate():
    result = frontier.analyze_rows(_inventory(), adapter_p95_ms=0.25)
    assert result["decision"] == "ADVANCE_VPM_DIRECT_RESIDUAL"
    assert result["gates"]["all_passed"] is True
    comparison = result["comparisons"]["DIRECT_ALIGNED_vs_DIRECT_SHUFFLED"]
    assert comparison["decoded_mse_unit_range"]["relative_improvement_percent"] > 10
    assert comparison["decoded_mse_unit_range"]["favorable_episode_fraction"] == 1


def test_population_only_or_no_improvement_cannot_pass():
    result = frontier.analyze_rows(
        _inventory(aligned=1.0, shuffled=1.01), adapter_p95_ms=0.25
    )
    assert result["decision"] == "STOP_VPM_DIRECT_RESIDUAL"
    assert result["gates"]["velocity_beats_VPM_OFF_3pct_ci1"] is False


def test_analysis_fails_closed_on_zero_mutation_and_nonfinite_metric():
    rows = _inventory()
    row = rows[1]
    assert row["endpoint"] == "ZERO"
    rows[1] = frontier.ladder.identity_payload(
        {
            **row,
            "tensor_sha256": {
                **row["tensor_sha256"],
                "final_video": _digest("mutated-zero"),
            },
        }
    )
    with pytest.raises(frontier.FrontierError, match="ZERO is not bit-exact"):
        frontier.analyze_rows(rows, adapter_p95_ms=0.25)

    rows = _inventory()
    row = rows[2]
    rows[2] = frontier.ladder.identity_payload(
        {**row, "metrics": {**row["metrics"], "video_future_nmse": "nan"}}
    )
    with pytest.raises(frontier.FrontierError, match="metric/hash payload"):
        frontier.analyze_rows(rows, adapter_p95_ms=0.25)


def test_protocol_contains_frozen_reserve_seeds_and_hard_controls():
    protocol = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "experiments"
        / "VPM_DIRECT_RESIDUAL_FRONTIER_PROTOCOL.md"
    ).read_text()
    for value in (
        "480--511",
        "20260836",
        "DIRECT_SHUFFLED",
        "bit-exact",
        "not dual diffusion",
    ):
        assert value in protocol
