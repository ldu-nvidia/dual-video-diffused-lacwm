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
                "morphology_index": _digest(f"morphology-{common}"),
                "action_control": _digest(f"action-control-{common}"),
                "auxiliary_noise": _digest(f"aux-noise-{common}"),
                "raw_history_input": _digest(f"raw-history-input-{common}"),
                "history_reference": _digest(f"reference-{common}"),
                "latent_flow_target": _digest(f"flow-target-{common}"),
                "raw_future_target": _digest(f"target-{common}"),
                "raw_history_boundary": _digest(f"history-{common}"),
                "scoring_future_uint8": _digest(f"target-u8-{common}"),
                "scoring_history_uint8": _digest(f"history-u8-{common}"),
                "final_video": _digest(
                    f"{'off' if endpoint in {'VPM_OFF', 'ZERO'} else endpoint}-{common}"
                ),
            },
            "scoring_target_constructed_after_all_endpoints": True,
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


def _registered_samples():
    return [
        {
            "clip_index": clip,
            "clip_id": f"clip-{clip}",
            "episode_dir": f"episode-{clip}",
        }
        for clip in range(*frontier.OUTCOME_RANGE)
    ]


def test_partition_contract_reserves_only_the_fresh_final_31_episodes():
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
    assert result["ranges"]["fresh_reserve_outcome"] == [480, 511]
    assert result["ranges"]["excluded_historical_constructor_probe"] == [511, 512]
    rows[500] = {**rows[500], "episode_dir": "episode-200"}
    with pytest.raises(frontier.FrontierError, match="identity differs"):
        frontier._validate_manifest(rows)


def test_synthetic_sample_specific_improvement_passes_every_frozen_gate():
    result = frontier.analyze_rows(
        _inventory(),
        adapter_p95_ms=0.25,
        registered_samples=_registered_samples(),
    )
    assert result["decision"] == "ADVANCE_VPM_DIRECT_RESIDUAL"
    assert result["gates"]["all_passed"] is True
    comparison = result["comparisons"]["DIRECT_ALIGNED_vs_DIRECT_SHUFFLED"]
    assert comparison["decoded_mse_unit_range"]["relative_improvement_percent"] > 10
    assert comparison["decoded_mse_unit_range"]["favorable_episode_fraction"] == 1


def test_population_only_or_no_improvement_cannot_pass():
    result = frontier.analyze_rows(
        _inventory(aligned=1.0, shuffled=1.01),
        adapter_p95_ms=0.25,
        registered_samples=_registered_samples(),
    )
    assert result["decision"] == "STOP_VPM_DIRECT_RESIDUAL"
    assert result["gates"]["velocity_beats_VPM_OFF_3pct_ci1"] is False


def test_negative_or_nonfinite_adapter_latency_can_never_pass():
    for value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(frontier.FrontierError, match="finite and nonnegative"):
            frontier.analyze_rows(
                _inventory(),
                adapter_p95_ms=value,
                registered_samples=_registered_samples(),
            )


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
        frontier.analyze_rows(
            rows,
            adapter_p95_ms=0.25,
            registered_samples=_registered_samples(),
        )

    rows = _inventory()
    row = rows[2]
    rows[2] = frontier.ladder.identity_payload(
        {**row, "metrics": {**row["metrics"], "video_future_nmse": "nan"}}
    )
    with pytest.raises(frontier.FrontierError, match="metric/hash payload"):
        frontier.analyze_rows(
            rows,
            adapter_p95_ms=0.25,
            registered_samples=_registered_samples(),
        )


def test_analysis_binds_rows_to_registered_clip_and_episode_identity():
    rows = _inventory()
    row = rows[2]
    rows[2] = frontier.ladder.identity_payload(
        {**row, "clip_id": "wrong-clip", "episode_dir": "wrong-episode"}
    )
    with pytest.raises(frontier.FrontierError, match="registered reserve identity"):
        frontier.analyze_rows(
            rows,
            adapter_p95_ms=0.25,
            registered_samples=_registered_samples(),
        )


def test_phase_dataset_wrapper_rejects_out_of_partition_and_counts_accesses():
    class Dataset:
        def __len__(self):
            return 512

        def __getitem__(self, index):
            return index

    wrapped = frontier._IndexAuditedDataset(Dataset(), (128, 129))
    assert wrapped[128] == 128
    assert wrapped[129] == 129
    wrapped.assert_exact_accesses(1)
    with pytest.raises(frontier.FrontierError, match="forbidden row 511"):
        wrapped[511]


def test_array_validation_probes_only_the_explicit_role_rows(monkeypatch):
    import numpy as np
    import sys
    import types

    base = types.ModuleType("robot_wm.datasets.base")
    base.Dataset = type("Dataset", (), {})
    monkeypatch.setitem(sys.modules, "robot_wm.datasets.base", base)
    source = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "abc"
        / "video_residual_anchor_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("role_scoped_dataset", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    ABCVideoResidualAnchorDataset = module.ABCVideoResidualAnchorDataset

    dataset = ABCVideoResidualAnchorDataset.__new__(ABCVideoResidualAnchorDataset)
    dataset.validation_sample_indices = (1,)
    rgbs = np.zeros((3, 2), dtype=np.float16)
    actions = np.zeros((3, 2), dtype=np.float32)
    rgbs[0, 0] = np.nan  # Forbidden row must not be sampled by validation.
    dataset._open_rgbs = lambda: rgbs
    dataset._open_actions = lambda: actions
    dataset._validate_arrays()
    rgbs[1, 0] = np.nan
    with pytest.raises(FloatingPointError, match="sample 1"):
        dataset._validate_arrays()


def test_no_auxiliary_dataset_disables_random_validity_substitution(monkeypatch):
    from types import SimpleNamespace

    import hydra.utils
    from omegaconf import OmegaConf

    observed = {}

    def instantiate(config):
        observed["future_validity"] = OmegaConf.to_container(
            config.future_validity, resolve=True
        )
        child = SimpleNamespace(
            auxiliary_target_array_opened=False,
            validation_sample_indices=tuple(config.datasets.ABC.validation_sample_indices),
        )
        validity = SimpleNamespace(
            enabled=bool(config.future_validity.enabled),
            max_retries=int(config.future_validity.max_retries),
        )
        return SimpleNamespace(datasets={"ABC": child}, future_validity=validity)

    monkeypatch.setattr(hydra.utils, "instantiate", instantiate)
    config = SimpleNamespace(
        dataset=OmegaConf.create(
            {
                "future_validity": {"enabled": True, "max_retries": 8},
                "datasets": {
                    "ABC": {
                        "_target_": "unused",
                        "expected_cache_id": "unused",
                        "validate_target_samples": True,
                    }
                },
            }
        )
    )
    registration = {
        "inputs": {"train_manifest": {"sha256": "a" * 64}},
        "cache_arrays": {
            "rgb": {"sha256": "b" * 64},
            "actions": {"sha256": "c" * 64},
        },
    }
    frontier.ladder._no_auxiliary_dataset(
        config, registration, validation_sample_indices=(128, 383)
    )
    assert observed["future_validity"] == {"enabled": False, "max_retries": 0}


def test_timing_trace_recomputes_all_summaries_and_singleton_batch(tmp_path):
    rows = []
    for noise_seed in frontier.OUTCOME_NOISE_SEEDS:
        for start in range(frontier.OUTCOME_RANGE[0], frontier.OUTCOME_RANGE[1], 2):
            clips = list(
                range(start, min(start + 2, frontier.OUTCOME_RANGE[1]))
            )
            rows.append(
                frontier.ladder.identity_payload(
                    {
                        "schema": frontier.TIMING_SCHEMA,
                        "batch_ordinal": len(rows),
                        "noise_seed": noise_seed,
                        "clip_indices": clips,
                        "batch_size": len(clips),
                        "latency_ms": {
                            "causal_preparation": 0.5,
                            "wan": 10.0,
                            "adapter": {arm: 0.2 for arm in frontier.ARMS},
                            "euler": {
                                endpoint: 0.1 for endpoint in frontier.ENDPOINTS
                            },
                            "decoder": {
                                endpoint: 5.0 for endpoint in frontier.ENDPOINTS
                            },
                            "full_aligned_endpoint": 15.8,
                        },
                    }
                )
            )
    assert len(rows) == 64
    assert rows[-1]["clip_indices"] == [510]
    path = tmp_path / "timing_rows.jsonl"
    frontier.ladder.exclusive_jsonl(path, rows)
    endpoint = {
        "timing_rows": frontier.ladder.file_record(path),
        "timing_row_count": 64,
        "adapter_latency_ms_per_batch": {
            arm: frontier._latency_summary([0.2] * 64) for arm in frontier.ARMS
        },
        "wan_latency_ms_per_batch": frontier._latency_summary([10.0] * 64),
        "causal_preparation_latency_ms_per_batch": frontier._latency_summary(
            [0.5] * 64
        ),
        "euler_latency_ms_per_batch": {
            endpoint: frontier._latency_summary([0.1] * 64)
            for endpoint in frontier.ENDPOINTS
        },
        "decoder_latency_ms_per_batch": {
            endpoint: frontier._latency_summary([5.0] * 64)
            for endpoint in frontier.ENDPOINTS
        },
        "full_aligned_endpoint_latency_ms_per_batch": frontier._latency_summary(
            [15.8] * 64
        ),
    }
    observed_path, aligned_p95 = frontier._validated_timing_trace(
        tmp_path, endpoint
    )
    assert observed_path == path
    assert aligned_p95 == pytest.approx(0.2)
    endpoint["adapter_latency_ms_per_batch"]["DIRECT_ALIGNED"]["p95"] = 0.3
    with pytest.raises(frontier.FrontierError, match="summary differs"):
        frontier._validated_timing_trace(tmp_path, endpoint)


def test_protocol_contains_frozen_reserve_seeds_and_hard_controls():
    protocol = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "experiments"
        / "VPM_DIRECT_RESIDUAL_FRONTIER_PROTOCOL.md"
    ).read_text()
    for value in (
        "480--510",
        "constructor probe",
        "20260836",
        "DIRECT_SHUFFLED",
        "bit-exact",
        "not dual diffusion",
    ):
        assert value in protocol
