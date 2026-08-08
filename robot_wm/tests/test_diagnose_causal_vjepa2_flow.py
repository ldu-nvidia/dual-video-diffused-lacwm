from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from tools import diagnose_causal_vjepa2_flow as diagnostic


class _ConstantCleanModel(torch.nn.Module):
    def forward(
        self,
        noisy_video,
        noisy_auxiliary,
        t_video,
        t_auxiliary,
        history,
        actions,
        **kwargs,
    ):
        del t_video, t_auxiliary, history, actions, kwargs
        return SimpleNamespace(
            video_x=torch.zeros_like(noisy_video),
            auxiliary_x=torch.full_like(noisy_auxiliary, 2.0),
        )


def _small_sampler_inputs():
    generator = torch.Generator().manual_seed(19)
    return {
        "history": torch.randn(2, 3, generator=generator),
        "actions": torch.randn(2, 4, generator=generator),
        "video_noise": torch.randn(2, 2, generator=generator),
        "auxiliary_noise": torch.randn(2, 3, 2, generator=generator),
    }


def test_registered_schedules_and_inventory_are_exact():
    uniform = diagnostic.clean_time_schedule(
        4, "uniform", device=torch.device("cpu")
    )
    dense = diagnostic.clean_time_schedule(
        4, "clean_dense", device=torch.device("cpu")
    )
    torch.testing.assert_close(
        uniform,
        torch.tensor((0.0, 0.25, 0.5, 0.75, 1.0)),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        dense,
        torch.tensor((0.0, 0.4375, 0.75, 0.9375, 1.0)),
        rtol=0,
        atol=0,
    )
    assert diagnostic.endpoint_solver_configs() == (
        ("direct", None, 1),
        ("euler", "uniform", 1),
        ("euler", "uniform", 2),
        ("euler", "uniform", 4),
        ("euler", "uniform", 8),
        ("euler", "clean_dense", 2),
        ("euler", "clean_dense", 4),
        ("euler", "clean_dense", 8),
        ("midpoint", "uniform", 2),
        ("midpoint", "uniform", 4),
        ("midpoint", "uniform", 8),
    )
    assert diagnostic.training_distribution_times() == (
        0.0,
        0.025,
        0.05,
        0.1,
        0.125,
        0.2,
        0.25,
        0.35,
        0.375,
        0.5,
        0.625,
        0.65,
        0.75,
        0.8,
        0.875,
        0.9,
        0.95,
    )
    assert diagnostic.expected_rows_per_clip() == 161
    assert diagnostic.expected_batched_calls() == 237


def test_solver_call_budgets_and_uniform_euler_pre_call_states():
    inputs = _small_sampler_inputs()
    model = _ConstantCleanModel().eval()
    for solver, schedule, calls in diagnostic.endpoint_solver_configs():
        sample = diagnostic.sample_autonomous_solver(
            model,
            **inputs,
            solver=solver,
            schedule=schedule,
            nfe=calls,
        )
        assert sample.model_calls == calls
        assert all(len(chain) == calls for chain in sample.state_sha256_chain_by_example)
        expected_points = calls if solver == "euler" else 0
        assert len(sample.pre_call_points) == expected_points
        if solver != "euler" or schedule == "uniform":
            torch.testing.assert_close(
                sample.prediction,
                torch.full_like(sample.prediction, 2.0),
                rtol=0,
                atol=2e-6,
            )


def test_invalid_direct_and_midpoint_cells_fail_closed():
    inputs = _small_sampler_inputs()
    model = _ConstantCleanModel().eval()
    with pytest.raises(ValueError, match="direct-x"):
        diagnostic.sample_autonomous_solver(
            model, **inputs, solver="direct", schedule="uniform", nfe=1
        )
    with pytest.raises(ValueError, match="direct-x"):
        diagnostic.sample_autonomous_solver(
            model, **inputs, solver="direct", schedule=None, nfe=2
        )
    with pytest.raises(ValueError, match="midpoint"):
        diagnostic.sample_autonomous_solver(
            model, **inputs, solver="midpoint", schedule="uniform", nfe=1
        )
    with pytest.raises(ValueError, match="midpoint"):
        diagnostic.sample_autonomous_solver(
            model, **inputs, solver="midpoint", schedule="clean_dense", nfe=2
        )


def test_autonomous_api_and_synthetic_end_to_end_contract():
    signatures = diagnostic.assert_autonomous_api_has_no_clean_future()
    assert set(signatures) == {
        "sample_autonomous_dense",
        "sample_autonomous_solver",
    }
    result = diagnostic.run_synthetic_smoke()
    assert result["status"] == "pass"
    assert result["records"] == 254
    assert result["expected_rows_per_clip"] == 127
    assert result["actual_batched_student_calls"] == 203
    assert result["teacher_model_calls"] == 0
    assert result["clean_future_target_entered_autonomous_sampler"] is False
    assert result["t0_training_autonomous_bit_identical"] is True


def test_production_uses_the_producer_attested_cache_bridge_only():
    production_source = inspect.getsource(diagnostic._production_command)
    checkpoint_source = inspect.getsource(diagnostic._validate_checkpoint)
    assert "construct_producer_attested_dataset" in production_source
    assert "screen._construct_dataset" not in production_source
    assert "producer_cache_attestation" in production_source
    assert "screen._dataset_source_record" not in checkpoint_source
    assert "producer_dataset_source" in checkpoint_source


def test_target_hash_guard_rejects_a_target_in_the_autonomous_chain():
    target = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    hashes = diagnostic._per_example_hashes(target)
    fixed = (
        {
            "history_sha256": "1" * 64,
            "actions_sha256": "2" * 64,
            "initial_video_noise_sha256": "3" * 64,
            "initial_auxiliary_noise_sha256": "4" * 64,
        },
        {
            "history_sha256": "5" * 64,
            "actions_sha256": "6" * 64,
            "initial_video_noise_sha256": "7" * 64,
            "initial_auxiliary_noise_sha256": "8" * 64,
        },
    )
    with pytest.raises(diagnostic.DiagnosticError, match="target hash"):
        diagnostic._assert_target_absent_from_autonomous_chain(
            target=target,
            state_hash_chains=((hashes[0],), ("9" * 64,)),
            fixed_input_records=fixed,
        )


def test_primary_diagnosis_uses_only_the_frozen_c4_t025_pair():
    records = []
    for clip_id in ("a", "b", "c"):
        records.extend(
            (
                {
                    "family": "trajectory",
                    "state_source": "training_distribution",
                    "action_control": "matched",
                    "nfe": None,
                    "clean_time": 0.25,
                    "clip_id": clip_id,
                    "temporal_difference_nmse": 0.4,
                },
                {
                    "family": "trajectory",
                    "state_source": "autonomous_uniform_euler",
                    "action_control": "matched",
                    "nfe": 4,
                    "clean_time": 0.25,
                    "clip_id": clip_id,
                    "temporal_difference_nmse": 0.6,
                },
                {
                    "family": "trajectory",
                    "state_source": "autonomous_uniform_euler",
                    "action_control": "matched",
                    "nfe": 8,
                    "clean_time": 0.25,
                    "clip_id": clip_id,
                    "temporal_difference_nmse": 99.0,
                },
            )
        )
    result = diagnostic.primary_diagnosis(
        records, bootstrap_resamples=100, seed=7
    )
    assert result["cell"]["actual_calls"] == 4
    assert result["cell"]["clean_time"] == 0.25
    assert result["training_distribution_temporal_nmse"] == pytest.approx(0.4)
    assert result["autonomous_temporal_nmse"] == pytest.approx(0.6)
    assert result["autonomous_relative_worsening"] == pytest.approx(0.5)
    assert result["relative_worsening_one_sided_lower_bound_95"] == pytest.approx(
        0.5
    )
    assert result["rollout_drift_primary"] is True
