from __future__ import annotations

import hashlib

import pytest
import torch

from tools import causal_compressibility_ladder as ladder
from tools import vpm_invertible_multirate_probe as probe


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("band", ("LL", "HH"))
def test_projectors_are_future_only_orthogonal_and_view_isolated(band: str):
    generator = torch.Generator().manual_seed(7)
    value = torch.randn((2, 3, 4, 6, 12), generator=generator)
    projected = probe.future_spatial_project(
        value, history_frames=2, band=band, view_count=3
    )
    complement = probe.future_spatial_complement(
        value, history_frames=2, band=band, view_count=3
    )
    target = probe._future_only(value, 2)
    assert torch.count_nonzero(projected[:, :, :2]) == 0
    assert torch.count_nonzero(complement[:, :, :2]) == 0
    torch.testing.assert_close(projected + complement, target, atol=2e-6, rtol=0)
    receipt = probe.projector_contract(
        value, history_frames=2, band=band, view_count=3
    )
    assert receipt["rank_fraction"] == 0.25
    assert receipt["view_isolation_bit_exact"] is True
    assert receipt["history_nonzero"] == 0


def test_ll_and_hh_project_exact_known_patterns():
    value = torch.zeros((1, 1, 2, 2, 6), dtype=torch.float32)
    value[:, :, 1, :, :2] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ll = probe.future_spatial_project(
        value, history_frames=1, band="LL", view_count=3
    )
    hh = probe.future_spatial_project(
        value, history_frames=1, band="HH", view_count=3
    )
    torch.testing.assert_close(
        ll[0, 0, 1, :, :2], torch.full((2, 2), 2.5), atol=0, rtol=0
    )
    torch.testing.assert_close(
        hh[0, 0, 1, :, :2],
        torch.tensor([[0.0, -0.0], [-0.0, 0.0]]),
        atol=0,
        rtol=0,
    )
    checkerboard = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    value[:, :, 1, :, :2] = checkerboard
    hh = probe.future_spatial_project(
        value, history_frames=1, band="HH", view_count=3
    )
    torch.testing.assert_close(hh[0, 0, 1, :, :2], checkerboard)


def test_materialization_shares_first_call_and_locks_declared_subspaces():
    generator = torch.Generator().manual_seed(11)
    initial = torch.randn((2, 2, 4, 4, 12), generator=generator)
    reference = torch.zeros_like(initial)
    reference[:, :, :2] = torch.randn((2, 2, 2, 4, 12), generator=generator)
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    timesteps = (torch.tensor(1000.0), torch.tensor(500.0))
    calls = []

    def velocity(state, timestep, label, sigma):
        calls.append(label)
        # State dependence ensures the second-call intervention is observable.
        return 0.2 * state.float() + 0.0001 * timestep.float() + 0.01 * sigma

    ledger = probe.EventLedger()
    result = probe.materialize_split_endpoints(
        initial=initial,
        reference=reference,
        history_frames=2,
        sigmas=sigmas,
        timesteps=timesteps,
        velocity_call=velocity,
        ledger=ledger,
        view_count=3,
    )
    assert calls == ["SHARED_FIRST", *probe.TWO_CALL_ENDPOINTS]
    assert result.actual_wan_calls == 6
    assert result.conceptual_wan_calls["VPM1"] == 1
    assert all(result.conceptual_wan_calls[name] == 2 for name in probe.TWO_CALL_ENDPOINTS)
    assert result.synchronous_identity_max_abs <= probe.MAX_ABS_TOLERANCE
    assert ledger.barrier_closed is True
    assert ledger.target_constructed is False

    aligned_p = probe.future_spatial_project(
        result.midpoints["LL_FIRST_ALIGNED"],
        history_frames=2,
        band="LL",
        view_count=3,
    )
    shuffled_p = probe.future_spatial_project(
        result.midpoints["LL_FIRST_EPISODE_SHUFFLED"],
        history_frames=2,
        band="LL",
        view_count=3,
    )
    reversed_p = probe.future_spatial_project(
        result.midpoints["LL_FIRST_TIME_REVERSED"],
        history_frames=2,
        band="LL",
        view_count=3,
    )
    torch.testing.assert_close(shuffled_p, torch.roll(aligned_p, 1, 0))
    expected_reversed = aligned_p.clone()
    expected_reversed[:, :, 2:] = aligned_p[:, :, 2:].flip(2)
    torch.testing.assert_close(reversed_p, expected_reversed)

    for endpoint, band in (
        ("LL_FIRST_ALIGNED", "LL"),
        ("LL_FIRST_EPISODE_SHUFFLED", "LL"),
        ("LL_FIRST_TIME_REVERSED", "LL"),
        ("HH_FIRST_RANK_MATCHED", "HH"),
    ):
        observed = probe.future_spatial_project(
            result.states[endpoint], history_frames=2, band=band, view_count=3
        )
        torch.testing.assert_close(
            observed, result.locked_states[endpoint], atol=2e-6, rtol=0
        )
    for state in result.states.values():
        torch.testing.assert_close(state[:, :, :2], reference[:, :, :2])

    ledger.construct_target()
    receipt = ledger.receipt()
    assert receipt["target_after_all_endpoints"] is True
    with pytest.raises(probe.MultirateError):
        ledger.require_pre_target()


def test_target_barrier_fails_closed():
    ledger = probe.EventLedger()
    with pytest.raises(probe.MultirateError):
        ledger.construct_target()
    for endpoint in probe.ENDPOINTS[:-1]:
        ledger.endpoint(endpoint)
    with pytest.raises(probe.MultirateError):
        ledger.close_endpoint_barrier()


def _analysis_fixture(*, shuffled_equals_primary: bool = False):
    rows = []
    common_names = {
        "initial_video",
        "actions",
        "morphology_index",
        "action_control",
        "auxiliary_noise",
        "raw_history_input",
        "history_reference",
        "first_velocity",
        "first_clean_estimate",
        "latent_future_target",
        "raw_future_target",
        "raw_history_boundary",
    }
    registered = []
    for clip in range(*probe.DEV_RANGE):
        registered.append(
            {
                "clip_index": clip,
                "clip_id": f"clip-{clip}",
                "episode_dir": f"episode-{clip}",
            }
        )
    for endpoint in probe.ENDPOINTS:
        if endpoint == probe.PRIMARY:
            primary_error = 0.80
        elif shuffled_equals_primary and endpoint == "LL_FIRST_EPISODE_SHUFFLED":
            primary_error = 0.80
        elif endpoint in {
            "LL_FIRST_EPISODE_SHUFFLED",
            "LL_FIRST_TIME_REVERSED",
            "HH_FIRST_RANK_MATCHED",
        }:
            primary_error = 0.90
        else:
            primary_error = 1.00
        for clip in range(*probe.DEV_RANGE):
            for seed in probe.DEV_NOISE_SEEDS:
                metrics = {name: primary_error for name in probe.ALL_METRICS}
                metrics["ll_temporal_delta_cosine"] = 0.5
                metrics["ll_prediction_energy_fraction"] = 0.6
                metrics["hh_prediction_energy_fraction"] = 0.1
                metrics["p_lock_max_abs_error"] = 0.0
                shared = {
                    name: _hash(f"{name}-{clip}-{seed}") for name in common_names
                }
                rows.append(
                    ladder.identity_payload(
                        {
                            "schema": probe.ROW_SCHEMA,
                            "endpoint": endpoint,
                            "clip_index": clip,
                            "clip_id": f"clip-{clip}",
                            "episode_dir": f"episode-{clip}",
                            "noise_seed": seed,
                            "conceptual_wan_calls": 1 if endpoint == "VPM1" else 2,
                            "shared_first_wan_call": True,
                            "actual_batch_wan_calls": 6,
                            "teacher_calls": 0,
                            "feature_encoder_calls": 0,
                            "auxiliary_target_calls": 0,
                            "optimizer_updates": 0,
                            "new_parameters": 0,
                            "condition_on_tf": False,
                            "condition_on_tf_clock": False,
                            "target_after_all_endpoints": True,
                            "event_ledger_sha256": _hash(f"ledger-{clip}-{seed}"),
                            "metrics": metrics,
                            "tensor_sha256": {
                                **shared,
                                "midpoint": _hash(f"mid-{endpoint}-{clip}-{seed}"),
                                "final_video": _hash(f"final-{endpoint}-{clip}-{seed}"),
                            },
                            "access": {
                                "prior_inspected_development_opened": True,
                                "fresh_reserve_480_510_opened": False,
                                "constructor_probe_511_opened": False,
                                "validation_opened": False,
                                "protected_test_opened": False,
                                "vjepa_target_array_opened": False,
                            },
                        }
                    )
                )
    timings = []
    ordinal = 0
    for seed in probe.DEV_NOISE_SEEDS:
        for start in range(probe.DEV_RANGE[0], probe.DEV_RANGE[1], 2):
            projection = {name: 0.0 for name in probe.ENDPOINTS}
            projection[probe.PRIMARY] = 1.0
            composed = {name: 100.0 for name in probe.ENDPOINTS}
            composed[probe.PRIMARY] = 104.0
            timings.append(
                {
                    "latency_ms": {
                        "projection": projection,
                        "deployable_composed": composed,
                    }
                }
            )
            ordinal += 1
    return rows, timings, registered


def test_analysis_has_exact_fifteen_test_family_and_can_pass_all_gates():
    rows, timings, registered = _analysis_fixture()
    result = probe._analysis_payload(
        rows,
        timings,
        endpoint_receipt={},
        registered_samples=registered,
    )
    assert result["simultaneous_family"]["test_count"] == 15
    assert len(result["comparisons"]) == 5
    assert sum(len(value) for value in result["comparisons"].values()) == 15
    assert result["gates"]["all_passed"] is True
    assert result["decision"] == "GO_INVERTIBLE_MULTIRATE"


def test_analysis_rejects_sample_insensitive_ll_state():
    rows, timings, registered = _analysis_fixture(shuffled_equals_primary=True)
    result = probe._analysis_payload(
        rows,
        timings,
        endpoint_receipt={},
        registered_samples=registered,
    )
    assert result["gates"]["aligned_vs_episode_shuffled"]["all_passed"] is False
    assert result["decision"] == "NO_GO_GENERIC_MIXED_STATE"


def test_frozen_inventory_and_bonferroni_constant():
    assert probe.BASE_COMMIT == "997a9dae79d63627a65773ef19cc41462db85c7d"
    assert probe.DEV_RANGE == (416, 480)
    assert probe.DEV_NOISE_SEEDS == (20261101, 20261102, 20261103, 20261104)
    assert probe.BOOTSTRAP_SEED == 20261105
    assert probe.SIMULTANEOUS_COMPARISON_COUNT == 15
    assert probe.BONFERRONI_ONE_SIDED_ALPHA == pytest.approx(0.05 / 15)
