from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import causal_compressibility_ladder as ladder
from tools import vpm_invertible_multirate_probe as probe


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _record_serving_accesses(
    ledger: probe.EventLedger, indexes=(416, 417)
) -> None:
    ledger.bind_batch(indexes)
    for row in indexes:
        for array, stop in (
            ("rgb", probe.HISTORY_RGB_FRAMES),
            ("actions", probe.TOTAL_RGB_FRAMES),
        ):
            started = ledger.begin_array_read(
                role="serving",
                array=array,
                row=row,
                frame_start=0,
                frame_stop=stop,
                path=f"/{array}.npy",
            )
            ledger.finish_array_read(started)


def _record_scoring_accesses(
    ledger: probe.EventLedger, indexes=(416, 417)
) -> None:
    for row in indexes:
        started = ledger.begin_array_read(
            role="scoring",
            array="rgb",
            row=row,
            frame_start=0,
            frame_stop=probe.TOTAL_RGB_FRAMES,
            path="/rgb.npy",
        )
        ledger.finish_array_read(started)


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
    _record_serving_accesses(ledger)
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

    _record_scoring_accesses(ledger)
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


def test_event_ledger_forbids_future_rgb_before_endpoint_barrier():
    ledger = probe.EventLedger()
    ledger.bind_batch((416, 417))
    with pytest.raises(probe.MultirateError):
        ledger.begin_array_read(
            role="serving",
            array="rgb",
            row=416,
            frame_start=0,
            frame_stop=13,
            path="/rgb.npy",
        )
    with pytest.raises(probe.MultirateError):
        ledger.begin_array_read(
            role="scoring",
            array="rgb",
            row=416,
            frame_start=0,
            frame_stop=13,
            path="/rgb.npy",
        )


def test_serving_type_has_no_full_rgb_field_and_rejects_thirteen_frames():
    batch = probe.ServingBatch(
        history_rgb=torch.zeros((2, 13, 3, 2, 2)),
        actions=torch.zeros((2, 13, 5, probe.PADDED_ACTION_DIM)),
        morphology_index=torch.full((2,), probe.ABC_MORPHOLOGY_INDEX),
        clip_index=torch.tensor([416, 417]),
    )
    assert not hasattr(batch, "rgb")
    with pytest.raises(probe.MultirateError, match="exactly the five"):
        probe._prepare_serving_inputs(
            SimpleNamespace(num_history_frames=5),
            batch,
            noise_seed=probe.DEV_NOISE_SEEDS[0],
            ledger=probe.EventLedger(),
        )


def test_audited_reader_uses_exact_memmap_slice_keys_and_barrier_order():
    class SliceSpy:
        def __init__(self):
            self.keys = []

        def __getitem__(self, key):
            self.keys.append(key)
            return np.zeros((1,), dtype=np.float16)

    reader = object.__new__(probe.AuditedABCClipReader)
    reader.allowed = frozenset(range(*probe.DEV_RANGE))
    reader.rgb_path = Path("/immutable/rgb.npy")
    reader.actions_path = Path("/immutable/actions.npy")
    reader._rgbs = SliceSpy()
    reader._actions = SliceSpy()
    ledger = probe.EventLedger()
    ledger.bind_batch((416, 417))
    reader._read_slice(
        ledger=ledger,
        role="serving",
        array="rgb",
        row=416,
        frame_start=0,
        frame_stop=5,
    )
    assert reader._rgbs.keys == [(416, slice(0, 5), Ellipsis)]
    for endpoint in probe.ENDPOINTS:
        ledger.endpoint(endpoint)
    ledger.close_endpoint_barrier()
    reader._read_slice(
        ledger=ledger,
        role="scoring",
        array="rgb",
        row=416,
        frame_start=0,
        frame_stop=13,
    )
    assert reader._rgbs.keys[-1] == (416, slice(0, 13), Ellipsis)


def test_public_deployable_capture_is_target_blind_and_restores_configuration():
    class Tokenizer:
        def decode_temporal(self, latent, out_hw=None):
            del out_hw
            value = latent.mean(dim=1, keepdim=True).repeat(1, 3, 8, 1, 1)
            return value

    class FakeModel:
        def __init__(self, initial, reference):
            self.initial = initial
            self.reference = reference
            self.forward_model = torch.nn.Identity()
            self.rgb_tokenizer = Tokenizer()
            self.evaluation_condition_sources = ("autonomous",)
            self.evaluation_nfe_steps = (4,)
            self.viz_num_steps = 4
            self.evaluation_noise_seed = 99
            self.capture_latent_trajectories = True
            self.artifact_batch_limit = 1
            self._last_sampling_counters = {"sentinel": 1}
            self._visualization_artifacts = {"sentinel": torch.tensor(1)}

        def sample_future_deployable(
            self,
            history_rgb,
            actions,
            morphology_index,
            *,
            collect_artifacts,
            sample_ids,
        ):
            del history_rgb, actions, morphology_index
            assert collect_artifacts is True
            nfe = self.viz_num_steps
            state = self.initial.clone()
            for _ in range(nfe):
                state = self.forward_model(state) + 0.125
            decoded = self.rgb_tokenizer.decode_temporal(state, out_hw=(2, 2))
            decoded_uint8 = ladder._to_uint8_video(decoded)
            self._visualization_artifacts = {
                "video_initial_state": self.initial.cpu().to(torch.float16),
                "reference_latents": self.reference.cpu().to(torch.float16),
                "sample_ids": sample_ids.cpu().to(torch.int64),
                "deployment_mode": torch.tensor([1]),
                "auxiliary_clean_available": torch.tensor([0]),
                f"video_final_off_nfe_{nfe}": state.cpu().to(torch.float16),
                f"decoded_future_off_nfe_{nfe}": decoded_uint8.cpu(),
            }
            self._last_sampling_counters = {
                "wan_calls_by_source_nfe": {f"off:nfe_{nfe}": nfe},
                "wan_calls_total": nfe,
                "online_teacher_calls": 0,
                "auxiliary_clean_available": 0,
                "artifacts_collected": 1,
                "deployment_mode": 1,
            }
            return decoded

        def pop_visualization_artifacts(self):
            value = self._visualization_artifacts
            self._visualization_artifacts = None
            return value

    initial = torch.zeros((2, 1, 1, 2, 2))
    reference = torch.ones_like(initial)
    model = FakeModel(initial, reference)
    batch = probe.ServingBatch(
        history_rgb=torch.zeros((2, 5, 3, 2, 2)),
        actions=torch.zeros((2, 13, 5, 157)),
        morphology_index=torch.tensor([9, 9]),
        clip_index=torch.tensor([416, 417]),
    )
    ledger = probe.EventLedger()
    _record_serving_accesses(ledger)
    saved = (
        model.evaluation_condition_sources,
        model.evaluation_nfe_steps,
        model.viz_num_steps,
        model.evaluation_noise_seed,
        model.capture_latent_trajectories,
        model.artifact_batch_limit,
    )
    capture = probe._capture_public_deployable(
        model,
        batch,
        {"initial_video": initial, "reference": reference},
        nfe=2,
        noise_seed=probe.DEV_NOISE_SEEDS[0],
        ledger=ledger,
    )
    assert capture.receipt["wan_calls"] == 2
    assert capture.receipt["official_artifact_final_matches_exact_capture"] is True
    assert not any(
        event.get("array") == "rgb" and event.get("frame_stop", 0) > 5
        for event in ledger.events
    )
    assert saved == (
        model.evaluation_condition_sources,
        model.evaluation_nfe_steps,
        model.viz_num_steps,
        model.evaluation_noise_seed,
        model.capture_latent_trajectories,
        model.artifact_batch_limit,
    )


def _path_guard() -> Path:
    return (
        Path(probe.__file__).resolve().parent
        / "slurm"
        / "vpm_invertible_multirate_paths.sh"
    )


def test_launcher_path_guard_creates_only_missing_parent_then_redirects(tmp_path):
    grandparent = tmp_path / "artifacts" / "dual_video_diffusion"
    grandparent.mkdir(parents=True)
    parent = grandparent / "vpm_invertible_multirate_probe"
    output = parent / "run-v2"
    runtime = Path(f"{output}.runtime_verification.json")
    prefix = f"{parent}/"
    command = """
source "$1"
prepare_vpm_invertible_multirate_artifact_parent "$2" "$3" "$4"
(set -o noclobber; printf '%s\n' '{"runtime":"ok"}' > "$3")
"""
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(_path_guard()), str(output), str(runtime), prefix],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert parent.is_dir() and not parent.is_symlink()
    assert not output.exists()
    assert runtime.read_text(encoding="utf-8") == '{"runtime":"ok"}\n'
    assert sorted(path.name for path in parent.iterdir()) == [runtime.name]


@pytest.mark.parametrize("existing", ("output", "runtime"))
def test_launcher_path_guard_refuses_existing_run_paths_without_mutation(
    tmp_path, existing
):
    grandparent = tmp_path / "artifacts" / "dual_video_diffusion"
    parent = grandparent / "vpm_invertible_multirate_probe"
    parent.mkdir(parents=True)
    output = parent / "run-v2"
    runtime = Path(f"{output}.runtime_verification.json")
    protected = output if existing == "output" else runtime
    if existing == "output":
        protected.mkdir()
        marker = protected / "marker"
    else:
        marker = protected
    marker.write_text("do-not-overwrite", encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; prepare_vpm_invertible_multirate_artifact_parent "$2" "$3" "$4"',
            "bash",
            str(_path_guard()),
            str(output),
            str(runtime),
            f"{parent}/",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert marker.read_text(encoding="utf-8") == "do-not-overwrite"
    if existing == "output":
        assert not runtime.exists()
    else:
        assert not output.exists()


def test_full_launcher_uses_guard_before_runtime_redirection_and_requires_v2():
    launcher = (
        Path(probe.__file__).resolve().parent
        / "slurm"
        / "vpm_invertible_multirate_probe.sbatch"
    ).read_text(encoding="utf-8")
    guard = "prepare_vpm_invertible_multirate_artifact_parent"
    redirect = '> "$RUNTIME_RECORD"'
    assert guard in launcher
    assert launcher.index(guard) < launcher.index(redirect)
    assert "${EXPECTED_COMMIT:0:7}-v2" in launcher
    assert 'mkdir -p "$OUTPUT_DIR"' not in launcher


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
                                "serving_rgb_frame_slice": [0, 5],
                                "full_rgb_materialized_only_after_endpoint_barrier": True,
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
