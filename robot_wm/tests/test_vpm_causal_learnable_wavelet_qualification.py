import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "vpm_causal_learnable_wavelet_qualification.py"
)
SPEC = importlib.util.spec_from_file_location(
    "vpm_causal_learnable_wavelet_qualification", SCRIPT
)
wavelet = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(wavelet)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _learned_filter():
    return wavelet.lattice_lowpass(torch.tensor([0.2, -0.1, 0.3]))


def test_paraunitary_lattice_is_admissible_nonattenuating_and_reconstructs():
    lowpass = _learned_filter()
    receipt = wavelet.basis_admissibility(lowpass)
    assert receipt["passed"] is True
    assert receipt["max_admissibility_residual"] <= wavelet.BASIS_ADMISSIBILITY_TOLERANCE
    assert (
        receipt["max_analysis_matrix_orthogonality_error"]
        <= wavelet.BASIS_MATRIX_ORTHOGONALITY_TOLERANCE
    )
    assert receipt["max_abs_reconstruction_error"] <= wavelet.BASIS_RECONSTRUCTION_TOLERANCE
    assert receipt["max_coefficient_input_energy_ratio_error"] <= wavelet.BASIS_ENERGY_RATIO_TOLERANCE
    assert receipt["threshold_gates_enabled"] is False
    assert receipt["terminal_band_gate_retention"] == [1.0] * 4
    assert abs(torch.dot(lowpass, wavelet.haar_lowpass()).item()) < 0.9999


def test_zero_padded_haar_uses_same_phase_and_matches_block_projection():
    value = torch.randn((2, 3, 2, 24, 40), generator=torch.Generator().manual_seed(5))
    observed = wavelet._spatial_lowpass_view(value, wavelet.haar_lowpass())
    blocks = value.reshape(2, 3, 2, 12, 2, 20, 2)
    expected = blocks.mean(dim=(4, 6), keepdim=True).expand_as(blocks).reshape_as(value)
    assert torch.allclose(observed, expected, atol=2e-6, rtol=0)
    assert wavelet.analysis_matrix(wavelet.haar_lowpass(), 24).shape == (12, 24)
    assert wavelet.analysis_matrix(_learned_filter(), 24).shape == (12, 24)


@pytest.mark.parametrize("basis", ("HAAR", "QMF"))
def test_three_band_partition_is_view_isolated_orthogonal_and_history_masked(basis):
    value = torch.randn(
        (3, *wavelet.LATENT_SHAPE), generator=torch.Generator().manual_seed(17)
    )
    if basis == "HAAR":
        bands = wavelet.haar_decompose(value, history_frames=2)
    else:
        bands = wavelet.qmf_decompose(value, lowpass=_learned_filter(), history_frames=2)
    receipt = wavelet.partition_contract(value, bands, history_frames=2)
    assert receipt["max_abs_reconstruction_error"] <= wavelet.HAAR_MAX_ABS_TOLERANCE
    assert receipt["relative_reconstruction_energy"] <= wavelet.HAAR_RELATIVE_ENERGY_TOLERANCE
    assert max(receipt["normalized_pairwise_inner_products"].values()) <= wavelet.HAAR_ORTHOGONALITY_TOLERANCE
    assert sum(receipt["band_energy_fractions"].values()) == pytest.approx(1.0, abs=5e-6)
    assert all(int(torch.count_nonzero(band[:, :, :2])) == 0 for band in bands.values())

    changed = value.clone()
    changed[..., : wavelet.VIEW_WIDTH] += 123
    if basis == "HAAR":
        changed_bands = wavelet.haar_decompose(changed, history_frames=2)
    else:
        changed_bands = wavelet.qmf_decompose(changed, lowpass=_learned_filter(), history_frames=2)
    for name in wavelet.BASE_BANDS:
        assert torch.equal(
            bands[name][..., wavelet.VIEW_WIDTH :],
            changed_bands[name][..., wavelet.VIEW_WIDTH :],
        )


def test_temporal_haar_endpoint_and_sign_contracts():
    constant = torch.zeros((1, *wavelet.LATENT_SHAPE))
    constant[:, :, 2:] = 1.0
    bands = wavelet.qmf_decompose(constant, lowpass=_learned_filter(), history_frames=2)
    assert bands["TEMPORAL_CHANGE"].abs().max().item() <= 2e-6
    assert bands["SPATIOTEMPORAL_DETAIL"].abs().max().item() <= 2e-6
    assert torch.allclose(bands["ST_COARSE"][:, :, 2:], constant[:, :, 2:], atol=2e-6)

    antisymmetric = torch.zeros_like(constant)
    antisymmetric[:, :, 2] = 1.0
    antisymmetric[:, :, 3] = -1.0
    bands = wavelet.qmf_decompose(antisymmetric, lowpass=_learned_filter(), history_frames=2)
    assert bands["ST_COARSE"].abs().max().item() <= 2e-6
    assert bands["SPATIOTEMPORAL_DETAIL"].abs().max().item() <= 2e-6
    assert torch.allclose(
        bands["TEMPORAL_CHANGE"][:, :, 2:], antisymmetric[:, :, 2:], atol=2e-6
    )


def test_each_named_projection_is_idempotent_and_other_bands_are_zero():
    value = torch.randn(
        (1, *wavelet.LATENT_SHAPE), generator=torch.Generator().manual_seed(2)
    )
    for decompose in (
        lambda tensor: wavelet.haar_decompose(tensor, history_frames=2),
        lambda tensor: wavelet.qmf_decompose(
            tensor, lowpass=_learned_filter(), history_frames=2
        ),
    ):
        bands = decompose(value)
        for name, band in bands.items():
            projected = decompose(band)
            assert torch.allclose(projected[name], band, atol=2e-6, rtol=0)
            for other in set(wavelet.BASE_BANDS) - {name}:
                assert projected[other].abs().max().item() <= 2e-6


def test_manifest_contract_bars_reserve_and_uses_disjoint_seeds():
    rows = [
        {
            "split": "train",
            "auxiliary_index": index,
            "clip_id": f"clip-{index}",
            "episode_dir": f"episode-{index}",
        }
        for index in range(512)
    ]
    result = wavelet._validate_manifest(rows)
    assert result["ranges"]["nested_fit"] == [128, 384]
    assert result["ranges"]["prior_inspected_exploratory_development"] == [416, 480]
    assert result["ranges"]["forbidden_fresh_reserve"] == [480, 511]
    assert result["ranges"]["excluded_constructor_probe"] == [511, 512]
    assert result["basis_seed"] == 20261001
    assert min(result["development_noise_seeds"]) > 20260919


def test_index_audit_fails_before_reserve_or_constructor_probe_access():
    class Dataset:
        def __len__(self):
            return 512

        def __getitem__(self, index):
            return index

    wrapped = wavelet._IndexAuditedDataset(Dataset(), range(*wavelet.DEV_RANGE))
    assert wrapped[416] == 416
    with pytest.raises(wavelet.WaveletError, match="forbidden row 480"):
        wrapped[480]
    with pytest.raises(wavelet.WaveletError, match="forbidden row 511"):
        wrapped[511]


def test_zero_head_audit_uses_sealed_ridge_intercept_schema():
    stats = {
        "count": 4,
        "sum_x": torch.zeros(2),
        "sum_xx": torch.eye(2) * 4,
        "sum_y": {"ZERO": torch.zeros(3)},
        "sum_xy": {"ZERO": torch.zeros(2, 3)},
        "sum_y2": {"ZERO": torch.zeros((), dtype=torch.float64)},
    }
    old_arms = wavelet.ladder.ARMS
    old_rungs = wavelet.ladder.CAPACITY_RUNGS
    old_lambdas = wavelet.ladder.RIDGE_LAMBDAS
    try:
        wavelet.ladder.ARMS = ("ZERO",)
        wavelet.ladder.CAPACITY_RUNGS = (2,)
        wavelet.ladder.RIDGE_LAMBDAS = (1.0,)
        head = wavelet.ladder.fit_ridge_head(
            stats, arm="ZERO", capacity=2, ridge_lambda=1.0
        )
    finally:
        wavelet.ladder.ARMS = old_arms
        wavelet.ladder.CAPACITY_RUNGS = old_rungs
        wavelet.ladder.RIDGE_LAMBDAS = old_lambdas
    assert set(("weight", "mean_y")).issubset(head)
    assert "bias" not in head
    assert all(int(torch.count_nonzero(head[name])) == 0 for name in ("weight", "mean_y"))


def _registered_samples():
    return [
        {
            "clip_index": clip,
            "clip_id": f"clip-{clip}",
            "episode_dir": f"episode-{clip}",
        }
        for clip in range(*wavelet.DEV_RANGE)
    ]


def _row(endpoint: str, clip: int, seed: int, error: float):
    common = f"{clip}-{seed}"
    off_or_zero = endpoint in {"VPM_OFF", "ZERO"}
    aligned = endpoint.endswith("_ALIGNED")
    return wavelet.ladder.identity_payload(
        {
            "schema": wavelet.ROW_SCHEMA,
            "endpoint": endpoint,
            "clip_index": clip,
            "clip_id": f"clip-{clip}",
            "episode_dir": f"episode-{clip}",
            "noise_seed": seed,
            "wan_calls": 1,
            "teacher_calls": 0,
            "vjepa_target_array_opened": False,
            "future_target_entered_correction": False,
            "scoring_target_constructed_after_all_endpoints": True,
            "post_selection_exploratory_dev_opened": True,
            "fresh_reserve_480_510_opened": False,
            "constructor_probe_511_opened": False,
            "validation_opened": False,
            "protected_test_opened": False,
            "metrics": {
                **{metric: error for metric in wavelet.ERROR_METRICS},
                "target_r2": 0.25 if aligned else 0.0,
                "target_cosine": 0.5 if aligned else 0.0,
                "target_energy": 1.0,
                "target_rms": 1.0,
                "prediction_energy": 0.2 if aligned else 0.0,
                "correction_rms": 0.2**0.5 if aligned else 0.0,
                "preprojection_offband_relative_energy": 0.1 if "ST_COARSE" in endpoint else 0.0,
                "postprojection_offband_relative_energy": 0.0,
            },
            "tensor_sha256": {
                **{
                    name: _digest(f"{name}-{common}")
                    for name in (
                        "initial_video",
                        "actions",
                        "morphology_index",
                        "action_control",
                        "auxiliary_noise",
                        "raw_history_input",
                        "history_reference",
                        "latent_flow_target",
                        "raw_future_target",
                        "raw_history_boundary",
                        "scoring_future_uint8",
                        "scoring_history_uint8",
                    )
                },
                "correction": _digest(
                    f"{'zero' if off_or_zero else endpoint}-correction-{common}"
                ),
                "final_video": _digest(
                    f"{'off' if off_or_zero else endpoint}-final-{common}"
                ),
            },
        }
    )


def _inventory(special: bool):
    rows = []
    for clip in range(*wavelet.DEV_RANGE):
        for seed in wavelet.DEV_NOISE_SEEDS:
            for endpoint in wavelet.ENDPOINTS:
                if endpoint in {"VPM_OFF", "ZERO"}:
                    error = 1.0 + 0.001 * (clip - wavelet.DEV_RANGE[0])
                elif "QMF_ST_COARSE_ALIGNED" in endpoint:
                    error = 0.78 if special else 0.995
                elif "QMF_ST_COARSE_SHUFFLED" in endpoint:
                    error = 0.96
                elif "HAAR_ST_COARSE_ALIGNED" in endpoint:
                    error = 0.91
                elif "FULL_DIRECT_ALIGNED" in endpoint:
                    error = 0.90
                elif endpoint.endswith("_ALIGNED"):
                    error = 0.98
                else:
                    error = 1.02
                rows.append(_row(endpoint, clip, seed, error))
    return rows


def _partition_receipts():
    receipt = {
        "checked_batches": 128,
        "max_abs_reconstruction_error": 1e-7,
        "max_relative_reconstruction_energy": 1e-15,
        "max_abs_normalized_pairwise_inner_product": 1e-8,
        "mean_band_energy_fraction": {
            "ST_COARSE": 0.2,
            "TEMPORAL_CHANGE": 0.1,
            "SPATIOTEMPORAL_DETAIL": 0.7,
        },
        "nontrivial_grouped_band_energy": True,
        "passed": True,
    }
    return {"HAAR": dict(receipt), "QMF": dict(receipt)}


def _basis_qualification(passed=True):
    result = {
        "admissibility_reconstruction_energy_pass": passed,
        "nontrivial_filter_change_pass": passed,
        "heldout_normalized_l1_sparsity_pass": passed,
        "nontrivial_terminal_energy_pass": passed,
        "threshold_retention_pass": True,
        "basis_frozen_before_any_residual_target": True,
    }
    result["all_passed"] = all(result.values())
    return result


def test_special_decision_requires_qmf_to_beat_shuffled_off_haar_and_full(monkeypatch):
    monkeypatch.setattr(wavelet, "BOOTSTRAP_REPLICATES", 100)
    result = wavelet.analyze_rows(
        _inventory(special=True),
        registered_samples=_registered_samples(),
        partition_contract_receipts=_partition_receipts(),
        basis_qualification=_basis_qualification(),
    )
    assert result["decision"] == "EXPLORATORY_LEARNED_QMF_SPECIAL"
    assert result["band_gates"]["QMF_ST_COARSE"]["any_dose_passed"] is True
    assert result["band_gates"]["HAAR_ST_COARSE"]["any_dose_passed"] is False
    assert len(result["off_error_quartile_heterogeneity_not_for_selection"][wavelet.ERROR_METRICS[0]]) == 4


def test_positive_r2_without_quality_is_no_causal_advantage(monkeypatch):
    monkeypatch.setattr(wavelet, "BOOTSTRAP_REPLICATES", 100)
    result = wavelet.analyze_rows(
        _inventory(special=False),
        registered_samples=_registered_samples(),
        partition_contract_receipts=_partition_receipts(),
        basis_qualification=_basis_qualification(),
    )
    assert result["decision"] == "EXPLORATORY_BASIS_ADAPTS_BUT_NO_CAUSAL_ADVANTAGE"
    assert result["aggregates"]["D256_QMF_ST_COARSE_ALIGNED"]["target_r2"]["mean"] > 0


def test_failed_unsupervised_basis_cannot_pass_even_with_quality(monkeypatch):
    monkeypatch.setattr(wavelet, "BOOTSTRAP_REPLICATES", 100)
    result = wavelet.analyze_rows(
        _inventory(special=True),
        registered_samples=_registered_samples(),
        partition_contract_receipts=_partition_receipts(),
        basis_qualification=_basis_qualification(False),
    )
    assert result["decision"] == "EXPLORATORY_NO_LEARNED_QMF_QUALIFICATION"
    assert result["global_gates"]["basis_qualification_pass"] is False


def test_protocol_freezes_primary_source_scope_and_claim_boundary():
    protocol = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "experiments"
        / "VPM_CAUSAL_LEARNABLE_WAVELET_QUALIFICATION_PROTOCOL.md"
    ).read_text()
    for value in (
        wavelet.FREQUENCY_FORCING_SOURCE_SHA256,
        "400 epochs",
        "two clean future latent frames",
        "100%",
        "416--479",
        "480--510",
        "post-selection exploratory",
        "not** a Frequency-Forcing reproduction",
    ):
        assert value in protocol
