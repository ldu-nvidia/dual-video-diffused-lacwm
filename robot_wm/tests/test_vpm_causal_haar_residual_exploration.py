import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "vpm_causal_haar_residual_exploration.py"
)
SPEC = importlib.util.spec_from_file_location("vpm_causal_haar_residual_exploration", SCRIPT)
haar = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(haar)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def test_view_isolated_three_band_haar_is_orthogonal_and_reconstructs():
    generator = torch.Generator().manual_seed(17)
    value = torch.randn((3, *haar.LATENT_SHAPE), generator=generator)
    bands = haar.haar_decompose(value, history_frames=haar.HISTORY_LATENT_FRAMES)
    receipt = haar.haar_contract(
        value, bands, history_frames=haar.HISTORY_LATENT_FRAMES
    )
    assert receipt["max_abs_reconstruction_error"] <= haar.HAAR_MAX_ABS_TOLERANCE
    assert receipt["relative_reconstruction_energy"] <= haar.HAAR_RELATIVE_ENERGY_TOLERANCE
    assert max(receipt["normalized_pairwise_inner_products"].values()) <= haar.HAAR_ORTHOGONALITY_TOLERANCE
    assert sum(receipt["band_energy_fractions"].values()) == pytest.approx(1.0, abs=5e-6)
    assert all(int(torch.count_nonzero(band[:, :, :2])) == 0 for band in bands.values())

    changed = value.clone()
    changed[..., : haar.VIEW_WIDTH] += 123
    changed_bands = haar.haar_decompose(changed, history_frames=2)
    for name in haar.BANDS:
        assert torch.equal(
            bands[name][..., haar.VIEW_WIDTH :],
            changed_bands[name][..., haar.VIEW_WIDTH :],
        )


def test_each_named_projection_is_idempotent_and_other_bands_are_zero():
    value = torch.randn((1, *haar.LATENT_SHAPE), generator=torch.Generator().manual_seed(2))
    bands = haar.haar_decompose(value, history_frames=2)
    for name, band in bands.items():
        projected = haar.haar_decompose(band, history_frames=2)
        assert torch.allclose(projected[name], band, atol=2e-6, rtol=0)
        for other in set(haar.BANDS) - {name}:
            assert projected[other].abs().max().item() <= 2e-6


def test_manifest_contract_bars_reserve_and_uses_new_disjoint_seeds():
    rows = [
        {
            "split": "train",
            "auxiliary_index": index,
            "clip_id": f"clip-{index}",
            "episode_dir": f"episode-{index}",
        }
        for index in range(512)
    ]
    result = haar._validate_manifest(rows)
    assert result["ranges"]["nested_fit"] == [128, 384]
    assert result["ranges"]["prior_inspected_exploratory_development"] == [416, 480]
    assert result["ranges"]["forbidden_fresh_reserve"] == [480, 511]
    assert result["ranges"]["excluded_constructor_probe"] == [511, 512]
    assert min(result["development_noise_seeds"]) > 20260841


def test_index_audit_fails_before_reserve_or_constructor_probe_access():
    class Dataset:
        def __len__(self):
            return 512

        def __getitem__(self, index):
            return index

    wrapped = haar._IndexAuditedDataset(Dataset(), range(*haar.DEV_RANGE))
    assert wrapped[416] == 416
    with pytest.raises(haar.HaarError, match="forbidden row 480"):
        wrapped[480]
    with pytest.raises(haar.HaarError, match="forbidden row 511"):
        wrapped[511]


def _registered_samples():
    return [
        {
            "clip_index": clip,
            "clip_id": f"clip-{clip}",
            "episode_dir": f"episode-{clip}",
        }
        for clip in range(*haar.DEV_RANGE)
    ]


def _row(endpoint: str, clip: int, seed: int, error: float):
    common = f"{clip}-{seed}"
    off_or_zero = endpoint in {"VPM_OFF", "ZERO"}
    aligned = endpoint.endswith("_ALIGNED")
    return haar.ladder.identity_payload(
        {
            "schema": haar.ROW_SCHEMA,
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
                **{metric: error for metric in haar.ERROR_METRICS},
                "target_r2": 0.25 if aligned else 0.0,
                "target_cosine": 0.5 if aligned else 0.0,
                "target_energy": 1.0,
                "prediction_energy": 0.2 if aligned else 0.0,
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
                "correction": _digest(f"{'zero' if off_or_zero else endpoint}-correction-{common}"),
                "final_video": _digest(f"{'off' if off_or_zero else endpoint}-final-{common}"),
            },
        }
    )


def _inventory(special: bool):
    rows = []
    for clip in range(*haar.DEV_RANGE):
        for seed in haar.DEV_NOISE_SEEDS:
            for endpoint in haar.ENDPOINTS:
                if endpoint in {"VPM_OFF", "ZERO"}:
                    error = 1.0
                elif "ST_COARSE_ALIGNED" in endpoint:
                    error = 0.84 if special else 0.995
                elif "_ALIGNED" in endpoint:
                    error = 0.98
                else:
                    error = 1.02
                rows.append(_row(endpoint, clip, seed, error))
    return rows


def _haar_receipt():
    return {
        "checked_batches": 128,
        "max_abs_reconstruction_error": 1e-7,
        "max_relative_reconstruction_energy": 1e-15,
        "max_abs_normalized_pairwise_inner_product": 1e-8,
        "mean_band_energy_fraction": {
            "ST_COARSE": 0.5,
            "TEMPORAL_CHANGE": 0.1,
            "SPATIOTEMPORAL_DETAIL": 0.4,
        },
        "passed": True,
    }


def test_special_decision_requires_better_quality_than_full_direct(monkeypatch):
    monkeypatch.setattr(haar, "BOOTSTRAP_REPLICATES", 100)
    result = haar.analyze_rows(
        _inventory(special=True),
        registered_samples=_registered_samples(),
        haar_contract_receipt=_haar_receipt(),
    )
    assert result["decision"] == "EXPLORATORY_HAAR_SPECIAL"
    assert result["band_gates"]["ST_COARSE"]["any_dose_passed"] is True
    assert result["aggregates"]["D256_ST_COARSE_ALIGNED"]["target_r2"]["mean"] > 0


def test_positive_r2_without_quality_or_sample_efficiency_is_not_special(monkeypatch):
    monkeypatch.setattr(haar, "BOOTSTRAP_REPLICATES", 100)
    result = haar.analyze_rows(
        _inventory(special=False),
        registered_samples=_registered_samples(),
        haar_contract_receipt=_haar_receipt(),
    )
    assert result["decision"] == "EXPLORATORY_NO_HAAR_ADVANTAGE"
    assert result["aggregates"]["D256_ST_COARSE_ALIGNED"]["target_r2"]["mean"] > 0


def test_protocol_freezes_exact_bands_and_claim_boundary():
    protocol = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "experiments"
        / "VPM_CAUSAL_HAAR_RESIDUAL_EXPLORATION_PROTOCOL.md"
    ).read_text()
    for value in (
        "`ST_COARSE`",
        "`TEMPORAL_CHANGE`",
        "`SPATIOTEMPORAL_DETAIL`",
        "416--479",
        "480--510",
        "post-selection exploratory",
        "Predictability alone is insufficient",
    ):
        assert value in protocol
