import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects/latent_action_models"
for path in (str(ROOT), str(PROJECT), str(ROOT / "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

from robot_wm.modeling.dual_diffusion.spectral_consistency import (  # noqa: E402
    spatiotemporal_spectral_consistency,
)
import tf_training_only_screen as contract  # noqa: E402


def _signal(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(2, 3, 2, 9, 16, generator=generator)


def test_spectral_consistency_is_zero_for_identical_signal():
    target = _signal()
    terms = spatiotemporal_spectral_consistency(target, target)
    assert terms.selected_bins > 1
    assert terms.total.item() == pytest.approx(0.0, abs=1.0e-7)
    assert terms.log_amplitude.item() == pytest.approx(0.0, abs=1.0e-7)
    assert terms.phase.item() == pytest.approx(0.0, abs=1.0e-7)


def test_amplitude_and_phase_coordinates_are_separated():
    target = _signal(4)
    scaled = spatiotemporal_spectral_consistency(1.7 * target, target)
    shifted = spatiotemporal_spectral_consistency(
        torch.roll(target, shifts=1, dims=-1), target
    )
    assert scaled.log_amplitude.item() > 0.01
    assert scaled.phase.item() < 1.0e-5
    # A circular spatial shift preserves every Fourier magnitude but rotates phase.
    assert shifted.log_amplitude.item() < 1.0e-6
    assert shifted.phase.item() > 0.01


def test_masked_region_has_exactly_zero_gradient():
    target = _signal(7)[:1]
    predicted = (target + 0.1 * _signal(8)[:1]).requires_grad_(True)
    mask = torch.ones(1, 1, 2, 1, 16)
    mask[..., 8:] = 0
    loss = spatiotemporal_spectral_consistency(
        predicted, target, validity_mask=mask
    ).total
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[..., :8].abs().sum().item() > 0
    assert torch.count_nonzero(predicted.grad[..., 8:]).item() == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spatial_band_fraction": 0.0},
        {"spatial_band_fraction": 1.1},
        {"phase_weight": -1.0},
        {"epsilon": 0.0},
    ],
)
def test_spectral_contract_rejects_invalid_hyperparameters(kwargs):
    target = _signal()
    with pytest.raises(ValueError):
        spatiotemporal_spectral_consistency(target, target, **kwargs)


def test_deployable_sampler_source_has_no_spectral_or_future_argument():
    # Import lazily because the full model depends on the production VideoX stack.
    source = (PROJECT / "lam/tf_training_only_model.py").read_text(encoding="utf-8")
    sampler = source.split("    def sample_future_deployable(", 1)[1]
    sampler = sampler.split("\n***", 1)[0] if "\n***" in sampler else sampler
    assert "spatiotemporal_spectral_consistency" not in sampler
    assert "auxiliary_target" not in sampler
    assert "future_rgb" not in sampler
    assert "history_rgb" in sampler
    assert "actions" in sampler


def _valid_rows(arm):
    rows = []
    for clip in range(contract.VALIDATION_CLIPS):
        for control in contract.ACTION_CONTROLS:
            rows.append(
                {
                    "schema": contract.RESULT_SCHEMA,
                    "arm": arm.code,
                    "clip_index": clip,
                    "action_control": control,
                    "action_donor_clip_index": (
                        clip
                        if control == "aligned"
                        else (clip + 1) % contract.VALIDATION_CLIPS
                        if control == "episode_shuffled"
                        else None
                    ),
                    "action_tensor_shape": list(contract.ACTION_SAMPLE_SHAPE),
                    "action_tensor_dtype": "torch.float32",
                    "action_tensor_sha256": (
                        contract.ZERO_ACTION_SHA256 if control == "zero" else control
                    ),
                    "nfe": 1,
                    "wan_calls": 1,
                    "history_rgb_frames_received": 5,
                    "future_rgb_frames_received": 0,
                    "spectral_loss_calls_at_inference": 0,
                    "auxiliary_inputs_at_inference": 0,
                    "auxiliary_modules_at_inference": 0,
                    "online_teacher_calls": 0,
                    "cached_vjepa_target_opened": False,
                    "protected_test_accessed": False,
                    "noise_seed": contract.EVALUATION_NOISE_SEED + clip,
                    "initial_noise_sha256": f"noise-{clip}",
                    "score_target_sha256": f"target-{clip}",
                    "latent_nmse": 1.0,
                    "decoded_mse": 1.0,
                    "temporal_mse": 1.0,
                }
            )
    return rows


def test_result_grid_requires_every_clip_and_action_control_once():
    arm = contract.ARM_BY_CODE["TFREG-ON"]
    rows = _valid_rows(arm)
    contract.validate_result_rows(rows, arm)
    rows[-1] = dict(rows[0])
    with pytest.raises(contract.TFREGContractError):
        contract.validate_result_rows(rows, arm)


def test_paired_bootstrap_reports_known_improvement():
    off = [1.0] * contract.VALIDATION_CLIPS
    on = [0.9] * contract.VALIDATION_CLIPS
    point, interval = contract._bootstrap_improvement(off, on, draws=256)
    assert point == pytest.approx(10.0)
    assert interval == pytest.approx([10.0, 10.0])


def test_protocol_has_one_parameter_matched_intervention():
    assert [arm.loss_weight for arm in contract.ARMS] == [0.0, 0.05]
    assert len({arm.run_name for arm in contract.ARMS}) == 2
    assert contract.ACTION_CONTROLS == ("aligned", "episode_shuffled", "zero")
    assert contract.TRAIN_UPDATES == 200
    assert contract.TRAIN_CLIPS == 512
    assert contract.VALIDATION_CLIPS == 64
