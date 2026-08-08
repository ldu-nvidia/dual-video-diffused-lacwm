import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects/latent_action_models"
for path in (str(ROOT), str(PROJECT), str(ROOT / "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

from robot_wm.modeling.dual_diffusion.low_frequency_motion import (  # noqa: E402
    _gaussian_kernel,
    _per_view_low_pass,
    low_frequency_motion_consistency,
)
import motion_band_training_only_screen as contract  # noqa: E402


def _trajectory(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    anchor = torch.randn(2, 3, 1, 9, 18, generator=generator)
    future = torch.randn(2, 3, 2, 9, 18, generator=generator)
    return anchor, future


def test_identical_trajectory_has_zero_loss_and_two_transitions_per_view():
    anchor, future = _trajectory()
    terms = low_frequency_motion_consistency(future, future, anchor)
    assert terms.total.item() == pytest.approx(0.0, abs=1.0e-7)
    assert terms.anchor_transition.item() == pytest.approx(0.0, abs=1.0e-7)
    assert terms.future_transition.item() == pytest.approx(0.0, abs=1.0e-7)
    assert terms.valid_transition_count == 2 * 3 * 2


def test_observed_anchor_turns_two_future_tokens_into_two_supervised_deltas():
    anchor = torch.zeros(1, 1, 1, 9, 18)
    target = torch.stack(
        (torch.ones(1, 1, 9, 18), 2.0 * torch.ones(1, 1, 9, 18)), dim=2
    )
    predicted = torch.stack(
        (torch.zeros(1, 1, 9, 18), 2.0 * torch.ones(1, 1, 9, 18)), dim=2
    )
    terms = low_frequency_motion_consistency(predicted, target, anchor)
    assert terms.anchor_transition.item() > 0.1
    assert terms.future_transition.item() > 0.1


def test_gaussian_filter_never_crosses_a_camera_seam():
    signal = torch.zeros(1, 1, 1, 9, 18)
    signal[..., 5] = 1.0  # last column of view zero
    kernel = _gaussian_kernel(5, 1.0, signal.device)
    filtered = _per_view_low_pass(signal, num_views=3, kernel=kernel)
    assert filtered[:, 0].abs().sum().item() > 0
    assert torch.count_nonzero(filtered[:, 1:]).item() == 0


def test_invalid_view_receives_exactly_zero_gradient():
    anchor, target = _trajectory(4)
    predicted = (target + 0.1 * _trajectory(5)[1]).requires_grad_(True)
    validity = torch.ones(2, 1, 3, 1, 18)
    validity[..., 6:12] = 0
    loss = low_frequency_motion_consistency(
        predicted, target, anchor, validity_mask=validity
    ).total
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[..., :6].abs().sum().item() > 0
    assert torch.count_nonzero(predicted.grad[..., 6:12]).item() == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_views": 4},
        {"kernel_size": 4},
        {"sigma": 0.0},
        {"beta": 0.0},
        {"epsilon": 0.0},
    ],
)
def test_loss_rejects_geometry_or_scale_drift(kwargs):
    anchor, future = _trajectory()
    with pytest.raises(ValueError):
        low_frequency_motion_consistency(future, future, anchor, **kwargs)


def test_protocol_is_parameter_matched_and_target_free_at_inference():
    assert [arm.loss_weight for arm in contract.ARMS] == [0.0, 0.05]
    assert contract.ACTION_CONTROLS == ("aligned", "episode_shuffled", "zero")
    assert contract.TRAIN_UPDATES == 200
    assert contract.TRAIN_CLIPS == 512
    assert contract.VALIDATION_CLIPS == 64
    source = (
        PROJECT / "lam/motion_band_training_only_model.py"
    ).read_text(encoding="utf-8")
    sampler = source.split("    def sample_future_deployable(", 1)[1]
    assert "low_frequency_motion_consistency" not in sampler
    assert "future_rgb" not in sampler
    assert "history_rgb" in sampler
    assert "actions" in sampler


def test_registered_geometry_discloses_two_token_limit_and_real_wan_height():
    source = (ROOT / "tools/motion_band_training_only_screen.py").read_text(
        encoding="utf-8"
    )
    assert "T=2,H=24,W=120" in source
    assert "cannot identify frame-level contact timing" in source


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
                    "motion_band_loss_calls_at_inference": 0,
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


def test_result_contract_requires_full_val64_action_control_grid():
    arm = contract.ARM_BY_CODE["LFMREG-ON"]
    rows = _valid_rows(arm)
    contract.validate_result_rows(rows, arm)
    rows[-1] = dict(rows[0])
    with pytest.raises(contract.LFMREGContractError):
        contract.validate_result_rows(rows, arm)


def test_paired_bootstrap_recovers_known_ten_percent_gain():
    off = [1.0] * contract.VALIDATION_CLIPS
    on = [0.9] * contract.VALIDATION_CLIPS
    point, interval = contract._bootstrap_improvement(off, on, draws=256)
    assert point == pytest.approx(10.0)
    assert interval == pytest.approx([10.0, 10.0])


def test_workflow_is_dry_run_by_default_and_has_no_test_mode():
    submit = (
        ROOT / "tools/slurm/submit_motion_band_training_only_screen.sh"
    ).read_text(encoding="utf-8")
    workflow = (
        ROOT / "tools/slurm/motion_band_training_only_workflow.sbatch"
    ).read_text(encoding="utf-8")
    assert "EXECUTE=0" in submit
    assert "--execute" in submit
    assert "Dry-run only" in submit
    assert "protected" not in workflow.lower()
    assert "test" not in workflow.split('[[ "$MODE" ==', 1)[1].split("]]", 1)[0]
