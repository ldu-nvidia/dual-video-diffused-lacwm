from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from robot_wm.modeling.dual_diffusion.action_cycle import (
    FEATURE_DIM,
    TARGET_DIM,
    ActionCycleError,
    FrozenStage0RidgeCritic,
    aligned_action_targets,
    critic_is_absent_from_model_state,
    latent_displacement_features,
    rf_predicted_clean,
)
from tools import action_cycle_recoverability as stage0
from tools import action_cycle_stage1 as protocol
from tools import analyze_action_cycle_stage1 as analysis


STAGE0_IDENTITY = "1" * 64


def _bundle(path: Path) -> tuple[Path, str]:
    feature_mean = np.zeros((3, 3, FEATURE_DIM), dtype=np.float32)
    feature_std = np.ones_like(feature_mean)
    feature_active = np.ones_like(feature_mean, dtype=np.bool_)
    target_mean = np.zeros((3, TARGET_DIM), dtype=np.float32)
    target_std = np.ones_like(target_mean)
    target_active = np.zeros_like(target_mean, dtype=np.bool_)
    target_active[:, :4] = True
    weight = np.zeros((3, 3, FEATURE_DIM, TARGET_DIM), dtype=np.float32)
    weight[..., :4] = 1.0 / FEATURE_DIM
    np.savez(
        path,
        selected_alpha=np.asarray(0.1, dtype=np.float64),
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_active=feature_active,
        target_mean=target_mean,
        target_std=target_std,
        target_active=target_active,
        aligned_weight=weight,
        stage0_registration_identity_sha256=np.asarray(STAGE0_IDENTITY),
        source_frozen_ridge_sha256=np.asarray("2" * 64),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def test_stage1_feature_is_word_for_word_stage0_equivalent() -> None:
    generator = torch.Generator().manual_seed(123)
    latent = torch.randn(2, 16, 4, 24, 120, generator=generator)
    expected = stage0.latent_displacement_features(latent)
    observed = latent_displacement_features(latent)
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_feature_never_crosses_camera_seam() -> None:
    latent = torch.randn(1, 16, 4, 24, 120)
    changed = latent.clone()
    changed[..., :40] += torch.randn_like(changed[..., :40])
    before = latent_displacement_features(latent)
    after = latent_displacement_features(changed)
    assert not torch.equal(before[:, :, 0], after[:, :, 0])
    torch.testing.assert_close(before[:, :, 1:], after[:, :, 1:], rtol=0.0, atol=0.0)


def test_aligned_targets_ignore_padding_and_terminal_chunk() -> None:
    actions = torch.arange(2 * 13 * 5 * 157, dtype=torch.float32).reshape(
        2, 13, 5, 157
    )
    target = aligned_action_targets(actions)
    assert target.shape == (2, 3, 460)
    for transition, (start, stop) in enumerate(((0, 4), (4, 8), (8, 12))):
        expected = actions[:, start:stop, :, :23].reshape(2, -1)
        torch.testing.assert_close(target[:, transition], expected)
    mutated = actions.clone()
    mutated[..., 23:] += 1.0e6
    mutated[:, 12, :, :23] += 1.0e6
    torch.testing.assert_close(aligned_action_targets(mutated), target)


def test_rf_predicted_clean_clock_sign_and_endpoints() -> None:
    noisy = torch.randn(2, 3, 4, 5, 6)
    velocity = torch.randn_like(noisy)
    sigma = torch.tensor([0.0, 1.0])
    observed = rf_predicted_clean(noisy, sigma, velocity)
    torch.testing.assert_close(observed[0], noisy[0])
    torch.testing.assert_close(observed[1], noisy[1] - velocity[1])


def test_frozen_critic_has_gradient_only_to_predicted_latent(tmp_path: Path) -> None:
    path, digest = _bundle(tmp_path / "critic.npz")
    critic = FrozenStage0RidgeCritic(
        path,
        expected_sha256=digest,
        expected_stage0_registration_identity=STAGE0_IDENTITY,
    )
    predicted = torch.randn(2, 16, 4, 24, 120, requires_grad=True)
    actions = torch.randn(2, 13, 5, 157)
    loss, telemetry = critic.predict_and_loss(predicted, actions)
    assert loss.shape == (2,)
    assert telemetry["prediction"].shape == (2, 3, TARGET_DIM)
    loss.mean().backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    with pytest.raises(ActionCycleError, match="no model state_dict"):
        critic.state_dict()


def test_critic_digest_is_fail_closed(tmp_path: Path) -> None:
    path, _ = _bundle(tmp_path / "critic.npz")
    with pytest.raises(ActionCycleError, match="digest differs"):
        FrozenStage0RidgeCritic(
            path,
            expected_sha256="0" * 64,
            expected_stage0_registration_identity=STAGE0_IDENTITY,
        )


def test_plain_critic_does_not_enter_module_state(tmp_path: Path) -> None:
    path, digest = _bundle(tmp_path / "critic.npz")
    critic = FrozenStage0RidgeCritic(
        path,
        expected_sha256=digest,
        expected_stage0_registration_identity=STAGE0_IDENTITY,
    )

    class Fake(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.action_cycle_critic = critic

    model = Fake()
    assert tuple(model.state_dict()) == ("weight",)
    assert critic_is_absent_from_model_state(model)


def test_protocol_is_contingent_and_parameter_matched() -> None:
    frozen = protocol.fixed_protocol()
    assert frozen["prerequisite"]["decision"] == (
        "go_to_generated_latent_action_cycle_stage1"
    )
    assert frozen["prerequisite"]["paired_bootstrap_all_passed"] is True
    assert frozen["action_cycle"]["critic_trainable_parameters"] == 0
    assert frozen["action_cycle"]["checkpoint_parameter_or_buffer_delta"] == 0
    assert frozen["action_cycle"]["extra_wan_calls"] == 0
    assert [arm.loss_weight for arm in protocol.ARMS] == [0.0, 0.05]
    assert frozen["analysis_gate"]["primary_claim_contrasts"] == 9


def test_official_stop_shape_cannot_authorize_stage1() -> None:
    registration = {"identity_sha256": "a" * 64}
    stopped = stage0.identity_payload(
        {
            "kind": stage0.ANALYSIS_KIND,
            "registration_identity_sha256": registration["identity_sha256"],
            "decision": "stop_or_revise_action_cycle_path",
            "paired_bootstrap_gate": {"all_passed": False},
            "validation_fit_observations": 0,
            "protected_test_accessed": False,
            "target_cache_array_opened": False,
        }
    )
    with pytest.raises(protocol.ActionCycleStage1Error, match="did not produce"):
        protocol.validate_stage0_go_result(registration, stopped)


def test_paired_effect_and_did_signs() -> None:
    reference = np.linspace(1.0, 2.0, 64)
    candidate = reference * 0.9
    effect = analysis.paired_effect(candidate, reference, label="unit-effect")
    assert effect["relative_improvement"] == pytest.approx(0.1)
    assert effect["one_sided_lower_bound"]["low"] > 0.09
    did = analysis.difference_in_differences(
        candidate_aligned=reference * 0.8,
        candidate_control=reference,
        reference_aligned=reference * 0.95,
        reference_control=reference,
        label="unit-did",
        simultaneous=True,
    )
    assert did["relative_difference_in_differences"] == pytest.approx(0.15 / 0.95)
    assert did["one_sided_lower_bound"]["low"] > 0.0


def test_endpoint_control_inventory_is_fixed() -> None:
    assert [(value.code, value.nfe, value.action_source) for value in protocol.ENDPOINTS] == [
        ("aligned_nfe_1", 1, "aligned"),
        ("aligned_nfe_4", 4, "aligned"),
        ("shuffled_nfe_1", 1, "shuffled"),
        ("zero_nfe_1", 1, "zero"),
    ]
