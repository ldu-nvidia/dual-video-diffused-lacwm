from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from robot_wm.modeling.dual_diffusion.causal_spectral_probe import (  # noqa: E402
    ACTION_DESCRIPTOR_DIM,
    ACTION_TARGET_DIM,
    SPECTRAL_FEATURE_DIM,
    ActionPCATransform,
    FrozenCausalSpectralProbe,
    action_descriptor,
    causal_spectral_features,
    control_targets,
    episode_disjoint_cyclic_donors,
    fit_action_pca,
    future_motion_latents,
    phase0_partition_indexes,
)
from tools import csip_analyze, csip_contract, csip_latent_cache, csip_train  # noqa: E402


def _latents_from_motion(motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert motion.shape == (1, 16, 2, 24, 120)
    history = torch.zeros(1, 16, 2, 24, 120)
    full = torch.zeros(1, 16, 4, 24, 120)
    full[:, :, 2] = motion[:, :, 0]
    full[:, :, 3] = motion[:, :, 0] + motion[:, :, 1]
    return full, history


def _volume_unit(features: torch.Tensor) -> torch.Tensor:
    block = 3 * 16 * 2 * 4 * 6
    real = features[:, block : 2 * block].reshape(1, 3, 16, 2, 4, 6)
    imag = features[:, 2 * block : 3 * block].reshape(1, 3, 16, 2, 4, 6)
    return torch.complex(real, imag)


def _increment_unit(features: torch.Tensor) -> torch.Tensor:
    volume_block = 3 * 16 * 2 * 4 * 6
    increment_block = 3 * 16 * 1 * 4 * 6
    start = 3 * volume_block
    real = features[:, start : start + increment_block].reshape(1, 3, 16, 1, 4, 6)
    imag = features[:, start + increment_block :].reshape(1, 3, 16, 1, 4, 6)
    return torch.complex(real, imag)


def test_phase0_partition_is_exact_fit448_cal64() -> None:
    fit = phase0_partition_indexes(512, "fit")
    calibration = phase0_partition_indexes(512, "calibration")
    assert len(fit) == 448
    assert len(calibration) == 64
    assert calibration == tuple(range(0, 512, 8))
    assert set(fit).isdisjoint(calibration)
    assert sorted((*fit, *calibration)) == list(range(512))


def test_future_motion_uses_independent_history_anchor() -> None:
    history = torch.randn(2, 16, 2, 24, 120)
    full = torch.randn(2, 16, 4, 24, 120)
    motion = future_motion_latents(full, history)
    torch.testing.assert_close(motion[:, :, 0], full[:, :, 2] - history[:, :, 1])
    torch.testing.assert_close(motion[:, :, 1], full[:, :, 3] - full[:, :, 2])


def test_every_fft_sees_one_camera_view_not_width_stacked_seams() -> None:
    full = torch.randn(1, 16, 4, 24, 120)
    history = torch.randn(1, 16, 2, 24, 120)
    observed_shapes: list[tuple[int, ...]] = []
    original = torch.fft.fftn

    def recording_fftn(value, *args, **kwargs):
        observed_shapes.append(tuple(value.shape))
        return original(value, *args, **kwargs)

    with mock.patch("torch.fft.fftn", side_effect=recording_fftn):
        features = causal_spectral_features(full, history)
    assert features.shape == (1, SPECTRAL_FEATURE_DIM)
    assert observed_shapes == [
        (1, 3, 16, 2, 24, 40),
        (1, 3, 16, 2, 24, 40),
    ]


def test_zero_energy_is_fully_masked_and_finite() -> None:
    features = causal_spectral_features(
        torch.zeros(2, 16, 4, 24, 120),
        torch.zeros(2, 16, 2, 24, 120),
    )
    assert features.shape == (2, SPECTRAL_FEATURE_DIM)
    assert torch.isfinite(features).all()
    assert torch.count_nonzero(features) == 0


def test_positive_spatial_shift_obeys_negative_fft_phase_sign() -> None:
    motion = torch.zeros(1, 16, 2, 24, 120)
    motion[:, 0, :, 7, 5] = 1.0
    shifted = motion.clone()
    shifted[:, :, :, :, :40] = torch.roll(motion[:, :, :, :, :40], shifts=1, dims=-1)
    base_full, history = _latents_from_motion(motion)
    shifted_full, shifted_history = _latents_from_motion(shifted)
    base_phase = _volume_unit(causal_spectral_features(base_full, history))
    shifted_phase = _volume_unit(
        causal_spectral_features(shifted_full, shifted_history)
    )
    frequencies = torch.fft.fftshift(torch.fft.fftfreq(40))[17:23]
    expected = torch.exp(
        torch.complex(torch.zeros_like(frequencies), -2 * torch.pi * frequencies)
    )
    # Identical motion tokens have zero energy in the shifted temporal Nyquist
    # bin (index 0), so phase is intentionally masked there.  Check the DC
    # temporal bin, whose energy is nonzero at every retained spatial bin.
    ratio = shifted_phase[0, 0, 0, 1] / base_phase[0, 0, 0, 1]
    torch.testing.assert_close(
        ratio,
        expected[None, :].expand_as(ratio),
        atol=2e-5,
        rtol=2e-5,
    )


def test_phase_increment_reverses_to_complex_conjugate() -> None:
    forward = torch.zeros(1, 16, 2, 24, 120)
    forward[:, 0, 0, 7, 5] = 1.0
    forward[:, 0, 1, 7, 6] = 1.0
    reverse = forward.flip(2)
    forward_full, history = _latents_from_motion(forward)
    reverse_full, reverse_history = _latents_from_motion(reverse)
    forward_phase = _increment_unit(causal_spectral_features(forward_full, history))
    reverse_phase = _increment_unit(
        causal_spectral_features(reverse_full, reverse_history)
    )
    torch.testing.assert_close(
        reverse_phase[0, 0, 0], forward_phase[0, 0, 0].conj(), atol=2e-5, rtol=2e-5
    )


def test_action_descriptor_and_controls_preserve_sign_semantics() -> None:
    actions = torch.zeros(4, 13, 5, 23)
    ramp = torch.arange(5, dtype=torch.float32)[None, :, None]
    actions[:, 4:12] = ramp
    descriptor = action_descriptor(actions)
    assert descriptor.shape == (4, ACTION_DESCRIPTOR_DIM)
    assert torch.all(descriptor == 1)
    components = torch.zeros(ACTION_TARGET_DIM, ACTION_DESCRIPTOR_DIM)
    components[:, :ACTION_TARGET_DIM] = torch.eye(ACTION_TARGET_DIM)
    pca = ActionPCATransform(
        mean=torch.zeros(ACTION_DESCRIPTOR_DIM),
        components=components,
        score_scale=torch.ones(ACTION_TARGET_DIM),
    )
    donors = torch.tensor([1, 2, 3, 0], dtype=torch.long)
    targets = control_targets(actions, pca, donors)
    torch.testing.assert_close(targets["inverse"], -targets["aligned"])
    assert torch.count_nonzero(targets["zero"]) == 0


def test_action_pca_rejects_any_population_other_than_fit448() -> None:
    with pytest.raises(ValueError, match="fit448"):
        fit_action_pca(torch.zeros(447, ACTION_DESCRIPTOR_DIM))


def test_cyclic_control_is_self_and_episode_disjoint() -> None:
    donors, shift = episode_disjoint_cyclic_donors(["a", "b", "c", "d"])
    assert shift == 1
    assert donors.tolist() == [1, 2, 3, 0]
    with pytest.raises(RuntimeError, match="no episode-disjoint"):
        episode_disjoint_cyclic_donors(["same", "same"])


def test_probe_capacity_and_output_geometry_are_frozen() -> None:
    model = FrozenCausalSpectralProbe()
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_382_096
    output = model(torch.randn(3, SPECTRAL_FEATURE_DIM))
    assert output.shape == (3, ACTION_TARGET_DIM)


def test_bootstrap_gate_uses_fixed_family_and_positive_direction() -> None:
    bootstrap = np.tile(np.arange(64), (10_000, 1))
    mse = csip_analyze.paired_effect(
        np.full(64, 0.5),
        np.full(64, 1.0),
        metric="mse",
        bootstrap_indexes=bootstrap,
    )
    cosine = csip_analyze.paired_effect(
        np.full(64, 0.8),
        np.full(64, 0.2),
        metric="cosine",
        bootstrap_indexes=bootstrap,
    )
    assert csip_analyze.FAMILY_CELLS == 6
    assert mse["pass"] and mse["mean_effect"] == pytest.approx(0.5)
    assert cosine["pass"] and cosine["mean_effect"] == pytest.approx(0.6)


def test_training_and_cache_clis_accept_no_test_split_or_path() -> None:
    train_options = {
        option
        for action in csip_train.build_parser()._actions
        for option in action.option_strings
    }
    cache_choices = (
        csip_latent_cache.build_parser()._subparsers._group_actions[0].choices
    )
    extract = cache_choices["extract"]
    split_action = next(
        action for action in extract._actions if "--split" in action.option_strings
    )
    all_cache_options = {
        option
        for parser in cache_choices.values()
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--validation-manifest" not in train_options
    assert "--test" not in train_options | all_cache_options
    assert tuple(split_action.choices) == ("train", "validation")


def test_validation_latent_extraction_requires_checkpoint_seal(tmp_path) -> None:
    args = SimpleNamespace(
        registration=tmp_path / "registration.json",
        split="validation",
        seal=None,
    )
    with mock.patch.object(
        csip_latent_cache.contract, "validate_registration", return_value={}
    ):
        with pytest.raises(RuntimeError, match="requires --seal"):
            csip_latent_cache._rank_zero_prepare(args)
    assert list(tmp_path.iterdir()) == []


def test_train_stage_registration_validation_does_not_open_val_records(
    tmp_path,
) -> None:
    stub = tmp_path / "stub"
    stub.write_text("x", encoding="utf-8")
    file_stub = {"path": str(stub), "bytes": 1, "sha256": "b" * 64}
    payload = csip_contract.with_identity(
        {
            "schema_version": csip_contract.SCHEMA_VERSION,
            "kind": csip_contract.REGISTRATION_KIND,
            "status": "registered_before_latent_extraction_or_training",
            "protected_test_clips_read": 0,
            "source": {
                "path": str(tmp_path),
                "git_commit": "a" * 40,
                "integration_base_commit": csip_contract.INTEGRATION_BASE_COMMIT,
            },
            "datasets": {
                split: {
                    "manifest": file_stub,
                    "cache_metadata": file_stub,
                    "arrays": {"rgb": file_stub, "actions": file_stub},
                }
                for split in ("train", "validation")
            },
            "runtime": {
                "python": {**file_stub, "launcher_path": str(stub)},
                "wan_config": file_stub,
                "wan_vae": file_stub,
                "videox_home": str(tmp_path),
                "videox_git_commit": csip_contract.EXPECTED_VIDEOX_COMMIT,
            },
            "planned_paths": {},
        }
    )
    labels: list[str] = []

    def record_label(_record, label):
        labels.append(label)
        return tmp_path / "stub"

    with (
        mock.patch.object(csip_contract, "regular_file", return_value=tmp_path / "reg"),
        mock.patch.object(csip_contract, "read_json", return_value=payload),
        mock.patch.object(csip_contract, "clean_source", return_value={}),
        mock.patch.object(
            csip_contract, "verify_file_record", side_effect=record_label
        ),
        mock.patch.object(
            csip_contract,
            "git_output",
            return_value=csip_contract.EXPECTED_VIDEOX_COMMIT,
        ),
    ):
        csip_contract.validate_registration(tmp_path / "reg", open_validation=False)
    assert any(label.startswith("train ") for label in labels)
    assert not any(label.startswith("validation ") for label in labels)


def test_fixed_checkpoint_validator_rejects_partition_contamination() -> None:
    registration = {
        "identity_sha256": "a" * 64,
        "wandb": {"run_id": "fixed-run"},
    }
    payload = {
        "schema_version": csip_contract.SCHEMA_VERSION,
        "kind": csip_contract.CHECKPOINT_KIND,
        "registration_identity_sha256": "a" * 64,
        "train_latent_cache_identity_sha256": "b" * 64,
        "completed_updates": 400,
        "selected_update": 400,
        "selection_rule": "fixed_final_update_not_metric_selected",
        "seed": 1234,
        "model_hidden_dim": 256,
        "spectral_feature_dim": SPECTRAL_FEATURE_DIM,
        "action_target_dim": ACTION_TARGET_DIM,
        "wandb_run_id": "fixed-run",
        "validation_clips_read": 0,
        "protected_test_clips_read": 0,
        "fit_indexes": torch.tensor(phase0_partition_indexes(512, "fit")),
        "calibration_indexes": torch.tensor(
            phase0_partition_indexes(512, "calibration")
        ),
        "action_pca": {
            "mean": torch.zeros(ACTION_DESCRIPTOR_DIM),
            "components": torch.zeros(ACTION_TARGET_DIM, ACTION_DESCRIPTOR_DIM),
            "score_scale": torch.ones(ACTION_TARGET_DIM),
        },
        "calibration_metrics": {str(value): {} for value in (0, 100, 200, 300, 400)},
    }
    csip_contract.validate_checkpoint_payload(
        payload,
        registration=registration,
        train_latent_cache_identity_sha256="b" * 64,
    )
    payload["fit_indexes"] = payload["fit_indexes"].clone()
    payload["fit_indexes"][0] = 0
    with pytest.raises(RuntimeError, match="checkpoint payload"):
        csip_contract.validate_checkpoint_payload(
            payload,
            registration=registration,
            train_latent_cache_identity_sha256="b" * 64,
        )


def test_content_identity_detects_tampering() -> None:
    payload = csip_contract.with_identity({"kind": "test", "value": 1})
    csip_contract.verify_identity(payload, "test")
    payload["value"] = 2
    with pytest.raises(RuntimeError, match="identity"):
        csip_contract.verify_identity(payload, "test")
