from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from robot_wm.modeling.dual_diffusion.causal_spectral_probe import (  # noqa: E402
    ACTION_DESCRIPTOR_DIM,
    ACTION_TARGET_DIM,
    PHASE_INCREMENT_COMPONENT_DIM,
    SPECTRAL_FEATURE_DIM,
    VOLUME_MAGNITUDE_FEATURE_DIM,
    VOLUME_PHASE_COMPONENT_DIM,
    ActionPCATransform,
    FrozenCausalSpectralProbe,
    _energy_mask,
    action_descriptor,
    angle_neutral_spectral_features,
    causal_spectral_features,
    control_targets,
    episode_disjoint_pair_donors,
    fit_action_pca,
    future_motion_latents,
    phase0_partition_indexes,
)
from tools import (  # noqa: E402
    csip_analyze,
    csip_contract,
    csip_latent_cache,
    csip_train,
    csip_workflow,
)


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


def test_angle_neutral_comparator_keeps_shape_magnitude_and_phase_support() -> None:
    features = torch.zeros(3, SPECTRAL_FEATURE_DIM)
    features[:, :VOLUME_MAGNITUDE_FEATURE_DIM] = torch.randn(
        3, VOLUME_MAGNITUDE_FEATURE_DIM
    )
    volume_real_start = VOLUME_MAGNITUDE_FEATURE_DIM
    volume_imag_start = volume_real_start + VOLUME_PHASE_COMPONENT_DIM
    increment_real_start = volume_imag_start + VOLUME_PHASE_COMPONENT_DIM
    increment_imag_start = increment_real_start + PHASE_INCREMENT_COMPONENT_DIM
    features[0, volume_real_start] = -0.6
    features[0, volume_imag_start + 1] = 0.8
    features[1, increment_real_start + 2] = -1.0
    features[2, increment_imag_start + 3] = 1.0

    neutral = angle_neutral_spectral_features(features)
    assert neutral.shape == features.shape
    torch.testing.assert_close(
        neutral[:, :VOLUME_MAGNITUDE_FEATURE_DIM],
        features[:, :VOLUME_MAGNITUDE_FEATURE_DIM],
    )
    volume_support = (
        (features[:, volume_real_start:volume_imag_start] != 0)
        | (features[:, volume_imag_start:increment_real_start] != 0)
    ).float()
    increment_support = (
        (features[:, increment_real_start:increment_imag_start] != 0)
        | (features[:, increment_imag_start:] != 0)
    ).float()
    torch.testing.assert_close(
        neutral[:, volume_real_start:volume_imag_start], volume_support
    )
    assert torch.count_nonzero(neutral[:, volume_imag_start:increment_real_start]) == 0
    torch.testing.assert_close(
        neutral[:, increment_real_start:increment_imag_start], increment_support
    )
    assert torch.count_nonzero(neutral[:, increment_imag_start:]) == 0


def test_phase_endpoint_mask_uses_one_rms_shared_across_both_motion_tokens() -> None:
    values = torch.tensor([1.0, 5.0e-4], dtype=torch.complex64).reshape(
        1, 1, 1, 2, 1, 1
    )
    _magnitude, mask = _energy_mask(values, relative_floor=1.0e-3)
    assert mask.flatten().tolist() == [True, False]


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
    assert torch.count_nonzero(targets["raw_no_action"]) == 0


def test_raw_no_action_is_raw_descriptor_zero_not_pca_coordinate_zero() -> None:
    actions = torch.zeros(2, 13, 5, 23)
    components = torch.zeros(ACTION_TARGET_DIM, ACTION_DESCRIPTOR_DIM)
    components[:, :ACTION_TARGET_DIM] = torch.eye(ACTION_TARGET_DIM)
    pca = ActionPCATransform(
        mean=torch.ones(ACTION_DESCRIPTOR_DIM),
        components=components,
        score_scale=torch.ones(ACTION_TARGET_DIM),
    )
    targets = control_targets(
        actions,
        pca,
        torch.tensor([1, 0], dtype=torch.long),
    )
    torch.testing.assert_close(
        targets["raw_no_action"], -torch.ones(2, ACTION_TARGET_DIM)
    )


def test_action_pca_rejects_any_population_other_than_fit448() -> None:
    with pytest.raises(ValueError, match="fit448"):
        fit_action_pca(torch.zeros(447, ACTION_DESCRIPTOR_DIM))


def test_pair_control_is_symmetric_self_disjoint_and_episode_disjoint() -> None:
    donors, pairs = episode_disjoint_pair_donors(["a", "b", "c", "d"])
    assert pairs == ((0, 1), (2, 3))
    assert donors.tolist() == [1, 0, 3, 2]
    for index, donor in enumerate(donors.tolist()):
        assert donor != index
        assert int(donors[donor]) == index
    with pytest.raises(RuntimeError, match="one clip per episode"):
        episode_disjoint_pair_donors(["same", "same"])
    with pytest.raises(ValueError, match="even number"):
        episode_disjoint_pair_donors(["a", "b", "c"])


def test_probe_capacity_and_output_geometry_are_frozen() -> None:
    model = FrozenCausalSpectralProbe()
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_382_096
    output = model(torch.randn(3, SPECTRAL_FEATURE_DIM))
    assert output.shape == (3, ACTION_TARGET_DIM)


def _fixed_pair_bootstrap() -> tuple[np.ndarray, np.ndarray]:
    pair_blocks = np.repeat(
        np.arange(csip_contract.EXPECTED_VALIDATION_PAIR_BLOCKS, dtype=np.int64), 2
    )
    bootstrap = np.tile(
        np.arange(csip_contract.EXPECTED_VALIDATION_PAIR_BLOCKS, dtype=np.int64),
        (csip_contract.BOOTSTRAP_REPLICATES, 1),
    )
    return pair_blocks, bootstrap


def test_bootstrap_gate_uses_fixed_family_and_positive_direction() -> None:
    pair_blocks, bootstrap = _fixed_pair_bootstrap()
    mse = csip_analyze.paired_effect(
        np.full(64, 0.5),
        np.full(64, 1.0),
        metric="mse",
        bootstrap_pair_indexes=bootstrap,
        pair_block_ids=pair_blocks,
    )
    cosine = csip_analyze.paired_effect(
        np.full(64, 0.8),
        np.full(64, 0.2),
        metric="cosine",
        bootstrap_pair_indexes=bootstrap,
        pair_block_ids=pair_blocks,
    )
    assert csip_analyze.FAMILY_CELLS == 8
    assert mse["pass"] and mse["mean_effect"] == pytest.approx(0.5)
    assert cosine["pass"] and cosine["mean_effect"] == pytest.approx(0.6)


def test_bootstrap_gate_enforces_frozen_practical_effect_thresholds() -> None:
    pair_blocks, bootstrap = _fixed_pair_bootstrap()
    too_small = csip_analyze.paired_effect(
        np.full(64, 0.96),
        np.full(64, 1.0),
        metric="mse",
        bootstrap_pair_indexes=bootstrap,
        pair_block_ids=pair_blocks,
        **csip_analyze.CONTROL_THRESHOLDS["mse"],
    )
    large_enough = csip_analyze.paired_effect(
        np.full(64, 0.90),
        np.full(64, 1.0),
        metric="mse",
        bootstrap_pair_indexes=bootstrap,
        pair_block_ids=pair_blocks,
        **csip_analyze.CONTROL_THRESHOLDS["mse"],
    )
    assert not too_small["pass"]
    assert large_enough["pass"]
    assert large_enough["relative_mse_improvement"] == pytest.approx(0.10)


def test_bootstrap_gate_rejects_overlapping_or_singleton_pair_blocks() -> None:
    _pair_blocks, bootstrap = _fixed_pair_bootstrap()
    invalid_blocks = np.arange(64, dtype=np.int64) % 31
    with pytest.raises(RuntimeError, match="block-bootstrap"):
        csip_analyze.paired_effect(
            np.zeros(64),
            np.ones(64),
            metric="mse",
            bootstrap_pair_indexes=bootstrap,
            pair_block_ids=invalid_blocks,
        )
    with pytest.raises(RuntimeError, match="effect inputs"):
        csip_analyze.paired_effect(
            np.zeros(64),
            np.ones(64),
            metric="mse",
            bootstrap_pair_indexes=bootstrap.astype(np.float64),
            pair_block_ids=_pair_blocks,
        )


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
        csip_latent_cache.contract, "validate_registration"
    ) as validate_registration:
        with pytest.raises(RuntimeError, match="requires --seal"):
            csip_latent_cache._rank_zero_prepare(args)
    validate_registration.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_validation_latent_extraction_verifies_seal_before_opening_source(
    tmp_path,
) -> None:
    registration_path = tmp_path / "registration.json"
    seal_path = tmp_path / "checkpoint-seal.json"
    output = tmp_path / "latents" / "validation"
    registration = {
        "identity_sha256": "a" * 64,
        "planned_paths": {
            "validation_latent_root": str(output),
            "validation_latent_metadata": str(output / "metadata.json"),
        },
    }
    registration_record = {
        "path": str(registration_path),
        "bytes": 1,
        "sha256": "b" * 64,
        "identity_sha256": registration["identity_sha256"],
    }
    seal = {
        "identity_sha256": "c" * 64,
        "registration": registration_record,
    }
    events: list[tuple[str, bool | str]] = []

    def validate_seal(path):
        events.append(("seal", str(path)))
        return seal, registration

    def validate_registration(_path, **kwargs):
        events.append(("registration", bool(kwargs.get("open_validation"))))
        return registration

    with (
        mock.patch.object(
            csip_workflow, "validate_seal", side_effect=validate_seal
        ),
        mock.patch.object(
            csip_latent_cache.contract,
            "validate_registration",
            side_effect=validate_registration,
        ),
        mock.patch.object(
            csip_latent_cache.contract,
            "registration_file_record",
            return_value=registration_record,
        ),
        mock.patch.object(
            csip_latent_cache.contract,
            "file_record",
            return_value={"path": str(seal_path), "bytes": 1, "sha256": "d" * 64},
        ),
    ):
        observed_registration, observed_output, observed_seal = (
            csip_latent_cache._rank_zero_prepare(
                SimpleNamespace(
                    registration=registration_path,
                    split="validation",
                    seal=seal_path,
                )
            )
        )

    assert events == [
        ("seal", str(seal_path)),
        ("registration", True),
    ]
    assert observed_registration is registration
    assert observed_output == output
    assert observed_seal["identity_sha256"] == seal["identity_sha256"]


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
                "wan_dir": str(tmp_path),
                "videox_home": str(tmp_path),
                "videox_git_commit": csip_contract.EXPECTED_VIDEOX_COMMIT,
                "videox_git_tree": csip_contract.EXPECTED_VIDEOX_COMMIT,
                "world_size": csip_contract.EXPECTED_WORLD_SIZE,
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
            side_effect=lambda _repo, *arguments: (
                ""
                if arguments and arguments[0] == "status"
                else csip_contract.EXPECTED_VIDEOX_COMMIT
            ),
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
        "feature_variants": ["full", "angle_neutral"],
        "paired_initialization": "identical_state_before_update_1",
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
        "model_states": {
            "full": {"weight": torch.zeros(1)},
            "angle_neutral": {"weight": torch.zeros(1)},
        },
        "calibration_metrics": {
            str(value): {
                probe: {"mse": 1.0, "cosine": 0.0}
                for probe in ("full", "angle_neutral")
            }
            for value in (0, 100, 200, 300, 400)
        },
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


def test_pinned_videox_validation_rejects_dirty_checkout(tmp_path) -> None:
    outputs = iter(
        (
            csip_contract.EXPECTED_VIDEOX_COMMIT,
            " M videox_fun/models/wan_vae.py",
        )
    )
    with (
        mock.patch.object(csip_contract, "canonical_directory", return_value=tmp_path),
        mock.patch.object(
            csip_contract, "git_output", side_effect=lambda *_: next(outputs)
        ),
        pytest.raises(RuntimeError, match="dirty"),
    ):
        csip_contract.clean_pinned_checkout(
            tmp_path,
            label="VideoX",
            expected_commit=csip_contract.EXPECTED_VIDEOX_COMMIT,
        )


def _render_registration(tmp_path) -> dict:
    return {
        "identity_sha256": "a" * 64,
        "study_root": str(tmp_path / "study"),
        "source": {"path": str(tmp_path / "repo")},
        "runtime": {"python": {"launcher_path": str(tmp_path / "python")}},
        "planned_paths": {
            "checkpoint_seal": str(tmp_path / "study" / "checkpoint-seal.json"),
            "checkpoint": str(tmp_path / "study" / "checkpoint.pt"),
            "training_report": str(tmp_path / "study" / "training-report.json"),
            "evaluation": str(tmp_path / "study" / "evaluation.json"),
        },
    }


def test_train_render_never_opens_validation_or_requires_seal(
    tmp_path, capsys
) -> None:
    registration_path = tmp_path / "registration.json"
    registration = _render_registration(tmp_path)
    events: list[tuple[str, bool]] = []

    def validate_registration(_path, **kwargs):
        events.append(("registration", bool(kwargs.get("open_validation"))))
        return registration

    with (
        mock.patch.object(
            csip_workflow.contract,
            "regular_file",
            return_value=registration_path,
        ),
        mock.patch.object(
            csip_workflow.contract,
            "validate_registration",
            side_effect=validate_registration,
        ),
        mock.patch.object(csip_workflow, "validate_seal") as validate_seal,
    ):
        csip_workflow.command_render(
            SimpleNamespace(registration=registration_path, stage="train")
        )

    payload = json.loads(capsys.readouterr().out)
    assert events == [("registration", False)]
    validate_seal.assert_not_called()
    assert payload["boundary_receipt"] == {
        "stage": "train",
        "validation_source_records_opened": False,
        "checkpoint_seal_required": False,
        "checkpoint_seal_validated": False,
    }


def test_validation_render_verifies_seal_before_opening_validation(
    tmp_path, capsys
) -> None:
    registration_path = tmp_path / "registration.json"
    registration = _render_registration(tmp_path)
    registration_record = {
        "path": str(registration_path),
        "bytes": 1,
        "sha256": "b" * 64,
        "identity_sha256": registration["identity_sha256"],
    }
    seal = {
        "identity_sha256": "c" * 64,
        "registration": registration_record,
    }
    events: list[tuple[str, bool | str]] = []

    def validate_registration(_path, **kwargs):
        events.append(("registration", bool(kwargs.get("open_validation"))))
        return registration

    def validate_seal(path):
        events.append(("seal", str(path)))
        return seal, registration

    with (
        mock.patch.object(
            csip_workflow.contract,
            "regular_file",
            return_value=registration_path,
        ),
        mock.patch.object(
            csip_workflow.contract,
            "validate_registration",
            side_effect=validate_registration,
        ),
        mock.patch.object(csip_workflow, "validate_seal", side_effect=validate_seal),
        mock.patch.object(
            csip_workflow.contract,
            "registration_file_record",
            return_value=registration_record,
        ),
    ):
        csip_workflow.command_render(
            SimpleNamespace(registration=registration_path, stage="validation")
        )

    payload = json.loads(capsys.readouterr().out)
    assert events == [
        ("registration", False),
        ("seal", registration["planned_paths"]["checkpoint_seal"]),
        ("registration", True),
    ]
    assert payload["boundary_receipt"] == {
        "stage": "validation",
        "validation_source_records_opened": True,
        "checkpoint_seal_required": True,
        "checkpoint_seal_validated": True,
        "checkpoint_seal_identity_sha256": seal["identity_sha256"],
    }


def test_validation_render_rejects_registration_identity_change(tmp_path) -> None:
    registration_path = tmp_path / "registration.json"
    registration = _render_registration(tmp_path)
    changed_registration = {
        **registration,
        "identity_sha256": "d" * 64,
    }
    registration_record = {
        "path": str(registration_path),
        "bytes": 1,
        "sha256": "b" * 64,
        "identity_sha256": registration["identity_sha256"],
    }
    seal = {
        "identity_sha256": "c" * 64,
        "registration": registration_record,
    }
    with (
        mock.patch.object(
            csip_workflow.contract,
            "regular_file",
            return_value=registration_path,
        ),
        mock.patch.object(
            csip_workflow.contract,
            "validate_registration",
            side_effect=(registration, changed_registration),
        ),
        mock.patch.object(
            csip_workflow,
            "validate_seal",
            return_value=(seal, registration),
        ),
        mock.patch.object(
            csip_workflow.contract,
            "registration_file_record",
            return_value=registration_record,
        ),
        pytest.raises(RuntimeError, match="changed while validation"),
    ):
        csip_workflow.command_render(
            SimpleNamespace(registration=registration_path, stage="validation")
        )


def test_csip_slurm_launch_is_nonrequeueable_dependency_ordered_and_excludes_bad_nodes() -> (
    None
):
    root = csip_contract.REPO_ROOT
    stage = (root / "tools/slurm/csip_phase0_stage.sbatch").read_text()
    submit = (root / "tools/slurm/submit_csip_phase0.sh").read_text()
    assert "#SBATCH --no-requeue" in stage
    assert "pool0-0081,pool0-0089,pool0-0200,pool0-0343" in stage
    assert "SLURM_RESTART_COUNT:-0" in stage
    assert "PYTHONNOUSERSITE=1" in stage
    assert "status --porcelain --untracked-files=all" in stage
    assert 'dependency="afterok:$TRAIN_JOB_ID"' in submit
    assert "--kill-on-invalid-dep=yes" in submit
    assert "ALLOW_ACTIVE_JOB_IDS" in submit
    assert '--registration "$REGISTRATION" --stage "$STAGE"' in stage
    assert 'stage-boundary-receipt.json' in stage
    assert '--registration "$REGISTRATION" --stage train' in submit
