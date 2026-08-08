from __future__ import annotations

import hashlib
import inspect
import json

import pytest


torch = pytest.importorskip("torch")

from lam.causal_motion_plan_model import (  # noqa: E402
    PLAN_CONDITION_SOURCES,
    CausalMotionPlanVPM,
    CausalMotionPlannerCalibrationModel,
)
from robot_wm.modeling.dual_diffusion.causal_motion_plan import (  # noqa: E402
    FROZEN_TRAIN_MANIFEST_SHA256,
    FROZEN_TRAIN_METADATA_SHA256,
    FROZEN_TRAIN_RGB_SHA256,
    NORMALIZATION_KIND,
    NORMALIZATION_SCHEMA_VERSION,
    CausalMotionPlanner,
    finalize_channel_moments,
    build_plan_condition,
    load_motion_plan_normalizer,
    motion_plan_target,
    planner_partition_indexes,
    pool_per_view,
    upsample_per_view,
)
from tools import causal_motion_plan_stats  # noqa: E402


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_stats(tmp_path, **changes):
    unsigned = {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "kind": NORMALIZATION_KIND,
        "status": "complete_before_planner_training",
        "split_rule": "auxiliary_index_mod_8_nonzero",
        "fit_clips": 448,
        "calibration_clips_excluded": 64,
        "validation_clips_read": 0,
        "protected_test_clips_read": 0,
        "elements_per_channel": 448 * 2 * 6 * 30,
        "train_manifest_sha256": FROZEN_TRAIN_MANIFEST_SHA256,
        "train_cache_metadata_sha256": FROZEN_TRAIN_METADATA_SHA256,
        "train_rgb_sha256": FROZEN_TRAIN_RGB_SHA256,
        "history_encoding": "independent_five_frame_observed_only",
        "future_tensor_used_for": "statistics_target_only",
        "causal_history_max_abs_tolerance": 1e-4,
        "causal_history_max_abs_observed": 0.0,
        "mean": [float(index) / 10.0 for index in range(16)],
        "std": [1.0 + float(index) / 100.0 for index in range(16)],
    }
    unsigned.update(changes)
    payload = {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }
    path = tmp_path / "motion-plan-stats.json"
    path.write_bytes(_canonical_json(payload) + b"\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def test_train_only_planner_partition_is_fixed_disjoint_and_complete() -> None:
    fit = planner_partition_indexes(512, "planner_fit")
    calibration = planner_partition_indexes(512, "planner_calibration")
    assert len(fit) == 448
    assert len(calibration) == 64
    assert set(fit).isdisjoint(calibration)
    assert sorted((*fit, *calibration)) == list(range(512))
    assert calibration == tuple(range(0, 512, 8))


def test_fit_only_stats_use_cpu_collectives_and_accumulators() -> None:
    # VAE work stays per-GPU; only provenance records and scalar moments move
    # between ranks. Guard the operational recovery without changing the fit.
    assert causal_motion_plan_stats.COLLECTIVE_BACKEND == "gloo"
    assert causal_motion_plan_stats.ACCUMULATOR_DEVICE == "cpu"


def test_per_view_pool_and_upsample_never_mix_camera_seams() -> None:
    value = torch.zeros(2, 16, 24, 120)
    value[..., :40] = 1
    value[..., 40:80] = 2
    value[..., 80:] = 3
    pooled = pool_per_view(value)
    assert pooled.shape == (2, 16, 6, 30)
    torch.testing.assert_close(pooled[..., :10], torch.ones_like(pooled[..., :10]))
    torch.testing.assert_close(
        pooled[..., 10:20], torch.full_like(pooled[..., 10:20], 2)
    )
    torch.testing.assert_close(
        pooled[..., 20:], torch.full_like(pooled[..., 20:], 3)
    )
    plan = torch.stack((pooled, pooled + 1), dim=2)
    upsampled = upsample_per_view(plan)
    assert upsampled.shape == (2, 16, 2, 24, 120)
    torch.testing.assert_close(
        upsampled[:, :, 0, :, :40],
        torch.ones_like(upsampled[:, :, 0, :, :40]),
    )
    torch.testing.assert_close(
        upsampled[:, :, 0, :, 40:80],
        torch.full_like(upsampled[:, :, 0, :, 40:80], 2),
    )


def test_motion_target_is_observed_anchor_then_future_increment() -> None:
    history = torch.randn(2, 16, 2, 24, 120)
    full = torch.randn(2, 16, 4, 24, 120)
    target = motion_plan_target(full, history)
    expected_first = pool_per_view(full[:, :, 2] - history[:, :, 1])
    expected_second = pool_per_view(full[:, :, 3] - full[:, :, 2])
    torch.testing.assert_close(target[:, :, 0], expected_first)
    torch.testing.assert_close(target[:, :, 1], expected_second)


def test_channel_moments_and_normalizer_roundtrip_are_exact(tmp_path) -> None:
    values = torch.randn(5, 16, 2, 6, 30, dtype=torch.float64)
    channel_sum = values.sum(dim=(0, 2, 3, 4))
    channel_square_sum = values.square().sum(dim=(0, 2, 3, 4))
    count = values.shape[0] * values.shape[2] * values.shape[3] * values.shape[4]
    mean, std = finalize_channel_moments(channel_sum, channel_square_sum, count)
    torch.testing.assert_close(mean, values.mean(dim=(0, 2, 3, 4)))
    torch.testing.assert_close(std, values.std(dim=(0, 2, 3, 4), correction=0))

    path, digest = _write_stats(tmp_path)
    normalizer = load_motion_plan_normalizer(
        path=str(path.resolve()), expected_sha256=digest
    )
    plan = torch.randn(2, 16, 2, 6, 30)
    torch.testing.assert_close(normalizer.denormalize(normalizer.normalize(plan)), plan)
    assert normalizer.artifact_sha256 == digest


def test_normalizer_rejects_calibration_contamination(tmp_path) -> None:
    path, digest = _write_stats(tmp_path, calibration_clips_excluded=63)
    with pytest.raises(RuntimeError, match="provenance"):
        load_motion_plan_normalizer(path=str(path.resolve()), expected_sha256=digest)


def test_plan_condition_contains_zero_history_and_generated_future_only() -> None:
    plan = torch.randn(2, 16, 2, 6, 30)
    condition = build_plan_condition(plan)
    assert condition.shape == (2, 16, 4, 24, 120)
    assert torch.count_nonzero(condition[:, :, :2]) == 0
    assert torch.count_nonzero(condition[:, :, 2:]) > 0


def test_small_planner_has_exact_two_call_autonomous_rollout() -> None:
    planner = CausalMotionPlanner(hidden_size=16, num_blocks=1, sigma_dim=8)
    history = torch.randn(2, 16, 2, 24, 120)
    actions = torch.randn(2, 13, 5, 157)
    morphology = torch.tensor([0, 1])
    noise = torch.randn(2, 16, 2, 6, 30)
    output = planner.rollout_two_step(noise, history, actions, morphology)
    assert output.calls == 2
    assert output.plan.shape == noise.shape
    assert torch.isfinite(output.plan).all()


def test_deployment_signatures_cannot_accept_clean_or_future_targets() -> None:
    planner_parameters = inspect.signature(
        CausalMotionPlanner.rollout_two_step
    ).parameters
    sampler_parameters = inspect.signature(
        CausalMotionPlanVPM.sample_causal_motion_plan
    ).parameters
    forbidden = ("clean", "target", "teacher", "future", "oracle")
    for parameters in (planner_parameters, sampler_parameters):
        assert not any(
            token in name for name in parameters for token in forbidden
        )


def test_action_shuffle_is_a_registered_two_call_diagnostic() -> None:
    assert PLAN_CONDITION_SOURCES == ("aligned", "off", "shuffled", "action_shuffled")
    sampler_source = inspect.getsource(CausalMotionPlanVPM.sample_causal_motion_plan)
    assert 'condition_source == "action_shuffled"' in sampler_source
    assert "roll_across_global_batch(actions)" in sampler_source
    assert sampler_source.count("rollout_two_step(") == 1


def test_calibration_model_is_the_only_model_path_with_clean_plan_target() -> None:
    calibration_source = inspect.getsource(CausalMotionPlannerCalibrationModel.forward)
    production_source = inspect.getsource(CausalMotionPlanVPM.forward)
    assert "motion_plan_target" in calibration_source
    assert "motion_plan_target" not in production_source
    assert "_autonomous_condition" in production_source
