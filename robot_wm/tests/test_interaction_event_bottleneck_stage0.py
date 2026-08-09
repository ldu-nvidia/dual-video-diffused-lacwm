from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from tools import interaction_event_bottleneck_stage0 as screen
from tools import interaction_event_token_stage0 as token_screen


def test_motion_stratified_selection_is_deterministic_and_donor_safe() -> None:
    eligible = [
        {
            "manifest_index": index,
            "clip_id": f"{index:064x}",
            "episode_dir": f"/immutable/train/episode-{index}",
            "planned_joint_motion_rms": float(index),
        }
        for index in range(400)
    ]
    fit, score = screen.select_fit_score(eligible)
    fit_again, score_again = screen.select_fit_score(list(reversed(eligible)))
    assert fit == fit_again
    assert score == score_again
    assert len(fit) == screen.FIT_CLIPS
    assert len(score) == screen.SCORE_CLIPS
    assert {item["episode_dir"] for item in fit}.isdisjoint(
        {item["episode_dir"] for item in score}
    )
    score_indexes = {item["manifest_index"] for item in score}
    by_index = {item["manifest_index"]: item for item in score}
    for item in score:
        assert item["donor_manifest_index"] in score_indexes
        donor = by_index[item["donor_manifest_index"]]
        assert donor["episode_dir"] != item["episode_dir"]
        assert donor["motion_stratum"] == item["motion_stratum"]


def test_transition_event_field_masks_robot_and_retains_nonrobot_change() -> None:
    source = np.zeros(screen.WORK_HW, dtype=np.float32)
    target = source.copy()
    target[10:16, 60:68] = 1.0
    robot_source = np.zeros(screen.RENDER_HW, dtype=bool)
    robot_target = np.zeros(screen.RENDER_HW, dtype=bool)
    robot_source[70:110, 130:190] = True
    robot_target[70:110, 136:196] = True
    field, audit = screen.transition_event_field(
        source, target, robot_source, robot_target
    )
    assert field.shape == (4, *screen.WORK_HW)
    assert np.isfinite(field).all()
    # The nonrobot square is retained as positive appearance.
    assert float(field[0, 10:16, 60:68].mean()) > 0.1
    # The central rendered-robot band is exactly excluded after dilation.
    assert np.count_nonzero(field[:, 18:28, 30:50]) == 0
    assert 0.0 < audit["robot_core_fraction"] < audit["robot_excluded_fraction"] < 1.0
    assert audit["nonrobot_event_mass"] > 0.0


def test_common_bootstrap_and_bonferroni_gate_are_reproducible() -> None:
    reference = np.linspace(1.0, 2.0, screen.SCORE_CLIPS)
    candidate = reference * 0.8
    first_indexes = screen.common_bootstrap_indices(screen.SCORE_CLIPS)
    second_indexes = screen.common_bootstrap_indices(screen.SCORE_CLIPS)
    assert np.array_equal(first_indexes, second_indexes)
    first = screen.paired_relative_effect(
        reference,
        candidate,
        first_indexes,
        family_size=screen.PRIMARY_FAMILY_SIZE,
    )
    second = screen.paired_relative_effect(
        reference,
        candidate,
        second_indexes,
        family_size=screen.PRIMARY_FAMILY_SIZE,
    )
    assert first == second
    assert first["relative_improvement_percent"] == pytest.approx(20.0)
    assert first["simultaneous_lower_bound_percent"] > 0.0
    assert screen.effect_gate(first, minimum_percent=5.0)["passed"] is True


def test_full_gate_requires_every_causal_and_reconstruction_contrast() -> None:
    base = np.linspace(1.0, 2.0, screen.SCORE_CLIPS)
    arrays: dict[str, np.ndarray] = {}
    for metric in ("coefficient", "field"):
        arrays[f"{metric}_error_aligned"] = base * 0.70
        arrays[f"{metric}_error_history_only"] = base
        arrays[f"{metric}_error_episode_shuffled"] = base * 1.05
        arrays[f"{metric}_error_train_mean"] = base * 1.02
        arrays[f"{metric}_error_raw_zero"] = base * 1.20
        arrays[f"{metric}_error_shift_minus1"] = base * 0.95
        arrays[f"{metric}_error_shift_plus1"] = base * 0.98
    arrays["reconstruction_error_oracle"] = base * 0.20
    arrays["reconstruction_error_fit_mean"] = base
    arrays["reconstruction_error_raw_zero"] = base * 1.20
    effects, gates = screen.analyze_error_arrays(arrays)
    assert len(effects["mandatory"]) == 8
    assert len(effects["diagnostic"]) == 6
    assert gates["all_passed"] is True

    arrays["field_error_episode_shuffled"] = base * 0.69
    _, failed = screen.analyze_error_arrays(arrays)
    assert failed["field:aligned_vs_episode_shuffled"]["passed"] is False
    assert failed["all_passed"] is False


def test_explicit_token_state_is_exact_spatial_channel_mean() -> None:
    field = np.arange(8 * 4 * 45 * 80, dtype=np.float32).reshape(8, 4, 45, 80)
    tokens = token_screen.token_state(field)
    assert tokens.shape == (8, 4)
    np.testing.assert_allclose(tokens, field.mean(axis=(-2, -1)), rtol=1e-6)


def test_explicit_token_gate_requires_all_three_references_and_salience() -> None:
    base = np.linspace(1.0, 2.0, screen.SCORE_CLIPS)
    errors: dict[str, np.ndarray] = {}
    for subset in ("all_token", "change", "transport"):
        errors[f"{subset}_error_aligned"] = base * 0.70
        errors[f"{subset}_error_history_only"] = base
        errors[f"{subset}_error_episode_shuffled"] = base * 1.05
        errors[f"{subset}_error_train_mean"] = base * 1.02
        errors[f"{subset}_error_raw_zero"] = base * 1.20
        errors[f"{subset}_error_shift_minus1"] = base * 0.95
        errors[f"{subset}_error_shift_plus1"] = base * 0.98
    effects, gates = token_screen.analyze_errors(
        errors,
        active_token_dims=32,
        active_score_fraction=1.0,
    )
    assert len(effects["mandatory"]) == 3
    assert len(effects["diagnostic"]) == 15
    assert gates["all_passed"] is True

    _, inactive = token_screen.analyze_errors(
        errors,
        active_token_dims=23,
        active_score_fraction=1.0,
    )
    assert inactive["target_salience"]["passed"] is False
    assert inactive["all_passed"] is False
