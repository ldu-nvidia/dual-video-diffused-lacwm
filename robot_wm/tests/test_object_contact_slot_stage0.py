from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from tools import interaction_event_bottleneck_stage0 as base
from tools import object_contact_slot_stage0 as slot


def _synthetic_parent() -> tuple[list[dict[str, object]], dict[str, object]]:
    eligible = [
        {
            "manifest_index": index,
            "clip_id": f"{index:064x}",
            "episode_dir": f"/immutable/train/episode-{index}",
            "planned_joint_motion_rms": float(index),
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
        for index in range(415)
    ]
    fit, score = base.select_fit_score(eligible)
    parent = base.seal(
        {
            "kind": "interaction_event_bottleneck_stage0_registration",
            "split": {"fit": fit, "score": score},
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    parent["identity_sha256"] = slot.PARENT_REGISTRATION_IDENTITY
    return eligible, parent


def test_fresh_selection_excludes_every_prior_score_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eligible, parent = _synthetic_parent()
    monkeypatch.setattr(slot.base, "validate_identity", lambda *_: None)
    first = slot.select_fresh_score(eligible, parent)
    second = slot.select_fresh_score(list(reversed(eligible)), parent)
    fit, fresh, pool, counts = first
    assert [item["manifest_index"] for item in fresh] == [
        item["manifest_index"] for item in second[1]
    ]
    prior = {item["manifest_index"] for item in parent["split"]["score"]}
    fit_indexes = {item["manifest_index"] for item in fit}
    fresh_indexes = {item["manifest_index"] for item in fresh}
    assert prior.isdisjoint(fresh_indexes)
    assert fit_indexes.isdisjoint(fresh_indexes)
    assert len(pool) == 95
    assert counts["fresh_score"] == 64
    assert counts["unused_untouched_after_fresh_selection"] == 31


def test_tracker_keeps_identity_and_emits_null_slots() -> None:
    def candidate(x: float, contact: float) -> dict[str, object]:
        return {
            "component_label": 1,
            "mask": np.zeros(slot.WORK_HW, dtype=bool),
            "centroid": np.asarray([x, 0.1], dtype=np.float32),
            "area_fraction": 0.01,
            "positive_mass": 0.002,
            "negative_mass": 0.001,
            "covariance_xx": 0.01,
            "covariance_xy": 0.0,
            "covariance_yy": 0.02,
            "signed_motion_u": 0.1,
            "signed_motion_v": 0.0,
            "contact_score": contact,
            "salience_mass": 2.0,
        }

    states, records = slot.track_components(
        [[candidate(-0.2, 0.1)], [candidate(-0.1, 0.4)], [], [candidate(0.1, 0.5)]]
    )
    assert states.shape == (4, slot.SLOT_COUNT, slot.SLOT_DIM)
    assert np.all(states[:2, 0, slot.PRESENCE_INDEX] == 1.0)
    assert states[2, 0, slot.PRESENCE_INDEX] == 0.0
    assert states[3, 0, slot.PRESENCE_INDEX] == 1.0
    assert states[1, 0, slot.SLOT_FEATURES.index("contact_onset")] == pytest.approx(0.3)
    assert np.count_nonzero(states[:, 1:]) == 0
    assert all(record["future_observations_consulted"] is False for record in records)


def test_slot_decoder_preserves_location_and_mass() -> None:
    state = np.zeros((1, slot.SLOT_COUNT, slot.SLOT_DIM), dtype=np.float32)
    state[0, 0, slot.PRESENCE_INDEX] = 1.0
    state[0, 0, 1:3] = [0.5, -0.5]
    state[0, 0, 4:6] = [0.01, 0.005]
    state[0, 0, 6:9] = [0.01, 0.0, 0.01]
    state[0, 0, 9:11] = [0.2, -0.1]
    decoded = slot.decode_slot_sequence(state)
    assert decoded.shape == (1, 4, *slot.WORK_HW)
    assert np.isfinite(decoded).all()
    assert float(decoded[0, 0].sum()) == pytest.approx(
        0.01 * np.prod(slot.WORK_HW), rel=1e-5
    )
    peak_y, peak_x = np.unravel_index(np.argmax(decoded[0, 0]), slot.WORK_HW)
    assert peak_x > slot.WORK_HW[1] // 2
    assert peak_y < slot.WORK_HW[0] // 2


def test_gate_requires_all_causal_timing_relevance_and_salience() -> None:
    base_error = np.linspace(1.0, 2.0, slot.SCORE_CLIPS)
    errors: dict[str, np.ndarray] = {}
    for metric in slot.CAUSAL_METRICS:
        errors[f"{metric}_error_aligned"] = base_error * 0.70
        for reference in slot.MANDATORY_REFERENCES:
            errors[f"{metric}_error_{reference}"] = base_error
    errors["reconstruction_error_oracle"] = base_error * 0.20
    errors["reconstruction_error_fit_mean"] = base_error
    errors["reconstruction_error_raw_zero"] = base_error
    salience = {
        "active_target_fraction_at_least_50_percent": True,
        "score_nonempty_fraction_at_least_90_percent": True,
        "score_persistent_fraction_at_least_60_percent": True,
        "passed": True,
    }
    effects, gates = slot.analyze_errors(errors, salience)
    assert len(effects["causal"]) == 12
    assert len(effects["reconstruction"]) == 2
    assert gates["all_passed"] is True

    errors["field_error_shift_plus1"] = base_error * 0.69
    _, failed = slot.analyze_errors(errors, salience)
    assert failed["causal"]["field:aligned_vs_shift_plus1"]["passed"] is False
    assert failed["all_passed"] is False
