from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from tools import object_contact_distribution_stage0 as dist
from tools import object_contact_distribution_pca_repair as repair
from tools import object_contact_slot_stage0 as slot


def test_compact_summary_is_slot_permutation_invariant_and_null_stable() -> None:
    states = np.zeros((2, 8, slot.SLOT_COUNT, slot.SLOT_DIM), dtype=np.float32)
    states[0, 2, 0, slot.PRESENCE_INDEX] = 1.0
    states[0, 2, 0, 1:3] = [0.4, -0.2]
    states[0, 2, 0, 4:6] = [0.01, 0.02]
    states[0, 2, 0, 9:12] = [0.3, -0.1, 0.8]
    summary = dist.compact_summary(states)
    permuted = states[:, :, [2, 0, 3, 1]]
    np.testing.assert_allclose(summary, dist.compact_summary(permuted))
    assert summary.shape == (2, dist.TARGET_DIM)
    assert np.count_nonzero(summary[1]) == 0
    assert summary[0, 0] > 0.0
    assert summary[0, 2] == pytest.approx(0.4)


def test_conditional_centers_use_fixed_equal_k_and_action_changes_neighbors() -> None:
    fit_history = np.zeros((64, dist.TARGET_DIM), dtype=np.float32)
    fit_history[:, 0] = np.linspace(-2.0, 2.0, 64)
    fit_target = np.repeat(np.arange(64, dtype=np.float32)[:, None], dist.TARGET_DIM, axis=1)
    fit_action = np.zeros((64, dist.ACTION_COMPONENTS), dtype=np.float32)
    fit_action[:, 0] = np.linspace(2.0, -2.0, 64)
    query_history = np.zeros((2, dist.TARGET_DIM), dtype=np.float32)
    query_action = np.zeros((2, dist.ACTION_COMPONENTS), dtype=np.float32)
    query_action[:, 0] = [2.0, -2.0]
    history_centers, history_indexes = dist.conditional_centers(
        fit_history, fit_target, query_history, components=8
    )
    action_centers, action_indexes = dist.conditional_centers(
        fit_history,
        fit_target,
        query_history,
        fit_action_z=fit_action,
        query_action_z=query_action,
        components=8,
    )
    assert history_centers.shape == action_centers.shape == (2, 8, dist.TARGET_DIM)
    assert history_indexes.shape == action_indexes.shape == (2, 8)
    assert not np.array_equal(action_indexes[0], action_indexes[1])


def test_proper_scores_favor_centered_distribution() -> None:
    rng = np.random.default_rng(7)
    target = rng.normal(size=(5, dist.TARGET_DIM))
    centered = np.repeat(target[:, None, :], dist.MIXTURE_COMPONENTS, axis=1)
    shifted = centered + 3.0
    centered_scores, _ = dist.distribution_scores(centered, 0.65, target)
    shifted_scores, _ = dist.distribution_scores(shifted, 0.65, target)
    for metric in dist.PROPER_METRICS:
        assert float(centered_scores[metric].mean()) < float(shifted_scores[metric].mean())
    assert float(centered_scores["coverage80"].mean()) == 1.0


def test_final_gate_requires_every_score_control_and_coverage() -> None:
    base_error = np.linspace(1.0, 2.0, dist.FINAL_CLIPS)
    errors: dict[str, np.ndarray] = {}
    for metric in dist.PROPER_METRICS:
        errors[f"{metric}_aligned"] = base_error * 0.70
        for reference in dist.REFERENCES:
            if metric == "nll":
                errors[f"{metric}_{reference}"] = errors[f"{metric}_aligned"] + 0.10
            else:
                errors[f"{metric}_{reference}"] = base_error
    for arm in dist.ARMS:
        errors[f"coverage80_{arm}"] = np.full(dist.FINAL_CLIPS, 0.80)
        errors[f"coverage95_{arm}"] = np.full(dist.FINAL_CLIPS, 0.95)
    errors["final_motion_strata"] = np.arange(dist.FINAL_CLIPS) % 4
    final_target = np.ones((dist.FINAL_CLIPS, dist.TARGET_DIM), dtype=np.float32)
    metrics, gates = dist.analyze_scores(errors, final_target)
    assert len(metrics["effects"]) == 12
    assert gates["all_passed"] is True

    errors["crps_episode_shuffled"] = errors["crps_aligned"] * 0.99
    _, failed = dist.analyze_scores(errors, final_target)
    assert failed["crps:aligned_vs_episode_shuffled"]["passed"] is False
    assert failed["all_passed"] is False


def test_remaining_partition_and_shuffle_donors_are_disjoint() -> None:
    pool = []
    index = 0
    for stratum, count in enumerate((24, 24, 24, 23)):
        for rank in range(count):
            pool.append(
                {
                    "manifest_index": index,
                    "clip_id": f"{index:064x}",
                    "episode_dir": f"/e/{index}",
                    "motion_stratum": stratum,
                    "fresh_selection_hash": f"{rank:064x}",
                    "selected_fresh_score": rank < 16,
                }
            )
            index += 1
    final = dist._remaining_final_items({"split": {"untouched_pool": pool}})
    assert len(final) == dist.FINAL_CLIPS
    assert [sum(int(item["motion_stratum"] == s) for item in final) for s in range(4)] == [8, 8, 8, 7]
    by_index = {item["manifest_index"]: item for item in final}
    for item in final:
        donor = by_index[item["donor_manifest_index"]]
        assert donor["episode_dir"] != item["episode_dir"]
        assert donor["motion_stratum"] == item["motion_stratum"]


def test_mechanical_pca_hydration_is_exact_for_nonwhitened_transform() -> None:
    result = repair.synthetic_transform_equivalence()
    assert result["whiten"] is False
    assert result["passed"] is True
    assert result["hydrated_max_abs_difference"] <= 1e-12
    assert result["explicit_map_max_abs_difference"] <= 1e-12
