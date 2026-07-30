import copy

import pytest
import torch

from tools import evaluate_vjepa2_quality as quality


def test_preregistered_quality_grids():
    intermediate = quality.quality_grid(800)
    assert len(intermediate) == 8
    assert set(intermediate) == {
        ("autonomous", 4),
        ("autonomous", 8),
        ("off", 4),
        ("off", 8),
        ("autonomous_shuffled", 4),
        ("autonomous_shuffled", 8),
        ("oracle_matched", 4),
        ("oracle_shuffled", 4),
    }
    final = quality.quality_grid(1000)
    assert len(final) == 35
    assert set(final) == {
        (source, nfe)
        for source in quality.SOURCES
        for nfe in quality.NFE_STEPS
    }
    with pytest.raises(quality.QualityEvaluationError):
        quality.quality_grid(600)


def test_rank_partitions_cover_all_clips_once():
    partitions = [
        quality._expected_rank_indexes(rank, quality.EXPECTED_WORLD_SIZE)
        for rank in range(quality.EXPECTED_WORLD_SIZE)
    ]
    assert all(len(indexes) == 16 for indexes in partitions)
    assert sorted(index for indexes in partitions for index in indexes) == list(
        range(quality.EXPECTED_TEST_CLIPS)
    )


def test_per_sample_reconstruction_metrics():
    target = torch.ones(2, 1, 3, 1, 1)
    prediction = target.clone()
    prediction[0, :, 0] = 10
    prediction[1, :, 1:] = -1
    assert quality._per_sample_nmse(
        prediction, target, history=1, key="video"
    ) == pytest.approx([0.0, 4.0])
    assert quality._per_sample_cosine(
        prediction, target, history=1, key="video"
    ) == pytest.approx([1.0, -1.0])

    decoded_target = torch.zeros(2, 3, 2, 1, 1, dtype=torch.uint8)
    decoded_prediction = decoded_target.clone()
    decoded_prediction[1, :, 1] = 255
    metrics = quality._per_sample_decoded(
        decoded_prediction, decoded_target
    )
    assert metrics["decoded_mse_unit_range"] == pytest.approx([0.0, 0.5])
    assert metrics[
        "decoded_temporal_difference_mse_unit_range"
    ] == pytest.approx([0.0, 1.0])
    history = torch.zeros(2, 3, 1, 1, 1, dtype=torch.uint8)
    boundary_metrics = quality._per_sample_decoded(
        decoded_prediction,
        decoded_target,
        prediction_history=history,
        target_history=history,
    )
    # The history→first-future transition is included in the denominator.
    assert boundary_metrics[
        "decoded_temporal_difference_mse_unit_range"
    ] == pytest.approx([0.0, 0.5])


def _row(clip_index, source, nfe, *, output_suffix="same"):
    digest = lambda text: __import__("hashlib").sha256(text.encode()).hexdigest()
    tensor_hashes = {
        "video_clean_sha256": digest(f"video-clean-{clip_index}"),
        "auxiliary_clean_sha256": digest(f"aux-clean-{clip_index}"),
        "ground_truth_sha256": digest(f"gt-{clip_index}"),
        "vae_ground_truth_sha256": digest(f"vae-gt-{clip_index}"),
        "raw_history_last_sha256": digest(f"raw-history-{clip_index}"),
        "vae_history_last_sha256": digest(f"vae-history-{clip_index}"),
        "cached_rgb_input_sha256": digest(f"rgb-input-{clip_index}"),
        "cached_actions_input_sha256": digest(
            f"actions-input-{clip_index}"
        ),
        "video_initial_state_sha256": digest(f"video-noise-{clip_index}"),
        "auxiliary_initial_state_sha256": digest(f"aux-noise-{clip_index}"),
        "auxiliary_initial_noise_sha256": digest(f"aux-noise-{clip_index}"),
        "video_final_sha256": digest(
            f"video-final-{clip_index}-{nfe}-{output_suffix}"
        ),
        "auxiliary_final_sha256": digest(
            f"aux-final-{clip_index}-{nfe}-{output_suffix}"
        ),
        "decoded_final_sha256": digest(
            f"decoded-final-{clip_index}-{nfe}-{output_suffix}"
        ),
    }
    return quality._identity_payload(
        {
            "schema_version": 1,
            "kind": "vjepa2_controlled_study_quality_clip",
            "arm_code": "VPM",
            "completed_updates": 800,
            "clip_index": clip_index,
            "clip_id": f"clip-{clip_index:03d}",
            "source": source,
            "oracle_leakage": source in quality.ORACLE_SOURCES,
            "deployable_evidence": source not in quality.ORACLE_SOURCES,
            "sampler_entrypoint": (
                "DualExplicitActionDiTModel._sample_future"
                if source in quality.ORACLE_SOURCES
                else "DualExplicitActionDiTModel.sample_future_deployable"
            ),
            "clean_future_or_auxiliary_passed_to_sampler": (
                source in quality.ORACLE_SOURCES
            ),
            "nfe": nfe,
            "video_history_latent_frames": 1,
            "auxiliary_history_latent_frames": 0,
            "online_teacher_call_count": 0,
            "actual_wan_call_count": nfe,
            "effective_state_gate": 0.0,
            "effective_clock_gate": 0.0,
            "metrics": {
                "video_future_nmse": 1.0,
                "auxiliary_future_nmse": 1.0,
                "auxiliary_future_cosine_similarity": 0.0,
                "decoded_mse_unit_range": 1.0,
                "decoded_psnr_db": 0.0,
                "decoded_temporal_difference_mse_unit_range": 1.0,
            },
            "diagnostic_metrics": {
                "prediction_vs_vae_reconstruction_mse_unit_range": 1.0,
                "prediction_vs_vae_reconstruction_psnr_db": 0.0,
                "prediction_vs_vae_reconstruction_temporal_mse_unit_range": 1.0,
                "vae_reconstruction_vs_raw_mse_unit_range": 1.0,
                "vae_reconstruction_vs_raw_psnr_db": 0.0,
                "vae_reconstruction_vs_raw_temporal_mse_unit_range": 1.0,
            },
            "tensor_sha256": tensor_hashes,
            "perceptual_metric": {
                "available": False,
                "reason": "unpinned downloads forbidden",
            },
        }
    )


def test_global_inventory_enforces_vpm_source_no_op():
    grid = quality.intermediate_grid()
    descriptors = [
        {"clip_id": f"clip-{index:03d}"}
        for index in range(quality.EXPECTED_TEST_CLIPS)
    ]
    rows = [
        _row(index, source, nfe)
        for index in range(quality.EXPECTED_TEST_CLIPS)
        for source, nfe in grid
    ]
    validation = quality._validate_global_rows(
        rows,
        arm="VPM",
        completed_updates=800,
        grid=grid,
        descriptors=descriptors,
    )
    assert validation["record_count"] == 128 * 8
    assert validation["causal_no_op_identities_passed"]

    broken = copy.deepcopy(rows)
    candidate = next(
        row
        for row in broken
        if row["clip_index"] == 0
        and row["source"] == "off"
        and row["nfe"] == 4
    )
    candidate["tensor_sha256"]["decoded_final_sha256"] = "f" * 64
    candidate.pop("identity_sha256")
    candidate.update(quality._identity_payload(candidate))
    with pytest.raises(
        quality.QualityEvaluationError, match="causal source no-op"
    ):
        quality._validate_global_rows(
            broken,
            arm="VPM",
            completed_updates=800,
            grid=grid,
            descriptors=descriptors,
        )
