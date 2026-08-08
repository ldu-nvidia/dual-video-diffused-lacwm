"""Protocol tests for deployable LaMo validation and call accounting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from tools import lamo_motion_drift_evaluate as evaluation


def _row(clip_index: int, endpoint: evaluation.Endpoint) -> dict:
    sampling_id = evaluation.VALIDATION_SAMPLE_ID_OFFSET + clip_index
    donor = (
        None
        if endpoint.action_source == "matched"
        else evaluation.VALIDATION_SAMPLE_ID_OFFSET
        + evaluation._expected_action_donor_index(clip_index)
    )
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    hashes = {
        field: digest(f"{field}:{clip_index}")
        for field in (
            "cached_rgb_input_sha256",
            "sampler_history_rgb_sha256",
            "cached_actions_input_sha256",
            "video_clean_scoring_sha256",
            "raw_ground_truth_sha256",
            "raw_history_last_sha256",
            "video_initial_noise_sha256",
            "tf_initial_noise_sha256",
        )
    }
    hashes["sampler_actions_sha256"] = (
        digest(f"cached_actions_input_sha256:{clip_index}")
        if endpoint.action_source == "matched"
        else digest(
            "cached_actions_input_sha256:"
            f"{evaluation._expected_action_donor_index(clip_index)}"
        )
    )
    hashes["video_final_sha256"] = digest(f"video:{clip_index}:{endpoint.code}")
    hashes["decoded_final_sha256"] = digest(f"decoded:{clip_index}:{endpoint.code}")
    return evaluation.identity_payload(
        {
            "schema_version": evaluation.SCHEMA_VERSION,
            "kind": evaluation.KIND_ROW,
            "registration_identity_sha256": "r" * 64,
            "arm": asdict(evaluation.ARM_BY_CODE["VPM-CONT"]),
            "evaluation_split": "validation",
            "protected_test_accessed": False,
            "clip_index": clip_index,
            "clip_id": f"clip-{clip_index}",
            "sampling_id": sampling_id,
            "endpoint": asdict(endpoint),
            "action_donor_sampling_id": donor,
            "clean_future_rgb_passed_to_sampler": False,
            "clean_video_latent_passed_to_sampler": False,
            "clean_auxiliary_passed_to_sampler": False,
            "target_cache_array_opened": False,
            "online_feature_or_teacher_call_count": 0,
            "scoring_constructed_after_all_sampling": True,
            "history_rgb_frames": 5,
            "future_rgb_frames": 8,
            "history_video_latent_tokens": 2,
            "future_video_latent_tokens": 2,
            "actual_transformer_call_count": endpoint.nfe,
            "declared_nfe": endpoint.nfe,
            "metrics": {
                "video_future_nmse": 1.0,
                "video_future_temporal_delta_nmse": 1.0,
                "decoded_mse_unit_range": 1.0,
                "decoded_psnr_db": 1.0,
                "decoded_temporal_difference_mse_unit_range": 1.0,
            },
            "tensor_sha256": hashes,
        }
    )


def _registration() -> dict:
    return {
        "identity_sha256": "r" * 64,
        "validation_descriptors": [
            {"clip_id": f"clip-{index}"}
            for index in range(evaluation.EXPECTED_VALIDATION_CLIPS)
        ],
    }


def _all_rows() -> list[dict]:
    return [
        _row(index, endpoint)
        for index in range(evaluation.EXPECTED_VALIDATION_CLIPS)
        for endpoint in evaluation.ENDPOINTS
    ]


def test_endpoints_are_fixed_and_action_shuffle_is_outside_gate():
    assert evaluation.NFE_GRID == (1, 2, 4)
    assert [endpoint.code for endpoint in evaluation.ENDPOINTS] == [
        "autonomous_nfe_1",
        "autonomous_nfe_2",
        "autonomous_nfe_4",
        "actions_shuffled_nfe_1",
    ]
    assert all(endpoint.primary_gate for endpoint in evaluation.ENDPOINTS[:3])
    assert evaluation.ENDPOINTS[-1].primary_gate is False
    assert evaluation.ENDPOINTS[-1].nfe == 1
    protocol = evaluation.fixed_protocol()
    assert protocol["motion_drift"] == {
        "epsilon": 1e-6,
        "tau": 1,
        "history_rgb_frames": 5,
        "future_rgb_frames": 8,
        "history_wan_tokens": 2,
        "future_wan_tokens": 2,
        "valid_future_future_deltas": 1,
        "history_excluded": True,
        "predicted_clean_conversion": "x0_hat=x_sigma-sigma*v_theta",
        "schedule_weight": "global_mean((1-sigma)^2)",
    }
    assert protocol["future_or_clean_feature_allowed_at_sampling"] is False
    assert protocol["train_validation_clip_and_episode_disjoint"] is True


def test_validation_rows_require_no_future_input_exact_calls_and_noise_pairing():
    rows = _all_rows()
    evaluation._validate_rows(
        rows, evaluation.ARM_BY_CODE["VPM-CONT"], _registration()
    )

    changed = dict(rows[0])
    changed["clean_future_rgb_passed_to_sampler"] = True
    rows[0] = evaluation.identity_payload(
        {key: value for key, value in changed.items() if key != "identity_sha256"}
    )
    with pytest.raises(evaluation.MotionDriftEvaluationError, match="protocol"):
        evaluation._validate_rows(
            rows, evaluation.ARM_BY_CODE["VPM-CONT"], _registration()
        )


def test_validation_rows_reject_changed_initial_noise_within_arm():
    rows = _all_rows()
    row = dict(rows[1])
    tensor_hashes = dict(row["tensor_sha256"])
    tensor_hashes["video_initial_noise_sha256"] = hashlib.sha256(
        b"different"
    ).hexdigest()
    row["tensor_sha256"] = tensor_hashes
    rows[1] = evaluation.identity_payload(
        {key: value for key, value in row.items() if key != "identity_sha256"}
    )
    with pytest.raises(evaluation.MotionDriftEvaluationError, match="changed within arm"):
        evaluation._validate_rows(
            rows, evaluation.ARM_BY_CODE["VPM-CONT"], _registration()
        )


def test_cli_exposes_no_test_lockbox_or_target_cache_path():
    parser = evaluation._parser()
    for command in parser._subparsers._group_actions[0].choices.values():
        options = {
            option
            for action in command._actions
            for option in action.option_strings
        }
        assert not any(
            forbidden in option
            for forbidden in ("test", "lockbox", "target", "teacher", "feature")
            for option in options
        )


def test_rank_shards_cover_validation_exactly_once():
    indexes = [
        index
        for rank in range(evaluation.EXPECTED_WORLD_SIZE)
        for index in evaluation._expected_rank_indexes(rank)
    ]
    assert sorted(indexes) == list(range(evaluation.EXPECTED_VALIDATION_CLIPS))
    assert len(indexes) == len(set(indexes))


def test_training_dataset_returns_only_rgb_actions_mask_and_identity():
    pytest.importorskip("h5py")
    from robot_wm.datasets.abc.lamo_motion_drift_fixed_dataset import (
        ABCLamoMotionDriftDataset,
    )

    dataset = object.__new__(ABCLamoMotionDriftDataset)
    dataset.clips = [{}, {}]
    rgbs = np.zeros((2, 13, 3, 2, 4), dtype=np.float16)
    actions = np.zeros((2, 13, 5, 23), dtype=np.float32)
    dataset._open_rgbs = lambda: rgbs
    dataset._open_actions = lambda: actions

    sample = dataset._get_sample(1)

    assert set(sample) == {"rgb", "actions", "mask", "clip_index"}
    assert "auxiliary_target" not in sample
    assert sample["rgb"].dtype == torch.float32
    assert sample["actions"].dtype == torch.float32
    assert sample["clip_index"].item() == 1
    rgbs[1] = 1
    assert torch.count_nonzero(sample["rgb"]) == 0


def test_point_of_use_revalidation_detects_registered_input_mutation(tmp_path):
    paths = {}
    for name in ("parent", "train_manifest", "train_meta", "train_rgb", "train_actions"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = evaluation._file_record(path.resolve())
    registration = {
        "controlled_study": {"parent_snapshot": paths["parent"]},
        "training": {
            "manifest": paths["train_manifest"],
            "cache_metadata": paths["train_meta"],
            "rgb": paths["train_rgb"],
            "actions": paths["train_actions"],
        },
    }

    result = evaluation.revalidate_registered_inputs(
        registration,
        include_parent=True,
        include_train=True,
        include_validation=False,
    )
    assert result["all_selected_registered_inputs_rehashed"] is True
    assert result["auxiliary_target_array_opened"] is False

    (tmp_path / "train_actions").write_bytes(b"changed")
    with pytest.raises(evaluation.MotionDriftEvaluationError, match="changed"):
        evaluation.revalidate_registered_inputs(
            registration,
            include_parent=True,
            include_train=True,
            include_validation=False,
        )


def _manifest_row(index: int, split: str, episode: int) -> dict:
    start = 10 + index
    return {
        "action_span": 65,
        "auxiliary_index": index,
        "chunk_size": 5,
        "clip_id": hashlib.sha256(f"{split}:{index}".encode()).hexdigest(),
        "episode_dir": f"/immutable/{split}/episode-{episode}",
        "frame_indices": [start + 5 * offset for offset in range(13)],
        "sample_size": 13,
        "split": split,
        "start": start,
    }


def test_manifest_identity_and_split_disjointness_fail_closed(tmp_path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_rows = [_manifest_row(index, "train", index) for index in range(3)]
    val_rows = [_manifest_row(index, "val", index + 10) for index in range(2)]
    train_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in train_rows),
        encoding="utf-8",
    )
    val_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in val_rows),
        encoding="utf-8",
    )
    parsed_train = evaluation._manifest_descriptors(
        train_path, expected_split="train", expected_count=3
    )
    parsed_val = evaluation._manifest_descriptors(
        val_path, expected_split="val", expected_count=2
    )
    result = evaluation._split_disjointness(parsed_train, parsed_val)
    assert result["episode_disjoint"] is True
    assert result["clip_id_overlap_count"] == 0

    overlapping = [dict(row) for row in parsed_val]
    overlapping[0]["episode_dir"] = parsed_train[0]["episode_dir"]
    with pytest.raises(evaluation.MotionDriftEvaluationError, match="episode"):
        evaluation._split_disjointness(parsed_train, overlapping)
