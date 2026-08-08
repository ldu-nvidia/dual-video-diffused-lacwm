"""Protocol tests for deployable two-clock consistency validation and call accounting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from tools import two_clock_consistency_evaluate as evaluation


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _row(clip_index: int, endpoint: evaluation.Endpoint) -> dict:
    sampling_id = evaluation.VALIDATION_SAMPLE_ID_OFFSET + clip_index
    donor = (
        None
        if endpoint.action_source == "matched"
        else evaluation.VALIDATION_SAMPLE_ID_OFFSET
        + evaluation._expected_action_donor_index(clip_index)
    )
    donor_index = evaluation._expected_action_donor_index(clip_index)
    hashes = {
        "cached_rgb_input_sha256": _hash(f"rgb:{clip_index}"),
        "sampler_history_rgb_sha256": _hash(f"history:{clip_index}"),
        "cached_actions_input_sha256": _hash(f"actions:{clip_index}"),
        "video_clean_scoring_sha256": _hash(f"clean:{clip_index}"),
        "raw_ground_truth_sha256": _hash(f"gt:{clip_index}"),
        "raw_history_last_sha256": _hash(f"last:{clip_index}"),
        "video_initial_noise_sha256": _hash(f"video-noise:{clip_index}"),
        "tf_initial_noise_sha256": _hash(f"tf-noise:{clip_index}"),
    }
    hashes["sampler_actions_sha256"] = (
        _hash(f"actions:{clip_index}")
        if endpoint.action_source == "matched"
        else _hash(f"actions:{donor_index}")
    )
    hashes["video_final_sha256"] = _hash(
        f"video:{clip_index}:{endpoint.code}"
    )
    hashes["decoded_final_sha256"] = _hash(
        f"decoded:{clip_index}:{endpoint.code}"
    )
    return evaluation.identity_payload(
        {
            "schema_version": evaluation.SCHEMA_VERSION,
            "kind": evaluation.KIND_ROW,
            "registration_identity_sha256": "r" * 64,
            "arm": asdict(evaluation.ARM_BY_CODE["TC-CONT"]),
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


def test_validation_rows_require_no_future_input_exact_calls_and_noise_pairing():
    rows = _all_rows()
    evaluation._validate_rows(
        rows, evaluation.ARM_BY_CODE["TC-CONT"], _registration()
    )

    changed = dict(rows[0])
    changed["clean_future_rgb_passed_to_sampler"] = True
    rows[0] = evaluation.identity_payload(
        {key: value for key, value in changed.items() if key != "identity_sha256"}
    )
    with pytest.raises(evaluation.TwoClockConsistencyEvaluationError, match="protocol"):
        evaluation._validate_rows(
            rows, evaluation.ARM_BY_CODE["TC-CONT"], _registration()
        )


def test_validation_rows_reject_changed_initial_noise_within_arm():
    rows = _all_rows()
    row = dict(rows[1])
    tensor_hashes = dict(row["tensor_sha256"])
    tensor_hashes["video_initial_noise_sha256"] = _hash("different")
    row["tensor_sha256"] = tensor_hashes
    rows[1] = evaluation.identity_payload(
        {key: value for key, value in row.items() if key != "identity_sha256"}
    )
    with pytest.raises(evaluation.TwoClockConsistencyEvaluationError, match="changed within arm"):
        evaluation._validate_rows(
            rows, evaluation.ARM_BY_CODE["TC-CONT"], _registration()
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


def test_manifest_parser_requires_dense_unique_episode_clips(tmp_path):
    rows = []
    for index in range(2):
        rows.append(
            {
                "action_span": 65,
                "auxiliary_index": index,
                "chunk_size": 5,
                "clip_id": f"{index + 1:064x}",
                "episode_dir": f"/immutable/episode-{index}",
                "frame_indices": list(range(10 + index, 75 + index, 5)),
                "sample_size": 13,
                "split": "train",
                "start": 10 + index,
            }
        )
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert len(
        evaluation._manifest_descriptors(
            path, expected_split="train", expected_count=2
        )
    ) == 2

    rows[1]["episode_dir"] = rows[0]["episode_dir"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(evaluation.TwoClockConsistencyEvaluationError, match="row 1"):
        evaluation._manifest_descriptors(
            path, expected_split="train", expected_count=2
        )


def test_rank_receipt_binds_exact_shard_grid_calls_and_rows(tmp_path):
    rank = 0
    arm = evaluation.ARM_BY_CODE["TC-CONT"]
    row_path = tmp_path / "rank_000.jsonl"
    row_bytes = b"{}\n" * (
        len(evaluation._expected_rank_indexes(rank)) * len(evaluation.ENDPOINTS)
    )
    row_path.write_bytes(row_bytes)
    receipt = evaluation.identity_payload(
        {
            "schema_version": evaluation.SCHEMA_VERSION,
            "kind": evaluation.KIND_RANK,
            "registration_identity_sha256": "r" * 64,
            "arm": asdict(arm),
            "rank": rank,
            "world_size": evaluation.EXPECTED_WORLD_SIZE,
            "batch_size_per_rank": evaluation.EXPECTED_BATCH_SIZE_PER_RANK,
            "assigned_clip_indexes": evaluation._expected_rank_indexes(rank),
            "endpoints": [asdict(endpoint) for endpoint in evaluation.ENDPOINTS],
            "actual_transformer_call_count": (
                evaluation._expected_rank_transformer_calls()
            ),
            "rows": {
                "path": str(row_path),
                "bytes": len(row_bytes),
                "sha256": hashlib.sha256(row_bytes).hexdigest(),
                "count": (
                    len(evaluation._expected_rank_indexes(rank))
                    * len(evaluation.ENDPOINTS)
                ),
            },
            "protected_test_accessed": False,
            "target_cache_array_opened": False,
        }
    )

    assert evaluation._validate_rank_manifest(
        receipt,
        expected_rank=rank,
        arm=arm,
        registration={"identity_sha256": "r" * 64},
        output_dir=tmp_path,
    ) == row_path

    changed = dict(receipt)
    changed["actual_transformer_call_count"] += 1
    changed = evaluation.identity_payload(
        {key: value for key, value in changed.items() if key != "identity_sha256"}
    )
    with pytest.raises(evaluation.TwoClockConsistencyEvaluationError, match="rank"):
        evaluation._validate_rank_manifest(
            changed,
            expected_rank=rank,
            arm=arm,
            registration={"identity_sha256": "r" * 64},
            output_dir=tmp_path,
        )


def test_training_dataset_returns_only_rgb_actions_mask_and_identity():
    pytest.importorskip("h5py")
    from robot_wm.datasets.abc.two_clock_consistency_fixed_dataset import (
        ABCTwoClockConsistencyDataset,
    )

    dataset = object.__new__(ABCTwoClockConsistencyDataset)
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
    with pytest.raises(evaluation.TwoClockConsistencyEvaluationError, match="changed"):
        evaluation.revalidate_registered_inputs(
            registration,
            include_parent=True,
            include_train=True,
            include_validation=False,
        )
