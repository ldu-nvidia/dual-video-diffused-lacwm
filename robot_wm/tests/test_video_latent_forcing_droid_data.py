import json
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from robot_wm.datasets.droid import video_latent_forcing as data_module
from robot_wm.datasets.droid.video_latent_forcing import (
    CAMERA,
    SCHEMA,
    DroidVideoLatentForcingDataset,
    DroidVideoLatentForcingError,
    assign_episode_splits,
    clip_indices,
    future_to_lowres_scratchpad,
    make_clip_row,
    episode_rank,
    patchify_lowres_rgb,
    rows_episode_ids,
    sha256_file,
    unpatchify_lowres_rgb,
    valid_start_count,
)
from tools import build_video_latent_forcing_droid as cache_builder
from tools.build_video_latent_forcing_droid import (
    CACHE_FORMAT,
    REPO_ROOT,
    build_artifact,
    validate_artifact_output,
)


def _write_manifest(path: Path, row: dict) -> Path:
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_endpoint_contract_and_no_padding_masks():
    assert valid_start_count(66) == 42
    indices = clip_indices(start=41, trajectory_length=66)
    assert indices["history"] == (41, 43, 45, 47, 49)
    assert indices["future"] == (51, 53, 55, 57, 59, 61, 63, 65)
    assert indices["actions"] == tuple(range(49, 65))
    assert indices["actions"][-1] + 1 == indices["future"][-1]
    with pytest.raises(DroidVideoLatentForcingError, match="future endpoint"):
        clip_indices(start=42, trajectory_length=66)


def test_lowres_scratchpad_shape_and_exact_pixel_unshuffle_inverse():
    future = torch.linspace(-1, 1, 3 * 8 * 64 * 112).reshape(3, 8, 64, 112)
    scratchpad, lowres = future_to_lowres_scratchpad(future)
    assert scratchpad.shape == (48, 8, 8, 14)
    assert lowres.shape == (3, 8, 32, 56)
    torch.testing.assert_close(unpatchify_lowres_rgb(scratchpad), lowres, rtol=0, atol=0)
    torch.testing.assert_close(patchify_lowres_rgb(lowres), scratchpad, rtol=0, atol=0)
    expected = torch.nn.functional.interpolate(
        future.permute(1, 0, 2, 3),
        size=(32, 56),
        mode="area",
    ).permute(1, 0, 2, 3)
    torch.testing.assert_close(lowres, expected, rtol=0, atol=0)


def test_live_resize_is_canonical_uint8_roundtrip():
    frames = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(1, 9, 13, 3)
    resized = data_module._resize_rgb_uint8(frames)
    roundtrip = (
        resized.add(1.0).mul(127.5).round().clamp(0, 255).div(127.5).sub(1.0)
    )
    assert torch.equal(resized, roundtrip)


def test_episode_hash_split_is_deterministic_disjoint_and_exact():
    inventory = {index: 66 + index % 17 for index in range(30)}
    counts = {"train": 20, "val": 5, "test": 5}
    first = assign_episode_splits(inventory, counts=counts)
    second = assign_episode_splits(dict(reversed(list(inventory.items()))), counts=counts)
    assert first == second
    episode_sets = {split: {episode for episode, _ in rows} for split, rows in first.items()}
    assert len(episode_sets["train"]) == 20
    assert len(episode_sets["val"]) == 5
    assert len(episode_sets["test"]) == 5
    assert episode_sets["train"].isdisjoint(episode_sets["val"])
    assert episode_sets["train"].isdisjoint(episode_sets["test"])
    assert episode_sets["val"].isdisjoint(episode_sets["test"])


def test_episode_rank_matches_frozen_protocol_golden_values():
    assert episode_rank(0) == "746b5f7db01c175cc6085fa4883528b780c948635c59eafbd0de43f3b1d908b1"
    assert episode_rank(42) == "ceb6d3157848206af7253a74bf8e43f3d393adb603677f7da09bf89c163f1026"


def test_manifest_rejects_camera_mixing_and_index_tampering(tmp_path):
    row = make_clip_row(split="train", episode_index=3, trajectory_length=80, start=7)
    row["camera"] = "exterior_image_2_left"
    manifest = _write_manifest(tmp_path / "bad-camera.jsonl", row)
    with pytest.raises(DroidVideoLatentForcingError, match="cross-view"):
        DroidVideoLatentForcingDataset(manifest, tmp_path)

    row = make_clip_row(split="train", episode_index=3, trajectory_length=80, start=7)
    row["future_indices"][-1] -= 1
    manifest = _write_manifest(tmp_path / "bad-endpoint.jsonl", row)
    with pytest.raises(DroidVideoLatentForcingError, match="future_indices"):
        DroidVideoLatentForcingDataset(manifest, tmp_path)


def test_protected_test_is_fail_closed_and_cannot_reference_cache(tmp_path):
    row = make_clip_row(split="test", episode_index=3, trajectory_length=80, start=7)
    manifest = _write_manifest(tmp_path / "test.identifiers.jsonl", row)
    with pytest.raises(DroidVideoLatentForcingError, match="protected test access"):
        DroidVideoLatentForcingDataset(manifest, tmp_path)

    row["cache_relpath"] = "cache/forbidden.npz"
    row["cache_sha256"] = "0" * 64
    manifest = _write_manifest(tmp_path / "test-with-cache.jsonl", row)
    with pytest.raises(DroidVideoLatentForcingError, match="identifiers only"):
        DroidVideoLatentForcingDataset(
            manifest,
            tmp_path,
            allow_protected_test=True,
            protected_test_purpose="final registered evaluation",
        )


def test_cached_dataset_shapes_actions_masks_and_one_camera(tmp_path):
    row = make_clip_row(split="val", episode_index=12, trajectory_length=90, start=9)
    cache = tmp_path / "cache" / "val" / f"{row['clip_id']}.npz"
    cache.parent.mkdir(parents=True)
    history = np.arange(3 * 5 * 64 * 112, dtype=np.uint8).reshape(3, 5, 64, 112)
    future = np.arange(3 * 8 * 64 * 112, dtype=np.uint8).reshape(3, 8, 64, 112)
    actions = np.arange(16 * 7, dtype=np.float32).reshape(16, 7)
    np.savez_compressed(cache, history_uint8=history, future_uint8=future, actions=actions)
    row["cache_relpath"] = str(cache.relative_to(tmp_path))
    row["cache_sha256"] = sha256_file(cache)
    manifest = _write_manifest(tmp_path / "val.jsonl", row)

    dataset = DroidVideoLatentForcingDataset(manifest, tmp_path)
    sample = dataset[0]
    assert sample["history"].shape == (3, 5, 64, 112)
    assert sample["future"].shape == (3, 8, 64, 112)
    assert sample["actions"].shape == (16, 7)
    assert sample["lowres_scratchpad"].shape == (48, 8, 8, 14)
    assert sample["camera"] == CAMERA
    assert sample["history_mask"].tolist() == [True] * 5
    assert sample["future_mask"].tolist() == [True] * 8
    assert sample["action_mask"].tolist() == [True] * 16
    torch.testing.assert_close(sample["actions"], torch.from_numpy(actions))
    assert dataset._verified_cache_paths == {cache.resolve()}
    # Re-reading the immutable sample retains a single first-use verification.
    dataset[0]
    assert dataset._verified_cache_paths == {cache.resolve()}


def test_rows_episode_ids_supports_split_leakage_audit():
    train = [make_clip_row(split="train", episode_index=1, trajectory_length=70, start=0)]
    val = [make_clip_row(split="val", episode_index=2, trajectory_length=70, start=0)]
    test = [make_clip_row(split="test", episode_index=3, trajectory_length=70, start=0)]
    assert rows_episode_ids(train).isdisjoint(rows_episode_ids(val))
    assert rows_episode_ids(train).isdisjoint(rows_episode_ids(test))


def test_artifact_output_policy_refuses_repo_and_non_mnt_paths(tmp_path):
    with pytest.raises(DroidVideoLatentForcingError, match="Git repository"):
        validate_artifact_output(REPO_ROOT / "artifacts" / "forbidden")
    with pytest.raises(DroidVideoLatentForcingError, match="must be under"):
        validate_artifact_output(tmp_path / "forbidden")
    accepted = validate_artifact_output("/mnt/data1/ldu/test-video-latent-forcing-artifact")
    assert str(accepted).startswith("/mnt/data1/")


def test_manifest_artifact_is_atomic_hashed_and_protected_test_is_not_cached():
    with tempfile.TemporaryDirectory(dir=REPO_ROOT.parent) as temporary:
        output = Path(temporary) / "artifact"
        provenance = build_artifact(
            data_root=Path(temporary) / "source-identity-only",
            output_root=output,
            episode_lengths={index: 66 + index for index in range(8)},
            counts={"test": 2, "train": 4, "val": 2},
            seed=123,
            clips_per_episode={"train": 2, "val": 1, "test": 1},
        )
        assert provenance["complete"] is True
        assert provenance["camera_count"] == 1
        assert provenance["manifests"]["train"]["episode_count"] == 4
        assert provenance["manifests"]["test"]["protected"] is True
        assert provenance["manifests"]["test"]["cached"] is False
        assert provenance["clips_per_episode"] == {"train": 2, "val": 1, "test": 1}
        assert provenance["cache"]["format"] == CACHE_FORMAT
        assert (output / "provenance.json").is_file()
        train_rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
        val_rows = [json.loads(line) for line in (output / "val.jsonl").read_text().splitlines()]
        test_rows = [
            json.loads(line)
            for line in (output / "test.identifiers.jsonl").read_text().splitlines()
        ]
        assert rows_episode_ids(train_rows).isdisjoint(rows_episode_ids(val_rows))
        assert rows_episode_ids(train_rows).isdisjoint(rows_episode_ids(test_rows))
        assert all(row["protected"] for row in test_rows)
        assert all("cache_relpath" not in row for row in test_rows)
        for split, record in provenance["manifests"].items():
            assert sha256_file(output / record["path"]) == record["sha256"], split

        with pytest.raises(DroidVideoLatentForcingError, match="already exists"):
            build_artifact(
                data_root=Path(temporary) / "source-identity-only",
                output_root=output,
                episode_lengths={index: 66 + index for index in range(8)},
                counts={"train": 4, "val": 2, "test": 2},
            )


def test_cache_builder_decodes_and_reads_actions_once_per_episode(tmp_path, monkeypatch):
    episode_index = 12
    source_root = tmp_path / "source"
    parquet, video = cache_builder.droid_paths(source_root, episode_index)
    parquet.parent.mkdir(parents=True)
    video.parent.mkdir(parents=True)
    parquet.touch()
    video.touch()
    rows = [
        make_clip_row(
            split="train",
            episode_index=episode_index,
            trajectory_length=100,
            start=start,
        )
        for start in (0, 3, 7, 11, 15, 19, 23, 27)
    ]
    calls = {"decode": 0, "actions": 0}

    def fake_decode(path, indices):
        calls["decode"] += 1
        assert path == video.resolve()
        return np.stack(
            [np.full((64, 112, 3), index % 256, dtype=np.uint8) for index in indices]
        )

    def fake_actions(path, indices):
        calls["actions"] += 1
        assert path == parquet.resolve()
        return torch.tensor([[float(index)] * 7 for index in indices], dtype=torch.float32)

    monkeypatch.setattr(cache_builder, "_decode_cache_frames", fake_decode)
    monkeypatch.setattr(cache_builder, "_read_native_actions", fake_actions)
    cached = cache_builder._cache_rows_by_episode(
        rows=rows,
        source_root=source_root,
        stage_root=tmp_path / "stage",
        cache_workers=1,
    )

    assert calls == {"decode": 1, "actions": 1}
    assert [row["clip_id"] for row in cached] == [row["clip_id"] for row in rows]
    for original, row in zip(rows, cached, strict=True):
        assert row["cache_format"] == CACHE_FORMAT
        cache_path = tmp_path / "stage" / row["cache_relpath"]
        assert sha256_file(cache_path) == row["cache_sha256"]
        with np.load(cache_path, allow_pickle=False) as payload:
            assert payload["history_uint8"].shape == (3, 5, 64, 112)
            assert payload["future_uint8"].shape == (3, 8, 64, 112)
            assert payload["actions"].shape == (16, 7)
            assert payload["actions"][-1, 0] == original["action_indices"][-1]


def test_parallel_cache_preserves_interleaved_manifest_order(tmp_path, monkeypatch):
    episode_ids = (12, 1012)
    source_root = tmp_path / "source"
    paths = {}
    for episode_index in episode_ids:
        parquet, video = cache_builder.droid_paths(source_root, episode_index)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True, exist_ok=True)
        parquet.touch()
        video.touch()
        paths[episode_index] = (parquet.resolve(), video.resolve())
    rows = [
        make_clip_row(
            split="train",
            episode_index=episode_index,
            trajectory_length=100,
            start=start,
        )
        for episode_index, start in ((12, 0), (1012, 2), (12, 5), (1012, 7))
    ]
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    decode_calls = {episode_index: 0 for episode_index in episode_ids}
    action_calls = {episode_index: 0 for episode_index in episode_ids}
    worker_threads: set[int] = set()

    def episode_for_path(path, position):
        return next(
            episode_index
            for episode_index, episode_paths in paths.items()
            if episode_paths[position] == path
        )

    def fake_decode(path, indices):
        episode_index = episode_for_path(path, 1)
        with lock:
            decode_calls[episode_index] += 1
            worker_threads.add(threading.get_ident())
        barrier.wait(timeout=5)
        return np.stack(
            [np.full((64, 112, 3), index % 256, dtype=np.uint8) for index in indices]
        )

    def fake_actions(path, indices):
        episode_index = episode_for_path(path, 0)
        with lock:
            action_calls[episode_index] += 1
        return torch.tensor([[float(index)] * 7 for index in indices], dtype=torch.float32)

    monkeypatch.setattr(cache_builder, "_decode_cache_frames", fake_decode)
    monkeypatch.setattr(cache_builder, "_read_native_actions", fake_actions)
    cached = cache_builder._cache_rows_by_episode(
        rows=rows,
        source_root=source_root,
        stage_root=tmp_path / "stage",
        cache_workers=2,
    )

    assert len(worker_threads) == 2
    assert decode_calls == {12: 1, 1012: 1}
    assert action_calls == {12: 1, 1012: 1}
    assert [row["clip_id"] for row in cached] == [row["clip_id"] for row in rows]
    assert all((tmp_path / "stage" / row["cache_relpath"]).is_file() for row in cached)


def test_parallel_cache_error_leaves_no_published_or_staged_artifact(monkeypatch):
    with tempfile.TemporaryDirectory(dir=REPO_ROOT.parent) as temporary:
        parent = Path(temporary)
        output = parent / "parallel-cache-artifact"

        def fail_episode(**kwargs):
            episode_index = int(kwargs["rows"][0]["episode_index"])
            raise RuntimeError(f"mock cache failure for episode {episode_index}")

        monkeypatch.setattr(cache_builder, "_cache_episode_rows", fail_episode)
        with pytest.raises(RuntimeError, match="mock cache failure"):
            build_artifact(
                data_root=parent / "source",
                output_root=output,
                episode_lengths={12: 100, 1012: 100},
                counts={"train": 2, "val": 0, "test": 0},
                clips_per_episode={"train": 2, "val": 1, "test": 1},
                cache_splits=("train",),
                cache_workers=2,
            )

        assert not output.exists()
        assert list(parent.glob(f".{output.name}.*")) == []


def test_manifest_identity_fixes_camera_and_schema():
    row = make_clip_row(split="train", episode_index=42, trajectory_length=100, start=11)
    assert row["schema"] == SCHEMA
    assert row["camera"] == CAMERA
    assert row["protected"] is False
