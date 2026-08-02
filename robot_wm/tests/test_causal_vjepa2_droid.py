import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from robot_wm.datasets.droid import causal_vjepa2 as causal_targets
from robot_wm.datasets.droid.causal_vjepa2 import (
    CACHE_ARTIFACT_TYPE,
    FORMAT_VERSION,
    PCA_ARTIFACT_TYPE,
    PCA_CLIP_COUNT,
    PCA_RANK_PREFIX,
    POOLED_TOKEN_GRID,
    PRODUCTION_NUMERICAL_CONTRACT,
    TARGET_KIND,
    TARGET_SHAPE,
    TEACHER_SIZE,
    TOKENS_PER_CLIP,
    VJEPA2_CHECKPOINT_SHA256,
    VJEPA2_LICENSE_SHA256,
    VJEPA2_SOURCE_COMMIT,
    CausalVJEPA2DroidDataset,
    CausalVJEPA2Error,
    build_prefix_causal_clips,
    extract_causal_vjepa2_tokens,
    identity_sha256,
    manifest_order_sha256,
    pca_clip_rank,
    select_and_pool_last_tubelet,
    select_pca_rows,
    validate_frozen_base_record,
    validate_pca_artifact,
)
from robot_wm.datasets.droid.video_latent_forcing import make_clip_row, sha256_file
from robot_wm.modeling.dual_diffusion.vjepa2_target import (
    PCAWhiteningStats,
    VJEPA2_1_MODEL_NAME,
)
from tools import build_causal_vjepa2_droid as cache_builder
from tools.build_causal_vjepa2_droid import (
    REPO_ROOT,
    BuildError,
    DistributedContext,
    _cache_rank_expected,
    _fit_whitening,
    _rank_pca_shard_record,
    approved_artifact_path,
)


RUNTIME_FIXTURE = {
    "python": "3.fixture",
    "torch": "torch.fixture",
    "cuda": "cuda.fixture",
    "numpy": "numpy.fixture",
}


class _TinyPairEncoder(nn.Module):
    """Return dense tokens whose value depends on each local frame pair."""

    def forward(self, video, training=False):
        del training
        batch, _, frames, height, width = video.shape
        assert (frames, height, width) == (16, 6, 6)
        pair_value = video.reshape(batch, 3, 8, 2, 6, 6).mean((1, 3, 4, 5))
        grid = pair_value[:, :, None, None, None].expand(batch, 8, 3, 3, 4)
        return grid.reshape(batch, 8 * 3 * 3, 4)


def _rgb_frames(values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 1, 1).expand(
        1, 3, -1, 2, 2
    )


def test_prefixes_are_exact_native_length_and_end_in_required_pair():
    history = _rgb_frames([-0.9, -0.7, -0.5, -0.3, -0.1])
    future = _rgb_frames([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    prefixes = build_prefix_causal_clips(history, future)

    assert prefixes.shape == (1, 8, 3, 16, 2, 2)
    for future_index in range(8):
        expected = torch.cat(
            (
                history[:, :, :1].expand(-1, -1, 10 - future_index, -1, -1),
                history,
                future[:, :, : future_index + 1],
            ),
            dim=2,
        )
        torch.testing.assert_close(prefixes[:, future_index], expected, rtol=0, atol=0)
        if future_index == 0:
            torch.testing.assert_close(prefixes[:, future_index, :, -2], history[:, :, -1])
        else:
            torch.testing.assert_close(
                prefixes[:, future_index, :, -2], future[:, :, future_index - 1]
            )
        torch.testing.assert_close(
            prefixes[:, future_index, :, -1], future[:, :, future_index]
        )


def test_changing_later_future_cannot_change_earlier_causal_targets():
    history = _rgb_frames([-0.8, -0.6, -0.4, -0.2, 0.0])
    future = _rgb_frames([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    changed = future.clone()
    changed[:, :, 4:] = -0.95
    encoder = _TinyPairEncoder()

    baseline = extract_causal_vjepa2_tokens(
        history,
        future,
        encoder=encoder,
        encoder_dtype=torch.float32,
        teacher_size=(6, 6),
        patch_size=2,
        pool_kernel=(3, 3),
        expected_source_dim=4,
    )
    perturbed = extract_causal_vjepa2_tokens(
        history,
        changed,
        encoder=encoder,
        encoder_dtype=torch.float32,
        teacher_size=(6, 6),
        patch_size=2,
        pool_kernel=(3, 3),
        expected_source_dim=4,
    )

    torch.testing.assert_close(baseline[:, :4], perturbed[:, :4], rtol=0, atol=0)
    assert not torch.equal(baseline[:, 4:], perturbed[:, 4:])


def test_last_tubelet_selection_and_nonoverlapping_pool_are_exact():
    # Eight causal prefixes, eight temporal tokens, 3x3 spatial tokens.
    tokens = torch.zeros(8, 8 * 3 * 3, 2)
    grid = tokens.reshape(1, 8, 8, 3, 3, 2)
    for step in range(8):
        for temporal in range(8):
            grid[0, step, temporal, :, :, 0] = 100 * step + 10 * temporal
            grid[0, step, temporal, :, :, 1] = torch.arange(9).reshape(3, 3)

    pooled = select_and_pool_last_tubelet(
        tokens,
        batch_size=1,
        teacher_size=(6, 6),
        patch_size=2,
        pool_kernel=(3, 3),
        expected_source_dim=2,
    )

    assert pooled.shape == (1, 8, 1, 1, 2)
    torch.testing.assert_close(
        pooled[0, :, 0, 0, 0], torch.arange(8, dtype=torch.float32) * 100 + 70
    )
    torch.testing.assert_close(pooled[0, :, 0, 0, 1], torch.full((8,), 4.0))


def test_pca_selection_uses_frozen_exact_rank_and_all_256_clips():
    rows = [{"split": "train", "clip_id": f"clip-{index:04d}"} for index in range(300)]
    selected = select_pca_rows(rows)
    independent = sorted(
        enumerate(rows),
        key=lambda item: (
            hashlib.sha256(
                ("causal-vjepa2-pca-v1:" + item[1]["clip_id"]).encode()
            ).hexdigest(),
            item[1]["clip_id"],
            item[0],
        ),
    )[:256]

    assert selected == independent
    assert len(selected) == PCA_CLIP_COUNT
    assert pca_clip_rank("clip-0042") == hashlib.sha256(
        b"causal-vjepa2-pca-v1:clip-0042"
    ).hexdigest()
    assert PCA_CLIP_COUNT * TOKENS_PER_CLIP == 229_376


def test_exact_covariance_eigh_pca_is_reproducible_and_whitens():
    generator = torch.Generator().manual_seed(73)
    matrix = torch.randn(256, 9, generator=generator)
    first = _fit_whitening(matrix, channels=4)
    second = _fit_whitening(matrix.clone(), channels=4)

    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.components, second.components)
    assert torch.equal(first.eigenvalues, second.eigenvalues)
    projected = first.project(matrix)
    covariance = torch.cov(projected.T)
    torch.testing.assert_close(covariance, torch.eye(4), rtol=2e-4, atol=2e-4)


def test_eight_rank_assignments_are_disjoint_and_complete():
    cache_assignments = [
        set(_cache_rank_expected(rank, 8, 64_000)["assigned_indices"])
        for rank in range(8)
    ]
    assert set.union(*cache_assignments) == set(range(64_000))
    assert sum(map(len, cache_assignments)) == 64_000
    assert all(
        cache_assignments[left].isdisjoint(cache_assignments[right])
        for left in range(8)
        for right in range(left + 1, 8)
    )

    rows = [{"split": "train", "clip_id": f"clip-{index:04d}"} for index in range(300)]
    selected = select_pca_rows(rows)
    positions = [
        set(
            _rank_pca_shard_record(
                rank=rank, world_size=8, selected=selected
            )["selected_positions"]
        )
        for rank in range(8)
    ]
    assert set.union(*positions) == set(range(256))
    assert sum(map(len, positions)) == 256
    bound = _cache_rank_expected(
        3,
        8,
        64_000,
        cache_id="a" * 64,
        runtime=RUNTIME_FIXTURE,
        numerical_contract=PRODUCTION_NUMERICAL_CONTRACT,
    )
    assert bound["cache_id"] == "a" * 64
    assert bound["runtime"] == RUNTIME_FIXTURE
    assert bound["numerical_contract"] == PRODUCTION_NUMERICAL_CONTRACT


def test_large_artifact_policy_rejects_repo_and_root_disk(tmp_path):
    with pytest.raises(BuildError, match="inside the Git repo"):
        approved_artifact_path(REPO_ROOT / "forbidden-cache")
    with pytest.raises(BuildError, match="must be under"):
        approved_artifact_path(tmp_path / "root-disk-cache")
    assert approved_artifact_path("/mnt/data1/ldu/causal-vjepa-cache") == Path(
        "/mnt/data1/ldu/causal-vjepa-cache"
    )


def _write_base_manifest_and_cache(root: Path):
    row = make_clip_row(
        split="val", episode_index=12, trajectory_length=90, start=9
    )
    sample_path = root / "base" / "sample.npz"
    sample_path.parent.mkdir(parents=True)
    np.savez_compressed(
        sample_path,
        history_uint8=np.zeros((3, 5, 64, 112), dtype=np.uint8),
        future_uint8=np.ones((3, 8, 64, 112), dtype=np.uint8),
        actions=np.zeros((16, 7), dtype=np.float32),
    )
    row["cache_relpath"] = str(sample_path.relative_to(root))
    row["cache_sha256"] = sha256_file(sample_path)
    manifest = root / "val.jsonl"
    manifest.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return manifest, row


def _implementation_fixture():
    return {
        "repo_root": "/unused/repo",
        "repo_commit": "e" * 40,
        "builder_source": {
            "path": "/unused/builder.py",
            "sha256": "f" * 64,
            "bytes": 1,
        },
        "dataset_source": {
            "path": "/unused/dataset.py",
            "sha256": "1" * 64,
            "bytes": 1,
        },
    }


def _base_record_fixture(train_manifest="/unused/train.jsonl", val_manifest="/unused/val.jsonl"):
    return {
        "schema": "fixture-base-droid",
        "data_root": "/unused/data",
        "provenance": {"path": "/unused/provenance.json", "sha256": "2" * 64, "bytes": 1},
        "manifests": {
            "train": {"path": str(train_manifest)},
            "val": {"path": str(val_manifest)},
            "test": {"path": "/unused/test.identifiers.jsonl"},
        },
    }


def _frozen_base_record_fixture():
    manifests = {}
    names = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.identifiers.jsonl"}
    for split in ("train", "val", "test"):
        manifests[split] = {
            "path": f"/frozen/{names[split]}",
            "sha256": causal_targets.FROZEN_BASE_MANIFEST_SHA256[split],
            "bytes": 1,
            "clip_count": causal_targets.FROZEN_BASE_CLIP_COUNTS[split],
            "episode_count": causal_targets.FROZEN_BASE_EPISODE_COUNTS[split],
            "episode_ids_sha256": causal_targets.FROZEN_SPLIT_EPISODE_IDS_SHA256[split],
            "protected": split == "test",
            "cached": split != "test",
        }
    return {
        "schema": "frozen-droid-video-latent-forcing-poc-v1",
        "artifact_root": "/frozen",
        "data_root": causal_targets.FROZEN_DROID_DATA_ROOT,
        "provenance": {
            "path": "/frozen/provenance.json",
            "sha256": causal_targets.FROZEN_BASE_PROVENANCE_SHA256,
            "bytes": 1,
        },
        "manifests": manifests,
        "eligible_inventory_sha256": causal_targets.FROZEN_ELIGIBLE_INVENTORY_SHA256,
        "split_episode_ids_sha256": dict(
            causal_targets.FROZEN_SPLIT_EPISODE_IDS_SHA256
        ),
        "clip_counts": dict(causal_targets.FROZEN_BASE_CLIP_COUNTS),
        "episode_counts": dict(causal_targets.FROZEN_BASE_EPISODE_COUNTS),
        "split_disjoint": True,
        "protected_test_payload_cached": False,
    }


def _write_pca(
    path: Path,
    train_manifest_sha256: str,
    selection_rows=None,
    *,
    base_droid=None,
    runtime=None,
    implementation=None,
):
    if selection_rows is None:
        selection_rows = [
            {"split": "train", "clip_id": f"pca-{index:04d}"} for index in range(300)
        ]
    selected = select_pca_rows(selection_rows)
    stats = PCAWhiteningStats(
        mean=torch.zeros(768),
        components=torch.eye(768)[:48].contiguous(),
        eigenvalues=torch.ones(48),
    )
    payload = stats.to_payload()
    identity = {
        "schema": "droid-causal-vjepa2.1-v1",
        "artifact_type": PCA_ARTIFACT_TYPE,
        "source_commit": VJEPA2_SOURCE_COMMIT,
        "checkpoint_sha256": VJEPA2_CHECKPOINT_SHA256,
        "checkpoint_evidence": {
            "path": "/unused/checkpoint.pt",
            "sha256": VJEPA2_CHECKPOINT_SHA256,
        },
        "source_archive_sha256": "b" * 64,
        "source_license": {
            "path": "/unused/LICENSE",
            "sha256": VJEPA2_LICENSE_SHA256,
        },
        "teacher_size": list(TEACHER_SIZE),
        "pooled_token_grid": list(POOLED_TOKEN_GRID),
        "tokens_per_clip": TOKENS_PER_CLIP,
        "selected_clip_ids": [row["clip_id"] for _, row in selected],
        "selected_manifest_indices": [index for index, _ in selected],
        "selected_clip_ranks": [
            pca_clip_rank(row["clip_id"]) for _, row in selected
        ],
        "pca_rank_prefix": PCA_RANK_PREFIX,
        "sampled_token_count": PCA_CLIP_COUNT * TOKENS_PER_CLIP,
        "pca_clip_count": PCA_CLIP_COUNT,
        "pca_training_split_only": True,
        "test_rows_used": 0,
        "pca_algorithm": "exact-centered-covariance-eigh",
        "pca_covariance_dtype": "float32",
        "pca_tf32": False,
        "whitening_eps": 1e-6,
        "implementation": implementation or _implementation_fixture(),
        "base_droid": base_droid or _base_record_fixture(),
        "runtime": runtime or RUNTIME_FIXTURE,
        "numerical_contract": dict(PRODUCTION_NUMERICAL_CONTRACT),
        "train_manifest_sha256": train_manifest_sha256,
    }
    payload.update(identity)
    payload.update(
        {
            "artifact_identity": identity,
            "artifact_id": identity_sha256(identity),
            "model_name": VJEPA2_1_MODEL_NAME,
        }
    )
    torch.save(payload, path)


def _relax_artifact_validators(monkeypatch):
    monkeypatch.setattr(
        causal_targets,
        "_validate_implementation_record",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        causal_targets,
        "validate_frozen_base_record",
        lambda value, **kwargs: dict(value),
    )


def test_frozen_base_record_rejects_population_and_protected_cache_tampering():
    record = _frozen_base_record_fixture()
    validated = validate_frozen_base_record(record, verify_files=False)
    assert validated["clip_counts"] == {"train": 64_000, "val": 890, "test": 890}

    tampered = json.loads(json.dumps(record))
    tampered["manifests"]["val"]["episode_ids_sha256"] = "0" * 64
    with pytest.raises(CausalVJEPA2Error, match="identity mismatch"):
        validate_frozen_base_record(tampered, verify_files=False)

    tampered = json.loads(json.dumps(record))
    tampered["protected_test_payload_cached"] = True
    with pytest.raises(CausalVJEPA2Error, match="frozen POC population"):
        validate_frozen_base_record(tampered, verify_files=False)


def test_pca_validator_rejects_eps_and_explicit_identity_tampering(
    tmp_path, monkeypatch
):
    _relax_artifact_validators(monkeypatch)
    pca = tmp_path / "pca.pt"
    _write_pca(pca, "c" * 64)
    _, validated_payload, _ = validate_pca_artifact(pca)
    assert validated_payload["artifact_id"] == identity_sha256(
        validated_payload["artifact_identity"]
    )
    # Reload without mmap before overwriting the same test file.
    payload = torch.load(pca, map_location="cpu", weights_only=True)

    payload["eps"] = 1e-3
    torch.save(payload, pca)
    with pytest.raises(CausalVJEPA2Error, match="epsilon"):
        validate_pca_artifact(pca)

    payload["eps"] = 1e-6
    payload["artifact_id"] = "0" * 64
    torch.save(payload, pca)
    with pytest.raises(CausalVJEPA2Error, match="artifact_id"):
        validate_pca_artifact(pca)


def test_pca_identity_changes_with_runtime_and_numerical_contract(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("fixture\n", encoding="utf-8")
    rows = [{"split": "train", "clip_id": f"clip-{index:04d}"} for index in range(300)]
    selected = select_pca_rows(rows)
    kwargs = {
        "train_manifest": manifest,
        "rows": rows,
        "selected": selected,
        "provenance": {
            "source_commit": VJEPA2_SOURCE_COMMIT,
            "checkpoint_sha256": VJEPA2_CHECKPOINT_SHA256,
            "checkpoint_evidence": {"path": "/checkpoint", "sha256": VJEPA2_CHECKPOINT_SHA256},
            "source_archive_sha256": "b" * 64,
            "source_license": {"path": "/LICENSE", "sha256": VJEPA2_LICENSE_SHA256},
        },
        "implementation": _implementation_fixture(),
        "base_droid": _base_record_fixture(),
        "runtime": RUNTIME_FIXTURE,
        "numerical_contract": PRODUCTION_NUMERICAL_CONTRACT,
        "world_size": 8,
    }
    baseline = cache_builder._pca_identity(**kwargs)
    changed_runtime = {**kwargs, "runtime": {**RUNTIME_FIXTURE, "torch": "different"}}
    changed_numerics = {
        **kwargs,
        "numerical_contract": {
            **PRODUCTION_NUMERICAL_CONTRACT,
            "encoder_dtype": "float16",
        },
    }
    assert identity_sha256(baseline) != identity_sha256(
        cache_builder._pca_identity(**changed_runtime)
    )
    assert identity_sha256(baseline) != identity_sha256(
        cache_builder._pca_identity(**changed_numerics)
    )


def test_deployable_dataset_uses_split_metadata_and_read_only_memmap(
    tmp_path, monkeypatch
):
    _relax_artifact_validators(monkeypatch)
    manifest, row = _write_base_manifest_and_cache(tmp_path)
    semantic_root = tmp_path / "semantic"
    split_root = semantic_root / "val"
    split_root.mkdir(parents=True)
    pca = semantic_root / "pca.pt"
    train_manifest_sha = "c" * 64
    implementation = _implementation_fixture()
    base_droid = _base_record_fixture(val_manifest=manifest)
    _write_pca(
        pca,
        train_manifest_sha,
        base_droid=base_droid,
        implementation=implementation,
    )

    target_path = split_root / "targets.fp16.npy"
    target = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.float16, shape=(1, *TARGET_SHAPE)
    )
    target[...] = np.float16(0.25)
    target.flush()
    del target
    cache_identity = {
        "artifact_type": CACHE_ARTIFACT_TYPE,
        "split": "val",
        "clip_count": 1,
        "manifest_sha256": sha256_file(manifest),
        "manifest_order_sha256": manifest_order_sha256([row]),
        "source_commit": VJEPA2_SOURCE_COMMIT,
        "checkpoint_sha256": VJEPA2_CHECKPOINT_SHA256,
        "checkpoint_evidence": {
            "path": "/unused/checkpoint.pt",
            "sha256": VJEPA2_CHECKPOINT_SHA256,
        },
        "source_archive_sha256": "b" * 64,
        "source_license": {"path": "/unused/LICENSE", "sha256": VJEPA2_LICENSE_SHA256},
        "target_kind": TARGET_KIND,
        "auxiliary_target_shape": list(TARGET_SHAPE),
        "target_dtype": "float16",
        "target_shape": [1, *TARGET_SHAPE],
        "target_file": target_path.name,
        "target_sha256": sha256_file(target_path),
        "pca_file": str(pca),
        "pca_sha256": sha256_file(pca),
        "train_manifest_sha256": train_manifest_sha,
        "protected_test_access": False,
        "allowed_splits": ["train", "val"],
        "test_rows_extracted": 0,
        "implementation": implementation,
        "base_droid": base_droid,
        "runtime": RUNTIME_FIXTURE,
        "numerical_contract": dict(PRODUCTION_NUMERICAL_CONTRACT),
    }
    metadata = {
        "format_version": FORMAT_VERSION,
        **cache_identity,
        "artifact_identity": cache_identity,
        "cache_id": identity_sha256(cache_identity),
        "complete": True,
    }
    (split_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    dataset = CausalVJEPA2DroidDataset(manifest, tmp_path, semantic_root)
    assert dataset._targets.flags.writeable is False
    first = dataset[0]
    assert first["auxiliary_target"].shape == TARGET_SHAPE
    assert first["auxiliary_target"].dtype == torch.float16
    assert torch.all(first["auxiliary_target"] == 0.25)
    assert first["auxiliary_cache_id"] == identity_sha256(cache_identity)
    first["auxiliary_target"].zero_()
    assert torch.all(dataset[0]["auxiliary_target"] == 0.25)

    with target_path.open("r+b") as handle:
        handle.seek(-1, 2)
        final = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([final[0] ^ 1]))
    with pytest.raises(CausalVJEPA2Error, match="content hash"):
        CausalVJEPA2DroidDataset(manifest, tmp_path, semantic_root)


def test_pca_resume_recovers_orphan_artifact_without_companion(
    tmp_path, monkeypatch
):
    _relax_artifact_validators(monkeypatch)
    train_rows = [
        make_clip_row(
            split="train",
            episode_index=index,
            trajectory_length=90,
            start=index % 20,
        )
        for index in range(PCA_CLIP_COUNT)
    ]
    train_manifest = tmp_path / "train.jsonl"
    train_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    base_droid = _base_record_fixture(train_manifest=train_manifest)
    implementation = _implementation_fixture()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    provenance = {
        "model_name": VJEPA2_1_MODEL_NAME,
        "source_path": str(tmp_path / "source"),
        "source_commit": VJEPA2_SOURCE_COMMIT,
        "source_archive_sha256": "b" * 64,
        "source_license": {"path": "/unused/LICENSE", "sha256": VJEPA2_LICENSE_SHA256},
        "checkpoint_path": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": VJEPA2_CHECKPOINT_SHA256,
        "checkpoint_evidence": {
            "path": "/unused/checkpoint.pt",
            "sha256": VJEPA2_CHECKPOINT_SHA256,
        },
    }
    selected = select_pca_rows(train_rows)
    identity = cache_builder._pca_identity(
        train_manifest=train_manifest,
        rows=train_rows,
        selected=selected,
        provenance=provenance,
        implementation=implementation,
        base_droid=base_droid,
        runtime=RUNTIME_FIXTURE,
        numerical_contract=PRODUCTION_NUMERICAL_CONTRACT,
        world_size=1,
    )
    pca = tmp_path / "orphan.pt"
    _write_pca(
        pca,
        sha256_file(train_manifest),
        train_rows,
        base_droid=base_droid,
        implementation=implementation,
    )
    payload = torch.load(pca, map_location="cpu", weights_only=True)
    payload.update(identity)
    payload["artifact_identity"] = identity
    payload["artifact_id"] = identity_sha256(identity)
    torch.save(payload, pca)

    monkeypatch.setattr(
        cache_builder, "validate_teacher_inputs", lambda **kwargs: provenance
    )
    monkeypatch.setattr(
        cache_builder, "validate_builder_source", lambda: implementation
    )
    monkeypatch.setattr(
        cache_builder, "validate_frozen_base_droid", lambda **kwargs: base_droid
    )
    monkeypatch.setattr(
        cache_builder,
        "production_numerical_contract",
        lambda **kwargs: dict(PRODUCTION_NUMERICAL_CONTRACT),
    )
    monkeypatch.setattr(cache_builder, "runtime_record", lambda: dict(RUNTIME_FIXTURE))
    context = DistributedContext(0, 1, 0, torch.device("cpu"), False)
    result = cache_builder.fit_pca(
        train_manifest=train_manifest,
        data_root=tmp_path,
        source_path=tmp_path / "source",
        checkpoint_path=checkpoint,
        output_path=pca,
        context=context,
        encoder_dtype=torch.float32,
        batch_size=1,
        pca_device="cpu",
        resume=True,
    )
    companion = pca.with_suffix(".pt.metadata.json")
    assert companion.is_file()
    assert result["artifact_id"] == identity_sha256(identity)
    assert json.loads(companion.read_text())["artifact_sha256"] == sha256_file(pca)


def test_builder_atomically_publishes_and_resumes_exact_cache(
    tmp_path, monkeypatch
):
    _relax_artifact_validators(monkeypatch)
    val_manifest, _ = _write_base_manifest_and_cache(tmp_path)
    train_rows = [
        make_clip_row(
            split="train",
            episode_index=index,
            trajectory_length=90,
            start=index % 20,
        )
        for index in range(PCA_CLIP_COUNT)
    ]
    train_manifest = tmp_path / "train.jsonl"
    train_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    implementation = _implementation_fixture()
    base_droid = _base_record_fixture(
        train_manifest=train_manifest, val_manifest=val_manifest
    )
    pca = tmp_path / "pca.pt"
    _write_pca(
        pca,
        sha256_file(train_manifest),
        train_rows,
        base_droid=base_droid,
        implementation=implementation,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    provenance = {
        "model_name": VJEPA2_1_MODEL_NAME,
        "source_path": str(tmp_path / "source"),
        "source_commit": VJEPA2_SOURCE_COMMIT,
        "source_archive_sha256": "b" * 64,
        "source_license": {"path": "/unused/LICENSE", "sha256": VJEPA2_LICENSE_SHA256},
        "checkpoint_path": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": VJEPA2_CHECKPOINT_SHA256,
        "checkpoint_evidence": {
            "path": "/unused/checkpoint.pt",
            "sha256": VJEPA2_CHECKPOINT_SHA256,
        },
    }
    monkeypatch.setattr(
        cache_builder, "validate_teacher_inputs", lambda **kwargs: provenance
    )
    monkeypatch.setattr(
        cache_builder, "validate_builder_source", lambda: implementation
    )
    monkeypatch.setattr(
        cache_builder,
        "validate_frozen_base_droid",
        lambda **kwargs: base_droid,
    )
    monkeypatch.setattr(
        cache_builder,
        "production_numerical_contract",
        lambda **kwargs: dict(PRODUCTION_NUMERICAL_CONTRACT),
    )
    monkeypatch.setattr(cache_builder, "runtime_record", lambda: dict(RUNTIME_FIXTURE))
    monkeypatch.setattr(cache_builder, "_load_teacher", lambda *args, **kwargs: nn.Identity())

    def fake_target(history, future, **kwargs):
        del future, kwargs
        return torch.full(
            (history.shape[0], *TARGET_SHAPE), 0.125, dtype=torch.float16
        )

    monkeypatch.setattr(cache_builder, "extract_causal_vjepa2_target", fake_target)
    context = DistributedContext(0, 1, 0, torch.device("cpu"), False)
    semantic_root = tmp_path / "semantic-cache"
    result = cache_builder.extract_cache(
        manifest=val_manifest,
        train_manifest=train_manifest,
        data_root=tmp_path,
        source_path=tmp_path / "source",
        checkpoint_path=checkpoint,
        pca_path=pca,
        semantic_cache_root=semantic_root,
        context=context,
        encoder_dtype=torch.float32,
        batch_size=1,
        resume=False,
    )

    assert result["complete"] is True
    assert result["target_kind"] == TARGET_KIND
    assert result["test_rows_extracted"] == 0
    assert (semantic_root / "val" / "metadata.json").is_file()
    assert not (semantic_root / ".val.building").exists()
    targets = np.load(semantic_root / "val" / "targets.fp16.npy", mmap_mode="r")
    assert targets.shape == (1, *TARGET_SHAPE)
    assert np.all(targets == np.float16(0.125))

    # Reproduce the narrow crash after complete metadata but before the final
    # directory rename. Exact resume must validate and publish it.
    os.replace(semantic_root / "val", semantic_root / ".val.building")
    recovered = cache_builder.extract_cache(
        manifest=val_manifest,
        train_manifest=train_manifest,
        data_root=tmp_path,
        source_path=tmp_path / "source",
        checkpoint_path=checkpoint,
        pca_path=pca,
        semantic_cache_root=semantic_root,
        context=context,
        encoder_dtype=torch.float32,
        batch_size=1,
        resume=True,
    )
    assert recovered["cache_id"] == result["cache_id"]
    assert (semantic_root / "val" / "metadata.json").is_file()
    assert not (semantic_root / ".val.building").exists()
    resumed = cache_builder.extract_cache(
        manifest=val_manifest,
        train_manifest=train_manifest,
        data_root=tmp_path,
        source_path=tmp_path / "source",
        checkpoint_path=checkpoint,
        pca_path=pca,
        semantic_cache_root=semantic_root,
        context=context,
        encoder_dtype=torch.float32,
        batch_size=1,
        resume=True,
    )
    assert resumed["cache_id"] == result["cache_id"]
    with pytest.raises(BuildError, match="collision"):
        cache_builder.extract_cache(
            manifest=val_manifest,
            train_manifest=train_manifest,
            data_root=tmp_path,
            source_path=tmp_path / "source",
            checkpoint_path=checkpoint,
            pca_path=pca,
            semantic_cache_root=semantic_root,
            context=context,
            encoder_dtype=torch.float32,
            batch_size=1,
            resume=False,
        )


def test_production_cli_is_frozen_and_exposes_real_preflight_and_resume_smoke():
    parser = cache_builder._parser()
    preflight = parser.parse_args(
        [
            "preflight-teacher",
            "--source-path",
            "/source",
            "--checkpoint-path",
            "/checkpoint",
            "--train-manifest",
            "/train.jsonl",
            "--data-root",
            "/data",
            "--output",
            "/mnt/data1/preflight.json",
        ]
    )
    assert preflight.device == "cuda"
    smoke = parser.parse_args(
        [
            "smoke-mini-cache",
            "--source-path",
            "/source",
            "--checkpoint-path",
            "/checkpoint",
            "--train-manifest",
            "/train.jsonl",
            "--data-root",
            "/data",
            "--smoke-root",
            "/mnt/data1/smoke",
            "--mode",
            "resume",
        ]
    )
    assert smoke.rows == 16 and smoke.mode == "resume" and smoke.device == "cuda"
    smoke_prefix = [
        "smoke-mini-cache",
        "--source-path",
        "/source",
        "--checkpoint-path",
        "/checkpoint",
        "--train-manifest",
        "/train.jsonl",
        "--data-root",
        "/data",
        "--smoke-root",
        "/mnt/data1/smoke",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(smoke_prefix)
    with pytest.raises(SystemExit):
        parser.parse_args([*smoke_prefix, "--mode", "full"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "fit-pca",
                "--source-path",
                "/source",
                "--checkpoint-path",
                "/checkpoint",
                "--train-manifest",
                "/train.jsonl",
                "--data-root",
                "/data",
                "--output",
                "/mnt/data1/pca.pt",
                "--encoder-dtype",
                "float16",
            ]
        )
    with pytest.raises(BuildError, match="batch-size is frozen"):
        cache_builder.main(
            [
                "fit-pca",
                "--source-path",
                "/source",
                "--checkpoint-path",
                "/checkpoint",
                "--train-manifest",
                "/train.jsonl",
                "--data-root",
                "/data",
                "--output",
                "/mnt/data1/pca.pt",
                "--batch-size",
                "2",
            ]
        )


def test_smoke_only_metadata_is_rejected_by_production_cache_validator(tmp_path):
    manifest, _ = _write_base_manifest_and_cache(tmp_path)
    metadata = tmp_path / "smoke-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "artifact_type": cache_builder.SMOKE_CACHE_ARTIFACT_TYPE,
                "complete": True,
                "production_eligible": False,
                "synthetic_pca": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CausalVJEPA2Error, match="explicit artifact_identity"):
        causal_targets.validate_causal_cache(
            manifest_path=manifest,
            cache_metadata_path=metadata,
            expected_split="val",
        )


def test_smoke_cache_reference_partial_resume_is_byte_exact_and_pending_only(
    tmp_path, monkeypatch
):
    class _SyntheticDataset:
        def __getitem__(self, index):
            value = float(index) / 10.0
            return {
                "history": torch.full((3, 5, 2, 2), value),
                "future": torch.full((3, 8, 2, 2), value + 0.01),
            }

    calls = []

    def fake_target(history, future, **kwargs):
        del future, kwargs
        values = history[:, 0, 0, 0, 0]
        calls.extend(round(float(value) * 10) for value in values)
        return values[:, None, None, None, None].expand(
            -1, *TARGET_SHAPE
        ).to(torch.float16)

    monkeypatch.setattr(cache_builder, "extract_causal_vjepa2_target", fake_target)
    selected = [
        (index, {"clip_id": f"smoke-{index:02d}"}) for index in range(4)
    ]
    context = DistributedContext(0, 1, 0, torch.device("cpu"), False)
    stats = PCAWhiteningStats(
        mean=torch.zeros(768),
        components=torch.eye(768)[:48].contiguous(),
        eigenvalues=torch.ones(48),
    )
    identity = {
        "artifact_type": cache_builder.SMOKE_CACHE_ARTIFACT_TYPE,
        "selected_clip_ids": [row["clip_id"] for _, row in selected],
        "world_size": 1,
    }
    kwargs = {
        "smoke_root": tmp_path / "smoke",
        "identity": identity,
        "selected": selected,
        "dataset": _SyntheticDataset(),
        "encoder": nn.Identity(),
        "stats": stats,
        "context": context,
        "runtime": RUNTIME_FIXTURE,
        "numerical_contract": PRODUCTION_NUMERICAL_CONTRACT,
    }

    reference = cache_builder._run_smoke_cache_pass(
        label="reference", phase="fresh", **kwargs
    )
    assert calls == [0, 1, 2, 3]
    assert reference["complete"] is True

    calls.clear()
    partial = cache_builder._run_smoke_cache_pass(
        label="resumed", phase="partial", **kwargs
    )
    assert calls == [0]
    assert partial["status"] == "intentional_graceful_stop"
    partial_sidecar_path = tmp_path / "smoke" / ".resumed.building" / "rank-00000.json"
    partial_sidecar = json.loads(partial_sidecar_path.read_text())
    assert [record["index"] for record in partial_sidecar["completed_rows"]] == [0]
    completed_zero_hash = partial_sidecar["completed_rows"][0]["sha256"]

    calls.clear()
    resumed = cache_builder._run_smoke_cache_pass(
        label="resumed", phase="resume", **kwargs
    )
    assert calls == [1, 2, 3]
    assert resumed["byte_equivalent_to_reference"] is True
    reference_target = tmp_path / "smoke" / "reference" / "targets.fp16.npy"
    resumed_target = tmp_path / "smoke" / "resumed" / "targets.fp16.npy"
    assert reference_target.read_bytes() == resumed_target.read_bytes()
    final_sidecar = json.loads(
        (tmp_path / "smoke" / "resumed" / "rank-00000.json").read_text()
    )
    assert [record["index"] for record in final_sidecar["completed_rows"]] == [0, 1, 2, 3]
    assert final_sidecar["completed_rows"][0]["sha256"] == completed_zero_hash
    assert resumed["production_eligible"] is False
