import json
from pathlib import Path

import pytest

from tools import build_vjepa2_clip_manifests as clips
from tools import vjepa2_frontier_lockbox as lockbox


def _write(path: Path, payload: bytes) -> dict:
    path.write_bytes(payload)
    return {
        "path": str(path),
        "sha256": lockbox.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_rows(path: Path, rows: list[dict]) -> dict:
    payload = b"".join(lockbox.canonical_json(row) + b"\n" for row in rows)
    return {**_write(path, payload), "entries": len(rows)}


@pytest.fixture
def population(tmp_path, monkeypatch):
    monkeypatch.setattr(clips, "inspect_episode", lambda _episode: 65)
    episodes = []
    for index in range(832):
        episode = tmp_path / "episodes" / f"episode_{index:04d}"
        episode.mkdir(parents=True)
        episodes.append(episode.resolve())
    source = tmp_path / "episodes.txt"
    source_record = _write(
        source,
        "".join(f"{episode}\n" for episode in episodes).encode(),
    )
    counts = {"train": 512, "validation": 64, "test": 128}
    physical = {"train": "train", "validation": "val", "test": "test"}
    cursor = 0
    splits = {}
    for name, count in counts.items():
        rows = [
            clips._clip_row(
                split=physical[name],
                episode=episode,
                start=0,
                auxiliary_index=index,
            )
            for index, episode in enumerate(episodes[cursor : cursor + count])
        ]
        cursor += count
        manifest = tmp_path / f"{name}.jsonl"
        splits[name] = {"clip_manifest": _write_rows(manifest, rows)}
    training_commit = "1" * 40
    study = lockbox.identity_payload(
        {
            "schema_version": 1,
            "kind": "vjepa2_controlled_video_diffusion_study",
            "study_root": str(tmp_path),
            "inputs": {
                "repository": {
                    "root": str(tmp_path),
                    "git_commit": training_commit,
                },
                "cache_build": {"episode_manifest": source_record},
                "splits": splits,
                "vjepa": {
                    "model_name": "vjepa2-vitg-fpc64-256",
                    "pca_stats": {"sha256": "2" * 64},
                    "checkpoint": {"sha256": "3" * 64, "bytes": 123},
                    "source": {"commit": "4" * 40},
                },
            },
        }
    )
    study_path = tmp_path / "study_manifest.json"
    study_path.write_text(json.dumps(study, sort_keys=True) + "\n")
    compatibility = {
        "training_commit_is_ancestor": True,
        "inference_critical_paths_unchanged": True,
        "paths": {},
    }
    return {
        "root": tmp_path,
        "episodes": episodes,
        "study": study,
        "study_path": study_path,
        "compatibility": compatibility,
        "training_commit": training_commit,
        "registration_commit": "5" * 40,
    }


def _registered_lockbox(population, label: str):
    root = population["root"]
    construction = lockbox.build_manifest(
        study_path=population["study_path"],
        output_dir=root / label,
        construction_commit=population["registration_commit"],
        inference_compatibility=population["compatibility"],
    )
    construction_path = root / label / "lockbox_construction.json"
    manifest_path = Path(construction["clip_manifest"]["path"])
    arrays = {}
    for name in ("target", "rgb", "actions"):
        path = root / label / f"{name}.npy"
        path.write_bytes(f"{label}-{name}".encode())
        arrays[name] = {
            "path": path,
            "sha256": lockbox.sha256_file(path),
        }
    metadata = {
        "format_version": 1,
        "artifact_type": "vjepa2.1-wan-grid-cache",
        "complete": True,
        "split": "test",
        "clip_count": 128,
        "clip_manifest": str(manifest_path),
        "clip_manifest_sha256": lockbox.sha256_file(manifest_path),
        "train_manifest_sha256": population["study"]["inputs"]["splits"][
            "train"
        ]["clip_manifest"]["sha256"],
        "pca_sha256": "2" * 64,
        "checkpoint_sha256": "3" * 64,
        "source_commit": "4" * 40,
        "model_name": "vjepa2-vitg-fpc64-256",
        "checkpoint_bytes": 123,
        "sample_size": 13,
        "chunk_size": 5,
        "action_span": 65,
        "frame_offsets": list(range(0, 65, 5)),
        "camera_order": ["top", "left_wrist", "right_wrist"],
        "cache_id": "6" * 64,
        "target_shape": [128, 64, 4, 24, 120],
        "target_dtype": "float16",
        "rgb_shape": [128, 13, 3, 180, 960],
        "rgb_dtype": "float16",
        "actions_shape": [128, 13, 5, 23],
        "actions_dtype": "float32",
    }
    for name, record in arrays.items():
        metadata[f"{name}_file"] = str(record["path"])
        metadata[f"{name}_sha256"] = record["sha256"]
    metadata_path = root / label / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n")
    registration = lockbox.build_registration(
        construction_path=construction_path,
        cache_metadata_path=metadata_path,
        study_path=population["study_path"],
        registration_commit=population["registration_commit"],
        attested_never_scored=True,
        inference_compatibility=population["compatibility"],
    )
    return registration, arrays


def test_constructs_deterministic_next_unused_episode_lockbox(population):
    first, audit = lockbox._derive_rows(
        population["study"], seed=lockbox.LOCKBOX_SEED
    )
    second, second_audit = lockbox._derive_rows(
        population["study"], seed=lockbox.LOCKBOX_SEED
    )
    original = set(population["episodes"][:704])

    assert first == second
    assert audit == second_audit
    assert len(first) == 128
    assert not original.intersection(
        Path(row["episode_dir"]) for row in first
    )
    assert audit["lockbox_original_episode_overlap_counts"] == {
        "train": 0,
        "validation": 0,
        "test": 0,
    }


def test_original_episode_overlap_is_rejected(population):
    study = population["study"]
    validation_path = Path(
        study["inputs"]["splits"]["validation"]["clip_manifest"]["path"]
    )
    validation_rows = [
        json.loads(line) for line in validation_path.read_text().splitlines()
    ]
    validation_rows[0] = clips._clip_row(
        split="val",
        episode=population["episodes"][0],
        start=0,
        auxiliary_index=0,
    )
    study["inputs"]["splits"]["validation"]["clip_manifest"] = _write_rows(
        validation_path, validation_rows
    )

    with pytest.raises(lockbox.LockboxError, match="episodes overlap"):
        lockbox.study_population(study)


def test_registration_requires_never_scored_attestation(population):
    construction = lockbox.build_manifest(
        study_path=population["study_path"],
        output_dir=population["root"] / "no_attestation",
        construction_commit=population["registration_commit"],
        inference_compatibility=population["compatibility"],
    )

    with pytest.raises(lockbox.LockboxError, match="never-scored attestation"):
        lockbox.build_registration(
            construction_path=population["root"]
            / "no_attestation"
            / "lockbox_construction.json",
            cache_metadata_path=Path(construction["clip_manifest"]["path"]),
            study_path=population["study_path"],
            registration_commit=population["registration_commit"],
            attested_never_scored=False,
            inference_compatibility=population["compatibility"],
        )


def test_manifest_and_cache_tampering_are_rejected(population):
    registration, arrays = _registered_lockbox(population, "tamper")
    Path(registration["manifest"]["path"]).write_bytes(b"tampered\n")
    with pytest.raises(lockbox.LockboxError, match="path/hash/bytes differ"):
        lockbox.validate_registration(
            registration,
            study=population["study"],
            rehash_arrays=True,
        )

    registration, arrays = _registered_lockbox(population, "array_tamper")
    arrays["rgb"]["path"].write_bytes(b"tampered-rgb")
    with pytest.raises(lockbox.LockboxError, match="rgb array SHA-256 differs"):
        lockbox.validate_registration(
            registration,
            study=population["study"],
            rehash_arrays=True,
        )


def test_self_hashed_forged_isolation_audit_is_recomputed(population):
    registration, _arrays = _registered_lockbox(population, "forged")
    forged = dict(registration)
    forged["episode_isolation"] = {
        **registration["episode_isolation"],
        "eligible_unused_episodes_examined": 999,
    }
    forged.pop("identity_sha256")
    forged = lockbox.identity_payload(forged)

    with pytest.raises(lockbox.LockboxError, match="deterministic next unused"):
        lockbox.validate_registration(
            forged,
            study=population["study"],
            rehash_arrays=False,
            verify_construction=True,
        )
