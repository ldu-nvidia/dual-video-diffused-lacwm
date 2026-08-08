from __future__ import annotations

from pathlib import Path

import pytest

from tools import causal_vjepa2_cache_bridge as bridge
from tools import video_latent_forcing_poc as vlf


def _record(path: Path) -> dict[str, object]:
    return vlf.file_record(path)


def _producer_metadata(tmp_path: Path) -> tuple[dict, Path]:
    root = tmp_path / "producer"
    (root / ".git").mkdir(parents=True)
    dataset = root / "robot_wm" / "datasets" / "droid" / "causal_vjepa2.py"
    base = root / "robot_wm" / "datasets" / "droid" / "video_latent_forcing.py"
    builder = root / "tools" / "build_causal_vjepa2_droid.py"
    dataset.parent.mkdir(parents=True)
    builder.parent.mkdir(parents=True)
    dataset.write_text("# exact producer dataset\n", encoding="utf-8")
    builder.write_text("# exact producer builder\n", encoding="utf-8")
    active_base = Path(
        bridge.inspect.getsourcefile(bridge.DroidVideoLatentForcingDataset)
    ).resolve()
    base.write_bytes(active_base.read_bytes())
    return {
        "implementation": {
            "repo_commit": bridge.FROZEN_CACHE_PRODUCER_COMMIT,
            "repo_root": str(root),
            "dataset_source": _record(dataset),
            "builder_source": _record(builder),
        }
    }, root


def test_recorded_producer_requires_exact_clean_commit(tmp_path, monkeypatch):
    metadata, root = _producer_metadata(tmp_path)

    def clean_git(repo, *arguments):
        assert repo == root.resolve()
        return (
            bridge.FROZEN_CACHE_PRODUCER_COMMIT
            if arguments == ("rev-parse", "HEAD")
            else ""
        )

    monkeypatch.setattr(bridge, "_git_output", clean_git)
    result = bridge._validate_recorded_producer(
        metadata,
        expected_commit=bridge.FROZEN_CACHE_PRODUCER_COMMIT,
    )
    assert result["repo_commit"] == bridge.FROZEN_CACHE_PRODUCER_COMMIT
    assert result["dataset_source"] == metadata["implementation"]["dataset_source"]

    monkeypatch.setattr(
        bridge,
        "_git_output",
        lambda repo, *arguments: (
            "f" * 40 if arguments == ("rev-parse", "HEAD") else ""
        ),
    )
    with pytest.raises(bridge.ProducerCacheBridgeError, match="moved"):
        bridge._validate_recorded_producer(
            metadata,
            expected_commit=bridge.FROZEN_CACHE_PRODUCER_COMMIT,
        )


def test_recorded_producer_rejects_dirty_checkout(tmp_path, monkeypatch):
    metadata, _ = _producer_metadata(tmp_path)

    def dirty_git(repo, *arguments):
        del repo
        return (
            bridge.FROZEN_CACHE_PRODUCER_COMMIT
            if arguments == ("rev-parse", "HEAD")
            else "?? untracked"
        )

    monkeypatch.setattr(bridge, "_git_output", dirty_git)
    with pytest.raises(bridge.ProducerCacheBridgeError, match="dirty"):
        bridge._validate_recorded_producer(
            metadata,
            expected_commit=bridge.FROZEN_CACHE_PRODUCER_COMMIT,
        )


def test_recorded_producer_rejects_other_commit_before_git_access(
    tmp_path, monkeypatch
):
    metadata, _ = _producer_metadata(tmp_path)
    metadata["implementation"]["repo_commit"] = "f" * 40
    monkeypatch.setattr(
        bridge,
        "_git_output",
        lambda *args: pytest.fail("Git must not be consulted for wrong identity"),
    )
    with pytest.raises(bridge.ProducerCacheBridgeError, match="differs"):
        bridge._validate_recorded_producer(
            metadata,
            expected_commit=bridge.FROZEN_CACHE_PRODUCER_COMMIT,
        )


def test_bridge_rejects_protected_test_before_reading_cache(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "read_clip_manifest",
        lambda path: [{"split": "test", "clip_id": "protected"}],
    )
    with pytest.raises(bridge.ProducerCacheBridgeError, match="only train"):
        bridge.ProducerAttestedCausalDataset("manifest", "data", "cache")
