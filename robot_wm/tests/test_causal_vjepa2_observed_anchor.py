from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from robot_wm.modeling.dual_diffusion.vjepa2_target import PCAWhiteningStats
from tools import causal_vjepa2_observed_anchor as observed


def _normalization() -> observed.ObservedIncrementNormalization:
    mean = torch.linspace(-0.2, 0.2, observed.TARGET_CHANNELS)
    std = torch.linspace(0.3, 1.3, observed.TARGET_CHANNELS)
    return observed.ObservedIncrementNormalization(mean, std, {"split": "train"})


def test_preregistration_is_bound_to_frozen_git_bytes() -> None:
    record = observed.preregistration_record()
    assert record["commit"] == observed.PREREGISTRATION_COMMIT
    assert record["file"]["sha256"] == observed.PREREGISTRATION_SHA256
    assert record["commit_is_ancestor"] is True
    assert record["frozen_before_candidate_metrics"] is True
    assert record["prospective_matched_control_recovery_before_any_artifact"] is True
    assert record["running_temporal_doe_used_for_contingency_only"] is True
    assert record["continuation_local_abs_required"] is True


def test_observed_prefix_is_exact_and_ends_in_h3_h4() -> None:
    values = torch.linspace(-0.8, 0.8, 5)
    history = values.reshape(1, 1, 5, 1, 1).expand(1, 3, 5, 2, 3).clone()
    prefix = observed.build_observed_prefix(history)

    assert prefix.shape == (1, 3, 16, 2, 3)
    assert observed.OBSERVED_PREFIX_FRAME_MAP == (0,) * 12 + (1, 2, 3, 4)
    assert torch.equal(prefix[:, :, :12], history[:, :, :1].expand(-1, -1, 12, -1, -1))
    assert torch.equal(prefix[:, :, -2], history[:, :, 3])
    assert torch.equal(prefix[:, :, -1], history[:, :, 4])


def test_observed_prefix_prepares_official_teacher_geometry() -> None:
    history = torch.zeros(1, 3, 5, 64, 112)
    prepared = observed.prepare_observed_anchor_teacher_input(history)

    assert prepared.shape == (1, 3, 16, 384, 672)
    assert prepared.dtype == torch.float32
    assert torch.isfinite(prepared).all()


def test_public_anchor_extractor_cannot_read_future_and_is_future_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Encoder(nn.Module):
        def forward(self, prepared: torch.Tensor, *, training: bool) -> torch.Tensor:
            assert training is False
            return prepared.mean(dim=(1, 2, 3, 4)).reshape(-1, 1, 1)

    monkeypatch.setattr(
        observed,
        "prepare_observed_anchor_teacher_input",
        lambda history: history,
    )
    monkeypatch.setattr(
        observed,
        "select_and_pool_observed_anchor",
        lambda tokens, *, batch_size: tokens.reshape(batch_size, 1, 1, 1).expand(
            -1, 8, 14, 1
        ),
    )
    monkeypatch.setattr(
        observed,
        "project_observed_anchor_tokens",
        lambda tokens, stats, *, quantize_online: (
            tokens.permute(0, 3, 1, 2).expand(-1, 48, -1, -1).half().float()
        ),
    )
    parameters = inspect.signature(observed.extract_observed_anchor).parameters
    assert "future" not in parameters
    assert "target" not in parameters

    history = torch.zeros(2, 3, 5, 4, 6)
    future = torch.randn(2, 3, 8, 4, 6)
    stats = object()
    first = observed.extract_observed_anchor(
        history,
        encoder=Encoder(),
        stats=stats,  # type: ignore[arg-type]
    )
    future.add_(1000.0)
    second = observed.extract_observed_anchor(
        history,
        encoder=Encoder(),
        stats=stats,  # type: ignore[arg-type]
    )

    assert first.shape == (2, 48, 8, 14)
    assert torch.equal(first, second)


def test_last_tubelet_selection_and_nonoverlapping_pooling() -> None:
    # Small geometry exercises the same eight temporal tubelets and 2x3 pool.
    tokens = torch.zeros(1, 8, 4, 6, 2)
    for temporal_index in range(8):
        tokens[:, temporal_index, :, :, 0] = temporal_index
        tokens[:, temporal_index, :, :, 1] = torch.arange(24).reshape(4, 6)
    pooled = observed.select_and_pool_observed_anchor(
        tokens.reshape(1, 8 * 4 * 6, 2),
        batch_size=1,
        teacher_size=(4, 6),
        patch_size=1,
        tubelet_size=2,
        pool_kernel=(2, 3),
        expected_source_dim=None,
    )

    assert pooled.shape == (1, 2, 2, 2)
    assert torch.equal(pooled[..., 0], torch.full((1, 2, 2), 7.0))
    expected = torch.tensor([[[4.0, 7.0], [16.0, 19.0]]])
    assert torch.equal(pooled[..., 1], expected)


def test_projection_uses_same_pca_and_float16_roundtrip() -> None:
    mean = torch.zeros(768)
    components = torch.zeros(48, 768)
    components[:, :48] = torch.eye(48)
    stats = PCAWhiteningStats(
        mean=mean,
        components=components,
        eigenvalues=torch.ones(48),
    )
    tokens = torch.zeros(1, 8, 14, 768)
    tokens[..., :48] = torch.linspace(-0.9, 0.9, 48)
    result = observed.project_observed_anchor_tokens(tokens, stats)

    assert result.shape == (1, 48, 8, 14)
    assert result.dtype == torch.float32
    assert torch.equal(result, result.half().float())
    assert torch.equal(result[0, :, 0, 0], torch.linspace(-0.9, 0.9, 48).half().float())


def test_anchored_increment_transform_and_inverse() -> None:
    torch.manual_seed(4)
    anchor = torch.randn(2, *observed.ANCHOR_SHAPE)
    semantic = torch.randn(2, *observed.INCREMENT_SHAPE)
    increments = observed.anchored_increments(semantic, anchor)
    recovered = observed.decode_anchored_increments(increments, anchor)

    assert torch.equal(increments[:, :, 0], semantic[:, :, 0] - anchor)
    assert torch.equal(increments[:, :, 1:], semantic[:, :, 1:] - semantic[:, :, :-1])
    assert torch.allclose(recovered, semantic, atol=2e-6, rtol=0)


def test_normalized_target_roundtrip_and_exact_controls() -> None:
    torch.manual_seed(8)
    normalization = _normalization()
    anchor = torch.randn(2, *observed.ANCHOR_SHAPE)
    semantic = torch.randn(2, *observed.INCREMENT_SHAPE)
    normalized = observed.encode_normalized_increment_target(
        semantic, anchor, normalization
    )
    recovered = observed.decode_normalized_increment_prediction(
        normalized, anchor, normalization
    )
    static_q, static = observed.anchor_static_control(anchor, normalization)
    mean_q, mean = observed.mean_increment_control(anchor, normalization)

    assert torch.allclose(recovered, semantic, atol=3e-6, rtol=0)
    expected_static_q = (
        (-normalization.mean / normalization.std)
        .reshape(1, 48, 1, 1, 1)
        .expand_as(static_q)
    )
    assert torch.allclose(static_q, expected_static_q, atol=1e-7, rtol=0)
    assert torch.equal(
        static,
        anchor.unsqueeze(2).expand(-1, -1, 8, -1, -1),
    )
    assert torch.equal(mean_q, torch.zeros_like(mean_q))
    expected_mean = anchor.unsqueeze(2) + torch.arange(
        1, 9, dtype=torch.float32
    ).reshape(1, 1, 8, 1, 1) * normalization.mean.reshape(1, 48, 1, 1, 1)
    assert torch.allclose(mean, expected_mean, atol=2e-6, rtol=0)


def test_shuffling_only_decode_anchor_cannot_change_temporal_differences() -> None:
    """Regression proof for the prospective correction to the frozen gate."""
    torch.manual_seed(9)
    normalization = _normalization()
    normalized = torch.randn(4, *observed.INCREMENT_SHAPE)
    own_anchor = torch.randn(4, *observed.ANCHOR_SHAPE)
    donor_anchor = own_anchor.roll(1, dims=0)

    own = observed.decode_normalized_increment_prediction(
        normalized, own_anchor, normalization
    )
    shuffled = observed.decode_normalized_increment_prediction(
        normalized, donor_anchor, normalization
    )

    # The additive anchor changes absolute content but cancels identically in
    # adjacent differences.  A temporal-NMSE donor-anchor gate is impossible.
    assert not torch.equal(own, shuffled)
    assert torch.allclose(own.diff(dim=2), shuffled.diff(dim=2), atol=2e-6, rtol=0)


def test_protected_split_is_rejected_before_cache_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        observed,
        "read_clip_manifest",
        lambda path, expected_split=None: [
            {"split": "test", "protected": True, "clip_id": "x", "episode_index": 1}
        ],
    )
    with pytest.raises(observed.ObservedAnchorError, match="train/validation only"):
        observed._split_manifest(tmp_path / "test.identifiers.jsonl")  # noqa: SLF001


def test_normalization_loader_binds_train_only_artifact(tmp_path: Path) -> None:
    evidence = {}
    for name in ("train_manifest", "semantic_cache_metadata", "anchor_cache_metadata"):
        evidence_path = tmp_path / f"{name}.json"
        evidence_path.write_text(f'{{"name":"{name}"}}\n', encoding="utf-8")
        evidence[name] = observed.vlf.file_record(evidence_path)
    implementation_dependencies = {
        name: observed.vlf.file_record(path)
        for name, path in {
            "entrypoint": Path(observed.__file__),
            "cache_bridge": observed.REPO_ROOT
            / "tools"
            / "causal_vjepa2_cache_bridge.py",
            "causal_dataset": observed.REPO_ROOT
            / "robot_wm"
            / "datasets"
            / "droid"
            / "causal_vjepa2.py",
            "vjepa_target": observed.REPO_ROOT
            / "robot_wm"
            / "modeling"
            / "dual_diffusion"
            / "vjepa2_target.py",
            "preregistration": observed.PREREGISTRATION_PATH,
        }.items()
    }
    payload = {
        "schema": observed.NORMALIZATION_SCHEMA,
        "status": "complete",
        "complete": True,
        "target_kind": observed.INCREMENT_TARGET_KIND,
        "split": "train",
        "clips": 64_000,
        "target_shape": list(observed.INCREMENT_SHAPE),
        "source_storage_dtype": "float16",
        "statistics_compute_dtype": "float64",
        "encode_decode_dtype": "float32",
        "std_floor": observed.STD_FLOOR,
        "statistics_source": "frozen_train_semantic_and_observed_anchor_caches_only",
        "population_elements_per_channel": observed.EXPECTED_INCREMENT_ELEMENTS_PER_CHANNEL,
        "channel_axis": observed.CHANNEL_AXIS,
        "temporal_axis": observed.TEMPORAL_AXIS,
        "increment_definition": "D0=S0-A; Dj=Sj-S[j-1] for j=1..7",
        "declared_roundtrip_max_abs_tolerance": observed.ROUNDTRIP_TOLERANCE,
        "roundtrip_max_abs_error": 0.0,
        "protected_test_accessed": False,
        "test_rows_used": 0,
        "train_manifest_sha256": evidence["train_manifest"]["sha256"],
        "semantic_cache_metadata_sha256": evidence["semantic_cache_metadata"]["sha256"],
        "anchor_cache_metadata_sha256": evidence["anchor_cache_metadata"]["sha256"],
        **evidence,
        "implementation": {
            "dirty": False,
            "dependencies": implementation_dependencies,
        },
        "increment_mean": [0.0] * 48,
        "increment_std": [1.0] * 48,
    }
    path = tmp_path / "normalization.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    normalization, record = observed.load_increment_normalization(
        path,
        expected_train_manifest_sha256=evidence["train_manifest"]["sha256"],
        expected_semantic_cache_metadata_sha256=evidence["semantic_cache_metadata"]["sha256"],
        expected_anchor_cache_metadata_sha256=evidence["anchor_cache_metadata"]["sha256"],
    )

    assert torch.equal(normalization.mean, torch.zeros(48))
    assert torch.equal(normalization.std, torch.ones(48))
    assert record["payload_sha256"] == observed._canonical_sha256(payload)  # noqa: SLF001
