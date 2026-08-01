import json

import numpy as np
import pytest
import torch

from robot_wm.evaluation.video_latent_forcing_quality import (
    LPIPS_ALEX_FRAME_METRIC,
    LPIPS_ALEX_TEMPORAL_DIFFERENCE_METRIC,
    R3D18_FEATURE_DIM,
    R3D18_FRECHET_METRIC,
    R3D18_SHA256,
    ALEXNET_SHA256,
    QualityBatch,
    QualityMetricError,
    FrozenR3D18AvgPool,
    _required_weight,
    evaluate_quality_batch,
    lpips_alex_per_frame_per_example,
    merge_quality_batches,
    paired_bootstrap_mean_difference,
    preprocessing_specification,
    quality_metric_provenance,
    quality_summary,
    r3d18_avgpool_features,
    r3d18_frechet,
    temporal_difference_lpips_per_example,
    clamp_video_for_quality,
    upsample_lowres_video_for_quality,
)


class FakePerceptual:
    provenance = {"extractor": "test-perceptual", "weights": [{"sha256": "a" * 64}]}

    def __init__(self):
        self.calls = []

    def __call__(self, reference, candidate):
        self.calls.append((reference.detach().clone(), candidate.detach().clone()))
        return (reference - candidate).square().mean(dim=(1, 2, 3))


class FakeR3D:
    provenance = {"extractor": "test-r3d", "weights": [{"sha256": "b" * 64}]}

    def __call__(self, video):
        summary = video.mean(dim=(1, 2, 3, 4), keepdim=False)
        return summary[:, None].repeat(1, R3D18_FEATURE_DIM)


def test_per_frame_lpips_is_per_example_mean_over_eight_frames():
    reference = torch.zeros(2, 3, 8, 64, 112)
    candidate = torch.zeros_like(reference)
    candidate[0] = 0.5
    candidate[1] = 1.0
    scores = lpips_alex_per_frame_per_example(reference, candidate, FakePerceptual())
    torch.testing.assert_close(scores, torch.tensor([0.25, 1.0]))


def test_temporal_difference_lpips_divides_differences_by_two_explicitly():
    reference = torch.zeros(1, 3, 8, 64, 112)
    candidate = torch.stack(
        [torch.full((3, 64, 112), -0.7 + 0.2 * time) for time in range(8)],
        dim=1,
    ).unsqueeze(0)
    extractor = FakePerceptual()
    scores = temporal_difference_lpips_per_example(reference, candidate, extractor)
    torch.testing.assert_close(scores, torch.tensor([0.01]), rtol=1e-5, atol=1e-7)
    ref_delta, gen_delta = extractor.calls[0]
    assert ref_delta.shape == (7, 3, 64, 112)
    torch.testing.assert_close(gen_delta, torch.full_like(gen_delta, 0.1))


def test_video_validation_fails_outside_range_instead_of_clipping():
    reference = torch.zeros(1, 3, 8, 64, 112)
    candidate = reference.clone()
    candidate[0, 0, 0, 0, 0] = 1.1
    with pytest.raises(QualityMetricError, match="without clipping"):
        lpips_alex_per_frame_per_example(reference, candidate, FakePerceptual())


def test_explicit_candidate_clamp_reports_exact_fraction_and_is_not_implicit():
    candidate = torch.zeros(1, 3, 8, 64, 112)
    candidate.reshape(-1)[:3] = torch.tensor([-1.2, 1.1, 2.0])
    clamped, audit = clamp_video_for_quality(candidate)
    assert clamped.reshape(-1)[:3].tolist() == pytest.approx([-1.0, 1.0, 1.0])
    assert audit["clipped_below_count"] == 1
    assert audit["clipped_above_count"] == 2
    assert audit["clipped_count"] == 3
    assert audit["clipped_fraction"] == pytest.approx(3 / candidate.numel())
    assert len(audit["provenance_sha256"]) == 64


def test_phase1_lowres_helper_upsamples_bilinearly_without_temporal_mixing():
    lowres = torch.zeros(3, 8, 32, 56)
    lowres[:, 4] = 0.75
    upsampled = upsample_lowres_video_for_quality(lowres)
    assert upsampled.shape == (3, 8, 64, 112)
    assert torch.all(upsampled[:, :4] == 0)
    assert torch.all(upsampled[:, 4] == 0.75)
    batched = upsample_lowres_video_for_quality(lowres.unsqueeze(0))
    torch.testing.assert_close(batched[0], upsampled)


def test_injectable_r3d_feature_extractor_contract():
    video = torch.zeros(3, 3, 8, 64, 112)
    video[1] = 0.25
    features = r3d18_avgpool_features(video, FakeR3D())
    assert features.shape == (3, 512)
    assert torch.all(features[0] == 0)
    assert torch.all(features[1] == 0.25)


def test_r3d18_frechet_is_float64_stable_and_has_frozen_label():
    real = np.array([[0.0], [2.0]], dtype=np.float32)
    generated = np.array([[1.0], [3.0]], dtype=np.float32)
    assert R3D18_FRECHET_METRIC == "r3d18_frechet"
    assert r3d18_frechet(real, real.copy()) == pytest.approx(0.0, abs=1e-12)
    assert r3d18_frechet(real, generated) == pytest.approx(1.0, abs=1e-12)
    singular = np.random.default_rng(7).normal(size=(3, 512))
    assert r3d18_frechet(singular, singular.copy()) == pytest.approx(0.0, abs=1e-12)


def test_paired_bootstrap_aligns_by_clip_id_and_is_reproducible():
    reference = [("clip-b", 2.0), ("clip-a", 1.0), ("clip-c", 3.0)]
    candidate = {"clip-c": 3.5, "clip-a": 1.5, "clip-b": 2.5}
    first = paired_bootstrap_mean_difference(
        reference, candidate, samples=500, confidence=0.95, seed=17
    )
    second = paired_bootstrap_mean_difference(
        reference, candidate, samples=500, confidence=0.95, seed=17
    )
    assert first == second
    assert first["estimate"] == pytest.approx(0.5)
    assert first["ci_low"] == pytest.approx(0.5)
    assert first["ci_high"] == pytest.approx(0.5)
    assert first["paired_count"] == 3


def test_paired_bootstrap_rejects_duplicates_and_unaligned_ids():
    with pytest.raises(QualityMetricError, match="duplicate clip ID"):
        paired_bootstrap_mean_difference(
            [("a", 1.0), ("a", 2.0)], {"a": 2.0}, samples=100
        )
    with pytest.raises(QualityMetricError, match="paired clip IDs differ"):
        paired_bootstrap_mean_difference({"a": 1.0}, {"b": 1.0}, samples=100)


def test_online_batch_gather_sorts_ids_and_summarizes():
    reference = torch.zeros(4, 3, 8, 64, 112)
    candidate = reference.clone()
    candidate[0] = 0.1
    candidate[2] = 0.2
    perceptual = FakePerceptual()
    feature = FakeR3D()
    left = evaluate_quality_batch(
        reference[:2],
        candidate[:2],
        ["clip-d", "clip-a"],
        perceptual_extractor=perceptual,
        video_feature_extractor=feature,
    )
    right = evaluate_quality_batch(
        reference[2:],
        candidate[2:],
        ["clip-c", "clip-b"],
        perceptual_extractor=perceptual,
        video_feature_extractor=feature,
    )
    merged = merge_quality_batches([left, right])
    assert merged.clip_ids == ("clip-a", "clip-b", "clip-c", "clip-d")
    summary = quality_summary(merged)
    assert set(summary) == {
        LPIPS_ALEX_FRAME_METRIC,
        LPIPS_ALEX_TEMPORAL_DIFFERENCE_METRIC,
        R3D18_FRECHET_METRIC,
    }
    assert summary[LPIPS_ALEX_FRAME_METRIC] == pytest.approx((0.01 + 0.04) / 4)


def test_gather_rejects_duplicate_clip_ids():
    zeros = torch.zeros(1, dtype=torch.float64)
    features = torch.zeros(1, 512, dtype=torch.float64)
    batch = QualityBatch(("same",), zeros, zeros, features, features)
    with pytest.raises(QualityMetricError, match="duplicate clip IDs"):
        merge_quality_batches([batch, batch])


def test_preprocessing_and_actual_weight_hashes_are_in_provenance(tmp_path):
    specification = preprocessing_specification()
    assert len(specification["sha256"]) == 64
    assert specification[LPIPS_ALEX_TEMPORAL_DIFFERENCE_METRIC]["difference"].endswith("/2")
    assert specification["phase1_lowres_integration"]["required_before_quality_metrics"] is True
    # This is the full digest measured from the official file, not its filename prefix.
    assert ALEXNET_SHA256 == "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
    assert R3D18_SHA256 == "b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d"
    perceptual = FakePerceptual()
    feature = FakeR3D()
    provenance = quality_metric_provenance(perceptual, feature)
    assert len(provenance["sha256"]) == 64
    assert provenance["metrics"][-1] == "r3d18_frechet"
    json.dumps(provenance, sort_keys=True)

    weight = tmp_path / "weight.pth"
    weight.write_bytes(b"immutable-weight")
    record = _required_weight(
        weight,
        role="test",
        expected_sha256=None,
        expected_prefix=None,
        url="https://invalid.example/weight",
    )
    assert record["sha256"] == __import__("hashlib").sha256(b"immutable-weight").hexdigest()


def test_missing_or_mismatched_weight_fails_without_downloading(tmp_path):
    with pytest.raises(QualityMetricError, match="unavailable"):
        _required_weight(
            tmp_path / "missing.pth",
            role="test",
            expected_sha256="0" * 64,
            url="https://invalid.example/weight",
        )
    weight = tmp_path / "wrong.pth"
    weight.write_bytes(b"wrong")
    with pytest.raises(QualityMetricError, match="SHA-256 mismatch"):
        _required_weight(
            weight,
            role="test",
            expected_sha256="0" * 64,
            url="https://invalid.example/weight",
        )


def test_r3d_publication_extractor_rejects_filename_prefix_as_expected_hash(tmp_path):
    with pytest.raises(QualityMetricError, match="full 64-character"):
        FrozenR3D18AvgPool(
            weight_path=tmp_path / "missing.pth",
            expected_sha256="b3b3357e",
        )

    with pytest.raises(QualityMetricError, match="preregistered official"):
        FrozenR3D18AvgPool(
            weight_path=tmp_path / "missing.pth",
            expected_sha256="0" * 64,
        )
