from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools import action_cycle_recoverability as probe


def test_actual_four_bin_contract_yields_three_aligned_transitions() -> None:
    protocol = probe.fixed_protocol()
    assert protocol["vae"]["causal_input_blocks"] == [[0, 1], [1, 5], [5, 9], [9, 13]]
    assert protocol["alignment"]["latent_displacements"] == [[0, 1], [1, 2], [2, 3]]
    assert protocol["alignment"]["action_chunk_intervals"] == [[0, 4], [4, 8], [8, 12]]
    assert protocol["alignment"]["unused_terminal_action_chunk"] == 12
    assert protocol["alignment"]["future_relevant_guardrail"] == [1, 2]


def test_action_target_uses_full_aligned_segments_not_within_chunk_delta() -> None:
    actions = np.arange(2 * 13 * 5 * 23, dtype=np.float32).reshape(2, 13, 5, 23)
    target = probe.aligned_action_targets(actions)
    assert target.shape == (2, 3, 460)
    np.testing.assert_array_equal(target[:, 0], actions[:, 0:4].reshape(2, -1))
    np.testing.assert_array_equal(target[:, 1], actions[:, 4:8].reshape(2, -1))
    np.testing.assert_array_equal(target[:, 2], actions[:, 8:12].reshape(2, -1))
    assert not np.isin(actions[:, 12].reshape(-1), target.reshape(-1)).all()
    temporal = probe.temporally_misaligned_action_targets(actions)
    np.testing.assert_array_equal(temporal[:, 0], actions[:, 1:5].reshape(2, -1))
    np.testing.assert_array_equal(temporal[:, 1], actions[:, 5:9].reshape(2, -1))
    np.testing.assert_array_equal(temporal[:, 2], actions[:, 9:13].reshape(2, -1))
    control = probe.fixed_protocol()["controls"][
        "same_clip_task_matched_temporal_misalignment"
    ]
    assert control["same_clip_episode_and_task"] is True
    assert control["offset_low_level_actions"] == 5


def test_latent_features_split_three_views_and_keep_three_displacements() -> None:
    generator = torch.Generator().manual_seed(4)
    latent = torch.randn((2, *probe.LATENT_SHAPE), generator=generator)
    feature = probe.latent_displacement_features(latent)
    assert feature.shape == (2, 3, 3, 960)
    assert torch.isfinite(feature).all()
    changed = latent.clone()
    changed[:, :, 3, :, 80:120] += torch.linspace(0, 1, 40).reshape(1, 1, 1, 40)
    delta = probe.latent_displacement_features(changed) - feature
    assert torch.count_nonzero(delta[:, :2]) == 0
    assert torch.count_nonzero(delta[:, 2, :2]) == 0
    assert torch.count_nonzero(delta[:, 2, 2]) > 0


class _FakeCausalTokenizer:
    def encode_temporal(self, video: torch.Tensor, sample: bool = False) -> torch.Tensor:
        assert sample is False
        bins = (video.shape[2] - 1) // 4 + 1
        # Values depend only on the corresponding prefix, so repeated prefix
        # calls are byte identical to the full call's emitted prefix.
        values = torch.arange(bins, device=video.device, dtype=torch.float32)
        return values.reshape(1, 1, bins, 1, 1).expand(1, 16, bins, 24, 120).clone()


def test_runtime_prefix_canary_audits_actual_outputs() -> None:
    rgb = torch.zeros((1, *probe.RGB_SHAPE), dtype=torch.float32)
    result = probe.runtime_prefix_alignment_audit(_FakeCausalTokenizer(), rgb)
    assert result["passed"] is True
    assert result["emitted_bins"] == 4
    assert result["adjacent_displacements"] == 3
    assert [row["prefix_frame_count"] for row in result["prefix_equivalence"]] == [1, 5, 9, 13]
    assert all(row["max_abs_error"] == 0 for row in result["prefix_equivalence"])


def test_episode_disjoint_shuffle_is_deterministic_bijective_and_disjoint() -> None:
    episodes = [f"episode-{index}" for index in range(64)]
    first = probe.episode_disjoint_permutation(episodes, seed=123)
    second = probe.episode_disjoint_permutation(episodes, seed=123)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(64))
    assert all(episodes[index] != episodes[donor] for index, donor in enumerate(first))
    with pytest.raises(probe.ActionCycleProbeError, match="no episode-disjoint"):
        probe.episode_disjoint_permutation(["a", "a", "a", "b"], seed=1)


def test_train_only_ridge_recovers_synthetic_linear_signal() -> None:
    rng = np.random.default_rng(9)
    x_train = rng.normal(size=(96, 12))
    weight = rng.normal(size=(12, 5))
    y_train = x_train @ weight
    x_val = rng.normal(size=(16, 12))
    prediction, fitted = probe._fit_predict_ridge(x_train, y_train, x_val, 1.0e-4)
    assert fitted.shape == weight.shape
    assert np.mean((prediction - x_val @ weight) ** 2) < 1.0e-6
    scores = probe._loo_scores(x_train, y_train, probe.ALPHA_GRID)
    assert scores.shape == (len(probe.ALPHA_GRID),)
    assert np.isfinite(scores).all()


def test_paired_bootstrap_requires_point_threshold_and_simultaneous_bound() -> None:
    baseline = np.full(64, 1.0)
    candidate = np.full(64, 0.5)
    aligned_cosine = np.full(64, 0.4)
    shuffled_cosine = np.full(64, 0.1)
    aligned_hit = np.ones(64)
    shuffled_hit = np.zeros(64)
    result = probe.paired_bootstrap_gate(
        {
            "x/mse_vs_train_mean": ("relative", baseline, candidate),
            "x/cosine_vs_shuffled_fit": ("difference", aligned_cosine, shuffled_cosine),
            "x/retrieval_vs_shuffled_target": ("difference", aligned_hit, shuffled_hit),
        },
        samples=200,
        seed=5,
    )
    assert result["all_passed"] is True
    assert all(row["simultaneous_lower_95"] > 0 for row in result["comparisons"].values())


def test_full_preconsumption_hash_rejects_same_size_middle_mutation(tmp_path) -> None:
    artifact = tmp_path / "array.bin"
    artifact.write_bytes(b"a" * 128)
    record = probe.file_record(artifact)
    receipt = probe.full_preconsumption_rehash(
        record,
        label="synthetic array",
        registration_identity_sha256="1" * 64,
    )
    assert receipt["full_file_hashed"] is True
    assert receipt["sha256"] == record["sha256"]
    with artifact.open("r+b") as handle:
        handle.seek(64)
        handle.write(b"z")
    assert artifact.stat().st_size == record["bytes"]
    with pytest.raises(probe.ActionCycleProbeError, match="same-size middle mutation"):
        probe.full_preconsumption_rehash(
            record,
            label="synthetic array",
            registration_identity_sha256="1" * 64,
        )


def test_slurm_contract_is_one_gpu_encode_then_cpu_analysis() -> None:
    root = Path(probe.__file__).resolve().parents[1]
    encode = (root / "tools/slurm/action_cycle_recoverability_encode.sbatch").read_text()
    analyze = (root / "tools/slurm/action_cycle_recoverability_analyze.sbatch").read_text()
    submit = (root / "tools/slurm/submit_action_cycle_recoverability.sh").read_text()
    assert "#SBATCH --nodes=1" in encode
    assert "#SBATCH --gpus-per-node=8" in encode
    assert "--nproc_per_node=8" in encode
    assert "#SBATCH --qos=short" in encode and "#SBATCH --time=02:00:00" in encode
    assert "--gpus" not in analyze
    assert "analyze --registration" in analyze
    assert '--dependency="afterok:$ENCODE_JOB"' in submit
    assert "--kill-on-invalid-dep=yes" in submit
    assert "wandb-check" in encode and "wandb-check" in analyze
    exact_base = "/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train"
    assert f"LUSTRE_BASE={exact_base}" in encode
    assert f"LUSTRE_BASE={exact_base}" in analyze
    assert f"LUSTRE_BASE={exact_base}" in submit
    assert 'export LACWM_ALLOWED_RUN_ROOTS=$LUSTRE_BASE' in submit
    assert 'export PYTHONPATH="$REPO:$VIDEOX_HOME"' in submit
    assert submit.index('export PYTHONPATH="$REPO:$VIDEOX_HOME"') < submit.index(
        '"$PYTHON_BIN" "$TOOL" register'
    )
    assert "--lustre-base \"$LUSTRE_BASE\"" in submit
    assert "full_preconsumption_rehash" not in submit  # implemented inside bound tools
