from __future__ import annotations

import inspect
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
    assert set(protocol["consumption_integrity"]["full_sha256_before_and_after"]) == {
        "registration_oracle_train_actions", "train_rgb_encode",
        "validation_rgb_encode", "rank_index_and_feature_shards_merge",
        "train_features_analysis",
        "validation_features_analysis", "train_actions_analysis",
        "validation_actions_analysis",
    }
    assert protocol["consumption_integrity"]["rank_shard_producer_seal_required"] is True


def test_action_target_uses_full_aligned_segments_not_within_chunk_delta() -> None:
    actions = np.arange(2 * 13 * 5 * 23, dtype=np.float32).reshape(2, 13, 5, 23)
    target = probe.aligned_action_targets(actions)
    assert target.shape == (2, 3, 460)
    np.testing.assert_array_equal(target[:, 0], actions[:, 0:4].reshape(2, -1))
    np.testing.assert_array_equal(target[:, 1], actions[:, 4:8].reshape(2, -1))
    np.testing.assert_array_equal(target[:, 2], actions[:, 8:12].reshape(2, -1))
    assert not np.isin(actions[:, 12].reshape(-1), target.reshape(-1)).all()
    temporal = probe.temporally_misaligned_action_targets(actions)
    np.testing.assert_array_equal(temporal[:, 0], actions[:, 4:8].reshape(2, -1))
    np.testing.assert_array_equal(temporal[:, 1], actions[:, 8:12].reshape(2, -1))
    np.testing.assert_array_equal(temporal[:, 2], actions[:, 0:4].reshape(2, -1))
    control = probe.fixed_protocol()["controls"][
        "same_clip_task_matched_temporal_misalignment"
    ]
    assert control["same_clip_episode_and_task"] is True
    assert control["donor_transition_by_recipient"] == [1, 2, 0]
    assert control["nonoverlapping_with_recipient"] is True
    for aligned, negative in zip(
        probe.ACTION_CHUNK_INTERVALS,
        probe.TEMPORALLY_MISALIGNED_ACTION_CHUNK_INTERVALS,
    ):
        assert not (set(range(*aligned)) & set(range(*negative)))


def test_temporal_negative_train_oracle_can_attain_frozen_cosine_gate() -> None:
    rng = np.random.default_rng(20260808)
    actions = rng.normal(size=(64, *probe.ACTION_SHAPE)).astype(np.float32)
    result = probe.temporal_control_oracle_feasibility(actions)
    probe.validate_temporal_oracle_payload(result)
    assert result["passed"] is True
    for row in result["subsets"].values():
        assert row["perfect_predictor_cosine_gap"] >= 0.10
        assert row["perfect_predictor_mse_gap"] > 0
    register_source = inspect.getsource(probe.command_register)
    assert (
        register_source.index("oracle_action_pre_rehash = full_preconsumption_rehash")
        < register_source.index("train_actions_for_oracle = np.load")
        < register_source.index("temporal_control_oracle_feasibility")
        < register_source.index("oracle_action_post_rehash = full_postconsumption_rehash")
        < register_source.index("output_root.mkdir")
    )


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
    producer = probe.full_producer_seal(
        record,
        label="synthetic array producer seal",
        binding_identity_sha256="1" * 64,
    )
    receipt = probe.full_preconsumption_rehash(
        record,
        label="synthetic array",
        binding_identity_sha256="1" * 64,
    )
    probe.validate_same_file_identity(producer, receipt)
    assert receipt["full_file_hashed"] is True
    assert receipt["sha256"] == record["sha256"]
    artifact.read_bytes()  # stand in for a complete, unchanged consumer
    post = probe.full_postconsumption_rehash(
        record,
        label="synthetic array after consumption",
        binding_identity_sha256="1" * 64,
    )
    probe.validate_consumption_window(receipt, post)
    probe.validate_same_file_identity(producer, receipt, post)
    with artifact.open("r+b") as handle:
        handle.seek(64)
        handle.write(b"z")
    assert artifact.stat().st_size == record["bytes"]
    with pytest.raises(probe.ActionCycleProbeError, match="same-size middle mutation"):
        probe.full_postconsumption_rehash(
            record,
            label="synthetic array after mutated consumption",
            binding_identity_sha256="1" * 64,
        )


def test_shard_merge_brackets_memmaps_and_consumes_only_copies() -> None:
    merge_source = " ".join(inspect.getsource(probe._merge_encoding_shards).split())
    assert (
        merge_source.index("feature_pre_rehash = full_preconsumption_rehash")
        < merge_source.index("features_mmap = np.load")
        < merge_source.index("features = np.array(features_mmap, copy=True)")
        < merge_source.index("feature_post_rehash = full_postconsumption_rehash")
        < merge_source.index("all_features.append(features)")
        < merge_source.index("values = np.concatenate(all_features)")
    )
    assert (
        "validate_same_file_identity( feature_producer_seal, "
        "feature_pre_rehash, feature_post_rehash )"
    ) in merge_source


def test_shard_merge_persists_producer_pre_and_post_identities(tmp_path) -> None:
    output = tmp_path / "encoded"
    output.mkdir()
    binding = "2" * 64
    registration = {
        "identity_sha256": binding,
        "inputs": {"train": {"rgb": {"synthetic": True}, "actions": {}}},
    }
    index_path = output / "indices.rank00.int64.npy"
    feature_path = output / "features.rank00.float32.npy"
    np.save(index_path, np.arange(2, dtype=np.int64), allow_pickle=False)
    np.save(
        feature_path,
        np.zeros((2, 3, 3, probe.FEATURE_DIM), dtype=np.float32),
        allow_pickle=False,
    )
    index_record = probe.file_record(index_path)
    feature_record = probe.file_record(feature_path)
    index_seal = probe.full_producer_seal(
        index_record, label="index producer", binding_identity_sha256=binding
    )
    feature_seal = probe.full_producer_seal(
        feature_record, label="feature producer", binding_identity_sha256=binding
    )
    receipt = probe.identity_payload(
        {
            "kind": "action-cycle-recoverability-encoding-rank-v1",
            "registration_identity_sha256": binding,
            "split": "train",
            "rank": 0,
            "world_size": 1,
            "indices_sha256": index_record["sha256"],
            "features_sha256": feature_record["sha256"],
            "indices_file": index_record,
            "features_file": feature_record,
            "indices_producer_seal": index_seal,
            "features_producer_seal": feature_seal,
            "rows": 2,
        }
    )
    probe.exclusive_json(output / "receipt.rank00.json", receipt)
    rgb_path = tmp_path / "rgb.bin"
    rgb_path.write_bytes(b"rgb-consumption-window")
    rgb_record = probe.file_record(rgb_path)
    rgb_before = probe.full_preconsumption_rehash(
        rgb_record, label="RGB before", binding_identity_sha256=binding
    )
    rgb_after = probe.full_postconsumption_rehash(
        rgb_record, label="RGB after", binding_identity_sha256=binding
    )

    metadata = probe._merge_encoding_shards(
        output,
        split="train",
        count=2,
        world_size=1,
        registration=registration,
        canary={"passed": True},
        rgb_pre_rehash=rgb_before,
        rgb_post_rehash=rgb_after,
    )

    shard = metadata["shards"][0]
    for key, seal in (
        ("index_merge_consumption_window", index_seal),
        ("feature_merge_consumption_window", feature_seal),
    ):
        window = shard[key]
        assert window["unchanged"] is True
        probe.validate_consumption_window(window["before"], window["after"])
        probe.validate_same_file_identity(
            seal, window["producer"], window["before"], window["after"]
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
    assert "#SBATCH --gpus-per-node=1" in analyze
    assert probe.fixed_protocol()["analysis_device"] == "cpu"
    assert probe.fixed_protocol()["analysis_scheduler_bookkeeping_gpus"] == 1
    assert probe.fixed_protocol()["analysis_cuda_usage_allowed"] is False
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
