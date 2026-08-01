from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.video_latent_forcing_poc import (
    CHECKPOINT_SCHEMA,
    GATE_SCHEMA,
    FROZEN_CLEAN_TIME_EPS,
    FROZEN_EMA_DECAY,
    FROZEN_GLOBAL_BATCH_SIZE,
    FROZEN_LEARNING_RATE,
    FROZEN_WARMUP_UPDATES,
    FROZEN_WEIGHT_DECAY,
    ModelEMA,
    PocError,
    DeterministicDistributedBatchSampler,
    DistributedContext,
    build_parser,
    canonical_quality_video,
    clean_time_euler_from_x,
    corrupt_clean_time,
    evaluation_batch_indexes,
    instantiate_model,
    load_checkpoint,
    masked_branch_loss,
    model_forward,
    per_example_x_prediction_flow_mse,
    paired_global_derangement,
    paired_rank_evaluation_layout,
    per_example_metrics,
    reconcile_resume_artifacts,
    sample_control,
    sample_training_clocks,
    sha256_json,
    stable_noise_like,
    stable_within_batch_shuffle_indices,
    validate_args,
    validate_phase1_gate_record,
    validate_nfe_pairs,
)


def _model_args(arm: str = "L1") -> Namespace:
    return Namespace(arm=arm, width=16, depth=1, heads=4, mlp_ratio=2.0)


def _training_cli(command: str = "calibrate") -> list[str]:
    values = [
        command,
        "--arm",
        "L1",
        "--data-root",
        "/mnt/data1/data",
        "--train-manifest",
        "/mnt/data1/train.jsonl",
        "--validation-manifest",
        "/mnt/data1/val.jsonl",
        "--artifact-root",
        "/mnt/data1/runs",
        "--run-id",
        "unit",
    ]
    if command == "train":
        values.extend(("--calibration-record", "/mnt/data1/calibration.json"))
    return values


def test_cluster_launchers_use_the_case_sensitive_b200_feature_name():
    repo_root = Path(__file__).resolve().parents[2]
    launcher_paths = (
        "tools/slurm/build_video_latent_forcing_droid.sbatch",
        "tools/slurm/submit_build_video_latent_forcing_droid.sh",
        "tools/slurm/submit_video_latent_forcing_poc.sh",
        "tools/slurm/submit_video_latent_forcing_eval.sh",
    )
    for relative_path in launcher_paths:
        launcher_text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "B200" in launcher_text, relative_path
        assert "CONSTRAINT=\"b200\"" not in launcher_text, relative_path
        assert "--constraint=b200" not in launcher_text, relative_path
        assert "Default: b200" not in launcher_text, relative_path


def test_slurm_launchers_pass_repository_root_instead_of_using_spool_source():
    repo_root = Path(__file__).resolve().parents[2]
    submitters = (
        "tools/slurm/submit_build_video_latent_forcing_droid.sh",
        "tools/slurm/submit_video_latent_forcing_poc.sh",
        "tools/slurm/submit_video_latent_forcing_eval.sh",
    )
    batch_scripts = (
        "tools/slurm/build_video_latent_forcing_droid.sbatch",
        "tools/slurm/video_latent_forcing_poc.sbatch",
    )
    for relative_path in submitters:
        launcher_text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert '--repo-root "$REPO_ROOT"' in launcher_text, relative_path
    for relative_path in batch_scripts:
        batch_text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert '"--repo-root"' in batch_text, relative_path
        assert 'dirname "${BASH_SOURCE[0]}"' not in batch_text, relative_path


def test_parser_locks_frozen_optimizer_defaults_and_exact_calibration_length():
    args = build_parser().parse_args(_training_cli())
    validate_args(args)
    assert args.global_batch_size == FROZEN_GLOBAL_BATCH_SIZE == 256
    assert args.learning_rate == FROZEN_LEARNING_RATE == 5e-5
    assert args.warmup_updates == FROZEN_WARMUP_UPDATES == 500
    assert args.weight_decay == FROZEN_WEIGHT_DECAY == 0.0
    assert args.ema_decay == FROZEN_EMA_DECAY == 0.9999

    args.learning_rate = 1e-4
    with pytest.raises(PocError, match="optimizer contract is frozen"):
        validate_args(args)

    args = build_parser().parse_args(_training_cli())
    args.width = 256
    with pytest.raises(PocError, match="model contract is frozen"):
        validate_args(args)

    args = build_parser().parse_args(_training_cli())
    args.seed = 2234
    with pytest.raises(PocError, match="Phase-1|optimizer seed"):
        args.arm = "phase1"
        validate_args(args)


def test_cli_width_maps_to_hidden_size_and_never_image_width():
    model, config = instantiate_model(_model_args())
    assert config["hidden_size"] == 16
    assert config["width"] == 112
    assert config["height"] == 64
    assert model.config.future_shape == (3, 8, 64, 112)
    assert model.config.auxiliary_shape == (48, 8, 8, 14)


def test_clean_time_endpoints_positive_euler_and_released_eps_loss():
    clean = torch.tensor([[2.5]])
    noise = torch.tensor([[-3.0]])
    time = torch.tensor([0.99])
    noisy = corrupt_clean_time(clean, noise, time)
    prediction = torch.tensor([[3.0]])
    loss = per_example_x_prediction_flow_mse(
        prediction, noisy, clean, noise, time
    )
    assert FROZEN_CLEAN_TIME_EPS == 0.05
    torch.testing.assert_close(loss, torch.tensor([(0.5 / 0.05) ** 2]))
    torch.testing.assert_close(
        clean_time_euler_from_x(noise, clean, torch.tensor([0.0]), torch.tensor([1.0])),
        clean,
    )


def test_schedule_masks_and_full_batch_normalization_are_exact():
    generator = torch.Generator().manual_seed(1234)
    clocks = sample_training_clocks("A1", 100_000, torch.device("cpu"), generator=generator)
    auxiliary = clocks.auxiliary_loss_mask.bool()
    assert float(auxiliary.float().mean()) == pytest.approx(0.4, abs=0.005)
    assert torch.equal(clocks.auxiliary_condition_mask, auxiliary)
    assert torch.all(clocks.video_time[auxiliary] == 0)
    assert torch.all((clocks.auxiliary_time[~auxiliary] >= 0.75))
    assert torch.all((clocks.auxiliary_time[~auxiliary] <= 1.0))
    assert torch.equal(clocks.video_loss_mask.bool(), ~auxiliary)
    assert masked_branch_loss(torch.tensor([2.0, 8.0]), torch.tensor([1.0, 0.0])) == 1.0


def test_step_addressed_sampler_resume_and_accumulation_are_identical():
    complete = DeterministicDistributedBatchSampler(
        17,
        global_batch_size=8,
        rank=1,
        world_size=2,
        seed=9,
        start_update=0,
        end_update=6,
        micro_batch_size=2,
    )
    resumed = DeterministicDistributedBatchSampler(
        17,
        global_batch_size=8,
        rank=1,
        world_size=2,
        seed=9,
        start_update=3,
        end_update=6,
        micro_batch_size=2,
    )
    complete_batches = list(complete)
    assert complete.accumulation_steps == 2
    assert complete_batches[6:] == list(resumed)


class _FusionRecordingModel:
    def __init__(self):
        self.masks = []

    def __call__(
        self,
        noisy_video,
        noisy_auxiliary,
        t_video,
        t_auxiliary,
        history,
        actions,
        *,
        auxiliary_fusion_mask,
    ):
        self.masks.append(auxiliary_fusion_mask)
        auxiliary = (
            torch.zeros(noisy_video.shape[0], 1, 2, 1, 1)
            if noisy_auxiliary is None
            else torch.zeros_like(noisy_auxiliary)
        )
        return SimpleNamespace(video_x=torch.zeros_like(noisy_video), auxiliary_x=auxiliary)


def test_model_forward_uses_explicit_per_example_fusion_mask():
    model = _FusionRecordingModel()
    mask = torch.tensor([True, False])
    model_forward(
        model,
        noisy_video=torch.zeros(2, 1, 2, 1, 1),
        noisy_auxiliary=torch.zeros(2, 1, 2, 1, 1),
        t_video=torch.zeros(2),
        t_auxiliary=torch.zeros(2),
        history=torch.zeros(2, 1, 1, 1, 1),
        actions=torch.zeros(2, 1, 1),
        condition_on_auxiliary=mask,
    )
    assert torch.equal(model.masks[0], mask)


def _sampling_inputs():
    history = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1, 1)
    actions = torch.zeros(2, 1, 1)
    video_noise = torch.ones(2, 1, 2, 1, 1)
    auxiliary_noise = -torch.ones(2, 1, 2, 1, 1)
    clean_auxiliary = torch.tensor([3.0, 4.0]).reshape(2, 1, 1, 1, 1).expand_as(
        auxiliary_noise
    )
    return history, actions, video_noise, auxiliary_noise, clean_auxiliary


def test_shared_boundary_exact_calls_shuffle_and_b0_no_auxiliary_input():
    history, actions, video_noise, auxiliary_noise, clean_auxiliary = _sampling_inputs()
    model = _FusionRecordingModel()
    generated = torch.tensor([5.0, 7.0]).reshape(2, 1, 1, 1, 1).expand_as(
        auxiliary_noise
    )
    indexes, source_ids = stable_within_batch_shuffle_indices(
        ["clip-b", "clip-a"], torch.device("cpu")
    )
    shuffled = sample_control(
        model,
        "L1",
        history,
        actions,
        None,
        control="shuffled",
        auxiliary_steps=3,
        video_steps=2,
        video_noise=video_noise,
        auxiliary_noise=auxiliary_noise,
        generated_auxiliary=generated,
        shuffle_indices=indexes,
    )
    assert shuffled.model_calls == 2
    assert source_ids == ["clip-a", "clip-b"]
    torch.testing.assert_close(shuffled.conditioning_auxiliary, generated[indexes])

    b0_model = _FusionRecordingModel()
    b0 = sample_control(
        b0_model,
        "B0",
        history,
        actions,
        None,
        control="off",
        auxiliary_steps=0,
        video_steps=4,
        video_noise=video_noise,
        auxiliary_noise=None,
    )
    assert b0.model_calls == 4
    assert b0.generated_auxiliary is None
    assert b0.initial_auxiliary_noise is None
    assert all(mask is False for mask in b0_model.masks)


def test_fixed_noise_is_clip_keyed_and_independent_of_batch_order():
    reference = torch.empty(2, 2, 3)
    first = stable_noise_like(reference, ["a", "b"], 17, "video")
    reverse = stable_noise_like(reference, ["b", "a"], 17, "video")
    torch.testing.assert_close(first[0], reverse[1], rtol=0, atol=0)
    torch.testing.assert_close(first[1], reverse[0], rtol=0, atol=0)
    digest = hashlib.sha256(b"a:17:video").digest()
    generator = torch.Generator().manual_seed(
        int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    )
    expected = torch.randn((2, 3), generator=generator)
    torch.testing.assert_close(first[0], expected, rtol=0, atol=0)


def test_nfe_contract_and_singleton_batch_merge_fail_closed():
    validate_nfe_pairs("phase1", [(25, 0)])
    validate_nfe_pairs("B0", [(0, 50)])
    validate_nfe_pairs("L1", [(25, 25)])
    with pytest.raises(PocError):
        validate_nfe_pairs("L1", [(0, 50)])
    assert [len(batch) for batch in evaluation_batch_indexes(17, 8, require_derangement=True)] == [8, 9]
    with pytest.raises(PocError, match="at least two"):
        evaluation_batch_indexes(1, 8, require_derangement=True)


def test_manifest_global_derangement_is_world_and_batch_invariant():
    rows = [{"clip_id": f"clip-{index}"} for index in range(10)]
    mapping, digest = paired_global_derangement(rows)
    assert digest == paired_global_derangement(rows)[1]
    assert mapping["clip-0"] == "clip-1"
    assert mapping["clip-1"] == "clip-0"
    covered = []
    for rank in range(2):
        indexes, batches = paired_rank_evaluation_layout(
            len(rows), 4, rank=rank, world_size=2
        )
        assert all(len(batch) % 2 == 0 for batch in batches)
        covered.extend(indexes)
        for start in range(0, len(indexes), 2):
            left, right = indexes[start : start + 2]
            assert mapping[rows[left]["clip_id"]] == rows[right]["clip_id"]
    assert sorted(covered) == list(range(len(rows)))


def test_quality_canonicalization_upsamples_and_reports_per_example_clip_fraction():
    video = torch.zeros(2, 3, 8, 32, 56)
    video[0].reshape(-1)[:2] = torch.tensor([-2.0, 1.5])
    canonical, fractions = canonical_quality_video(video, upsample_lowres=True)
    assert canonical.shape == (2, 3, 8, 64, 112)
    assert canonical.min() >= -1 and canonical.max() <= 1
    assert fractions[0] == pytest.approx(2 / video[0].numel())
    assert fractions[1] == 0


def test_auxiliary_cosine_is_mean_of_aligned_token_cosines():
    prediction_video = torch.zeros(1, 1, 2, 1, 1)
    target_video = torch.zeros_like(prediction_video)
    generated = torch.tensor([1.0, 0.0, 0.0, 100.0]).reshape(1, 2, 2, 1, 1)
    target = torch.tensor([1.0, 100.0, 0.0, 0.0]).reshape(1, 2, 2, 1, 1)
    metrics = per_example_metrics(
        prediction_video,
        target_video,
        generated,
        target,
    )
    torch.testing.assert_close(metrics["auxiliary_cosine"], torch.tensor([0.5]))


def test_ema_checkpoint_roundtrip_and_resume_log_reconciliation(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    ema = ModelEMA(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    expected = {name: value.clone() for name, value in ema.shadow.items()}
    context = DistributedContext(0, 1, 0, torch.device("cpu"))
    checkpoint = tmp_path / "update_000003.pt"
    from tools.video_latent_forcing_poc import save_checkpoint

    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        update=3,
        arm="L1",
        model_config={"hidden_size": 2},
        config_sha256="a" * 64,
        context=context,
        cumulative_optimizer_wall_seconds=12.5,
    )
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["schema"] == CHECKPOINT_SCHEMA
    assert saved["cumulative_optimizer_wall_seconds"] == 12.5
    with torch.no_grad():
        model.weight.zero_()
    ema.shadow["weight"].zero_()
    load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        expected_config_sha256="a" * 64,
        context=context,
    )
    for name, value in expected.items():
        torch.testing.assert_close(ema.shadow[name], value)

    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    metrics = run_dir / "metrics.jsonl"
    metrics.write_text(
        "".join(json.dumps({"update": update}) + "\n" for update in (1, 3, 4)),
        encoding="utf-8",
    )
    assert reconcile_resume_artifacts(run_dir, 3) == 1
    assert [json.loads(line)["update"] for line in metrics.read_text().splitlines()] == [1, 3]


def test_phase1_gate_handoff_recomputes_raw_evidence_and_rejects_resigned_edit(
    tmp_path, monkeypatch
):
    from tools import analyze_video_latent_forcing_poc as analyzer
    from tools.video_latent_forcing_poc import file_record

    checkpoint_path = tmp_path / "phase1.pt"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "arm": "phase1",
            "completed_updates": 5_000,
            "ema": {"decay": FROZEN_EMA_DECAY, "shadow": {"w": torch.ones(1)}},
        },
        checkpoint_path,
    )
    evaluation_root = tmp_path / "evaluation"
    evaluation_root.mkdir()
    evidence = evaluation_root / "metrics.jsonl"
    evidence.write_text("{}\n", encoding="utf-8")
    payload = {
        "schema": GATE_SCHEMA,
        "phase": "phase1",
        "status": "pass",
        "frozen": True,
        "validation_only": True,
        "phase1_gate_passed": True,
        "source_commit": "a" * 40,
        "selected_nfe_pair": [4, 0],
        "checkpoint": file_record(checkpoint_path),
        "evaluation": {
            "root": str(evaluation_root),
            "per_clip_metrics": file_record(evidence),
        },
        "criteria": {"4": {"passed": True}},
    }
    payload["decision_sha256"] = sha256_json(payload)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(analyzer, "analyze_phase1_evaluation", lambda _: payload)
    assert validate_phase1_gate_record(gate_path, expected_commit="a" * 40) == payload

    edited = {**payload, "criteria": {"4": {"passed": False, "edited": True}}}
    edited["decision_sha256"] = sha256_json(
        {key: value for key, value in edited.items() if key != "decision_sha256"}
    )
    gate_path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(PocError, match="differs from the decision recomputed"):
        validate_phase1_gate_record(gate_path, expected_commit="a" * 40)
