from __future__ import annotations

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel

from robot_wm.modeling.video_latent_forcing import (
    VideoLatentForcingConfig,
    VideoLatentForcingModel,
)
from tools import causal_vjepa2_screen as screen


class _IdentityAuxiliaryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        noisy_video,
        noisy_auxiliary,
        t_video,
        t_auxiliary,
        history,
        actions,
        *,
        auxiliary_fusion_mask,
        predict_video,
    ):
        self.calls += 1
        assert auxiliary_fusion_mask is True
        assert predict_video is False
        return SimpleNamespace(
            video_x=torch.zeros_like(noisy_video),
            auxiliary_x=torch.zeros_like(noisy_auxiliary),
        )


def _semantic_batch(batch: int = 2):
    generator = torch.Generator().manual_seed(7)
    return {
        "history": torch.randn(batch, 3, 5, 64, 112, generator=generator),
        "future": torch.randn(batch, 3, 8, 64, 112, generator=generator),
        "actions": torch.randn(batch, 16, 7, generator=generator),
        "auxiliary_target": torch.randn(
            batch, *screen.TARGET_SHAPE, generator=generator
        ),
    }


def test_zero_and_oracle_metrics_have_analytic_references():
    target = _semantic_batch()["auxiliary_target"]
    zero = screen.semantic_metrics(torch.zeros_like(target), target)
    oracle = screen.semantic_metrics(target, target)
    torch.testing.assert_close(zero["semantic_nmse"], torch.ones(2))
    torch.testing.assert_close(zero["temporal_difference_nmse"], torch.ones(2))
    torch.testing.assert_close(zero["retained_utility"], torch.zeros(2))
    torch.testing.assert_close(oracle["semantic_nmse"], torch.zeros(2))
    torch.testing.assert_close(oracle["temporal_difference_nmse"], torch.zeros(2))
    torch.testing.assert_close(oracle["semantic_token_cosine"], torch.ones(2))
    torch.testing.assert_close(
        oracle["temporal_difference_token_cosine"], torch.ones(2)
    )


def test_deployable_sampler_has_no_clean_target_and_hashes_every_call():
    assert "target" not in inspect.signature(screen.sample_semantic).parameters
    batch = _semantic_batch()
    video_noise = torch.randn_like(batch["future"])
    auxiliary_noise = torch.randn_like(batch["auxiliary_target"])
    model = _IdentityAuxiliaryModel()
    first = screen.sample_semantic(
        model,
        batch["history"],
        batch["actions"],
        video_noise=video_noise,
        auxiliary_noise=auxiliary_noise,
        steps=2,
    )
    assert first.model_calls == 2
    assert model.calls == 2
    assert len(first.call_input_sha256_by_example) == 2
    assert all(len(trace) == 2 for trace in first.call_input_sha256_by_example)
    assert all(
        len(digest) == 64
        for trace in first.call_input_sha256_by_example
        for digest in trace
    )
    model_again = _IdentityAuxiliaryModel()
    second = screen.sample_semantic(
        model_again,
        batch["history"],
        batch["actions"],
        video_noise=video_noise,
        auxiliary_noise=auxiliary_noise,
        steps=2,
    )
    assert first.call_input_sha256_by_example == second.call_input_sha256_by_example
    torch.testing.assert_close(first.prediction, second.prediction)


def test_sampler_hashes_each_changing_batch_with_one_host_transfer(monkeypatch):
    batch = _semantic_batch(batch=3)
    video_noise = torch.randn_like(batch["future"])
    auxiliary_noise = torch.randn_like(batch["auxiliary_target"])
    original = screen.tensor_sha256_by_example
    calls = []

    def observed(value):
        calls.append(tuple(value.shape))
        return original(value)

    monkeypatch.setattr(screen, "tensor_sha256_by_example", observed)
    result = screen.sample_semantic(
        _IdentityAuxiliaryModel(),
        batch["history"],
        batch["actions"],
        video_noise=video_noise,
        auxiliary_noise=auxiliary_noise,
        steps=4,
    )
    assert len(result.call_input_sha256_by_example) == 3
    assert calls == [
        tuple(video_noise.shape),
        tuple(batch["history"].shape),
        tuple(batch["actions"].shape),
        *[tuple(auxiliary_noise.shape)] * 4,
    ]
    assert original(auxiliary_noise) == tuple(
        screen.tensor_sha256(auxiliary_noise[item]) for item in range(3)
    )


def test_training_step_uses_future_only_as_noise_shape(monkeypatch):
    batch = _semantic_batch(batch=1)
    seen_video_noise = []

    def fake_forward(
        model,
        *,
        noisy_video,
        noisy_auxiliary,
        t_video,
        t_auxiliary,
        history,
        actions,
        condition_on_auxiliary,
        predict_video,
    ):
        del model, t_video, t_auxiliary, history, actions
        assert condition_on_auxiliary is True
        assert predict_video is False
        seen_video_noise.append(noisy_video.clone())
        return torch.zeros_like(noisy_video), torch.zeros_like(noisy_auxiliary)

    monkeypatch.setattr(screen.vlf, "model_forward", fake_forward)
    torch.manual_seed(11)
    first, first_telemetry = screen.semantic_training_step(object(), batch)
    changed = dict(batch)
    changed["future"] = torch.full_like(batch["future"], 123.0)
    torch.manual_seed(11)
    second, second_telemetry = screen.semantic_training_step(object(), changed)
    torch.testing.assert_close(seen_video_noise[0], seen_video_noise[1])
    torch.testing.assert_close(first, second)
    assert set(first_telemetry) == {
        "auxiliary_loss",
        "weighted_auxiliary_loss",
        "auxiliary_branch_count",
    }
    assert "video_loss" not in second_telemetry


def test_semantic_only_ddp_completes_consecutive_backward_passes(tmp_path):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("single-process Gloo is required for the DDP regression")
    if torch.distributed.is_initialized():
        pytest.skip("the test process already owns a distributed process group")

    init_path = tmp_path / "semantic-ddp-init"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=0,
        world_size=1,
    )
    try:
        config = VideoLatentForcingConfig(
            hidden_size=16,
            depth=1,
            num_heads=4,
            mlp_ratio=2.0,
        )
        bare_model = VideoLatentForcingModel(config)
        model = DistributedDataParallel(
            bare_model,
            find_unused_parameters=True,
        )
        batch = _semantic_batch(batch=1)
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            loss, _ = screen.semantic_training_step(model, batch)
            loss.backward()
            assert bare_model.video_output_head.weight.grad is None
            assert bare_model.video_output_head.bias.grad is None
    finally:
        torch.distributed.destroy_process_group()


def test_sampler_input_digest_is_key_order_invariant():
    record = {
        "history_sha256": "1" * 64,
        "actions_sha256": "2" * 64,
        "initial_video_noise_sha256": "3" * 64,
        "initial_auxiliary_noise_sha256": "4" * 64,
    }
    assert screen.sampler_input_sha256(record) == screen.sampler_input_sha256(
        dict(reversed(list(record.items())))
    )
    with pytest.raises(ValueError, match="malformed"):
        screen.sampler_input_sha256({**record, "extra": "5" * 64})


def test_wandb_destination_is_private_and_frozen():
    parser = screen.build_parser()
    args = parser.parse_args(
        [
            "eval",
            "--artifact-root",
            "/mnt/data1/artifacts",
            "--run-id",
            "unit",
            "--data-root",
            "/mnt/data1/data",
            "--semantic-cache-root",
            "/mnt/data1/cache",
            "--manifest",
            "/mnt/data1/val.jsonl",
            "--checkpoint",
            "/mnt/data1/checkpoints/update_005000.pt",
            "--wandb",
            "--wandb-entity",
            "some-team",
            "--wandb-project",
            "dual-video-diffusion-private",
            "--wandb-private-project-ack",
        ]
    )
    with pytest.raises(screen.ScreenError, match="frozen to private"):
        screen.validate_args(args)


def test_evaluation_wandb_payloads_use_unique_steps_and_bind_checkpoint():
    summaries = [
        {"nfe": nfe, "control": control, "semantic_nmse": 0.5}
        for nfe, control in ((1, "autonomous"), (1, "zero"), (2, "autonomous"))
    ]
    payloads = screen.evaluation_logging_payloads(summaries)
    assert [payload["update"] for payload in payloads] == [5000, 5001, 5002]
    assert all(payload["checkpoint_update"] == 5000 for payload in payloads)
    assert [payload["evaluation_cell_index"] for payload in payloads] == [0, 1, 2]
    assert [payload["nfe"] for payload in payloads] == [1, 1, 2]


def test_training_cache_pair_requires_one_pca_and_actual_train_manifest():
    train_manifest = {"path": "/train", "sha256": "1" * 64, "bytes": 10}
    val_manifest = {"path": "/val", "sha256": "2" * 64, "bytes": 20}
    shared = {
        "pca_sha256": "3" * 64,
        "implementation": {"repo_commit": "4" * 40},
        "source_commit": "5" * 40,
        "checkpoint_sha256": "6" * 64,
        "checkpoint_evidence": {"path": "/teacher", "sha256": "6" * 64},
        "source_archive_sha256": "7" * 64,
        "source_license": {"path": "/license", "sha256": "8" * 64},
        "train_manifest_sha256": train_manifest["sha256"],
        "teacher_size": [384, 672],
        "teacher_frames": 16,
        "last_temporal_token": 7,
        "pooled_token_grid": [8, 14],
        "base_droid": {"artifact": "frozen"},
        "runtime": {"python": "same"},
        "numerical_contract": {"encoder_dtype": "bfloat16"},
    }

    def cache(split, manifest):
        return {
            **shared,
            "split": split,
            "manifest_sha256": manifest["sha256"],
            "evidence": {
                "manifest": dict(manifest),
                "train_manifest": dict(train_manifest),
                "pca": {"path": "/pca", "sha256": shared["pca_sha256"]},
            },
        }

    train_cache = cache("train", train_manifest)
    val_cache = cache("val", val_manifest)
    screen._validate_training_cache_pair(
        train_cache,
        val_cache,
        train_manifest=train_manifest,
        validation_manifest=val_manifest,
    )

    changed_pca = copy.deepcopy(val_cache)
    changed_pca["pca_sha256"] = "9" * 64
    changed_pca["evidence"]["pca"]["sha256"] = "9" * 64
    with pytest.raises(screen.ScreenError, match="share one PCA"):
        screen._validate_training_cache_pair(
            train_cache,
            changed_pca,
            train_manifest=train_manifest,
            validation_manifest=val_manifest,
        )

    wrong_train_manifest = copy.deepcopy(val_cache)
    wrong_train_manifest["train_manifest_sha256"] = "a" * 64
    wrong_train_manifest["evidence"]["train_manifest"]["sha256"] = "a" * 64
    with pytest.raises(screen.ScreenError, match="actual frozen manifests"):
        screen._validate_training_cache_pair(
            train_cache,
            wrong_train_manifest,
            train_manifest=train_manifest,
            validation_manifest=val_manifest,
        )
