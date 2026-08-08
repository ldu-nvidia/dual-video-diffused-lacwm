from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from robot_wm.modeling.dual_diffusion.haar_lowpass import PerViewHaarLowpass
from tools import evaluate_frequency_forcing as evaluator
from tools import frequency_forcing_screen as screen


def test_haar_lowpass_exact_dc_motion_and_view_isolation():
    transform = PerViewHaarLowpass(
        num_views=2,
        output_size=(2, 4),
        window_size=4,
        pad_multiple=None,
    )
    video = torch.empty(1, 5, 1, 4, 8)
    for frame in range(5):
        video[:, frame, :, :, :4] = float(frame)
        video[:, frame, :, :, 4:] = float(10 + frame)

    target = transform(video)

    assert target.shape == (1, 2, 2, 2, 4)
    # Width-stacked views remain independent.
    assert torch.equal(target[0, 0, 0, :, :2], torch.zeros(2, 2))
    assert torch.equal(target[0, 0, 0, :, 2:], torch.full((2, 2), 20.0))
    assert torch.equal(target[0, 1, 0], torch.zeros(2, 4))
    # DC=(1+2+3+4)/2 and motion=(3+4-1-2)/2.
    assert torch.equal(target[0, 0, 1, :, :2], torch.full((2, 2), 5.0))
    assert torch.equal(target[0, 0, 1, :, 2:], torch.full((2, 2), 25.0))
    assert torch.equal(target[0, 1, 1], torch.full((2, 4), 2.0))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_haar_lowpass_production_shape_and_dtype(dtype):
    transform = PerViewHaarLowpass()
    video = torch.zeros(2, 13, 3, 180, 960, dtype=dtype)
    target = transform(video)
    assert target.shape == (2, 6, 4, 24, 120)
    assert target.dtype == dtype
    assert torch.isfinite(target).all()


def test_haar_lowpass_future_perturbation_changes_only_aligned_bin():
    transform = PerViewHaarLowpass(
        num_views=1,
        output_size=(2, 2),
        pad_multiple=None,
    )
    baseline = torch.zeros(1, 13, 3, 4, 4)
    changed = baseline.clone()
    changed[:, 12] = 1.0

    delta = transform(changed) - transform(baseline)

    assert torch.count_nonzero(delta[:, :, :3]) == 0
    assert torch.count_nonzero(delta[:, :, 3]) > 0
    assert torch.count_nonzero(delta[:, :3, 3]) > 0
    assert torch.count_nonzero(delta[:, 3:, 3]) > 0


def test_haar_lowpass_rejects_non_haar_spatial_ratio():
    transform = PerViewHaarLowpass(
        num_views=1,
        output_size=(3, 3),
        pad_multiple=None,
    )
    with pytest.raises(ValueError, match="powers of two"):
        transform(torch.zeros(1, 5, 3, 9, 9))


def test_fixed_rgb_action_dataset_never_returns_cached_target():
    pytest.importorskip("h5py")
    from robot_wm.datasets.abc.fixed_rgb_action_dataset import (
        ABCFixedRGBActionDataset,
    )

    dataset = object.__new__(ABCFixedRGBActionDataset)
    rgbs = np.zeros((2, 13, 3, 4, 8), dtype=np.float16)
    actions = np.zeros((2, 13, 5, 23), dtype=np.float32)
    dataset._open_rgbs = lambda: rgbs
    dataset._open_actions = lambda: actions

    sample = dataset._get_sample(1)

    assert set(sample) == {"rgb", "actions", "mask", "clip_index"}
    assert "auxiliary_target" not in sample
    assert sample["rgb"].dtype == torch.float32
    assert sample["actions"].dtype == torch.float32
    assert sample["clip_index"].item() == 1
    rgbs[1] = 1
    assert torch.count_nonzero(sample["rgb"]) == 0


def test_fixed_rgb_action_dataset_validation_never_opens_target():
    pytest.importorskip("h5py")
    from robot_wm.datasets.abc.fixed_rgb_action_dataset import (
        ABCFixedRGBActionDataset,
    )

    dataset = object.__new__(ABCFixedRGBActionDataset)
    dataset.clips = [{}, {}]
    dataset._rgbs = None
    dataset._actions = None
    dataset._open_targets = lambda: pytest.fail("V-JEPA target was opened")
    dataset._open_rgbs = lambda: np.zeros(
        (2, 13, 3, 4, 8), dtype=np.float16
    )
    dataset._open_actions = lambda: np.zeros(
        (2, 13, 5, 23), dtype=np.float32
    )

    dataset._validate_cached_arrays(1)

    assert dataset._rgbs is None
    assert dataset._actions is None


def test_frequency_screen_arm_decomposition_and_no_test_cli():
    assert [arm.code for arm in screen.ARMS] == ["FPM", "FAUX", "FSYNC", "FLEAD"]
    by_code = screen.ARM_BY_CODE
    assert by_code["FPM"].parameter_matched_control
    assert by_code["FPM"].auxiliary_loss_weight == 0
    assert not by_code["FAUX"].condition_on_state
    assert by_code["FAUX"].auxiliary_loss_weight == 1
    assert by_code["FSYNC"].schedule_mode == "aligned"
    assert by_code["FLEAD"].schedule_mode == "tf_leads"
    assert by_code["FLEAD"].lead_logit == 1
    assert screen.NFE_GRID == [1, 2, 4, 8]
    assert screen.DEPLOYABLE_SOURCES == [
        "autonomous",
        "off",
        "autonomous_shuffled",
    ]
    parser = screen.build_parser()
    register = next(
        action
        for action in parser._actions
        if getattr(action, "dest", None) == "command"
    ).choices["register"]
    option_strings = {
        option
        for action in register._actions
        for option in action.option_strings
    }
    assert not any("test" in option for option in option_strings)


def test_frequency_screen_identity_is_content_bound():
    payload = screen.identity_payload({"schema": "unit", "value": 1})
    assert screen.validate_identity(payload)
    payload["value"] = 2
    assert not screen.validate_identity(payload)


def test_frequency_training_validation_contract_is_exact_val64_and_cycles():
    from omegaconf import OmegaConf

    config = OmegaConf.load(screen.COMMON_CONFIG)
    validation = config.trainer.config.validation
    loader = config.val_data_loader[0]
    max_iter = int(config.trainer.config.max_iter)
    val_every = int(validation.val_every)
    observed = {
        "dataset_infinite": bool(config.val_dataset.infinite),
        "dataset_seed": int(config.val_dataset.seed),
        "image_augmentation": bool(config.val_dataset.img_augment),
        "future_validity_enabled": bool(
            config.val_dataset.future_validity.enabled
        ),
        "future_validity_max_retries": int(
            config.val_dataset.future_validity.max_retries
        ),
        "single_iterator_reused": True,
        "drop_last": bool(loader.drop_last),
        "batch_size_per_rank": int(loader.batch_size),
        "loader_workers_per_rank": int(loader.num_workers),
        "persistent_workers": bool(loader.persistent_workers),
        "local_batches_per_event": int(validation.n_val_samples),
        "local_clips_per_event": int(loader.batch_size)
        * int(validation.n_val_samples),
        "global_clips_per_event": screen.WORLD_SIZE
        * int(loader.batch_size)
        * int(validation.n_val_samples),
        "iterations": [
            iteration
            for iteration in range(max_iter)
            if iteration % val_every == 0 or iteration + 1 == max_iter
        ],
        "one_complete_registered_validation_pass_per_event": True,
    }

    assert screen.validate_training_validation_contract(observed) == observed
    assert observed["global_clips_per_event"] == screen.EXPECTED_VALIDATION_CLIPS

    changed = dict(observed, local_batches_per_event=8)
    with pytest.raises(screen.ContractError, match="validation iterator"):
        screen.validate_training_validation_contract(changed)


def test_registration_rejects_self_signed_validation_contract_change(tmp_path):
    base = {
        "schema": screen.SCHEMA,
        "protected_test": {"allowed": False},
        "training": {
            "validation_iterator": dict(screen.TRAIN_VALIDATION_CONTRACT)
        },
    }
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(screen.identity_payload(base)), encoding="utf-8")
    assert screen.load_registration(path, verify_files=False)["schema"] == screen.SCHEMA

    changed = dict(screen.TRAIN_VALIDATION_CONTRACT)
    changed["local_batches_per_event"] = 8
    base["training"] = {"validation_iterator": changed}
    path.write_text(json.dumps(screen.identity_payload(base)), encoding="utf-8")
    with pytest.raises(screen.ContractError, match="validation iterator"):
        screen.load_registration(path, verify_files=False)


def test_standalone_evaluation_bounds_infinite_validation_to_four_batches():
    batches = [
        {"clip_index": torch.tensor([2 * index, 2 * index + 1])}
        for index in range(screen.TRAIN_VALIDATION_LOCAL_BATCHES_PER_EVENT)
    ]

    def infinite_batches():
        while True:
            yield from batches

    observed = list(evaluator._exact_validation_batches(infinite_batches()))
    assert len(observed) == screen.TRAIN_VALIDATION_LOCAL_BATCHES_PER_EVENT
    assert torch.cat([batch["clip_index"] for batch in observed]).tolist() == list(
        range(screen.TRAIN_VALIDATION_LOCAL_CLIPS_PER_EVENT)
    )

    with pytest.raises(evaluator.EvaluationError, match="ended before"):
        list(evaluator._exact_validation_batches(iter(batches[:3])))

    repeated = list(batches)
    repeated[-1] = {"clip_index": torch.tensor([0, 7])}
    with pytest.raises(evaluator.EvaluationError, match="repeats"):
        list(evaluator._exact_validation_batches(iter(repeated)))


def _manifest_row(index: int, *, split: str, episode: int | None = None):
    episode = index if episode is None else episode
    return {
        "clip_id": f"{index + (0 if split == 'train' else 10_000):064x}",
        "episode_dir": f"/episodes/{split}/{episode}",
        "start": index * 5,
        "auxiliary_index": index,
        "split": split,
    }


def test_manifest_identity_requires_episode_disjoint_splits():
    train = screen._manifest_identity(
        [_manifest_row(index, split="train") for index in range(2)],
        split="train",
        expected_clips=2,
    )
    validation = screen._manifest_identity(
        [_manifest_row(index, split="val") for index in range(2)],
        split="val",
        expected_clips=2,
    )
    assert screen._split_isolation(train, validation) == {
        "clip_id_overlap": 0,
        "episode_dir_overlap": 0,
        "episode_start_overlap": 0,
        "one_clip_per_episode": True,
    }

    leaked = screen.ManifestIdentity(
        clip_ids=validation.clip_ids,
        episode_dirs=frozenset(
            {*validation.episode_dirs, next(iter(train.episode_dirs))}
        ),
        episode_starts=validation.episode_starts,
        summary=validation.summary,
    )
    with pytest.raises(screen.ContractError, match="overlap"):
        screen._split_isolation(train, leaked)


def test_manifest_identity_rejects_repeated_episode():
    rows = [
        _manifest_row(0, split="val", episode=7),
        _manifest_row(1, split="val", episode=7),
    ]
    with pytest.raises(screen.ContractError, match="repeats an episode"):
        screen._manifest_identity(rows, split="val", expected_clips=2)


def test_validation_loader_uses_iterable_rank_sharding_without_sampler():
    class RankShard(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield from range(0, 8, 2)

    loader = evaluator._validation_loader(RankShard(), pin_memory=False)
    assert loader.batch_size == screen.TRAIN_VALIDATION_BATCH_SIZE_PER_RANK
    assert loader.drop_last is False
    assert [batch.tolist() for batch in loader] == [[0, 2], [4, 6]]


def test_merged_grid_requires_every_clip_source_nfe_exactly_once():
    rows = [
        {
            "arm": "FPM",
            "source": source,
            "nfe": nfe,
            "clip_index": clip,
        }
        for source in screen.SOURCES
        for nfe in screen.NFE_GRID
        for clip in range(screen.EXPECTED_VALIDATION_CLIPS)
    ]
    evaluator._validate_merged_grid(rows, arm_code="FPM")

    rows[-1] = dict(rows[-2])
    with pytest.raises(evaluator.EvaluationError, match="duplicate/missing"):
        evaluator._validate_merged_grid(rows, arm_code="FPM")


def test_nfe1_shuffle_is_an_exact_negative_control():
    fields = {
        "video_final_sha256": "a" * 64,
        "auxiliary_final_sha256": "b" * 64,
        "video_nmse": 1.0,
        "decoded_mse": 1.0,
        "temporal_mse": 1.0,
        "auxiliary_future_nmse": 1.0,
        "auxiliary_future_cosine": 0.0,
        "auxiliary_dc_nmse": 1.0,
        "auxiliary_motion_nmse": 1.0,
    }
    keyed = {
        (arm.code, source, 1, clip): dict(fields)
        for arm in screen.ARMS
        for source in ("autonomous", "autonomous_shuffled")
        for clip in range(screen.EXPECTED_VALIDATION_CLIPS)
    }
    screen._validate_nfe1_shuffle_negative_control(keyed)

    keyed[("FLEAD", "autonomous_shuffled", 1, 7)]["video_nmse"] = 1.1
    with pytest.raises(screen.ContractError, match="negative control"):
        screen._validate_nfe1_shuffle_negative_control(keyed)


def test_evaluation_receipt_binds_exact_rows_and_frozen_grid(tmp_path):
    output = tmp_path / "fpm_video_only"
    output.mkdir()
    rows_path = output / "rows.jsonl"
    rows_path.write_text('{"row": 1}\n', encoding="utf-8")
    registration_identity = "a" * 64
    receipt = screen.identity_payload(
        {
            "schema": screen.EVALUATION_COMPLETE_SCHEMA,
            "registration_identity_sha256": registration_identity,
            "arm_identity_sha256": "b" * 64,
            "arm": "FPM",
            "rows": (
                screen.EXPECTED_VALIDATION_CLIPS
                * len(screen.NFE_GRID)
                * len(screen.SOURCES)
            ),
            "validation_clips": screen.EXPECTED_VALIDATION_CLIPS,
            "nfe": screen.NFE_GRID,
            "sources": screen.SOURCES,
            "world_size": screen.WORLD_SIZE,
            "rows_sha256": screen.sha256_file(rows_path),
            "protected_test_accessed": False,
        }
    )
    (output / "complete.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )

    assert screen._validate_evaluation_receipt(
        rows_path,
        registration_identity_sha256=registration_identity,
    ) == receipt

    rows_path.write_text('{"row": 2}\n', encoding="utf-8")
    with pytest.raises(screen.ContractError, match="completion receipt"):
        screen._validate_evaluation_receipt(
            rows_path,
            registration_identity_sha256=registration_identity,
        )


def test_protocol_declares_exact_shapes_and_no_protected_test():
    protocol = screen.PROTOCOL.read_text(encoding="utf-8")
    assert "`[B,6,4,24,120]`" in protocol
    assert "`[B,16,4,24,120]`" in protocol
    assert "no protected-test phase" in protocol.lower()
    assert "exactly `NFE` Wan calls" in protocol
    assert "4 batches x 2 clips x 8 ranks = 64 clips" in protocol
    assert "fresh source commit" in protocol


def test_frequency_wandb_finish_is_bounded_and_preserves_local_sync_files():
    from omegaconf import OmegaConf

    config = OmegaConf.load(screen.COMMON_CONFIG)
    settings = OmegaConf.to_container(config.wandb.settings, resolve=True)
    assert settings == {
        "start_method": "thread",
        "save_code": False,
        "finish_timeout": 120.0,
        "finish_timeout_raises": False,
    }
    protocol = screen.PROTOCOL.read_text(encoding="utf-8")
    runbook = (
        screen.REPO_ROOT
        / "docs/experiments/VIDEO_FREQUENCY_FORCING_RUNBOOK.md"
    ).read_text(encoding="utf-8")
    assert "later `wandb sync`" in protocol
    assert "must not be deleted" in runbook


@pytest.mark.parametrize(
    "relative_path",
    [
        "tools/slurm/frequency_forcing_screen.sbatch",
        "tools/slurm/frequency_forcing_evaluate.sbatch",
    ],
)
def test_frequency_launcher_exports_registered_runtime_before_activation(
    relative_path,
):
    source = (screen.REPO_ROOT / relative_path).read_text(encoding="utf-8")
    activation = source.index('\nsource "')
    assert source.index('export WAN_DIR="$WAN_DIR_VALUE"') < activation
    assert source.index('export VIDEOX_HOME="$VIDEOX_HOME_VALUE"') < activation
    assert "/mnt/data2/" not in source


def test_frequency_training_workflow_binds_validation_contract_and_registration():
    source = screen.TRAIN_SBATCH.read_text(encoding="utf-8")
    assert '--registration "$FREQ_SCREEN_REGISTRATION"' in source
    assert '"val_dataset.infinite=true"' in source
    assert '"val_dataset.seed=1234"' in source
    assert '"val_dataset.img_augment=false"' in source
    assert '"val_dataset.future_validity.enabled=false"' in source
    assert '"val_dataset.future_validity.max_retries=0"' in source
    assert '"val_data_loader.0.batch_size=2"' in source
    assert '"val_data_loader.0.drop_last=false"' in source
    assert '"val_data_loader.0.num_workers=2"' in source
    assert '"val_data_loader.0.persistent_workers=false"' in source
    assert '"trainer.config.validation.n_val_samples=4"' in source

    preflight = screen.WARMSTART_PREFLIGHT.read_text(encoding="utf-8")
    assert "validate_training_validation_contract(" in preflight
    assert 'parser.add_argument("--registration", type=Path, required=True)' in preflight


def test_completion_receipt_must_be_final_and_identity_bound(tmp_path):
    snapshot = tmp_path / "snapshot.pt"
    identity = "a" * 64
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "completed_updates": screen.TRAIN_UPDATES,
        "max_iter": screen.TRAIN_UPDATES,
        "run_identity_sha256": identity,
        "snapshot": str(snapshot),
    }
    evaluator._validate_completion_receipt(
        receipt,
        snapshot_path=snapshot,
        arm_identity_sha256=identity,
    )
    interrupted = dict(receipt, completed_updates=screen.TRAIN_UPDATES - 1)
    with pytest.raises(evaluator.EvaluationError, match="completion receipt"):
        evaluator._validate_completion_receipt(
            interrupted,
            snapshot_path=snapshot,
            arm_identity_sha256=identity,
        )


def test_decoded_metrics_include_history_future_boundary():
    rgb = torch.zeros(2, 13, 3, 4, 5)
    decoded = torch.full((2, 3, 8, 4, 5), 128, dtype=torch.uint8)

    metrics = evaluator._decoded_metrics(decoded, rgb, future_frames=8)

    assert torch.equal(metrics["decoded_mse"], torch.zeros(2))
    assert torch.equal(metrics["temporal_mse"], torch.zeros(2))
    assert len(metrics["target_hash"]) == 2
