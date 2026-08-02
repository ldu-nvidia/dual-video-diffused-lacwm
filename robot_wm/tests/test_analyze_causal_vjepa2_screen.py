from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import analyze_causal_vjepa2_screen as analyzer


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(analyzer.canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _runtime_record():
    return {
        "utc": "2026-08-01T00:00:00+00:00",
        "hostname": "synthetic-b200",
        "python": "3.10.synthetic",
        "platform": "linux-synthetic",
        "torch_cuda": "12.8",
        "cuda_available": True,
        "packages": {"torch": "synthetic"},
        "slurm": {"SLURM_JOB_ID": "123"},
        "gpu_inventory": ["NVIDIA B200, synthetic"],
    }


def _provenance(config, *, command: str):
    return {
        "schema": config["schema"],
        "source": config["source"],
        "runtime": _runtime_record(),
        "command": [
            "/synthetic/python",
            config["entrypoint"]["path"],
            command,
        ],
        "resolved_config_sha256": analyzer.sha256_json(config),
        "secrets_persisted": False,
    }


def _manifest_record(path: Path, rows, split: str):
    record = analyzer.file_record(path)
    episodes = [int(row["episode_index"]) for row in rows]
    clip_ids = [str(row["clip_id"]) for row in rows]
    return {
        **record,
        "split": split,
        "clips": len(rows),
        "episodes": len(set(episodes)),
        "ordered_clip_ids_sha256": analyzer.sha256_json(clip_ids),
        "ordered_episode_ids_sha256": analyzer.sha256_json(episodes),
    }


def _cache_metadata(
    *,
    split: str,
    count: int,
    manifest,
    train_manifest,
    target,
    pca,
    teacher_checkpoint,
    source_license,
    implementation,
    base_droid,
    runtime,
):
    identity = {
        "schema": "droid-causal-vjepa2.1-v1",
        "artifact_type": analyzer.CACHE_ARTIFACT_TYPE,
        "target_kind": analyzer.TARGET_KIND,
        "split": split,
        "clip_count": count,
        "manifest_sha256": manifest["sha256"],
        "train_manifest_sha256": train_manifest["sha256"],
        "pca_sha256": pca["sha256"],
        "pca_file": pca["path"],
        "source_commit": "d" * 40,
        "checkpoint_sha256": teacher_checkpoint["sha256"],
        "checkpoint_evidence": teacher_checkpoint,
        "source_archive_sha256": _digest("teacher-source-archive"),
        "source_license": source_license,
        "implementation": implementation,
        "base_droid": base_droid,
        "runtime": runtime,
        "numerical_contract": dict(analyzer.PRODUCTION_NUMERICAL_CONTRACT),
        "auxiliary_target_shape": list(analyzer.TARGET_SHAPE),
        "target_shape": [count, *analyzer.TARGET_SHAPE],
        "target_dtype": "float16",
        "teacher_size": [384, 672],
        "teacher_frames": 16,
        "last_temporal_token": 7,
        "pooled_token_grid": [8, 14],
        "world_size": 1,
        "protected_test_access": False,
        "allowed_splits": ["train", "val"],
        "test_rows_extracted": 0,
    }
    return {
        "format_version": 1,
        **identity,
        "artifact_identity": identity,
        "cache_id": analyzer.sha256_json(identity),
        "complete": True,
        "target_file": target["path"],
        "target_sha256": target["sha256"],
        "evidence": {
            "target": target,
            "manifest": analyzer._base_file_record(manifest),
            "train_manifest": analyzer._base_file_record(train_manifest),
            "pca": pca,
            "checkpoint": teacher_checkpoint,
            "source_license": source_license,
        },
    }


def _run_config(*, command: str, source, datasets, entrypoint, dataset_source, calibration):
    calibrating = command == "calibrate"
    config = {
        "schema": analyzer.RUN_SCHEMA,
        "source": source,
        "entrypoint": entrypoint,
        "dataset_source": dataset_source,
        "command": command,
        "target_kind": analyzer.TARGET_KIND,
        "cache_artifact_type": analyzer.CACHE_ARTIFACT_TYPE,
        "target_shape": list(analyzer.TARGET_SHAPE),
        "clock_convention": analyzer.CLOCK_CONVENTION,
        "clean_time_epsilon": analyzer.CLEAN_TIME_EPSILON,
        "updates": analyzer.CALIBRATION_UPDATES if calibrating else analyzer.TRAIN_UPDATES,
        "checkpoint_updates": (
            [analyzer.CALIBRATION_UPDATES]
            if calibrating
            else list(analyzer.TRAIN_CHECKPOINTS)
        ),
        "seed": analyzer.TRAIN_SEED,
        "global_batch_size": analyzer.GLOBAL_BATCH_SIZE,
        "world_size": 1,
        "local_optimizer_batch_size": analyzer.GLOBAL_BATCH_SIZE,
        "micro_batch_size_per_rank": analyzer.GLOBAL_BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "dtype": "bfloat16",
        "target_storage_dtype": "float16",
        "target_flow_compute_dtype": "float32",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 5e-5,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "warmup_updates": 500,
            "after_warmup": "constant",
            "gradient_clip_norm": 1.0,
        },
        "ema": {
            "decay": 0.9999,
            "schedule": analyzer.EMA_SCHEDULE,
            "reported_samples_use": True,
            "short_run_initialization_bias_corrected": True,
        },
        "phase1_schedule": dict(analyzer.PHASE1_SCHEDULE),
        "model": dict(analyzer.MODEL_CONFIG),
        "parameter_count": analyzer.MODEL_PARAMETER_COUNT,
        "datasets": datasets,
        "data_root": "/mnt/data1/synthetic-droid",
        "semantic_cache_root": "/mnt/data1/synthetic-causal-vjepa2",
        "workers_per_rank": 0,
        "calibration_record": calibration,
        "wandb": {
            "enabled": False,
            "entity": None,
            "project": None,
            "group": None,
            "private_project_acknowledged": False,
        },
    }
    config["experiment_identity_sha256"] = analyzer.sha256_json(
        {key: config[key] for key in analyzer.TRAINING_IDENTITY_KEYS}
    )
    return config


def _metric_values(control: str):
    nmse, cosine, temporal, temporal_cosine = {
        "autonomous": (0.30, 0.82, 0.30, 0.80),
        "donor_target": (0.52, 0.55, 0.52, 0.53),
        "context_shuffled": (0.48, 0.60, 0.48, 0.58),
        "history_shuffled": (0.42, 0.67, 0.42, 0.65),
        "actions_shuffled": (0.40, 0.69, 0.40, 0.67),
        "zero": (1.0, 0.0, 1.0, 0.0),
        "oracle_clean": (0.0, 1.0, 0.0, 1.0),
    }[control]
    return {
        "semantic_nmse": nmse,
        "semantic_token_cosine": cosine,
        "temporal_difference_nmse": temporal,
        "temporal_difference_token_cosine": temporal_cosine,
        "retained_utility": 1.0 - nmse,
        "temporal_retained_utility": 1.0 - temporal,
    }


def _metric_rows(*, clips, checkpoint_sha256, training_config_sha256):
    clip_ids = [str(row["clip_id"]) for row in clips]
    episodes = {str(row["clip_id"]): int(row["episode_index"]) for row in clips}
    inputs = {
        clip_id: {
            "history_sha256": _digest(f"history:{clip_id}"),
            "actions_sha256": _digest(f"actions:{clip_id}"),
            "initial_video_noise_sha256": _digest(f"video-noise:{clip_id}"),
            "initial_auxiliary_noise_sha256": _digest(f"aux-noise:{clip_id}"),
        }
        for clip_id in clip_ids
    }
    targets = {clip_id: _digest(f"target:{clip_id}") for clip_id in clip_ids}
    records = []
    for nfe in analyzer.NFE_GRID:
        autonomous_predictions = {
            clip_id: _digest(f"prediction:{nfe}:autonomous:{clip_id}")
            for clip_id in clip_ids
        }
        autonomous_calls = {
            clip_id: [
                _digest(f"call:{nfe}:autonomous:{clip_id}:{step}")
                for step in range(nfe)
            ]
            for clip_id in clip_ids
        }
        for control in analyzer.CONTROLS:
            for index, clip_id in enumerate(clip_ids):
                donor_id = clip_ids[index ^ 1]
                history_source = (
                    donor_id
                    if control in {"context_shuffled", "history_shuffled"}
                    else clip_id
                )
                action_source = (
                    donor_id
                    if control in {"context_shuffled", "actions_shuffled"}
                    else clip_id
                )
                input_record = {
                    "history_sha256": inputs[history_source]["history_sha256"],
                    "actions_sha256": inputs[action_source]["actions_sha256"],
                    "initial_video_noise_sha256": inputs[clip_id][
                        "initial_video_noise_sha256"
                    ],
                    "initial_auxiliary_noise_sha256": inputs[clip_id][
                        "initial_auxiliary_noise_sha256"
                    ],
                }
                if control == "donor_target":
                    call_hashes = autonomous_calls[clip_id]
                    generated = autonomous_predictions[clip_id]
                elif control in analyzer.GENERATED_CONTROLS:
                    call_hashes = (
                        autonomous_calls[clip_id]
                        if control == "autonomous"
                        else [
                            _digest(f"call:{nfe}:{control}:{clip_id}:{step}")
                            for step in range(nfe)
                        ]
                    )
                    generated = (
                        autonomous_predictions[clip_id]
                        if control == "autonomous"
                        else _digest(f"prediction:{nfe}:{control}:{clip_id}")
                    )
                else:
                    call_hashes = []
                    generated = (
                        targets[clip_id]
                        if control == "oracle_clean"
                        else _digest(f"prediction:{nfe}:{control}:{clip_id}")
                    )
                conceptual_calls = (
                    nfe
                    if control in {*analyzer.GENERATED_CONTROLS, "donor_target"}
                    else 0
                )
                actual_calls = nfe if control in analyzer.GENERATED_CONTROLS else 0
                records.append(
                    {
                        "clip_id": clip_id,
                        "episode_index": episodes[clip_id],
                        "donor_clip_id": donor_id,
                        "donor_episode_index": episodes[donor_id],
                        "control": control,
                        "nfe": nfe,
                        "evaluation_seed": analyzer.EVALUATION_SEED,
                        "checkpoint_sha256": checkpoint_sha256,
                        "training_config_sha256": training_config_sha256,
                        "source_clip_id": (
                            donor_id if control == "context_shuffled" else clip_id
                        ),
                        "target_source_clip_id": (
                            donor_id if control == "donor_target" else clip_id
                        ),
                        **input_record,
                        "sampler_input_sha256": analyzer.sha256_json(
                            {"schema": "causal-vjepa2-sampler-input-v1", **input_record}
                        ),
                        "model_call_input_sha256": call_hashes,
                        "model_call_input_chain_sha256": analyzer.sha256_json(
                            call_hashes
                        ),
                        "conceptual_path_model_calls": conceptual_calls,
                        "actual_evaluator_model_calls": actual_calls,
                        "generation_reused_from": (
                            "autonomous" if control == "donor_target" else None
                        ),
                        "generated_auxiliary_sha256": generated,
                        "metric_target_sha256": targets[
                            donor_id if control == "donor_target" else clip_id
                        ],
                        "generation_deployable": control
                        in {*analyzer.GENERATED_CONTROLS, "donor_target"},
                        "control_deployable": control in analyzer.GENERATED_CONTROLS,
                        "metric_comparison_only": control
                        in {"donor_target", "zero", "oracle_clean"},
                        "metric_target_available_at_inference": False,
                        "clean_future_target_entered_sampler": False,
                        "teacher_model_calls": 0,
                        **_metric_values(control),
                    }
                )
    return sorted(records, key=lambda row: (row["nfe"], row["control"], row["clip_id"]))


def _synthetic_evaluation(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(analyzer, "CLIPS", 2)
    monkeypatch.setattr(analyzer, "TRAIN_CLIPS", 4)
    monkeypatch.setattr(analyzer, "TRAIN_EPISODES", 2)
    monkeypatch.setattr(analyzer, "BOOTSTRAP_SAMPLES", 100)
    source = {"commit": "c" * 40, "dirty": False}
    monkeypatch.setattr(analyzer, "git_record", lambda: dict(source))
    train_rows = [
        {"clip_id": f"train-{index}", "episode_index": index // 2, "split": "train"}
        for index in range(4)
    ]
    val_rows = [
        {"clip_id": f"val-{index}", "episode_index": 10 + index, "split": "val"}
        for index in range(2)
    ]

    def read_manifest(path, *, expected_split):
        del path
        return train_rows if expected_split == "train" else val_rows

    monkeypatch.setattr(analyzer, "read_clip_manifest", read_manifest)
    files = tmp_path / "files"
    files.mkdir()
    train_manifest_path = files / "train.jsonl"
    val_manifest_path = files / "val.jsonl"
    _write_jsonl(train_manifest_path, train_rows)
    _write_jsonl(val_manifest_path, val_rows)
    train_manifest = _manifest_record(train_manifest_path, train_rows, "train")
    val_manifest = _manifest_record(val_manifest_path, val_rows, "val")

    artifact_paths = {}
    for name in (
        "train-target",
        "val-target",
        "pca",
        "teacher-checkpoint",
        "license",
        "builder-source",
        "dataset-source",
        "screen-entrypoint",
    ):
        path = files / f"{name}.bin"
        path.write_bytes(name.encode("utf-8"))
        artifact_paths[name] = analyzer.file_record(path)
    implementation = {
        "repo_commit": "e" * 40,
        "builder_source": artifact_paths["builder-source"],
        "dataset_source": artifact_paths["dataset-source"],
    }
    base_droid = {
        "schema": "frozen-droid-video-latent-forcing-poc-v1",
        "provenance": artifact_paths["license"],
        "manifests": {
            "train": analyzer._base_file_record(train_manifest),
            "val": analyzer._base_file_record(val_manifest),
        },
        "split_disjoint": True,
        "protected_test_payload_cached": False,
    }
    runtime = {
        "python": "3.synthetic",
        "torch": "synthetic",
        "cuda": "synthetic",
        "numpy": "synthetic",
    }
    train_cache = _cache_metadata(
        split="train",
        count=4,
        manifest=train_manifest,
        train_manifest=train_manifest,
        target=artifact_paths["train-target"],
        pca=artifact_paths["pca"],
        teacher_checkpoint=artifact_paths["teacher-checkpoint"],
        source_license=artifact_paths["license"],
        implementation=implementation,
        base_droid=base_droid,
        runtime=runtime,
    )
    val_cache = _cache_metadata(
        split="val",
        count=2,
        manifest=val_manifest,
        train_manifest=train_manifest,
        target=artifact_paths["val-target"],
        pca=artifact_paths["pca"],
        teacher_checkpoint=artifact_paths["teacher-checkpoint"],
        source_license=artifact_paths["license"],
        implementation=implementation,
        base_droid=base_droid,
        runtime=runtime,
    )
    datasets = {
        "train": train_manifest,
        "validation": val_manifest,
        "semantic_cache": {"train": train_cache, "validation": val_cache},
    }

    calibration_dir = tmp_path / "calibration"
    calibration_config = _run_config(
        command="calibrate",
        source=source,
        datasets=datasets,
        entrypoint=artifact_paths["screen-entrypoint"],
        dataset_source=artifact_paths["dataset-source"],
        calibration=None,
    )
    calibration_config_path = calibration_dir / "resolved_config.json"
    _write_json(calibration_config_path, calibration_config)
    calibration_provenance_path = calibration_dir / "provenance.json"
    _write_json(
        calibration_provenance_path,
        _provenance(calibration_config, command="calibrate"),
    )
    calibration_checkpoint_path = calibration_dir / "checkpoints" / "update_000200.pt"
    calibration_checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "schema": "video-latent-forcing-poc-checkpoint-v1",
            "arm": "phase1",
            "completed_updates": analyzer.CALIBRATION_UPDATES,
            "model_config": dict(analyzer.MODEL_CONFIG),
            "config_sha256": analyzer.sha256_json(calibration_config),
            "ema": {
                "decay": 0.9999,
                "schedule": analyzer.EMA_SCHEDULE,
                "num_updates": analyzer.CALIBRATION_UPDATES,
                "shadow": {},
            },
        },
        calibration_checkpoint_path,
    )
    calibration_complete = {
        "schema": analyzer.RUN_SCHEMA,
        "status": "complete",
        "command": "calibrate",
        "completed_updates": analyzer.CALIBRATION_UPDATES,
        "nonfinite_updates": 0,
        "only_supervised_target": "auxiliary_target",
        "video_loss_enabled": False,
        "clock_convention": analyzer.CLOCK_CONVENTION,
        "parameter_counts": {
            "total": analyzer.MODEL_PARAMETER_COUNT,
            "trainable": analyzer.MODEL_PARAMETER_COUNT,
        },
        "experiment_identity_sha256": calibration_config[
            "experiment_identity_sha256"
        ],
        "source": source,
        "resolved_config_sha256": analyzer.sha256_json(calibration_config),
        "resolved_config": analyzer.file_record(calibration_config_path),
        "provenance": analyzer.file_record(calibration_provenance_path),
        "checkpoint": analyzer.file_record(calibration_checkpoint_path),
    }
    calibration_complete_path = calibration_dir / "complete.json"
    _write_json(calibration_complete_path, calibration_complete)

    training_dir = tmp_path / "training"
    training_config = _run_config(
        command="train",
        source=source,
        datasets=datasets,
        entrypoint=artifact_paths["screen-entrypoint"],
        dataset_source=artifact_paths["dataset-source"],
        calibration=analyzer.file_record(calibration_complete_path),
    )
    assert (
        training_config["experiment_identity_sha256"]
        == calibration_config["experiment_identity_sha256"]
    )
    training_config_path = training_dir / "resolved_config.json"
    _write_json(training_config_path, training_config)
    training_provenance_path = training_dir / "provenance.json"
    _write_json(
        training_provenance_path, _provenance(training_config, command="train")
    )
    checkpoint_path = training_dir / "checkpoints" / "update_005000.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "schema": "video-latent-forcing-poc-checkpoint-v1",
            "arm": "phase1",
            "completed_updates": analyzer.TRAIN_UPDATES,
            "model_config": dict(analyzer.MODEL_CONFIG),
            "config_sha256": analyzer.sha256_json(training_config),
            "ema": {
                "decay": 0.9999,
                "schedule": analyzer.EMA_SCHEDULE,
                "num_updates": analyzer.TRAIN_UPDATES,
                "shadow": {},
            },
        },
        checkpoint_path,
    )
    training_config_record = analyzer.file_record(training_config_path)
    checkpoint_record = analyzer.file_record(checkpoint_path)
    _write_json(
        training_dir / "complete.json",
        {
            "schema": analyzer.RUN_SCHEMA,
            "status": "complete",
            "command": "train",
            "completed_updates": analyzer.TRAIN_UPDATES,
            "nonfinite_updates": 0,
            "only_supervised_target": "auxiliary_target",
            "video_loss_enabled": False,
            "parameter_counts": {
                "total": analyzer.MODEL_PARAMETER_COUNT,
                "trainable": analyzer.MODEL_PARAMETER_COUNT,
            },
            "resolved_config": training_config_record,
            "provenance": analyzer.file_record(training_provenance_path),
            "checkpoint": checkpoint_record,
            "resolved_config_sha256": analyzer.sha256_json(training_config),
        },
    )

    evaluation_root = tmp_path / "evaluation"
    evaluation_root.mkdir()
    donor = {"val-0": "val-1", "val-1": "val-0"}
    config = {
        "schema": analyzer.EVALUATION_SCHEMA,
        "source": source,
        "entrypoint": artifact_paths["screen-entrypoint"],
        "dataset_source": artifact_paths["dataset-source"],
        "target_kind": analyzer.TARGET_KIND,
        "cache_artifact_type": analyzer.CACHE_ARTIFACT_TYPE,
        "target_storage_dtype": "float16",
        "target_flow_compute_dtype": "float32",
        "target_shape": list(analyzer.TARGET_SHAPE),
        "clock_convention": analyzer.CLOCK_CONVENTION,
        "split": "val",
        "validation_clips": 2,
        "checkpoint": checkpoint_record,
        "training_config": training_config_record,
        "checkpoint_update": analyzer.TRAIN_UPDATES,
        "weights": {
            "kind": "ema",
            "decay": 0.9999,
            "schedule": analyzer.EMA_SCHEDULE,
            "updates": analyzer.TRAIN_UPDATES,
        },
        "manifest": val_manifest,
        "semantic_cache": val_cache,
        "data_root": "/mnt/data1/synthetic-droid",
        "semantic_cache_root": "/mnt/data1/synthetic-causal-vjepa2",
        "seed": analyzer.EVALUATION_SEED,
        "nfe_grid": list(analyzer.NFE_GRID),
        "controls": list(analyzer.CONTROLS),
        "fixed_clip_noise": True,
        "fixed_noise_key": "sha256(f'{clip_id}:{eval_seed}:video|aux')",
        "donor_rule": "manifest-adjacent xor-1; pairs are episode-disjoint",
        "donor_mapping_sha256": analyzer.sha256_json(donor),
        "clean_target_sampler_policy": (
            "forbidden for every generated control; oracle-clean is metric-only"
        ),
        "world_size": 1,
        "eval_batch_size": 2,
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cublas_workspace_config": ":4096:8",
            "nvidia_tf32_override": "0",
            "torch_allow_tf32_cublas_override": "0",
            "autocast": "cuda-bfloat16",
        },
        "wandb": {
            "enabled": False,
            "entity": None,
            "project": None,
            "group": None,
            "private_project_acknowledged": False,
        },
    }
    _write_json(evaluation_root / "resolved_config.json", config)
    evaluation_provenance_path = evaluation_root / "provenance.json"
    _write_json(evaluation_provenance_path, _provenance(config, command="eval"))
    records = _metric_rows(
        clips=val_rows,
        checkpoint_sha256=checkpoint_record["sha256"],
        training_config_sha256=training_config_record["sha256"],
    )
    merged_path = evaluation_root / "per_clip_metrics.jsonl"
    shard_path = evaluation_root / "rank_metrics" / "rank_0000.jsonl"
    _write_jsonl(merged_path, records)
    _write_jsonl(shard_path, records)
    actual_calls = len(analyzer.GENERATED_CONTROLS) * sum(analyzer.NFE_GRID)
    summary = {
        "schema": analyzer.EVALUATION_SCHEMA,
        "status": "complete",
        "target_kind": analyzer.TARGET_KIND,
        "split": "val",
        "checkpoint": checkpoint_record,
        "training_config": training_config_record,
        "manifest": val_manifest,
        "semantic_cache": val_cache,
        "provenance": analyzer.file_record(evaluation_provenance_path),
        "record_count": len(records),
        "cell_count": len(analyzer.NFE_GRID) * len(analyzer.CONTROLS),
        "summaries": [
            {
                "nfe": nfe,
                "control": control,
                "clips": 2,
                **_metric_values(control),
            }
            for nfe in analyzer.NFE_GRID
            for control in analyzer.CONTROLS
        ],
        "per_clip_metrics": analyzer.file_record(merged_path),
        "rank_shards": [
            {
                "rank": 0,
                "records": len(records),
                "actual_batched_transformer_calls": actual_calls,
                "file": analyzer.file_record(shard_path),
            }
        ],
        "actual_batched_transformer_calls": actual_calls,
        "donor_target_generation_reused_bit_identically": True,
        "clean_future_target_entered_deployable_sampler": False,
        "teacher_model_calls": 0,
        "zero_nmse_reference": 1.0,
        "oracle_clean_nmse_reference": 0.0,
    }
    summary_path = evaluation_root / "summary.json"
    _write_json(summary_path, summary)
    return (
        evaluation_root,
        shard_path,
        summary_path,
        Path(train_cache["evidence"]["target"]["path"]),
    )


def _gate_values(clips: int, *, autonomous_nmse: float = 0.30):
    def cell(nmse, cosine, temporal):
        return {
            "semantic_nmse": np.full(clips, nmse, dtype=np.float64),
            "semantic_token_cosine": np.full(clips, cosine, dtype=np.float64),
            "temporal_difference_nmse": np.full(clips, temporal, dtype=np.float64),
            "temporal_difference_token_cosine": np.full(
                clips, cosine - 0.02, dtype=np.float64
            ),
            "retained_utility": np.full(clips, 1.0 - nmse, dtype=np.float64),
            "temporal_retained_utility": np.full(
                clips, 1.0 - temporal, dtype=np.float64
            ),
        }

    return {
        "autonomous": cell(autonomous_nmse, 0.82, 0.30),
        "donor_target": cell(0.52, 0.55, 0.52),
        "context_shuffled": cell(0.48, 0.60, 0.48),
        "history_shuffled": cell(0.42, 0.67, 0.42),
        "actions_shuffled": cell(0.40, 0.69, 0.40),
        "zero": cell(1.0, 0.0, 1.0),
        "oracle_clean": cell(0.0, 1.0, 0.0),
    }


def test_gate_cell_passes_only_with_absolute_and_paired_advantages(monkeypatch):
    monkeypatch.setattr(analyzer, "CLIPS", 32)
    monkeypatch.setattr(analyzer, "BOOTSTRAP_SAMPLES", 500)
    passed = analyzer.gate_cell(4, _gate_values(32))
    assert passed["passed"] is True
    assert len(passed["checks"]) == 9
    assert all(check["passed"] for check in passed["checks"])
    failed = analyzer.gate_cell(4, _gate_values(32, autonomous_nmse=0.60))
    assert failed["passed"] is False
    assert next(
        check for check in failed["checks"] if check["id"] == "autonomous_semantic_nmse"
    )["passed"] is False


def test_paired_bootstrap_is_deterministic(monkeypatch):
    monkeypatch.setattr(analyzer, "CLIPS", 16)
    monkeypatch.setattr(analyzer, "BOOTSTRAP_SAMPLES", 200)
    candidate = np.linspace(0.1, 0.2, 16)
    reference = np.linspace(0.2, 0.4, 16)
    first = analyzer.paired_bootstrap(
        candidate,
        reference,
        label="unit-relative",
        statistic=analyzer.relative_improvement,
    )
    second = analyzer.paired_bootstrap(
        candidate,
        reference,
        label="unit-relative",
        statistic=analyzer.relative_improvement,
    )
    assert first == second
    assert first["estimate"] == pytest.approx(0.5)
    assert first["ci_low"] > 0.0


def test_row_audit_distinguishes_deployable_generation_from_metric_control():
    component = {
        "history_sha256": "1" * 64,
        "actions_sha256": "2" * 64,
        "initial_video_noise_sha256": "3" * 64,
        "initial_auxiliary_noise_sha256": "4" * 64,
    }
    calls = ["5" * 64, "6" * 64]
    row = {
        "clip_id": "clip",
        "episode_index": 1,
        "donor_clip_id": "donor",
        "donor_episode_index": 2,
        "control": "donor_target",
        "nfe": 2,
        "evaluation_seed": analyzer.EVALUATION_SEED,
        "checkpoint_sha256": "7" * 64,
        "training_config_sha256": "8" * 64,
        "source_clip_id": "clip",
        "target_source_clip_id": "donor",
        **component,
        "sampler_input_sha256": analyzer.sha256_json(
            {"schema": "causal-vjepa2-sampler-input-v1", **component}
        ),
        "model_call_input_sha256": calls,
        "model_call_input_chain_sha256": analyzer.sha256_json(calls),
        "conceptual_path_model_calls": 2,
        "actual_evaluator_model_calls": 0,
        "generation_reused_from": "autonomous",
        "generated_auxiliary_sha256": "9" * 64,
        "metric_target_sha256": "a" * 64,
        "generation_deployable": True,
        "control_deployable": False,
        "metric_comparison_only": True,
        "metric_target_available_at_inference": False,
        "clean_future_target_entered_sampler": False,
        "teacher_model_calls": 0,
        "semantic_nmse": 0.5,
        "semantic_token_cosine": 0.5,
        "temporal_difference_nmse": 0.5,
        "temporal_difference_token_cosine": 0.5,
        "retained_utility": 0.5,
        "temporal_retained_utility": 0.5,
    }
    analyzer._validate_row_common(
        row,
        nfe=2,
        control="donor_target",
        clip_id="clip",
        episode_index=1,
        checkpoint_sha256="7" * 64,
        training_config_sha256="8" * 64,
    )
    row["control_deployable"] = True
    with pytest.raises(analyzer.GateError, match="call accounting"):
        analyzer._validate_row_common(
            row,
            nfe=2,
            control="donor_target",
            clip_id="clip",
            episode_index=1,
            checkpoint_sha256="7" * 64,
            training_config_sha256="8" * 64,
        )


def test_full_analyzer_accepts_bound_evidence_and_rejects_rehashed_shard_tamper(
    tmp_path, monkeypatch
):
    evaluation_root, shard_path, summary_path, train_target_path = _synthetic_evaluation(
        tmp_path, monkeypatch
    )
    evaluation_config = analyzer.load_json(
        evaluation_root / "resolved_config.json", "evaluation config"
    )
    training_config = analyzer.load_json(
        evaluation_config["training_config"]["path"], "training config"
    )
    caches = training_config["datasets"]["semantic_cache"]
    changed_val_cache = copy.deepcopy(caches["validation"])
    different_pca_path = tmp_path / "files" / "different-pca.bin"
    different_pca_path.write_bytes(b"different-pca")
    different_pca = analyzer.file_record(different_pca_path)
    changed_val_cache["pca_sha256"] = different_pca["sha256"]
    changed_val_cache["pca_file"] = different_pca["path"]
    changed_val_cache["evidence"]["pca"] = different_pca
    changed_val_cache["artifact_identity"]["pca_sha256"] = different_pca[
        "sha256"
    ]
    changed_val_cache["artifact_identity"]["pca_file"] = different_pca["path"]
    changed_val_cache["cache_id"] = analyzer.sha256_json(
        changed_val_cache["artifact_identity"]
    )
    with pytest.raises(analyzer.GateError, match="do not share one PCA"):
        analyzer._validate_training_cache_pair(
            caches["train"],
            changed_val_cache,
            train_manifest=training_config["datasets"]["train"],
            validation_manifest=training_config["datasets"]["validation"],
        )

    result = analyzer.analyze_evaluation(evaluation_root)
    assert result["status"] == "pass"
    assert result["selected_nfe"] == 1

    train_target_path.write_bytes(b"tampered-train-target")
    with pytest.raises(analyzer.GateError, match="train semantic cache evidence changed"):
        analyzer.analyze_evaluation(evaluation_root)
    train_target_path.write_bytes(b"train-target")

    shard_rows = analyzer._read_jsonl(shard_path)
    shard_rows[0]["semantic_nmse"] = 0.123456
    _write_jsonl(shard_path, shard_rows)
    summary = analyzer.load_json(summary_path, "summary")
    summary["rank_shards"][0]["file"] = analyzer.file_record(shard_path)
    _write_json(summary_path, summary)
    with pytest.raises(analyzer.GateError, match="shard differs from merged metrics"):
        analyzer.analyze_evaluation(evaluation_root)


def test_full_analyzer_rejects_cross_nfe_noise_tamper(tmp_path, monkeypatch):
    evaluation_root, shard_path, summary_path, _ = _synthetic_evaluation(
        tmp_path, monkeypatch
    )
    merged_path = evaluation_root / "per_clip_metrics.jsonl"
    rows = analyzer._read_jsonl(merged_path)
    for row in rows:
        if row["nfe"] == 2:
            row["initial_auxiliary_noise_sha256"] = "f" * 64
            row["sampler_input_sha256"] = analyzer._sampler_input_sha256(row)
    _write_jsonl(merged_path, rows)
    _write_jsonl(shard_path, rows)
    summary = analyzer.load_json(summary_path, "summary")
    summary["per_clip_metrics"] = analyzer.file_record(merged_path)
    summary["rank_shards"][0]["file"] = analyzer.file_record(shard_path)
    _write_json(summary_path, summary)
    with pytest.raises(analyzer.GateError, match="cross-NFE fixed noise/target"):
        analyzer.analyze_evaluation(evaluation_root)


def test_full_analyzer_requires_semantic_evaluation_runtime_provenance(
    tmp_path, monkeypatch
):
    evaluation_root, _, summary_path, _ = _synthetic_evaluation(tmp_path, monkeypatch)
    provenance_path = evaluation_root / "provenance.json"
    provenance = analyzer.load_json(provenance_path, "evaluation provenance")
    provenance.pop("runtime")
    _write_json(provenance_path, provenance)
    summary = analyzer.load_json(summary_path, "summary")
    summary["provenance"] = analyzer.file_record(provenance_path)
    _write_json(summary_path, summary)
    with pytest.raises(analyzer.GateError, match="lacks a runtime record"):
        analyzer.analyze_evaluation(evaluation_root)


def test_full_analyzer_requires_semantic_training_runtime_provenance(
    tmp_path, monkeypatch
):
    evaluation_root, _, _, _ = _synthetic_evaluation(tmp_path, monkeypatch)
    evaluation_config = analyzer.load_json(
        evaluation_root / "resolved_config.json", "evaluation config"
    )
    training_config_path = Path(evaluation_config["training_config"]["path"])
    training_root = training_config_path.parent
    provenance_path = training_root / "provenance.json"
    provenance = analyzer.load_json(provenance_path, "training provenance")
    provenance.pop("runtime")
    _write_json(provenance_path, provenance)
    complete_path = training_root / "complete.json"
    complete = analyzer.load_json(complete_path, "training completion")
    complete["provenance"] = analyzer.file_record(provenance_path)
    _write_json(complete_path, complete)
    with pytest.raises(analyzer.GateError, match="lacks a runtime record"):
        analyzer.analyze_evaluation(evaluation_root)


def test_full_analyzer_rechecks_cache_evidence_after_gate_math(tmp_path, monkeypatch):
    evaluation_root, _, _, train_target_path = _synthetic_evaluation(
        tmp_path, monkeypatch
    )
    original_gate_cell = analyzer.gate_cell
    changed = False

    def mutate_after_snapshot(nfe, values):
        nonlocal changed
        if not changed:
            train_target_path.write_bytes(b"changed-after-cache-snapshot")
            changed = True
        return original_gate_cell(nfe, values)

    monkeypatch.setattr(analyzer, "gate_cell", mutate_after_snapshot)
    with pytest.raises(analyzer.GateError, match="evidence changed while"):
        analyzer.analyze_evaluation(evaluation_root)
