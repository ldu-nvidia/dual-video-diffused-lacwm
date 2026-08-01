from __future__ import annotations

import copy
import contextlib
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import qualify_video_latent_forcing_determinism as qualification
from tools import video_latent_forcing_poc as vlf


def _topology(job_id: str = "101", node: str = "node-a") -> dict:
    return {
        "slurm_job_id": job_id,
        "slurm_nodelist": node,
        "nodes_requested": 1,
        "world_size": 8,
        "per_rank_batch_size": 8,
        "global_probe_clips": 64,
        "nodes": [node],
        "ranks": [
            {
                "rank": rank,
                "local_rank": rank,
                "hostname": node,
                "device_name": "NVIDIA B200",
                "compute_capability": [10, 0],
                "cuda_visible_devices": "0,1,2,3,4,5,6,7",
            }
            for rank in range(8)
        ],
        "visible_gpu_inventory": [
            {
                "uuid": f"GPU-{node}-{rank}",
                "name": "NVIDIA B200",
                "driver": "1",
                "memory_mib": "1",
            }
            for rank in range(8)
        ],
    }


def _expected_probe_indexes() -> list[int]:
    indexes = []
    for rank in range(8):
        local, batches = vlf.paired_rank_evaluation_layout(
            890, 8, rank=rank, world_size=8
        )
        indexes.extend(local[position] for position in batches[0])
    return sorted(indexes)


def _hex(*values) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()


def _clip_id(index: int) -> str:
    return _hex("clip", index)


def _probe(output_seed: str = "eight") -> dict:
    rank_records = []
    clips = []
    for rank in range(8):
        local, batches = vlf.paired_rank_evaluation_layout(
            890, 8, rank=rank, world_size=8
        )
        indexes = [local[position] for position in batches[0]]
        rank_clips = []
        for offset, index in enumerate(indexes):
            output_hash = _hex(output_seed, index)
            hashes = {
                control: output_hash for control in qualification.CONTROLS
            }
            rank_clips.append(
                {
                    "manifest_index": index,
                    "clip_id": _clip_id(index),
                    "history_sha256": _hex("history", index),
                    "actions_sha256": _hex("actions", index),
                    "target_video_sha256": _hex("target", index),
                    "target_auxiliary_sha256": _hex("aux-target", index),
                    "initial_video_noise_sha256": _hex("noise", index),
                    "initial_auxiliary_noise_sha256": _hex("aux-noise", index),
                    "shuffle_source_clip_id": _clip_id(indexes[offset ^ 1]),
                    "phase_boundary_sha256": _hex("boundary", index),
                    "generated_video_sha256_by_control": hashes,
                }
            )
        batch_output = _hex(output_seed, "batch", rank)
        rank_records.append(
            {
                "rank": rank,
                "batch_generated_video_sha256_by_control": {
                    control: batch_output for control in qualification.CONTROLS
                },
                "batch_phase_boundary_sha256": _hex("batch-boundary", rank),
                "clips": rank_clips,
            }
        )
        clips.extend(rank_clips)
    clips.sort(key=lambda row: row["manifest_index"])
    return {
        "arm": "A1",
        "weights": "ema",
        "ema_updates": 200,
        "nfe_pair": [25, 25],
        "controls": list(qualification.CONTROLS),
        "autocast": "cuda-bfloat16",
        "qualified_clips": 64,
        "ranks": rank_records,
        "clips": clips,
    }


def _job(tmp_path, *, job_id: str, node: str, output_hash: str = "8" * 64) -> dict:
    common = tmp_path / "common.bin"
    if not common.exists():
        common.write_bytes(b"immutable")
    record = vlf.file_record(common)
    matrix_path = tmp_path / "real.npy"
    if not matrix_path.exists():
        with matrix_path.open("wb") as handle:
            np.save(handle, np.zeros((890, 512), dtype=np.float32), allow_pickle=False)
    matrix = np.load(matrix_path, allow_pickle=False)
    target_hashes = [
        {
            "index": index,
            "clip_id": _clip_id(index),
            "target_video_sha256": _hex("target", index),
        }
        for index in range(890)
    ]
    packages = {"torch": str(torch.__version__)}
    environment = {
        "requested_python": sys.executable,
        "resolved_python": vlf.file_record(sys.executable),
        "python_version": sys.version,
        "packages": packages,
        "packages_sha256": qualification._sha256_json(packages),
        "torch_cuda": "13.0",
        "cudnn_version": 1,
        "determinism": dict(qualification.DETERMINISM_CONTRACT),
        "environment_variables": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        },
    }
    environment["identity_sha256"] = qualification._digest_payload(
        environment, "identity_sha256"
    )
    payload = {
        "schema": qualification.JOB_SCHEMA,
        "status": "pass",
        "frozen": True,
        "validation_only": True,
        "source": {"commit": "a" * 40, "branch": "test", "dirty": False},
        "source_files": {name: record for name in ("qualifier", "poc", "model")},
        "environment": environment,
        "topology": _topology(job_id, node),
        "determinism_contract": dict(qualification.DETERMINISM_CONTRACT),
        "inputs": {
            **{
                name: record
                for name in (
                    "manifest",
                    "data_provenance",
                    "checkpoint",
                    "training_config",
                    "completion",
                    "r3d18_weight",
                    "phase1_gate",
                )
            },
            "data_root": str(tmp_path),
            "manifest_expected_sha256": record["sha256"],
            "checkpoint_expected_sha256": record["sha256"],
            "r3d18_expected_sha256": qualification.R3D18_SHA256,
            "r3d18_source_url": qualification.R3D18_URL,
            "nontrivial_parameter_evidence": {
                group: [
                    {
                        "name": f"{prefix}{name}",
                        "shape": [2, 2],
                        "nonzero": 4,
                        "l2_norm": 2.0,
                        "sha256": _hex(group, name),
                    }
                    for name in (
                        "clock_modulation.1.weight",
                        "video_output_head.weight",
                        "auxiliary_output_head.weight",
                    )
                ]
                for group, prefix in (("raw_model", "model."), ("ema", "ema."))
            },
        },
        "r3d18_real_target": {
            "clips": 890,
            "shape": [890, 512],
            "dtype": "float32-little-endian-c-contiguous",
            "manifest_order": True,
            "matrix_sha256": qualification.canonical_float32_matrix_sha256(matrix),
            "clip_id_sorted_matrix_sha256": (
                qualification.clip_id_sorted_float32_matrix_sha256(
                    matrix, [row["clip_id"] for row in target_hashes]
                )
            ),
            "ordered_clip_ids_sha256": qualification._sha256_json(
                [row["clip_id"] for row in target_hashes]
            ),
            "target_video_hashes": target_hashes,
            "target_video_hashes_sha256": qualification._sha256_json(target_hashes),
            "file": vlf.file_record(matrix_path),
        },
        "a1_sample_control_probe": _probe(output_hash),
    }
    payload["record_sha256"] = qualification._digest_payload(payload, "record_sha256")
    return payload


def test_canonical_float32_matrix_hash_detects_one_bit_change():
    first = np.arange(12, dtype=np.float64).reshape(3, 4)
    second = first.copy()
    second[0, 0] = np.nextafter(np.float32(0), np.float32(1), dtype=np.float32)
    assert qualification.canonical_float32_matrix_sha256(first) != qualification.canonical_float32_matrix_sha256(second)


def test_clip_id_sorted_digest_matches_production_reorder_not_manifest_order():
    matrix = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
    clip_ids = ["b" * 64, "a" * 64, "c" * 64]
    manifest_digest = qualification.canonical_float32_matrix_sha256(matrix)
    sorted_digest = qualification.clip_id_sorted_float32_matrix_sha256(
        matrix, clip_ids
    )
    assert sorted_digest == qualification.canonical_float32_matrix_sha256(
        matrix[[1, 0, 2]]
    )
    assert sorted_digest != manifest_digest


def test_topology_rejects_wrong_world_gpu_and_duplicate_uuid():
    qualification.validate_topology_record(_topology())
    wrong_world = _topology()
    wrong_world["world_size"] = 4
    with pytest.raises(qualification.QualificationError, match="eight"):
        qualification.validate_topology_record(wrong_world)
    duplicate = _topology()
    duplicate["visible_gpu_inventory"][1]["uuid"] = duplicate["visible_gpu_inventory"][0]["uuid"]
    with pytest.raises(qualification.QualificationError, match="UUID"):
        qualification.validate_topology_record(duplicate)
    floating_node = _topology()
    floating_node["nodes"] = ["node-b"]
    with pytest.raises(qualification.QualificationError, match="rank hostnames"):
        qualification.validate_topology_record(floating_node)


def test_checkpoint_rejects_zero_output_or_modulation_and_accepts_trained_like():
    model_config = {"initialization": vlf.FROZEN_INITIALIZATION}
    config = {
        "schema": vlf.SCHEMA,
        "command": "calibrate",
        "arm": "A1",
        "updates": 200,
        "checkpoint_updates": [200],
        "global_batch_size": 256,
        "world_size": 8,
        "local_optimizer_batch_size": 32,
        "dtype": "bfloat16",
        "model": model_config,
        "source": {"commit": "a" * 40, "dirty": False},
    }
    names = (
        "clock_modulation.1.weight",
        "video_output_head.weight",
        "auxiliary_output_head.weight",
    )
    state = {name: torch.ones(2, 2) for name in names}
    checkpoint = {
        "schema": vlf.CHECKPOINT_SCHEMA,
        "arm": "A1",
        "completed_updates": 200,
        "config_sha256": vlf.sha256_json(config),
        "model_config": model_config,
        "model": copy.deepcopy(state),
        "ema": {
            "decay": vlf.FROZEN_EMA_DECAY,
            "schedule": vlf.FROZEN_EMA_SCHEDULE,
            "num_updates": 200,
            "shadow": copy.deepcopy(state),
        },
    }
    evidence = qualification.validate_nontrivial_a1_checkpoint(
        checkpoint,
        config,
        expected_commit="a" * 40,
        expected_model_config=model_config,
    )
    assert len(evidence["raw_model"]) == len(evidence["ema"]) == 3
    checkpoint["ema"]["shadow"]["video_output_head.weight"].zero_()
    with pytest.raises(qualification.QualificationError, match="trivial"):
        qualification.validate_nontrivial_a1_checkpoint(
            checkpoint,
            config,
            expected_commit="a" * 40,
            expected_model_config=model_config,
        )


def test_job_record_tamper_mismatch_and_independent_node_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(vlf, "validate_phase1_gate_record", lambda *args, **kwargs: {})
    common = tmp_path / "common.bin"
    common.write_bytes(b"immutable")
    monkeypatch.setattr(qualification, "R3D18_SHA256", vlf.sha256_file(common))
    first = _job(tmp_path, job_id="101", node="node-a")
    second = _job(tmp_path, job_id="102", node="node-b")
    assert qualification._comparison_payload(first, second)["passed"] is True

    tampered = copy.deepcopy(first)
    tampered["a1_sample_control_probe"]["clips"][0][
        "generated_video_sha256_by_control"
    ]["off"] = "9" * 64
    with pytest.raises(qualification.QualificationError, match="schema"):
        qualification.validate_job_record(tampered)

    mismatched = _job(tmp_path, job_id="103", node="node-c", output_hash="9" * 64)
    with pytest.raises(qualification.QualificationError, match="differ"):
        qualification._comparison_payload(first, mismatched)

    same_job = _job(tmp_path, job_id="101", node="node-d")
    with pytest.raises(qualification.QualificationError, match="job IDs"):
        qualification._comparison_payload(first, same_job)
    same_node = _job(tmp_path, job_id="104", node="node-a")
    with pytest.raises(qualification.QualificationError, match="disjoint"):
        qualification._comparison_payload(first, same_node)

    wrong_pin = copy.deepcopy(first)
    wrong_pin["inputs"]["manifest_expected_sha256"] = "f" * 64
    wrong_pin["record_sha256"] = qualification._digest_payload(
        wrong_pin, "record_sha256"
    )
    with pytest.raises(qualification.QualificationError, match="identity changed"):
        qualification.validate_job_record(wrong_pin)

    float64_path = tmp_path / "real-float64.npy"
    with float64_path.open("wb") as handle:
        np.save(handle, np.zeros((890, 512), dtype=np.float64), allow_pickle=False)
    wrong_dtype = copy.deepcopy(first)
    wrong_dtype["r3d18_real_target"]["file"] = vlf.file_record(float64_path)
    wrong_dtype["record_sha256"] = qualification._digest_payload(
        wrong_dtype, "record_sha256"
    )
    with pytest.raises(qualification.QualificationError, match="persisted R3D"):
        qualification.validate_job_record(wrong_dtype)

    fake_environment = copy.deepcopy(first)
    fake_environment["environment"] = {"name": "not-an-environment"}
    fake_environment["environment"]["identity_sha256"] = qualification._digest_payload(
        fake_environment["environment"], "identity_sha256"
    )
    fake_environment["record_sha256"] = qualification._digest_payload(
        fake_environment, "record_sha256"
    )
    with pytest.raises(qualification.QualificationError, match="environment is malformed"):
        qualification.validate_job_record(fake_environment)


def test_job_json_roundtrip_preserves_control_semantics(tmp_path, monkeypatch):
    common = tmp_path / "common.bin"
    common.write_bytes(b"immutable")
    monkeypatch.setattr(qualification, "R3D18_SHA256", vlf.sha256_file(common))
    monkeypatch.setattr(vlf, "validate_phase1_gate_record", lambda *args, **kwargs: {})
    first = _job(tmp_path, job_id="201", node="node-a")
    second = _job(tmp_path, job_id="202", node="node-b")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    vlf.atomic_write_json(first_path, first, exclusive=True)
    vlf.atomic_write_json(second_path, second, exclusive=True)
    first_loaded = qualification._load_json(first_path, "first")
    second_loaded = qualification._load_json(second_path, "second")
    qualification.validate_job_record(first_loaded)
    qualification.validate_job_record(second_loaded)
    assert qualification._comparison_payload(first_loaded, second_loaded)["passed"]


def test_qualification_rejects_contradictory_authorization(tmp_path):
    payload = {
        "schema": qualification.QUALIFICATION_SCHEMA,
        "status": "pass",
        "frozen": True,
        "validation_only": True,
        "source_commit": "a" * 40,
        "authorization": {
            **qualification.AUTHORIZATION,
            "phase2_full_training": False,
        },
    }
    payload["qualification_sha256"] = qualification._digest_payload(
        payload, "qualification_sha256"
    )
    path = tmp_path / "qualification.json"
    vlf.atomic_write_json(path, payload, exclusive=True)
    with pytest.raises(qualification.QualificationError, match="missing, failed, or changed"):
        qualification.load_and_validate_qualification(path, expected_commit="a" * 40)


def test_deterministic_runtime_rejects_ambient_tf32_override(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    with pytest.raises(vlf.PocError, match="NVIDIA_TF32_OVERRIDE"):
        vlf.configure_deterministic_evaluation_runtime()
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    with pytest.raises(vlf.PocError, match="TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        vlf.configure_deterministic_evaluation_runtime()


def test_dual_train_and_eval_fail_closed_without_qualification():
    train = vlf.build_parser().parse_args(
        [
            "train",
            "--arm", "A1",
            "--data-root", "/mnt/data1/data",
            "--train-manifest", "/mnt/data1/train.jsonl",
            "--validation-manifest", "/mnt/data1/val.jsonl",
            "--artifact-root", "/mnt/data1/runs",
            "--run-id", "train",
            "--calibration-record", "/mnt/data1/calibration.json",
            "--phase1-gate-record", "/mnt/data1/phase1.json",
        ]
    )
    with pytest.raises(vlf.PocError, match="deterministic-evaluation qualification"):
        vlf.validate_args(train)
    train.determinism_qualification_record = "/mnt/data1/qualification.json"
    vlf.validate_args(train)

    evaluate = vlf.build_parser().parse_args(
        [
            "eval",
            "--arm", "B0",
            "--data-root", "/mnt/data1/data",
            "--manifest", "/mnt/data1/val.jsonl",
            "--checkpoint", "/mnt/data1/checkpoint.pt",
            "--artifact-root", "/mnt/data1/runs",
            "--run-id", "eval",
        ]
    )
    with pytest.raises(vlf.PocError, match="deterministic-evaluation qualification"):
        vlf.validate_args(evaluate)


def test_a1_probe_generates_auxiliary_once_and_reuses_exact_object(monkeypatch):
    class Dataset:
        def __len__(self):
            return 890

        def __getitem__(self, index):
            return {
                "clip_id": f"clip-{index}",
                "history": torch.zeros(3, 5, 64, 112),
                "future": torch.ones(3, 8, 64, 112),
                "actions": torch.zeros(16, 7),
                "lowres_scratchpad": torch.ones(48, 8, 8, 14),
            }

    calls = {"auxiliary": 0, "controls": [], "shared_ids": []}

    def fake_auxiliary_phase(
        model, history, actions, *, video_noise, auxiliary_noise, steps
    ):
        del model, history, actions, video_noise
        calls["auxiliary"] += 1
        assert steps == 25
        return torch.full_like(auxiliary_noise, 2.0), 25

    def fake_sample_control(
        model,
        arm,
        history,
        actions,
        clean_auxiliary,
        *,
        control,
        auxiliary_steps,
        video_steps,
        video_noise,
        auxiliary_noise,
        generated_auxiliary,
        shuffle_indices,
    ):
        del model, history, actions, clean_auxiliary, shuffle_indices
        assert arm == "A1" and (auxiliary_steps, video_steps) == (25, 25)
        assert generated_auxiliary is not None
        calls["controls"].append(control)
        calls["shared_ids"].append(id(generated_auxiliary))
        video = torch.ones_like(video_noise)
        return vlf.CascadeResult(
            video=video,
            generated_auxiliary=generated_auxiliary,
            conditioning_auxiliary=generated_auxiliary,
            initial_video_noise=video_noise,
            initial_auxiliary_noise=auxiliary_noise,
            model_calls=25,
            phase_boundary_sha256=vlf.tensor_sha256(generated_auxiliary),
        )

    monkeypatch.setattr(vlf, "stable_noise_like", lambda reference, *args: torch.zeros_like(reference))
    monkeypatch.setattr(vlf, "_autocast", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(vlf, "sample_auxiliary_phase", fake_auxiliary_phase)
    monkeypatch.setattr(vlf, "sample_control", fake_sample_control)
    context = SimpleNamespace(
        rank=0,
        world_size=8,
        device=torch.device("cpu"),
        gather_objects=lambda value: [value],
    )
    records = qualification._sample_a1_probe(context, Dataset(), object())
    assert len(records) == 1
    assert calls["auxiliary"] == 1
    assert calls["controls"] == list(qualification.CONTROLS)
    assert len(set(calls["shared_ids"])) == 1


def test_qualification_launcher_supports_safe_node_exclusion():
    launcher = (
        qualification.REPO_ROOT
        / "tools/slurm/submit_video_latent_forcing_determinism_qualification.sh"
    ).read_text(encoding="utf-8")
    assert "--exclude-node" in launcher
    assert 'SBATCH_ARGS+=(--exclude "$EXCLUDE_NODE")' in launcher
    assert "A-Za-z0-9_.\\[\\],-" in launcher
    generic_sbatch = (
        qualification.REPO_ROOT / "tools/slurm/video_latent_forcing_poc.sbatch"
    ).read_text(encoding="utf-8")
    qualifier_sbatch = (
        qualification.REPO_ROOT
        / "tools/slurm/video_latent_forcing_determinism_qualification.sbatch"
    ).read_text(encoding="utf-8")
    for variable in (
        "NVIDIA_TF32_OVERRIDE=0",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0",
    ):
        assert variable in generic_sbatch
        assert variable in qualifier_sbatch
    assert '${3:-}" == "B0"' in generic_sbatch
    assert '${3:-}" != "phase1"' in generic_sbatch
