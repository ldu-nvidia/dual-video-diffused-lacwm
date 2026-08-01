#!/usr/bin/env python3
"""Fail-closed deterministic-evaluation qualification for Video Latent Forcing.

``run`` is executed by torchrun on one exclusive 8xB200 node.  It extracts the
complete frozen validation population with the pinned R3D-18 extractor and
executes the real A1 EMA sampling path under CUDA bfloat16 autocast.  ``finalize``
compares records from two independently scheduled, disjoint-node jobs.  The
result is an authorization artifact, not a quality result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.datasets.droid.video_latent_forcing import (  # noqa: E402
    DroidVideoLatentForcingDataset,
    sha256_file,
)
from robot_wm.evaluation.video_latent_forcing_quality import (  # noqa: E402
    FrozenR3D18AvgPool,
    R3D18_FEATURE_DIM,
    R3D18_SHA256,
    R3D18_URL,
    r3d18_avgpool_features,
)
from tools import video_latent_forcing_poc as vlf  # noqa: E402


JOB_SCHEMA = "video-latent-forcing-determinism-job-v1"
QUALIFICATION_SCHEMA = "video-latent-forcing-determinism-qualification-v1"
QUALIFICATION_SEED = 20260801
WORLD_SIZE = 8
PER_RANK_BATCH = 8
QUALIFICATION_CLIPS = WORLD_SIZE * PER_RANK_BATCH
NFE_PAIR = (25, 25)
CONTROLS = ("autonomous", "off", "shuffled", "oracle_clean")
DETERMINISM_CONTRACT = dict(vlf.DETERMINISTIC_EVALUATION_CONTRACT)
HEX64 = re.compile(r"[0-9a-f]{64}")
AUTHORIZATION = {
    "phase2_full_training": True,
    "phase2_validation_evaluation": True,
    "quality_or_speed_claim": False,
}


class QualificationError(RuntimeError):
    """A deterministic-evaluation qualification contract was violated."""


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must be a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _digest_payload(payload: Mapping[str, Any], digest_key: str) -> str:
    return _sha256_json({key: value for key, value in payload.items() if key != digest_key})


def canonical_float32_matrix_sha256(matrix: Tensor | np.ndarray) -> str:
    """Hash canonical little-endian, C-contiguous float32 matrix bytes."""
    if isinstance(matrix, Tensor):
        array = matrix.detach().cpu().numpy()
    else:
        array = np.asarray(matrix)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise QualificationError("feature matrix must be finite and two-dimensional")
    canonical = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def clip_id_sorted_float32_matrix_sha256(
    matrix: Tensor | np.ndarray,
    clip_ids: Sequence[str],
) -> str:
    array = matrix.detach().cpu().numpy() if isinstance(matrix, Tensor) else np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != len(clip_ids) or len(set(clip_ids)) != len(clip_ids):
        raise QualificationError("feature rows require one unique clip ID each")
    order = sorted(range(len(clip_ids)), key=lambda index: clip_ids[index])
    return canonical_float32_matrix_sha256(array[order])


def configure_deterministic_runtime() -> None:
    try:
        vlf.configure_deterministic_evaluation_runtime()
    except vlf.PocError as exc:
        raise QualificationError(str(exc)) from exc


def _package_versions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    return dict(sorted(packages.items()))


def environment_record(expected_python: str | Path) -> dict[str, Any]:
    requested = Path(expected_python).expanduser()
    if not requested.is_absolute() or not requested.exists():
        raise QualificationError("expected Python must be an existing absolute path")
    try:
        same_python = os.path.samefile(requested, sys.executable)
    except OSError as exc:
        raise QualificationError("cannot compare the requested and running Python") from exc
    if not same_python:
        raise QualificationError("running Python differs from the launcher-pinned environment")
    packages = _package_versions()
    stable = {
        "requested_python": str(requested),
        "resolved_python": vlf.file_record(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "packages": packages,
        "packages_sha256": _sha256_json(packages),
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "determinism": dict(DETERMINISM_CONTRACT),
        "environment_variables": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get(
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
            ),
        },
    }
    stable["identity_sha256"] = _sha256_json(stable)
    return stable


def validate_environment_record(environment: Mapping[str, Any]) -> None:
    required_keys = {
        "requested_python",
        "resolved_python",
        "python_version",
        "packages",
        "packages_sha256",
        "torch_cuda",
        "cudnn_version",
        "determinism",
        "environment_variables",
        "identity_sha256",
    }
    packages = environment.get("packages")
    variables = environment.get("environment_variables")
    requested = environment.get("requested_python")
    if (
        set(environment) != required_keys
        or not isinstance(requested, str)
        or not Path(requested).is_absolute()
        or not isinstance(environment.get("python_version"), str)
        or not isinstance(packages, Mapping)
        or not packages
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in packages.items())
        or environment.get("packages_sha256") != _sha256_json(dict(packages))
        or not isinstance(environment.get("torch_cuda"), str)
        or not environment.get("torch_cuda")
        or isinstance(environment.get("cudnn_version"), bool)
        or not isinstance(environment.get("cudnn_version"), int)
        or environment.get("cudnn_version") <= 0
        or environment.get("determinism") != DETERMINISM_CONTRACT
        or variables
        != {
            "CUBLAS_WORKSPACE_CONFIG": DETERMINISM_CONTRACT[
                "cublas_workspace_config"
            ],
            "NVIDIA_TF32_OVERRIDE": DETERMINISM_CONTRACT[
                "nvidia_tf32_override"
            ],
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": DETERMINISM_CONTRACT[
                "torch_allow_tf32_cublas_override"
            ],
        }
        or environment.get("identity_sha256")
        != _digest_payload(environment, "identity_sha256")
    ):
        raise QualificationError("qualified Python/package/CUDA environment is malformed")
    resolved = _verify_file_record(environment.get("resolved_python"), "resolved Python")
    try:
        if not os.path.samefile(requested, resolved["path"]):
            raise QualificationError("requested and resolved qualified Python differ")
    except OSError as exc:
        raise QualificationError("qualified Python path is unavailable") from exc


def _visible_gpu_inventory() -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise QualificationError("nvidia-smi returned a malformed GPU inventory")
        inventory.append(
            {"uuid": fields[0], "name": fields[1], "driver": fields[2], "memory_mib": fields[3]}
        )
    return inventory


def validate_topology_record(record: Mapping[str, Any]) -> None:
    ranks = record.get("ranks")
    nodes = record.get("nodes")
    gpus = record.get("visible_gpu_inventory")
    if (
        set(record)
        != {
            "slurm_job_id",
            "slurm_nodelist",
            "nodes_requested",
            "world_size",
            "per_rank_batch_size",
            "global_probe_clips",
            "nodes",
            "ranks",
            "visible_gpu_inventory",
        }
        or not re.fullmatch(r"[0-9]+", str(record.get("slurm_job_id", "")))
        or not isinstance(record.get("slurm_nodelist"), str)
        or record.get("world_size") != WORLD_SIZE
        or record.get("nodes_requested") != 1
        or record.get("per_rank_batch_size") != PER_RANK_BATCH
        or record.get("global_probe_clips") != QUALIFICATION_CLIPS
        or not isinstance(ranks, list)
        or len(ranks) != WORLD_SIZE
        or not isinstance(nodes, list)
        or len(nodes) != 1
        or not isinstance(gpus, list)
        or len(gpus) != WORLD_SIZE
    ):
        raise QualificationError("qualification requires one Slurm node with exactly eight ranks/GPUs")
    rank_keys = {
        "rank",
        "local_rank",
        "hostname",
        "device_name",
        "compute_capability",
        "cuda_visible_devices",
    }
    if any(
        not isinstance(row, Mapping)
        or set(row) != rank_keys
        or not isinstance(row.get("hostname"), str)
        or not row.get("hostname")
        or not isinstance(row.get("compute_capability"), list)
        or len(row["compute_capability"]) != 2
        or any(not isinstance(value, int) for value in row["compute_capability"])
        or not isinstance(row.get("cuda_visible_devices"), str)
        for row in ranks
    ):
        raise QualificationError("rank/GPU binding schema is malformed")
    if {row.get("rank") for row in ranks if isinstance(row, Mapping)} != set(range(WORLD_SIZE)):
        raise QualificationError("distributed rank inventory is incomplete")
    if {row.get("local_rank") for row in ranks if isinstance(row, Mapping)} != set(
        range(WORLD_SIZE)
    ):
        raise QualificationError("local-rank topology is not one process per GPU")
    if any("B200" not in str(row.get("device_name", "")) for row in ranks):
        raise QualificationError("every qualification rank must execute on B200")
    if {row["hostname"] for row in ranks} != set(nodes):
        raise QualificationError("declared node set differs from rank hostnames")
    uuids = [row.get("uuid") for row in gpus if isinstance(row, Mapping)]
    if (
        any(
            not isinstance(row, Mapping)
            or set(row) != {"uuid", "name", "driver", "memory_mib"}
            or "B200" not in str(row.get("name", ""))
            or not str(row.get("driver", ""))
            or not str(row.get("memory_mib", ""))
            for row in gpus
        )
        or len(set(uuids)) != WORLD_SIZE
        or any(not str(uuid).startswith("GPU-") for uuid in uuids)
    ):
        raise QualificationError("allocated GPU UUID inventory is incomplete or duplicated")


def topology_record(context: vlf.DistributedContext) -> dict[str, Any]:
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "hostname": socket.gethostname(),
        "device_name": torch.cuda.get_device_name(context.device),
        "compute_capability": list(torch.cuda.get_device_capability(context.device)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    ranks = context.gather_objects(local)
    inventory = _visible_gpu_inventory()
    record = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "nodes_requested": int(os.environ.get("SLURM_NNODES", "0")),
        "world_size": context.world_size,
        "per_rank_batch_size": PER_RANK_BATCH,
        "global_probe_clips": QUALIFICATION_CLIPS,
        "nodes": sorted({str(row["hostname"]) for row in ranks}),
        "ranks": sorted(ranks, key=lambda row: int(row["rank"])),
        "visible_gpu_inventory": inventory,
    }
    validate_topology_record(record)
    return record


def _tensor_nontrivial_stats(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise QualificationError(f"checkpoint tensor is missing: {name}")
    tensor = value.detach().float()
    if not bool(torch.isfinite(tensor).all()):
        raise QualificationError(f"checkpoint tensor is nonfinite: {name}")
    nonzero = int(torch.count_nonzero(tensor))
    norm = float(torch.linalg.vector_norm(tensor))
    if nonzero == 0 or not math.isfinite(norm) or norm <= 0:
        raise QualificationError(f"checkpoint remains trivial/untrained at {name}")
    return {
        "name": name,
        "shape": list(tensor.shape),
        "nonzero": nonzero,
        "l2_norm": norm,
        "sha256": vlf.tensor_sha256(tensor),
    }


def validate_nontrivial_a1_checkpoint(
    checkpoint: Mapping[str, Any],
    training_config: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_model_config: Mapping[str, Any],
) -> dict[str, Any]:
    source = training_config.get("source")
    ema = checkpoint.get("ema")
    if (
        checkpoint.get("schema") != vlf.CHECKPOINT_SCHEMA
        or checkpoint.get("arm") != "A1"
        or checkpoint.get("completed_updates") != vlf.CALIBRATION_UPDATES
        or checkpoint.get("config_sha256") != vlf.sha256_json(training_config)
        or checkpoint.get("model_config") != expected_model_config
        or training_config.get("schema") != vlf.SCHEMA
        or training_config.get("command") != "calibrate"
        or training_config.get("arm") != "A1"
        or training_config.get("updates") != vlf.CALIBRATION_UPDATES
        or training_config.get("checkpoint_updates") != [vlf.CALIBRATION_UPDATES]
        or training_config.get("global_batch_size") != vlf.FROZEN_GLOBAL_BATCH_SIZE
        or training_config.get("world_size") != WORLD_SIZE
        or training_config.get("local_optimizer_batch_size") != 32
        or training_config.get("dtype") != "bfloat16"
        or training_config.get("model") != expected_model_config
        or not isinstance(source, Mapping)
        or source.get("commit") != expected_commit
        or source.get("dirty") is not False
        or not isinstance(ema, Mapping)
        or ema.get("decay") != vlf.FROZEN_EMA_DECAY
        or ema.get("schedule") != vlf.FROZEN_EMA_SCHEDULE
        or ema.get("num_updates") != vlf.CALIBRATION_UPDATES
        or not isinstance(ema.get("shadow"), Mapping)
        or not isinstance(checkpoint.get("model"), Mapping)
    ):
        raise QualificationError("A1 calibration checkpoint/config/EMA provenance is invalid")
    names = (
        "clock_modulation.1.weight",
        "video_output_head.weight",
        "auxiliary_output_head.weight",
    )
    return {
        "raw_model": [
            _tensor_nontrivial_stats(checkpoint["model"].get(name), name=f"model.{name}")
            for name in names
        ],
        "ema": [
            _tensor_nontrivial_stats(ema["shadow"].get(name), name=f"ema.{name}")
            for name in names
        ],
    }


def _load_a1_calibration(
    checkpoint_path: Path,
    *,
    expected_commit: str,
    expected_model_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if checkpoint_path.parent.name != "checkpoints" or checkpoint_path.name != "update_000200.pt":
        raise QualificationError("qualification requires the canonical update-200 checkpoint path")
    run_dir = checkpoint_path.parent.parent
    config_path = run_dir / "resolved_config.json"
    complete_path = run_dir / "complete.json"
    config = vlf.load_json(config_path, "A1 calibration resolved config")
    # Reuse the training-side immutable completion/checkpoint audit before the
    # qualification-specific nontriviality checks below.
    vlf.validate_calibration_record(
        complete_path,
        "A1",
        expected_experiment_identity_sha256=str(config.get("experiment_identity_sha256", "")),
        expected_commit=expected_commit,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    nontrivial = validate_nontrivial_a1_checkpoint(
        checkpoint,
        config,
        expected_commit=expected_commit,
        expected_model_config=expected_model_config,
    )
    records = {
        "checkpoint": vlf.file_record(checkpoint_path),
        "training_config": vlf.file_record(config_path),
        "completion": vlf.file_record(complete_path),
        "nontrivial_parameter_evidence": nontrivial,
    }
    return checkpoint, config, records


@torch.inference_mode()
def _extract_real_features(
    context: vlf.DistributedContext,
    dataset: DroidVideoLatentForcingDataset,
    rows: Sequence[Mapping[str, Any]],
    extractor: FrozenR3D18AvgPool,
) -> list[dict[str, Any]]:
    local: list[dict[str, Any]] = []
    local_indexes, local_batches = vlf.paired_rank_evaluation_layout(
        len(dataset),
        PER_RANK_BATCH,
        rank=context.rank,
        world_size=context.world_size,
    )
    for batch_positions in local_batches:
        indexes = [local_indexes[position] for position in batch_positions]
        raw = default_collate([dataset[index] for index in indexes])
        batch = vlf.move_training_batch(raw, context.device)
        quality_reference, _ = vlf.canonical_quality_video(
            batch["future"], upsample_lowres=False
        )
        features = r3d18_avgpool_features(
            quality_reference, extractor
        ).detach().float().cpu()
        for offset, index in enumerate(indexes):
            clip_id = str(raw["clip_id"][offset])
            if clip_id != str(rows[index]["clip_id"]):
                raise QualificationError("dataset order differs from the pinned manifest")
            local.append(
                {
                    "index": index,
                    "clip_id": clip_id,
                    "target_video_sha256": vlf.tensor_sha256(raw["future"][offset]),
                    "feature": features[offset].numpy(),
                }
            )
    gathered = context.gather_objects(local)
    return [item for rank_rows in gathered for item in rank_rows]


@torch.inference_mode()
def _sample_a1_probe(
    context: vlf.DistributedContext,
    dataset: DroidVideoLatentForcingDataset,
    model: torch.nn.Module,
) -> list[dict[str, Any]]:
    local_indexes, local_batches = vlf.paired_rank_evaluation_layout(
        len(dataset),
        PER_RANK_BATCH,
        rank=context.rank,
        world_size=context.world_size,
    )
    indexes = [local_indexes[position] for position in local_batches[0]]
    if len(indexes) != PER_RANK_BATCH:
        raise QualificationError("every rank must probe one complete eight-clip eval batch")
    raw = default_collate([dataset[index] for index in indexes])
    batch = vlf.move_training_batch(raw, context.device)
    clip_ids = [str(value) for value in raw["clip_id"]]
    video_noise = vlf.stable_noise_like(batch["future"], clip_ids, QUALIFICATION_SEED, "video")
    auxiliary_noise = vlf.stable_noise_like(
        batch["lowres_scratchpad"], clip_ids, QUALIFICATION_SEED, "aux"
    )
    shuffle_indices = torch.arange(
        len(clip_ids), device=context.device, dtype=torch.long
    ).bitwise_xor(1)
    shuffled_sources = [clip_ids[int(index)] for index in shuffle_indices.cpu().tolist()]
    with vlf._autocast(context.device):
        generated_shared, auxiliary_calls = vlf.sample_auxiliary_phase(
            model,
            batch["history"],
            batch["actions"],
            video_noise=video_noise,
            auxiliary_noise=auxiliary_noise,
            steps=NFE_PAIR[0],
        )
    if auxiliary_calls != NFE_PAIR[0]:
        raise QualificationError("A1 probe auxiliary phase did not execute 25 calls")
    control_results: dict[str, vlf.CascadeResult] = {}
    for control in CONTROLS:
        oracle = batch["lowres_scratchpad"] if control == "oracle_clean" else None
        with vlf._autocast(context.device):
            result = vlf.sample_control(
                model,
                "A1",
                batch["history"],
                batch["actions"],
                oracle,
                control=control,
                auxiliary_steps=NFE_PAIR[0],
                video_steps=NFE_PAIR[1],
                video_noise=video_noise,
                auxiliary_noise=auxiliary_noise,
                generated_auxiliary=generated_shared,
                shuffle_indices=shuffle_indices,
            )
        if auxiliary_calls + result.model_calls != sum(NFE_PAIR):
            raise QualificationError("A1 probe did not execute the conceptual exact 25+25 calls")
        control_results[control] = result
    boundaries = {result.phase_boundary_sha256 for result in control_results.values()}
    if len(boundaries) != 1:
        raise QualificationError("A1 controls did not regenerate one bit-identical phase boundary")
    batch_hashes = {control: vlf.tensor_sha256(result.video) for control, result in control_results.items()}
    if len(set(batch_hashes.values())) != 1:
        raise QualificationError("A1 fusion-off raw generated videos differ across controls")
    rows_out: list[dict[str, Any]] = []
    for offset, clip_id in enumerate(clip_ids):
        output_hashes = {
            control: vlf.tensor_sha256(result.video[offset])
            for control, result in control_results.items()
        }
        if len(set(output_hashes.values())) != 1:
            raise QualificationError("A1 fusion-off per-clip generated video differs across controls")
        if output_hashes["autonomous"] == vlf.tensor_sha256(video_noise[offset]):
            raise QualificationError("A1 generated video is a trivial unchanged-noise output")
        rows_out.append(
            {
                "manifest_index": indexes[offset],
                "clip_id": clip_id,
                "history_sha256": vlf.tensor_sha256(batch["history"][offset]),
                "actions_sha256": vlf.tensor_sha256(batch["actions"][offset]),
                "target_video_sha256": vlf.tensor_sha256(batch["future"][offset]),
                "target_auxiliary_sha256": vlf.tensor_sha256(
                    batch["lowres_scratchpad"][offset]
                ),
                "initial_video_noise_sha256": vlf.tensor_sha256(video_noise[offset]),
                "initial_auxiliary_noise_sha256": vlf.tensor_sha256(
                    auxiliary_noise[offset]
                ),
                "shuffle_source_clip_id": shuffled_sources[offset],
                "phase_boundary_sha256": vlf.tensor_sha256(
                    generated_shared[offset]
                ),
                "generated_video_sha256_by_control": output_hashes,
            }
        )
    return context.gather_objects(
        {
            "rank": context.rank,
            "batch_generated_video_sha256_by_control": batch_hashes,
            "batch_phase_boundary_sha256": vlf.tensor_sha256(generated_shared),
            "clips": rows_out,
        }
    )


def _validate_sample_probe(probe: Mapping[str, Any]) -> None:
    if set(probe) != {
        "arm",
        "weights",
        "ema_updates",
        "nfe_pair",
        "controls",
        "autocast",
        "qualified_clips",
        "ranks",
        "clips",
    }:
        raise QualificationError("A1 sampling probe keys differ from the frozen schema")
    ranks = probe.get("ranks")
    clips = probe.get("clips")
    if (
        probe.get("arm") != "A1"
        or probe.get("weights") != "ema"
        or probe.get("ema_updates") != vlf.CALIBRATION_UPDATES
        or probe.get("nfe_pair") != list(NFE_PAIR)
        or probe.get("controls") != list(CONTROLS)
        or probe.get("autocast") != "cuda-bfloat16"
        or not isinstance(ranks, list)
        or len(ranks) != WORLD_SIZE
        or any(not isinstance(row, Mapping) for row in ranks)
        or not isinstance(clips, list)
        or len(clips) != QUALIFICATION_CLIPS
        or any(not isinstance(row, Mapping) for row in clips)
        or probe.get("qualified_clips") != QUALIFICATION_CLIPS
    ):
        raise QualificationError("A1 sampling probe inventory differs from the frozen contract")
    expected_indexes: list[int] = []
    for rank in range(WORLD_SIZE):
        local_indexes, local_batches = vlf.paired_rank_evaluation_layout(
            vlf.FROZEN_VALIDATION_CLIPS,
            PER_RANK_BATCH,
            rank=rank,
            world_size=WORLD_SIZE,
        )
        expected_indexes.extend(local_indexes[position] for position in local_batches[0])
    if [row.get("manifest_index") for row in clips] != sorted(expected_indexes):
        raise QualificationError("A1 probe does not cover each rank's first exact eval batch")
    clip_keys = {
        "manifest_index",
        "clip_id",
        "history_sha256",
        "actions_sha256",
        "target_video_sha256",
        "target_auxiliary_sha256",
        "initial_video_noise_sha256",
        "initial_auxiliary_noise_sha256",
        "shuffle_source_clip_id",
        "phase_boundary_sha256",
        "generated_video_sha256_by_control",
    }
    for row in clips:
        required_hashes = (
            "history_sha256",
            "actions_sha256",
            "target_video_sha256",
            "target_auxiliary_sha256",
            "initial_video_noise_sha256",
            "initial_auxiliary_noise_sha256",
            "phase_boundary_sha256",
        )
        if (
            not isinstance(row, Mapping)
            or set(row) != clip_keys
            or not HEX64.fullmatch(str(row.get("clip_id", "")))
            or not HEX64.fullmatch(str(row.get("shuffle_source_clip_id", "")))
            or any(not HEX64.fullmatch(str(row.get(name, ""))) for name in required_hashes)
        ):
            raise QualificationError("A1 probe input/output hash inventory is malformed")
        hashes = row.get("generated_video_sha256_by_control")
        if not isinstance(hashes, Mapping) or set(hashes) != set(CONTROLS):
            raise QualificationError("A1 probe control output inventory is malformed")
        if any(not HEX64.fullmatch(str(value)) for value in hashes.values()) or len(set(hashes.values())) != 1:
            raise QualificationError("A1 within-job raw outputs are not bit-identical")
        if hashes["autonomous"] == row["initial_video_noise_sha256"]:
            raise QualificationError("A1 probe generated video is unchanged input noise")
    if [row.get("rank") for row in ranks if isinstance(row, Mapping)] != list(
        range(WORLD_SIZE)
    ):
        raise QualificationError("A1 probe rank inventory is incomplete or reordered")
    for row in ranks:
        if set(row) != {
            "rank",
            "batch_generated_video_sha256_by_control",
            "batch_phase_boundary_sha256",
            "clips",
        } or not HEX64.fullmatch(str(row.get("batch_phase_boundary_sha256", ""))):
            raise QualificationError("A1 rank-local probe schema is malformed")
        hashes = row.get("batch_generated_video_sha256_by_control")
        if (
            not isinstance(hashes, Mapping)
            or set(hashes) != set(CONTROLS)
            or any(not HEX64.fullmatch(str(value)) for value in hashes.values())
            or len(set(hashes.values())) != 1
        ):
            raise QualificationError("A1 within-job batch outputs are not bit-identical")
        rank_clips_in_order = row.get("clips")
        if not isinstance(rank_clips_in_order, list) or len(rank_clips_in_order) != PER_RANK_BATCH:
            raise QualificationError("A1 rank-local probe batch is not exactly eight clips")
        for offset, clip in enumerate(rank_clips_in_order):
            if clip.get("shuffle_source_clip_id") != rank_clips_in_order[offset ^ 1].get(
                "clip_id"
            ):
                raise QualificationError("A1 probe donor mapping is not adjacent-pair XOR")
    rank_clips = sorted(
        [clip for row in ranks for clip in row.get("clips", [])],
        key=lambda row: int(row["manifest_index"]),
    )
    if rank_clips != clips:
        raise QualificationError("rank-local A1 probe evidence differs from the global inventory")
    if (
        len({row["clip_id"] for row in clips}) != QUALIFICATION_CLIPS
        or len(
            {
                row["generated_video_sha256_by_control"]["autonomous"]
                for row in clips
            }
        )
        < 2
        or len({row["phase_boundary_sha256"] for row in clips}) < 2
        or len({row["batch_phase_boundary_sha256"] for row in ranks}) < 2
    ):
        raise QualificationError("A1 probe is trivial or lacks cross-clip diversity")


def _verify_file_record(record: Any, label: str) -> dict[str, Any]:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise QualificationError(f"{label} file record is malformed")
    path = Path(record["path"]).expanduser().resolve()
    if (
        not path.is_file()
        or not HEX64.fullmatch(str(record.get("sha256", "")))
        or sha256_file(path) != record["sha256"]
        or path.stat().st_size != record.get("bytes")
    ):
        raise QualificationError(f"{label} file is missing or changed")
    return dict(record)


def _validate_nontrivial_evidence(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"raw_model", "ema"}:
        raise QualificationError("A1 nontrivial-parameter evidence is malformed")
    base_names = (
        "clock_modulation.1.weight",
        "video_output_head.weight",
        "auxiliary_output_head.weight",
    )
    for group, prefix in (("raw_model", "model."), ("ema", "ema.")):
        entries = value.get(group)
        if not isinstance(entries, list) or len(entries) != len(base_names):
            raise QualificationError("A1 nontrivial-parameter evidence is incomplete")
        if [entry.get("name") for entry in entries if isinstance(entry, Mapping)] != [
            prefix + name for name in base_names
        ]:
            raise QualificationError("A1 nontrivial-parameter names differ")
        for entry in entries:
            shape = entry.get("shape")
            nonzero = entry.get("nonzero")
            norm = entry.get("l2_norm")
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"name", "shape", "nonzero", "l2_norm", "sha256"}
                or not isinstance(shape, list)
                or not shape
                or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape)
                or isinstance(nonzero, bool)
                or not isinstance(nonzero, int)
                or not 0 < nonzero <= math.prod(shape)
                or isinstance(norm, bool)
                or not isinstance(norm, (int, float))
                or not math.isfinite(norm)
                or norm <= 0
                or not HEX64.fullmatch(str(entry.get("sha256", "")))
            ):
                raise QualificationError("A1 nontrivial-parameter statistics are invalid")


def validate_job_record(payload: Mapping[str, Any], *, expected_commit: str | None = None) -> None:
    source = payload.get("source")
    features = payload.get("r3d18_real_target")
    probe = payload.get("a1_sample_control_probe")
    topology = payload.get("topology")
    environment = payload.get("environment")
    inputs = payload.get("inputs")
    if set(payload) != {
        "schema",
        "status",
        "frozen",
        "validation_only",
        "source",
        "source_files",
        "environment",
        "topology",
        "determinism_contract",
        "inputs",
        "r3d18_real_target",
        "a1_sample_control_probe",
        "record_sha256",
    }:
        raise QualificationError("determinism job keys differ from the frozen schema")
    if (
        payload.get("schema") != JOB_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("frozen") is not True
        or payload.get("validation_only") is not True
        or not isinstance(source, Mapping)
        or set(source) != {"commit", "branch", "dirty"}
        or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit", "")))
        or not isinstance(source.get("branch"), str)
        or source.get("dirty") is not False
        or (expected_commit is not None and source.get("commit") != expected_commit)
        or payload.get("record_sha256") != _digest_payload(payload, "record_sha256")
        or payload.get("determinism_contract") != DETERMINISM_CONTRACT
        or not isinstance(features, Mapping)
        or features.get("shape") != [vlf.FROZEN_VALIDATION_CLIPS, R3D18_FEATURE_DIM]
        or features.get("clips") != vlf.FROZEN_VALIDATION_CLIPS
        or features.get("manifest_order") is not True
        or features.get("dtype") != "float32-little-endian-c-contiguous"
        or not HEX64.fullmatch(str(features.get("matrix_sha256", "")))
        or not isinstance(probe, Mapping)
        or not isinstance(topology, Mapping)
        or not isinstance(environment, Mapping)
        or not isinstance(inputs, Mapping)
    ):
        raise QualificationError("determinism job record violates its frozen schema")
    validate_environment_record(environment)
    expected_input_keys = {
        "manifest",
        "manifest_expected_sha256",
        "data_provenance",
        "data_root",
        "checkpoint",
        "training_config",
        "completion",
        "nontrivial_parameter_evidence",
        "checkpoint_expected_sha256",
        "r3d18_weight",
        "r3d18_expected_sha256",
        "r3d18_source_url",
        "phase1_gate",
    }
    if set(inputs) != expected_input_keys:
        raise QualificationError("qualification input keys differ from the frozen schema")
    _validate_nontrivial_evidence(inputs.get("nontrivial_parameter_evidence"))
    validate_topology_record(topology)
    _validate_sample_probe(probe)
    if set(features) != {
        "clips",
        "shape",
        "dtype",
        "manifest_order",
        "matrix_sha256",
        "clip_id_sorted_matrix_sha256",
        "ordered_clip_ids_sha256",
        "target_video_hashes",
        "target_video_hashes_sha256",
        "file",
    }:
        raise QualificationError("R3D target evidence keys differ from the frozen schema")
    matrix_record = _verify_file_record(features.get("file"), "R3D matrix")
    matrix = np.load(matrix_record["path"], allow_pickle=False)
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.dtype.str != "<f4"
        or not matrix.flags.c_contiguous
        or matrix.dtype.fields is not None
        or list(matrix.shape) != features["shape"]
        or canonical_float32_matrix_sha256(matrix) != features["matrix_sha256"]
    ):
        raise QualificationError("persisted R3D matrix differs from its canonical digest")
    target_hashes = features.get("target_video_hashes")
    if (
        not isinstance(target_hashes, list)
        or len(target_hashes) != vlf.FROZEN_VALIDATION_CLIPS
        or [row.get("index") for row in target_hashes]
        != list(range(vlf.FROZEN_VALIDATION_CLIPS))
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"index", "clip_id", "target_video_sha256"}
            or not HEX64.fullmatch(str(row.get("clip_id", "")))
            or not HEX64.fullmatch(str(row.get("target_video_sha256", "")))
            for row in target_hashes
        )
        or features.get("target_video_hashes_sha256") != _sha256_json(target_hashes)
        or features.get("ordered_clip_ids_sha256")
        != _sha256_json([row["clip_id"] for row in target_hashes])
    ):
        raise QualificationError("full-validation target input hashes are incomplete or changed")
    if features.get("clip_id_sorted_matrix_sha256") != clip_id_sorted_float32_matrix_sha256(
        matrix, [row["clip_id"] for row in target_hashes]
    ):
        raise QualificationError("clip-ID-sorted R3D target digest is missing or changed")
    targets_by_index = {int(row["index"]): row for row in target_hashes}
    for probe_row in probe["clips"]:
        target_row = targets_by_index.get(int(probe_row["manifest_index"]))
        if (
            target_row is None
            or target_row["clip_id"] != probe_row["clip_id"]
            or target_row["target_video_sha256"]
            != probe_row["target_video_sha256"]
        ):
            raise QualificationError(
                "A1 probe clip/target identity differs from full-validation extraction"
            )
    verified_inputs = {
        label: _verify_file_record(inputs.get(label), label)
        for label in (
            "manifest",
            "data_provenance",
            "checkpoint",
            "training_config",
            "completion",
            "r3d18_weight",
            "phase1_gate",
        )
    }
    if (
        verified_inputs["manifest"]["sha256"] != inputs.get("manifest_expected_sha256")
        or verified_inputs["checkpoint"]["sha256"]
        != inputs.get("checkpoint_expected_sha256")
        or verified_inputs["r3d18_weight"]["sha256"] != R3D18_SHA256
        or inputs.get("r3d18_expected_sha256") != R3D18_SHA256
        or inputs.get("r3d18_source_url") != R3D18_URL
        or not isinstance(inputs.get("data_root"), str)
        or not Path(inputs["data_root"]).is_absolute()
        or not Path(inputs["data_root"]).is_dir()
    ):
        raise QualificationError("launcher-pinned manifest/checkpoint/R3D identity changed")
    try:
        vlf.validate_phase1_gate_record(
            verified_inputs["phase1_gate"]["path"],
            expected_commit=str(source.get("commit")),
        )
    except vlf.PocError as exc:
        raise QualificationError(f"passed same-source Phase-1 gate is invalid: {exc}") from exc
    source_files = payload.get("source_files")
    if not isinstance(source_files, Mapping) or set(source_files) != {
        "qualifier",
        "poc",
        "model",
    }:
        raise QualificationError("qualification source-file inventory is missing")
    for label in ("qualifier", "poc", "model"):
        _verify_file_record(source_files.get(label), f"source {label}")


def _comparison_payload(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    already_validated: bool = False,
) -> dict[str, Any]:
    if not already_validated:
        validate_job_record(first)
        validate_job_record(second)
    first_topology = first["topology"]
    second_topology = second["topology"]
    if str(first_topology["slurm_job_id"]) == str(second_topology["slurm_job_id"]):
        raise QualificationError("qualification jobs must have distinct Slurm job IDs")
    if set(first_topology["nodes"]) & set(second_topology["nodes"]):
        raise QualificationError("qualification jobs must execute on disjoint node sets")
    exact_fields = (
        "source",
        "environment",
        "determinism_contract",
        "inputs",
        "r3d18_real_target",
        "a1_sample_control_probe",
    )
    # Artifact paths and GPU IDs are intentionally job-specific. Compare the
    # canonical scientific content instead of the matrix file record/path.
    first_features = dict(first["r3d18_real_target"])
    second_features = dict(second["r3d18_real_target"])
    first_features.pop("file", None)
    second_features.pop("file", None)
    first_probe = first["a1_sample_control_probe"]
    second_probe = second["a1_sample_control_probe"]
    checks = {
        "source": first["source"] == second["source"],
        "environment": first["environment"] == second["environment"],
        "determinism_contract": first["determinism_contract"] == second["determinism_contract"],
        "inputs": first["inputs"] == second["inputs"],
        "r3d18_real_target": first_features == second_features,
        "a1_sample_control_probe": first_probe == second_probe,
    }
    if not all(checks.values()):
        mismatch = [name for name in exact_fields if not checks[name]]
        raise QualificationError("independent qualification jobs differ: " + ", ".join(mismatch))
    return {
        "passed": True,
        "distinct_slurm_job_ids": True,
        "disjoint_node_sets": True,
        "exact_match_checks": checks,
        "r3d18_real_matrix_sha256": first_features[
            "clip_id_sorted_matrix_sha256"
        ],
        "r3d18_real_manifest_order_matrix_sha256": first_features[
            "matrix_sha256"
        ],
        "r3d18_real_clip_id_sorted_matrix_sha256": first_features[
            "clip_id_sorted_matrix_sha256"
        ],
        "sample_probe_sha256": _sha256_json(first_probe),
    }


def load_and_validate_qualification(
    path: str | Path, *, expected_commit: str
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = _load_json(resolved, "determinism qualification")
    if (
        payload.get("schema") != QUALIFICATION_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("frozen") is not True
        or payload.get("validation_only") is not True
        or payload.get("source_commit") != expected_commit
        or payload.get("authorization") != AUTHORIZATION
        or payload.get("qualification_sha256")
        != _digest_payload(payload, "qualification_sha256")
    ):
        raise QualificationError("determinism qualification is missing, failed, or changed")
    records = payload.get("job_records")
    if not isinstance(records, list) or len(records) != 2:
        raise QualificationError("determinism qualification must bind exactly two jobs")
    jobs = []
    for index, record in enumerate(records):
        verified = _verify_file_record(record, f"qualification job {index}")
        job = _load_json(verified["path"], f"qualification job {index}")
        validate_job_record(job, expected_commit=expected_commit)
        jobs.append(job)
    comparison = _comparison_payload(jobs[0], jobs[1], already_validated=True)
    if comparison != payload.get("comparison"):
        raise QualificationError("qualification comparison cannot be reproduced")
    phase1_records = [job["inputs"]["phase1_gate"] for job in jobs]
    if phase1_records[0] != phase1_records[1] or payload.get("phase1_gate") != phase1_records[0]:
        raise QualificationError("qualification changed the passed Phase-1 gate binding")
    if (
        jobs[0]["environment"] != jobs[1]["environment"]
        or payload.get("qualified_environment") != jobs[0]["environment"]
    ):
        raise QualificationError("qualification changed the exact evaluator environment")
    return payload


def load_qualified_environment_preflight(
    path: str | Path, *, expected_commit: str
) -> dict[str, Any]:
    """Read the stable qualified env before any evaluator CUDA initialization."""
    payload = _load_json(path, "determinism qualification preflight")
    environment = payload.get("qualified_environment")
    if (
        payload.get("schema") != QUALIFICATION_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("source_commit") != expected_commit
        or payload.get("qualification_sha256")
        != _digest_payload(payload, "qualification_sha256")
        or not isinstance(environment, Mapping)
        or environment.get("identity_sha256")
        != _digest_payload(environment, "identity_sha256")
    ):
        raise QualificationError("qualified environment preflight is invalid")
    return dict(environment)


def run_command(args: argparse.Namespace) -> int:
    configure_deterministic_runtime()
    context = vlf.initialize_distributed()
    try:
        if context.world_size != WORLD_SIZE or not torch.cuda.is_available():
            raise QualificationError("qualification must run under torchrun with eight CUDA ranks")
        source = vlf.git_record()
        if source.get("dirty") is not False or source.get("commit") != args.expected_source_commit:
            raise QualificationError("qualification source is dirty or differs from the expected commit")
        environment = environment_record(args.expected_python)
        topology = topology_record(context)
        run_dir = vlf.validated_run_dir(args.artifact_root, args.run_id, resume=False)
        if context.is_primary:
            run_dir.mkdir(parents=True, exist_ok=False)
        context.barrier()

        manifest_path, rows, manifest_record = vlf._manifest_record(
            args.manifest, "val", data_root=args.data_root
        )
        if sha256_file(manifest_path) != args.manifest_sha256:
            raise QualificationError("validation manifest differs from its launcher-pinned hash")
        phase1_gate = vlf.validate_phase1_gate_record(
            args.phase1_gate_record, expected_commit=args.expected_source_commit
        )
        del phase1_gate
        model_args = argparse.Namespace(arm="A1", width=512, depth=12, heads=8, mlp_ratio=4.0)
        model, model_config = vlf.instantiate_model(model_args)
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if sha256_file(checkpoint_path) != args.checkpoint_sha256:
            raise QualificationError("A1 calibration checkpoint differs from its pinned hash")
        checkpoint, training_config, checkpoint_records = _load_a1_calibration(
            checkpoint_path,
            expected_commit=args.expected_source_commit,
            expected_model_config=model_config,
        )
        del training_config
        model.to(context.device)
        model.load_state_dict(checkpoint["ema"]["shadow"], strict=True)
        model.eval()
        del checkpoint

        if args.r3d18_sha256 != R3D18_SHA256 or sha256_file(args.r3d18_weight) != R3D18_SHA256:
            raise QualificationError("R3D-18 weight is not the pinned Kinetics-400 V1 artifact")
        extractor = FrozenR3D18AvgPool(
            weight_path=args.r3d18_weight,
            expected_sha256=args.r3d18_sha256,
            device=context.device,
        )
        dataset = DroidVideoLatentForcingDataset(manifest_path, args.data_root)
        feature_rows = _extract_real_features(context, dataset, rows, extractor)
        sample_ranks = _sample_a1_probe(context, dataset, model)

        if context.is_primary:
            feature_rows.sort(key=lambda row: int(row["index"]))
            if [row["index"] for row in feature_rows] != list(range(vlf.FROZEN_VALIDATION_CLIPS)):
                raise QualificationError("full validation R3D extraction is incomplete")
            matrix = np.ascontiguousarray(
                np.stack([row.pop("feature") for row in feature_rows]), dtype=np.dtype("<f4")
            )
            matrix_path = run_dir / "r3d18_real_target_features.f32.npy"
            with matrix_path.open("xb") as handle:
                np.save(handle, matrix, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            sample_clips = sorted(
                [clip for rank_record in sample_ranks for clip in rank_record["clips"]],
                key=lambda row: int(row["manifest_index"]),
            )
            probe = {
                "arm": "A1",
                "weights": "ema",
                "ema_updates": vlf.CALIBRATION_UPDATES,
                "nfe_pair": list(NFE_PAIR),
                "controls": list(CONTROLS),
                "autocast": "cuda-bfloat16",
                "qualified_clips": QUALIFICATION_CLIPS,
                "ranks": sample_ranks,
                "clips": sample_clips,
            }
            _validate_sample_probe(probe)
            provenance_path = manifest_path.parent / "provenance.json"
            payload: dict[str, Any] = {
                "schema": JOB_SCHEMA,
                "status": "pass",
                "frozen": True,
                "validation_only": True,
                "source": source,
                "source_files": {
                    "qualifier": vlf.file_record(Path(__file__)),
                    "poc": vlf.file_record(REPO_ROOT / "tools/video_latent_forcing_poc.py"),
                    "model": vlf.file_record(
                        REPO_ROOT / "robot_wm/modeling/video_latent_forcing/model.py"
                    ),
                },
                "environment": environment,
                "topology": topology,
                "determinism_contract": dict(DETERMINISM_CONTRACT),
                "inputs": {
                    "manifest": manifest_record,
                    "manifest_expected_sha256": args.manifest_sha256,
                    "data_provenance": vlf.file_record(provenance_path),
                    "data_root": str(Path(args.data_root).expanduser().resolve()),
                    **checkpoint_records,
                    "checkpoint_expected_sha256": args.checkpoint_sha256,
                    "r3d18_weight": vlf.file_record(args.r3d18_weight),
                    "r3d18_expected_sha256": args.r3d18_sha256,
                    "r3d18_source_url": R3D18_URL,
                    "phase1_gate": vlf.file_record(args.phase1_gate_record),
                },
                "r3d18_real_target": {
                    "clips": vlf.FROZEN_VALIDATION_CLIPS,
                    "shape": [vlf.FROZEN_VALIDATION_CLIPS, R3D18_FEATURE_DIM],
                    "dtype": "float32-little-endian-c-contiguous",
                    "manifest_order": True,
                    "matrix_sha256": canonical_float32_matrix_sha256(matrix),
                    "clip_id_sorted_matrix_sha256": (
                        clip_id_sorted_float32_matrix_sha256(
                            matrix, [str(row["clip_id"]) for row in rows]
                        )
                    ),
                    "ordered_clip_ids_sha256": _sha256_json(
                        [str(row["clip_id"]) for row in rows]
                    ),
                    "target_video_hashes": feature_rows,
                    "target_video_hashes_sha256": _sha256_json(feature_rows),
                    "file": vlf.file_record(matrix_path),
                },
                "a1_sample_control_probe": probe,
            }
            payload["record_sha256"] = _digest_payload(payload, "record_sha256")
            vlf.atomic_write_json(run_dir / "job_record.json", payload, exclusive=True)
        context.barrier()
        return 0
    finally:
        vlf.close_distributed(context)


def finalize_command(args: argparse.Namespace) -> int:
    if len(args.job_record) != 2:
        raise QualificationError("finalization requires exactly two job records")
    current = vlf.git_record()
    if current.get("dirty") is not False:
        raise QualificationError("finalization requires a clean source tree")
    records = []
    jobs = []
    for index, value in enumerate(args.job_record):
        record = vlf.file_record(value)
        job = vlf.load_json(record["path"], f"qualification job {index}")
        validate_job_record(job, expected_commit=str(current["commit"]))
        records.append(record)
        jobs.append(job)
    comparison = _comparison_payload(jobs[0], jobs[1], already_validated=True)
    output = vlf.approved_artifact_path(args.output)
    if output.exists() or output.parent == REPO_ROOT or REPO_ROOT in output.parents:
        raise QualificationError("qualification output must be fresh and outside Git")
    if any(output == Path(record["path"]).resolve() for record in records):
        raise QualificationError("qualification output cannot overwrite a job record")
    payload: dict[str, Any] = {
        "schema": QUALIFICATION_SCHEMA,
        "status": "pass",
        "frozen": True,
        "validation_only": True,
        "source_commit": current["commit"],
        "phase1_gate": jobs[0]["inputs"]["phase1_gate"],
        "qualified_environment": jobs[0]["environment"],
        "job_records": records,
        "comparison": comparison,
        "authorization": {
            **AUTHORIZATION,
        },
    }
    payload["qualification_sha256"] = _digest_payload(payload, "qualification_sha256")
    vlf.atomic_write_json(output, payload, exclusive=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--data-root", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--checkpoint-sha256", required=True)
    run.add_argument("--r3d18-weight", required=True)
    run.add_argument("--r3d18-sha256", required=True)
    run.add_argument("--phase1-gate-record", required=True)
    run.add_argument("--artifact-root", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--expected-source-commit", required=True)
    run.add_argument("--expected-python", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--job-record", action="append", required=True)
    finalize.add_argument("--output", required=True)
    return parser


def _validate_sha(value: str, label: str) -> None:
    if not HEX64.fullmatch(value):
        raise QualificationError(f"{label} must be a lowercase full SHA-256")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            for value, label in (
                (args.manifest_sha256, "manifest hash"),
                (args.checkpoint_sha256, "checkpoint hash"),
                (args.r3d18_sha256, "R3D-18 hash"),
            ):
                _validate_sha(value, label)
            if not re.fullmatch(r"[0-9a-f]{40}", args.expected_source_commit):
                raise QualificationError("expected source commit must be a full Git SHA")
            return run_command(args)
        return finalize_command(args)
    except (QualificationError, vlf.PocError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Video Latent Forcing determinism qualification error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
