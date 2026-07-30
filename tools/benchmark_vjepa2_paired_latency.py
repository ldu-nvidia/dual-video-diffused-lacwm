#!/usr/bin/env python3
"""Same-B200 paired latency benchmark for the final V-JEPA2 speed claim.

This benchmark is deliberately narrower than the per-arm telemetry grid.  It
compares exactly:

* J1 autonomous deployable sampling at NFE=4; and
* VPM autonomous deployable sampling at NFE=8.

Both checkpoints remain resident in one process on one B200.  Every timed
round invokes both arms on the identical immutable history/action tensors, and
the first/second execution order alternates deterministically.  Timed calls use
only ``sample_future_deployable(..., collect_artifacts=False)``.  Future RGB,
clean V-JEPA targets, online teachers, and trajectory capture are unavailable.

LACWM clock convention: ``sigma=1`` is Gaussian noise and ``sigma=0`` is clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import benchmark_vjepa2_inference as single  # noqa: E402
from tools import vjepa2_paired_recovery as recovery  # noqa: E402


SCHEMA_VERSION = 1
WARMUP_PAIRS = 20
TIMED_PAIRS = 100
BATCH_SIZE = 1
SAMPLE_INDEX = 0
ARM_SPECS = {
    "J1": {
        "directory": "j1_joint_auxiliary_leads",
        "nfe": 4,
        "array_task_id": 4,
    },
    "VPM": {
        "directory": "vpm_parameter_matched_video",
        "nfe": 8,
        "array_task_id": 1,
    },
}
PAIR_LABEL = "J1_autonomous_nfe4_vs_VPM_autonomous_nfe8"
OUTPUT_BASENAME = "paired_j1_nfe4_vs_vpm_nfe8.json"


class PairedLatencyError(RuntimeError):
    """Raised when the paired latency claim would be incomparable."""


def counterbalanced_order(index: int) -> tuple[str, str]:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise PairedLatencyError("counterbalance index must be nonnegative")
    return ("J1", "VPM") if index % 2 == 0 else ("VPM", "J1")


def _summary(values: Sequence[float]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    if (
        not normalized
        or any(not math.isfinite(value) or value <= 0 for value in normalized)
    ):
        raise PairedLatencyError("latency vector is empty or invalid")
    ordered = sorted(normalized)
    return {
        "count": len(normalized),
        "p50": single._percentile(ordered, 50.0),
        "p95": single._percentile(ordered, 95.0),
        "mean": sum(normalized) / len(normalized),
        "min": min(normalized),
        "max": max(normalized),
        "values_sha256": hashlib.sha256(
            single._canonical_json([round(value, 9) for value in normalized])
        ).hexdigest(),
        "values": normalized,
    }


def _canonical_directory(value: str | Path, label: str) -> Path:
    try:
        return single._canonical_directory(value, label)
    except single.BenchmarkError as exc:
        raise PairedLatencyError(str(exc)) from exc


def _canonical_file(value: str | Path, label: str) -> Path:
    try:
        return single._canonical_file(value, label)
    except single.BenchmarkError as exc:
        raise PairedLatencyError(str(exc)) from exc


def _promote_scientific_paths(repo: Path, project_root: Path) -> None:
    """Put scientific packages ahead of the recovery controller checkout."""

    paths = (str(repo), str(project_root))
    for path in paths:
        while path in sys.path:
            sys.path.remove(path)
    # The project supplies ``lam`` and the repository supplies ``robot_wm``.
    # Inserting in this order leaves project_root first and repo second.
    for path in paths:
        sys.path.insert(0, path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return single._read_json(path, label)
    except single.BenchmarkError as exc:
        raise PairedLatencyError(str(exc)) from exc


def _identity_valid(payload: Mapping[str, Any]) -> bool:
    return single._identity_is_valid(payload)


def _record(path: Path, *, identity: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": single._sha256(path),
        "bytes": path.stat().st_size,
    }
    if identity is not None:
        result["identity_sha256"] = identity
    return result


def _instantiated_class_origin(
    instance: Any,
    *,
    package: str,
    root: Path,
    label: str,
) -> dict[str, str]:
    """Bind an instantiated scientific class to its loaded source module."""

    cls = type(instance)
    module_name = str(cls.__module__)
    if module_name != package and not module_name.startswith(f"{package}."):
        raise PairedLatencyError(
            f"{label} class does not come from the {package} package"
        )
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise PairedLatencyError(f"{label} class module lacks a source file")
    origin = Path(module_file).resolve(strict=True)
    if not origin.is_relative_to(root):
        raise PairedLatencyError(
            f"{label} class module escaped the scientific repository"
        )
    return {
        "qualified_name": f"{module_name}.{cls.__qualname__}",
        "module": module_name,
        "file": str(origin),
    }


def _loaded_scientific_module_origins(
    *,
    repo: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Reject and record every loaded robot_wm/lam module with a file."""

    origins: dict[str, str] = {}
    for module_name, module in sorted(sys.modules.items()):
        if module_name == "robot_wm" or module_name.startswith("robot_wm."):
            expected_root = repo
        elif module_name == "lam" or module_name.startswith("lam."):
            expected_root = project_root
        else:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        if not isinstance(module_file, str):
            raise PairedLatencyError(
                f"loaded {module_name} module has an invalid source path"
            )
        origin = Path(module_file).resolve(strict=True)
        if not origin.is_relative_to(expected_root):
            raise PairedLatencyError(
                "loaded robot_wm/lam modules escaped the scientific repository: "
                f"{module_name} -> {origin}"
            )
        origins[module_name] = str(origin)
    if "robot_wm" not in origins or "lam" not in origins:
        raise PairedLatencyError(
            "loaded scientific package provenance is incomplete"
        )
    canonical = single._canonical_json(origins)
    return {
        "count": len(origins),
        "origins": origins,
        "origins_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _file_record_matches(
    record: Any,
    path: Path,
    *,
    sha256: str | None = None,
) -> bool:
    """Require the producer's exact, content-bound file-record schema."""

    expected = {
        "path": str(path),
        "sha256": sha256 if sha256 is not None else single._sha256(path),
        "bytes": path.stat().st_size,
    }
    return isinstance(record, Mapping) and dict(record) == expected


def _recursive_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if child_key == key:
                found.append(child)
            found.extend(_recursive_values(child, key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_recursive_values(child, key))
    return found


def _validate_test_cache(study: Mapping[str, Any]) -> dict[str, Any]:
    test = study.get("inputs", {}).get("splits", {}).get("test", {})
    manifest_record = test.get("clip_manifest")
    cache = test.get("cache")
    if (
        not isinstance(manifest_record, dict)
        or manifest_record.get("entries") != 128
        or not isinstance(cache, dict)
    ):
        raise PairedLatencyError("study lacks the immutable 128-clip test cache")
    manifest = _canonical_file(
        manifest_record.get("path", ""), "test clip manifest"
    )
    if single._sha256(manifest) != manifest_record.get("sha256"):
        raise PairedLatencyError("test clip manifest digest differs")
    metadata_record = cache.get("metadata")
    if not isinstance(metadata_record, dict):
        raise PairedLatencyError("study lacks test cache metadata")
    metadata_path = _canonical_file(
        metadata_record.get("path", ""), "test cache metadata"
    )
    if single._sha256(metadata_path) != metadata_record.get("sha256"):
        raise PairedLatencyError("test cache metadata digest differs")
    metadata = _read_json(metadata_path, "test cache metadata")
    arrays: dict[str, Any] = {}
    for name, file_key, sha_key in (
        ("target", "target_file", "target_sha256"),
        ("rgb", "rgb_file", "rgb_sha256"),
        ("actions", "actions_file", "actions_sha256"),
    ):
        study_record = cache.get(name)
        if not isinstance(study_record, dict):
            raise PairedLatencyError(f"study lacks test {name} array record")
        path = _canonical_file(
            study_record.get("path", ""), f"test {name} array"
        )
        metadata_path_value = Path(str(metadata.get(file_key, "")))
        if not metadata_path_value.is_absolute():
            metadata_path_value = metadata_path.parent / metadata_path_value
        metadata_array = _canonical_file(
            metadata_path_value, f"metadata test {name} array"
        )
        digest = single._sha256(path)
        if (
            path != metadata_array
            or digest != study_record.get("sha256")
            or digest != metadata.get(sha_key)
            or study_record.get("bytes") != path.stat().st_size
        ):
            raise PairedLatencyError(
                f"test {name} array differs from immutable study provenance"
            )
        arrays[name] = {
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "full_sha256_verified": True,
        }
    return {
        "clip_manifest": _record(manifest),
        "cache_metadata": _record(metadata_path),
        "arrays": arrays,
    }


def _validate_submission(
    *,
    path: Path,
    study: Mapping[str, Any],
    slurm_job_id: str,
    bind_current_job: bool = True,
) -> dict[str, Any]:
    payload = _read_json(path, "Slurm submission record")
    paired = payload.get("paired_latency_job")
    recorded_job_id = (
        str(paired.get("job_id", "")).split(";", 1)[0].split("_", 1)[0]
        if isinstance(paired, dict)
        else ""
    )
    if (
        not _identity_valid(payload)
        or payload.get("kind") != "vjepa2_controlled_study_submission"
        or payload.get("study_identity_sha256") != study["identity_sha256"]
        or not isinstance(paired, dict)
        or paired.get("dependency") != "afterok:final_stage_array"
        or paired.get("comparison") != PAIR_LABEL
        or paired.get("nodes") != 1
        or paired.get("gpus") != 1
        or paired.get("same_allocation_pairing") is not True
        or paired.get("runs_final_analyzer_after_benchmark") is not True
        or recorded_job_id
        != (
            slurm_job_id
            if bind_current_job
            else recovery.FAILED_JOB_ID
        )
    ):
        raise PairedLatencyError(
            "submission record does not bind this paired-latency allocation"
        )
    return payload


def _arm_provenance(
    *,
    study_root: Path,
    study: Mapping[str, Any],
    arm_code: str,
) -> dict[str, Any]:
    spec = ARM_SPECS[arm_code]
    run_dir = _canonical_directory(
        study_root / spec["directory"], f"{arm_code} run directory"
    )
    arm_path = _canonical_file(
        run_dir / "arm_manifest.json", f"{arm_code} arm manifest"
    )
    stage_path = _canonical_file(
        run_dir / "stage_manifest_update_1000.json",
        f"{arm_code} final stage manifest",
    )
    stage_outcome_path = _canonical_file(
        run_dir / "stage_outcome_update_1000.json",
        f"{arm_code} final stage outcome",
    )
    outcome_path = _canonical_file(
        run_dir / "outcome.json", f"{arm_code} arm outcome"
    )
    resolved_path = _canonical_file(
        run_dir / "resolved_update_1000.yaml",
        f"{arm_code} resolved config",
    )
    snapshot_path = _canonical_file(
        run_dir / "snapshot.pt", f"{arm_code} final snapshot"
    )
    arm = _read_json(arm_path, f"{arm_code} arm manifest")
    stage = _read_json(stage_path, f"{arm_code} final stage manifest")
    stage_outcome = _read_json(
        stage_outcome_path, f"{arm_code} final stage outcome"
    )
    outcome = _read_json(outcome_path, f"{arm_code} arm outcome")
    observed_snapshot = stage_outcome.get("snapshot_observed_at_stage_end")
    snapshot_sha256 = single._sha256(snapshot_path)
    if (
        not _identity_valid(arm)
        or arm.get("study_identity_sha256") != study["identity_sha256"]
        or arm.get("run_dir") != str(run_dir)
        or arm.get("array_task_id") != spec["array_task_id"]
        or arm.get("arm", {}).get("code") != arm_code
        or arm.get("arm", {}).get("dual_enabled") is not True
        or not _identity_valid(stage)
        or stage.get("arm_identity_sha256") != arm["identity_sha256"]
        or stage.get("arm_code") != arm_code
        or stage.get("stage_endpoint_completed_updates") != 1000
        or stage.get("trainer_terminal_iteration") != 999
        or not _file_record_matches(stage.get("resolved_config"), resolved_path)
        or not _identity_valid(stage_outcome)
        or stage_outcome.get("arm_identity_sha256")
        != arm["identity_sha256"]
        or stage_outcome.get("stage_identity_sha256")
        != stage["identity_sha256"]
        or stage_outcome.get("completed_updates") != 1000
        or not _file_record_matches(
            observed_snapshot,
            snapshot_path,
            sha256=snapshot_sha256,
        )
        or not _identity_valid(outcome)
        or outcome.get("arm_identity_sha256") != arm["identity_sha256"]
        or outcome.get("arm_code") != arm_code
        or outcome.get("completed_updates") != 1000
        or outcome.get("final_snapshot") != observed_snapshot
        or outcome.get("quality_evidence", {}).get(
            "reconstruction_metrics_only"
        )
        is not True
        or outcome.get("latency_evidence", {}).get("complete") is not True
    ):
        raise PairedLatencyError(
            f"{arm_code} final provenance/snapshot identity differs"
        )
    return {
        "code": arm_code,
        "nfe": int(spec["nfe"]),
        "run_dir": run_dir,
        "arm_path": arm_path,
        "arm": arm,
        "stage_path": stage_path,
        "stage": stage,
        "stage_outcome_path": stage_outcome_path,
        "stage_outcome": stage_outcome,
        "outcome_path": outcome_path,
        "outcome": outcome,
        "resolved_path": resolved_path,
        "snapshot_path": snapshot_path,
        "snapshot_sha256": snapshot_sha256,
    }


def _load_model_and_sample(
    *,
    provenance: Mapping[str, Any],
    device: Any,
    repo: Path,
    project_root: Path,
) -> tuple[Any, Any, dict[str, dict[str, str]]]:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(provenance["resolved_path"])
    if config.get("wandb", {}).get("enabled", False):
        config.wandb.enabled = False
    target = str(config.model.get("_target_", ""))
    if not target.endswith(".DualExplicitActionDiTModel"):
        raise PairedLatencyError(
            f"{provenance['code']} config is not a dual explicit-action model"
        )
    test_cache = provenance["study_test_cache"]
    config_value = OmegaConf.to_container(config.viz_dataset, resolve=True)
    manifest_values = {
        str(value)
        for value in _recursive_values(config_value, "clip_manifest")
    }
    metadata_values = {
        str(value)
        for value in _recursive_values(config_value, "cache_metadata")
    }
    if manifest_values != {test_cache["clip_manifest"]["path"]} or (
        metadata_values != {test_cache["cache_metadata"]["path"]}
    ):
        raise PairedLatencyError(
            f"{provenance['code']} viz dataset is not the pinned test cache"
        )
    dataset = instantiate(config.viz_dataset)
    dataset_origin = _instantiated_class_origin(
        dataset,
        package="robot_wm",
        root=repo,
        label=f"{provenance['code']} dataset",
    )
    if len(dataset) != 128:
        raise PairedLatencyError(
            f"{provenance['code']} fixed-test dataset length is not 128"
        )
    sample = dataset[SAMPLE_INDEX]
    del dataset

    model = instantiate(config.model)
    model_origin = _instantiated_class_origin(
        model,
        package="lam",
        root=project_root,
        label=f"{provenance['code']} model",
    )
    snapshot = torch.load(
        provenance["snapshot_path"],
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("_start_iter") != 1000
        or snapshot.get("run_identity_sha256")
        != provenance["arm"]["identity_sha256"]
        or "model" not in snapshot
    ):
        raise PairedLatencyError(
            f"{provenance['code']} snapshot is not the final bound checkpoint"
        )
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise PairedLatencyError(
            f"{provenance['code']} strict checkpoint load failed: {incompatible}"
        )
    del snapshot
    model = model.to(device=device).eval()
    single._assert_teacher_absent(model, config)
    if getattr(model, "time_frequency_transform", None) is not None:
        raise PairedLatencyError(
            f"{provenance['code']} unexpectedly registers an online transform"
        )
    model.evaluation_condition_sources = ("autonomous",)
    model.evaluation_nfe_steps = (provenance["nfe"],)
    model.viz_num_steps = provenance["nfe"]
    model.capture_latent_trajectories = False
    return model, sample, {
        "dataset": dataset_origin,
        "model": model_origin,
    }


def _host_sample(sample: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    import torch

    required = {
        "rgb",
        "actions",
        "morphology_index",
        "clip_index",
    }
    missing = sorted(required - sample.keys())
    if missing:
        raise PairedLatencyError(f"{arm} sample lacks {missing}")
    result: dict[str, Any] = {}
    for key in required:
        value = sample[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        result[key] = value.detach().cpu().contiguous()
    if int(result["clip_index"].reshape(-1)[0]) != SAMPLE_INDEX:
        raise PairedLatencyError(f"{arm} dataset substituted another clip")
    return result


def _sample_identity(sample: Mapping[str, Any], history_frames: int) -> dict[str, Any]:
    rgb = sample["rgb"].unsqueeze(0)
    actions = sample["actions"].unsqueeze(0)
    morphology = sample["morphology_index"].unsqueeze(0)
    history = rgb[:, :history_frames]
    return {
        "sample_index": SAMPLE_INDEX,
        "full_rgb_sha256": single._tensor_sha256(rgb),
        "history_rgb_sha256": single._tensor_sha256(history),
        "actions_sha256": single._tensor_sha256(actions),
        "morphology_index_sha256": single._tensor_sha256(morphology),
    }


def command_benchmark(args: argparse.Namespace) -> int:
    import torch

    recovery_values = (
        args.recovery_protocol,
        args.recovery_submission_record,
        args.recovery_runtime_record,
        args.controller_commit,
    )
    recovery_mode = all(value is not None for value in recovery_values)
    if any(value is not None for value in recovery_values) and not recovery_mode:
        raise PairedLatencyError(
            "recovery protocol, submission, runtime, and controller commit "
            "are all required"
        )

    repo = _canonical_directory(args.repo_root, "repository root")
    try:
        single._assert_clean_commit(repo, args.expected_commit)
    except single.BenchmarkError as exc:
        raise PairedLatencyError(str(exc)) from exc
    project_root = repo / "projects" / "latent_action_models"
    if not project_root.is_dir():
        raise PairedLatencyError("latent-action project root is missing")
    _promote_scientific_paths(repo, project_root)
    controller_repo: Path | None = None
    scientific_import_origins: dict[str, Any] | None = None
    if recovery_mode:
        controller_repo = _canonical_directory(
            REPO_ROOT, "recovery controller repository"
        )
        if controller_repo == repo:
            raise PairedLatencyError(
                "recovery controller and scientific repositories must differ"
            )
        try:
            single._assert_clean_commit(
                controller_repo, args.controller_commit
            )
        except single.BenchmarkError as exc:
            raise PairedLatencyError(str(exc)) from exc
        import importlib.util

        robot_spec = importlib.util.find_spec("robot_wm")
        lam_spec = importlib.util.find_spec("lam")
        if (
            robot_spec is None
            or robot_spec.origin is None
            or not Path(robot_spec.origin).resolve().is_relative_to(repo)
            or lam_spec is None
            or lam_spec.origin is None
            or not Path(lam_spec.origin).resolve().is_relative_to(project_root)
        ):
            raise PairedLatencyError(
                "robot_wm/lam do not resolve from the scientific repository"
            )
        import lam
        import robot_wm

        robot_origin = Path(robot_wm.__file__).resolve()
        lam_origin = Path(lam.__file__).resolve()
        if (
            not robot_origin.is_relative_to(repo)
            or not lam_origin.is_relative_to(project_root)
        ):
            raise PairedLatencyError(
                "loaded robot_wm/lam modules escaped the scientific repository"
            )
        scientific_import_origins = {
            "bootstrap": {
                "robot_wm": str(robot_origin),
                "lam": str(lam_origin),
            },
            "instantiated_classes": {},
        }

    study_root = _canonical_directory(args.study_root, "study root")
    study_path = _canonical_file(
        study_root / "study_manifest.json", "study manifest"
    )
    study = _read_json(study_path, "study manifest")
    if (
        not _identity_valid(study)
        or study.get("kind")
        != "vjepa2_controlled_video_diffusion_study"
        or study.get("study_root") != str(study_root)
        or study.get("inputs", {}).get("repository", {}).get("git_commit")
        != args.expected_commit
    ):
        raise PairedLatencyError("study manifest identity/provenance differs")
    submission_path = _canonical_file(
        args.submission_record, "Slurm submission record"
    )
    if submission_path != study_root / "slurm_submission.json":
        raise PairedLatencyError(
            "submission record must be study_root/slurm_submission.json"
        )
    submission = _validate_submission(
        path=submission_path,
        study=study,
        slurm_job_id=args.slurm_job_id,
        bind_current_job=not recovery_mode,
    )
    recovery_protocol_path: Path | None = None
    recovery_protocol: Mapping[str, Any] | None = None
    recovery_submission_path: Path | None = None
    recovery_submission: Mapping[str, Any] | None = None
    recovery_runtime_path: Path | None = None
    recovery_runtime: Mapping[str, Any] | None = None
    if recovery_mode:
        try:
            (
                recovery_submission_path,
                recovery_submission,
                recovery_protocol_path,
                recovery_protocol,
            ) = recovery.validate_submission(
                args.recovery_submission_record,
                protocol_path=args.recovery_protocol,
                slurm_job_id=args.slurm_job_id,
                output=args.output,
                controller_repo_root=controller_repo,
                controller_commit=args.controller_commit,
                scientific_repo_root=repo,
                scientific_commit=args.expected_commit,
            )
        except recovery.RecoveryError as exc:
            raise PairedLatencyError(str(exc)) from exc
        try:
            recovery_runtime_path, recovery_runtime = (
                recovery.validate_runtime_evidence(
                    args.recovery_runtime_record,
                    protocol_path=args.recovery_protocol,
                    submission_path=args.recovery_submission_record,
                    slurm_job_id=args.slurm_job_id,
                )
            )
        except recovery.RecoveryError as exc:
            raise PairedLatencyError(str(exc)) from exc
    test_cache = _validate_test_cache(study)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise PairedLatencyError("paired latency requires a CUDA device")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if "B200" not in properties.name.upper():
        raise PairedLatencyError(
            f"paired latency requires NVIDIA B200, found {properties.name}"
        )
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    slurm_node = os.environ.get("SLURMD_NODENAME", "")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (
        not slurm_job_id
        or slurm_job_id != args.slurm_job_id
        or os.environ.get("SLURM_JOB_NUM_NODES", "1") != "1"
        or not slurm_node
        or not cuda_visible_devices
        or torch.cuda.device_count() != 1
        or (
            recovery_mode
            and (
                recovery_runtime is None
                or recovery_runtime.get("slurm_environment", {}).get("node")
                != slurm_node
                or recovery_runtime.get("slurm_environment", {}).get(
                    "cuda_visible_devices"
                )
                != cuda_visible_devices
            )
        )
    ):
        raise PairedLatencyError(
            "paired latency requires one recorded Slurm node and one visible GPU"
        )

    provenance = {
        arm: _arm_provenance(
            study_root=study_root, study=study, arm_code=arm
        )
        for arm in ("J1", "VPM")
    }
    for value in provenance.values():
        value["study_test_cache"] = test_cache

    models: dict[str, Any] = {}
    host_samples: dict[str, Any] = {}
    instantiated_classes: dict[str, Any] = {}
    for arm in ("VPM", "J1"):
        model, sample, class_origins = _load_model_and_sample(
            provenance=provenance[arm],
            device=device,
            repo=repo,
            project_root=project_root,
        )
        models[arm] = model
        host_samples[arm] = _host_sample(sample, arm=arm)
        instantiated_classes[arm] = class_origins
        del sample
    if recovery_mode:
        assert scientific_import_origins is not None
        scientific_import_origins["instantiated_classes"] = (
            instantiated_classes
        )
        scientific_import_origins["before_timing"] = (
            _loaded_scientific_module_origins(
                repo=repo,
                project_root=project_root,
            )
        )
    history_frames = int(models["J1"].num_history_frames)
    future_frames = int(models["J1"].num_future_frames)
    if (
        history_frames != 5
        or future_frames != 8
        or int(models["VPM"].num_history_frames) != history_frames
        or int(models["VPM"].num_future_frames) != future_frames
    ):
        raise PairedLatencyError("paired arms disagree on 5-history/8-future")
    identities = {
        arm: _sample_identity(host_samples[arm], history_frames)
        for arm in ("J1", "VPM")
    }
    if identities["J1"] != identities["VPM"]:
        raise PairedLatencyError(
            "J1 and VPM immutable latency inputs are not bit-identical"
        )

    shared = host_samples["J1"]
    history_rgb = shared["rgb"][:history_frames].unsqueeze(0).to(device)
    actions = shared["actions"].unsqueeze(0).to(device)
    morphology = shared["morphology_index"].unsqueeze(0).to(device)
    sample_ids = torch.tensor([SAMPLE_INDEX], dtype=torch.long, device=device)
    # Future RGB and cached clean V-JEPA targets are not retained in the
    # deployable input bundle and are never accepted by the public sampler.
    del host_samples, shared

    hook_calls = {"J1": 0, "VPM": 0}
    hooks = []
    for arm in ("J1", "VPM"):
        def count_calls(_module, _inputs, _output, *, arm_code=arm):
            hook_calls[arm_code] += 1

        hooks.append(
            models[arm].forward_model.register_forward_hook(count_calls)
        )

    def invoke(arm: str, *, collect_artifacts: bool) -> None:
        model = models[arm]
        hook_calls[arm] = 0
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            output = model.sample_future_deployable(
                history_rgb,
                actions,
                morphology_index=morphology,
                collect_artifacts=collect_artifacts,
                sample_ids=sample_ids,
            )
        del output

    def validate_call(
        arm: str,
        *,
        artifacts_collected: int,
        verify_independent_hook: bool,
    ) -> dict[str, int]:
        nfe = int(provenance[arm]["nfe"])
        if verify_independent_hook and hook_calls[arm] != nfe:
            raise PairedLatencyError(
                f"{arm} Wan hook count {hook_calls[arm]} != NFE {nfe}"
            )
        counters = getattr(models[arm], "_last_sampling_counters", None)
        if not isinstance(counters, Mapping):
            raise PairedLatencyError(f"{arm} sampler counters are unavailable")
        try:
            normalized = single._validated_sampler_counters(
                counters,
                nfe=nfe,
                artifacts_collected=artifacts_collected,
                deployment_mode=1,
            )
        except single.BenchmarkError as exc:
            raise PairedLatencyError(str(exc)) from exc
        return normalized

    audit: dict[str, Any] = {}
    try:
        for arm in ("J1", "VPM"):
            invoke(arm, collect_artifacts=True)
            counters = validate_call(
                arm,
                artifacts_collected=1,
                verify_independent_hook=True,
            )
            artifacts = models[arm].pop_visualization_artifacts()
            if not isinstance(artifacts, Mapping):
                raise PairedLatencyError(f"{arm} audit artifacts are absent")
            nfe = int(provenance[arm]["nfe"])
            artifact_wan = int(
                artifacts[f"wan_call_count_nfe_{nfe}"].reshape(-1)[0]
            )
            teacher = int(
                artifacts["online_teacher_call_count"].reshape(-1)[0]
            )
            clean_available = int(
                artifacts["auxiliary_clean_available"].reshape(-1)[0]
            )
            deployment = int(
                artifacts["deployment_mode"].reshape(-1)[0]
            )
            forbidden = {
                "video_clean",
                "tf_clean",
                "ground_truth_future_uint8",
            }.intersection(artifacts)
            if (
                artifact_wan != nfe
                or teacher != 0
                or clean_available != 0
                or deployment != 1
                or forbidden
                or getattr(models[arm], "capture_latent_trajectories", True)
                is not False
            ):
                raise PairedLatencyError(
                    f"{arm} deployable audit violates clean/teacher/capture "
                    f"contract; forbidden={sorted(forbidden)}"
                )
            audit[arm] = {
                "nfe": nfe,
                "wan_calls": artifact_wan,
                "online_teacher_calls": teacher,
                "clean_auxiliary_available": clean_available,
                "deployment_mode": deployment,
                "trajectory_capture_enabled": False,
                "independent_forward_hook_wan_count": hook_calls[arm],
                "forward_hook_active_during_timing": False,
                "public_entrypoint": (
                    "DualExplicitActionDiTModel.sample_future_deployable"
                ),
                "sampler_counters": counters,
            }
            del artifacts

        # Forward hooks independently establish the configured Wan call count
        # above, then are removed so their per-call Python callbacks cannot
        # introduce an NFE-dependent timing bias.
        for hook in hooks:
            hook.remove()
        hooks.clear()

        def untimed_once(arm: str) -> None:
            invoke(arm, collect_artifacts=False)
            validate_call(
                arm,
                artifacts_collected=0,
                verify_independent_hook=False,
            )
            if models[arm].pop_visualization_artifacts() is not None:
                raise PairedLatencyError(
                    f"{arm} capture-free warmup materialized artifacts"
                )

        def timed_once(arm: str) -> float:
            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            invoke(arm, collect_artifacts=False)
            torch.cuda.synchronize(device)
            finished = time.perf_counter_ns()
            validate_call(
                arm,
                artifacts_collected=0,
                verify_independent_hook=False,
            )
            if models[arm].pop_visualization_artifacts() is not None:
                raise PairedLatencyError(
                    f"{arm} timed path materialized artifacts"
                )
            return (finished - started) / 1_000_000.0

        for pair_index in range(WARMUP_PAIRS):
            for arm in counterbalanced_order(pair_index):
                untimed_once(arm)

        torch.cuda.reset_peak_memory_stats(device)
        values = {"J1": [], "VPM": []}
        rounds = []
        for pair_index in range(TIMED_PAIRS):
            order = counterbalanced_order(pair_index)
            pair_values: dict[str, float] = {}
            for arm in order:
                pair_values[arm] = timed_once(arm)
            values["J1"].append(pair_values["J1"])
            values["VPM"].append(pair_values["VPM"])
            rounds.append(
                {
                    "pair_index": pair_index,
                    "execution_order": list(order),
                    "J1_latency_ms": pair_values["J1"],
                    "VPM_latency_ms": pair_values["VPM"],
                    "favorable_difference_ms": (
                        pair_values["VPM"] - pair_values["J1"]
                    ),
                    "relative_improvement": (
                        (pair_values["VPM"] - pair_values["J1"])
                        / pair_values["VPM"]
                    ),
                }
            )
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    finally:
        for hook in hooks:
            hook.remove()

    differences = [
        vpm - j1 for j1, vpm in zip(values["J1"], values["VPM"])
    ]
    relative = [
        (vpm - j1) / vpm
        for j1, vpm in zip(values["J1"], values["VPM"])
    ]
    if any(not math.isfinite(value) for value in differences + relative):
        raise PairedLatencyError("paired timing effect contains non-finite values")
    runtime_assets_after_timing: dict[str, Any] | None = None
    if recovery_mode:
        assert (
            scientific_import_origins is not None
            and recovery_protocol is not None
            and recovery_runtime is not None
        )
        scientific_import_origins["after_timing"] = (
            _loaded_scientific_module_origins(
                repo=repo,
                project_root=project_root,
            )
        )
        try:
            runtime_assets_after_timing = recovery.validate_runtime_assets(
                recovery_protocol["runtime"]["wan_dir"],
                recovery_protocol["runtime"]["videox_home"],
            )
        except recovery.RecoveryError as exc:
            raise PairedLatencyError(str(exc)) from exc
        if (
            runtime_assets_after_timing
            != recovery_runtime["runtime_assets_before_model_loading"]
        ):
            raise PairedLatencyError(
                "VideoX/Wan runtime assets changed during paired timing"
            )
    order_strata: dict[str, Any] = {}
    for first_arm in ("J1", "VPM"):
        indexes = [
            index
            for index in range(TIMED_PAIRS)
            if counterbalanced_order(index)[0] == first_arm
        ]
        order_strata[f"{first_arm}_first"] = {
            "count": len(indexes),
            "mean_favorable_difference_ms": (
                sum(differences[index] for index in indexes) / len(indexes)
            ),
            "mean_relative_improvement": (
                sum(relative[index] for index in indexes) / len(indexes)
            ),
        }
    arm_payloads: dict[str, Any] = {}
    for arm in ("J1", "VPM"):
        item = provenance[arm]
        latency = _summary(values[arm])
        arm_payloads[arm] = {
            "source": "autonomous",
            "nfe": item["nfe"],
            "arm_manifest": _record(
                item["arm_path"], identity=item["arm"]["identity_sha256"]
            ),
            "stage_manifest": _record(
                item["stage_path"], identity=item["stage"]["identity_sha256"]
            ),
            "stage_outcome": _record(
                item["stage_outcome_path"],
                identity=item["stage_outcome"]["identity_sha256"],
            ),
            "arm_outcome": _record(
                item["outcome_path"],
                identity=item["outcome"]["identity_sha256"],
            ),
            "resolved_config": _record(item["resolved_path"]),
            "snapshot": {
                "path": str(item["snapshot_path"]),
                "sha256": item["snapshot_sha256"],
                "bytes": item["snapshot_path"].stat().st_size,
                "checkpoint_start_iter": 1000,
                "run_identity_sha256": item["arm"]["identity_sha256"],
            },
            "latency_ms": latency,
            "generated_frames_per_second_at_p95": (
                future_frames * 1000.0 / latency["p95"]
            ),
            "audit": audit[arm],
        }
    rounds_sha256 = hashlib.sha256(
        single._canonical_json(rounds)
    ).hexdigest()
    unsigned_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "vjepa2_controlled_study_paired_latency",
        "created_at_utc": single._now(),
        "comparison": PAIR_LABEL,
        "git_commit": args.expected_commit,
        "study": _record(
            study_path, identity=study["identity_sha256"]
        ),
        "submission": _record(
            submission_path, identity=submission["identity_sha256"]
        ),
        "slurm": {
            "job_id": slurm_job_id,
            "same_allocation": True,
            "same_node": slurm_node,
            "cuda_visible_devices": cuda_visible_devices,
        },
        "device": {
            "name": properties.name,
            "index": torch.cuda.current_device(),
            "total_memory_bytes": properties.total_memory,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "peak_allocated_bytes_with_both_models_resident": peak_memory,
        },
        "protocol": {
            "batch_size": BATCH_SIZE,
            "sample_index": SAMPLE_INDEX,
            "warmup_pairs": WARMUP_PAIRS,
            "timed_pairs": TIMED_PAIRS,
            "counterbalance": (
                "even pair J1-first; odd pair VPM-first"
            ),
            "J1_first_pairs": TIMED_PAIRS // 2,
            "VPM_first_pairs": TIMED_PAIRS // 2,
            "same_process": True,
            "same_B200": True,
            "both_models_resident": True,
            "identical_immutable_batch_inputs": True,
            "entrypoint": (
                "DualExplicitActionDiTModel.sample_future_deployable"
            ),
            "collect_artifacts_timed": False,
            "trajectory_capture_timed": False,
            "forward_hooks_active_during_timing": False,
            "future_ground_truth_rgb_available_to_sampler": False,
            "clean_auxiliary_target_available_to_sampler": False,
            "online_teacher_calls": 0,
            "cuda_synchronize_before_and_after_each_arm": True,
            "timing_scope": (
                "history preparation inside model + Wan calls + VAE decode"
            ),
            "sigma_convention": "sigma=1 noise, sigma=0 clean",
        },
        "immutable_input": {
            **identities["J1"],
            "test_cache": test_cache,
        },
        "arms": arm_payloads,
        "paired": {
            "favorable_direction": "positive means J1@4 is faster",
            "favorable_difference_ms": differences,
            "relative_improvement": relative,
            "mean_favorable_difference_ms": (
                sum(differences) / len(differences)
            ),
            "mean_relative_improvement": (
                sum(relative) / len(relative)
            ),
            "order_strata": order_strata,
            "rounds_sha256": rounds_sha256,
            "rounds": rounds,
        },
    }
    if recovery_mode:
        assert (
            controller_repo is not None
            and recovery_protocol_path is not None
            and recovery_protocol is not None
            and recovery_submission_path is not None
            and recovery_submission is not None
            and recovery_runtime_path is not None
            and recovery_runtime is not None
            and scientific_import_origins is not None
            and runtime_assets_after_timing is not None
        )
        unsigned_payload["recovery"] = {
            "mode": "external_validator_only_recovery",
            "original_failed_job_id": recovery.FAILED_JOB_ID,
            "original_failure_started_timing": False,
            "legacy_study_tree_mutated": False,
            "controller": {
                "root": str(controller_repo),
                "commit": args.controller_commit,
            },
            "scientific_import_origins": dict(
                scientific_import_origins
            ),
            "runtime_assets_after_timing": dict(
                runtime_assets_after_timing
            ),
            "protocol": _record(
                recovery_protocol_path,
                identity=recovery_protocol["identity_sha256"],
            ),
            "submission": _record(
                recovery_submission_path,
                identity=recovery_submission["identity_sha256"],
            ),
            "runtime": _record(
                recovery_runtime_path,
                identity=recovery_runtime["identity_sha256"],
            ),
        }
    payload = single._identity_payload(unsigned_payload)
    output = Path(args.output).expanduser()
    if recovery_mode:
        assert recovery_protocol is not None
        expected_output = Path(recovery_protocol["paths"]["paired_latency"])
        expected_parent = _canonical_directory(
            expected_output.parent,
            "paired recovery output directory",
        )
        if expected_output.is_relative_to(study_root):
            raise PairedLatencyError(
                "paired recovery output must be outside the immutable study"
            )
    else:
        expected_parent = _canonical_directory(
            study_root / "paired_latency",
            "paired latency output directory",
        )
        expected_output = expected_parent / OUTPUT_BASENAME
    if (
        output.parent.resolve(strict=True) != expected_parent
        or output != expected_output
    ):
        raise PairedLatencyError(
            f"output must be {expected_output}"
        )
    try:
        single._exclusive_json(output, payload)
    except (OSError, single.BenchmarkError) as exc:
        raise PairedLatencyError(str(exc)) from exc
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(output),
                "comparison": PAIR_LABEL,
                "paired_rounds": TIMED_PAIRS,
                "mean_relative_improvement": payload["paired"][
                    "mean_relative_improvement"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--submission-record", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--recovery-protocol")
    parser.add_argument("--recovery-submission-record")
    parser.add_argument("--recovery-runtime-record")
    parser.add_argument("--controller-commit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return command_benchmark(args)
    except (
        PairedLatencyError,
        single.BenchmarkError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
