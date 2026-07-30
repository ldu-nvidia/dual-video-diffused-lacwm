#!/usr/bin/env python3
"""Benchmark one deployable dual-video sampler source/NFE on one B200.

The benchmark calls the public
``DualExplicitActionDiTModel.sample_future_deployable`` entrypoint with batch
size one and exactly the five observed RGB frames.  Dataset loading is outside
the timed region; each repetition includes history-video latent preparation,
action/context preparation, exactly N Wan backbone calls, and VAE decoding.
One separate untimed audited call checks artifacts and counters; timed calls
explicitly disable trajectory, final-state, uint8, and CPU evidence
materialization.

Before timing, an additional untimed audit compares the deployable prediction
with the prediction from the ordinary full-clip evaluation API under identical
noise.  The future predictions must agree to numerical tolerance.  The
full-clip call is used only for this audit; neither future RGB nor its decoded
ground truth is passed to the timed deployment entrypoint.

Protocol is fixed: 20 warmups, 100 synchronized wall-clock repetitions, one
source and one NFE per process.  The offline V-JEPA teacher must be absent from
the model and every sampler artifact must report zero online teacher calls.
Only ``autonomous`` and the same-checkpoint ``off`` intervention are valid
batch-one latency sources. ``autonomous_shuffled`` is a paired quality-only
mechanism control requiring an effective batch of at least two.

LACWM clock convention: sigma=1 is noise and sigma=0 is clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
BATCH_SIZE = 1
WARMUPS = 20
REPETITIONS = 100
ALLOWED_NFE = (1, 2, 4, 6, 8, 12, 20)
ALLOWED_SOURCES = ("autonomous", "off")
DEPLOYABLE_SOURCES = ALLOWED_SOURCES
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BenchmarkError(RuntimeError):
    """Raised when benchmark provenance or runtime counters differ."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(memoryview(value.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def _identity_is_valid(payload: Mapping[str, Any]) -> bool:
    recorded = payload.get("identity_sha256")
    if not isinstance(recorded, str) or SHA256_RE.fullmatch(recorded) is None:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest() == recorded


def _canonical_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise BenchmarkError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise BenchmarkError(f"{label} must be a non-symlink regular file")
    if info.st_size <= 0:
        raise BenchmarkError(f"{label} is empty")
    return path.resolve(strict=True)


def _canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise BenchmarkError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise BenchmarkError(f"{label} must be a non-symlink directory")
    return path.resolve(strict=True)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkError(f"{label} must contain an object")
    return payload


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise BenchmarkError(
            completed.stderr.strip() or completed.stdout.strip()
        )
    return completed.stdout.strip()


def _assert_clean_commit(repo: Path, expected_commit: str) -> None:
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise BenchmarkError("expected commit must be 40 lowercase hex characters")
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise BenchmarkError(f"Git commit differs: {actual} != {expected_commit}")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise BenchmarkError(
            "repository must be clean: " + status.replace("\n", "; ")
        )


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise BenchmarkError("cannot summarize an empty latency vector")
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _validated_sampler_counters(
    counters: Mapping[str, Any],
    *,
    nfe: int,
    artifacts_collected: int,
    deployment_mode: int = 1,
) -> dict[str, int]:
    expected_calls = {f"autonomous:nfe_{nfe}": nfe}
    call_map = dict(counters.get("wan_calls_by_source_nfe", {}))
    normalized = {
        "wan_calls": int(counters.get("wan_calls_total", -1)),
        "online_teacher_calls": int(
            counters.get("online_teacher_calls", -1)
        ),
        "auxiliary_clean_available": int(
            counters.get("auxiliary_clean_available", -1)
        ),
        "artifacts_collected": int(
            counters.get("artifacts_collected", -1)
        ),
        "deployment_mode": int(counters.get("deployment_mode", -1)),
    }
    if call_map != expected_calls or normalized["wan_calls"] != nfe:
        raise BenchmarkError(
            f"sampler counter map differs: {call_map!r} != {expected_calls!r}"
        )
    if normalized["online_teacher_calls"] != 0:
        raise BenchmarkError("sampler counter reports an online teacher call")
    if normalized["auxiliary_clean_available"] != 0:
        raise BenchmarkError("sampler counter reports a clean auxiliary target")
    if normalized["artifacts_collected"] != artifacts_collected:
        raise BenchmarkError(
            "sampler artifact flag differs: "
            f"{normalized['artifacts_collected']} != {artifacts_collected}"
        )
    if normalized["deployment_mode"] != deployment_mode:
        raise BenchmarkError(
            "sampler deployment-mode flag differs: "
            f"{normalized['deployment_mode']} != {deployment_mode}"
        )
    return normalized


def _to_batch(sample: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    batch: dict[str, Any] = {}
    required = ("rgb", "actions", "morphology_index")
    missing = [key for key in required if key not in sample]
    if missing:
        raise BenchmarkError(f"fixed evaluation sample lacks {missing}")
    for key in ("rgb", "actions", "mask", "morphology_index"):
        if key not in sample:
            continue
        value = sample[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        batch[key] = value.unsqueeze(0).to(device=device, non_blocking=False)
    if batch["rgb"].shape[0] != BATCH_SIZE:
        raise BenchmarkError("benchmark batch size differs from one")
    return batch


def _assert_teacher_absent(model: Any, config: Any) -> None:
    from omegaconf import OmegaConf

    suspicious_modules = []
    for name, module in model.named_modules():
        identity = f"{name}:{type(module).__module__}.{type(module).__name__}".lower()
        if "vjepa" in identity or "jepa" in identity or "teacher" in identity:
            suspicious_modules.append(identity)
    if suspicious_modules:
        raise BenchmarkError(
            "online teacher-like module is registered: "
            + "; ".join(suspicious_modules[:8])
        )
    serialized = json.dumps(
        OmegaConf.to_container(config.model, resolve=True),
        sort_keys=True,
    ).lower()
    for forbidden in (
        "vjepa2_target",
        "vjepa_source",
        "teacher_checkpoint",
        "teacher_encoder",
    ):
        if forbidden in serialized:
            raise BenchmarkError(
                f"model config contains forbidden online teacher marker: {forbidden}"
            )


def _load_model_and_sample(args: argparse.Namespace):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    repo = _canonical_directory(args.repo_root, "repository root")
    _assert_clean_commit(repo, args.expected_commit)
    project_root = repo / "projects" / "latent_action_models"
    if not project_root.is_dir():
        raise BenchmarkError("latent-action project root is missing")
    for import_root in (str(repo), str(project_root)):
        if import_root not in sys.path:
            sys.path.insert(0, import_root)

    run_dir = _canonical_directory(args.run_dir, "arm run directory")
    resolved = _canonical_file(args.resolved_config, "resolved config")
    snapshot_path = _canonical_file(args.snapshot, "snapshot")
    arm_manifest_path = _canonical_file(args.arm_manifest, "arm manifest")
    study_manifest_path = _canonical_file(
        args.study_manifest, "study manifest"
    )
    stage_manifest_path = _canonical_file(
        args.stage_manifest, "final stage manifest"
    )
    stage_outcome_path = _canonical_file(
        args.stage_outcome, "final stage outcome"
    )
    arm_manifest = _read_json(arm_manifest_path, "arm manifest")
    study_manifest = _read_json(study_manifest_path, "study manifest")
    stage_manifest = _read_json(stage_manifest_path, "final stage manifest")
    if not _identity_is_valid(arm_manifest):
        raise BenchmarkError("arm manifest identity SHA-256 is invalid")
    if Path(str(arm_manifest.get("run_dir", ""))) != run_dir:
        raise BenchmarkError("arm manifest points to another run directory")
    arm = arm_manifest.get("arm", {})
    if not isinstance(arm, dict) or arm.get("dual_enabled") is not True:
        raise BenchmarkError("true latency benchmark requires a dual arm")
    if str(arm.get("code")) not in {"VPM", "A1", "J0", "J1"}:
        raise BenchmarkError("unsupported dual arm code")
    if (
        not _identity_is_valid(study_manifest)
        or study_manifest.get("kind")
        != "vjepa2_controlled_video_diffusion_study"
        or study_manifest.get("identity_sha256")
        != arm_manifest.get("study_identity_sha256")
    ):
        raise BenchmarkError("study manifest identity/provenance differs")
    if (
        not _identity_is_valid(stage_manifest)
        or stage_manifest.get("kind")
        != "vjepa2_controlled_study_stage"
        or stage_manifest.get("arm_identity_sha256")
        != arm_manifest["identity_sha256"]
        or stage_manifest.get("arm_code") != arm.get("code")
        or stage_manifest.get("stage_endpoint_completed_updates") != 1000
        or stage_manifest.get("trainer_terminal_iteration") != 999
        or stage_manifest.get("resolved_config", {}).get("path")
        != str(resolved)
        or stage_manifest.get("resolved_config", {}).get("sha256")
        != _sha256(resolved)
    ):
        raise BenchmarkError("final stage manifest identity/provenance differs")
    stage_outcome = _read_json(stage_outcome_path, "final stage outcome")
    snapshot_record = stage_outcome.get("snapshot_observed_at_stage_end")
    if (
        not _identity_is_valid(stage_outcome)
        or stage_outcome.get("kind")
        != "vjepa2_controlled_study_stage_outcome"
        or stage_outcome.get("arm_identity_sha256")
        != arm_manifest["identity_sha256"]
        or stage_outcome.get("stage_identity_sha256")
        != stage_manifest["identity_sha256"]
        or stage_outcome.get("arm_code") != arm.get("code")
        or stage_outcome.get("completed_updates") != 1000
        or not isinstance(snapshot_record, dict)
        or snapshot_record.get("path") != str(snapshot_path)
        or not isinstance(snapshot_record.get("sha256"), str)
        or SHA256_RE.fullmatch(snapshot_record["sha256"]) is None
    ):
        raise BenchmarkError(
            "final stage outcome/snapshot provenance differs"
        )

    config = OmegaConf.load(resolved)
    model_target = str(config.model.get("_target_", ""))
    if not model_target.endswith(".DualExplicitActionDiTModel"):
        raise BenchmarkError("resolved config is not a dual explicit-action model")
    if config.get("wandb", {}).get("enabled", False):
        # This standalone benchmark must never initialize W&B; changing the
        # local config object does not alter the hashed resolved input.
        config.wandb.enabled = False
    model = instantiate(config.model)
    snapshot = torch.load(
        snapshot_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if (
        not isinstance(snapshot, Mapping)
        or "model" not in snapshot
        or snapshot.get("_start_iter") != 1000
    ):
        raise BenchmarkError(
            "snapshot lacks model state or is not update 1000"
        )
    if snapshot.get("run_identity_sha256") != arm_manifest["identity_sha256"]:
        raise BenchmarkError("snapshot and arm run identities differ")
    state = snapshot["model"]
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise BenchmarkError(f"strict snapshot load failed: {incompatible}")
    del snapshot, state

    device = torch.device(args.device)
    if device.type != "cuda":
        raise BenchmarkError("true latency benchmark requires a CUDA device")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if "B200" not in properties.name.upper():
        raise BenchmarkError(
            f"benchmark requires NVIDIA B200, found {properties.name}"
        )
    model = model.to(device=device).eval()
    _assert_teacher_absent(model, config)
    if getattr(model, "time_frequency_transform", None) is not None:
        raise BenchmarkError("online auxiliary transform is unexpectedly present")

    dataset = instantiate(config.viz_dataset)
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise BenchmarkError(
            f"sample index {args.sample_index} outside [0,{len(dataset) - 1}]"
        )
    sample = dataset[args.sample_index]
    batch = _to_batch(sample, device)
    del dataset, sample
    # ``_sample_future`` uses its autonomous slot as the returned primary
    # result. Map each deployable intervention into that one slot so exactly
    # one source (and therefore exactly N Wan calls) executes.
    model.evaluation_condition_sources = ("autonomous",)
    model.evaluation_nfe_steps = (args.nfe,)
    model.viz_num_steps = args.nfe
    model.capture_latent_trajectories = True
    if args.source == "off":
        model.condition_on_tf = False
        model.condition_on_tf_clock = False
    return (
        repo,
        run_dir,
        resolved,
        snapshot_path,
        arm_manifest_path,
        arm_manifest,
        study_manifest_path,
        study_manifest,
        stage_manifest_path,
        stage_manifest,
        stage_outcome_path,
        stage_outcome,
        config,
        model,
        batch,
        properties,
    )


def command_benchmark(args: argparse.Namespace) -> int:
    import inspect

    import torch

    if args.nfe not in ALLOWED_NFE:
        raise BenchmarkError(f"NFE must be one of {ALLOWED_NFE}")
    if args.source not in ALLOWED_SOURCES:
        raise BenchmarkError(f"source must be one of {ALLOWED_SOURCES}")
    if args.sample_index != 0:
        raise BenchmarkError(
            "controlled-study latency sample index is pinned to zero"
        )
    (
        repo,
        run_dir,
        resolved,
        snapshot,
        arm_manifest_path,
        arm_manifest,
        study_manifest_path,
        study_manifest,
        stage_manifest_path,
        stage_manifest,
        stage_outcome_path,
        stage_outcome,
        config,
        model,
        batch,
        properties,
    ) = _load_model_and_sample(args)

    calls = 0

    def count_wan_calls(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.forward_model.register_forward_hook(count_wan_calls)
    artifact_call_key = f"wan_call_count_nfe_{args.nfe}"
    deployable_signature = inspect.signature(model.sample_future_deployable)
    if "collect_artifacts" not in deployable_signature.parameters:
        handle.remove()
        raise BenchmarkError(
            "deployable sampler lacks collect_artifacts=False timed path"
        )
    history_rgb = batch["rgb"][:, : model.num_history_frames]
    if history_rgb.shape[1] != model.num_history_frames:
        handle.remove()
        raise BenchmarkError(
            "fixed evaluation sample does not contain the declared history"
        )
    sample_identity = {
        "sample_index": args.sample_index,
        "full_rgb_sha256": _tensor_sha256(batch["rgb"]),
        "history_rgb_sha256": _tensor_sha256(history_rgb),
        "actions_sha256": _tensor_sha256(batch["actions"]),
        "morphology_index_sha256": _tensor_sha256(
            batch["morphology_index"]
        ),
    }

    def invoke_sampler(*, collect_artifacts: bool):
        return model.sample_future_deployable(
            history_rgb,
            batch["actions"],
            morphology_index=batch["morphology_index"],
            collect_artifacts=collect_artifacts,
        )

    def history_invariance_audit() -> dict[str, Any]:
        """Prove full evaluation does not alter the deployable prediction."""
        nonlocal calls
        calls = 0
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        ):
            full_prediction, full_ground_truth = model._sample_future(
                batch["rgb"],
                batch["actions"],
                morphology_index=batch["morphology_index"],
                auxiliary_target=None,
                collect_artifacts=False,
                deployment_mode=False,
            )
        full_calls = calls
        if model.pop_visualization_artifacts() is not None:
            raise BenchmarkError(
                "full-clip invariance audit materialized visualization artifacts"
            )
        full_counters = getattr(model, "_last_sampling_counters", None)
        if not isinstance(full_counters, Mapping):
            raise BenchmarkError(
                "full-clip invariance audit lacks sampler counters"
            )
        _validated_sampler_counters(
            full_counters,
            nfe=args.nfe,
            artifacts_collected=0,
            deployment_mode=0,
        )
        if full_calls != args.nfe:
            raise BenchmarkError(
                f"full-clip audit hook count differs: {full_calls} != {args.nfe}"
            )
        if full_ground_truth is None:
            raise BenchmarkError(
                "ordinary full-clip evaluation did not construct ground truth"
            )
        full_future = full_prediction[:, :, -model.num_future_frames :]

        calls = 0
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        ):
            deployable_prediction = invoke_sampler(collect_artifacts=False)
        deployable_calls = calls
        if model.pop_visualization_artifacts() is not None:
            raise BenchmarkError(
                "deployable invariance audit materialized visualization artifacts"
            )
        deployable_counters = getattr(model, "_last_sampling_counters", None)
        if not isinstance(deployable_counters, Mapping):
            raise BenchmarkError(
                "deployable invariance audit lacks sampler counters"
            )
        _validated_sampler_counters(
            deployable_counters,
            nfe=args.nfe,
            artifacts_collected=0,
            deployment_mode=1,
        )
        if deployable_calls != args.nfe:
            raise BenchmarkError(
                "deployable audit hook count differs: "
                f"{deployable_calls} != {args.nfe}"
            )
        if tuple(full_future.shape) != tuple(deployable_prediction.shape):
            raise BenchmarkError(
                "full/deployable prediction shape differs: "
                f"{tuple(full_future.shape)} != "
                f"{tuple(deployable_prediction.shape)}"
            )
        delta = (
            full_future.detach().float()
            - deployable_prediction.detach().float()
        ).abs()
        max_abs = float(delta.max().item())
        mean_abs = float(delta.mean().item())
        tolerance = 1.0e-6
        exact_equal = bool(torch.equal(full_future, deployable_prediction))
        if max_abs > tolerance:
            raise BenchmarkError(
                "future RGB changed when unavailable full-clip frames were "
                f"removed: max_abs={max_abs:.9g} > {tolerance:.9g}"
            )
        return {
            "passed": True,
            "comparison": "full_clip_future_vs_history_only_deployable",
            "full_clip_call_timed": False,
            "full_clip_ground_truth_constructed_for_audit_only": True,
            "deployable_future_ground_truth_available": False,
            "history_frames": int(model.num_history_frames),
            "generated_future_frames": int(model.num_future_frames),
            "full_wan_calls": full_calls,
            "deployable_wan_calls": deployable_calls,
            "exact_equal": exact_equal,
            "absolute_tolerance": tolerance,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
        }

    def audited_counter_run() -> dict[str, int]:
        nonlocal calls
        calls = 0
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        ):
            outputs = invoke_sampler(collect_artifacts=True)
        del outputs
        artifacts = model.pop_visualization_artifacts()
        if not isinstance(artifacts, Mapping):
            raise BenchmarkError("sampler did not return audit artifacts")
        teacher = int(artifacts["online_teacher_call_count"].reshape(-1)[0])
        artifact_calls = int(artifacts[artifact_call_key].reshape(-1)[0])
        clean_available = int(
            artifacts["auxiliary_clean_available"].reshape(-1)[0]
        )
        deployment_mode = int(
            artifacts["deployment_mode"].reshape(-1)[0]
        )
        if calls != args.nfe or artifact_calls != args.nfe:
            raise BenchmarkError(
                "reported NFE differs from actual Wan calls: "
                f"hook={calls}, artifact={artifact_calls}, nfe={args.nfe}"
            )
        if teacher != 0:
            raise BenchmarkError(
                f"sampler reported {teacher} online teacher calls"
            )
        forbidden_artifacts = {
            "tf_clean",
            "video_clean",
            "ground_truth_future_uint8",
        }.intersection(artifacts)
        if clean_available != 0 or forbidden_artifacts:
            raise BenchmarkError(
                "deployable sampler consumed or exposed unavailable clean data: "
                f"{sorted(forbidden_artifacts)}"
            )
        if deployment_mode != 1:
            raise BenchmarkError(
                "artifact audit did not execute in deployment mode"
            )
        sampler_counters = getattr(model, "_last_sampling_counters", None)
        if not isinstance(sampler_counters, Mapping):
            raise BenchmarkError("audited sampler counters are unavailable")
        _validated_sampler_counters(
            sampler_counters,
            nfe=args.nfe,
            artifacts_collected=1,
            deployment_mode=1,
        )
        del artifacts
        return {
            "wan_calls": calls,
            "online_teacher_calls": teacher,
            "auxiliary_clean_available": clean_available,
            "deployment_mode": deployment_mode,
        }

    def timed_run_once() -> tuple[int, int]:
        nonlocal calls
        calls = 0
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        ):
            outputs = invoke_sampler(collect_artifacts=False)
        del outputs
        if model.pop_visualization_artifacts() is not None:
            raise BenchmarkError(
                "timed sampler path materialized trajectory/artifact tensors"
            )
        counters = getattr(model, "_last_sampling_counters", None)
        if not isinstance(counters, Mapping):
            raise BenchmarkError("timed sampler did not expose audit counters")
        normalized = _validated_sampler_counters(
            counters,
            nfe=args.nfe,
            artifacts_collected=0,
            deployment_mode=1,
        )
        wan_calls = normalized["wan_calls"]
        teacher_calls = normalized["online_teacher_calls"]
        if calls != args.nfe:
            raise BenchmarkError(
                f"timed hook count differs: {calls} != {args.nfe}"
            )
        return wan_calls, teacher_calls

    try:
        invariance_audit = history_invariance_audit()
        audit_counters = audited_counter_run()
        for _ in range(WARMUPS):
            timed_run_once()
        torch.cuda.synchronize()
        latency_ms: list[float] = []
        observed_wan_calls: list[int] = []
        observed_teacher_calls: list[int] = []
        for _ in range(REPETITIONS):
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            wan_calls, teacher_calls = timed_run_once()
            torch.cuda.synchronize()
            finished = time.perf_counter_ns()
            latency_ms.append((finished - started) / 1_000_000.0)
            observed_wan_calls.append(wan_calls)
            observed_teacher_calls.append(teacher_calls)
    finally:
        handle.remove()

    if observed_wan_calls != [args.nfe] * REPETITIONS:
        raise BenchmarkError("Wan-call vector is not exactly NFE per repetition")
    if observed_teacher_calls != [0] * REPETITIONS:
        raise BenchmarkError("teacher-call vector contains a nonzero value")
    ordered = sorted(latency_ms)
    p50_ms = _percentile(ordered, 50.0)
    p95_ms = _percentile(ordered, 95.0)
    mean_ms = sum(latency_ms) / len(latency_ms)
    generated_frames = int(model.num_future_frames)
    samples_digest = hashlib.sha256(
        _canonical_json([round(value, 9) for value in latency_ms])
    ).hexdigest()
    arm_code = str(arm_manifest["arm"]["code"])
    payload = _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "vjepa2_controlled_study_latency",
            "created_at_utc": _now(),
            "git_commit": args.expected_commit,
            "run_dir": str(run_dir),
            "arm_code": arm_code,
            "arm_identity_sha256": arm_manifest["identity_sha256"],
            "source": args.source,
            "source_is_deployable": True,
            "oracle_is_leakage_only": False,
            "nfe": args.nfe,
            "batch_size": BATCH_SIZE,
            "warmups": WARMUPS,
            "repetitions": REPETITIONS,
            "device": {
                "name": properties.name,
                "index": torch.cuda.current_device(),
                "total_memory_bytes": properties.total_memory,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            "inputs": {
                "resolved_config": {
                    "path": str(resolved),
                    "sha256": _sha256(resolved),
                },
                "snapshot": {
                    "path": str(snapshot),
                    "sha256": stage_outcome[
                        "snapshot_observed_at_stage_end"
                    ]["sha256"],
                },
                "arm_manifest": {
                    "path": str(arm_manifest_path),
                    "sha256": _sha256(arm_manifest_path),
                },
                "study_manifest": {
                    "path": str(study_manifest_path),
                    "sha256": _sha256(study_manifest_path),
                    "identity_sha256": study_manifest["identity_sha256"],
                },
                "stage_manifest": {
                    "path": str(stage_manifest_path),
                    "sha256": _sha256(stage_manifest_path),
                    "identity_sha256": stage_manifest["identity_sha256"],
                },
                "stage_outcome": {
                    "path": str(stage_outcome_path),
                    "sha256": _sha256(stage_outcome_path),
                    "identity_sha256": stage_outcome["identity_sha256"],
                },
                "sample_index": args.sample_index,
                "sample_identity": sample_identity,
            },
            "scope": {
                "entrypoint": (
                    "DualExplicitActionDiTModel.sample_future_deployable"
                ),
                "dataset_loading_timed": False,
                "observed_history_frames": int(model.num_history_frames),
                "generated_future_frames": generated_frames,
                "future_ground_truth_rgb_available": False,
                "clean_auxiliary_target_available": False,
                "history_video_latent_and_action_preparation_timed": True,
                "wan_backbone_calls_timed": True,
                "vae_decode_timed": True,
                "internal_evidence_tensor_materialization_timed": False,
                "trajectory_capture_timed": False,
                "one_source_and_nfe_per_process": True,
                "source_mapped_to_single_autonomous_sampler_slot": True,
                "full_clip_prediction_equivalence_audit_timed": False,
                "cuda_synchronize_before_and_after": True,
                "interpretation": (
                    "history-only deployable end-to-end sampler latency including "
                    "preparation, Wan calls, and VAE decode; no future RGB or "
                    "clean V-JEPA target is accepted by the timed entrypoint"
                ),
            },
            "history_invariance_audit": invariance_audit,
            "counters": {
                "separate_untimed_audit_run": audit_counters,
                "teacher_calls_per_repetition": 0,
                "teacher_calls_total": 0,
                "wan_calls_per_repetition": args.nfe,
                "wan_calls_total": args.nfe * REPETITIONS,
                "auxiliary_clean_available": 0,
            },
            "latency_ms": {
                "p50": p50_ms,
                "p95": p95_ms,
                "mean": mean_ms,
                "min": min(latency_ms),
                "max": max(latency_ms),
                "values_sha256": samples_digest,
                "values": latency_ms,
            },
            "throughput": {
                "generated_frames_per_clip": generated_frames,
                "clips_per_second_at_p50_latency": 1000.0 / p50_ms,
                "clips_per_second_at_p95_latency": 1000.0 / p95_ms,
                "generated_frames_per_second_at_p50_latency": (
                    generated_frames * 1000.0 / p50_ms
                ),
                "generated_frames_per_second_at_p95_latency": (
                    generated_frames * 1000.0 / p95_ms
                ),
            },
            "sigma_convention": "sigma=1 noise, sigma=0 clean",
        }
    )
    output = Path(args.output)
    expected_parent = run_dir / "latency"
    if not expected_parent.exists():
        raise BenchmarkError(
            "latency directory must be created once by the stage launcher"
        )
    expected_parent = _canonical_directory(expected_parent, "latency directory")
    if output.parent.resolve(strict=True) != expected_parent:
        raise BenchmarkError(
            f"output must be directly under {expected_parent}"
        )
    expected_name = f"source_{args.source}_nfe_{args.nfe}.json"
    if output.name != expected_name:
        raise BenchmarkError(f"output basename must be {expected_name}")
    _exclusive_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "arm": arm_code,
                "source": args.source,
                "nfe": args.nfe,
                "p50_ms": payload["latency_ms"]["p50"],
                "p95_ms": payload["latency_ms"]["p95"],
                "teacher_calls": 0,
                "wan_calls_per_repetition": args.nfe,
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--arm-manifest", required=True)
    parser.add_argument("--study-manifest", required=True)
    parser.add_argument("--stage-manifest", required=True)
    parser.add_argument("--stage-outcome", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--source", choices=ALLOWED_SOURCES, required=True)
    parser.add_argument("--nfe", type=int, choices=ALLOWED_NFE, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.set_defaults(handler=command_benchmark)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        BenchmarkError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
