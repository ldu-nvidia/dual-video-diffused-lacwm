#!/usr/bin/env python3
"""Three-endpoint same-B200 benchmark for a frozen NFE-frontier selection.

The validation selection fixes ``J1@k`` and a non-dominated ``VPM@m`` with
``2 <= k < m`` before this benchmark starts.  One process keeps both final
checkpoints resident on one B200 and measures three endpoints on identical
history/actions:

* J1@k;
* VPM@k, which identifies same-NFE JEPA overhead; and
* VPM@m, which is the selected quality-matched frontier comparator.

Every six rounds use all permutations of the three endpoints, balancing every
endpoint position and both pairwise execution orders.  The acceleration gate
requires a positive stratified-bootstrap CI-low, lower J1 p95, and a favorable
mean in both J1-first and VPM-frontier-first strata.  This file is opt-in and
does not alter the original v3 J1@4/VPM@8 artifact contract.
The immutable timing input is lockbox clip zero, with the same semantic
sampling ID used by lockbox quality evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import benchmark_vjepa2_inference as single  # noqa: E402
from tools import benchmark_vjepa2_paired_latency as legacy  # noqa: E402
from tools import vjepa2_nfe_frontier as frontier  # noqa: E402


SCHEMA_VERSION = 1
KIND = frontier.KIND_LATENCY
ENDPOINT_LABELS = ("J1_k", "VPM_k", "VPM_m")
BALANCED_ORDERS = tuple(itertools.permutations(ENDPOINT_LABELS))
DEFAULT_WARMUP_ROUNDS = 18
DEFAULT_TIMED_ROUNDS = 120
FUTURE_FRAMES = 8
LOCKBOX_CLIP_INDEX = 0
LOCKBOX_SAMPLE_ID = frontier.SAMPLE_ID_OFFSETS["lockbox"]


class FrontierLatencyError(RuntimeError):
    """Raised when the selected paired timing evidence is incomparable."""


def balanced_order(index: int) -> tuple[str, str, str]:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise FrontierLatencyError("round index must be a nonnegative integer")
    return BALANCED_ORDERS[index % len(BALANCED_ORDERS)]


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise FrontierLatencyError("latency vector is empty")
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    if (
        not normalized
        or any(not math.isfinite(value) or value <= 0 for value in normalized)
    ):
        raise FrontierLatencyError("latency values must be finite and positive")
    return {
        "count": len(normalized),
        "mean": sum(normalized) / len(normalized),
        "p50": _percentile(normalized, 50.0),
        "p95": _percentile(normalized, 95.0),
        "min": min(normalized),
        "max": max(normalized),
        "values_sha256": hashlib.sha256(
            single._canonical_json([round(value, 9) for value in normalized])
        ).hexdigest(),
        "values": normalized,
    }


def paired_timing_effect(
    left_ms: Sequence[float],
    reference_ms: Sequence[float],
    orders: Sequence[Sequence[str]],
    *,
    left_label: str,
    reference_label: str,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    label: str,
) -> dict[str, Any]:
    """Positive relative improvement means ``left`` is faster."""

    left = np.asarray(left_ms, dtype=np.float64)
    reference = np.asarray(reference_ms, dtype=np.float64)
    if (
        left.ndim != 1
        or left.shape != reference.shape
        or left.size < 6
        or left.size % 6
        or len(orders) != left.size
        or not np.isfinite(left).all()
        or not np.isfinite(reference).all()
        or np.any(left <= 0)
        or np.any(reference <= 0)
        or bootstrap_samples < 100
        or not 0.5 < confidence < 1.0
    ):
        raise FrontierLatencyError(f"invalid paired timing input for {label}")
    strata: dict[str, np.ndarray] = {}
    for first, second in (
        (left_label, reference_label),
        (reference_label, left_label),
    ):
        indexes = []
        for index, order in enumerate(orders):
            normalized = tuple(order)
            if set(normalized) != set(ENDPOINT_LABELS):
                raise FrontierLatencyError("timing order is not a permutation")
            if normalized.index(first) < normalized.index(second):
                indexes.append(index)
        if len(indexes) != left.size // 2:
            raise FrontierLatencyError(
                f"{label} execution-order strata are not balanced"
            )
        strata[f"{first}_before_{second}"] = np.asarray(indexes, dtype=np.int64)
    # The six-order counterbalance is the independent resampling unit.  This
    # preserves within-block order balance and does not pretend thermally
    # adjacent rounds are independent observations.
    block_count = left.size // len(BALANCED_ORDERS)
    left_blocks = left.reshape(block_count, len(BALANCED_ORDERS)).mean(axis=1)
    reference_blocks = reference.reshape(
        block_count, len(BALANCED_ORDERS)
    ).mean(axis=1)
    rng = np.random.default_rng(frontier._derived_seed(seed, label))
    draws = rng.integers(
        0,
        block_count,
        size=(bootstrap_samples, block_count),
        endpoint=False,
    )
    left_boot = left_blocks[draws].mean(axis=1)
    reference_boot = reference_blocks[draws].mean(axis=1)
    effects = (reference_boot - left_boot) / reference_boot
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(effects, [tail, 1.0 - tail])
    difference = reference - left
    order_strata = {}
    for name, indexes in strata.items():
        stratum_difference = difference[indexes]
        order_strata[name] = {
            "count": int(indexes.size),
            "left_mean_ms": float(left[indexes].mean()),
            "reference_mean_ms": float(reference[indexes].mean()),
            "mean_favorable_difference_ms": float(stratum_difference.mean()),
            "relative_improvement": float(
                (reference[indexes].mean() - left[indexes].mean())
                / reference[indexes].mean()
            ),
            "favorable_pair_fraction": float(
                np.mean(stratum_difference > 0)
            ),
        }
    relative = (reference.mean() - left.mean()) / reference.mean()
    return {
        "left": left_label,
        "reference": reference_label,
        "n_paired_rounds": int(left.size),
        "n_counterbalance_blocks": block_count,
        "bootstrap_unit": (
            "complete six-round counterbalance block; preserves every "
            "endpoint permutation and pairwise execution-order balance"
        ),
        "mean_favorable_difference_ms": float(difference.mean()),
        "relative_improvement": float(relative),
        "relative_improvement_percent": float(relative * 100.0),
        "bootstrap_ci": {
            "confidence": confidence,
            "low": float(low),
            "high": float(high),
        },
        "favorable_pair_fraction": float(np.mean(difference > 0)),
        "order_strata": order_strata,
    }


def timing_gate(
    effect: Mapping[str, Any],
    *,
    left_p95: float,
    reference_p95: float,
) -> dict[str, Any]:
    interval = effect.get("bootstrap_ci")
    strata = effect.get("order_strata")
    if not isinstance(interval, Mapping) or not isinstance(strata, Mapping):
        raise FrontierLatencyError("timing effect lacks CI or order strata")
    ci_low = float(interval["low"])
    p95_relative_reduction = (
        reference_p95 - left_p95
    ) / reference_p95
    checks = {
        "paired_speedup_ci_low_strictly_positive": ci_low > 0.0,
        "p95_relative_reduction_at_least_20_percent": (
            p95_relative_reduction >= 0.20
        ),
        "both_execution_order_strata_favorable": (
            len(strata) == 2
            and all(
                float(value["mean_favorable_difference_ms"]) > 0.0
                for value in strata.values()
            )
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "p95_relative_reduction": p95_relative_reduction,
        "rule": (
            "paired stratified-bootstrap relative speedup CI-low > 0; "
            "J1 p95 at least 20% lower; both pairwise order strata favorable"
        ),
    }


def _record(path: Path, identity: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": single._sha256(path),
        "bytes": path.stat().st_size,
    }
    if identity is not None:
        record["identity_sha256"] = identity
    return record


def _load_selection(path: Path) -> tuple[dict[str, Any], int, int]:
    payload = single._read_json(path, "frontier selection")
    if (
        not frontier.identity_valid(payload)
        or payload.get("kind") != frontier.KIND_SELECTION
        or payload.get("confirmatory_eligible") is not True
        or payload.get("selection_split") != "validation"
    ):
        raise FrontierLatencyError(
            "latency requires a confirmatory validation selection"
        )
    pair = payload.get("selected_pair")
    k = pair.get("left", {}).get("nfe") if isinstance(pair, Mapping) else None
    m = (
        pair.get("reference", {}).get("nfe")
        if isinstance(pair, Mapping)
        else None
    )
    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or isinstance(m, bool)
        or not isinstance(m, int)
        or k < frontier.MIN_CAUSAL_J1_NFE
        or k >= m
        or k not in frontier.NFE_GRID
        or m not in frontier.NFE_GRID
        or m not in payload.get("vpm_non_dominated_nfe_frontier", [])
    ):
        raise FrontierLatencyError("selection pair is not a valid frontier pair")
    return payload, k, m


def _load_model_and_lockbox_sample(
    *,
    provenance: Mapping[str, Any],
    lockbox_cache: Mapping[str, Any],
    device: Any,
) -> tuple[Any, Any]:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(provenance["resolved_path"])
    if config.get("wandb", {}).get("enabled", False):
        config.wandb.enabled = False
    target = str(config.model.get("_target_", ""))
    if not target.endswith(".DualExplicitActionDiTModel"):
        raise FrontierLatencyError(
            f"{provenance['code']} config is not a dual explicit-action model"
        )
    dataset_config = OmegaConf.create(
        OmegaConf.to_container(config.viz_dataset, resolve=True)
    )
    dataset_config.datasets.ABC.clip_manifest = lockbox_cache[
        "clip_manifest"
    ]["path"]
    dataset_config.datasets.ABC.cache_metadata = lockbox_cache[
        "cache_metadata"
    ]["path"]
    dataset = instantiate(dataset_config)
    if len(dataset) != frontier.lockbox.LOCKBOX_CLIPS:
        raise FrontierLatencyError(
            f"{provenance['code']} lockbox dataset length is not 128"
        )
    sample = dataset[LOCKBOX_CLIP_INDEX]
    observed_clip_index = int(sample["clip_index"].item())
    if observed_clip_index != LOCKBOX_CLIP_INDEX:
        raise FrontierLatencyError(
            f"{provenance['code']} lockbox dataset substituted clip "
            f"{observed_clip_index} for {LOCKBOX_CLIP_INDEX}"
        )
    del dataset

    model = instantiate(config.model)
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
        raise FrontierLatencyError(
            f"{provenance['code']} snapshot is not the final bound checkpoint"
        )
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise FrontierLatencyError(
            f"{provenance['code']} strict checkpoint load failed: {incompatible}"
        )
    del snapshot
    model = model.to(device=device).eval()
    single._assert_teacher_absent(model, config)
    if getattr(model, "time_frequency_transform", None) is not None:
        raise FrontierLatencyError(
            f"{provenance['code']} unexpectedly registers an online transform"
        )
    return model, sample


def command_benchmark(args: argparse.Namespace) -> int:
    import torch

    if (
        args.warmup_rounds != DEFAULT_WARMUP_ROUNDS
        or args.timed_rounds != DEFAULT_TIMED_ROUNDS
        or args.bootstrap_samples != frontier.DEFAULT_BOOTSTRAP_SAMPLES
        or args.confidence != frontier.DEFAULT_CONFIDENCE
        or args.seed != frontier.DEFAULT_SEED
    ):
        raise FrontierLatencyError(
            "confirmatory timing requires 18 warmup rounds, 120 timed rounds, "
            "10,000 bootstrap samples, 95% confidence, and seed 1234"
        )

    benchmark_commit = args.benchmark_commit or args.expected_commit
    repo = Path(args.repo_root).expanduser().resolve(strict=True)
    executed_roots = {
        Path(module.__file__).resolve().parents[1]
        for module in (single, legacy, frontier, frontier.lockbox)
    }
    executed_roots.add(Path(__file__).resolve().parents[1])
    if executed_roots != {repo}:
        raise FrontierLatencyError(
            "benchmark and imported helpers do not belong to --repo-root"
        )
    try:
        single._assert_clean_commit(repo, benchmark_commit)
    except single.BenchmarkError as exc:
        raise FrontierLatencyError(str(exc)) from exc
    try:
        benchmark_compatibility = frontier.git_inference_compatibility(
            repo,
            training_commit=args.expected_commit,
            tool_commit=benchmark_commit,
        )
    except frontier.FrontierError as exc:
        raise FrontierLatencyError(str(exc)) from exc
    project_root = repo / "projects" / "latent_action_models"
    for root in (str(repo), str(project_root)):
        if root not in sys.path:
            sys.path.insert(0, root)

    study_root = Path(args.study_root).expanduser().resolve(strict=True)
    study_path = (study_root / "study_manifest.json").resolve(strict=True)
    study = single._read_json(study_path, "study manifest")
    if (
        not single._identity_is_valid(study)
        or study.get("kind") != "vjepa2_controlled_video_diffusion_study"
        or study.get("study_root") != str(study_root)
        or study.get("inputs", {}).get("repository", {}).get("git_commit")
        != args.expected_commit
    ):
        raise FrontierLatencyError("study manifest provenance differs")
    videox_path = Path(
        str(study.get("inputs", {}).get("runtime", {}).get("videox_home", ""))
    ).expanduser()
    try:
        videox_runtime = frontier.git_runtime_provenance(videox_path)
    except frontier.FrontierError as exc:
        raise FrontierLatencyError(str(exc)) from exc
    selection_path = Path(args.selection).expanduser().resolve(strict=True)
    selection, k, m = _load_selection(selection_path)
    if (
        selection.get("study_identity_sha256") != study.get("identity_sha256")
        or selection.get("training_git_commit") != args.expected_commit
        or selection.get("videox_runtime_identity_sha256")
        != hashlib.sha256(single._canonical_json(videox_runtime)).hexdigest()
    ):
        raise FrontierLatencyError("selection belongs to a different study")
    evaluator_commit = selection.get("evaluator_git_commit")
    try:
        evaluator_compatibility = frontier.git_inference_compatibility(
            repo,
            training_commit=args.expected_commit,
            tool_commit=str(evaluator_commit),
        )
    except frontier.FrontierError as exc:
        raise FrontierLatencyError(str(exc)) from exc
    lockbox_registration = selection.get("lockbox_registration")
    if not isinstance(lockbox_registration, Mapping):
        raise FrontierLatencyError("selection does not bind a fresh lockbox")
    try:
        lockbox_validation = frontier.lockbox.validate_registration(
            lockbox_registration,
            study=study,
            rehash_arrays=True,
            verify_construction=True,
        )
    except frontier.lockbox.LockboxError as exc:
        raise FrontierLatencyError(str(exc)) from exc
    if (
        lockbox_registration.get("registration_git_commit")
        != evaluator_commit
        or lockbox_registration.get("inference_code_compatibility")
        != evaluator_compatibility
    ):
        raise FrontierLatencyError(
            "selection lockbox/evaluator code provenance differs"
        )
    lockbox_cache = {
        "clip_manifest": lockbox_registration["manifest"],
        "cache_metadata": lockbox_registration["cache"]["metadata"],
        "arrays": lockbox_registration["cache"]["arrays"],
        "registration_identity_sha256": lockbox_registration[
            "identity_sha256"
        ],
        "validation": lockbox_validation,
    }
    with Path(lockbox_cache["clip_manifest"]["path"]).open(
        encoding="utf-8"
    ) as handle:
        first_descriptor = json.loads(next(line for line in handle if line.strip()))
    if (
        first_descriptor.get("auxiliary_index") != LOCKBOX_CLIP_INDEX
        or not isinstance(first_descriptor.get("clip_id"), str)
        or not first_descriptor["clip_id"]
    ):
        raise FrontierLatencyError("lockbox clip-zero descriptor is invalid")
    lockbox_cache["selected_clip"] = {
        "clip_index": LOCKBOX_CLIP_INDEX,
        "clip_id": first_descriptor["clip_id"],
        "manifest_sha256": lockbox_cache["clip_manifest"]["sha256"],
    }

    device = torch.device(args.device)
    if device.type != "cuda":
        raise FrontierLatencyError("frontier timing requires CUDA")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if "B200" not in properties.name.upper():
        raise FrontierLatencyError(
            f"frontier timing requires NVIDIA B200, found {properties.name}"
        )
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    slurm_node = os.environ.get("SLURMD_NODENAME", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (
        not slurm_job_id
        or not slurm_node
        or not visible
        or os.environ.get("SLURM_JOB_NUM_NODES", "1") != "1"
        or torch.cuda.device_count() != 1
    ):
        raise FrontierLatencyError(
            "frontier timing requires one Slurm node and one visible GPU"
        )

    provenance = {
        arm: legacy._arm_provenance(
            study_root=study_root, study=study, arm_code=arm
        )
        for arm in ("J1", "VPM")
    }
    provenance["J1"]["nfe"] = k
    provenance["VPM"]["nfe"] = k
    for arm, value in provenance.items():
        if (
            selection.get("arm_identity_sha256", {}).get(arm)
            != value["arm"]["identity_sha256"]
            or selection.get("stage_identity_sha256", {}).get(arm)
            != value["stage"]["identity_sha256"]
        ):
            raise FrontierLatencyError(
                f"selection does not bind the loaded {arm} final checkpoint"
            )
    models: dict[str, Any] = {}
    host_samples: dict[str, Any] = {}
    for arm in ("VPM", "J1"):
        model, sample = _load_model_and_lockbox_sample(
            provenance=provenance[arm],
            lockbox_cache=lockbox_cache,
            device=device,
        )
        models[arm] = model
        host_samples[arm] = legacy._host_sample(sample, arm=arm)
        del sample
    history_frames = int(models["J1"].num_history_frames)
    future_frames = int(models["J1"].num_future_frames)
    if (
        history_frames != 5
        or future_frames != FUTURE_FRAMES
        or int(models["VPM"].num_history_frames) != history_frames
        or int(models["VPM"].num_future_frames) != future_frames
    ):
        raise FrontierLatencyError("models disagree on 5-history/8-future")
    identities = {
        arm: legacy._sample_identity(host_samples[arm], history_frames)
        for arm in ("J1", "VPM")
    }
    if identities["J1"] != identities["VPM"]:
        raise FrontierLatencyError("J1/VPM immutable inputs differ")

    shared = host_samples["J1"]
    history_rgb = shared["rgb"][:history_frames].unsqueeze(0).to(device)
    actions = shared["actions"].unsqueeze(0).to(device)
    morphology = shared["morphology_index"].unsqueeze(0).to(device)
    sample_ids = torch.tensor(
        [LOCKBOX_SAMPLE_ID], dtype=torch.long, device=device
    )
    del host_samples, shared

    endpoints = {
        "J1_k": {"arm": "J1", "nfe": k},
        "VPM_k": {"arm": "VPM", "nfe": k},
        "VPM_m": {"arm": "VPM", "nfe": m},
    }
    hook_calls = {"J1": 0, "VPM": 0}
    hooks = []
    for arm in ("J1", "VPM"):
        def count_calls(_module, _inputs, _output, *, arm_code=arm):
            hook_calls[arm_code] += 1

        hooks.append(models[arm].forward_model.register_forward_hook(count_calls))

    def configure(endpoint: str) -> tuple[Any, str, int]:
        spec = endpoints[endpoint]
        arm, nfe = str(spec["arm"]), int(spec["nfe"])
        model = models[arm]
        model.evaluation_condition_sources = ("autonomous",)
        model.evaluation_nfe_steps = (nfe,)
        model.viz_num_steps = nfe
        model.capture_latent_trajectories = False
        hook_calls[arm] = 0
        return model, arm, nfe

    def invoke(endpoint: str, *, collect_artifacts: bool) -> tuple[str, int]:
        model, arm, nfe = configure(endpoint)
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
        return arm, nfe

    def validate_call(
        endpoint: str,
        *,
        artifacts_collected: int,
        hook_required: bool,
    ) -> dict[str, int]:
        arm, nfe = endpoints[endpoint]["arm"], int(endpoints[endpoint]["nfe"])
        if hook_required and hook_calls[str(arm)] != nfe:
            raise FrontierLatencyError(
                f"{endpoint} hook calls {hook_calls[str(arm)]} != {nfe}"
            )
        counters = getattr(models[str(arm)], "_last_sampling_counters", None)
        try:
            return single._validated_sampler_counters(
                counters,
                nfe=nfe,
                artifacts_collected=artifacts_collected,
                deployment_mode=1,
            )
        except single.BenchmarkError as exc:
            raise FrontierLatencyError(str(exc)) from exc

    audits: dict[str, Any] = {}
    audit_peak: dict[str, int] = {}
    deployable_peak: dict[str, int] = {}
    try:
        for endpoint, spec in endpoints.items():
            torch.cuda.reset_peak_memory_stats(device)
            arm, nfe = invoke(endpoint, collect_artifacts=True)
            counters = validate_call(
                endpoint, artifacts_collected=1, hook_required=True
            )
            artifacts = models[arm].pop_visualization_artifacts()
            if not isinstance(artifacts, Mapping):
                raise FrontierLatencyError(f"{endpoint} audit lacks artifacts")
            observed_calls = int(
                artifacts[f"wan_call_count_nfe_{nfe}"].reshape(-1)[0]
            )
            forbidden = {
                "video_clean",
                "tf_clean",
                "ground_truth_future_uint8",
            }.intersection(artifacts)
            if (
                observed_calls != nfe
                or int(artifacts["online_teacher_call_count"].reshape(-1)[0])
                != 0
                or int(artifacts["auxiliary_clean_available"].reshape(-1)[0])
                != 0
                or int(artifacts["deployment_mode"].reshape(-1)[0]) != 1
                or forbidden
            ):
                raise FrontierLatencyError(
                    f"{endpoint} violates deployable audit; "
                    f"forbidden={sorted(forbidden)}"
                )
            audit_peak[endpoint] = int(torch.cuda.max_memory_allocated(device))
            audits[endpoint] = {
                "arm": arm,
                "nfe": nfe,
                "actual_wan_calls": observed_calls,
                "online_teacher_calls": 0,
                "clean_auxiliary_available": 0,
                "future_ground_truth_available": False,
                "trajectory_capture_enabled": False,
                "independent_forward_hook_wan_count": hook_calls[arm],
                "sampler_counters": counters,
            }
            del artifacts
        for hook in hooks:
            hook.remove()
        hooks.clear()

        def untimed_once(endpoint: str) -> None:
            arm, _nfe = invoke(endpoint, collect_artifacts=False)
            validate_call(endpoint, artifacts_collected=0, hook_required=False)
            if models[arm].pop_visualization_artifacts() is not None:
                raise FrontierLatencyError(
                    f"{endpoint} warmup materialized artifacts"
                )

        def timed_once(endpoint: str) -> float:
            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            arm, _nfe = invoke(endpoint, collect_artifacts=False)
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            validate_call(endpoint, artifacts_collected=0, hook_required=False)
            if models[arm].pop_visualization_artifacts() is not None:
                raise FrontierLatencyError(
                    f"{endpoint} timing materialized artifacts"
                )
            return elapsed

        for endpoint in ENDPOINT_LABELS:
            torch.cuda.reset_peak_memory_stats(device)
            untimed_once(endpoint)
            deployable_peak[endpoint] = int(
                torch.cuda.max_memory_allocated(device)
            )
        for round_index in range(args.warmup_rounds):
            for endpoint in balanced_order(round_index):
                untimed_once(endpoint)

        resident_bytes = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        values = {endpoint: [] for endpoint in ENDPOINT_LABELS}
        rounds = []
        orders = []
        for round_index in range(args.timed_rounds):
            order = balanced_order(round_index)
            measured = {}
            for endpoint in order:
                measured[endpoint] = timed_once(endpoint)
                values[endpoint].append(measured[endpoint])
            orders.append(list(order))
            rounds.append(
                {
                    "round_index": round_index,
                    "execution_order": list(order),
                    "latency_ms": measured,
                }
            )
        timed_peak = int(torch.cuda.max_memory_allocated(device))
    finally:
        for hook in hooks:
            hook.remove()

    summaries = {
        endpoint: latency_summary(values[endpoint])
        for endpoint in ENDPOINT_LABELS
    }
    frontier_effect = paired_timing_effect(
        values["J1_k"],
        values["VPM_m"],
        orders,
        left_label="J1_k",
        reference_label="VPM_m",
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
        label=f"frontier-latency:J1-{k}-vs-VPM-{m}",
    )
    same_effect = paired_timing_effect(
        values["J1_k"],
        values["VPM_k"],
        orders,
        left_label="J1_k",
        reference_label="VPM_k",
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
        label=f"same-nfe-latency:J1-{k}-vs-VPM-{k}",
    )
    same_ci = same_effect["bootstrap_ci"]
    same_overhead = {
        "comparison": f"J1@{k} vs VPM@{k}",
        "definition": "positive relative_overhead means J1 is slower",
        "relative_overhead": -float(same_effect["relative_improvement"]),
        "relative_overhead_percent": (
            -100.0 * float(same_effect["relative_improvement"])
        ),
        "bootstrap_ci": {
            "confidence": same_ci["confidence"],
            "low": -float(same_ci["high"]),
            "high": -float(same_ci["low"]),
        },
        "paired_speed_effect": same_effect,
    }
    acceleration = {
        "comparison": f"J1@{k} vs VPM@{m}",
        "paired_speed_effect": frontier_effect,
        "timing_gate": timing_gate(
            frontier_effect,
            left_p95=float(summaries["J1_k"]["p95"]),
            reference_p95=float(summaries["VPM_m"]["p95"]),
        ),
    }
    endpoint_payloads = {}
    for endpoint, spec in endpoints.items():
        summary = summaries[endpoint]
        endpoint_payloads[endpoint] = {
            **spec,
            "source": "autonomous",
            "actual_wan_calls": audits[endpoint]["actual_wan_calls"],
            "latency_ms": summary,
            "generated_frames_per_second_at_p95": (
                FUTURE_FRAMES * 1000.0 / float(summary["p95"])
            ),
            "peak_allocated_bytes_with_both_models_resident": deployable_peak[
                endpoint
            ],
            "audit_peak_allocated_bytes_with_artifact_capture": audit_peak[
                endpoint
            ],
            "audit": audits[endpoint],
        }

    payload = single._identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "created_at_utc": single._now(),
            "training_git_commit": args.expected_commit,
            "evaluator_git_commit": selection.get("evaluator_git_commit"),
            "benchmark_git_commit": benchmark_commit,
            "inference_code_compatibility": {
                "evaluator": evaluator_compatibility,
                "benchmark": benchmark_compatibility,
            },
            "videox_runtime": videox_runtime,
            "lockbox_registration_identity_sha256": lockbox_registration[
                "identity_sha256"
            ],
            "selection": _record(
                selection_path, identity=selection["identity_sha256"]
            ),
            "study": _record(study_path, identity=study["identity_sha256"]),
            "slurm": {
                "job_id": slurm_job_id,
                "same_allocation": True,
                "same_node": slurm_node,
                "one_visible_gpu": visible,
            },
            "device": {
                "name": properties.name,
                "index": torch.cuda.current_device(),
                "total_memory_bytes": properties.total_memory,
                "resident_allocated_bytes_before_timing": resident_bytes,
                "peak_allocated_bytes_during_timing": timed_peak,
                "both_models_resident": True,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            "protocol": {
                "confirmatory_protocol": True,
                "lockbox_clip_index": LOCKBOX_CLIP_INDEX,
                "sampling_id": LOCKBOX_SAMPLE_ID,
                "sampling_namespace": "lockbox",
                "warmup_rounds": args.warmup_rounds,
                "timed_rounds": args.timed_rounds,
                "bootstrap_samples": args.bootstrap_samples,
                "confidence": args.confidence,
                "bootstrap_seed": args.seed,
                "balanced_order_cycle": [list(order) for order in BALANCED_ORDERS],
                "same_process": True,
                "same_B200": True,
                "both_models_resident": True,
                "identical_immutable_inputs": True,
                "public_deployable_entrypoint": (
                    "DualExplicitActionDiTModel.sample_future_deployable"
                ),
                "future_ground_truth_available": False,
                "clean_auxiliary_available": False,
                "online_teacher_calls": 0,
                "timed_artifact_materialization": False,
                "cuda_synchronize_before_and_after_each_endpoint": True,
                "timing_scope": (
                    "history preparation inside model + Wan calls + VAE decode"
                ),
            },
            "immutable_input": {
                **identities["J1"],
                "lockbox_cache": lockbox_cache,
            },
            "model_provenance": {
                arm: {
                    "snapshot": _record(value["snapshot_path"]),
                    "arm_manifest": _record(
                        value["arm_path"], value["arm"]["identity_sha256"]
                    ),
                    "stage_manifest": _record(
                        value["stage_path"], value["stage"]["identity_sha256"]
                    ),
                }
                for arm, value in provenance.items()
            },
            "endpoints": endpoint_payloads,
            "same_nfe_overhead": same_overhead,
            "frontier_acceleration": acceleration,
            "rounds_sha256": hashlib.sha256(
                single._canonical_json(rounds)
            ).hexdigest(),
            "rounds": rounds,
        }
    )
    output = Path(args.output).expanduser()
    try:
        output_parent = output.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FrontierLatencyError(
            "output parent must already exist on durable storage"
        ) from exc
    try:
        output_parent.relative_to(study_root)
    except ValueError as exc:
        raise FrontierLatencyError("output must be beneath study root") from exc
    try:
        single._exclusive_json(output, payload)
    except single.BenchmarkError as exc:
        raise FrontierLatencyError(str(exc)) from exc
    print(
        json.dumps(
            {
                "status": (
                    "passed"
                    if acceleration["timing_gate"]["passed"]
                    else "not_demonstrated"
                ),
                "output": str(output),
                "comparison": acceleration["comparison"],
                "speedup_percent": frontier_effect[
                    "relative_improvement_percent"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--expected-commit",
        required=True,
        help="immutable training/checkpoint commit recorded by the study",
    )
    parser.add_argument(
        "--benchmark-commit",
        default=None,
        help=(
            "clean benchmark checkout commit; defaults to --expected-commit"
        ),
    )
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-rounds", type=int, default=DEFAULT_WARMUP_ROUNDS)
    parser.add_argument("--timed-rounds", type=int, default=DEFAULT_TIMED_ROUNDS)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=frontier.DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument("--confidence", type=float, default=frontier.DEFAULT_CONFIDENCE)
    parser.add_argument("--seed", type=int, default=frontier.DEFAULT_SEED)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return command_benchmark(args)
    except (
        FrontierLatencyError,
        frontier.FrontierError,
        legacy.PairedLatencyError,
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
