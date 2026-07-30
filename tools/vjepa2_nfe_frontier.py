#!/usr/bin/env python3
"""Validation-selected NFE frontier analysis for V-JEPA video diffusion.

This tool is an opt-in repair for the original fixed J1@4 versus VPM@8 speed
comparison.  It does not change or reinterpret the immutable v3 study:

* ``select`` constructs a robust VPM quality/NFE frontier using validation
  clips only, then freezes one J1@k versus VPM@m candidate with ``2 <= k < m``;
* ``confirm`` evaluates that one frozen pair on a fresh lockbox without
  reselecting it; and
* ``finalize`` combines the held-out quality result with a same-B200 frontier
  latency artifact.

Rows are the per-clip JSONL records emitted by
``tools/evaluate_vjepa2_quality.py``.  New confirmatory artifacts must carry an
explicit ``evaluation_split`` and lockbox rows additionally bind the frozen
selection and registration identities. Legacy rows can only be inspected with
``--allow-posthoc`` and are always labeled exploratory.

All three claim metrics are lower-is-better.  A lower-NFE VPM point robustly
dominates a higher-NFE point only when the paired 95% bootstrap CI is strictly
positive for temporal MSE and above -1% for both video NMSE and decoded MSE.
The same thresholds screen validation candidates and gate held-out quality.
NFE=1 is never a J1 causal candidate because its only Wan call sees pure
auxiliary noise.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools import vjepa2_frontier_lockbox as lockbox
except ModuleNotFoundError:  # Direct ``python tools/...`` invocation.
    import vjepa2_frontier_lockbox as lockbox


SCHEMA_VERSION = 1
KIND_SELECTION = "vjepa2_nfe_frontier_selection"
KIND_CONFIRMATION = "vjepa2_nfe_frontier_lockbox_confirmation"
KIND_FINAL = "vjepa2_nfe_frontier_final_report"
KIND_LATENCY = "vjepa2_nfe_frontier_paired_latency"
NFE_GRID = (1, 2, 4, 6, 8, 12, 20)
SAMPLE_ID_OFFSETS = {
    "validation": 1_000_000,
    "test": 2_000_000,
    "lockbox": 3_000_000,
}
MIN_CAUSAL_J1_NFE = 2
PRIMARY_METRIC = "decoded_temporal_difference_mse_unit_range"
GUARDRAIL_METRICS = ("video_future_nmse", "decoded_mse_unit_range")
CLAIM_METRICS = (PRIMARY_METRIC, *GUARDRAIL_METRICS)
METRIC_DIRECTIONS = {metric: "lower" for metric in CLAIM_METRICS}
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 1234
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
INFERENCE_CRITICAL_PATHS = (
    "projects/latent_action_models/lam",
    "robot_wm/modeling",
    "robot_wm/datasets",
    "tools/env/videox_shim",
)
EXPECTED_VIDEOX_COMMIT = "1d6d9c3e1540968466937129fef4b288041e06de"
TIMING_ENDPOINTS = ("J1_k", "VPM_k", "VPM_m")
BALANCED_TIMING_ORDERS = tuple(itertools.permutations(TIMING_ENDPOINTS))


class FrontierError(RuntimeError):
    """Raised when frontier evidence is incomplete or scientifically unsafe."""


def git_inference_compatibility(
    repo: Path,
    *,
    training_commit: str,
    tool_commit: str,
) -> dict[str, Any]:
    """Prove a descendant tool commit preserves checkpoint inference semantics."""

    if (
        COMMIT_RE.fullmatch(training_commit) is None
        or COMMIT_RE.fullmatch(tool_commit) is None
    ):
        raise FrontierError("training/tool commits must be full lowercase SHA-1s")

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise FrontierError(
                f"git {' '.join(arguments)} failed: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return completed.stdout.strip()

    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            training_commit,
            tool_commit,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    ancestor = completed.returncode == 0
    if not ancestor:
        raise FrontierError(
            "tool commit is not a descendant of the immutable training commit"
        )

    path_records: dict[str, Any] = {}
    for path in INFERENCE_CRITICAL_PATHS:
        training_tree = git("rev-parse", f"{training_commit}:{path}")
        tool_tree = git("rev-parse", f"{tool_commit}:{path}")
        unchanged = training_tree == tool_tree
        if not unchanged:
            raise FrontierError(
                "inference-critical code changed between training and tool "
                f"commits: {path}"
            )
        path_records[path] = {
            "training_object_sha": training_tree,
            "tool_object_sha": tool_tree,
            "unchanged": True,
        }
    return {
        "training_commit_is_ancestor": True,
        "inference_critical_paths_unchanged": True,
        "paths": path_records,
    }


def git_runtime_provenance(
    checkout: Path,
    *,
    expected_commit: str = EXPECTED_VIDEOX_COMMIT,
) -> dict[str, Any]:
    """Bind an external inference checkout by clean commit and root tree."""

    if not checkout.is_dir() or checkout.is_symlink():
        raise FrontierError(
            f"runtime checkout must be a non-symlink directory: {checkout}"
        )
    checkout = checkout.resolve(strict=True)

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise FrontierError(
                f"runtime git {' '.join(arguments)} failed: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if commit != expected_commit:
        raise FrontierError(
            f"VideoX-Fun commit differs: {commit} != {expected_commit}"
        )
    if git("status", "--porcelain", "--untracked-files=all"):
        raise FrontierError("VideoX-Fun checkout must be clean")
    return {
        "path": str(checkout),
        "git_commit": commit,
        "git_tree_sha": git("rev-parse", "HEAD^{tree}"),
        "clean": True,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: Mapping[str, Any]) -> bool:
    recorded = payload.get("identity_sha256")
    if not isinstance(recorded, str) or SHA256_RE.fullmatch(recorded) is None:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest() == recorded


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise FrontierError(f"{label} must contain one JSON object: {path}")
    return payload


def _read_jsonl(paths: Sequence[Path], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise FrontierError(f"{label} is not a regular file: {path}")
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        raise FrontierError(
                            f"{label} has blank row {line_number}: {path}"
                        )
                    row = json.loads(
                        line,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f"non-finite constant {value}")
                        ),
                    )
                    if not isinstance(row, dict):
                        raise FrontierError(
                            f"{label} row {line_number} is not an object"
                        )
                    rows.append(row)
        except FrontierError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise FrontierError(f"{label} is invalid JSONL: {path}") from exc
    if not rows:
        raise FrontierError(f"{label} contains no rows")
    return rows


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = _canonical_json(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FrontierError(f"refusing to overwrite output: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def paired_effect(
    left: Sequence[float],
    reference: Sequence[float],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    label: str,
) -> dict[str, Any]:
    """Return paired relative improvement; positive always favors ``left``."""

    left_array = np.asarray(left, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if (
        left_array.ndim != 1
        or left_array.shape != reference_array.shape
        or left_array.size < 2
        or not np.isfinite(left_array).all()
        or not np.isfinite(reference_array).all()
        or np.any(reference_array <= 0)
        or bootstrap_samples < 100
        or not 0.5 < confidence < 1.0
    ):
        raise FrontierError(f"invalid paired values/protocol for {label}")
    relative = (reference_array.mean() - left_array.mean()) / abs(
        reference_array.mean()
    )
    rng = np.random.default_rng(_derived_seed(seed, label))
    indexes = rng.integers(
        0,
        left_array.size,
        size=(bootstrap_samples, left_array.size),
        endpoint=False,
    )
    left_means = left_array[indexes].mean(axis=1)
    reference_means = reference_array[indexes].mean(axis=1)
    if np.any(reference_means <= 0):
        raise FrontierError(f"non-positive bootstrap reference for {label}")
    effects = (reference_means - left_means) / np.abs(reference_means)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(effects, [tail, 1.0 - tail])
    favorable = reference_array - left_array
    return {
        "n_paired_clips": int(left_array.size),
        "left_mean": float(left_array.mean()),
        "reference_mean": float(reference_array.mean()),
        "mean_favorable_delta": float(favorable.mean()),
        "relative_improvement": float(relative),
        "relative_improvement_percent": float(relative * 100.0),
        "bootstrap_ci": {
            "confidence": confidence,
            "low": float(low),
            "high": float(high),
        },
        "favorable_clip_fraction": float(np.mean(favorable > 0)),
        "bootstrap_unit": "paired immutable clip_id/clip_index",
    }


def quality_gate(comparison: Mapping[str, Any]) -> dict[str, Any]:
    metric_effects = comparison.get("metrics")
    if not isinstance(metric_effects, Mapping):
        raise FrontierError("comparison lacks metric effects")
    lows: dict[str, float] = {}
    for metric in CLAIM_METRICS:
        effect = metric_effects.get(metric)
        interval = effect.get("bootstrap_ci") if isinstance(effect, Mapping) else None
        low = interval.get("low") if isinstance(interval, Mapping) else None
        if (
            isinstance(low, bool)
            or not isinstance(low, (int, float))
            or not math.isfinite(float(low))
        ):
            raise FrontierError(f"comparison lacks finite CI-low for {metric}")
        lows[metric] = float(low)
    checks = {
        "temporal_ci_low_strictly_positive": lows[PRIMARY_METRIC] > 0.0,
        "video_nmse_ci_low_above_minus_one_percent": (
            lows["video_future_nmse"] > -0.01
        ),
        "decoded_mse_ci_low_above_minus_one_percent": (
            lows["decoded_mse_unit_range"] > -0.01
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "ci_lows": lows,
        "rule": (
            "temporal paired relative-improvement CI-low > 0; video-NMSE and "
            "decoded-MSE CI-lows > -0.01"
        ),
    }


def same_nfe_attribution_gate(
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the preregistered T1 temporal threshold for JEPA attribution."""

    base = quality_gate(comparison)
    checks = dict(base["checks"])
    checks["temporal_ci_low_at_least_three_percent"] = (
        base["ci_lows"][PRIMARY_METRIC] >= 0.03
    )
    return {
        **base,
        "passed": all(checks.values()),
        "checks": checks,
        "rule": (
            "same-NFE attribution requires temporal paired relative-improvement "
            "CI-low >= 0.03 and video-NMSE/decoded-MSE CI-lows > -0.01"
        ),
    }


def _row_unit(row: Mapping[str, Any]) -> tuple[str, int]:
    clip_id = row.get("clip_id")
    clip_index = row.get("clip_index")
    if (
        not isinstance(clip_id, str)
        or not clip_id
        or isinstance(clip_index, bool)
        or not isinstance(clip_index, int)
        or clip_index < 0
    ):
        raise FrontierError("quality row has invalid clip identity")
    return clip_id, clip_index


def validate_arm_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    split: str,
    required_nfes: Sequence[int],
    expected_clips: int,
    selection_identity: str | None,
    lockbox_identity: str | None = None,
    allow_posthoc: bool,
) -> tuple[dict[tuple[str, int], dict[int, Mapping[str, Any]]], bool]:
    """Validate autonomous rows and return per-clip/NFE lookup.

    The boolean is true only when every row explicitly binds the requested
    split and (for test) the frozen selection identity.
    """

    if arm not in {"J1", "VPM"} or split not in {
        "validation",
        "test",
        "lockbox",
    }:
        raise FrontierError("invalid arm or split")
    required = tuple(sorted(set(int(value) for value in required_nfes)))
    if not required or any(value not in NFE_GRID for value in required):
        raise FrontierError("required NFE set is empty or outside the fixed grid")
    observed: dict[tuple[str, int], dict[int, Mapping[str, Any]]] = {}
    explicit_binding = True
    for row in rows:
        if not identity_valid(row):
            raise FrontierError(f"{arm} quality row identity is invalid")
        # Final study JSONLs also contain off, shuffled, and oracle controls.
        # They are valid evidence for other questions but cannot enter this
        # deployable autonomous frontier.
        if row.get("source") != "autonomous":
            continue
        if (
            row.get("arm_code") != arm
            or row.get("completed_updates") != 1000
            or row.get("oracle_leakage") is not False
            or row.get("deployable_evidence") is not True
            or row.get("sampler_entrypoint")
            != "DualExplicitActionDiTModel.sample_future_deployable"
            or row.get("clean_future_or_auxiliary_passed_to_sampler") is not False
            or row.get("online_teacher_call_count") != 0
            or row.get("auxiliary_history_latent_frames") != 0
        ):
            raise FrontierError(f"{arm} row violates autonomous provenance")
        nfe = row.get("nfe")
        if isinstance(nfe, bool) or not isinstance(nfe, int) or nfe not in NFE_GRID:
            raise FrontierError(f"{arm} row has invalid NFE")
        if row.get("actual_wan_call_count") != nfe:
            raise FrontierError(f"{arm} row Wan-call count differs from NFE")
        row_split = row.get("evaluation_split")
        row_selection = row.get("frontier_selection_identity_sha256")
        if row_split != split:
            explicit_binding = False
        if split in {"test", "lockbox"} and row_selection != selection_identity:
            explicit_binding = False
        if (
            split == "lockbox"
            and row.get("lockbox_registration_identity_sha256")
            != lockbox_identity
        ):
            explicit_binding = False
        for identity_field in (
            "study_identity_sha256",
            "arm_identity_sha256",
            "stage_identity_sha256",
        ):
            identity_value = row.get(identity_field)
            if (
                not isinstance(identity_value, str)
                or SHA256_RE.fullmatch(identity_value) is None
            ):
                explicit_binding = False
        for commit_field in ("training_git_commit", "evaluator_git_commit"):
            commit_value = row.get(commit_field)
            if (
                not isinstance(commit_value, str)
                or COMMIT_RE.fullmatch(commit_value) is None
            ):
                explicit_binding = False
        compatibility_sha = row.get("inference_code_compatibility_sha256")
        if (
            not isinstance(compatibility_sha, str)
            or SHA256_RE.fullmatch(compatibility_sha) is None
        ):
            explicit_binding = False
        videox_sha = row.get("videox_runtime_identity_sha256")
        if (
            not isinstance(videox_sha, str)
            or SHA256_RE.fullmatch(videox_sha) is None
        ):
            explicit_binding = False
        if (
            row.get("evaluation_world_size") != 8
            or row.get("evaluation_batch_size_per_rank") != 2
            or row.get("sampling_namespace") != split
            or row.get("sampling_id")
            != SAMPLE_ID_OFFSETS[split] + _row_unit(row)[1]
        ):
            explicit_binding = False
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise FrontierError(f"{arm} row lacks metrics")
        for metric in CLAIM_METRICS:
            value = metrics.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise FrontierError(f"{arm} row has invalid {metric}")
        unit = _row_unit(row)
        bucket = observed.setdefault(unit, {})
        if nfe in bucket:
            raise FrontierError(f"duplicate {arm} row for {unit}, NFE={nfe}")
        bucket[nfe] = row
    if not explicit_binding and not allow_posthoc:
        raise FrontierError(
            f"{arm} {split} rows lack explicit split/selection binding; "
            "rerun the frontier evaluator or use --allow-posthoc"
        )
    if len(observed) != expected_clips:
        raise FrontierError(
            f"{arm} {split} clip count {len(observed)} != {expected_clips}"
        )
    required_set = set(required)
    for unit, by_nfe in observed.items():
        missing = required_set - by_nfe.keys()
        if missing:
            raise FrontierError(f"{arm} {unit} lacks NFEs {sorted(missing)}")
        if split == "lockbox" and set(by_nfe) != required_set:
            raise FrontierError(
                f"{arm} lockbox {unit} contains unfrozen extra NFEs"
            )
    return observed, explicit_binding


def _assert_paired(
    left: Mapping[tuple[str, int], Mapping[int, Mapping[str, Any]]],
    reference: Mapping[tuple[str, int], Mapping[int, Mapping[str, Any]]],
    *,
    left_nfe: int,
    reference_nfe: int,
) -> tuple[tuple[str, int], ...]:
    units = tuple(sorted(left))
    if units != tuple(sorted(reference)):
        raise FrontierError("J1 and VPM clip identities are not paired")
    identity_fields = (
        "video_clean_sha256",
        "auxiliary_clean_sha256",
        "ground_truth_sha256",
        "vae_ground_truth_sha256",
        "raw_history_last_sha256",
        "vae_history_last_sha256",
        "cached_rgb_input_sha256",
        "cached_actions_input_sha256",
        "video_initial_state_sha256",
        "auxiliary_initial_state_sha256",
        "auxiliary_initial_noise_sha256",
    )
    for unit in units:
        left_row = left[unit][left_nfe]
        reference_row = reference[unit][reference_nfe]
        left_hashes = left_row.get("tensor_sha256")
        reference_hashes = reference_row.get("tensor_sha256")
        if not isinstance(left_hashes, Mapping) or not isinstance(
            reference_hashes, Mapping
        ):
            raise FrontierError(f"paired rows for {unit} lack tensor hashes")
        for field in identity_fields:
            left_value = left_hashes.get(field)
            reference_value = reference_hashes.get(field)
            if (
                not isinstance(left_value, str)
                or SHA256_RE.fullmatch(left_value) is None
                or left_value != reference_value
            ):
                raise FrontierError(
                    f"paired immutable input differs for {unit}: {field}"
                )
    return units


def _provenance_identities(
    rows: Sequence[Mapping[str, Any]], arm: str
) -> dict[str, set[str]]:
    autonomous = [
        row
        for row in rows
        if row.get("source") == "autonomous" and row.get("arm_code") == arm
    ]
    return {
        field: {
            str(row.get(field))
            for row in autonomous
            if isinstance(row.get(field), str)
        }
        for field in (
            "study_identity_sha256",
            "arm_identity_sha256",
            "stage_identity_sha256",
            "training_git_commit",
            "evaluator_git_commit",
            "inference_code_compatibility_sha256",
            "videox_runtime_identity_sha256",
        )
    }


def compare_rows(
    left: Mapping[tuple[str, int], Mapping[int, Mapping[str, Any]]],
    reference: Mapping[tuple[str, int], Mapping[int, Mapping[str, Any]]],
    *,
    left_arm: str,
    reference_arm: str,
    left_nfe: int,
    reference_nfe: int,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    label: str,
) -> dict[str, Any]:
    units = _assert_paired(
        left,
        reference,
        left_nfe=left_nfe,
        reference_nfe=reference_nfe,
    )
    metrics: dict[str, Any] = {}
    for metric in CLAIM_METRICS:
        left_values = [
            float(left[unit][left_nfe]["metrics"][metric]) for unit in units
        ]
        reference_values = [
            float(reference[unit][reference_nfe]["metrics"][metric])
            for unit in units
        ]
        metrics[metric] = paired_effect(
            left_values,
            reference_values,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
            label=f"{label}:{metric}",
        )
    result = {
        "left": {"arm": left_arm, "source": "autonomous", "nfe": left_nfe},
        "reference": {
            "arm": reference_arm,
            "source": "autonomous",
            "nfe": reference_nfe,
        },
        "metrics": metrics,
    }
    result["quality_gate"] = quality_gate(result)
    return result


def robust_vpm_frontier(
    vpm: Mapping[tuple[str, int], Mapping[int, Mapping[str, Any]]],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    """Return the conservative CI-supported VPM quality/compute frontier.

    The burden of proof stays on extra compute: a higher NFE is admissible
    only if it passes the temporal-improvement and guardrail gates against
    every lower-NFE alternative.  Statistically unresolved points therefore
    fall back to the smaller NFE instead of inflating an acceleration claim.
    """

    frontier: list[int] = [NFE_GRID[0]]
    evidence: dict[str, Any] = {}
    for target_nfe in NFE_GRID:
        if target_nfe == NFE_GRID[0]:
            evidence[str(target_nfe)] = {
                "admitted_to_conservative_frontier": True,
                "reason": "minimum available compute",
                "required_lower_nfe_comparisons": [],
            }
            continue
        comparisons: list[dict[str, Any]] = []
        for candidate_nfe in NFE_GRID:
            if candidate_nfe >= target_nfe:
                continue
            comparison = compare_rows(
                vpm,
                vpm,
                left_arm="VPM",
                reference_arm="VPM",
                left_nfe=target_nfe,
                reference_nfe=candidate_nfe,
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                seed=seed,
                label=f"vpm-extra-compute:{target_nfe}-vs-{candidate_nfe}",
            )
            comparisons.append(comparison)
        admitted = all(
            comparison["quality_gate"]["passed"]
            for comparison in comparisons
        )
        if admitted:
            frontier.append(target_nfe)
        evidence[str(target_nfe)] = {
            "admitted_to_conservative_frontier": admitted,
            "burden_of_proof": (
                "extra compute must show temporal CI-low > 0 and both "
                "guardrail CI-lows > -0.01 against every lower NFE"
            ),
            "required_lower_nfe_comparisons": comparisons,
        }
    return frontier, evidence


def select_candidate(
    j1: Mapping[tuple[str, int], Mapping[int, Mapping[str, Any]]],
    vpm: Mapping[tuple[str, int], Mapping[int, Mapping[str, Any]]],
    frontier: Sequence[int],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for reference_nfe in sorted(frontier):
        for j1_nfe in NFE_GRID:
            if j1_nfe < MIN_CAUSAL_J1_NFE or j1_nfe >= reference_nfe:
                continue
            comparison = compare_rows(
                j1,
                vpm,
                left_arm="J1",
                reference_arm="VPM",
                left_nfe=j1_nfe,
                reference_nfe=reference_nfe,
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                seed=seed,
                label=f"candidate:J1-{j1_nfe}-vs-VPM-{reference_nfe}",
            )
            same_nfe = compare_rows(
                j1,
                vpm,
                left_arm="J1",
                reference_arm="VPM",
                left_nfe=j1_nfe,
                reference_nfe=j1_nfe,
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                seed=seed,
                label=f"candidate-attribution:J1-{j1_nfe}-vs-VPM-{j1_nfe}",
            )
            same_nfe["same_nfe_attribution_gate"] = (
                same_nfe_attribution_gate(same_nfe)
            )
            reduction = (reference_nfe - j1_nfe) / reference_nfe
            comparison["nfe_reduction_fraction"] = reduction
            comparison["same_nfe_quality_attribution"] = same_nfe
            comparison["eligible"] = (
                comparison["quality_gate"]["passed"]
                and same_nfe["same_nfe_attribution_gate"]["passed"]
            )
            candidates.append(comparison)
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:
        return None, candidates
    # Deterministic validation-only choice: maximize call reduction, then the
    # primary lower CI bound, then guardrail safety; remaining ties favor fewer
    # J1 calls and fewer reference calls.
    def ranking(item: Mapping[str, Any]) -> tuple[float, ...]:
        lows = item["quality_gate"]["ci_lows"]
        attribution_low = item["same_nfe_quality_attribution"][
            "same_nfe_attribution_gate"
        ]["ci_lows"][PRIMARY_METRIC]
        return (
            float(item["nfe_reduction_fraction"]),
            float(lows[PRIMARY_METRIC]),
            float(attribution_low),
            min(float(lows[metric]) for metric in GUARDRAIL_METRICS),
            -float(item["left"]["nfe"]),
            -float(item["reference"]["nfe"]),
        )

    chosen = max(eligible, key=ranking)
    return dict(chosen), candidates


def _input_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _validated_evidence_file(
    record: Mapping[str, Any],
    *,
    label: str,
    extra_keys: Sequence[str] = (),
) -> Path:
    expected_keys = {"path", "sha256", "bytes", *extra_keys}
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise FrontierError(f"{label} evidence record has an invalid schema")
    path_value = record.get("path")
    digest = record.get("sha256")
    size = record.get("bytes")
    if (
        not isinstance(path_value, str)
        or not path_value
        or not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise FrontierError(f"{label} evidence record is invalid")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise FrontierError(f"{label} evidence path must be absolute")
    try:
        canonical = path.resolve(strict=True)
    except OSError as exc:
        raise FrontierError(f"{label} evidence file is unavailable: {path}") from exc
    if not canonical.is_file() or canonical.is_symlink():
        raise FrontierError(f"{label} evidence is not a regular file: {canonical}")
    if str(canonical) != path_value:
        raise FrontierError(f"{label} evidence path is not canonical")
    if canonical.stat().st_size != size or _sha256(canonical) != digest:
        raise FrontierError(f"{label} evidence hash/size differs")
    return canonical


def _validated_jsonl_evidence(
    records: Any,
    *,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(record, Mapping) for record in records)
    ):
        raise FrontierError(f"{label} raw input evidence is missing")
    normalized = [dict(record) for record in records]
    paths = [
        _validated_evidence_file(record, label=f"{label}[{index}]")
        for index, record in enumerate(normalized)
    ]
    if len(set(paths)) != len(paths):
        raise FrontierError(f"{label} evidence contains duplicate paths")
    return _read_jsonl(paths, label), normalized


def _validated_lockbox_input(
    record: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise FrontierError("lockbox registration input evidence is missing")
    normalized = dict(record)
    path = _validated_evidence_file(
        normalized,
        label="lockbox registration",
        extra_keys=("identity_sha256",),
    )
    identity = normalized.get("identity_sha256")
    if not isinstance(identity, str) or SHA256_RE.fullmatch(identity) is None:
        raise FrontierError("lockbox registration evidence identity is invalid")
    registration = _read_json(path, "lockbox registration evidence")
    if (
        not identity_valid(registration)
        or registration.get("kind") != lockbox.KIND_REGISTRATION
        or registration.get("identity_sha256") != identity
    ):
        raise FrontierError("lockbox registration evidence identity differs")
    return registration, normalized


def _validated_lockbox_envelope(
    registration: Mapping[str, Any] | None,
    *,
    study_identity: str | None,
    training_commit: str | None,
) -> dict[str, Any] | None:
    if registration is None:
        return None
    manifest = registration.get("manifest")
    cache = registration.get("cache")
    construction = registration.get("construction")
    arrays = cache.get("arrays") if isinstance(cache, Mapping) else None
    audit = registration.get("episode_isolation")
    if (
        not lockbox.identity_valid(registration)
        or registration.get("kind") != lockbox.KIND_REGISTRATION
        or registration.get("study_identity_sha256") != study_identity
        or registration.get("training_git_commit") != training_commit
        or not isinstance(registration.get("registration_git_commit"), str)
        or COMMIT_RE.fullmatch(registration["registration_git_commit"]) is None
        or not isinstance(manifest, Mapping)
        or manifest.get("entries") != lockbox.LOCKBOX_CLIPS
        or not isinstance(cache, Mapping)
        or cache.get("clip_count") != lockbox.LOCKBOX_CLIPS
        or not isinstance(construction, Mapping)
        or not isinstance(arrays, Mapping)
        or set(arrays) != {"target", "rgb", "actions"}
        or not isinstance(audit, Mapping)
        or any(
            audit.get("lockbox_original_episode_overlap_counts", {}).get(name)
            != 0
            for name in ("train", "validation", "test")
        )
        or registration.get("operator_attestation", {}).get(
            "lockbox_never_scored_before_registration"
        )
        is not True
        or registration.get("operator_attestation", {}).get("text")
        != lockbox.ATTESTATION_TEXT
        or registration.get("inference_code_compatibility", {}).get(
            "training_commit_is_ancestor"
        )
        is not True
        or registration.get("inference_code_compatibility", {}).get(
            "inference_critical_paths_unchanged"
        )
        is not True
        or registration.get("selection_must_bind_registration_before_evaluation")
        is not True
    ):
        raise FrontierError("lockbox registration envelope is invalid")
    for record in (manifest, cache.get("metadata"), construction, *arrays.values()):
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("sha256"), str)
            or SHA256_RE.fullmatch(record["sha256"]) is None
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
        ):
            raise FrontierError("lockbox registration lacks a file hash/size")
    return dict(registration)


def build_selection(
    *,
    j1_rows: Sequence[Mapping[str, Any]],
    vpm_rows: Sequence[Mapping[str, Any]],
    split: str,
    expected_clips: int,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    allow_posthoc: bool,
    lockbox_registration: Mapping[str, Any] | None = None,
    lockbox_input: Mapping[str, Any] | None = None,
    j1_inputs: Sequence[Mapping[str, Any]] = (),
    vpm_inputs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    j1, j1_bound = validate_arm_rows(
        j1_rows,
        arm="J1",
        split=split,
        required_nfes=NFE_GRID,
        expected_clips=expected_clips,
        selection_identity=None,
        allow_posthoc=allow_posthoc,
    )
    vpm, vpm_bound = validate_arm_rows(
        vpm_rows,
        arm="VPM",
        split=split,
        required_nfes=NFE_GRID,
        expected_clips=expected_clips,
        selection_identity=None,
        allow_posthoc=allow_posthoc,
    )
    _assert_paired(j1, vpm, left_nfe=1, reference_nfe=1)
    j1_provenance = _provenance_identities(j1_rows, "J1")
    vpm_provenance = _provenance_identities(vpm_rows, "VPM")
    provenance_is_bound = (
        len(j1_provenance["study_identity_sha256"]) == 1
        and j1_provenance["study_identity_sha256"]
        == vpm_provenance["study_identity_sha256"]
        and len(j1_provenance["arm_identity_sha256"]) == 1
        and len(vpm_provenance["arm_identity_sha256"]) == 1
        and len(j1_provenance["stage_identity_sha256"]) == 1
        and len(vpm_provenance["stage_identity_sha256"]) == 1
        and len(j1_provenance["training_git_commit"]) == 1
        and j1_provenance["training_git_commit"]
        == vpm_provenance["training_git_commit"]
        and len(j1_provenance["evaluator_git_commit"]) == 1
        and j1_provenance["evaluator_git_commit"]
        == vpm_provenance["evaluator_git_commit"]
        and len(j1_provenance["inference_code_compatibility_sha256"]) == 1
        and j1_provenance["inference_code_compatibility_sha256"]
        == vpm_provenance["inference_code_compatibility_sha256"]
        and len(j1_provenance["videox_runtime_identity_sha256"]) == 1
        and j1_provenance["videox_runtime_identity_sha256"]
        == vpm_provenance["videox_runtime_identity_sha256"]
    )
    training_commit = (
        next(iter(j1_provenance["training_git_commit"]))
        if provenance_is_bound
        else None
    )
    study_identity = (
        next(iter(j1_provenance["study_identity_sha256"]))
        if provenance_is_bound
        else None
    )
    lockbox_record = _validated_lockbox_envelope(
        lockbox_registration,
        study_identity=study_identity,
        training_commit=training_commit,
    )
    if (
        lockbox_record is not None
        and provenance_is_bound
        and lockbox_record.get("registration_git_commit")
        != next(iter(j1_provenance["evaluator_git_commit"]))
    ):
        raise FrontierError(
            "lockbox registration and validation evaluator commits differ"
        )
    expected_compatibility_sha = (
        hashlib.sha256(
            _canonical_json(lockbox_record["inference_code_compatibility"])
        ).hexdigest()
        if lockbox_record is not None
        else None
    )
    if (
        lockbox_record is not None
        and provenance_is_bound
        and j1_provenance["inference_code_compatibility_sha256"]
        != {expected_compatibility_sha}
    ):
        raise FrontierError(
            "lockbox/evaluator inference compatibility evidence differs"
        )
    explicit_validation = (
        split == "validation"
        and expected_clips == 64
        and len(j1) == 64
        and len(vpm) == 64
        and bootstrap_samples == DEFAULT_BOOTSTRAP_SAMPLES
        and confidence == DEFAULT_CONFIDENCE
        and seed == DEFAULT_SEED
        and j1_bound
        and vpm_bound
        and lockbox_record is not None
    )
    explicit_validation = explicit_validation and provenance_is_bound
    if not explicit_validation and not allow_posthoc:
        raise FrontierError("candidate selection must use explicit validation rows")
    frontier, dominance = robust_vpm_frontier(
        vpm,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    chosen, candidates = select_candidate(
        j1,
        vpm,
        frontier,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    units = sorted(j1)
    status = (
        "validation_selected_confirmatory_candidate"
        if explicit_validation
        else "posthoc_exploratory_selection"
    )
    return identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_SELECTION,
            "status": status,
            "confirmatory_eligible": explicit_validation and chosen is not None,
            "selection_split": split,
            "training_git_commit": (
                training_commit
            ),
            "evaluator_git_commit": (
                next(iter(j1_provenance["evaluator_git_commit"]))
                if provenance_is_bound
                else None
            ),
            "inference_code_compatibility_sha256": (
                expected_compatibility_sha
            ),
            "videox_runtime_identity_sha256": (
                next(iter(j1_provenance["videox_runtime_identity_sha256"]))
                if provenance_is_bound
                else None
            ),
            "study_identity_sha256": (
                study_identity
            ),
            "lockbox_registration": lockbox_record,
            "arm_identity_sha256": {
                "J1": (
                    next(iter(j1_provenance["arm_identity_sha256"]))
                    if len(j1_provenance["arm_identity_sha256"]) == 1
                    else None
                ),
                "VPM": (
                    next(iter(vpm_provenance["arm_identity_sha256"]))
                    if len(vpm_provenance["arm_identity_sha256"]) == 1
                    else None
                ),
            },
            "stage_identity_sha256": {
                "J1": (
                    next(iter(j1_provenance["stage_identity_sha256"]))
                    if len(j1_provenance["stage_identity_sha256"]) == 1
                    else None
                ),
                "VPM": (
                    next(iter(vpm_provenance["stage_identity_sha256"]))
                    if len(vpm_provenance["stage_identity_sha256"]) == 1
                    else None
                ),
            },
            "selection_used_test_metrics": split == "test",
            "sampling_id_namespace": {
                "split": split,
                "offset": SAMPLE_ID_OFFSETS[split],
                "ids": [
                    SAMPLE_ID_OFFSETS[split],
                    SAMPLE_ID_OFFSETS[split] + expected_clips - 1,
                ],
                "validation_and_lockbox_namespaces_disjoint": True,
            },
            "legacy_or_unbound_rows_used": not (j1_bound and vpm_bound),
            "expected_clip_count": expected_clips,
            "observed_clip_count": len(units),
            "selection_clip_units": [
                [clip_id, index] for clip_id, index in units
            ],
            "clip_identity_sha256": hashlib.sha256(
                _canonical_json([[clip_id, index] for clip_id, index in units])
            ).hexdigest(),
            "nfe_grid": list(NFE_GRID),
            "nfe_one_policy": (
                "eligible for VPM frontier estimation but excluded from J1 "
                "causal candidates because the first call sees auxiliary noise"
            ),
            "metrics": {
                "primary": PRIMARY_METRIC,
                "guardrails": list(GUARDRAIL_METRICS),
                "direction": "lower_is_better",
            },
            "bootstrap": {
                "samples": bootstrap_samples,
                "confidence": confidence,
                "seed": seed,
                "unit": "paired immutable clip_id/clip_index",
            },
            "dominance_rule": (
                "extra compute is admitted only when the higher-NFE VPM "
                "point passes the three quality gates against every lower "
                "NFE; unresolved comparisons fall back to lower compute"
            ),
            "vpm_non_dominated_nfe_frontier": frontier,
            "vpm_dominance_evidence": dominance,
            "candidate_ranking": (
                "among candidates passing both frontier preservation and "
                "same-NFE JEPA attribution: max NFE reduction; then temporal "
                "CI-low; attribution CI-low; minimum guardrail CI-low; then "
                "lower J1 NFE and lower VPM NFE"
            ),
            "candidate_count": len(candidates),
            "eligible_candidate_count": sum(
                int(item["eligible"]) for item in candidates
            ),
            "candidates": candidates,
            "selected_pair": chosen,
            "input_evidence": {
                "J1": list(j1_inputs),
                "VPM": list(vpm_inputs),
                "lockbox_registration": (
                    dict(lockbox_input)
                    if isinstance(lockbox_input, Mapping)
                    else None
                ),
            },
            "claim_restriction": (
                "posthoc selections are exploratory and cannot support a "
                "held-out inference-acceleration claim"
                if not explicit_validation
                else (
                    "pair is frozen from validation; test quality and paired "
                    "same-B200 timing remain required"
                )
            ),
        }
    )


def validate_confirmatory_selection(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Reload validation evidence and reproduce the exact frozen winner."""

    if not identity_valid(selection) or selection.get("kind") != KIND_SELECTION:
        raise FrontierError("selection manifest identity/kind is invalid")
    bootstrap = selection.get("bootstrap")
    namespace = selection.get("sampling_id_namespace")
    if (
        selection.get("schema_version") != SCHEMA_VERSION
        or selection.get("status")
        != "validation_selected_confirmatory_candidate"
        or selection.get("confirmatory_eligible") is not True
        or selection.get("selection_split") != "validation"
        or selection.get("selection_used_test_metrics") is not False
        or selection.get("legacy_or_unbound_rows_used") is not False
        or selection.get("expected_clip_count") != 64
        or selection.get("observed_clip_count") != 64
        or selection.get("nfe_grid") != list(NFE_GRID)
        or bootstrap
        != {
            "samples": DEFAULT_BOOTSTRAP_SAMPLES,
            "confidence": DEFAULT_CONFIDENCE,
            "seed": DEFAULT_SEED,
            "unit": "paired immutable clip_id/clip_index",
        }
        or namespace
        != {
            "split": "validation",
            "offset": SAMPLE_ID_OFFSETS["validation"],
            "ids": [
                SAMPLE_ID_OFFSETS["validation"],
                SAMPLE_ID_OFFSETS["validation"] + 63,
            ],
            "validation_and_lockbox_namespaces_disjoint": True,
        }
    ):
        raise FrontierError("selection is not the pinned confirmatory protocol")
    evidence = selection.get("input_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "J1",
        "VPM",
        "lockbox_registration",
    }:
        raise FrontierError("selection raw input evidence is missing")
    j1_rows, j1_inputs = _validated_jsonl_evidence(
        evidence["J1"], label="selection J1 rows"
    )
    vpm_rows, vpm_inputs = _validated_jsonl_evidence(
        evidence["VPM"], label="selection VPM rows"
    )
    registration, registration_input = _validated_lockbox_input(
        evidence["lockbox_registration"]
    )
    embedded_registration = selection.get("lockbox_registration")
    if (
        not isinstance(embedded_registration, Mapping)
        or dict(embedded_registration) != registration
    ):
        raise FrontierError(
            "selection embedded lockbox differs from registered evidence"
        )
    recomputed = build_selection(
        j1_rows=j1_rows,
        vpm_rows=vpm_rows,
        split="validation",
        expected_clips=64,
        bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
        confidence=DEFAULT_CONFIDENCE,
        seed=DEFAULT_SEED,
        allow_posthoc=False,
        lockbox_registration=registration,
        lockbox_input=registration_input,
        j1_inputs=j1_inputs,
        vpm_inputs=vpm_inputs,
    )
    if dict(selection) != recomputed:
        raise FrontierError(
            "selection does not reproduce from its raw validation evidence"
        )
    return {
        "selection": dict(selection),
        "j1_rows": j1_rows,
        "vpm_rows": vpm_rows,
        "lockbox_registration": registration,
    }


def build_confirmation(
    *,
    selection: Mapping[str, Any],
    j1_rows: Sequence[Mapping[str, Any]],
    vpm_rows: Sequence[Mapping[str, Any]],
    expected_clips: int,
    allow_posthoc: bool,
    j1_inputs: Sequence[Mapping[str, Any]] = (),
    vpm_inputs: Sequence[Mapping[str, Any]] = (),
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    if allow_posthoc:
        raise FrontierError("lockbox confirmation cannot use posthoc evidence")
    validate_confirmatory_selection(selection)
    loaded_j1_rows, normalized_j1_inputs = _validated_jsonl_evidence(
        list(j1_inputs), label="confirmation J1 rows"
    )
    loaded_vpm_rows, normalized_vpm_inputs = _validated_jsonl_evidence(
        list(vpm_inputs), label="confirmation VPM rows"
    )
    if (
        list(j1_rows) != loaded_j1_rows
        or list(vpm_rows) != loaded_vpm_rows
    ):
        raise FrontierError(
            "confirmation rows differ from their hashed raw input evidence"
        )
    j1_rows = loaded_j1_rows
    vpm_rows = loaded_vpm_rows
    pair = selection.get("selected_pair")
    if not isinstance(pair, Mapping):
        raise FrontierError("selection contains no eligible pair")
    k = pair.get("left", {}).get("nfe")
    m = pair.get("reference", {}).get("nfe")
    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or isinstance(m, bool)
        or not isinstance(m, int)
        or k < MIN_CAUSAL_J1_NFE
        or k >= m
        or k not in NFE_GRID
        or m not in NFE_GRID
    ):
        raise FrontierError("frozen selection pair is invalid")
    selection_id = str(selection["identity_sha256"])
    lockbox_registration = selection.get("lockbox_registration")
    lockbox_record = _validated_lockbox_envelope(
        lockbox_registration if isinstance(lockbox_registration, Mapping) else None,
        study_identity=(
            str(selection.get("study_identity_sha256"))
            if isinstance(selection.get("study_identity_sha256"), str)
            else None
        ),
        training_commit=(
            str(selection.get("training_git_commit"))
            if isinstance(selection.get("training_git_commit"), str)
            else None
        ),
    )
    lockbox_identity = (
        str(lockbox_record["identity_sha256"])
        if lockbox_record is not None
        else None
    )
    j1, j1_bound = validate_arm_rows(
        j1_rows,
        arm="J1",
        split="lockbox",
        required_nfes=(k,),
        expected_clips=expected_clips,
        selection_identity=selection_id,
        lockbox_identity=lockbox_identity,
        allow_posthoc=allow_posthoc,
    )
    vpm, vpm_bound = validate_arm_rows(
        vpm_rows,
        arm="VPM",
        split="lockbox",
        required_nfes=(k, m),
        expected_clips=expected_clips,
        selection_identity=selection_id,
        lockbox_identity=lockbox_identity,
        allow_posthoc=allow_posthoc,
    )
    j1_provenance = _provenance_identities(j1_rows, "J1")
    vpm_provenance = _provenance_identities(vpm_rows, "VPM")
    selected_study = selection.get("study_identity_sha256")
    test_provenance_bound = (
        isinstance(selected_study, str)
        and j1_provenance["study_identity_sha256"] == {selected_study}
        and vpm_provenance["study_identity_sha256"] == {selected_study}
        and j1_provenance["arm_identity_sha256"]
        == {selection.get("arm_identity_sha256", {}).get("J1")}
        and vpm_provenance["arm_identity_sha256"]
        == {selection.get("arm_identity_sha256", {}).get("VPM")}
        and j1_provenance["stage_identity_sha256"]
        == {selection.get("stage_identity_sha256", {}).get("J1")}
        and vpm_provenance["stage_identity_sha256"]
        == {selection.get("stage_identity_sha256", {}).get("VPM")}
        and j1_provenance["training_git_commit"]
        == {selection.get("training_git_commit")}
        and vpm_provenance["training_git_commit"]
        == {selection.get("training_git_commit")}
        and j1_provenance["evaluator_git_commit"]
        == {selection.get("evaluator_git_commit")}
        and vpm_provenance["evaluator_git_commit"]
        == {selection.get("evaluator_git_commit")}
        and j1_provenance["inference_code_compatibility_sha256"]
        == {selection.get("inference_code_compatibility_sha256")}
        and vpm_provenance["inference_code_compatibility_sha256"]
        == {selection.get("inference_code_compatibility_sha256")}
        and j1_provenance["videox_runtime_identity_sha256"]
        == {selection.get("videox_runtime_identity_sha256")}
        and vpm_provenance["videox_runtime_identity_sha256"]
        == {selection.get("videox_runtime_identity_sha256")}
    )
    validation_units = {
        tuple(value)
        for value in selection.get("selection_clip_units", [])
        if isinstance(value, list) and len(value) == 2
    }
    lockbox_units = set(j1)
    disjoint = not validation_units.intersection(lockbox_units)
    # Older selection schema instances may only expose the digest.  Such
    # evidence cannot prove disjointness and therefore remains posthoc.
    explicit_confirmatory = (
        selection.get("confirmatory_eligible") is True
        and selection.get("expected_clip_count") == 64
        and selection.get("observed_clip_count") == 64
        and len(validation_units) == 64
        and expected_clips == 128
        and len(j1) == 128
        and len(vpm) == 128
        and j1_bound
        and vpm_bound
        and test_provenance_bound
        and bool(validation_units)
        and disjoint
        and lockbox_record is not None
    )
    if not explicit_confirmatory and not allow_posthoc:
        raise FrontierError(
            "test evidence is not bound after a disjoint validation selection"
        )
    bootstrap = selection.get("bootstrap", {})
    comparison = compare_rows(
        j1,
        vpm,
        left_arm="J1",
        reference_arm="VPM",
        left_nfe=k,
        reference_nfe=m,
        bootstrap_samples=int(bootstrap["samples"]),
        confidence=float(bootstrap["confidence"]),
        seed=int(bootstrap["seed"]),
        label=f"held-out-lockbox:J1-{k}-vs-VPM-{m}",
    )
    same_nfe = compare_rows(
        j1,
        vpm,
        left_arm="J1",
        reference_arm="VPM",
        left_nfe=k,
        reference_nfe=k,
        bootstrap_samples=int(bootstrap["samples"]),
        confidence=float(bootstrap["confidence"]),
        seed=int(bootstrap["seed"]),
        label=f"held-out-lockbox:J1-{k}-vs-VPM-{k}",
    )
    frontier_quality_pass = comparison["quality_gate"]["passed"]
    same_nfe["same_nfe_attribution_gate"] = same_nfe_attribution_gate(same_nfe)
    same_nfe_attribution_pass = same_nfe[
        "same_nfe_attribution_gate"
    ]["passed"]
    confirmatory_pass = (
        explicit_confirmatory
        and frontier_quality_pass
        and same_nfe_attribution_pass
    )
    return identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_CONFIRMATION,
            "created_at_utc": created_at_utc or _now(),
            "selection": {
                "identity_sha256": selection_id,
                "status": selection.get("status"),
                "selected_J1_nfe": k,
                "selected_VPM_frontier_nfe": m,
            },
            "training_git_commit": selection.get("training_git_commit"),
            "evaluator_git_commit": selection.get("evaluator_git_commit"),
            "study_identity_sha256": selection.get("study_identity_sha256"),
            "videox_runtime_identity_sha256": selection.get(
                "videox_runtime_identity_sha256"
            ),
            "evaluation_split": "lockbox",
            "lockbox_registration_identity_sha256": lockbox_identity,
            "expected_clip_count": expected_clips,
            "observed_clip_count": len(lockbox_units),
            "validation_lockbox_clip_units_disjoint": disjoint,
            "validation_lockbox_sampling_id_namespaces_disjoint": True,
            "rows_explicitly_bound_to_frozen_selection": j1_bound and vpm_bound,
            "rows_bound_to_selection_study_and_final_stages": (
                test_provenance_bound
            ),
            "lockbox_selection_decisions": 0,
            "confirmatory_evidence": explicit_confirmatory,
            "posthoc_exploratory": not explicit_confirmatory,
            "frontier_quality_comparison": comparison,
            "frontier_quality_gate_passed": frontier_quality_pass,
            "same_nfe_quality_attribution": same_nfe,
            "same_nfe_quality_attribution_gate_passed": (
                same_nfe_attribution_pass
            ),
            "quality_gate_passed": confirmatory_pass,
            "input_evidence": {
                "J1": normalized_j1_inputs,
                "VPM": normalized_vpm_inputs,
            },
        }
    )


def _timing_percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower, upper = int(math.floor(rank)), int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timing_summary(values: Sequence[float]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    if (
        len(normalized) != 120
        or any(not math.isfinite(value) or value <= 0 for value in normalized)
    ):
        raise FrontierError("timing values must be 120 finite positive latencies")
    return {
        "count": len(normalized),
        "mean": sum(normalized) / len(normalized),
        "p50": _timing_percentile(normalized, 50.0),
        "p95": _timing_percentile(normalized, 95.0),
        "min": min(normalized),
        "max": max(normalized),
        "values_sha256": hashlib.sha256(
            _canonical_json([round(value, 9) for value in normalized])
        ).hexdigest(),
        "values": normalized,
    }


def _timing_effect(
    left_values: Sequence[float],
    reference_values: Sequence[float],
    orders: Sequence[Sequence[str]],
    *,
    left_label: str,
    reference_label: str,
    seed: int,
    label: str,
) -> dict[str, Any]:
    left = np.asarray(left_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    if (
        left.shape != (120,)
        or reference.shape != (120,)
        or len(orders) != 120
        or not np.isfinite(left).all()
        or not np.isfinite(reference).all()
        or np.any(left <= 0)
        or np.any(reference <= 0)
    ):
        raise FrontierError(f"invalid paired timing input for {label}")
    strata: dict[str, np.ndarray] = {}
    for first, second in (
        (left_label, reference_label),
        (reference_label, left_label),
    ):
        indexes = [
            index
            for index, order in enumerate(orders)
            if tuple(order).index(first) < tuple(order).index(second)
        ]
        if len(indexes) != 60:
            raise FrontierError(f"{label} timing order strata are unbalanced")
        strata[f"{first}_before_{second}"] = np.asarray(
            indexes, dtype=np.int64
        )
    block_count = 20
    left_blocks = left.reshape(block_count, 6).mean(axis=1)
    reference_blocks = reference.reshape(block_count, 6).mean(axis=1)
    rng = np.random.default_rng(_derived_seed(seed, label))
    draws = rng.integers(
        0,
        block_count,
        size=(DEFAULT_BOOTSTRAP_SAMPLES, block_count),
        endpoint=False,
    )
    left_boot = left_blocks[draws].mean(axis=1)
    reference_boot = reference_blocks[draws].mean(axis=1)
    effects = (reference_boot - left_boot) / reference_boot
    low, high = np.quantile(effects, [0.025, 0.975])
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
        "n_paired_rounds": 120,
        "n_counterbalance_blocks": 20,
        "bootstrap_unit": (
            "complete six-round counterbalance block; preserves every "
            "endpoint permutation and pairwise execution-order balance"
        ),
        "mean_favorable_difference_ms": float(difference.mean()),
        "relative_improvement": float(relative),
        "relative_improvement_percent": float(relative * 100.0),
        "bootstrap_ci": {
            "confidence": DEFAULT_CONFIDENCE,
            "low": float(low),
            "high": float(high),
        },
        "favorable_pair_fraction": float(np.mean(difference > 0)),
        "order_strata": order_strata,
    }


def _timing_gate_from_effect(
    effect: Mapping[str, Any],
    *,
    left_p95: float,
    reference_p95: float,
) -> dict[str, Any]:
    checks = {
        "paired_speedup_ci_low_strictly_positive": (
            float(effect["bootstrap_ci"]["low"]) > 0.0
        ),
        "left_p95_lower_than_reference_p95": left_p95 < reference_p95,
        "both_execution_order_strata_favorable": (
            len(effect["order_strata"]) == 2
            and all(
                float(value["mean_favorable_difference_ms"]) > 0.0
                for value in effect["order_strata"].values()
            )
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "rule": (
            "paired stratified-bootstrap relative speedup CI-low > 0; "
            "J1 p95 lower; both pairwise order strata favorable"
        ),
    }


def build_final_report(
    *,
    selection: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    latency: Mapping[str, Any],
) -> dict[str, Any]:
    for payload, kind, label in (
        (selection, KIND_SELECTION, "selection"),
        (confirmation, KIND_CONFIRMATION, "confirmation"),
        (latency, KIND_LATENCY, "latency"),
    ):
        if not identity_valid(payload) or payload.get("kind") != kind:
            raise FrontierError(f"{label} identity/kind is invalid")
    validate_confirmatory_selection(selection)
    confirmation_evidence = confirmation.get("input_evidence")
    if (
        not isinstance(confirmation_evidence, Mapping)
        or set(confirmation_evidence) != {"J1", "VPM"}
        or confirmation.get("posthoc_exploratory") is not False
        or confirmation.get("confirmatory_evidence") is not True
        or confirmation.get("quality_gate_passed") is not True
        or not isinstance(confirmation.get("created_at_utc"), str)
        or not confirmation["created_at_utc"]
    ):
        raise FrontierError(
            "confirmation is posthoc or lacks raw confirmatory evidence"
        )
    raw_j1_rows, raw_j1_inputs = _validated_jsonl_evidence(
        confirmation_evidence["J1"], label="confirmation J1 rows"
    )
    raw_vpm_rows, raw_vpm_inputs = _validated_jsonl_evidence(
        confirmation_evidence["VPM"], label="confirmation VPM rows"
    )
    recomputed_confirmation = build_confirmation(
        selection=selection,
        j1_rows=raw_j1_rows,
        vpm_rows=raw_vpm_rows,
        expected_clips=128,
        allow_posthoc=False,
        j1_inputs=raw_j1_inputs,
        vpm_inputs=raw_vpm_inputs,
        created_at_utc=confirmation["created_at_utc"],
    )
    if dict(confirmation) != recomputed_confirmation:
        raise FrontierError(
            "confirmation does not reproduce from its raw lockbox evidence"
        )
    selection_id = selection["identity_sha256"]
    selected_pair = selection.get("selected_pair")
    if not isinstance(selected_pair, Mapping):
        raise FrontierError("selection lacks a frozen pair")
    k = selected_pair.get("left", {}).get("nfe")
    m = selected_pair.get("reference", {}).get("nfe")
    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or isinstance(m, bool)
        or not isinstance(m, int)
        or k < MIN_CAUSAL_J1_NFE
        or k >= m
        or m not in selection.get("vpm_non_dominated_nfe_frontier", [])
    ):
        raise FrontierError("selection frozen pair/frontier is invalid")
    lockbox_registration = selection.get("lockbox_registration")
    lockbox_record = _validated_lockbox_envelope(
        lockbox_registration if isinstance(lockbox_registration, Mapping) else None,
        study_identity=(
            str(selection.get("study_identity_sha256"))
            if isinstance(selection.get("study_identity_sha256"), str)
            else None
        ),
        training_commit=(
            str(selection.get("training_git_commit"))
            if isinstance(selection.get("training_git_commit"), str)
            else None
        ),
    )
    if lockbox_record is None:
        raise FrontierError("selection does not bind a registered lockbox")
    lockbox_id = lockbox_record["identity_sha256"]
    if (
        confirmation.get("selection", {}).get("identity_sha256") != selection_id
        or latency.get("selection", {}).get("identity_sha256") != selection_id
        or confirmation.get("selection", {}).get("selected_J1_nfe") != k
        or confirmation.get("selection", {}).get(
            "selected_VPM_frontier_nfe"
        )
        != m
        or confirmation.get("evaluation_split") != "lockbox"
        or confirmation.get("lockbox_registration_identity_sha256")
        != lockbox_id
        or latency.get("lockbox_registration_identity_sha256") != lockbox_id
        or selection.get("expected_clip_count") != 64
        or selection.get("observed_clip_count") != 64
        or confirmation.get("expected_clip_count") != 128
        or confirmation.get("observed_clip_count") != 128
        or confirmation.get("validation_lockbox_clip_units_disjoint") is not True
        or confirmation.get(
            "validation_lockbox_sampling_id_namespaces_disjoint"
        )
        is not True
        or confirmation.get("rows_explicitly_bound_to_frozen_selection")
        is not True
        or confirmation.get(
            "rows_bound_to_selection_study_and_final_stages"
        )
        is not True
        or confirmation.get("lockbox_selection_decisions") != 0
    ):
        raise FrontierError(
            "quality/timing artifacts bind different selection/lockbox endpoints"
        )
    training_commit = selection.get("training_git_commit")
    evaluator_commit = selection.get("evaluator_git_commit")
    benchmark_commit = latency.get("benchmark_git_commit")
    latency_compatibility = latency.get("inference_code_compatibility")
    if not isinstance(latency_compatibility, Mapping):
        raise FrontierError("latency lacks inference-code compatibility")
    latency_evaluator_compatibility = latency_compatibility.get("evaluator")
    latency_benchmark_compatibility = latency_compatibility.get("benchmark")
    if not isinstance(latency_evaluator_compatibility, Mapping) or not isinstance(
        latency_benchmark_compatibility, Mapping
    ):
        raise FrontierError("latency compatibility records are invalid")
    if (
        confirmation.get("training_git_commit") != training_commit
        or confirmation.get("evaluator_git_commit") != evaluator_commit
        or confirmation.get("videox_runtime_identity_sha256")
        != selection.get("videox_runtime_identity_sha256")
        or confirmation.get("study_identity_sha256")
        != selection.get("study_identity_sha256")
        or latency.get("training_git_commit") != training_commit
        or latency.get("evaluator_git_commit") != evaluator_commit
        or not isinstance(latency.get("videox_runtime"), Mapping)
        or latency["videox_runtime"].get("git_commit")
        != EXPECTED_VIDEOX_COMMIT
        or latency["videox_runtime"].get("clean") is not True
        or hashlib.sha256(
            _canonical_json(latency["videox_runtime"])
        ).hexdigest()
        != selection.get("videox_runtime_identity_sha256")
        or hashlib.sha256(
            _canonical_json(latency_evaluator_compatibility)
        ).hexdigest()
        != selection.get("inference_code_compatibility_sha256")
        or not isinstance(benchmark_commit, str)
        or COMMIT_RE.fullmatch(benchmark_commit) is None
        or latency_evaluator_compatibility
        .get("training_commit_is_ancestor")
        is not True
        or latency_evaluator_compatibility
        .get("inference_critical_paths_unchanged")
        is not True
        or latency_benchmark_compatibility.get("training_commit_is_ancestor")
        is not True
        or latency_benchmark_compatibility.get(
            "inference_critical_paths_unchanged"
        )
        is not True
    ):
        raise FrontierError("quality/timing Git or study provenance differs")

    frontier_quality = confirmation.get("frontier_quality_comparison")
    same_quality = confirmation.get("same_nfe_quality_attribution")
    if (
        not isinstance(frontier_quality, Mapping)
        or frontier_quality.get("left")
        != {"arm": "J1", "source": "autonomous", "nfe": k}
        or frontier_quality.get("reference")
        != {"arm": "VPM", "source": "autonomous", "nfe": m}
        or not isinstance(same_quality, Mapping)
        or same_quality.get("left")
        != {"arm": "J1", "source": "autonomous", "nfe": k}
        or same_quality.get("reference")
        != {"arm": "VPM", "source": "autonomous", "nfe": k}
    ):
        raise FrontierError("confirmation quality endpoints differ from selection")
    for comparison in (frontier_quality, same_quality):
        metrics = comparison.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(CLAIM_METRICS):
            raise FrontierError("confirmation metric inventory differs")
        for metric in CLAIM_METRICS:
            effect = metrics[metric]
            if (
                not isinstance(effect, Mapping)
                or effect.get("n_paired_clips") != 128
                or effect.get("bootstrap_ci", {}).get("confidence")
                != DEFAULT_CONFIDENCE
            ):
                raise FrontierError(
                    "confirmation paired-count/bootstrap protocol differs"
                )
    frontier_quality_pass = quality_gate(frontier_quality)["passed"]
    same_quality_pass = same_nfe_attribution_gate(same_quality)["passed"]
    if (
        confirmation.get("frontier_quality_gate_passed")
        is not frontier_quality_pass
        or confirmation.get("same_nfe_quality_attribution_gate_passed")
        is not same_quality_pass
        or confirmation.get("quality_gate_passed")
        is not (
            confirmation.get("confirmatory_evidence") is True
            and frontier_quality_pass
            and same_quality_pass
        )
    ):
        raise FrontierError("confirmation quality booleans do not recompute")

    endpoints = latency.get("endpoints")
    expected_endpoints = {
        "J1_k": ("J1", k),
        "VPM_k": ("VPM", k),
        "VPM_m": ("VPM", m),
    }
    if not isinstance(endpoints, Mapping) or set(endpoints) != set(
        expected_endpoints
    ):
        raise FrontierError("latency endpoint inventory differs")
    rounds = latency.get("rounds")
    if (
        not isinstance(rounds, list)
        or len(rounds) != 120
        or latency.get("rounds_sha256")
        != hashlib.sha256(_canonical_json(rounds)).hexdigest()
    ):
        raise FrontierError("latency raw-round inventory/digest differs")
    orders: list[list[str]] = []
    timing_values = {name: [] for name in TIMING_ENDPOINTS}
    for index, round_record in enumerate(rounds):
        expected_order = list(BALANCED_TIMING_ORDERS[index % 6])
        if (
            not isinstance(round_record, Mapping)
            or round_record.get("round_index") != index
            or round_record.get("execution_order") != expected_order
            or not isinstance(round_record.get("latency_ms"), Mapping)
            or set(round_record["latency_ms"]) != set(TIMING_ENDPOINTS)
        ):
            raise FrontierError(f"latency raw round {index} is invalid")
        orders.append(expected_order)
        for endpoint_name in TIMING_ENDPOINTS:
            value = round_record["latency_ms"][endpoint_name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise FrontierError(
                    f"latency raw round {index} has invalid {endpoint_name}"
                )
            timing_values[endpoint_name].append(float(value))
    recomputed_summaries = {
        name: _timing_summary(values)
        for name, values in timing_values.items()
    }
    for name, (arm, nfe) in expected_endpoints.items():
        endpoint = endpoints[name]
        latency_summary = (
            endpoint.get("latency_ms") if isinstance(endpoint, Mapping) else None
        )
        if (
            not isinstance(endpoint, Mapping)
            or endpoint.get("arm") != arm
            or endpoint.get("source") != "autonomous"
            or endpoint.get("nfe") != nfe
            or endpoint.get("actual_wan_calls") != nfe
            or endpoint.get("audit", {}).get("actual_wan_calls") != nfe
            or not isinstance(latency_summary, Mapping)
            or dict(latency_summary) != recomputed_summaries[name]
            or any(
                isinstance(latency_summary.get(field), bool)
                or not isinstance(latency_summary.get(field), (int, float))
                or not math.isfinite(float(latency_summary[field]))
                or float(latency_summary[field]) <= 0
                for field in ("mean", "p50", "p95")
            )
            or not isinstance(
                endpoint.get("generated_frames_per_second_at_p95"),
                (int, float),
            )
            or float(endpoint["generated_frames_per_second_at_p95"]) <= 0
            or not isinstance(
                endpoint.get("peak_allocated_bytes_with_both_models_resident"),
                int,
            )
            or endpoint["peak_allocated_bytes_with_both_models_resident"] <= 0
            or not isinstance(
                endpoint.get("audit_peak_allocated_bytes_with_artifact_capture"),
                int,
            )
            or endpoint["audit_peak_allocated_bytes_with_artifact_capture"] <= 0
        ):
            raise FrontierError(f"latency endpoint accounting differs: {name}")
    for arm in ("J1", "VPM"):
        provenance = latency.get("model_provenance", {}).get(arm, {})
        if (
            provenance.get("arm_manifest", {}).get("identity_sha256")
            != selection.get("arm_identity_sha256", {}).get(arm)
            or provenance.get("stage_manifest", {}).get("identity_sha256")
            != selection.get("stage_identity_sha256", {}).get(arm)
        ):
            raise FrontierError(
                f"latency {arm} arm/stage identity differs from selection"
            )
    timing_gate = latency.get("frontier_acceleration", {}).get("timing_gate")
    frontier_effect = latency.get("frontier_acceleration", {}).get(
        "paired_speed_effect"
    )
    same_effect = latency.get("same_nfe_overhead", {}).get(
        "paired_speed_effect"
    )
    if (
        not isinstance(timing_gate, Mapping)
        or not isinstance(frontier_effect, Mapping)
        or frontier_effect.get("left") != "J1_k"
        or frontier_effect.get("reference") != "VPM_m"
        or not isinstance(same_effect, Mapping)
        or same_effect.get("left") != "J1_k"
        or same_effect.get("reference") != "VPM_k"
        or frontier_effect.get("n_paired_rounds") != 120
        or frontier_effect.get("n_counterbalance_blocks") != 20
        or frontier_effect.get("bootstrap_ci", {}).get("confidence")
        != DEFAULT_CONFIDENCE
        or same_effect.get("n_paired_rounds") != 120
        or same_effect.get("n_counterbalance_blocks") != 20
        or same_effect.get("bootstrap_ci", {}).get("confidence")
        != DEFAULT_CONFIDENCE
    ):
        raise FrontierError("latency artifact lacks frontier timing gate")
    recomputed_frontier_effect = _timing_effect(
        timing_values["J1_k"],
        timing_values["VPM_m"],
        orders,
        left_label="J1_k",
        reference_label="VPM_m",
        seed=DEFAULT_SEED,
        label=f"frontier-latency:J1-{k}-vs-VPM-{m}",
    )
    recomputed_same_effect = _timing_effect(
        timing_values["J1_k"],
        timing_values["VPM_k"],
        orders,
        left_label="J1_k",
        reference_label="VPM_k",
        seed=DEFAULT_SEED,
        label=f"same-nfe-latency:J1-{k}-vs-VPM-{k}",
    )
    if (
        dict(frontier_effect) != recomputed_frontier_effect
        or dict(same_effect) != recomputed_same_effect
    ):
        raise FrontierError("latency paired effects do not recompute from rounds")
    recomputed_timing_gate = _timing_gate_from_effect(
        recomputed_frontier_effect,
        left_p95=float(recomputed_summaries["J1_k"]["p95"]),
        reference_p95=float(recomputed_summaries["VPM_m"]["p95"]),
    )
    recomputed_timing_pass = recomputed_timing_gate["passed"]
    if (
        dict(timing_gate) != recomputed_timing_gate
    ):
        raise FrontierError("latency timing gate does not recompute")
    same_ci = recomputed_same_effect["bootstrap_ci"]
    expected_same_overhead = {
        "comparison": f"J1@{k} vs VPM@{k}",
        "definition": "positive relative_overhead means J1 is slower",
        "relative_overhead": -float(
            recomputed_same_effect["relative_improvement"]
        ),
        "relative_overhead_percent": (
            -100.0 * float(recomputed_same_effect["relative_improvement"])
        ),
        "bootstrap_ci": {
            "confidence": same_ci["confidence"],
            "low": -float(same_ci["high"]),
            "high": -float(same_ci["low"]),
        },
        "paired_speed_effect": recomputed_same_effect,
    }
    if latency.get("same_nfe_overhead") != expected_same_overhead:
        raise FrontierError("same-NFE latency overhead does not recompute")
    if latency.get("frontier_acceleration", {}).get("comparison") != (
        f"J1@{k} vs VPM@{m}"
    ):
        raise FrontierError("frontier latency comparison label differs")
    timing_protocol = latency.get("protocol")
    timing_protocol_confirmatory = (
        isinstance(timing_protocol, Mapping)
        and timing_protocol.get("confirmatory_protocol") is True
        and timing_protocol.get("warmup_rounds") == 18
        and timing_protocol.get("timed_rounds") == 120
        and timing_protocol.get("bootstrap_samples")
        == DEFAULT_BOOTSTRAP_SAMPLES
        and timing_protocol.get("confidence") == DEFAULT_CONFIDENCE
        and timing_protocol.get("bootstrap_seed") == DEFAULT_SEED
        and timing_protocol.get("same_process") is True
        and timing_protocol.get("same_B200") is True
        and timing_protocol.get("both_models_resident") is True
        and timing_protocol.get("sampling_namespace") == "lockbox"
        and timing_protocol.get("sampling_id")
        == SAMPLE_ID_OFFSETS["lockbox"]
        and timing_protocol.get("balanced_order_cycle")
        == [list(order) for order in BALANCED_TIMING_ORDERS]
    )
    device = latency.get("device")
    timing_protocol_confirmatory = (
        timing_protocol_confirmatory
        and isinstance(device, Mapping)
        and "B200" in str(device.get("name", "")).upper()
        and device.get("both_models_resident") is True
        and isinstance(device.get("resident_allocated_bytes_before_timing"), int)
        and device["resident_allocated_bytes_before_timing"] > 0
        and isinstance(device.get("peak_allocated_bytes_during_timing"), int)
        and device["peak_allocated_bytes_during_timing"] > 0
    )
    quality_pass = frontier_quality_pass and same_quality_pass
    timing_pass = timing_protocol_confirmatory and recomputed_timing_pass
    confirmatory = (
        selection.get("confirmatory_eligible") is True
        and confirmation.get("confirmatory_evidence") is True
    )
    joint = confirmatory and quality_pass and timing_pass
    return identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_FINAL,
            "created_at_utc": _now(),
            "selection_identity_sha256": selection_id,
            "confirmation_identity_sha256": confirmation["identity_sha256"],
            "latency_identity_sha256": latency["identity_sha256"],
            "lockbox_registration_identity_sha256": lockbox_id,
            "training_git_commit": training_commit,
            "evaluator_git_commit": evaluator_commit,
            "benchmark_git_commit": benchmark_commit,
            "selection_status": selection.get("status"),
            "confirmatory_evidence": confirmatory,
            "quality_gate_passed": quality_pass,
            "frontier_quality_gate_passed": confirmation.get(
                "frontier_quality_gate_passed"
            ),
            "same_nfe_quality_attribution_gate_passed": confirmation.get(
                "same_nfe_quality_attribution_gate_passed"
            ),
            "timing_gate_passed": timing_pass,
            "timing_protocol_confirmatory": timing_protocol_confirmatory,
            "faster_with_better_held_out_reconstruction_demonstrated": joint,
            "status": (
                "PASS"
                if joint
                else (
                    "POSTHOC_EXPLORATORY"
                    if not confirmatory
                    else "NOT_DEMONSTRATED"
                )
            ),
            "quality": confirmation["frontier_quality_comparison"],
            "same_nfe_quality_attribution": confirmation[
                "same_nfe_quality_attribution"
            ],
            "frontier_acceleration": latency["frontier_acceleration"],
            "same_nfe_latency_overhead": latency["same_nfe_overhead"],
            "device": latency.get("device"),
            "endpoint_accounting": latency.get("endpoints"),
            "claim_scope": (
                "paired held-out latent/RGB reconstruction and temporal-"
                "difference errors only; no FVD, perceptual, or diversity "
                "claim; lockbox is held out from controlled fine-tuning and "
                "NFE selection, but shared warm-start pretraining data may "
                "include ABC episodes; clip bootstrap covers sample "
                "uncertainty for one seed-1234 checkpoint pair, not "
                "training-run variance or a general method-level claim; "
                "timing p95 is over 120 counterbalanced repeats of one fixed "
                "lockbox clip, not a latency distribution over clips"
            ),
        }
    )


def _paths(values: Sequence[str]) -> list[Path]:
    paths = [Path(value).expanduser().resolve() for value in values]
    if len(set(paths)) != len(paths):
        raise FrontierError("duplicate JSONL input path")
    return paths


def _command_select(args: argparse.Namespace) -> int:
    if not args.allow_posthoc and (
        args.split != "validation"
        or args.expected_clips != 64
        or args.bootstrap_samples != DEFAULT_BOOTSTRAP_SAMPLES
        or args.confidence != DEFAULT_CONFIDENCE
        or args.seed != DEFAULT_SEED
        or args.lockbox_registration is None
    ):
        raise FrontierError(
            "confirmatory selection requires the pinned 64-clip validation "
            "split, 10,000 bootstrap samples, 95% confidence, seed 1234, "
            "and an already registered fresh lockbox"
        )
    j1_paths, vpm_paths = _paths(args.j1_rows), _paths(args.vpm_rows)
    lockbox_registration = (
        None
        if args.lockbox_registration is None
        else _read_json(
            Path(args.lockbox_registration).expanduser(),
            "lockbox registration",
        )
    )
    lockbox_input = (
        None
        if args.lockbox_registration is None
        else {
            "path": str(Path(args.lockbox_registration).expanduser().resolve()),
            "sha256": _sha256(
                Path(args.lockbox_registration).expanduser().resolve()
            ),
            "bytes": Path(args.lockbox_registration)
            .expanduser()
            .resolve()
            .stat()
            .st_size,
            "identity_sha256": lockbox_registration["identity_sha256"],
        }
    )
    payload = build_selection(
        j1_rows=_read_jsonl(j1_paths, "J1 rows"),
        vpm_rows=_read_jsonl(vpm_paths, "VPM rows"),
        split=args.split,
        expected_clips=args.expected_clips,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
        allow_posthoc=args.allow_posthoc,
        lockbox_registration=lockbox_registration,
        lockbox_input=lockbox_input,
        j1_inputs=_input_records(j1_paths),
        vpm_inputs=_input_records(vpm_paths),
    )
    _exclusive_json(Path(args.output).expanduser(), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": args.output,
                "frontier": payload["vpm_non_dominated_nfe_frontier"],
                "selected_pair": payload["selected_pair"],
            },
            sort_keys=True,
        )
    )
    return 0


def _command_confirm(args: argparse.Namespace) -> int:
    if not args.allow_posthoc and args.expected_clips != 128:
        raise FrontierError(
            "confirmatory lockbox requires all 128 immutable clips"
        )
    selection = _read_json(Path(args.selection).expanduser(), "selection")
    j1_paths, vpm_paths = _paths(args.j1_rows), _paths(args.vpm_rows)
    payload = build_confirmation(
        selection=selection,
        j1_rows=_read_jsonl(j1_paths, "J1 lockbox rows"),
        vpm_rows=_read_jsonl(vpm_paths, "VPM lockbox rows"),
        expected_clips=args.expected_clips,
        allow_posthoc=args.allow_posthoc,
        j1_inputs=_input_records(j1_paths),
        vpm_inputs=_input_records(vpm_paths),
    )
    _exclusive_json(Path(args.output).expanduser(), payload)
    print(
        json.dumps(
            {
                "status": (
                    "passed"
                    if payload["quality_gate_passed"]
                    else (
                        "posthoc"
                        if payload["posthoc_exploratory"]
                        else "not_demonstrated"
                    )
                ),
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_finalize(args: argparse.Namespace) -> int:
    payload = build_final_report(
        selection=_read_json(Path(args.selection).expanduser(), "selection"),
        confirmation=_read_json(
            Path(args.confirmation).expanduser(), "confirmation"
        ),
        latency=_read_json(Path(args.latency).expanduser(), "latency"),
    )
    _exclusive_json(Path(args.output).expanduser(), payload)
    print(json.dumps({"status": payload["status"], "output": args.output}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--j1-rows", nargs="+", required=True)
    select.add_argument("--vpm-rows", nargs="+", required=True)
    select.add_argument("--split", choices=("validation", "test"), default="validation")
    select.add_argument("--expected-clips", type=int, default=64)
    select.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    select.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    select.add_argument("--seed", type=int, default=DEFAULT_SEED)
    select.add_argument("--allow-posthoc", action="store_true")
    select.add_argument(
        "--lockbox-registration",
        help=(
            "already-registered fresh lockbox; required before confirmatory "
            "validation selection"
        ),
    )
    select.add_argument("--output", required=True)
    select.set_defaults(handler=_command_select)

    confirm = commands.add_parser("confirm")
    confirm.add_argument("--selection", required=True)
    confirm.add_argument("--j1-rows", nargs="+", required=True)
    confirm.add_argument("--vpm-rows", nargs="+", required=True)
    confirm.add_argument("--expected-clips", type=int, default=128)
    confirm.add_argument("--allow-posthoc", action="store_true")
    confirm.add_argument("--output", required=True)
    confirm.set_defaults(handler=_command_confirm)

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--selection", required=True)
    finalize.add_argument("--confirmation", required=True)
    finalize.add_argument("--latency", required=True)
    finalize.add_argument("--output", required=True)
    finalize.set_defaults(handler=_command_finalize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FrontierError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
