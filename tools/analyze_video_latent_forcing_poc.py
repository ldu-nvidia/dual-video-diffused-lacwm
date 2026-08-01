#!/usr/bin/env python3
"""Fail-closed paired gate analysis for the Video Latent Forcing POC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "docs/experiments/VIDEO_LATENT_FORCING_POC_PROTOCOL.md"
BUILDER_PATH = REPO_ROOT / "tools/build_video_latent_forcing_droid.py"
GATE_SCHEMA = "video-latent-forcing-poc-gate-v1"
PHASE2_GATE_SCHEMA = "video-latent-forcing-poc-phase2-gate-v1"
EVALUATION_SCHEMA = "video-latent-forcing-poc-evaluation-v1"
RUN_SCHEMA = "video-latent-forcing-poc-run-v1"
CHECKPOINT_SCHEMA = "video-latent-forcing-poc-checkpoint-v1"
CONTROLS = ("autonomous", "off", "shuffled", "oracle_clean", "context_shuffled")
NFE_GRID = (1, 2, 4, 8, 12, 20, 25)
GATE_NFE_GRID = (1, 2, 4, 8, 12)
DUAL_ARMS = ("B0", "A1", "L1")
DUAL_CONTROLS = ("autonomous", "off", "shuffled", "oracle_clean")
DUAL_CHECKPOINT_UPDATES = (500, 1_000, 2_000, 5_000, 10_000, 16_000, 20_000)
TRAINING_EFFICIENCY_UPDATES = (500, 1_000, 2_000, 5_000, 10_000, 16_000)
FROZEN_EVALUATION_WORLD_SIZE = 8
FROZEN_EVALUATION_BATCH_SIZE = 8
B0_FRONTIER = ((0, 1), (0, 2), (0, 4), (0, 8), (0, 12), (0, 20), (0, 25), (0, 50))
DUAL_FRONTIER = ((1, 1), (2, 2), (4, 4), (6, 6), (10, 10), (25, 25))
PRIMARY_NFE = {"B0": (0, 50), "A1": (25, 25), "L1": (25, 25)}
PRIMARY_CONTROL = {"B0": "off", "A1": "off", "L1": "autonomous"}
PAIRED_PRIMARY_METRICS = (
    "lpips_alex_frame",
    "lpips_alex_temporal_difference",
    "temporal_difference_mse",
)
DISTRIBUTION_PRIMARY_METRIC = "r3d18_frechet"
ALL_PRIMARY_METRICS = (DISTRIBUTION_PRIMARY_METRIC, *PAIRED_PRIMARY_METRICS)
MAX_PRIMARY_REGRESSION = 0.01
MIN_FRECHET_IMPROVEMENT = 0.10
CLIPS = 890
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_BASE_SEED = 20260801
EMA_SCHEDULE = "min(target_decay,(1+completed_updates)/(10+completed_updates))-v1"
INITIALIZATION_SCHEME = "latent-forcing-zero-adaln-and-output-heads-v1"
APPROVED_ROOTS = (Path("/lustre"), Path("/mnt/data1"), Path("/mnt/data2"))
ELIGIBLE_INVENTORY_SHA256 = (
    "3bc6f2c06abe74f1a60ddc4f9a44ce734fb8fa85f9ec94ac99e7bcc954993651"
)
VALIDATION_EPISODE_IDS_SHA256 = (
    "58f1a863a7be8f273212030c902c568b32ed75df5aa79993a8aa5c1a7a0252e6"
)


class GateError(RuntimeError):
    """Evidence is incomplete, inconsistent, mutable, or outside the protocol."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def gate_decision_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_json(
        {key: value for key, value in payload.items() if key != "decision_sha256"}
    )


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise GateError(f"required evidence file is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def load_json(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object")
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_output_path(path: str | Path, evidence_root: Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise GateError(f"gate output already exists: {output}")
    if _is_relative_to(output, evidence_root) or _is_relative_to(output, REPO_ROOT):
        raise GateError("gate output must be outside both evidence and Git roots")
    if not any(_is_relative_to(output, root) for root in APPROVED_ROOTS):
        raise GateError("gate output must be under /lustre, /mnt/data1, or /mnt/data2")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def atomic_publish_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise GateError(f"gate output appeared during publication: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def git_record() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _verify_file_record(record: Any, label: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise GateError(f"{label} file record is missing")
    path = record.get("path")
    digest = record.get("sha256")
    if not isinstance(path, str) or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
        raise GateError(f"{label} file record is malformed")
    actual = file_record(path)
    if actual["sha256"] != digest or (
        record.get("bytes") is not None and record.get("bytes") != actual["bytes"]
    ):
        raise GateError(f"{label} file record changed")
    return actual


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"invalid metric row {line_number}") from exc
            if not isinstance(row, dict):
                raise GateError(f"metric row {line_number} is not an object")
            rows.append(row)
    return rows


def _validation_manifest_identity(path: Path) -> tuple[list[str], list[int]]:
    clip_ids: list[str] = []
    episode_ids: list[int] = []
    seen_episodes: set[int] = set()
    for line_number, row in enumerate(_read_rows(path), 1):
        clip_id = row.get("clip_id")
        episode_index = row.get("episode_index")
        if (
            row.get("split") != "val"
            or not isinstance(clip_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", clip_id)
            or isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
        ):
            raise GateError(f"validation manifest identity is malformed at row {line_number}")
        clip_ids.append(clip_id)
        if episode_index not in seen_episodes:
            seen_episodes.add(episode_index)
            episode_ids.append(episode_index)
    if len(clip_ids) != CLIPS or len(set(clip_ids)) != CLIPS or len(episode_ids) != CLIPS:
        raise GateError("validation manifest must contain 890 unique one-clip episodes")
    if sha256_json(episode_ids) != VALIDATION_EPISODE_IDS_SHA256:
        raise GateError("validation manifest episode population differs from preregistration")
    return clip_ids, episode_ids


def _finite_number(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GateError(f"metric {name} must be finite for every row")
    return float(value)


def _bootstrap_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_BASE_SEED}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def paired_bootstrap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    label: str,
    statistic: Callable[[np.ndarray, np.ndarray], float],
) -> dict[str, Any]:
    if first.shape != (CLIPS,) or second.shape != (CLIPS,):
        raise GateError(f"paired bootstrap requires exactly {CLIPS} aligned clips")
    estimate = float(statistic(first, second))
    if not math.isfinite(estimate):
        raise GateError(f"nonfinite point statistic: {label}")
    generator = np.random.default_rng(_bootstrap_seed(label))
    values = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    chunk = 1_000
    for start in range(0, BOOTSTRAP_SAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_SAMPLES - start)
        indexes = generator.integers(0, CLIPS, size=(count, CLIPS), endpoint=False)
        first_means = first[indexes].mean(axis=1)
        second_means = second[indexes].mean(axis=1)
        if statistic is relative_improvement:
            with np.errstate(divide="ignore", invalid="ignore"):
                values[start : start + count] = (
                    second_means - first_means
                ) / second_means
        elif statistic is mean_first_minus_second:
            values[start : start + count] = first_means - second_means
        else:
            for offset in range(count):
                values[start + offset] = statistic(
                    first[indexes[offset]], second[indexes[offset]]
                )
    if not np.isfinite(values).all():
        raise GateError(f"nonfinite bootstrap statistic: {label}")
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) * 0.5
    low, high = np.quantile(values, (tail, 1.0 - tail), method="linear")
    return {
        "label": label,
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "seed": _bootstrap_seed(label),
    }


def relative_improvement(generated: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(reference.mean())
    if denominator <= 0:
        return float("nan")
    return float((denominator - generated.mean()) / denominator)


def mean_first_minus_second(first: np.ndarray, second: np.ndarray) -> float:
    return float(first.mean() - second.mean())


def _gate_cell(
    nfe: int,
    values: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    generated = values["autonomous"]
    checks: list[dict[str, Any]] = []

    def add(
        identifier: str,
        estimate: float | None,
        rule: str,
        passed: bool,
        **extra: Any,
    ) -> None:
        finite_estimate = (
            float(estimate)
            if estimate is not None and math.isfinite(float(estimate))
            else None
        )
        checks.append(
            {
                "id": identifier,
                "estimate": finite_estimate,
                "rule": rule,
                "passed": bool(passed),
                **extra,
            }
        )

    nmse = float(generated["auxiliary_nmse"].mean())
    cosine = float(generated["auxiliary_cosine"].mean())
    add("auxiliary_nmse", nmse, "mean <= 0.50", nmse <= 0.50)
    add("auxiliary_cosine", cosine, "mean >= 0.70", cosine >= 0.70)

    for metric in ("lpips_alex_frame", "temporal_difference_mse"):
        for reference_control in ("off", "shuffled"):
            label = f"nfe={nfe}|{metric}|autonomous_vs_{reference_control}|relative_improvement"
            result = paired_bootstrap(
                generated[metric],
                values[reference_control][metric],
                label=label,
                statistic=relative_improvement,
            )
            passed = result["estimate"] >= 0.05 and result["ci_low"] > 0.0
            add(
                f"{metric}_vs_{reference_control}",
                result["estimate"],
                "relative improvement >= 0.05 and paired CI low > 0",
                passed,
                bootstrap=result,
            )

        off_mean = float(values["off"][metric].mean())
        generated_mean = float(generated[metric].mean())
        oracle_mean = float(values["oracle_clean"][metric].mean())
        denominator = off_mean - oracle_mean
        retention = (off_mean - generated_mean) / denominator if denominator > 0 else None
        add(
            f"{metric}_retained_utility",
            retention,
            "(off-generated)/(off-oracle) >= 0.50 with positive denominator",
            retention is not None and math.isfinite(retention) and retention >= 0.50,
            denominator=denominator,
        )

    nmse_bootstrap = paired_bootstrap(
        generated["auxiliary_nmse"],
        values["context_shuffled"]["auxiliary_nmse"],
        label=f"nfe={nfe}|context_causality|auxiliary_nmse_relative_improvement",
        statistic=relative_improvement,
    )
    add(
        "context_causality_nmse",
        nmse_bootstrap["estimate"],
        "paired relative-improvement CI low > 0",
        nmse_bootstrap["ci_low"] > 0.0,
        bootstrap=nmse_bootstrap,
    )
    cosine_bootstrap = paired_bootstrap(
        generated["auxiliary_cosine"],
        values["context_shuffled"]["auxiliary_cosine"],
        label=f"nfe={nfe}|context_causality|auxiliary_cosine_advantage",
        statistic=mean_first_minus_second,
    )
    add(
        "context_causality_cosine",
        cosine_bootstrap["estimate"],
        "paired autonomous-minus-context cosine CI low > 0",
        cosine_bootstrap["ci_low"] > 0.0,
        bootstrap=cosine_bootstrap,
    )
    return {
        "nfe_pair": [nfe, 0],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _positive_relative_change(candidate: float, reference: float) -> float | None:
    """Return ``(candidate-reference)/reference`` for a positive reference."""
    if not math.isfinite(candidate) or not math.isfinite(reference) or reference <= 0:
        return None
    return (candidate - reference) / reference


def _phase2_comparison(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Evaluate one conservative Phase-2 attribution comparison.

    R3D18-Frechet is distribution-level and therefore receives only the frozen
    one-percent non-regression check.  The other three metrics retain their
    aligned clip values and must include at least one strictly favorable paired
    bootstrap interval while none regresses by more than one percent.
    """

    checks: list[dict[str, Any]] = []
    significant: list[str] = []
    candidate_means = candidate.get("means")
    reference_means = reference.get("means")
    candidate_paired = candidate.get("paired")
    reference_paired = reference.get("paired")
    if not all(
        isinstance(value, Mapping)
        for value in (candidate_means, reference_means, candidate_paired, reference_paired)
    ):
        raise GateError(f"{label} comparison evidence is malformed")

    for metric in PAIRED_PRIMARY_METRICS:
        candidate_values = np.asarray(candidate_paired.get(metric), dtype=np.float64)
        reference_values = np.asarray(reference_paired.get(metric), dtype=np.float64)
        result = paired_bootstrap(
            candidate_values,
            reference_values,
            label=f"phase2|{label}|{metric}|candidate_minus_reference",
            statistic=mean_first_minus_second,
        )
        candidate_mean = float(candidate_means.get(metric, float("nan")))
        reference_mean = float(reference_means.get(metric, float("nan")))
        regression = _positive_relative_change(candidate_mean, reference_mean)
        nonregression = regression is not None and regression <= MAX_PRIMARY_REGRESSION
        favorable = result["ci_high"] < 0.0
        if favorable:
            significant.append(metric)
        checks.append(
            {
                "id": f"{metric}_paired_attribution",
                "candidate_mean": candidate_mean,
                "reference_mean": reference_mean,
                "relative_regression": regression,
                "rule": (
                    "paired candidate-minus-reference CI high < 0 for attribution; "
                    "relative regression <= 0.01 for non-regression"
                ),
                "favorable_ci": favorable,
                "nonregression_passed": nonregression,
                "passed": nonregression,
                "bootstrap": result,
            }
        )

    candidate_frechet = float(candidate_means.get(DISTRIBUTION_PRIMARY_METRIC, float("nan")))
    reference_frechet = float(reference_means.get(DISTRIBUTION_PRIMARY_METRIC, float("nan")))
    frechet_regression = _positive_relative_change(candidate_frechet, reference_frechet)
    frechet_passed = (
        frechet_regression is not None and frechet_regression <= MAX_PRIMARY_REGRESSION
    )
    checks.append(
        {
            "id": "r3d18_frechet_nonregression",
            "candidate": candidate_frechet,
            "reference": reference_frechet,
            "relative_regression": frechet_regression,
            "rule": "distribution-level relative regression <= 0.01; reference > 0",
            "passed": frechet_passed,
        }
    )
    return {
        "label": label,
        "passed": bool(significant) and all(check["passed"] for check in checks),
        "significantly_improved_paired_metrics": significant,
        "rule": (
            "at least one paired primary metric has CI high < 0 and every paired "
            "metric plus distribution-level R3D18-Frechet regresses by at most 1%"
        ),
        "checks": checks,
    }


def phase2_gate_decision(
    cells: Mapping[tuple[str, int, str], Mapping[str, Any]],
    checkpoint_wall_seconds: Mapping[tuple[str, int], float],
) -> dict[str, Any]:
    """Compute the exact frozen one-seed Phase-2 decision from audited cells."""

    def cell(arm: str, update: int, control: str) -> Mapping[str, Any]:
        try:
            value = cells[(arm, update, control)]
        except KeyError as exc:
            raise GateError(f"missing Phase-2 gate cell: {arm}@{update}/{control}") from exc
        means = value.get("means")
        paired = value.get("paired")
        if not isinstance(means, Mapping) or not isinstance(paired, Mapping):
            raise GateError(f"malformed Phase-2 gate cell: {arm}@{update}/{control}")
        for metric in ALL_PRIMARY_METRICS:
            raw = means.get(metric)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(raw)
                or raw < 0
            ):
                raise GateError(f"invalid Phase-2 mean: {arm}@{update}/{control}/{metric}")
        for metric in PAIRED_PRIMARY_METRICS:
            values = np.asarray(paired.get(metric), dtype=np.float64)
            if values.shape != (CLIPS,) or not np.isfinite(values).all():
                raise GateError(
                    f"Phase-2 paired metric must contain {CLIPS} finite clips: "
                    f"{arm}@{update}/{control}/{metric}"
                )
        return value

    baseline = cell("B0", 20_000, "off")
    baseline_means = baseline["means"]
    baseline_wall = checkpoint_wall_seconds.get(("B0", 20_000))
    if (
        isinstance(baseline_wall, bool)
        or not isinstance(baseline_wall, (int, float))
        or not math.isfinite(baseline_wall)
        or baseline_wall <= 0
    ):
        raise GateError("B0 update-20000 checkpoint wall time must be positive and finite")

    reach_candidates: list[dict[str, Any]] = []
    selected_reach: dict[str, Any] | None = None
    for update in TRAINING_EFFICIENCY_UPDATES:
        candidate = cell("L1", update, "autonomous")
        candidate_wall = checkpoint_wall_seconds.get(("L1", update))
        if (
            isinstance(candidate_wall, bool)
            or not isinstance(candidate_wall, (int, float))
            or not math.isfinite(candidate_wall)
            or candidate_wall <= 0
        ):
            raise GateError(f"L1 update-{update} checkpoint wall time must be positive and finite")
        metric_checks = {
            metric: {
                "candidate": float(candidate["means"][metric]),
                "reference": float(baseline_means[metric]),
                "passed": float(candidate["means"][metric])
                <= float(baseline_means[metric]),
            }
            for metric in ALL_PRIMARY_METRICS
        }
        wall_passed = float(candidate_wall) < float(baseline_wall)
        candidate_record = {
            "update": update,
            "nfe_pair": list(PRIMARY_NFE["L1"]),
            "control": "autonomous",
            "checkpoint_wall_seconds": float(candidate_wall),
            "reference_b0_update": 20_000,
            "reference_b0_wall_seconds": float(baseline_wall),
            "metric_checks": metric_checks,
            "wall_time_passed": wall_passed,
            "passed": wall_passed
            and all(check["passed"] for check in metric_checks.values()),
        }
        reach_candidates.append(candidate_record)
        if selected_reach is None and candidate_record["passed"]:
            selected_reach = candidate_record
    training_efficiency = {
        "passed": selected_reach is not None,
        "rule": (
            "earliest L1 autonomous 25+25 checkpoint at update <=16000 has every "
            "primary metric <= B0 off 0+50 at update 20000 and strictly lower "
            "cumulative optimizer wall time"
        ),
        "selected_reach": selected_reach,
        "candidates": reach_candidates,
    }

    l1_endpoint = cell("L1", 20_000, "autonomous")
    b0_endpoint = baseline
    b0_frechet = float(b0_endpoint["means"][DISTRIBUTION_PRIMARY_METRIC])
    l1_frechet = float(l1_endpoint["means"][DISTRIBUTION_PRIMARY_METRIC])
    frechet_improvement = (
        (b0_frechet - l1_frechet) / b0_frechet if b0_frechet > 0 else None
    )
    same_update_sample_nfe_checks: list[dict[str, Any]] = [
        {
            "id": "r3d18_frechet_improvement",
            "candidate": l1_frechet,
            "reference": b0_frechet,
            "relative_improvement": frechet_improvement,
            "rule": "relative improvement >= 0.10; reference > 0",
            "passed": frechet_improvement is not None
            and frechet_improvement >= MIN_FRECHET_IMPROVEMENT,
        }
    ]
    for metric in PAIRED_PRIMARY_METRICS:
        candidate_mean = float(l1_endpoint["means"][metric])
        reference_mean = float(b0_endpoint["means"][metric])
        regression = _positive_relative_change(candidate_mean, reference_mean)
        same_update_sample_nfe_checks.append(
            {
                "id": f"{metric}_nonregression",
                "candidate": candidate_mean,
                "reference": reference_mean,
                "relative_regression": regression,
                "rule": "relative regression <= 0.01; reference > 0",
                "passed": regression is not None
                and regression <= MAX_PRIMARY_REGRESSION,
            }
        )
    same_update_sample_nfe_budget = {
        "passed": all(check["passed"] for check in same_update_sample_nfe_checks),
        "comparison": "L1@20000 autonomous 25+25 vs B0@20000 off 0+50",
        "matched_budget": {
            "optimizer_updates": 20_000,
            "training_examples": 20_000 * 256,
            "inference_transformer_calls": 50,
        },
        "not_matched": (
            "training FLOPs are not equal because L1 trains an auxiliary head; "
            "this is a same-update/sample/NFE quality comparison, not a speed claim"
        ),
        "checks": same_update_sample_nfe_checks,
    }

    attribution = {
        "l1_vs_a1": _phase2_comparison(
            l1_endpoint,
            cell("A1", 20_000, "off"),
            label="L1@20000_autonomous_vs_A1@20000_off",
        ),
        "autonomous_vs_off": _phase2_comparison(
            l1_endpoint,
            cell("L1", 20_000, "off"),
            label="L1@20000_autonomous_vs_off",
        ),
        "autonomous_vs_shuffled": _phase2_comparison(
            l1_endpoint,
            cell("L1", 20_000, "shuffled"),
            label="L1@20000_autonomous_vs_shuffled",
        ),
    }
    attribution["passed"] = all(
        value["passed"] for key, value in attribution.items() if key != "passed"
    )
    criteria = {
        "training_efficiency": training_efficiency,
        "same_update_sample_nfe_budget": same_update_sample_nfe_budget,
        "attribution": attribution,
    }
    return {
        "passed": all(value["passed"] for value in criteria.values()),
        "criteria": criteria,
        "selected_arm": "L1",
        "selected_checkpoint_update": 20_000,
        "selected_nfe_pair": list(PRIMARY_NFE["L1"]),
        "selected_control": "autonomous",
    }


def _validate_phase2_checkpoint_walls(
    checkpoint_wall_seconds: Mapping[tuple[str, int], float],
) -> None:
    """Require positive, strictly increasing cumulative wall time for each arm."""
    for arm in DUAL_ARMS:
        walls: list[float] = []
        for update in DUAL_CHECKPOINT_UPDATES:
            raw = checkpoint_wall_seconds.get((arm, update))
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(raw)
                or raw <= 0
            ):
                raise GateError(f"{arm} update-{update} wall time is not positive and finite")
            walls.append(float(raw))
        if any(
            later <= earlier
            for earlier, later in zip(walls[:-1], walls[1:], strict=True)
        ):
            raise GateError(f"{arm} checkpoint cumulative wall time is not strictly increasing")


def analyze_phase1_evaluation(evaluation_root: str | Path) -> dict[str, Any]:
    """Recompute the complete Phase-1 decision from immutable raw evidence."""
    root = Path(evaluation_root).expanduser().resolve()
    if not root.is_dir():
        raise GateError(f"evaluation root is unavailable: {root}")
    config_path = root / "resolved_config.json"
    summary_path = root / "summary.json"
    metrics_path = root / "per_clip_metrics.jsonl"
    before = {
        "resolved_config": file_record(config_path),
        "summary": file_record(summary_path),
        "per_clip_metrics": file_record(metrics_path),
        "protocol": file_record(PROTOCOL_PATH),
    }
    config = load_json(config_path, "evaluation resolved config")
    summary = load_json(summary_path, "evaluation summary")
    source = config.get("source")
    expected_pairs = [[nfe, 0] for nfe in NFE_GRID]
    quality_config = config.get("publication_quality_metrics")
    weights = config.get("weights")
    if (
        config.get("schema") != EVALUATION_SCHEMA
        or config.get("arm") != "phase1"
        or config.get("split") != "val"
        or config.get("checkpoint_update") != 5_000
        or config.get("nfe_pairs") != expected_pairs
        or config.get("controls") != list(CONTROLS)
        or config.get("seed") != BOOTSTRAP_BASE_SEED
        or config.get("quality_metric_suite_complete") is not True
        or not isinstance(quality_config, Mapping)
        or quality_config.get("enabled") is not True
        or not isinstance(quality_config.get("provenance"), Mapping)
        or not isinstance(weights, Mapping)
        or weights.get("kind") != "ema"
        or weights.get("decay") != 0.9999
        or weights.get("schedule") != EMA_SCHEDULE
        or weights.get("num_updates") != 5_000
        or not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or summary.get("schema") != EVALUATION_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("arm") != "phase1"
        or summary.get("split") != "val"
        or summary.get("record_count") != CLIPS * len(CONTROLS) * len(NFE_GRID)
        or summary.get("reported_weight_source") != "ema"
        or summary.get("ema_decay") != 0.9999
        or summary.get("ema_schedule") != EMA_SCHEDULE
        or summary.get("ema_updates") != 5_000
        or summary.get("quality_metric_suite_complete") is not True
        or summary.get("quality_metric_provenance")
        != quality_config.get("provenance")
    ):
        raise GateError("evaluation does not match the frozen update-5000 Phase-1 frontier")
    quality_audit = _validate_quality_provenance(quality_config["provenance"])
    config_checkpoint_record = _verify_file_record(config.get("checkpoint"), "config checkpoint")
    summary_checkpoint_record = _verify_file_record(summary.get("checkpoint"), "summary checkpoint")
    if config_checkpoint_record != summary_checkpoint_record:
        raise GateError("evaluation config/summary checkpoint records differ")
    training_config_record = _verify_file_record(
        config.get("training_config"), "training resolved config"
    )
    training_config = load_json(
        training_config_record["path"], "training resolved config"
    )
    training_source = training_config.get("source")
    training_ema = training_config.get("ema")
    training_model = training_config.get("model")
    if (
        training_config.get("schema") != RUN_SCHEMA
        or training_config.get("command") != "train"
        or training_config.get("arm") != "phase1"
        or training_config.get("seed") != 1234
        or training_config.get("updates") != 5_000
        or training_config.get("checkpoint_updates") != [500, 1_000, 2_000, 5_000]
        or training_config.get("model") != config.get("model")
        or not isinstance(training_model, Mapping)
        or training_model.get("initialization") != INITIALIZATION_SCHEME
        or not isinstance(training_source, Mapping)
        or training_source != source
        or not isinstance(training_ema, Mapping)
        or training_ema.get("decay") != 0.9999
        or training_ema.get("schedule") != EMA_SCHEDULE
        or training_ema.get("short_run_initialization_bias_corrected") is not True
        or training_ema.get("reported_samples_use") is not True
    ):
        raise GateError("checkpoint training configuration violates the frozen Phase-1 contract")
    try:
        checkpoint = torch.load(
            config_checkpoint_record["path"], map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise GateError(f"cannot load Phase-1 checkpoint metadata: {exc}") from exc
    checkpoint_ema = checkpoint.get("ema")
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("arm") != "phase1"
        or checkpoint.get("completed_updates") != 5_000
        or checkpoint.get("config_sha256") != sha256_json(training_config)
        or checkpoint.get("model_config") != config.get("model")
        or not isinstance(checkpoint_ema, Mapping)
        or checkpoint_ema.get("decay") != 0.9999
        or checkpoint_ema.get("schedule") != EMA_SCHEDULE
        or checkpoint_ema.get("num_updates") != 5_000
        or not isinstance(checkpoint_ema.get("shadow"), Mapping)
    ):
        raise GateError("checkpoint payload is not the frozen update-5000 Phase-1 EMA model")
    del checkpoint
    if _verify_file_record(summary.get("per_clip_metrics"), "per-clip metrics") != before[
        "per_clip_metrics"
    ]:
        raise GateError("summary per-clip record does not identify the gate input")
    manifest = config.get("manifest")
    if not isinstance(manifest, Mapping) or manifest.get("clips") != CLIPS:
        raise GateError("validation manifest does not contain exactly 890 clips")
    manifest_record = _verify_file_record(manifest, "validation manifest")
    provenance_record = _verify_file_record(manifest.get("data_provenance"), "data provenance")
    provenance = load_json(provenance_record["path"], "DROID data provenance")
    split_episode_ids = provenance.get("split_episode_ids_sha256")
    builder_source = provenance.get("builder_source")
    if (
        not isinstance(builder_source, Mapping)
        or builder_source.get("commit") != source.get("commit")
        or builder_source.get("dirty") is not False
        or builder_source.get("builder_tool_sha256") != sha256_file(BUILDER_PATH)
        or provenance.get("split_rank_expression")
        != "sha256('video-latent-forcing-poc-v1:<episode_id>')"
        or provenance.get("clip_start_seed") != 20260801
        or provenance.get("clips_per_episode")
        != {"train": 8, "val": 1, "test": 1}
        or provenance.get("minimum_episode_frames") != 66
        or provenance.get("eligible_episode_count") != 9_780
        or provenance.get("eligible_inventory_sha256") != ELIGIBLE_INVENTORY_SHA256
        or not isinstance(split_episode_ids, Mapping)
        or split_episode_ids.get("val") != VALIDATION_EPISODE_IDS_SHA256
    ):
        raise GateError("DROID validation population differs from preregistration")
    manifest_clip_ids, _ = _validation_manifest_identity(Path(manifest_record["path"]))
    training_manifests = training_config.get("manifests")
    training_validation_manifest = (
        training_manifests.get("validation")
        if isinstance(training_manifests, Mapping)
        else None
    )
    if (
        not isinstance(training_validation_manifest, Mapping)
        or training_validation_manifest.get("clips") != CLIPS
        or _verify_file_record(
            training_validation_manifest, "training validation manifest"
        )
        != manifest_record
        or training_config.get("data_root") != config.get("data_root")
    ):
        raise GateError("evaluation does not use the checkpoint's frozen validation population")
    summary_manifest = summary.get("manifest")
    if (
        not isinstance(summary_manifest, Mapping)
        or summary_manifest.get("clips") != CLIPS
        or _verify_file_record(summary_manifest, "summary validation manifest") != manifest_record
    ):
        raise GateError("evaluation config/summary validation manifests differ")
    feature_entries = summary.get("quality_feature_artifacts")
    if not isinstance(feature_entries, list):
        raise GateError("quality feature artifact inventory is missing")
    expected_feature_keys = {
        (nfe, 0, control) for nfe in NFE_GRID for control in CONTROLS
    }
    feature_records: dict[tuple[int, int, str], dict[str, Any]] = {}
    for feature in feature_entries:
        if not isinstance(feature, Mapping):
            raise GateError("quality feature artifact is malformed")
        key = (
            feature.get("auxiliary_nfe"),
            feature.get("video_nfe"),
            feature.get("control"),
        )
        if key not in expected_feature_keys or key in feature_records:
            raise GateError("quality feature artifact inventory is malformed")
        feature_records[key] = _verify_file_record(
            feature.get("file"), "quality feature artifact"
        )
    if set(feature_records) != expected_feature_keys:
        raise GateError("quality feature artifact inventory is incomplete")

    donor_mapping: dict[str, str] = {}
    for position in range(0, len(manifest_clip_ids), 2):
        left, right = manifest_clip_ids[position : position + 2]
        donor_mapping[left] = right
        donor_mapping[right] = left
    donor_mapping_sha256 = sha256_json(donor_mapping)
    if config.get("shuffle_mapping_sha256") != donor_mapping_sha256:
        raise GateError("evaluation shuffle mapping differs from the manifest-global derangement")

    rows = _read_rows(metrics_path)
    expected_count = CLIPS * len(CONTROLS) * len(NFE_GRID)
    if len(rows) != expected_count:
        raise GateError(f"expected {expected_count} metric rows, found {len(rows)}")
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    group_ids: dict[tuple[int, str], set[str]] = {}
    for row in rows:
        nfe = row.get("auxiliary_nfe")
        control = row.get("control")
        clip_id = row.get("clip_id")
        if (
            isinstance(nfe, bool)
            or not isinstance(nfe, int)
            or nfe not in NFE_GRID
            or isinstance(row.get("video_nfe"), bool)
            or row.get("video_nfe") != 0
            or control not in CONTROLS
            or not isinstance(clip_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", clip_id)
            or row.get("path_model_calls") != nfe
            or row.get("evaluation_auxiliary_generation_calls") != nfe
            or row.get("evaluation_seed") != BOOTSTRAP_BASE_SEED
            or row.get("optimizer_seed") != 1234
            or row.get("ema_decay") != 0.9999
            or row.get("ema_schedule") != EMA_SCHEDULE
            or row.get("ema_updates") != 5_000
            or row.get("teacher_model_calls") != 0
            or row.get("clean_future_used_as_condition") != (control == "oracle_clean")
            or row.get("deployable") != (control != "oracle_clean")
            or row.get("auxiliary_frozen_assertion_executed") is not False
        ):
            raise GateError("per-clip row identity/call/leakage contract failed")
        key = (int(nfe), str(control), clip_id)
        if key in indexed:
            raise GateError("duplicate per-clip NFE/control identity")
        for metric in (
            "lpips_alex_frame",
            "temporal_difference_mse",
            "auxiliary_nmse",
            "auxiliary_cosine",
        ):
            _finite_number(row, metric)
        for field in (
            "phase_boundary_sha256",
            "conditioning_auxiliary_sha256",
            "pre_video_auxiliary_sha256",
            "post_video_auxiliary_sha256",
            "target_auxiliary_sha256",
            "zero_auxiliary_sha256",
            "history_sha256",
            "actions_sha256",
            "initial_video_noise_sha256",
            "initial_auxiliary_noise_sha256",
            "checkpoint_sha256",
            "training_config_sha256",
            "shuffle_mapping_sha256",
        ):
            if not isinstance(row.get(field), str) or not re.fullmatch(
                r"[0-9a-f]{64}", row[field]
            ):
                raise GateError(f"per-clip evidence hash is missing or malformed: {field}")
        if (
            row["checkpoint_sha256"] != config_checkpoint_record["sha256"]
            or row["training_config_sha256"] != training_config_record["sha256"]
            or row["shuffle_mapping_sha256"] != donor_mapping_sha256
            or row["pre_video_auxiliary_sha256"]
            != row["conditioning_auxiliary_sha256"]
            or row["post_video_auxiliary_sha256"]
            != row["conditioning_auxiliary_sha256"]
        ):
            raise GateError("per-clip evidence is not anchored to the evaluated run")
        indexed[key] = row
        group_ids.setdefault((nfe, control), set()).add(clip_id)
    expected_groups = {(nfe, control) for nfe in NFE_GRID for control in CONTROLS}
    if set(group_ids) != expected_groups:
        raise GateError("NFE/control group inventory is incomplete")
    canonical_ids = group_ids[(NFE_GRID[0], CONTROLS[0])]
    if len(canonical_ids) != CLIPS or any(
        group_ids[group] != canonical_ids for group in expected_groups
    ):
        raise GateError("NFE/control groups do not contain identical 890-clip populations")
    if canonical_ids != set(manifest_clip_ids):
        raise GateError("metric clip IDs differ from the hashed validation manifest")
    sorted_ids = sorted(canonical_ids)

    zero_conditioning_hashes: set[str] = set()
    for nfe in NFE_GRID:
        for clip_id in sorted_ids:
            donor_id = donor_mapping[clip_id]
            aligned = [indexed[(nfe, control, clip_id)] for control in CONTROLS[:4]]
            if len({row.get("phase_boundary_sha256") for row in aligned}) != 1:
                raise GateError("aligned Phase-1 controls do not share one generated boundary")
            invariant_fields = (
                "initial_video_noise_sha256",
                "initial_auxiliary_noise_sha256",
                "checkpoint_sha256",
                "training_config_sha256",
                "shuffle_mapping_sha256",
                "target_auxiliary_sha256",
            )
            for field in invariant_fields:
                if len({indexed[(nfe, control, clip_id)].get(field) for control in CONTROLS}) != 1:
                    raise GateError(f"paired control field changed: {field}")
            autonomous = indexed[(nfe, "autonomous", clip_id)]
            off = indexed[(nfe, "off", clip_id)]
            shuffled = indexed[(nfe, "shuffled", clip_id)]
            oracle = indexed[(nfe, "oracle_clean", clip_id)]
            context = indexed[(nfe, "context_shuffled", clip_id)]
            donor_autonomous = indexed[(nfe, "autonomous", donor_id)]
            if (
                autonomous["conditioning_auxiliary_sha256"]
                != autonomous["phase_boundary_sha256"]
                or context["conditioning_auxiliary_sha256"]
                != context["phase_boundary_sha256"]
                or shuffled["conditioning_auxiliary_sha256"]
                != donor_autonomous["phase_boundary_sha256"]
                or off["conditioning_auxiliary_sha256"]
                != off["zero_auxiliary_sha256"]
                or oracle["conditioning_auxiliary_sha256"]
                != oracle["target_auxiliary_sha256"]
            ):
                raise GateError("control conditioning tensor does not match its generated boundary")
            zero_conditioning_hashes.add(off["conditioning_auxiliary_sha256"])
            for regular in (autonomous, off, oracle):
                if (
                    regular.get("conditioning_source_clip_id") != clip_id
                    or regular.get("auxiliary_conditioning_source_clip_id") != clip_id
                    or regular.get("history_action_source_clip_id") != clip_id
                    or regular["history_sha256"] != autonomous["history_sha256"]
                    or regular["actions_sha256"] != autonomous["actions_sha256"]
                ):
                    raise GateError("aligned control source identity is inconsistent")
            if (
                shuffled.get("conditioning_source_clip_id") != donor_id
                or shuffled.get("auxiliary_conditioning_source_clip_id") != donor_id
                or shuffled.get("history_action_source_clip_id") != clip_id
                or shuffled["history_sha256"] != autonomous["history_sha256"]
                or shuffled["actions_sha256"] != autonomous["actions_sha256"]
                or context.get("conditioning_source_clip_id") != donor_id
                or context.get("auxiliary_conditioning_source_clip_id") != clip_id
                or context.get("history_action_source_clip_id") != donor_id
                or context["history_sha256"] != donor_autonomous["history_sha256"]
                or context["actions_sha256"] != donor_autonomous["actions_sha256"]
            ):
                raise GateError("shuffled control does not implement the frozen donor mapping")

    if len(zero_conditioning_hashes) != 1:
        raise GateError("Phase-1 off control is not one stable all-zero auxiliary tensor")
    for clip_id in sorted_ids:
        for control in CONTROLS:
            across_nfe = [indexed[(nfe, control, clip_id)] for nfe in NFE_GRID]
            for field in (
                "initial_video_noise_sha256",
                "initial_auxiliary_noise_sha256",
                "checkpoint_sha256",
                "training_config_sha256",
                "shuffle_mapping_sha256",
                "history_sha256",
                "actions_sha256",
                "target_auxiliary_sha256",
                "zero_auxiliary_sha256",
            ):
                if len({row[field] for row in across_nfe}) != 1:
                    raise GateError(f"frontier changed frozen paired input field: {field}")

    metrics = (
        "lpips_alex_frame",
        "temporal_difference_mse",
        "auxiliary_nmse",
        "auxiliary_cosine",
    )
    criteria: dict[str, Any] = {}
    for nfe in GATE_NFE_GRID:
        values = {
            control: {
                metric: np.asarray(
                    [_finite_number(indexed[(nfe, control, clip_id)], metric) for clip_id in sorted_ids],
                    dtype=np.float64,
                )
                for metric in metrics
            }
            for control in CONTROLS
        }
        criteria[str(nfe)] = _gate_cell(nfe, values)
    passing = [nfe for nfe in GATE_NFE_GRID if criteria[str(nfe)]["passed"]]
    selected = [passing[0], 0] if passing else None

    checkpoint_record = config_checkpoint_record
    payload: dict[str, Any] = {
        "schema": GATE_SCHEMA,
        "phase": "phase1",
        "arm": "phase1",
        "status": "pass" if selected is not None else "fail",
        "frozen": True,
        "validation_only": True,
        "phase1_gate_passed": selected is not None,
        "source_commit": source["commit"],
        "protocol": before["protocol"],
        "checkpoint": checkpoint_record,
        "training_config": training_config_record,
        "evaluation": {
            "root": str(root),
            "resolved_config": before["resolved_config"],
            "summary": before["summary"],
            "per_clip_metrics": before["per_clip_metrics"],
            "validation_manifest": manifest_record,
            "quality_feature_artifacts": summary.get("quality_feature_artifacts", []),
            "quality_provenance": quality_audit,
        },
        "selected_nfe_pair": selected,
        "selection_rule": "smallest passing NFE among [1,2,4,8,12] at update 5000",
        "bootstrap": {
            "method": "paired clip-level percentile",
            "samples": BOOTSTRAP_SAMPLES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "base_seed": BOOTSTRAP_BASE_SEED,
            "seed_derivation": "first63bits(sha256(base_seed + NUL + statistic_label))",
            "unit": "immutable clip_id",
        },
        "criteria": criteria,
        "validation_clip_count": CLIPS,
        "validation_clip_ids_sha256": sha256_json(sorted_ids),
        "shuffle_mapping_sha256": donor_mapping_sha256,
        "protected_test_accessed": False,
    }
    payload["decision_sha256"] = gate_decision_sha256(payload)

    after = {
        "resolved_config": file_record(config_path),
        "summary": file_record(summary_path),
        "per_clip_metrics": file_record(metrics_path),
        "protocol": file_record(PROTOCOL_PATH),
        "checkpoint": file_record(checkpoint_record["path"]),
        "training_config": file_record(training_config_record["path"]),
        "validation_manifest": file_record(manifest_record["path"]),
        "data_provenance": file_record(provenance_record["path"]),
        "quality_feature_artifacts": {
            str(key): file_record(record["path"])
            for key, record in sorted(feature_records.items())
        },
        "quality_weights": {
            key: file_record(record["path"])
            for key, record in sorted(quality_audit["weights"].items())
        },
    }
    before_complete = {
        **before,
        "checkpoint": checkpoint_record,
        "training_config": training_config_record,
        "validation_manifest": manifest_record,
        "data_provenance": provenance_record,
        "quality_feature_artifacts": {
            str(key): record for key, record in sorted(feature_records.items())
        },
        "quality_weights": quality_audit["weights"],
    }
    if after != before_complete:
        raise GateError("gate evidence changed while it was being analyzed")
    current_source = git_record()
    if current_source["dirty"] or current_source["commit"] != source["commit"]:
        raise GateError("analyzer source is dirty or differs from evaluated source")
    return payload


def _expected_phase2_pairs(arm: str) -> tuple[tuple[int, int], ...]:
    if arm == "B0":
        return B0_FRONTIER
    if arm in {"A1", "L1"}:
        return DUAL_FRONTIER
    raise GateError(f"unsupported Phase-2 arm: {arm}")


def _expected_phase2_controls(arm: str) -> tuple[str, ...]:
    return ("off",) if arm == "B0" else DUAL_CONTROLS


def _load_checkpoint_metadata(
    record: Mapping[str, Any],
    *,
    arm: str,
    update: int,
    training_config: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    try:
        try:
            checkpoint = torch.load(
                record["path"], map_location="cpu", weights_only=False, mmap=True
            )
        except TypeError:  # pragma: no cover - older supported torch fallback
            checkpoint = torch.load(record["path"], map_location="cpu", weights_only=False)
    except Exception as exc:
        raise GateError(f"cannot load {arm} update-{update} checkpoint metadata: {exc}") from exc
    ema = checkpoint.get("ema")
    wall = checkpoint.get("cumulative_optimizer_wall_seconds")
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("arm") != arm
        or checkpoint.get("completed_updates") != update
        or checkpoint.get("config_sha256") != sha256_json(training_config)
        or checkpoint.get("model_config") != training_config.get("model")
        or not isinstance(ema, Mapping)
        or ema.get("decay") != 0.9999
        or ema.get("schedule") != EMA_SCHEDULE
        or ema.get("num_updates") != update
        or not isinstance(ema.get("shadow"), Mapping)
        or isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(wall)
        or wall <= 0
    ):
        raise GateError(f"{arm} update-{update} checkpoint violates the frozen contract")
    metadata = {
        "schema": checkpoint["schema"],
        "arm": checkpoint["arm"],
        "completed_updates": checkpoint["completed_updates"],
        "model_config": checkpoint["model_config"],
        "config_sha256": checkpoint["config_sha256"],
        "ema": {
            "decay": ema["decay"],
            "schedule": ema["schedule"],
            "num_updates": ema["num_updates"],
            "shadow_keys_sha256": sha256_json(sorted(str(key) for key in ema["shadow"])),
        },
        "cumulative_optimizer_wall_seconds": float(wall),
    }
    del checkpoint
    return metadata, float(wall)


def _validate_phase2_training_config(
    training_config: Mapping[str, Any],
    *,
    arm: str,
    source: Mapping[str, Any],
) -> None:
    expected_schedule = {
        "auxiliary_branch_probability": 0.4,
        "auxiliary_logit_normal_mean": -1.2,
        "auxiliary_logit_normal_std": 1.0,
        "video_logit_normal_mean": -0.4,
        "video_logit_normal_std": 0.8,
        "video_low_time_replacement_probability": 0.1,
        "video_low_time_interval": [0.0, 0.5],
        "video_branch_auxiliary_time_interval": [0.75, 1.0],
        "auxiliary_loss_coefficient": 0.333,
        "loss_mask_normalization": "unchanged_global_batch",
    }
    expected_optimizer = {
        "name": "AdamW",
        "learning_rate": 5e-5,
        "warmup_updates": 500,
        "after_warmup": "constant",
        "betas": [0.9, 0.95],
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
    }
    model = training_config.get("model")
    ema = training_config.get("ema")
    if (
        training_config.get("schema") != RUN_SCHEMA
        or training_config.get("command") != "train"
        or training_config.get("arm") != arm
        or training_config.get("seed") != 1234
        or training_config.get("updates") != 20_000
        or training_config.get("checkpoint_updates") != list(DUAL_CHECKPOINT_UPDATES)
        or training_config.get("global_batch_size") != 256
        or training_config.get("world_size") != 8
        or training_config.get("local_optimizer_batch_size") != 32
        or training_config.get("dtype") != "bfloat16"
        or training_config.get("clock_convention")
        != "clean-time: t=0 noise, t=1 clean; velocity=clean-noise"
        or training_config.get("clean_time_epsilon") != 0.05
        or training_config.get("optimizer") != expected_optimizer
        or training_config.get("schedule") != expected_schedule
        or training_config.get("source") != source
        or not isinstance(model, Mapping)
        or model.get("initialization") != INITIALIZATION_SCHEME
        or model.get("parameter_matched_video_only") != (arm == "B0")
        or not isinstance(ema, Mapping)
        or ema.get("decay") != 0.9999
        or ema.get("schedule") != EMA_SCHEDULE
        or ema.get("reported_samples_use") is not True
        or ema.get("short_run_initialization_bias_corrected") is not True
    ):
        raise GateError(f"{arm} training configuration violates the frozen Phase-2 contract")
    accumulation = training_config.get("gradient_accumulation_steps")
    microbatch = training_config.get("micro_batch_size_per_rank")
    if (
        isinstance(accumulation, bool)
        or not isinstance(accumulation, int)
        or accumulation < 1
        or isinstance(microbatch, bool)
        or not isinstance(microbatch, int)
        or microbatch < 1
        or accumulation * microbatch != 32
    ):
        raise GateError(f"{arm} does not preserve the frozen global batch through accumulation")


def _phase2_summary_index(
    summary: Mapping[str, Any],
    expected_keys: set[tuple[int, int, str]],
) -> dict[tuple[int, int, str], Mapping[str, Any]]:
    entries = summary.get("summaries")
    if not isinstance(entries, list):
        raise GateError("Phase-2 evaluation summary groups are missing")
    indexed: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise GateError("Phase-2 evaluation summary group is malformed")
        key = (
            entry.get("auxiliary_nfe"),
            entry.get("video_nfe"),
            entry.get("control"),
        )
        if key not in expected_keys or key in indexed or entry.get("clips") != CLIPS:
            raise GateError("Phase-2 evaluation summary group inventory differs")
        if entry.get("total_nfe") != key[0] + key[1]:
            raise GateError("Phase-2 summary total NFE is not the actual call count")
        for metric in ALL_PRIMARY_METRICS:
            if _finite_number(entry, metric) < 0:
                raise GateError(f"Phase-2 summary metric must be nonnegative: {metric}")
        indexed[key] = entry
    if set(indexed) != expected_keys:
        raise GateError("Phase-2 evaluation summary groups are incomplete")
    return indexed


def _phase2_feature_inventory(
    summary: Mapping[str, Any],
    expected_keys: set[tuple[int, int, str]],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    entries = summary.get("quality_feature_artifacts")
    if not isinstance(entries, list):
        raise GateError("Phase-2 quality feature inventory is missing")
    indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise GateError("Phase-2 quality feature record is malformed")
        key = (
            entry.get("auxiliary_nfe"),
            entry.get("video_nfe"),
            entry.get("control"),
        )
        if key not in expected_keys or key in indexed:
            raise GateError("Phase-2 quality feature inventory differs")
        indexed[key] = _verify_file_record(entry.get("file"), "Phase-2 quality features")
    if set(indexed) != expected_keys:
        raise GateError("Phase-2 quality feature inventory is incomplete")
    return indexed


def _validate_pinned_quality_weight(
    record: Any,
    *,
    role: str,
    expected_sha256: str,
    source_url: str,
) -> dict[str, Any]:
    """Verify one extractor weight against the exact preregistered identity."""
    if not isinstance(record, Mapping) or set(record) != {
        "role",
        "path",
        "size_bytes",
        "sha256",
        "expected_sha256",
        "expected_sha256_prefix",
        "source_url",
    }:
        raise GateError(f"{role} provenance record is malformed")
    path = record.get("path")
    size = record.get("size_bytes")
    if (
        record.get("role") != role
        or not isinstance(path, str)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or record.get("sha256") != expected_sha256
        or record.get("expected_sha256") != expected_sha256
        or record.get("expected_sha256_prefix") is not None
        or record.get("source_url") != source_url
    ):
        raise GateError(f"{role} provenance differs from the frozen weight contract")
    actual = file_record(path)
    if actual["sha256"] != expected_sha256 or actual["bytes"] != size:
        raise GateError(f"{role} weight file is missing, changed, or not pinned")
    return actual


def _validate_quality_provenance(provenance: Any) -> dict[str, Any]:
    """Require exact metric code, preprocessing, packages, and weight hashes."""
    from robot_wm.evaluation.video_latent_forcing_quality import (
        ALEXNET_SHA256,
        ALEXNET_URL,
        LPIPS_ALEX_FRAME_METRIC,
        LPIPS_ALEX_TEMPORAL_DIFFERENCE_METRIC,
        LPIPS_LINEAR_SHA256,
        LPIPS_PACKAGE_VERSION,
        R3D18_FRECHET_METRIC,
        R3D18_SHA256,
        R3D18_URL,
        TORCHVISION_PACKAGE_VERSION,
        preprocessing_specification,
    )

    if not isinstance(provenance, Mapping) or set(provenance) != {
        "metrics",
        "perceptual_extractor",
        "video_feature_extractor",
        "preprocessing",
        "sha256",
    }:
        raise GateError("publication-quality provenance is malformed")
    unsigned = dict(provenance)
    embedded_sha256 = unsigned.pop("sha256")
    if embedded_sha256 != sha256_json(unsigned):
        raise GateError("publication-quality provenance self-hash differs")
    preprocessing = preprocessing_specification()
    if (
        provenance.get("metrics")
        != [
            LPIPS_ALEX_FRAME_METRIC,
            LPIPS_ALEX_TEMPORAL_DIFFERENCE_METRIC,
            R3D18_FRECHET_METRIC,
        ]
        or provenance.get("preprocessing") != preprocessing
    ):
        raise GateError("publication metric names or preprocessing differ from the pin")

    perceptual = provenance.get("perceptual_extractor")
    video = provenance.get("video_feature_extractor")
    if (
        not isinstance(perceptual, Mapping)
        or set(perceptual) != {"extractor", "package", "weights", "preprocessing"}
        or perceptual.get("extractor") != "FrozenLPIPSAlex"
        or perceptual.get("package")
        != {"name": "lpips", "version": LPIPS_PACKAGE_VERSION}
        or perceptual.get("preprocessing") != preprocessing
        or not isinstance(perceptual.get("weights"), list)
        or len(perceptual["weights"]) != 2
    ):
        raise GateError("LPIPS extractor provenance differs from the frozen contract")
    if (
        not isinstance(video, Mapping)
        or set(video)
        != {"extractor", "package", "weights_enum", "weights", "preprocessing"}
        or video.get("extractor") != "FrozenR3D18AvgPool"
        or video.get("package")
        != {"name": "torchvision", "version": TORCHVISION_PACKAGE_VERSION}
        or video.get("weights_enum") != "R3D_18_Weights.KINETICS400_V1"
        or video.get("preprocessing") != preprocessing
        or not isinstance(video.get("weights"), list)
        or len(video["weights"]) != 1
    ):
        raise GateError("R3D18 extractor provenance differs from the frozen contract")

    weight_records = {
        "lpips_linear": _validate_pinned_quality_weight(
            perceptual["weights"][0],
            role="LPIPS-Alex v0.1 linear calibration",
            expected_sha256=LPIPS_LINEAR_SHA256,
            source_url=(
                "https://github.com/richzhang/PerceptualSimilarity/"
                "tree/master/lpips/weights/v0.1"
            ),
        ),
        "alexnet": _validate_pinned_quality_weight(
            perceptual["weights"][1],
            role="ImageNet AlexNet backbone",
            expected_sha256=ALEXNET_SHA256,
            source_url=ALEXNET_URL,
        ),
        "r3d18": _validate_pinned_quality_weight(
            video["weights"][0],
            role="torchvision R3D-18 Kinetics-400 V1",
            expected_sha256=R3D18_SHA256,
            source_url=R3D18_URL,
        ),
    }
    return {
        "provenance_sha256": embedded_sha256,
        "serialized_provenance_sha256": sha256_json(provenance),
        "preprocessing_sha256": preprocessing["sha256"],
        "weights": weight_records,
    }


def _recompute_primary_frechet(
    record: Mapping[str, Any],
    *,
    clip_ids: Sequence[str],
) -> tuple[float, str, str]:
    try:
        payload = torch.load(record["path"], map_location="cpu", weights_only=False)
    except Exception as exc:
        raise GateError(f"cannot load primary R3D18 feature evidence: {exc}") from exc
    real = payload.get("r3d18_real_features")
    generated = payload.get("r3d18_generated_features")
    ids = payload.get("clip_ids")
    if (
        payload.get("feature_name") != "torchvision_r3d18_kinetics400_v1_avgpool"
        or payload.get("feature_dimension") != 512
        or payload.get("stored_dtype") != "float32_exact_extractor_output"
        or not isinstance(ids, (list, tuple))
        or tuple(ids) != tuple(clip_ids)
        or not isinstance(real, torch.Tensor)
        or not isinstance(generated, torch.Tensor)
        or real.dtype != torch.float32
        or generated.dtype != torch.float32
        or tuple(real.shape) != (CLIPS, 512)
        or tuple(generated.shape) != (CLIPS, 512)
        or not bool(torch.isfinite(real).all())
        or not bool(torch.isfinite(generated).all())
    ):
        raise GateError("primary R3D18 feature payload violates the frozen contract")
    from robot_wm.evaluation.video_latent_forcing_quality import r3d18_frechet

    value = r3d18_frechet(real, generated)
    real_digest = hashlib.sha256(real.float().contiguous().numpy().tobytes()).hexdigest()
    generated_digest = hashlib.sha256(
        generated.float().contiguous().numpy().tobytes()
    ).hexdigest()
    return value, real_digest, generated_digest


def _validate_phase2_control_group(
    *,
    arm: str,
    clip_id: str,
    donor_id: str | None,
    rows_by_control: Mapping[str, Mapping[str, Any]],
    donor_phase_boundary_sha256: str | None,
) -> dict[str, int]:
    """Validate one clip/NFE control group and return audited evidence counts."""
    controls = _expected_phase2_controls(arm)
    if set(rows_by_control) != set(controls):
        raise GateError("Phase-2 control group inventory differs")
    rows = [rows_by_control[control] for control in controls]
    invariant = (
        "checkpoint_sha256",
        "training_config_sha256",
        "shuffle_mapping_sha256",
        "history_sha256",
        "actions_sha256",
        "initial_video_noise_sha256",
        "target_auxiliary_sha256",
        "zero_auxiliary_sha256",
    )
    if arm != "B0":
        invariant += ("initial_auxiliary_noise_sha256", "phase_boundary_sha256")
    for field in invariant:
        if len({row.get(field) for row in rows}) != 1:
            raise GateError(f"Phase-2 paired control changed invariant field: {field}")
    for control, row in rows_by_control.items():
        digest = row.get("generated_video_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise GateError("Phase-2 generated-video evidence hash is malformed")
        if (
            type(row.get("teacher_model_calls")) is not int
            or row["teacher_model_calls"] != 0
            or row.get("deployable") is not (control != "oracle_clean")
            or row.get("clean_future_used_as_condition")
            is not (control == "oracle_clean")
        ):
            raise GateError("Phase-2 control leakage/call identity differs")

    if arm == "B0":
        row = rows_by_control["off"]
        if (
            row.get("conditioning_source_clip_id") != clip_id
            or row.get("auxiliary_conditioning_source_clip_id") != clip_id
            or row.get("history_action_source_clip_id") != clip_id
            or row.get("auxiliary_frozen_assertion_executed") is not False
            or row.get("phase_boundary_sha256") is not None
            or row.get("conditioning_auxiliary_sha256") is not None
            or row.get("pre_video_auxiliary_sha256") is not None
            or row.get("post_video_auxiliary_sha256") is not None
            or row.get("initial_auxiliary_noise_sha256") is not None
        ):
            raise GateError("B0 is not a strict auxiliary-state no-op")
        return {
            "boundary_control_comparisons": 0,
            "registered_auxiliary_bindings": 0,
            "deployable_generated_auxiliary_bindings": 0,
            "shuffled_donor_bindings": 0,
            "a1_generated_video_noop_comparisons": 0,
        }

    if donor_id is None or donor_phase_boundary_sha256 is None:
        raise GateError("dual Phase-2 control group lacks its shuffled donor")
    for row in rows:
        if (
            row.get("auxiliary_frozen_assertion_executed") is not True
            or row.get("pre_video_auxiliary_sha256")
            != row.get("conditioning_auxiliary_sha256")
            or row.get("post_video_auxiliary_sha256")
            != row.get("conditioning_auxiliary_sha256")
        ):
            raise GateError("dual auxiliary changed during the frozen video phase")
    autonomous = rows_by_control["autonomous"]
    off = rows_by_control["off"]
    shuffled = rows_by_control["shuffled"]
    oracle = rows_by_control["oracle_clean"]
    if (
        autonomous.get("conditioning_auxiliary_sha256")
        != autonomous.get("phase_boundary_sha256")
        or off.get("conditioning_auxiliary_sha256") != off.get("phase_boundary_sha256")
        or shuffled.get("conditioning_auxiliary_sha256")
        != donor_phase_boundary_sha256
        or oracle.get("conditioning_auxiliary_sha256")
        != oracle.get("target_auxiliary_sha256")
    ):
        raise GateError("Phase-2 control does not use the registered auxiliary tensor")
    for control, row in rows_by_control.items():
        expected_aux_source = donor_id if control == "shuffled" else clip_id
        if (
            row.get("conditioning_source_clip_id") != expected_aux_source
            or row.get("auxiliary_conditioning_source_clip_id") != expected_aux_source
            or row.get("history_action_source_clip_id") != clip_id
        ):
            raise GateError("Phase-2 control source identity differs from the frozen pairing")
    a1_noop = 0
    if arm == "A1":
        if len({row["generated_video_sha256"] for row in rows}) != 1:
            raise GateError("A1 video tensor changed under an auxiliary-state control")
        for metric in (
            *PAIRED_PRIMARY_METRICS,
            "rgb_mse",
            "rgb_psnr",
            "generated_pixel_clipped_fraction",
        ):
            if len({row.get(metric) for row in rows}) != 1:
                raise GateError("A1 video output changed under an auxiliary-state control")
        a1_noop = 1
    return {
        "boundary_control_comparisons": 1,
        "registered_auxiliary_bindings": len(controls),
        "deployable_generated_auxiliary_bindings": len(controls) - 1,
        "shuffled_donor_bindings": 1,
        "a1_generated_video_noop_comparisons": a1_noop,
    }


def _audit_phase2_evaluation(evaluation_root: str | Path) -> dict[str, Any]:
    """Audit one arm/checkpoint frontier and expose only bound gate evidence."""

    root = Path(evaluation_root).expanduser().resolve()
    if not root.is_dir():
        raise GateError(f"Phase-2 evaluation root is unavailable: {root}")
    config_path = root / "resolved_config.json"
    summary_path = root / "summary.json"
    metrics_path = root / "per_clip_metrics.jsonl"
    top_records = {
        "resolved_config": file_record(config_path),
        "summary": file_record(summary_path),
        "per_clip_metrics": file_record(metrics_path),
    }
    config = load_json(config_path, "Phase-2 evaluation resolved config")
    summary = load_json(summary_path, "Phase-2 evaluation summary")
    arm = config.get("arm")
    update = config.get("checkpoint_update")
    if arm not in DUAL_ARMS or update not in DUAL_CHECKPOINT_UPDATES:
        raise GateError("Phase-2 evaluation arm/checkpoint is outside the frozen matrix")
    arm = str(arm)
    update = int(update)
    pairs = _expected_phase2_pairs(arm)
    controls = _expected_phase2_controls(arm)
    expected_keys = {(aux, video, control) for aux, video in pairs for control in controls}
    source = config.get("source")
    quality = config.get("publication_quality_metrics")
    weights = config.get("weights")
    if (
        config.get("schema") != EVALUATION_SCHEMA
        or config.get("split") != "val"
        or config.get("seed") != BOOTSTRAP_BASE_SEED
        or config.get("nfe_pairs") != [list(pair) for pair in pairs]
        or config.get("controls") != list(controls)
        or config.get("fixed_noise_by_clip_id") is not True
        or config.get("world_size") != FROZEN_EVALUATION_WORLD_SIZE
        or config.get("eval_batch_size") != FROZEN_EVALUATION_BATCH_SIZE
        or config.get("quality_metric_suite_complete") is not True
        or not isinstance(quality, Mapping)
        or quality.get("enabled") is not True
        or not isinstance(quality.get("provenance"), Mapping)
        or not isinstance(weights, Mapping)
        or weights.get("kind") != "ema"
        or weights.get("decay") != 0.9999
        or weights.get("schedule") != EMA_SCHEDULE
        or weights.get("num_updates") != update
        or not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or summary.get("schema") != EVALUATION_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("arm") != arm
        or summary.get("split") != "val"
        or summary.get("record_count") != CLIPS * len(expected_keys)
        or summary.get("reported_weight_source") != "ema"
        or summary.get("ema_decay") != 0.9999
        or summary.get("ema_schedule") != EMA_SCHEDULE
        or summary.get("ema_updates") != update
        or summary.get("quality_metric_suite_complete") is not True
        or summary.get("quality_metric_provenance") != quality.get("provenance")
    ):
        raise GateError(f"{arm} update-{update} evaluation violates its frozen frontier")
    quality_audit = _validate_quality_provenance(quality["provenance"])

    checkpoint_record = _verify_file_record(config.get("checkpoint"), "Phase-2 checkpoint")
    if _verify_file_record(summary.get("checkpoint"), "summary checkpoint") != checkpoint_record:
        raise GateError("Phase-2 config/summary checkpoint records differ")
    training_config_record = _verify_file_record(
        config.get("training_config"), "Phase-2 training config"
    )
    training_config = load_json(training_config_record["path"], "Phase-2 training config")
    _validate_phase2_training_config(training_config, arm=arm, source=source)
    checkpoint_metadata, checkpoint_wall = _load_checkpoint_metadata(
        checkpoint_record,
        arm=arm,
        update=update,
        training_config=training_config,
    )
    training_root = Path(training_config_record["path"]).parent
    complete_record = file_record(training_root / "complete.json")
    complete = load_json(complete_record["path"], f"{arm} training completion")
    final_checkpoint_record = file_record(
        training_root / "checkpoints" / "update_020000.pt"
    )
    complete_wall = complete.get("cumulative_optimizer_wall_seconds")
    if (
        complete.get("schema") != RUN_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("command") != "train"
        or complete.get("arm") != arm
        or complete.get("completed_updates") != 20_000
        or complete.get("nonfinite_updates") != 0
        or complete.get("source") != source
        or complete.get("resolved_config_sha256") != sha256_json(training_config)
        or complete.get("checkpoint") != final_checkpoint_record
        or isinstance(complete_wall, bool)
        or not isinstance(complete_wall, (int, float))
        or not math.isfinite(complete_wall)
        or complete_wall <= 0
        or (update == 20_000 and complete_wall < checkpoint_wall)
    ):
        raise GateError(f"{arm} full training completion is missing or inconsistent")

    phase1_record = _verify_file_record(
        training_config.get("phase1_gate_record"), "Phase-1 handoff"
    )
    phase1 = load_json(phase1_record["path"], "Phase-1 handoff")
    if (
        phase1.get("schema") != GATE_SCHEMA
        or phase1.get("phase") != "phase1"
        or phase1.get("status") != "pass"
        or phase1.get("phase1_gate_passed") is not True
        or phase1.get("source_commit") != source.get("commit")
    ):
        raise GateError("Phase-2 training lacks a passed same-source Phase-1 handoff")
    _verify_file_record(training_config.get("calibration_record"), f"{arm} calibration")

    if _verify_file_record(summary.get("per_clip_metrics"), "Phase-2 per-clip metrics") != top_records[
        "per_clip_metrics"
    ]:
        raise GateError("Phase-2 summary does not bind its raw per-clip metrics")
    manifest = config.get("manifest")
    if not isinstance(manifest, Mapping) or manifest.get("clips") != CLIPS:
        raise GateError("Phase-2 validation manifest must contain exactly 890 clips")
    manifest_record = _verify_file_record(manifest, "Phase-2 validation manifest")
    provenance_record = _verify_file_record(manifest.get("data_provenance"), "DROID provenance")
    provenance = load_json(provenance_record["path"], "DROID provenance")
    builder_source = provenance.get("builder_source")
    split_ids = provenance.get("split_episode_ids_sha256")
    if (
        not isinstance(builder_source, Mapping)
        or builder_source.get("commit") != source.get("commit")
        or builder_source.get("dirty") is not False
        or builder_source.get("builder_tool_sha256") != sha256_file(BUILDER_PATH)
        or provenance.get("eligible_inventory_sha256") != ELIGIBLE_INVENTORY_SHA256
        or not isinstance(split_ids, Mapping)
        or split_ids.get("val") != VALIDATION_EPISODE_IDS_SHA256
    ):
        raise GateError("Phase-2 data provenance differs from preregistration")
    manifest_clip_ids, _ = _validation_manifest_identity(Path(manifest_record["path"]))
    sorted_ids = sorted(manifest_clip_ids)
    training_manifests = training_config.get("manifests")
    training_val = (
        training_manifests.get("validation")
        if isinstance(training_manifests, Mapping)
        else None
    )
    if (
        not isinstance(training_val, Mapping)
        or _verify_file_record(training_val, "training validation manifest") != manifest_record
        or training_config.get("data_root") != config.get("data_root")
        or _verify_file_record(summary.get("manifest"), "summary validation manifest")
        != manifest_record
    ):
        raise GateError("Phase-2 evaluation does not use its training validation population")

    donor_mapping: dict[str, str] = {}
    for position in range(0, len(manifest_clip_ids), 2):
        left, right = manifest_clip_ids[position : position + 2]
        donor_mapping[left] = right
        donor_mapping[right] = left
    donor_digest = sha256_json(donor_mapping)
    if config.get("shuffle_mapping_sha256") != donor_digest:
        raise GateError("Phase-2 shuffled mapping differs from the manifest-global pairing")

    summary_index = _phase2_summary_index(summary, expected_keys)
    feature_records = _phase2_feature_inventory(summary, expected_keys)
    rows = _read_rows(metrics_path)
    if len(rows) != CLIPS * len(expected_keys):
        raise GateError("Phase-2 per-clip record count is incomplete")
    indexed: dict[tuple[int, int, str, str], Mapping[str, Any]] = {}
    group_ids: dict[tuple[int, int, str], set[str]] = {}
    for row in rows:
        key3 = (row.get("auxiliary_nfe"), row.get("video_nfe"), row.get("control"))
        clip_id = row.get("clip_id")
        if (
            type(key3[0]) is not int
            or type(key3[1]) is not int
            or key3 not in expected_keys
            or not isinstance(clip_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", clip_id)
            or row.get("arm") != arm
            or type(row.get("path_model_calls")) is not int
            or row.get("path_model_calls") != key3[0] + key3[1]
            or type(row.get("evaluation_auxiliary_generation_calls")) is not int
            or row.get("evaluation_auxiliary_generation_calls") != key3[0]
            or type(row.get("optimizer_seed")) is not int
            or row.get("optimizer_seed") != 1234
            or type(row.get("evaluation_seed")) is not int
            or row.get("evaluation_seed") != BOOTSTRAP_BASE_SEED
            or row.get("checkpoint_sha256") != checkpoint_record["sha256"]
            or row.get("training_config_sha256") != training_config_record["sha256"]
            or row.get("ema_decay") != 0.9999
            or row.get("ema_schedule") != EMA_SCHEDULE
            or type(row.get("ema_updates")) is not int
            or row.get("ema_updates") != update
            or type(row.get("teacher_model_calls")) is not int
            or row.get("teacher_model_calls") != 0
            or row.get("deployable") is not (key3[2] != "oracle_clean")
            or row.get("clean_future_used_as_condition")
            is not (key3[2] == "oracle_clean")
            or row.get("auxiliary_frozen_assertion_executed") is not (arm != "B0")
            or row.get("shuffle_mapping_sha256") != donor_digest
        ):
            raise GateError(f"{arm} update-{update} per-clip identity/call/leakage contract failed")
        key4 = (*key3, clip_id)
        if key4 in indexed:
            raise GateError("duplicate Phase-2 per-clip group identity")
        for metric in (
            *PAIRED_PRIMARY_METRICS,
            "rgb_mse",
            "rgb_psnr",
            "generated_pixel_clipped_fraction",
        ):
            if _finite_number(row, metric) < 0:
                raise GateError(f"Phase-2 per-clip metric must be nonnegative: {metric}")
        if not 0 <= float(row["generated_pixel_clipped_fraction"]) <= 1:
            raise GateError("generated clipped fraction lies outside [0,1]")
        if arm == "B0":
            if (
                row.get("phase_boundary_sha256") is not None
                or row.get("conditioning_auxiliary_sha256") is not None
                or row.get("pre_video_auxiliary_sha256") is not None
                or row.get("post_video_auxiliary_sha256") is not None
                or row.get("initial_auxiliary_noise_sha256") is not None
                or row.get("auxiliary_nmse") is not None
                or row.get("auxiliary_cosine") is not None
            ):
                raise GateError("B0 is not a strict auxiliary-state no-op in raw evidence")
        else:
            for field in (
                "phase_boundary_sha256",
                "conditioning_auxiliary_sha256",
                "pre_video_auxiliary_sha256",
                "post_video_auxiliary_sha256",
                "initial_auxiliary_noise_sha256",
            ):
                if not isinstance(row.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", row[field]):
                    raise GateError(f"dual Phase-2 evidence hash is malformed: {field}")
            _finite_number(row, "auxiliary_nmse")
            _finite_number(row, "auxiliary_cosine")
            if (
                row["pre_video_auxiliary_sha256"] != row["conditioning_auxiliary_sha256"]
                or row["post_video_auxiliary_sha256"] != row["conditioning_auxiliary_sha256"]
            ):
                raise GateError("dual auxiliary changed during the frozen video phase")
        for field in (
            "generated_video_sha256",
            "target_auxiliary_sha256",
            "zero_auxiliary_sha256",
            "history_sha256",
            "actions_sha256",
            "initial_video_noise_sha256",
        ):
            if not isinstance(row.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", row[field]):
                raise GateError(f"Phase-2 paired evidence hash is malformed: {field}")
        indexed[key4] = row
        group_ids.setdefault(key3, set()).add(clip_id)
    if set(group_ids) != expected_keys or any(
        ids != set(manifest_clip_ids) for ids in group_ids.values()
    ):
        raise GateError("Phase-2 frontier groups do not share the exact validation population")

    boundary_control_comparisons = 0
    registered_auxiliary_bindings = 0
    deployable_generated_auxiliary_bindings = 0
    shuffled_donor_bindings = 0
    a1_generated_video_noop_comparisons = 0
    for aux_nfe, video_nfe in pairs:
        for clip_id in sorted_ids:
            rows_by_control = {
                control: indexed[(aux_nfe, video_nfe, control, clip_id)]
                for control in controls
            }
            donor_id = donor_mapping[clip_id] if arm != "B0" else None
            donor_boundary = (
                indexed[(aux_nfe, video_nfe, "autonomous", donor_id)][
                    "phase_boundary_sha256"
                ]
                if donor_id is not None
                else None
            )
            counts = _validate_phase2_control_group(
                arm=arm,
                clip_id=clip_id,
                donor_id=donor_id,
                rows_by_control=rows_by_control,
                donor_phase_boundary_sha256=donor_boundary,
            )
            boundary_control_comparisons += counts["boundary_control_comparisons"]
            registered_auxiliary_bindings += counts["registered_auxiliary_bindings"]
            deployable_generated_auxiliary_bindings += counts[
                "deployable_generated_auxiliary_bindings"
            ]
            shuffled_donor_bindings += counts["shuffled_donor_bindings"]
            a1_generated_video_noop_comparisons += counts[
                "a1_generated_video_noop_comparisons"
            ]

    deployable_rows = sum(row["deployable"] is True for row in indexed.values())
    oracle_rows = sum(row["control"] == "oracle_clean" for row in indexed.values())
    clean_future_condition_rows = sum(
        row["clean_future_used_as_condition"] is True for row in indexed.values()
    )
    dual_freeze_rows = sum(
        row["auxiliary_frozen_assertion_executed"] is True
        for row in indexed.values()
    )
    b0_auxiliary_noop_rows = sum(
        row["phase_boundary_sha256"] is None
        and row["conditioning_auxiliary_sha256"] is None
        and row["initial_auxiliary_noise_sha256"] is None
        for row in indexed.values()
    )
    structural_evidence = {
        "audited_rows": len(indexed),
        "teacher_model_call_sum": sum(int(row["teacher_model_calls"]) for row in indexed.values()),
        "deployable_rows": deployable_rows,
        "oracle_clean_rows": oracle_rows,
        "clean_future_condition_rows": clean_future_condition_rows,
        "deployable_clean_future_condition_rows": sum(
            row["deployable"] is True
            and row["clean_future_used_as_condition"] is True
            for row in indexed.values()
        ),
        "dual_auxiliary_freeze_rows": dual_freeze_rows,
        "b0_auxiliary_strict_noop_rows": b0_auxiliary_noop_rows,
        "boundary_control_comparisons": boundary_control_comparisons,
        "registered_auxiliary_bindings": registered_auxiliary_bindings,
        "deployable_generated_auxiliary_bindings": (
            deployable_generated_auxiliary_bindings
        ),
        "shuffled_donor_bindings": shuffled_donor_bindings,
        "a1_generated_video_noop_comparisons": a1_generated_video_noop_comparisons,
    }

    frozen_input_fields = (
        "history_sha256",
        "actions_sha256",
        "initial_video_noise_sha256",
        "initial_auxiliary_noise_sha256",
        "target_auxiliary_sha256",
        "zero_auxiliary_sha256",
    )
    input_hashes: dict[str, dict[str, Any]] = {}
    first_pair = pairs[0]
    first_control = controls[0]
    for clip_id in sorted_ids:
        first = indexed[(*first_pair, first_control, clip_id)]
        input_hashes[clip_id] = {field: first.get(field) for field in frozen_input_fields}
        for aux_nfe, video_nfe in pairs:
            for control in controls:
                row = indexed[(aux_nfe, video_nfe, control, clip_id)]
                for field in frozen_input_fields:
                    if row.get(field) != first.get(field):
                        raise GateError(f"Phase-2 frontier changed paired input field: {field}")

    cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    for key3, entry in summary_index.items():
        aux_nfe, video_nfe, control = key3
        group_rows = [indexed[(aux_nfe, video_nfe, control, clip_id)] for clip_id in sorted_ids]
        paired = {
            metric: np.asarray([_finite_number(row, metric) for row in group_rows], dtype=np.float64)
            for metric in PAIRED_PRIMARY_METRICS
        }
        means = {metric: float(entry[metric]) for metric in ALL_PRIMARY_METRICS}
        for metric in PAIRED_PRIMARY_METRICS:
            recomputed = float(paired[metric].mean())
            if not math.isclose(means[metric], recomputed, rel_tol=1e-12, abs_tol=1e-12):
                raise GateError(f"Phase-2 summary mean differs from raw {metric}")
        if key3 == (*PRIMARY_NFE[arm], PRIMARY_CONTROL[arm]):
            recomputed_frechet, real_digest, generated_digest = _recompute_primary_frechet(
                feature_records[key3], clip_ids=sorted_ids
            )
            _require_frechet_match(
                means[DISTRIBUTION_PRIMARY_METRIC],
                recomputed_frechet,
                label="primary",
            )
            cells[(arm, update, control)] = {
                "means": means,
                "paired": paired,
                "real_feature_sha256": real_digest,
                "generated_feature_sha256": generated_digest,
            }
        elif arm == "L1" and update == 20_000 and key3 in {
            (*PRIMARY_NFE["L1"], "off"),
            (*PRIMARY_NFE["L1"], "shuffled"),
        }:
            recomputed_frechet, real_digest, generated_digest = _recompute_primary_frechet(
                feature_records[key3], clip_ids=sorted_ids
            )
            _require_frechet_match(
                means[DISTRIBUTION_PRIMARY_METRIC],
                recomputed_frechet,
                label="attribution",
            )
            cells[(arm, update, control)] = {
                "means": means,
                "paired": paired,
                "real_feature_sha256": real_digest,
                "generated_feature_sha256": generated_digest,
            }

    evidence_records = {
        **top_records,
        "checkpoint": checkpoint_record,
        "training_config": training_config_record,
        "training_complete": complete_record,
        "final_checkpoint": final_checkpoint_record,
        "phase1_gate": phase1_record,
        "manifest": manifest_record,
        "data_provenance": provenance_record,
        "quality_weights": quality_audit["weights"],
        "quality_features": {str(key): value for key, value in sorted(feature_records.items())},
    }
    return {
        "root": str(root),
        "arm": arm,
        "update": update,
        "source": dict(source),
        "training_root": str(training_root),
        "training_config": training_config,
        "training_config_record": training_config_record,
        "checkpoint_record": checkpoint_record,
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_wall_seconds": checkpoint_wall,
        "manifest_record": manifest_record,
        "quality_provenance_sha256": quality_audit[
            "serialized_provenance_sha256"
        ],
        "quality_audit": quality_audit,
        "clip_ids": sorted_ids,
        "donor_mapping_sha256": donor_digest,
        "input_hashes": input_hashes,
        "cells": cells,
        "structural_evidence": structural_evidence,
        "evidence_records": evidence_records,
    }


def _phase2_structural_criteria(
    audits: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    """Turn fail-closed row audits into explicit mechanism/leakage criteria."""
    totals = {
        "audited_rows": 0,
        "teacher_model_call_sum": 0,
        "deployable_rows": 0,
        "oracle_clean_rows": 0,
        "clean_future_condition_rows": 0,
        "deployable_clean_future_condition_rows": 0,
        "dual_auxiliary_freeze_rows": 0,
        "b0_auxiliary_strict_noop_rows": 0,
        "boundary_control_comparisons": 0,
        "registered_auxiliary_bindings": 0,
        "deployable_generated_auxiliary_bindings": 0,
        "shuffled_donor_bindings": 0,
        "a1_generated_video_noop_comparisons": 0,
    }
    for (arm, update), audit in audits.items():
        evidence = audit.get("structural_evidence")
        if not isinstance(evidence, Mapping):
            raise GateError(f"{arm} update-{update} structural evidence is missing")
        pair_clips = CLIPS * len(_expected_phase2_pairs(arm))
        expected_rows = pair_clips * len(_expected_phase2_controls(arm))
        expected = {
            "audited_rows": expected_rows,
            "teacher_model_call_sum": 0,
            "deployable_rows": expected_rows if arm == "B0" else pair_clips * 3,
            "oracle_clean_rows": 0 if arm == "B0" else pair_clips,
            "clean_future_condition_rows": 0 if arm == "B0" else pair_clips,
            "deployable_clean_future_condition_rows": 0,
            "dual_auxiliary_freeze_rows": 0 if arm == "B0" else expected_rows,
            "b0_auxiliary_strict_noop_rows": expected_rows if arm == "B0" else 0,
            "boundary_control_comparisons": 0 if arm == "B0" else pair_clips,
            "registered_auxiliary_bindings": 0 if arm == "B0" else pair_clips * 4,
            "deployable_generated_auxiliary_bindings": (
                0 if arm == "B0" else pair_clips * 3
            ),
            "shuffled_donor_bindings": 0 if arm == "B0" else pair_clips,
            "a1_generated_video_noop_comparisons": pair_clips if arm == "A1" else 0,
        }
        for name, value in expected.items():
            if evidence.get(name) != value:
                raise GateError(
                    f"{arm} update-{update} structural evidence count differs: {name}"
                )
            totals[name] += value
    return {
        "auxiliary_mechanism": {
            "passed": True,
            "rule": (
                "B0 contains no auxiliary state; every dual row executes the per-step "
                "freeze assertion; generated phase boundaries are shared across controls; "
                "each control uses the registered tensor; shuffled uses its registered donor; "
                "and A1 raw generated video is bit-identical across controls"
            ),
            "evidence": {
                key: totals[key]
                for key in (
                    "dual_auxiliary_freeze_rows",
                    "b0_auxiliary_strict_noop_rows",
                    "boundary_control_comparisons",
                    "registered_auxiliary_bindings",
                    "shuffled_donor_bindings",
                    "a1_generated_video_noop_comparisons",
                )
            },
        },
        "no_inference_leakage": {
            "passed": True,
            "rule": (
                "every deployable row has zero teacher calls and conditions only on its "
                "autonomously generated aligned/donor auxiliary; clean future conditioning "
                "appears only in explicitly nondeployable oracle_clean rows"
            ),
            "evidence": {
                key: totals[key]
                for key in (
                    "audited_rows",
                    "teacher_model_call_sum",
                    "deployable_rows",
                    "oracle_clean_rows",
                    "clean_future_condition_rows",
                    "deployable_clean_future_condition_rows",
                    "deployable_generated_auxiliary_bindings",
                )
            },
        },
    }


def _validate_phase2_gate_cells(
    cells: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> str:
    """Require the exact decision-cell inventory and one shared real target."""
    expected = {
        (arm, update, PRIMARY_CONTROL[arm])
        for arm in DUAL_ARMS
        for update in DUAL_CHECKPOINT_UPDATES
    } | {
        ("L1", 20_000, "off"),
        ("L1", 20_000, "shuffled"),
    }
    if set(cells) != expected:
        raise GateError("Phase-2 gate-cell inventory is incomplete or contains extras")
    digests = {cell.get("real_feature_sha256") for cell in cells.values()}
    if len(digests) != 1:
        raise GateError("Phase-2 gate cells do not share one bit-identical R3D target matrix")
    digest = next(iter(digests))
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise GateError("Phase-2 real-target R3D feature digest is malformed")
    return digest


def _require_identical_phase2_inputs(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if candidate != reference:
        raise GateError(f"Phase-2 matrix changed {label}")


def _require_frechet_match(reported: float, recomputed: float, *, label: str) -> None:
    if (
        not math.isfinite(reported)
        or not math.isfinite(recomputed)
        or not math.isclose(reported, recomputed, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise GateError(f"{label} R3D18-Frechet differs from pinned raw features")


def _validate_source_binding(source: Mapping[str, Any]) -> None:
    current = git_record()
    if current.get("dirty") is not False or current.get("commit") != source.get("commit"):
        raise GateError("analyzer source is dirty or differs from Phase-2 evidence")


def _index_phase2_audits(
    evaluation_roots: Sequence[str | Path],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Audit and uniquely index the frozen 3-arm/7-checkpoint matrix."""
    expected = {
        (arm, update) for arm in DUAL_ARMS for update in DUAL_CHECKPOINT_UPDATES
    }
    if len(evaluation_roots) != len(expected):
        raise GateError(f"Phase-2 gate requires exactly {len(expected)} evaluation roots")
    audits: dict[tuple[str, int], dict[str, Any]] = {}
    for root in evaluation_roots:
        audit = _audit_phase2_evaluation(root)
        key = (audit["arm"], audit["update"])
        if key in audits:
            raise GateError(f"duplicate Phase-2 evaluation root for {key[0]}@{key[1]}")
        audits[key] = audit
    if set(audits) != expected:
        raise GateError("Phase-2 evaluation roots do not form the frozen 3-arm/7-checkpoint matrix")
    return audits


def analyze_phase2_evaluations(evaluation_roots: Sequence[str | Path]) -> dict[str, Any]:
    """Recompute the complete one-seed Phase-2 gate from 21 immutable frontiers."""
    audits = _index_phase2_audits(evaluation_roots)

    canonical = audits[("B0", DUAL_CHECKPOINT_UPDATES[0])]
    source = canonical["source"]
    manifest_record = canonical["manifest_record"]
    clip_ids = canonical["clip_ids"]
    quality_provenance = canonical["quality_provenance_sha256"]
    donor_digest = canonical["donor_mapping_sha256"]
    video_inputs = {
        clip_id: {
            key: value
            for key, value in canonical["input_hashes"][clip_id].items()
            if key != "initial_auxiliary_noise_sha256"
        }
        for clip_id in clip_ids
    }
    dual_auxiliary_inputs: dict[str, str] | None = None
    training_roots: dict[str, str] = {}
    training_configs: dict[str, Mapping[str, Any]] = {}
    checkpoint_walls: dict[tuple[str, int], float] = {}
    cells: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for (arm, update), audit in sorted(audits.items()):
        if (
            audit["source"] != source
            or audit["manifest_record"] != manifest_record
            or audit["clip_ids"] != clip_ids
            or audit["quality_provenance_sha256"] != quality_provenance
            or audit["donor_mapping_sha256"] != donor_digest
        ):
            raise GateError("Phase-2 matrix changed source/data/metric/control identity")
        current_video_inputs = {
            clip_id: {
                key: value
                for key, value in audit["input_hashes"][clip_id].items()
                if key != "initial_auxiliary_noise_sha256"
            }
            for clip_id in clip_ids
        }
        _require_identical_phase2_inputs(
            video_inputs,
            current_video_inputs,
            label="history/actions/video noise or target identity",
        )
        if arm != "B0":
            current_aux = {
                clip_id: audit["input_hashes"][clip_id]["initial_auxiliary_noise_sha256"]
                for clip_id in clip_ids
            }
            if dual_auxiliary_inputs is None:
                dual_auxiliary_inputs = current_aux
            else:
                _require_identical_phase2_inputs(
                    dual_auxiliary_inputs,
                    current_aux,
                    label="A1/L1 clip-keyed auxiliary noise",
                )
        previous_root = training_roots.setdefault(arm, audit["training_root"])
        if previous_root != audit["training_root"]:
            raise GateError(f"{arm} checkpoints do not come from one immutable training run")
        previous_config = training_configs.setdefault(arm, audit["training_config"])
        if previous_config != audit["training_config"]:
            raise GateError(f"{arm} checkpoint evaluations changed training configuration")
        checkpoint_walls[(arm, update)] = audit["checkpoint_wall_seconds"]
        cells.update(audit["cells"])

    _validate_phase2_checkpoint_walls(checkpoint_walls)
    common_config_keys = (
        "source",
        "clock_convention",
        "clean_time_epsilon",
        "seed",
        "updates",
        "checkpoint_updates",
        "global_batch_size",
        "world_size",
        "local_optimizer_batch_size",
        "dtype",
        "optimizer",
        "ema",
        "schedule",
        "manifests",
        "data_root",
    )
    reference_common = {key: training_configs["B0"].get(key) for key in common_config_keys}
    for arm in ("A1", "L1"):
        if {key: training_configs[arm].get(key) for key in common_config_keys} != reference_common:
            raise GateError("Phase-2 arms are not data/optimizer/schedule matched")
    b0_model = dict(training_configs["B0"]["model"])
    b0_model.pop("parameter_matched_video_only", None)
    for arm in ("A1", "L1"):
        model = dict(training_configs[arm]["model"])
        model.pop("parameter_matched_video_only", None)
        if model != b0_model:
            raise GateError("Phase-2 arms are not architecture/parameter matched")

    real_feature_sha256 = _validate_phase2_gate_cells(cells)
    decision = phase2_gate_decision(cells, checkpoint_walls)
    structural_criteria = _phase2_structural_criteria(audits)
    decision["criteria"].update(structural_criteria)
    decision["passed"] = all(
        criterion["passed"] for criterion in decision["criteria"].values()
    )
    evidence = {
        f"{arm}@{update}": {
            "root": audit["root"],
            "checkpoint": audit["checkpoint_record"],
            "training_config": audit["training_config_record"],
            "records": audit["evidence_records"],
        }
        for (arm, update), audit in sorted(audits.items())
    }
    payload: dict[str, Any] = {
        "schema": PHASE2_GATE_SCHEMA,
        "phase": "phase2",
        "status": "pass" if decision["passed"] else "fail",
        "frozen": True,
        "validation_only": True,
        "one_seed_gate": True,
        "phase2_gate_passed": decision["passed"],
        "source_commit": source["commit"],
        "protocol": file_record(PROTOCOL_PATH),
        "optimizer_seed": 1234,
        "evaluation_seed": BOOTSTRAP_BASE_SEED,
        "matrix": {
            "arms": list(DUAL_ARMS),
            "checkpoint_updates": list(DUAL_CHECKPOINT_UPDATES),
            "primary": {
                arm: {
                    "nfe_pair": list(PRIMARY_NFE[arm]),
                    "control": PRIMARY_CONTROL[arm],
                }
                for arm in DUAL_ARMS
            },
            "arm_roles": {
                "B0": "practical equal-50-call video-only quality baseline",
                "A1": (
                    "schedule/multitask attribution control; auxiliary fusion is off "
                    "and 25 auxiliary calls are not a competitive inference baseline"
                ),
                "L1": "generated-auxiliary dual cascade under test",
            },
            "validation_clips": CLIPS,
            "validation_clip_ids_sha256": sha256_json(clip_ids),
            "manifest": manifest_record,
            "shuffle_mapping_sha256": donor_digest,
            "quality_provenance_sha256": quality_provenance,
            "r3d18_real_target_features_sha256": real_feature_sha256,
        },
        "bootstrap": {
            "method": "paired clip-level percentile",
            "samples": BOOTSTRAP_SAMPLES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "base_seed": BOOTSTRAP_BASE_SEED,
            "seed_derivation": "first63bits(sha256(base_seed + NUL + statistic_label))",
            "paired_metrics": list(PAIRED_PRIMARY_METRICS),
            "distribution_metric": DISTRIBUTION_PRIMARY_METRIC,
            "multiplicity_control": "none; preregistered one-seed screening intervals",
            "frechet_uncertainty_interval": False,
        },
        "decision": decision,
        "evidence": evidence,
        "interpretation": {
            "screening_only": True,
            "obvious_advantage_claim_evaluable": False,
            "speed_claim_evaluable": False,
            "dagger_or_controllability_claim_evaluable": False,
            "reason": (
                "L1 adds auxiliary training compute; the one-seed screen has multiple "
                "checkpoint/metric looks without familywise correction or Frechet uncertainty; "
                "and no independent action-conditioned evaluator is implemented"
            ),
            "confirmation_required": (
                "three optimizer/evaluation seeds and one protected-test confirmation under "
                "a separately frozen familywise/Frechet uncertainty rule"
            ),
        },
        "protected_test_accessed": False,
        "confirmation_seeds_unlocked": False,
    }
    payload["decision_sha256"] = gate_decision_sha256(payload)

    for audit in audits.values():
        for record in audit["evidence_records"].values():
            if isinstance(record, Mapping) and "path" in record:
                _verify_file_record(record, "Phase-2 evidence recheck")
            elif isinstance(record, Mapping):
                for child in record.values():
                    _verify_file_record(child, "Phase-2 feature evidence recheck")
    if file_record(PROTOCOL_PATH) != payload["protocol"]:
        raise GateError("protocol changed while Phase-2 evidence was analyzed")
    _validate_source_binding(source)
    return payload


def validate_phase2_output_path(path: str | Path, roots: Sequence[str | Path]) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise GateError(f"gate output already exists: {output}")
    evidence_roots = [Path(root).expanduser().resolve() for root in roots]
    if _is_relative_to(output, REPO_ROOT) or any(
        _is_relative_to(output, root) for root in evidence_roots
    ):
        raise GateError("Phase-2 gate output must be outside every evidence and Git root")
    if not any(_is_relative_to(output, root) for root in APPROVED_ROOTS):
        raise GateError("gate output must be under /lustre, /mnt/data1, or /mnt/data2")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def phase2_gate(evaluation_roots: Sequence[str | Path], output: str | Path) -> dict[str, Any]:
    output_path = validate_phase2_output_path(output, evaluation_roots)
    payload = analyze_phase2_evaluations(evaluation_roots)
    atomic_publish_exclusive(output_path, payload)
    return payload


def phase1_gate(evaluation_root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(evaluation_root).expanduser().resolve()
    if not root.is_dir():
        raise GateError(f"evaluation root is unavailable: {root}")
    output_path = validate_output_path(output, root)
    payload = analyze_phase1_evaluation(root)
    atomic_publish_exclusive(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    phase1 = subparsers.add_parser("phase1")
    phase1.add_argument("--evaluation-root", required=True)
    phase1.add_argument("--output", required=True)
    phase2 = subparsers.add_parser("phase2")
    phase2.add_argument(
        "--evaluation-root",
        action="append",
        required=True,
        help="repeat exactly once for every B0/A1/L1 checkpoint frontier",
    )
    phase2.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = (
            phase1_gate(args.evaluation_root, args.output)
            if args.command == "phase1"
            else phase2_gate(args.evaluation_root, args.output)
        )
    except (GateError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Video Latent Forcing gate error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
