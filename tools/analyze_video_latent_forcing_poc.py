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
EVALUATION_SCHEMA = "video-latent-forcing-poc-evaluation-v1"
RUN_SCHEMA = "video-latent-forcing-poc-run-v1"
CHECKPOINT_SCHEMA = "video-latent-forcing-poc-checkpoint-v1"
CONTROLS = ("autonomous", "off", "shuffled", "oracle_clean", "context_shuffled")
NFE_GRID = (1, 2, 4, 8, 12, 20, 25)
GATE_NFE_GRID = (1, 2, 4, 8, 12)
CLIPS = 890
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_BASE_SEED = 20260801
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
        or not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or summary.get("schema") != EVALUATION_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("arm") != "phase1"
        or summary.get("split") != "val"
        or summary.get("record_count") != CLIPS * len(CONTROLS) * len(NFE_GRID)
        or summary.get("reported_weight_source") != "ema"
        or summary.get("ema_decay") != 0.9999
        or summary.get("quality_metric_suite_complete") is not True
    ):
        raise GateError("evaluation does not match the frozen update-5000 Phase-1 frontier")
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
    if (
        training_config.get("schema") != RUN_SCHEMA
        or training_config.get("command") != "train"
        or training_config.get("arm") != "phase1"
        or training_config.get("seed") != 1234
        or training_config.get("updates") != 5_000
        or training_config.get("checkpoint_updates") != [500, 1_000, 2_000, 5_000]
        or training_config.get("model") != config.get("model")
        or not isinstance(training_source, Mapping)
        or training_source != source
        or not isinstance(training_ema, Mapping)
        or training_ema.get("decay") != 0.9999
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
    }
    if after != before_complete:
        raise GateError("gate evidence changed while it was being analyzed")
    current_source = git_record()
    if current_source["dirty"] or current_source["commit"] != source["commit"]:
        raise GateError("analyzer source is dirty or differs from evaluated source")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = phase1_gate(args.evaluation_root, args.output)
    except (GateError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Video Latent Forcing gate error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
