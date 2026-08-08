#!/usr/bin/env python3
"""Paired development analysis for the video residual-anchor screen."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import video_residual_anchor_evaluate as evaluation  # noqa: E402


ANALYSIS_KIND = "video_residual_anchor_development_analysis"
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_REPLICATES = 10_000
LOWER_BETTER_METRICS = (
    "video_future_nmse",
    "video_future_temporal_delta_nmse",
    "decoded_mse_unit_range",
    "decoded_temporal_difference_mse_unit_range",
)
PRIMARY_METRICS = (
    "video_future_nmse",
    "decoded_mse_unit_range",
    "decoded_temporal_difference_mse_unit_range",
)
BONFERRONI_CONTRASTS = len(evaluation.NFE_GRID) * len(LOWER_BETTER_METRICS)
ONE_SIDED_ALPHA = 0.05 / BONFERRONI_CONTRASTS


class VideoResidualAnchorAnalysisError(RuntimeError):
    """An evidence receipt, pair identity, or statistical contract changed."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VideoResidualAnchorAnalysisError(f"{label} is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise VideoResidualAnchorAnalysisError(f"{label} must contain one object")
    return value


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = evaluation._canonical_json(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise VideoResidualAnchorAnalysisError(
            f"refusing to replace analysis output: {path}"
        ) from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _load_arm_rows(
    inventory_path: Path,
    arm: evaluation.Arm,
    registration: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory_path = inventory_path.resolve(strict=True)
    inventory = _read_json(inventory_path, f"{arm.code} inventory")
    if (
        not evaluation.identity_valid(inventory)
        or inventory.get("kind") != evaluation.KIND_INVENTORY
        or inventory.get("arm") != evaluation.asdict(arm)
        or inventory.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or inventory.get("status") != "complete"
        or inventory.get("clips") != evaluation.EXPECTED_VALIDATION_CLIPS
        or inventory.get("rows")
        != evaluation.EXPECTED_VALIDATION_CLIPS * len(evaluation.ENDPOINTS)
        or inventory.get("endpoints")
        != [evaluation.asdict(endpoint) for endpoint in evaluation.ENDPOINTS]
        or inventory.get("protected_test_accessed") is not False
        or inventory.get("target_cache_array_opened") is not False
    ):
        raise VideoResidualAnchorAnalysisError(f"{arm.code} inventory differs")
    rank_files = inventory.get("rank_files")
    rank_receipts = inventory.get("rank_receipts")
    if (
        not isinstance(rank_files, list)
        or not isinstance(rank_receipts, list)
        or len(rank_files) != evaluation.EXPECTED_WORLD_SIZE
        or len(rank_receipts) != evaluation.EXPECTED_WORLD_SIZE
    ):
        raise VideoResidualAnchorAnalysisError("rank receipt inventory is incomplete")
    rows: list[dict[str, Any]] = []
    for rank, (record, receipt_record) in enumerate(zip(rank_files, rank_receipts)):
        if not isinstance(record, Mapping) or not isinstance(receipt_record, Mapping):
            raise VideoResidualAnchorAnalysisError("rank file receipt is malformed")
        path = Path(str(record.get("path", ""))).resolve(strict=True)
        expected_rows_path = inventory_path.parent / f"rank_{rank:02d}.jsonl"
        if path != expected_rows_path.resolve(strict=True):
            raise VideoResidualAnchorAnalysisError("rank rows path is not canonical")
        if (
            evaluation._sha256(path) != record.get("sha256")
            or path.stat().st_size != record.get("bytes")
        ):
            raise VideoResidualAnchorAnalysisError("rank rows changed after inventory")
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
        receipt_path = Path(str(receipt_record.get("path", ""))).resolve(strict=True)
        expected_receipt_path = inventory_path.parent / f"rank_{rank:02d}.json"
        if (
            receipt_path != expected_receipt_path.resolve(strict=True)
            or evaluation._file_record(receipt_path) != dict(receipt_record)
        ):
            raise VideoResidualAnchorAnalysisError("rank receipt changed after inventory")
        receipt = _read_json(receipt_path, f"{arm.code} rank {rank} receipt")
        evaluation._validate_rank_receipt_payload(
            receipt,
            rank=rank,
            arm=arm,
            rows_record=record,
        )
    evaluation._validate_rows(rows, arm, registration)
    return rows, inventory


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {
        (int(row["clip_index"]), str(row["endpoint"]["code"])): row
        for row in rows
    }


def _validate_cross_arm_pairing(
    absolute: Mapping[tuple[int, str], Mapping[str, Any]],
    residual: Mapping[tuple[int, str], Mapping[str, Any]],
) -> None:
    if set(absolute) != set(residual):
        raise VideoResidualAnchorAnalysisError("arm row keys are not paired")
    invariant_hashes = (
        "cached_rgb_input_sha256",
        "cached_actions_input_sha256",
        "sampler_actions_sha256",
        "video_clean_scoring_sha256",
        "raw_ground_truth_sha256",
        "raw_history_last_sha256",
        "video_initial_noise_sha256",
        "auxiliary_initial_noise_sha256",
    )
    for key in absolute:
        left = absolute[key]
        right = residual[key]
        if (
            left.get("clip_id") != right.get("clip_id")
            or left.get("sampling_id") != right.get("sampling_id")
            or left.get("endpoint") != right.get("endpoint")
            or left.get("action_donor_sampling_id")
            != right.get("action_donor_sampling_id")
            or any(
                left["tensor_sha256"].get(field)
                != right["tensor_sha256"].get(field)
                for field in invariant_hashes
            )
        ):
            raise VideoResidualAnchorAnalysisError(
                f"cross-arm pairing differs at {key}"
            )


def _training_events(trace_path: Path) -> tuple[dict[str, Any], dict[int, Mapping[str, Any]]]:
    with trace_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or not isinstance(rows[0], dict):
        raise VideoResidualAnchorAnalysisError("training trace is empty")
    events = {}
    for row in rows[1:]:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, Mapping) or "train_loss/loss" not in metrics:
            continue
        iteration = metrics.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise VideoResidualAnchorAnalysisError("training event iteration is invalid")
        if iteration in events:
            raise VideoResidualAnchorAnalysisError("duplicate training iteration")
        events[iteration] = row
    if set(events) != set(range(200)):
        raise VideoResidualAnchorAnalysisError("training trace lacks 200 update events")
    return rows[0], events


def _validate_training_pairing(
    absolute_inventory: Mapping[str, Any],
    residual_inventory: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    traces = {}
    for code, inventory in (
        ("VPM-ABS", absolute_inventory),
        ("VPM-RESIDUAL", residual_inventory),
    ):
        rank_file = inventory["rank_files"][0]
        # Evaluation rows live at <root>/evaluation/<arm>/rank_00.jsonl.
        # The corresponding training trace is explicitly embedded in every row's
        # arm snapshot run directory, so derive it from the first row below.
        with Path(rank_file["path"]).open(encoding="utf-8") as handle:
            first = json.loads(next(handle))
        snapshot = Path(first["arm_snapshot"]["path"]).resolve(strict=True)
        arm = evaluation.ARM_BY_CODE[code]
        validated_trace = evaluation._validate_training_trace(
            snapshot.parent, arm, registration
        )
        trace = Path(validated_trace["trace"]["path"]).resolve(strict=True)
        traces[code] = _training_events(trace)
    abs_header, abs_events = traces["VPM-ABS"]
    residual_header, residual_events = traces["VPM-RESIDUAL"]
    if (
        abs_header.get("parent_snapshot_sha256")
        != residual_header.get("parent_snapshot_sha256")
        or abs_header.get("parent_run_identity_sha256")
        != residual_header.get("parent_run_identity_sha256")
        or abs_header.get("optimizer_state_policy")
        != residual_header.get("optimizer_state_policy")
        or abs_header.get("ema_policy") != residual_header.get("ema_policy")
    ):
        raise VideoResidualAnchorAnalysisError("arm training parent/policy differs")
    audit_keys = (
        "train_loss/paired_audit/clip_index_mean",
        "train_loss/paired_audit/clip_index_square_mean",
        "train_loss/paired_audit/timestep_mean",
        "train_loss/paired_audit/timestep_square_mean",
        "train_loss/paired_audit/noise_probe",
    )
    for iteration in range(200):
        left = abs_events[iteration]["metrics"]
        right = residual_events[iteration]["metrics"]
        if any(key not in left or key not in right or left[key] != right[key] for key in audit_keys):
            raise VideoResidualAnchorAnalysisError(
                f"paired training input/RNG audit differs at update {iteration}"
            )
    return {
        "paired_updates": 200,
        "audit_keys": list(audit_keys),
        "all_audit_values_exactly_equal": True,
        "parent_snapshot_sha256": abs_header["parent_snapshot_sha256"],
        "optimizer_state_policy": abs_header["optimizer_state_policy"],
        "ema_policy": abs_header["ema_policy"],
    }


def _paired_effect(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    if (
        baseline.shape != candidate.shape
        or baseline.ndim != 1
        or baseline.size != evaluation.EXPECTED_VALIDATION_CLIPS
        or not np.isfinite(baseline).all()
        or not np.isfinite(candidate).all()
        or float(baseline.mean()) <= 0
    ):
        raise VideoResidualAnchorAnalysisError("paired metric vector is invalid")
    point = float((baseline.mean() - candidate.mean()) / baseline.mean())
    rng = np.random.default_rng(seed)
    indexes = rng.integers(
        0,
        baseline.size,
        size=(BOOTSTRAP_REPLICATES, baseline.size),
        endpoint=False,
    )
    base_means = baseline[indexes].mean(axis=1)
    candidate_means = candidate[indexes].mean(axis=1)
    bootstrap = (base_means - candidate_means) / np.maximum(base_means, 1e-15)
    lower = float(np.quantile(bootstrap, ONE_SIDED_ALPHA, method="linear"))
    return {
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
        "relative_improvement": point,
        "one_sided_bonferroni_lower_bound": lower,
        "one_sided_alpha": ONE_SIDED_ALPHA,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": seed,
        "n": int(baseline.size),
    }


def _metric_vector(
    rows: Mapping[tuple[int, str], Mapping[str, Any]], endpoint: str, metric: str
) -> np.ndarray:
    return np.asarray(
        [float(rows[(index, endpoint)]["metrics"][metric]) for index in range(64)],
        dtype=np.float64,
    )


def _action_diagnostic(
    rows: Mapping[tuple[int, str], Mapping[str, Any]], metric: str
) -> dict[str, float]:
    matched = _metric_vector(rows, "autonomous_nfe_1", metric)
    shuffled = _metric_vector(rows, "actions_shuffled_nfe_1", metric)
    denominator = max(float(matched.mean()), 1e-15)
    return {
        "matched_mean": float(matched.mean()),
        "shuffled_mean": float(shuffled.mean()),
        "relative_worsening_when_actions_shuffled": float(
            (shuffled.mean() - matched.mean()) / denominator
        ),
    }


def command_analyze(args: argparse.Namespace) -> int:
    registration = evaluation._validate_registration(args.registration)
    absolute_rows, absolute_inventory = _load_arm_rows(
        args.absolute_inventory, evaluation.ARM_BY_CODE["VPM-ABS"], registration
    )
    residual_rows, residual_inventory = _load_arm_rows(
        args.residual_inventory,
        evaluation.ARM_BY_CODE["VPM-RESIDUAL"],
        registration,
    )
    absolute = _index(absolute_rows)
    residual = _index(residual_rows)
    _validate_cross_arm_pairing(absolute, residual)
    training_pairing = _validate_training_pairing(
        absolute_inventory, residual_inventory, registration
    )

    effects: dict[str, Any] = {}
    selected_nfe = None
    for nfe_index, nfe in enumerate(evaluation.NFE_GRID):
        endpoint = f"autonomous_nfe_{nfe}"
        effects[str(nfe)] = {}
        for metric_index, metric in enumerate(LOWER_BETTER_METRICS):
            effects[str(nfe)][metric] = _paired_effect(
                _metric_vector(absolute, endpoint, metric),
                _metric_vector(residual, endpoint, metric),
                seed=BOOTSTRAP_SEED + 100 * nfe_index + metric_index,
            )
        decoded = effects[str(nfe)]["decoded_mse_unit_range"]
        temporal = effects[str(nfe)][
            "decoded_temporal_difference_mse_unit_range"
        ]
        latent = effects[str(nfe)]["video_future_nmse"]
        latent_temporal = effects[str(nfe)]["video_future_temporal_delta_nmse"]
        gate = (
            decoded["relative_improvement"] >= 0.02
            and decoded["one_sided_bonferroni_lower_bound"] > 0.0
            and temporal["relative_improvement"] >= 0.02
            and temporal["one_sided_bonferroni_lower_bound"] > 0.0
            and latent["relative_improvement"] > -0.01
            and latent["one_sided_bonferroni_lower_bound"] > -0.01
            and latent_temporal["relative_improvement"] > -0.01
            and latent_temporal["one_sided_bonferroni_lower_bound"] > -0.01
        )
        effects[str(nfe)]["screen_gate_passed"] = gate
        if selected_nfe is None and gate:
            selected_nfe = nfe

    diagnostics = {
        arm: {
            metric: _action_diagnostic(indexed, metric)
            for metric in LOWER_BETTER_METRICS
        }
        for arm, indexed in (("VPM-ABS", absolute), ("VPM-RESIDUAL", residual))
    }
    conclusion = (
        "screening_signal_for_residual_coordinate_followup"
        if selected_nfe is not None
        else "no_controlled_residual_coordinate_advantage_in_quick_screen"
    )
    payload = evaluation.identity_payload(
        {
            "schema_version": evaluation.SCHEMA_VERSION,
            "kind": ANALYSIS_KIND,
            "registration_identity_sha256": registration["identity_sha256"],
            "analysis_split": "validation",
            "protected_test_accessed": False,
            "claim_scope": "adjacent_structural_baseline_not_dual_diffusion",
            "training_pairing": training_pairing,
            "evaluation_pairing": {
                "clips": evaluation.EXPECTED_VALIDATION_CLIPS,
                "same_targets_actions_noise_across_arms": True,
                "same_transformer_calls_at_each_nfe": True,
                "representation_inverted_before_all_metrics": True,
            },
            "statistical_protocol": {
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed_base": BOOTSTRAP_SEED,
                "one_sided_family_alpha": 0.05,
                "bonferroni_contrasts": BONFERRONI_CONTRASTS,
                "per_contrast_alpha": ONE_SIDED_ALPHA,
                "selection_uses_validation_only": True,
            },
            "effects": effects,
            "selected_nfe": selected_nfe,
            "action_shuffled_diagnostic_not_used_for_selection": diagnostics,
            "screen_gate": {
                "decoded_mse_relative_improvement_point": ">=0.02",
                "decoded_mse_lower_bound": ">0",
                "decoded_temporal_mse_relative_improvement_point": ">=0.02",
                "decoded_temporal_mse_lower_bound": ">0",
                "latent_nmse_point_and_lower_bound": ">-0.01",
                "latent_temporal_nmse_point_and_lower_bound": ">-0.01",
            },
            "conclusion": conclusion,
            "limitations": [
                "64 validation clips and 200 continuation updates only",
                "no protected test, FVD, latency, DAgger, or generalization claim",
                "coordinate reparameterization is not dual diffusion",
                "no train-derived residual whitening in this first screen",
                (
                    "the parent is functionally pretrained for absolute coordinates; "
                    "the residual arm spends part of 200 updates adapting its state "
                    "and velocity target"
                ),
                (
                    "both arms use an exact-clean-history clamp that differs from "
                    "the parent training trajectory"
                ),
            ],
        }
    )
    _exclusive_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--absolute-inventory", type=Path, required=True)
    parser.add_argument("--residual-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        return command_analyze(parser.parse_args(argv))
    except (
        VideoResidualAnchorAnalysisError,
        evaluation.VideoResidualAnchorEvaluationError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
