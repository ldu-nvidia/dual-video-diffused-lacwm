#!/usr/bin/env python3
"""Apply the preregistered paired-bootstrap gate to sealed CSIP val64 rows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import csip_contract as contract  # noqa: E402
from tools.csip_workflow import validate_seal  # noqa: E402


CONTROLS = ("episode_disjoint_paired_shuffled", "raw_no_action", "inverse")
METRICS = ("mse", "cosine")
PROBES = ("full", "angle_neutral")
FAMILY_CELLS = len(CONTROLS) * len(METRICS) + len(METRICS)
FAMILYWISE_ALPHA = 0.05
CONTROL_THRESHOLDS = {
    "mse": {"minimum_point": 0.05, "minimum_lower_bound": 0.01, "relative": True},
    "cosine": {
        "minimum_point": 0.05,
        "minimum_lower_bound": 0.01,
        "relative": False,
    },
}
ANGLE_THRESHOLDS = {
    "mse": {"minimum_point": 0.03, "minimum_lower_bound": 0.01, "relative": True},
    "cosine": {
        "minimum_point": 0.02,
        "minimum_lower_bound": 0.005,
        "relative": False,
    },
}


def paired_effect(
    aligned: np.ndarray,
    control: np.ndarray,
    *,
    metric: str,
    bootstrap_pair_indexes: np.ndarray,
    pair_block_ids: np.ndarray,
    minimum_point: float = 0.0,
    minimum_lower_bound: float = 0.0,
    relative: bool | None = None,
) -> dict[str, Any]:
    if (
        aligned.shape != (contract.EXPECTED_VALIDATION_CLIPS,)
        or control.shape != (contract.EXPECTED_VALIDATION_CLIPS,)
        or bootstrap_pair_indexes.shape
        != (
            contract.BOOTSTRAP_REPLICATES,
            contract.EXPECTED_VALIDATION_PAIR_BLOCKS,
        )
        or not np.issubdtype(bootstrap_pair_indexes.dtype, np.integer)
        or pair_block_ids.shape != (contract.EXPECTED_VALIDATION_CLIPS,)
        or not np.issubdtype(pair_block_ids.dtype, np.integer)
        or not np.isfinite(aligned).all()
        or not np.isfinite(control).all()
        or metric not in METRICS
    ):
        raise contract.CSIPContractError("paired CSIP effect inputs differ")
    if relative is None:
        relative = metric == "mse"
    if relative != (metric == "mse") or minimum_point < 0 or minimum_lower_bound < 0:
        raise contract.CSIPContractError(
            "paired CSIP practical-effect contract differs"
        )
    expected_blocks = np.arange(contract.EXPECTED_VALIDATION_PAIR_BLOCKS)
    if (
        not np.array_equal(np.unique(pair_block_ids), expected_blocks)
        or any(
            np.count_nonzero(pair_block_ids == block) != 2
            for block in expected_blocks
        )
        or bootstrap_pair_indexes.min() < 0
        or bootstrap_pair_indexes.max() >= contract.EXPECTED_VALIDATION_PAIR_BLOCKS
    ):
        raise contract.CSIPContractError("paired CSIP block-bootstrap contract differs")

    # Positive is always favorable to aligned.  Average the two members of each
    # disjoint donor pair first; only the 32 independent pair blocks are sampled.
    differences = control - aligned if metric == "mse" else aligned - control
    block_differences = np.asarray(
        [differences[pair_block_ids == block].mean() for block in expected_blocks],
        dtype=np.float64,
    )
    bootstrap = block_differences[bootstrap_pair_indexes].mean(axis=1)
    alpha_cell = FAMILYWISE_ALPHA / FAMILY_CELLS
    effect = float(differences.mean())
    lower_bound = float(np.quantile(bootstrap, alpha_cell, method="linear"))
    result: dict[str, Any] = {
        "direction": "control_minus_aligned"
        if metric == "mse"
        else "aligned_minus_control",
        "mean_effect": effect,
        "simultaneous_one_sided_lower_bound": lower_bound,
        "bootstrap_nonpositive_fraction_with_plus_one": float(
            (1 + np.count_nonzero(bootstrap <= 0.0)) / (len(bootstrap) + 1)
        ),
        "cell_alpha": alpha_cell,
        "practical_threshold": {
            "minimum_point": minimum_point,
            "minimum_lower_bound": minimum_lower_bound,
            "scale": "relative_to_reference_mean" if relative else "absolute",
        },
    }
    if metric == "mse":
        denominator = float(control.mean())
        point_for_gate = effect / denominator if denominator > 0.0 else float("-inf")
        control_by_block = np.asarray(
            [control[pair_block_ids == block].mean() for block in expected_blocks],
            dtype=np.float64,
        )
        bootstrap_denominator = control_by_block[bootstrap_pair_indexes].mean(axis=1)
        if denominator > 0.0 and bool((bootstrap_denominator > 0.0).all()):
            relative_bootstrap = bootstrap / bootstrap_denominator
            lower_for_gate = float(
                np.quantile(relative_bootstrap, alpha_cell, method="linear")
            )
        else:
            lower_for_gate = float("-inf")
        result["relative_mse_improvement"] = (
            point_for_gate if math.isfinite(point_for_gate) else None
        )
        result["relative_simultaneous_one_sided_lower_bound"] = (
            lower_for_gate if math.isfinite(lower_for_gate) else None
        )
    else:
        point_for_gate = effect
        lower_for_gate = lower_bound
    result["pass"] = bool(
        effect > 0.0
        and lower_bound > 0.0
        and point_for_gate >= minimum_point
        and lower_for_gate >= minimum_lower_bound
    )
    return result


def _validated_evaluation(
    path: Path, seal: Mapping[str, Any], registration: Mapping[str, Any]
) -> dict[str, Any]:
    expected = Path(registration["planned_paths"]["evaluation"])
    if path.resolve() != expected:
        raise contract.CSIPContractError("evaluation path differs from registration")
    payload = contract.read_json(path, "CSIP sealed evaluation")
    contract.verify_identity(payload, "CSIP sealed evaluation")
    if (
        payload.get("schema_version") != contract.SCHEMA_VERSION
        or payload.get("kind") != contract.EVALUATION_KIND
        or payload.get("status") != "sealed_val64_complete"
        or payload.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or payload.get("checkpoint") != seal["checkpoint"]
        or payload.get("validation_clips") != contract.EXPECTED_VALIDATION_CLIPS
        or payload.get("conditions")
        != ["aligned", "episode_disjoint_paired_shuffled", "raw_no_action", "inverse"]
        or payload.get("probes") != ["full", "angle_neutral"]
        or payload.get("angle_comparator")
        != {
            "kind": "matched_9216_input_phasors_replaced_by_support_zero",
            "support_encoding": "each_masked_unit_phasor_becomes_mask_zero",
            "same_initialization": True,
            "same_batches_optimizer_updates_targets_architecture": True,
        }
        or payload.get("protected_test_clips_read") != 0
    ):
        raise contract.CSIPContractError("sealed evaluation contract differs")
    rows = payload.get("clips")
    if (
        not isinstance(rows, list)
        or len(rows) != contract.EXPECTED_VALIDATION_CLIPS
    ):
        raise contract.CSIPContractError("sealed evaluation must contain 64 rows")
    if [row.get("auxiliary_index") for row in rows] != list(
        range(contract.EXPECTED_VALIDATION_CLIPS)
    ):
        raise contract.CSIPContractError("sealed evaluation row order differs")
    if (
        len({str(row.get("clip_id", "")) for row in rows})
        != contract.EXPECTED_VALIDATION_CLIPS
    ):
        raise contract.CSIPContractError("sealed evaluation clip identities differ")
    expected_pairs = [
        {
            "pair_block": block,
            "left_index": 2 * block,
            "right_index": 2 * block + 1,
        }
        for block in range(contract.EXPECTED_VALIDATION_PAIR_BLOCKS)
    ]
    if payload.get("donor_control") != {
        "kind": "episode_disjoint_two_episode_pairs",
        "pairing_rule": "adjacent_immutable_manifest_indexes_swap_targets",
        "pair_count": contract.EXPECTED_VALIDATION_PAIR_BLOCKS,
        "pairs": expected_pairs,
        "self_donors": 0,
        "same_episode_donors": 0,
        "overlapping_pair_blocks": 0,
    }:
        raise contract.CSIPContractError("sealed evaluation donor-pair audit differs")
    for index, row in enumerate(rows):
        expected_donor = index + 1 if index % 2 == 0 else index - 1
        if (
            row.get("donor_index") != expected_donor
            or row.get("donor_pair_block") != index // 2
            or rows[expected_donor].get("donor_index") != index
            or row.get("donor_clip_id")
            != rows[expected_donor].get("clip_id")
            or row.get("donor_episode_dir")
            != rows[expected_donor].get("episode_dir")
        ):
            raise contract.CSIPContractError(
                "sealed donor pair is not a symmetric block"
            )
        if row.get("episode_dir") == row.get("donor_episode_dir") or row.get(
            "clip_id"
        ) == row.get("donor_clip_id"):
            raise contract.CSIPContractError("sealed evaluation donor is not disjoint")
        probes = row.get("metrics")
        if not isinstance(probes, Mapping) or set(probes) != set(PROBES):
            raise contract.CSIPContractError("sealed evaluation probes differ")
        for values in probes.values():
            if not isinstance(values, Mapping) or set(values) != {
                "aligned",
                *CONTROLS,
            }:
                raise contract.CSIPContractError("sealed evaluation conditions differ")
            for condition in values.values():
                if not isinstance(condition, Mapping) or any(
                    isinstance(condition.get(metric), bool)
                    or not isinstance(condition.get(metric), (int, float))
                    or not math.isfinite(float(condition[metric]))
                    for metric in METRICS
                ):
                    raise contract.CSIPContractError(
                        "sealed evaluation metric is invalid"
                    )
    return payload


def command_analyze(args: argparse.Namespace) -> int:
    seal, sealed_registration = validate_seal(args.seal)
    registration = contract.validate_registration(
        seal["registration"]["path"],
        require_train_cache=True,
        require_validation_cache=True,
        open_validation=True,
    )
    if registration["identity_sha256"] != sealed_registration["identity_sha256"]:
        raise contract.CSIPContractError("sealed registration changed")
    evaluation = _validated_evaluation(args.evaluation, seal, registration)
    output = Path(registration["planned_paths"]["analysis"])
    if output.exists() or output.is_symlink():
        raise contract.CSIPContractError("CSIP analysis output must be fresh")
    rng = np.random.default_rng(contract.BOOTSTRAP_SEED)
    bootstrap_pair_indexes = rng.integers(
        0,
        contract.EXPECTED_VALIDATION_PAIR_BLOCKS,
        size=(
            contract.BOOTSTRAP_REPLICATES,
            contract.EXPECTED_VALIDATION_PAIR_BLOCKS,
        ),
        endpoint=False,
        dtype=np.int64,
    )
    rows = evaluation["clips"]
    pair_block_ids = np.asarray(
        [row["donor_pair_block"] for row in rows], dtype=np.int64
    )
    control_cells: dict[str, Any] = {}
    for control in CONTROLS:
        control_cells[control] = {}
        for metric in METRICS:
            aligned = np.asarray(
                [row["metrics"]["full"]["aligned"][metric] for row in rows],
                dtype=np.float64,
            )
            reference = np.asarray(
                [row["metrics"]["full"][control][metric] for row in rows],
                dtype=np.float64,
            )
            control_cells[control][metric] = paired_effect(
                aligned,
                reference,
                metric=metric,
                bootstrap_pair_indexes=bootstrap_pair_indexes,
                pair_block_ids=pair_block_ids,
                **CONTROL_THRESHOLDS[metric],
            )
    angle_cells: dict[str, Any] = {}
    for metric in METRICS:
        full = np.asarray(
            [row["metrics"]["full"]["aligned"][metric] for row in rows],
            dtype=np.float64,
        )
        angle_neutral = np.asarray(
            [row["metrics"]["angle_neutral"]["aligned"][metric] for row in rows],
            dtype=np.float64,
        )
        angle_cells[metric] = paired_effect(
            full,
            angle_neutral,
            metric=metric,
            bootstrap_pair_indexes=bootstrap_pair_indexes,
            pair_block_ids=pair_block_ids,
            **ANGLE_THRESHOLDS[metric],
        )
    controls_pass = all(
        bool(control_cells[control][metric]["pass"])
        for control in CONTROLS
        for metric in METRICS
    )
    angle_pass = all(bool(angle_cells[metric]["pass"]) for metric in METRICS)
    passed = controls_pass and angle_pass
    if passed:
        conclusion = "advance_to_training_only_generator_regularization_ablation"
    elif not controls_pass:
        conclusion = "stop_csip_path_no_action_specific_spectral_signal_demonstrated"
    else:
        conclusion = "stop_csip_path_no_incremental_spectral_angle_contribution"
    analysis = contract.with_identity(
        {
            "schema_version": contract.SCHEMA_VERSION,
            "kind": contract.ANALYSIS_KIND,
            "created_at_utc": contract.now_utc(),
            "status": "complete",
            "registration": contract.registration_file_record(
                seal["registration"]["path"]
            ),
            "registration_identity_sha256": registration["identity_sha256"],
            "checkpoint_seal": {
                **contract.file_record(args.seal, "checkpoint seal"),
                "identity_sha256": seal["identity_sha256"],
            },
            "evaluation": {
                **contract.file_record(args.evaluation, "sealed evaluation"),
                "identity_sha256": evaluation["identity_sha256"],
            },
            "bootstrap": {
                "unit": "disjoint_two_episode_validation_pair_block",
                "clips": contract.EXPECTED_VALIDATION_CLIPS,
                "pair_blocks": contract.EXPECTED_VALIDATION_PAIR_BLOCKS,
                "clips_per_block": 2,
                "replicates": contract.BOOTSTRAP_REPLICATES,
                "seed": contract.BOOTSTRAP_SEED,
                "familywise_alpha": FAMILYWISE_ALPHA,
                "cells": FAMILY_CELLS,
                "correction": "bonferroni_one_sided",
                "shared_resample_index_matrix_across_all_cells": True,
            },
            "practical_effect_thresholds": {
                "full_probe_vs_target_controls": CONTROL_THRESHOLDS,
                "full_probe_vs_matched_angle_neutral": ANGLE_THRESHOLDS,
            },
            "cells": {
                "full_probe_target_controls": control_cells,
                "spectral_angle_contribution_over_angle_neutral": angle_cells,
            },
            "target_control_gate_pass": controls_pass,
            "spectral_angle_contribution_gate_pass": angle_pass,
            "gate_pass": passed,
            "decision": conclusion,
            "claim_boundary": (
                "A pass is one-seed probe feasibility on immutable val64. It does not "
                "demonstrate improved generated-video quality, lower NFE, real-time "
                "DAgger, or protected-test generalization."
            ),
            "generator_changes": 0,
            "protected_test_clips_read": 0,
        }
    )
    contract.exclusive_json(output, analysis)
    print(
        json.dumps(
            {
                "analysis": str(output),
                "gate_pass": passed,
                "decision": conclusion,
                "identity_sha256": analysis["identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    return parser


def main() -> int:
    return command_analyze(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
