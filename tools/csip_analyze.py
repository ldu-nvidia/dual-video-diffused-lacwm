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


CONTROLS = ("episode_disjoint_cyclic_shuffled", "zero", "inverse")
METRICS = ("mse", "cosine")
FAMILY_CELLS = len(CONTROLS) * len(METRICS)
FAMILYWISE_ALPHA = 0.05


def paired_effect(
    aligned: np.ndarray,
    control: np.ndarray,
    *,
    metric: str,
    bootstrap_indexes: np.ndarray,
) -> dict[str, Any]:
    if (
        aligned.shape != (64,)
        or control.shape != (64,)
        or bootstrap_indexes.shape != (contract.BOOTSTRAP_REPLICATES, 64)
        or not np.isfinite(aligned).all()
        or not np.isfinite(control).all()
        or metric not in METRICS
    ):
        raise contract.CSIPContractError("paired CSIP effect inputs differ")
    # Positive is always favorable to aligned.
    differences = control - aligned if metric == "mse" else aligned - control
    bootstrap = differences[bootstrap_indexes].mean(axis=1)
    alpha_cell = FAMILYWISE_ALPHA / FAMILY_CELLS
    effect = float(differences.mean())
    result: dict[str, Any] = {
        "direction": "control_minus_aligned"
        if metric == "mse"
        else "aligned_minus_control",
        "mean_effect": effect,
        "simultaneous_one_sided_lower_bound": float(
            np.quantile(bootstrap, alpha_cell, method="linear")
        ),
        "bootstrap_nonpositive_fraction_with_plus_one": float(
            (1 + np.count_nonzero(bootstrap <= 0.0)) / (len(bootstrap) + 1)
        ),
        "cell_alpha": alpha_cell,
    }
    if metric == "mse":
        denominator = float(control.mean())
        result["relative_mse_improvement"] = (
            effect / denominator if denominator > 0.0 else None
        )
    result["pass"] = bool(
        result["mean_effect"] > 0.0
        and result["simultaneous_one_sided_lower_bound"] > 0.0
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
        or payload.get("validation_clips") != 64
        or payload.get("conditions")
        != ["aligned", "episode_disjoint_cyclic_shuffled", "zero", "inverse"]
        or payload.get("protected_test_clips_read") != 0
    ):
        raise contract.CSIPContractError("sealed evaluation contract differs")
    rows = payload.get("clips")
    if not isinstance(rows, list) or len(rows) != 64:
        raise contract.CSIPContractError("sealed evaluation must contain 64 rows")
    if [row.get("auxiliary_index") for row in rows] != list(range(64)):
        raise contract.CSIPContractError("sealed evaluation row order differs")
    if len({str(row.get("clip_id", "")) for row in rows}) != 64:
        raise contract.CSIPContractError("sealed evaluation clip identities differ")
    for row in rows:
        if row.get("episode_dir") == row.get("donor_episode_dir") or row.get(
            "clip_id"
        ) == row.get("donor_clip_id"):
            raise contract.CSIPContractError("sealed evaluation donor is not disjoint")
        values = row.get("metrics")
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
                raise contract.CSIPContractError("sealed evaluation metric is invalid")
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
    bootstrap_indexes = rng.integers(
        0,
        64,
        size=(contract.BOOTSTRAP_REPLICATES, 64),
        endpoint=False,
        dtype=np.int64,
    )
    rows = evaluation["clips"]
    cells: dict[str, Any] = {}
    for control in CONTROLS:
        cells[control] = {}
        for metric in METRICS:
            aligned = np.asarray(
                [row["metrics"]["aligned"][metric] for row in rows], dtype=np.float64
            )
            reference = np.asarray(
                [row["metrics"][control][metric] for row in rows], dtype=np.float64
            )
            cells[control][metric] = paired_effect(
                aligned,
                reference,
                metric=metric,
                bootstrap_indexes=bootstrap_indexes,
            )
    passed = all(
        bool(cells[control][metric]["pass"])
        for control in CONTROLS
        for metric in METRICS
    )
    conclusion = (
        "advance_to_training_only_generator_regularization_ablation"
        if passed
        else "stop_csip_path_no_action_specific_spectral_signal_demonstrated"
    )
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
                "unit": "paired_validation_clip",
                "clips": 64,
                "replicates": contract.BOOTSTRAP_REPLICATES,
                "seed": contract.BOOTSTRAP_SEED,
                "familywise_alpha": FAMILYWISE_ALPHA,
                "cells": FAMILY_CELLS,
                "correction": "bonferroni_one_sided",
            },
            "cells": cells,
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
