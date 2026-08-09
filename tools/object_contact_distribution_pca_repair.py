#!/usr/bin/env python3
"""Target-blind execution shim for the failed distributional Stage-0 run.

The registered scientific source is intentionally left byte-for-byte intact.
scikit-learn 1.7.2 requires ``explained_variance_`` to exist when
``PCA.transform`` discovers an array namespace, even though a non-whitened
transform does not use its value.  This shim hydrates only that fitted metadata
attribute and delegates every scientific operation to the registered module.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import interaction_event_bottleneck_stage0 as base
from tools import object_contact_distribution_stage0 as study


SCHEMA_VERSION = 1
ORIGINAL_REGISTRATION_IDENTITY = (
    "b39d6b2760bd970c9903cf663ab28387e0497668f044664789cc739475ee7225"
)
ORIGINAL_CALIBRATION_IDENTITY = (
    "ff6faca4537b7000572d8189135599b6138099690a0cbbf8d82ff9f18b2ffaea"
)
FAILED_JOB_ID = "507387"
EXPECTED_EXCEPTION = "AttributeError: 'PCA' object has no attribute 'explained_variance_'"

_REGISTERED_ACTION_PCA_FROM_ARRAYS = study._action_pca_from_arrays


def hydrated_action_pca_from_arrays(arrays: Mapping[str, np.ndarray]) -> Any:
    """Hydrate metadata required by sklearn; preserve the non-whitened map."""

    pca = _REGISTERED_ACTION_PCA_FROM_ARRAYS(arrays)
    pca.explained_variance_ = np.ones(
        study.ACTION_COMPONENTS,
        dtype=np.asarray(pca.components_).dtype,
    )
    return pca


# Delegated ``study.run`` resolves this module global at call time.
study._action_pca_from_arrays = hydrated_action_pca_from_arrays


def synthetic_transform_equivalence() -> dict[str, Any]:
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(20260817)
    fit = rng.normal(size=(96, 23)).astype(np.float64)
    query = rng.normal(size=(37, 23)).astype(np.float64)
    fitted = PCA(n_components=study.ACTION_COMPONENTS, svd_solver="full").fit(fit)
    hydrated = hydrated_action_pca_from_arrays(
        {
            "action_pca_mean": fitted.mean_,
            "action_pca_components": fitted.components_,
        }
    )
    reference = fitted.transform(query)
    repaired = hydrated.transform(query)
    explicit = (query - fitted.mean_) @ fitted.components_.T
    max_hydrated = float(np.max(np.abs(reference - repaired)))
    max_explicit = float(np.max(np.abs(reference - explicit)))
    return {
        "seed": 20260817,
        "fit_shape": list(fit.shape),
        "query_shape": list(query.shape),
        "whiten": bool(fitted.whiten),
        "hydrated_max_abs_difference": max_hydrated,
        "explicit_map_max_abs_difference": max_explicit,
        "atol": 1e-12,
        "rtol": 1e-12,
        "passed": bool(
            np.allclose(reference, repaired, rtol=1e-12, atol=1e-12)
            and np.allclose(reference, explicit, rtol=1e-12, atol=1e-12)
        ),
    }


def calibration_transform_equivalence(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    model_path = Path(registration["calibration"]["model"]["path"])
    with np.load(model_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    action_path = Path(registration["inputs"]["actions"]["path"])
    actions = np.load(action_path, mmap_mode="r", allow_pickle=False)
    calibration_indices = np.asarray(
        [item["manifest_index"] for item in registration["split"]["calibration"]],
        dtype=np.int64,
    )
    values = np.asarray(
        actions[
            calibration_indices,
            study.slot.ACTION_WINDOW[0] : study.slot.ACTION_WINDOW[1],
            :,
            : study.base.ACTION_DIM,
        ],
        dtype=np.float32,
    )
    pca = hydrated_action_pca_from_arrays(arrays)
    transformed = study._transform_action_pca8(
        pca,
        values,
        arrays["action_mean"],
        arrays["action_std"],
        arrays["action_active"],
        arrays["action_pc_mean"],
        arrays["action_pc_std"],
    )
    stored = arrays["calibration_action_z"]
    difference = float(np.max(np.abs(transformed - stored)))
    return {
        "rows": len(calibration_indices),
        "coordinates": int(transformed.shape[1]),
        "max_abs_difference_from_registered_calibration_transform": difference,
        "atol": 1e-6,
        "rtol": 1e-6,
        "passed": bool(np.allclose(transformed, stored, rtol=1e-6, atol=1e-6)),
        "final31_action_or_target_opened_for_equivalence": False,
    }


def repair_register(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    repair_path = output / "mechanical_repair_registration.json"
    if repair_path.exists():
        raise FileExistsError("mechanical repair registration is immutable")
    registration = base.read_json(output / "registration.json")
    calibration = base.read_json(output / "calibration.json")
    base.validate_identity(registration, "original registration")
    base.validate_identity(calibration, "original calibration")
    if registration["identity_sha256"] != ORIGINAL_REGISTRATION_IDENTITY:
        raise RuntimeError("original registration identity differs")
    if calibration["identity_sha256"] != ORIGINAL_CALIBRATION_IDENTITY:
        raise RuntimeError("original calibration identity differs")
    forbidden = (
        "analysis.json",
        "run_complete.json",
        "audit.json",
        "distribution_predictions.npz",
        "final_input_provenance.jsonl",
        "final_per_clip_scores.jsonl",
    )
    present = [name for name in forbidden if (output / name).exists()]
    if present:
        raise RuntimeError(f"failed run unexpectedly persisted outcome artifacts: {present}")
    failed_stdout = args.failed_stdout.resolve()
    failed_stderr = args.failed_stderr.resolve()
    stderr_text = failed_stderr.read_text()
    if EXPECTED_EXCEPTION not in stderr_text:
        raise RuntimeError("failed log does not contain the registered mechanical exception")

    synthetic = synthetic_transform_equivalence()
    calibration_equivalence = calibration_transform_equivalence(registration)
    if not synthetic["passed"] or not calibration_equivalence["passed"]:
        raise RuntimeError("mechanical transform equivalence failed")

    source = Path(__file__).resolve()
    registered_source = Path(study.__file__).resolve()
    if base.sha256_file(registered_source) != registration["source"]["sha256"]:
        raise RuntimeError("registered scientific source was modified")
    frozen_wrapper = output / "frozen_object_contact_distribution_pca_repair.py"
    shutil.copy2(source, frozen_wrapper)
    payload = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "object_contact_distribution_mechanical_repair_registration",
            "status": "target_blind_mechanical_repair_registered_before_retry",
            "created_at_utc": study.now(),
            "repair_source": {
                **base.file_record(source),
                "git_commit": args.expected_commit or base.git_commit(REPO_ROOT),
                "frozen_copy": base.file_record(frozen_wrapper),
            },
            "original_contract": {
                "registration": base.file_record(output / "registration.json"),
                "registration_identity_sha256": registration["identity_sha256"],
                "calibration": base.file_record(output / "calibration.json"),
                "calibration_identity_sha256": calibration["identity_sha256"],
                "registered_scientific_source": base.file_record(registered_source),
                "scientific_source_bytes_unchanged": True,
                "split_target_model_scores_gates_bootstrap_unchanged": True,
                "deterministic_point_mse_stop_unchanged": True,
            },
            "failed_execution": {
                "slurm_job_id": FAILED_JOB_ID,
                "state": "FAILED",
                "exit_code": "1:0",
                "exception": EXPECTED_EXCEPTION,
                "stdout": base.file_record(failed_stdout),
                "stderr": base.file_record(failed_stderr),
                "all_final31_targets_opened_in_memory_before_exception": True,
                "any_final31_value_printed_or_inspected": False,
                "score_or_distribution_computation_reached": False,
                "persisted_outcome_artifacts": [],
            },
            "repair": {
                "allowlist": [
                    "set reconstructed non-whitened PCA.explained_variance_ to an eight-vector of ones"
                ],
                "why_value_is_inert": (
                    "sklearn 1.7.2 inspects explained_variance_ for namespace selection; "
                    "PCA.transform with whiten=False computes only (X - mean_) @ components_.T"
                ),
                "synthetic_transform_equivalence": synthetic,
                "calibration_only_transform_equivalence": calibration_equivalence,
                "final31_used_to_design_or_test_repair": False,
                "retry_must_delegate_original_registered_run_and_audit": True,
            },
            "claim_boundary": (
                "Execution repair only. Final31 is no longer untouched, but no value, distribution, "
                "score, or derived artifact was observed before this target-blind repair was frozen."
            ),
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(repair_path, payload)
    return payload


def repair_audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    repair = base.read_json(output / "mechanical_repair_registration.json")
    registration = base.read_json(output / "registration.json")
    analysis = base.read_json(output / "analysis.json")
    complete = base.read_json(output / "run_complete.json")
    audit = base.read_json(output / "audit.json")
    for name, document in (
        ("repair registration", repair),
        ("original registration", registration),
        ("analysis", analysis),
        ("completion", complete),
        ("scientific audit", audit),
    ):
        base.validate_identity(document, name)
    if repair["original_contract"]["registration_identity_sha256"] != registration["identity_sha256"]:
        raise RuntimeError("repair-original binding differs")
    if analysis["registration_identity_sha256"] != registration["identity_sha256"]:
        raise RuntimeError("analysis-original binding differs")
    if audit["analysis_identity_sha256"] != analysis["identity_sha256"]:
        raise RuntimeError("scientific audit-analysis binding differs")
    if base.sha256_file(Path(study.__file__).resolve()) != registration["source"]["sha256"]:
        raise RuntimeError("registered scientific source changed")
    if base.sha256_file(Path(__file__).resolve()) != repair["repair_source"]["sha256"]:
        raise RuntimeError("repair source changed")
    frozen = output / "frozen_object_contact_distribution_pca_repair.py"
    if base.sha256_file(frozen) != repair["repair_source"]["frozen_copy"]["sha256"]:
        raise RuntimeError("frozen repair source changed")
    synthetic = synthetic_transform_equivalence()
    calibration_equivalence = calibration_transform_equivalence(registration)
    if synthetic != repair["repair"]["synthetic_transform_equivalence"]:
        raise RuntimeError("synthetic repair equivalence differs")
    if calibration_equivalence != repair["repair"]["calibration_only_transform_equivalence"]:
        raise RuntimeError("calibration repair equivalence differs")
    return base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "object_contact_distribution_mechanical_repair_audit",
            "audited_at_utc": study.now(),
            "status": "audit_passed",
            "decision": analysis["decision"],
            "repair_registration_identity_sha256": repair["identity_sha256"],
            "original_registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "completion_identity_sha256": complete["identity_sha256"],
            "scientific_audit_identity_sha256": audit["identity_sha256"],
            "scientific_source_bytes_unchanged": True,
            "synthetic_equivalence_passed": synthetic["passed"],
            "calibration_equivalence_passed": calibration_equivalence["passed"],
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )


def build_repair_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("repair-register")
    register.add_argument("--output", type=Path, required=True)
    register.add_argument("--failed-stdout", type=Path, required=True)
    register.add_argument("--failed-stderr", type=Path, required=True)
    register.add_argument("--expected-commit")
    audit = subparsers.add_parser("repair-audit")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"repair-register", "repair-audit"}:
        args = build_repair_parser().parse_args(arguments)
        if args.command == "repair-register":
            result = repair_register(args)
            event = "repair_registered"
        else:
            result = repair_audit(args)
            if args.report is not None:
                base.write_json(args.report, result)
            event = "repair_audited"
        print(json.dumps({"event": event, **result}, sort_keys=True), flush=True)
        return 0
    return study.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
