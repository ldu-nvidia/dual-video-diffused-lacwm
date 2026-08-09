#!/usr/bin/env python3
"""Second target-blind execution shim for distributional Stage-0.

Retry 507412 computed and wrote per-clip proper scores, but Python rejected the
subsequent ``np.savez_compressed`` call because ``final_motion_strata`` was
supplied both explicitly and through ``**errors``.  This shim compiles the
registered ``run`` function after removing exactly that redundant explicit AST
keyword.  The complete AST is proven identical after reinserting the keyword.
No target, score, model, control, gate, or statistic is changed.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import interaction_event_bottleneck_stage0 as base
from tools import object_contact_distribution_pca_repair as repair1
from tools import object_contact_distribution_stage0 as study


SCHEMA_VERSION = 1
ORIGINAL_REGISTRATION_IDENTITY = repair1.ORIGINAL_REGISTRATION_IDENTITY
ORIGINAL_CALIBRATION_IDENTITY = repair1.ORIGINAL_CALIBRATION_IDENTITY
REPAIR1_REGISTRATION_IDENTITY = (
    "55410442f1c61c7409d570546b0b1eb7a557f0469f4d879d42c573a959d39653"
)
FAILED_JOB1_ID = "507387"
FAILED_JOB2_ID = "507412"
EXPECTED_EXCEPTION_2 = (
    "TypeError: numpy.savez_compressed() got multiple values for keyword argument "
    "'final_motion_strata'"
)
OPAQUE_SCORE_COPY = "failed_507412_final_per_clip_scores.opaque.jsonl"
OPAQUE_PROVENANCE_COPY = "failed_507412_final_input_provenance.opaque.jsonl"


def _registered_run_node(source_text: str) -> ast.FunctionDef:
    tree = ast.parse(source_text)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
    ]
    if len(nodes) != 1 or not isinstance(nodes[0], ast.FunctionDef):
        raise RuntimeError("registered source must contain exactly one synchronous run")
    return nodes[0]


def _savez_call(run_node: ast.FunctionDef) -> ast.Call:
    calls = []
    for node in ast.walk(run_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (
            node.func.attr == "savez_compressed"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
        ):
            calls.append(node)
    if len(calls) != 1:
        raise RuntimeError(f"expected one np.savez_compressed call, got {len(calls)}")
    return calls[0]


def _final_motion_strata_assignment(run_node: ast.FunctionDef) -> ast.Assign:
    assignments = []
    for node in ast.walk(run_node):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "errors"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "final_motion_strata"
        ):
            assignments.append(node)
    if len(assignments) != 1:
        raise RuntimeError(
            "expected one errors['final_motion_strata'] assignment, "
            f"got {len(assignments)}"
        )
    assignment = assignments[0]
    if not isinstance(assignment.value, ast.Name) or assignment.value.id != "strata":
        raise RuntimeError("errors['final_motion_strata'] value must be Name('strata')")
    return assignment


def transform_registered_run(source_text: str) -> tuple[ast.FunctionDef, dict[str, Any]]:
    original = _registered_run_node(source_text)
    assignment = _final_motion_strata_assignment(original)
    original_call = _savez_call(original)
    errors_expansions = [
        keyword
        for keyword in original_call.keywords
        if keyword.arg is None
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "errors"
    ]
    if len(errors_expansions) != 1:
        raise RuntimeError(
            "expected one **errors expansion in np.savez_compressed, "
            f"got {len(errors_expansions)}"
        )
    errors_expansion = errors_expansions[0]
    if int(assignment.lineno) >= int(original_call.lineno):
        raise RuntimeError("final_motion_strata assignment must precede save call")

    transformed = copy.deepcopy(original)
    call = _savez_call(transformed)
    matching = [
        (index, keyword)
        for index, keyword in enumerate(call.keywords)
        if keyword.arg == "final_motion_strata"
    ]
    if len(matching) != 1:
        raise RuntimeError("expected one explicit final_motion_strata keyword")
    keyword_index, removed = matching[0]
    if not isinstance(removed.value, ast.Name) or removed.value.id != "strata":
        raise RuntimeError("duplicate keyword value must be exactly Name('strata')")
    del call.keywords[keyword_index]

    restored = copy.deepcopy(transformed)
    restored_call = _savez_call(restored)
    restored_call.keywords.insert(keyword_index, copy.deepcopy(removed))
    original_dump = ast.dump(original, annotate_fields=True, include_attributes=False)
    transformed_dump = ast.dump(
        transformed, annotate_fields=True, include_attributes=False
    )
    restored_dump = ast.dump(restored, annotate_fields=True, include_attributes=False)
    if restored_dump != original_dump:
        raise RuntimeError("reinserting the one keyword does not exactly restore run AST")
    proof = {
        "transform": "remove one explicit np.savez_compressed keyword",
        "removed_keyword_arg": removed.arg,
        "removed_keyword_value_ast": ast.dump(
            removed.value, annotate_fields=True, include_attributes=False
        ),
        "keyword_index": keyword_index,
        "removed_keyword_lineno": int(removed.lineno),
        "prior_errors_assignment_count": 1,
        "prior_errors_assignment_lineno": int(assignment.lineno),
        "prior_errors_assignment_value_ast": ast.dump(
            assignment.value, annotate_fields=True, include_attributes=False
        ),
        "same_savez_errors_expansion_count": 1,
        "same_savez_errors_expansion_lineno": int(errors_expansion.lineno),
        "assignment_precedes_savez_call": True,
        "same_savez_contains_errors_expansion": True,
        "original_run_ast_sha256": hashlib.sha256(original_dump.encode()).hexdigest(),
        "transformed_run_ast_sha256": hashlib.sha256(
            transformed_dump.encode()
        ).hexdigest(),
        "restored_run_ast_sha256": hashlib.sha256(restored_dump.encode()).hexdigest(),
        "restoration_is_byte_identical_ast_dump": restored_dump == original_dump,
        "scientific_call_target": "np.savez_compressed",
        "any_other_ast_difference": False,
    }
    return transformed, proof


def apply_run_ast_shim() -> dict[str, Any]:
    source_path = Path(study.__file__).resolve()
    transformed, proof = transform_registered_run(source_path.read_text())
    module = ast.Module(body=[transformed], type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, str(source_path), "exec")
    exec(code, study.__dict__, study.__dict__)
    return proof


def _copy_opaque(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    source_record = base.file_record(source)
    destination_record = base.file_record(destination)
    if source_record["bytes"] != destination_record["bytes"]:
        raise RuntimeError("opaque copy byte count differs")
    if source_record["sha256"] != destination_record["sha256"]:
        raise RuntimeError("opaque copy hash differs")
    return {
        "original": source_record,
        "opaque_copy": destination_record,
        "copied_without_json_parsing": True,
    }


def repair2_register(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    path = output / "mechanical_repair2_registration.json"
    if path.exists():
        raise FileExistsError("repair-2 registration is immutable")
    registration = base.read_json(output / "registration.json")
    calibration = base.read_json(output / "calibration.json")
    repair1_document = base.read_json(output / "mechanical_repair_registration.json")
    for name, document in (
        ("original registration", registration),
        ("calibration", calibration),
        ("repair-1 registration", repair1_document),
    ):
        base.validate_identity(document, name)
    if registration["identity_sha256"] != ORIGINAL_REGISTRATION_IDENTITY:
        raise RuntimeError("original registration differs")
    if calibration["identity_sha256"] != ORIGINAL_CALIBRATION_IDENTITY:
        raise RuntimeError("calibration differs")
    if repair1_document["identity_sha256"] != REPAIR1_REGISTRATION_IDENTITY:
        raise RuntimeError("repair-1 registration differs")

    forbidden = (
        "analysis.json",
        "run_complete.json",
        "audit.json",
        "distribution_predictions.npz",
    )
    present = [name for name in forbidden if (output / name).exists()]
    if present:
        raise RuntimeError(f"retry unexpectedly persisted terminal artifacts: {present}")
    partial_scores = output / "final_per_clip_scores.jsonl"
    partial_provenance = output / "final_input_provenance.jsonl"
    if not partial_scores.is_file() or not partial_provenance.is_file():
        raise RuntimeError("retry partial artifacts are unavailable")
    score_evidence = _copy_opaque(partial_scores, output / OPAQUE_SCORE_COPY)
    provenance_evidence = _copy_opaque(
        partial_provenance, output / OPAQUE_PROVENANCE_COPY
    )

    failed1_stdout = args.failed1_stdout.resolve()
    failed1_stderr = args.failed1_stderr.resolve()
    failed2_stdout = args.failed2_stdout.resolve()
    failed2_stderr = args.failed2_stderr.resolve()
    if repair1.EXPECTED_EXCEPTION not in failed1_stderr.read_text():
        raise RuntimeError("job-1 exception differs")
    if EXPECTED_EXCEPTION_2 not in failed2_stderr.read_text():
        raise RuntimeError("job-2 exception differs")

    registered_source = Path(study.__file__).resolve()
    if base.sha256_file(registered_source) != registration["source"]["sha256"]:
        raise RuntimeError("registered scientific source was modified")
    _, proof = transform_registered_run(registered_source.read_text())
    source = Path(__file__).resolve()
    frozen = output / "frozen_object_contact_distribution_duplicate_keyword_repair.py"
    shutil.copy2(source, frozen)
    payload = base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "object_contact_distribution_mechanical_repair2_registration",
            "status": "target_blind_duplicate_keyword_repair_registered_before_retry2",
            "created_at_utc": study.now(),
            "repair2_source": {
                **base.file_record(source),
                "git_commit": args.expected_commit or base.git_commit(REPO_ROOT),
                "frozen_copy": base.file_record(frozen),
            },
            "chain": {
                "original_registration_identity_sha256": registration["identity_sha256"],
                "calibration_identity_sha256": calibration["identity_sha256"],
                "repair1_registration": base.file_record(
                    output / "mechanical_repair_registration.json"
                ),
                "repair1_registration_identity_sha256": repair1_document[
                    "identity_sha256"
                ],
                "registered_scientific_source": base.file_record(registered_source),
                "scientific_source_bytes_unchanged": True,
                "point_mse_stop_and_every_distributional_gate_unchanged": True,
            },
            "failed_job_507387": {
                "slurm_job_id": FAILED_JOB1_ID,
                "stdout": base.file_record(failed1_stdout),
                "stderr": base.file_record(failed1_stderr),
                "exception": repair1.EXPECTED_EXCEPTION,
                "scores_computed": False,
            },
            "failed_job_507412": {
                "slurm_job_id": FAILED_JOB2_ID,
                "stdout": base.file_record(failed2_stdout),
                "stderr": base.file_record(failed2_stderr),
                "exception": EXPECTED_EXCEPTION_2,
                "scores_computed": True,
                "scores_human_or_agent_inspected": False,
                "terminal_analysis_or_decision_computed": False,
                "partial_score_evidence": score_evidence,
                "partial_provenance_evidence": provenance_evidence,
            },
            "repair2": {
                "allowlist": [
                    "remove explicit final_motion_strata=strata keyword from np.savez_compressed because **errors already contains the identical final_motion_strata array"
                ],
                "registered_run_ast_proof": proof,
                "source_or_ast_changes_beyond_one_keyword": False,
                "target_model_controls_scores_gates_bootstrap_or_decision_changed": False,
                "partial_jsonl_parsed_to_design_or_test_repair": False,
                "retry_requirement": (
                    "regenerated final_per_clip_scores.jsonl SHA-256 must exactly equal "
                    "the opaque 507412 copy before audit; provenance equality is diagnostic only"
                ),
            },
            "claim_boundary": (
                "Second execution repair only. Proper-score bytes exist but were copied and hashed "
                "opaquely without JSON parsing before this exact one-keyword AST shim was frozen."
            ),
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )
    base.write_json(path, payload)
    return payload


def verify_regenerated_score_hash(output: Path) -> dict[str, Any]:
    repair = base.read_json(output / "mechanical_repair2_registration.json")
    base.validate_identity(repair, "repair-2 registration")
    regenerated = base.file_record(output / "final_per_clip_scores.jsonl")
    expected = repair["failed_job_507412"]["partial_score_evidence"]["opaque_copy"]
    score_match = bool(
        regenerated["bytes"] == expected["bytes"]
        and regenerated["sha256"] == expected["sha256"]
    )
    provenance = base.file_record(output / "final_input_provenance.jsonl")
    provenance_expected = repair["failed_job_507412"]["partial_provenance_evidence"][
        "opaque_copy"
    ]
    result = {
        "regenerated_score": regenerated,
        "opaque_failed_score": expected,
        "score_bytes_and_sha256_identical": score_match,
        "regenerated_provenance": provenance,
        "opaque_failed_provenance": provenance_expected,
        "provenance_bytes_and_sha256_identical_diagnostic": bool(
            provenance["bytes"] == provenance_expected["bytes"]
            and provenance["sha256"] == provenance_expected["sha256"]
        ),
    }
    if not score_match:
        raise RuntimeError("regenerated score JSONL differs from opaque failed evidence")
    return result


def repair2_audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    repair = base.read_json(output / "mechanical_repair2_registration.json")
    registration = base.read_json(output / "registration.json")
    analysis = base.read_json(output / "analysis.json")
    complete = base.read_json(output / "run_complete.json")
    scientific_audit = base.read_json(output / "audit.json")
    repair1_audit = base.read_json(output / "mechanical_repair_audit.json")
    for name, document in (
        ("repair-2", repair),
        ("registration", registration),
        ("analysis", analysis),
        ("completion", complete),
        ("scientific audit", scientific_audit),
        ("repair-1 audit", repair1_audit),
    ):
        base.validate_identity(document, name)
    if repair["chain"]["original_registration_identity_sha256"] != registration[
        "identity_sha256"
    ]:
        raise RuntimeError("repair-2 original binding differs")
    if scientific_audit["analysis_identity_sha256"] != analysis["identity_sha256"]:
        raise RuntimeError("scientific audit binding differs")
    if repair1_audit["analysis_identity_sha256"] != analysis["identity_sha256"]:
        raise RuntimeError("repair-1 audit binding differs")
    source = Path(__file__).resolve()
    if base.sha256_file(source) != repair["repair2_source"]["sha256"]:
        raise RuntimeError("repair-2 source differs")
    frozen = output / "frozen_object_contact_distribution_duplicate_keyword_repair.py"
    if base.sha256_file(frozen) != repair["repair2_source"]["frozen_copy"]["sha256"]:
        raise RuntimeError("frozen repair-2 source differs")
    _, proof = transform_registered_run(Path(study.__file__).resolve().read_text())
    if proof != repair["repair2"]["registered_run_ast_proof"]:
        raise RuntimeError("repair-2 AST proof differs")
    hash_verification = verify_regenerated_score_hash(output)
    return base.seal(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "object_contact_distribution_mechanical_repair2_audit",
            "audited_at_utc": study.now(),
            "status": "audit_passed",
            "decision": analysis["decision"],
            "repair2_registration_identity_sha256": repair["identity_sha256"],
            "original_registration_identity_sha256": registration["identity_sha256"],
            "analysis_identity_sha256": analysis["identity_sha256"],
            "completion_identity_sha256": complete["identity_sha256"],
            "scientific_audit_identity_sha256": scientific_audit["identity_sha256"],
            "repair1_audit_identity_sha256": repair1_audit["identity_sha256"],
            "score_regeneration": hash_verification,
            "registered_scientific_source_unchanged": True,
            "ast_restoration_exact": proof[
                "restoration_is_byte_identical_ast_dump"
            ],
            "validation_accessed": False,
            "protected_test_accessed": False,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("repair2-register")
    register.add_argument("--output", type=Path, required=True)
    register.add_argument("--failed1-stdout", type=Path, required=True)
    register.add_argument("--failed1-stderr", type=Path, required=True)
    register.add_argument("--failed2-stdout", type=Path, required=True)
    register.add_argument("--failed2-stderr", type=Path, required=True)
    register.add_argument("--expected-commit")
    audit = subparsers.add_parser("repair2-audit")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"repair2-register", "repair2-audit"}:
        args = build_parser().parse_args(arguments)
        if args.command == "repair2-register":
            result = repair2_register(args)
            event = "repair2_registered"
        else:
            result = repair2_audit(args)
            if args.report is not None:
                base.write_json(args.report, result)
            event = "repair2_audited"
        print(json.dumps({"event": event, **result}, sort_keys=True), flush=True)
        return 0
    apply_run_ast_shim()
    result = repair1.main(arguments)
    if arguments and arguments[0] == "run" and result == 0:
        parsed = study.build_parser().parse_args(arguments)
        verify_regenerated_score_hash(parsed.output.resolve())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
