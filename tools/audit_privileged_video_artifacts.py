#!/usr/bin/env python3
"""Fail-closed bitwise audit for the three privileged-TF video evaluations.

The evaluated roots are immutable inputs.  This command creates exactly one
caller-selected JSON report, outside those roots, with exclusive-create
semantics.  LACWM uses ``sigma=1`` for noise and ``sigma=0`` for clean data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tools.audit_stage_faithful_artifacts import (
        ArtifactRecord,
        StageArtifactAuditError as PrivilegedArtifactAuditError,
        _artifact_set_sha256,
        _canonical_directory,
        _canonical_regular_file,
        _comparison,
        _discover,
        _exclusive_json,
        _forbidden_training_outputs,
        _get_tensor,
        _integer_values,
        _load_json_strict,
        _output_path,
        _rank_artifact_inventory,
        _require_pass,
        _sha256_file,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from audit_stage_faithful_artifacts import (  # type: ignore[no-redef]
        ArtifactRecord,
        StageArtifactAuditError as PrivilegedArtifactAuditError,
        _artifact_set_sha256,
        _canonical_directory,
        _canonical_regular_file,
        _comparison,
        _discover,
        _exclusive_json,
        _forbidden_training_outputs,
        _get_tensor,
        _integer_values,
        _load_json_strict,
        _output_path,
        _rank_artifact_inventory,
        _require_pass,
        _sha256_file,
    )


SCHEMA_VERSION = 1
SIGMA_CONVENTION = "1=noise,0=clean"
WORLD_SIZE = 8
ARTIFACT_ITERATION = 199
EVALUATION_NOISE_SEED = 20_260_726
NFE_STEPS = (1, 2, 4, 8)
SOURCE_CODES = (0, 1)
ARM_NAMES = ("trained_matched", "trained_shuffled", "trained_off")
PROVENANCE_NAME = "privileged_video_evaluation_provenance.json"
PARENT_BASE = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/"
    "lacwm_train/runs/dual_video_diffusion/ztf_first_cascade_screen/"
    "abc200-tf-cascade3-s1234-18318ed-v1"
)
EXPECTED_PARENTS = {
    "trained_matched": {
        "arm": "cascade_matched_s010",
        "snapshot": f"{PARENT_BASE}/cascade_matched_s010/snapshot.pt",
        "snapshot_sha256": (
            "5e96584ac70af54463ddebde1d2581c982c0be1aabbac1335b26722a1c03164d"
        ),
        "run_identity_sha256": (
            "ea6963718edb2b7827b189f3c622e5affe9b1706fb1cb623159731c8c29486e5"
        ),
    },
    "trained_shuffled": {
        "arm": "cascade_shuffled_s010",
        "snapshot": f"{PARENT_BASE}/cascade_shuffled_s010/snapshot.pt",
        "snapshot_sha256": (
            "1b5e70982d1a93b4069b8ad1c33b25ba1b4d106c560dcaad63e1d2dd23c3eb76"
        ),
        "run_identity_sha256": (
            "151cf2b0a349878839f515b34ea4f8edd6a34528c2814c12b1fdc5afc9a3645f"
        ),
    },
    "trained_off": {
        "arm": "cascade_off_s000",
        "snapshot": f"{PARENT_BASE}/cascade_off_s000/snapshot.pt",
        "snapshot_sha256": (
            "a147acb27dec8fb9f793d665861149ebc8d203b63ab1e6d107760f62d0b36e6b"
        ),
        "run_identity_sha256": (
            "8861147ccfcc0a2909480400d7f09452ae192298ac758f1cf73f71802d0b5f9b"
        ),
    },
}
IDENTITY_TENSORS = (
    "video_clean",
    "tf_clean",
    "video_initial_state",
    "tf_initial_state",
    "tf_initial_noise",
    "history_latent_frames",
    "evaluation_noise_seed",
    "ground_truth_future_uint8",
    "raw_actions",
    "raw_morphology_index",
)
DIAGNOSTIC_TENSORS = ("z_control",)
EXACT_INTEGER_CONTRACTS = {
    "evaluation_condition_source_codes": SOURCE_CODES,
    "evaluation_nfe_steps": NFE_STEPS,
    "evaluation_noise_seed": (EVALUATION_NOISE_SEED,),
    "condition_on_tf": (0,),
    "condition_mode_code": (0,),
    "cascade_stage_faithful_inference": (0,),
    "evaluation_disable_tf_clock": (1,),
    "evaluation_tf_clock_enabled": (0,),
    "evaluation_all_video_schedule": (1,),
    "raw_actions_present": (1,),
    "raw_morphology_index_present": (1,),
}
FINAL_PREFIXES = ("video_final", "tf_final", "decoded_future")


def _expected_final_keys() -> set[str]:
    return {
        f"{prefix}{infix}_nfe_{nfe}"
        for prefix in FINAL_PREFIXES
        for infix in ("", "_off")
        for nfe in NFE_STEPS
    }


def _validate_final_inventory(record: ArtifactRecord, arm: str) -> None:
    expected = _expected_final_keys()
    actual = {
        key for key in record.keys if key.startswith(FINAL_PREFIXES)
    }
    if actual != expected:
        raise PrivilegedArtifactAuditError(
            f"{arm} rank {record.rank} final tensor inventory mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _validate_roots(
    roots: Mapping[str, str | Path],
) -> dict[str, Path]:
    if tuple(roots) != ARM_NAMES:
        raise PrivilegedArtifactAuditError(
            f"root labels/order must be exactly {list(ARM_NAMES)}"
        )
    canonical = {
        arm: _canonical_directory(value, f"{arm} evaluation root")
        for arm, value in roots.items()
    }
    for left_index, left_arm in enumerate(ARM_NAMES):
        for right_arm in ARM_NAMES[left_index + 1 :]:
            left = canonical[left_arm]
            right = canonical[right_arm]
            if (
                left == right
                or left.is_relative_to(right)
                or right.is_relative_to(left)
            ):
                raise PrivilegedArtifactAuditError(
                    f"{left_arm} and {right_arm} roots must be separate, "
                    "non-nested directories"
                )
    return canonical


def _exact_contracts(record: ArtifactRecord, arm: str) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for key, expected in EXACT_INTEGER_CONTRACTS.items():
        values = _integer_values(
            record,
            key,
            expected_length=len(expected),
        )
        if values != expected:
            raise PrivilegedArtifactAuditError(
                f"{arm} rank {record.rank} {key} must be {expected}, "
                f"got {values}"
            )
        observed[key] = {
            "observed": list(values),
            "expected": list(expected),
            "pass": True,
        }
    _validate_final_inventory(record, arm)
    for key in DIAGNOSTIC_TENSORS:
        _get_tensor(record, key)
    return observed


def _validate_provenance(
    *,
    arm: str,
    root: Path,
    scope: Path,
) -> dict[str, Any]:
    path = _canonical_regular_file(
        root / PROVENANCE_NAME,
        f"{arm} privileged-video evaluation provenance",
    )
    payload = _load_json_strict(path)
    expected_parent = EXPECTED_PARENTS[arm]
    if (
        payload.get("schema_version") != 1
        or payload.get("kind")
        != "dual_video_diffusion_privileged_video_evaluation"
        or payload.get("status") != "visualization_completed"
        or payload.get("evaluation_only") is not True
        or payload.get("evaluation_optimizer_updates") != 0
        or payload.get("evaluation_total_observations") != 0
        or payload.get("artifact_iteration") != ARTIFACT_ITERATION
        or payload.get("viz_skip_batches") != 4
        or payload.get("evaluation_condition_sources")
        != ["autonomous", "off"]
        or payload.get("evaluation_nfe_steps") != list(NFE_STEPS)
        or payload.get("snapshot_written") is not False
        or payload.get("training_completion_written") is not False
    ):
        raise PrivilegedArtifactAuditError(
            f"{arm} evaluation provenance contract is invalid: {path}"
        )
    runtime = payload.get("runtime_intervention")
    if runtime != {
        "schedule_mode": "aligned",
        "tf_content_disabled": True,
        "tf_clock_disabled": True,
        "all_model_calls_advance_video": True,
    }:
        raise PrivilegedArtifactAuditError(
            f"{arm} provenance runtime intervention is not exact"
        )
    parent = payload.get("parent")
    if not isinstance(parent, dict):
        raise PrivilegedArtifactAuditError(
            f"{arm} provenance parent must be an object"
        )
    exact_parent_fields = {
        key: parent.get(key)
        for key in (
            "arm",
            "snapshot",
            "snapshot_sha256",
            "run_identity_sha256",
        )
    }
    if exact_parent_fields != expected_parent:
        raise PrivilegedArtifactAuditError(
            f"{arm} provenance parent mapping mismatch: "
            f"observed={exact_parent_fields}, expected={expected_parent}"
        )
    if (
        parent.get("completed_updates") != 200
        or isinstance(parent.get("total_observations"), bool)
        or not isinstance(parent.get("total_observations"), int)
        or parent["total_observations"] <= 0
    ):
        raise PrivilegedArtifactAuditError(
            f"{arm} provenance parent completion counters are invalid"
        )
    try:
        declared_artifact_root = Path(
            str(payload.get("artifact_root"))
        ).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PrivilegedArtifactAuditError(
            f"{arm} provenance artifact root is invalid"
        ) from exc
    if declared_artifact_root != scope:
        raise PrivilegedArtifactAuditError(
            f"{arm} provenance artifact root {declared_artifact_root} "
            f"does not equal discovered scope {scope}"
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "parent": dict(expected_parent),
        "completed_updates": 200,
        "total_observations": parent["total_observations"],
        "runtime_intervention": dict(runtime),
        "pass": True,
    }


def _compare(
    left_record: ArtifactRecord,
    left_key: str,
    right_record: ArtifactRecord,
    right_key: str,
    *,
    left_label: str,
    right_label: str,
    failure: str,
) -> dict[str, Any]:
    result = _comparison(
        _get_tensor(left_record, left_key),
        _get_tensor(right_record, right_key),
        left_label=left_label,
        right_label=right_label,
    )
    _require_pass(result, failure)
    return result


def audit(
    *,
    trained_matched_root: str | Path,
    trained_shuffled_root: str | Path,
    trained_off_root: str | Path,
) -> dict[str, Any]:
    """Audit three immutable evaluation roots and return signed evidence."""

    roots = _validate_roots(
        {
            "trained_matched": trained_matched_root,
            "trained_shuffled": trained_shuffled_root,
            "trained_off": trained_off_root,
        }
    )
    forbidden: dict[str, list[str]] = {}
    scopes: dict[str, Path] = {}
    records: dict[str, dict[int, ArtifactRecord]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        forbidden[arm] = _forbidden_training_outputs(roots[arm])
        if forbidden[arm]:
            raise PrivilegedArtifactAuditError(
                f"{arm} evaluation root contains forbidden training outputs: "
                + ", ".join(forbidden[arm])
            )
        scopes[arm], records[arm] = _discover(
            roots[arm],
            f"{arm} evaluation",
        )
        provenance[arm] = _validate_provenance(
            arm=arm,
            root=roots[arm],
            scope=scopes[arm],
        )

    dataset_sets = {
        arm: {record.dataset for record in records[arm].values()}
        for arm in ARM_NAMES
    }
    if len({tuple(sorted(value)) for value in dataset_sets.values()}) != 1:
        raise PrivilegedArtifactAuditError(
            f"artifact datasets differ across arms: {dataset_sets}"
        )

    rank_audits: list[dict[str, Any]] = []
    for rank in range(WORLD_SIZE):
        arm_contracts = {
            arm: _exact_contracts(records[arm][rank], arm)
            for arm in ARM_NAMES
        }

        cross_arm_identity: dict[str, Any] = {}
        for comparison_arm in ("trained_shuffled", "trained_off"):
            tensors: dict[str, Any] = {}
            for key in IDENTITY_TENSORS:
                tensors[key] = _compare(
                    records["trained_matched"][rank],
                    key,
                    records[comparison_arm][rank],
                    key,
                    left_label=f"trained_matched:{key}",
                    right_label=f"{comparison_arm}:{key}",
                    failure=(
                        f"rank {rank} trained_matched/{comparison_arm} "
                        f"identity mismatch for {key}"
                    ),
                )
            cross_arm_identity[
                f"trained_matched_vs_{comparison_arm.removeprefix('trained_')}"
            ] = {"pass": True, "tensors": tensors}

        runtime_noop: dict[str, Any] = {}
        for arm in ARM_NAMES:
            per_nfe: dict[str, Any] = {}
            for nfe in NFE_STEPS:
                per_nfe[str(nfe)] = {
                    "video_final": _compare(
                        records[arm][rank],
                        f"video_final_nfe_{nfe}",
                        records[arm][rank],
                        f"video_final_off_nfe_{nfe}",
                        left_label=f"{arm}:autonomous:video:nfe:{nfe}",
                        right_label=f"{arm}:off:video:nfe:{nfe}",
                        failure=(
                            f"{arm} rank {rank} NFE {nfe} autonomous/off "
                            "video finals are not bitwise identical"
                        ),
                    ),
                    "decoded_future": _compare(
                        records[arm][rank],
                        f"decoded_future_nfe_{nfe}",
                        records[arm][rank],
                        f"decoded_future_off_nfe_{nfe}",
                        left_label=f"{arm}:autonomous:decoded:nfe:{nfe}",
                        right_label=f"{arm}:off:decoded:nfe:{nfe}",
                        failure=(
                            f"{arm} rank {rank} NFE {nfe} autonomous/off "
                            "decoded finals are not bitwise identical"
                        ),
                    ),
                }
            runtime_noop[arm] = {"pass": True, "nfe": per_nfe}

        rank_audits.append(
            {
                "rank": rank,
                "dataset": records["trained_matched"][rank].dataset,
                "pass": True,
                "contracts": {"pass": True, "arms": arm_contracts},
                "cross_arm_input_identity": {
                    "pass": True,
                    "comparisons": cross_arm_identity,
                },
                "autonomous_off_runtime_noop": {
                    "pass": True,
                    "arms": runtime_noop,
                },
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "privileged_tf_video_bitwise_artifact_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sigma_convention": SIGMA_CONVENTION,
        "read_only_inputs": True,
        "overall_pass": True,
        "contracts": {
            "pass": True,
            "arm_names": {
                "observed": list(ARM_NAMES),
                "expected": list(ARM_NAMES),
                "pass": True,
            },
            "world_size": {
                "observed": {
                    arm: len(records[arm]) for arm in ARM_NAMES
                },
                "expected": WORLD_SIZE,
                "pass": True,
            },
            "paired_ranks": {
                "observed": {
                    arm: sorted(records[arm]) for arm in ARM_NAMES
                },
                "expected": list(range(WORLD_SIZE)),
                "pass": True,
            },
            "artifact_iteration": {
                "observed": ARTIFACT_ITERATION,
                "expected": ARTIFACT_ITERATION,
                "pass": True,
            },
            "source_codes": {
                "expected": list(SOURCE_CODES),
                "names": ["autonomous", "off"],
                "pass": True,
            },
            "nfe_steps": {
                "expected": list(NFE_STEPS),
                "pass": True,
            },
            "evaluation_noise_seed": {
                "expected": EVALUATION_NOISE_SEED,
                "pass": True,
            },
            "tf_content_disabled": {
                "condition_on_tf": 0,
                "condition_mode_code": 0,
                "pass": True,
            },
            "tf_clock_disabled": {
                "evaluation_disable_tf_clock": 1,
                "evaluation_tf_clock_enabled": 0,
                "pass": True,
            },
            "noncascade_all_video_schedule": {
                "cascade_stage_faithful_inference": 0,
                "evaluation_all_video_schedule": 1,
                "pass": True,
            },
            "raw_causal_inputs_present": {
                "raw_actions_present": 1,
                "raw_morphology_index_present": 1,
                "pass": True,
            },
            "cross_arm_input_identity": {"pass": True},
            "autonomous_off_runtime_noop": {"pass": True},
            "exact_parent_provenance": {
                "expected": EXPECTED_PARENTS,
                "pass": True,
            },
            "raw_action_morphology_input_identity": {
                "tensors": ["raw_actions", "raw_morphology_index"],
                "meaning": (
                    "exact raw causal inputs supplied to each checkpoint's "
                    "independently learned action encoder"
                ),
                "pass": True,
            },
            "learned_action_control_diagnostic": {
                "tensor": "z_control",
                "cross_arm_equality_required": False,
                "meaning": (
                    "checkpoint-specific learned action control retained for "
                    "mechanism analysis, not treated as a paired raw input"
                ),
                "pass": True,
            },
            "forbidden_training_outputs": {
                "observed": forbidden,
                "expected": {arm: [] for arm in ARM_NAMES},
                "pass": True,
            },
            "sidecar_hashes_schema_and_sigma_convention": {"pass": True},
        },
        "inputs": {
            arm: {
                "root": str(roots[arm]),
                "artifact_scope": str(scopes[arm]),
                "artifact_set_sha256": _artifact_set_sha256(records[arm]),
                "evaluation_provenance": provenance[arm],
                "ranks": [
                    _rank_artifact_inventory(record)
                    for _, record in sorted(records[arm].items())
                ],
            }
            for arm in ARM_NAMES
        },
        "rank_audits": rank_audits,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    payload["identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARM_NAMES:
        parser.add_argument(
            f"--{arm.replace('_', '-')}-root",
            required=True,
            help=(
                f"Fresh {arm} evaluation run root containing "
                f"{PROVENANCE_NAME}"
            ),
        )
    parser.add_argument(
        "--output",
        required=True,
        help="New external .json report path; must not already exist",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = {
        "trained_matched": args.trained_matched_root,
        "trained_shuffled": args.trained_shuffled_root,
        "trained_off": args.trained_off_root,
    }
    try:
        canonical_roots = [
            _canonical_directory(value, f"{arm} evaluation root")
            for arm, value in roots.items()
        ]
        output = _output_path(args.output, canonical_roots)
        if output.suffix.lower() != ".json":
            raise PrivilegedArtifactAuditError(
                "output path must end in .json"
            )
        payload = audit(
            trained_matched_root=roots["trained_matched"],
            trained_shuffled_root=roots["trained_shuffled"],
            trained_off_root=roots["trained_off"],
        )
        output_sha256 = _exclusive_json(output, payload)
    except (PrivilegedArtifactAuditError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": output_sha256,
                "identity_sha256": payload["identity_sha256"],
                "overall_pass": payload["overall_pass"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
