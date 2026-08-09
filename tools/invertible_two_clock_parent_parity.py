#!/usr/bin/env python3
"""Seal independently produced historical/current parent replay receipts.

This tool does not run a model or open any dataset.  An independent auditor
runs the untouched parent sampler in the clean historical and current source
worktrees, then this comparator verifies lineage, target blindness, and exact
NFE-1/2/4 tensor equality before producing the registration receipt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import invertible_two_clock_pilot as pilot  # noqa: E402


KIND_REPLAY = "ipq_tc1_parent_parity_replay"
KIND_COMPARISON = "ipq_tc1_parent_parity_comparison"
HISTORICAL_SAMPLER_GIT_BLOB = "abc6d4df165f684c2c93920d079585c600562dbb"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ParentParityError(RuntimeError):
    """A replay receipt is not exact, target-blind, or lineage-compatible."""


def _replay(
    path: Path,
    *,
    role: str,
    expected_source_commit: str,
    expected_current_commit: str,
) -> dict[str, Any]:
    value = pilot.read_json(path.resolve(strict=True), f"{role} replay receipt")
    if (
        not pilot.identity_valid(value)
        or value.get("kind") != KIND_REPLAY
        or value.get("source_role") != role
        or value.get("source_commit") != expected_source_commit
        or value.get("parent_snapshot_sha256") != pilot.PARENT_SNAPSHOT_SHA256
        or value.get("parent_run_identity_sha256")
        != pilot.PARENT_RUN_IDENTITY_SHA256
        or value.get("parent_canonical_model_state_sha256")
        != pilot.PARENT_CANONICAL_MODEL_STATE_SHA256
        or value.get("parent_resolved_config_sha256")
        != pilot.PARENT_RESOLVED_CONFIG_SHA256
        or value.get("nfe_grid") != list(pilot.NFE_GRID)
        or value.get("strict_model_load") is not True
        or value.get("deployable_target_blind_sampler") is not True
        or value.get("validation_array_opened") is not False
        or value.get("protected_test_accessed") is not False
        or value.get("teacher_feature_encoder_calls") != 0
        or not isinstance(value.get("input_sha256"), str)
        or SHA256_RE.fullmatch(value["input_sha256"]) is None
        or set(value.get("outputs", {})) != {str(nfe) for nfe in pilot.NFE_GRID}
    ):
        raise ParentParityError(f"{role} replay identity differs")
    if role == "historical":
        if value.get("sampler_git_blob") != HISTORICAL_SAMPLER_GIT_BLOB:
            raise ParentParityError("historical sampler blob differs")
    elif (
        value.get("source_commit") != expected_current_commit
        or value.get("sampler_sha256") != pilot.CURRENT_PARENT_SAMPLER_SHA256
    ):
        raise ParentParityError("current sampler source differs")
    for nfe in pilot.NFE_GRID:
        output = value["outputs"][str(nfe)]
        if not isinstance(output, Mapping) or set(output) != {
            "initial_noise_sha256",
            "final_latent_sha256",
            "decoded_uint8_sha256",
            "actual_wan_calls",
        }:
            raise ParentParityError(f"{role} NFE-{nfe} output schema differs")
        if output["actual_wan_calls"] != nfe or any(
            not isinstance(output[field], str)
            or SHA256_RE.fullmatch(output[field]) is None
            for field in (
                "initial_noise_sha256",
                "final_latent_sha256",
                "decoded_uint8_sha256",
            )
        ):
            raise ParentParityError(f"{role} NFE-{nfe} output differs")
    return value


def compare(
    historical_path: Path,
    current_path: Path,
    *,
    current_source_commit: str,
) -> dict[str, Any]:
    historical = _replay(
        historical_path,
        role="historical",
        expected_source_commit=pilot.PARENT_TRAINING_SOURCE_COMMIT,
        expected_current_commit=current_source_commit,
    )
    current = _replay(
        current_path,
        role="current",
        expected_source_commit=current_source_commit,
        expected_current_commit=current_source_commit,
    )
    if historical["input_sha256"] != current["input_sha256"]:
        raise ParentParityError("historical/current replay inputs differ")
    per_nfe = {}
    for nfe in pilot.NFE_GRID:
        left = historical["outputs"][str(nfe)]
        right = current["outputs"][str(nfe)]
        fields = (
            "initial_noise_sha256",
            "final_latent_sha256",
            "decoded_uint8_sha256",
        )
        exact = {field: left[field] == right[field] for field in fields}
        if not all(exact.values()):
            raise ParentParityError(f"historical/current NFE-{nfe} is not bit-exact")
        per_nfe[str(nfe)] = {**exact, "hashes": {field: left[field] for field in fields}}
    return pilot.identity_payload(
        {
            "kind": KIND_COMPARISON,
            "status": "PASS_BIT_EXACT",
            "parent_snapshot_sha256": pilot.PARENT_SNAPSHOT_SHA256,
            "parent_run_identity_sha256": pilot.PARENT_RUN_IDENTITY_SHA256,
            "parent_canonical_model_state_sha256": (
                pilot.PARENT_CANONICAL_MODEL_STATE_SHA256
            ),
            "parent_resolved_config_sha256": pilot.PARENT_RESOLVED_CONFIG_SHA256,
            "historical_source_commit": pilot.PARENT_TRAINING_SOURCE_COMMIT,
            "current_source_commit": current_source_commit,
            "historical_sampler_git_blob": HISTORICAL_SAMPLER_GIT_BLOB,
            "current_sampler_sha256": pilot.CURRENT_PARENT_SAMPLER_SHA256,
            "input_sha256": historical["input_sha256"],
            "nfe_grid": list(pilot.NFE_GRID),
            "per_nfe": per_nfe,
            "initial_noise_bit_exact": True,
            "final_latent_bit_exact": True,
            "decoded_uint8_bit_exact": True,
            "historical_receipt": pilot.file_record(historical_path),
            "current_receipt": pilot.file_record(current_path),
            "strict_model_load_both": True,
            "deployable_target_blind_sampler_both": True,
            "validation_array_opened": False,
            "protected_test_accessed": False,
            "teacher_feature_encoder_calls": 0,
        }
    )


def command_compare(args: argparse.Namespace) -> int:
    result = compare(
        args.historical,
        args.current,
        current_source_commit=args.current_source_commit,
    )
    pilot.exclusive_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--current-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=command_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
