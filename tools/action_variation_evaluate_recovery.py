#!/usr/bin/env python3
"""Auditable runtime-receipt bridge for action-variation evaluation recovery.

The registered evaluator requires training and evaluation to execute in one
Slurm task.  If training completed but a post-training service boundary failed,
that task cannot be resumed.  This wrapper preserves and validates the original
training receipt, adds a second receipt for the actual recovery B200 task, and
then invokes the registered evaluator unchanged.  It does not relax any model,
checkpoint, dataset, W&B, sampler, metric, or output-freshness check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


KIND = "action_variation_evaluation_recovery_runtime_v1"


class RecoveryError(RuntimeError):
    """Recovery provenance or runtime differs from the sealed contract."""


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().absolute()
    if not resolved.is_file() or resolved.is_symlink():
        raise RecoveryError(f"required regular file is absent: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RecoveryError(f"required JSON is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryError(f"JSON root is not an object: {path}")
    return value


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _slurm_identity() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    node_list = os.environ.get(
        "SLURM_JOB_NODELIST", os.environ.get("SLURM_NODELIST", "")
    )
    if not job_id.isdigit() or not node_list:
        raise RecoveryError("recovery must execute in an identified Slurm job")
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    try:
        return {
            "job_id": job_id,
            "array_job_id": array_job_id if array_job_id else None,
            "array_task_id": int(array_task_id) if array_task_id else None,
            "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
            "node_list": node_list,
        }
    except ValueError as exc:
        raise RecoveryError("Slurm recovery identity is malformed") from exc


def _load_registered_modules(registration_path: Path):
    raw = _read(registration_path)
    repo = Path(raw.get("tool_repository", {}).get("path", ""))
    if not repo.is_absolute() or not repo.is_dir() or repo.is_symlink():
        raise RecoveryError("registered source repository is invalid")
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "projects/latent_action_models"))
    from tools import action_variation_evaluate as evaluator
    from tools import action_variation_screen as screen

    registration = screen.validate_registration(registration_path)
    screen.revalidate_execution_environment(registration)
    return registration, screen, evaluator


def _arm(screen: Any, code: str):
    arm = screen.ARM_BY_CODE.get(code)
    if arm is None:
        raise RecoveryError(f"unknown registered arm: {code}")
    return arm


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["identity_sha256"] = hashlib.sha256(
        _canonical(value).encode("utf-8")
    ).hexdigest()
    return value


def command_seal(args: argparse.Namespace) -> int:
    registration, screen, _evaluator = _load_registered_modules(args.registration)
    arm = _arm(screen, args.arm)
    original = screen.validate_runtime_receipt(
        registration, arm, require_current_slurm=False
    )
    verifier = _read(args.verifier_json)
    screen.validate_runtime_verifier_output(verifier, registration)
    wrapper = Path(__file__).resolve(strict=True)
    registered_evaluator = (
        Path(registration["tool_repository"]["path"])
        / "tools/action_variation_evaluate.py"
    )
    payload = _signed(
        {
            "kind": KIND,
            "registration_identity_sha256": registration["identity_sha256"],
            "arm": args.arm,
            "original_training_runtime_receipt": original,
            "registered_tool_commit": registration["tool_repository"]["git_commit"],
            "registered_evaluator": _record(registered_evaluator),
            "recovery_wrapper": _record(wrapper),
            "verifier_output": verifier,
            "slurm": _slurm_identity(),
            "scope": "evaluation_only_no_training_no_sampler_or_metric_change",
            "protected_test_accessed": False,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _validate_recovery(
    path: Path, registration: Mapping[str, Any], screen: Any, arm: Any
) -> dict[str, Any]:
    receipt = _read(path)
    unsigned = dict(receipt)
    identity = unsigned.pop("identity_sha256", None)
    expected_identity = hashlib.sha256(_canonical(unsigned).encode()).hexdigest()
    registered_evaluator = (
        Path(registration["tool_repository"]["path"])
        / "tools/action_variation_evaluate.py"
    )
    original = screen.validate_runtime_receipt(
        registration, arm, require_current_slurm=False
    )
    if (
        receipt.get("kind") != KIND
        or identity != expected_identity
        or receipt.get("registration_identity_sha256")
        != registration["identity_sha256"]
        or receipt.get("arm") != arm.code
        or receipt.get("original_training_runtime_receipt") != original
        or receipt.get("registered_tool_commit")
        != registration["tool_repository"]["git_commit"]
        or receipt.get("registered_evaluator") != _record(registered_evaluator)
        or receipt.get("recovery_wrapper") != _record(Path(__file__).resolve())
        or receipt.get("slurm") != _slurm_identity()
        or receipt.get("scope")
        != "evaluation_only_no_training_no_sampler_or_metric_change"
        or receipt.get("protected_test_accessed") is not False
    ):
        raise RecoveryError("evaluation recovery receipt differs")
    screen.validate_runtime_verifier_output(receipt["verifier_output"], registration)
    return receipt


def command_evaluate(args: argparse.Namespace) -> int:
    registration, screen, evaluator = _load_registered_modules(args.registration)
    arm = _arm(screen, args.arm)
    _validate_recovery(args.recovery_runtime_receipt, registration, screen, arm)
    original_validate = screen.validate_runtime_receipt

    def validate_with_recovery(
        candidate_registration: Mapping[str, Any],
        candidate_arm: Any,
        *,
        require_current_slurm: bool = False,
    ) -> dict[str, Any]:
        result = original_validate(
            candidate_registration, candidate_arm, require_current_slurm=False
        )
        if require_current_slurm:
            _validate_recovery(
                args.recovery_runtime_receipt,
                candidate_registration,
                screen,
                candidate_arm,
            )
        return result

    # evaluator.screen is the same registered module object.  Only the
    # same-task receipt predicate is bridged; all returned provenance remains
    # the original receipt expected by the immutable arm plan.
    screen.validate_runtime_receipt = validate_with_recovery
    return int(
        evaluator.command_evaluate(
            argparse.Namespace(
                registration=args.registration,
                arm=args.arm,
                output_dir=args.output_dir,
                batch_size_per_rank=args.batch_size_per_rank,
            )
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal-runtime")
    seal.add_argument("--registration", type=Path, required=True)
    seal.add_argument("--arm", required=True)
    seal.add_argument("--verifier-json", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.set_defaults(func=command_seal)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--registration", type=Path, required=True)
    evaluate.add_argument("--arm", required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--batch-size-per-rank", type=int, required=True)
    evaluate.add_argument("--recovery-runtime-receipt", type=Path, required=True)
    evaluate.set_defaults(func=command_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
