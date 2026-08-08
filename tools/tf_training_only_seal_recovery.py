#!/usr/bin/env python3
"""Operationally recover a TFREG seal after missing Hydra environment exports.

The registered scientific source remains the implementation under
``--source-repo``.  This controller only restores immutable paths already
bound in registration and switches the arm-specific W&B identity immediately
before the original config validator resolves that arm's Hydra interpolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--original-failed-job", action="append", default=[])
    args = parser.parse_args()

    repo = args.source_repo.resolve(strict=True)
    registration_path = args.registration.resolve(strict=True)
    if (
        not repo.is_dir()
        or repo.is_symlink()
        or _git(repo, "rev-parse", "HEAD") != args.expected_commit
        or _git(repo, "status", "--porcelain", "--untracked-files=all")
    ):
        raise SystemExit("registered source repository is not exact and clean")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    source = registration.get("source", {})
    study = Path(registration.get("study_root", ""))
    if (
        source.get("git_commit") != args.expected_commit
        or Path(source.get("path", "")).resolve(strict=True) != repo
        or registration_path != (study / "registration.json").resolve(strict=True)
        or registration.get("design", {}).get("protected_test_accessed") is not False
    ):
        raise SystemExit("registration/source boundary differs")

    os.environ.update(
        {
            "TFREG_PARENT_SNAPSHOT": registration["parent_snapshot"]["path"],
            "TFREG_TRAIN_MANIFEST": registration["training"]["manifest"]["path"],
            "TFREG_TRAIN_METADATA": registration["training"]["metadata"]["path"],
            "TFREG_VAL_MANIFEST": registration["validation"]["manifest"]["path"],
            "TFREG_VAL_METADATA": registration["validation"]["metadata"]["path"],
            "TFREG_RUN_ROOT": str(study / "training"),
        }
    )
    sys.path[:0] = [str(repo), str(repo / "projects/latent_action_models")]
    from tools import tf_training_only_screen as screen  # noqa: PLC0415

    original_validate = screen.validate_resolved_config

    def validate_with_registered_arm_identity(config, observed_registration, arm):
        os.environ["TFREG_RUN_IDENTITY"] = screen.arm_identity(
            observed_registration, arm
        )
        return original_validate(config, observed_registration, arm)

    screen.validate_resolved_config = validate_with_registered_arm_identity
    result = int(
        screen.command_seal(SimpleNamespace(registration=registration_path))
    )
    seal = study / "post_training_seal.json"
    if result != 0 or not seal.is_file() or seal.is_symlink():
        raise SystemExit("registered seal recovery did not complete")

    controller = Path(__file__).resolve(strict=True)
    payload = {
        "kind": "tfreg-post-training-seal-operational-recovery-v1",
        "registration_identity_sha256": registration["identity_sha256"],
        "registered_source_commit": args.expected_commit,
        "registered_scientific_configuration_changed": False,
        "recovery_scope": (
            "restore registered Hydra environment and arm-specific run identity"
        ),
        "original_failed_job_ids": list(args.original_failed_job),
        "controller": {
            "path": str(controller),
            "sha256": _sha256(controller),
            "bytes": controller.stat().st_size,
        },
        "post_training_seal": {
            "path": str(seal),
            "sha256": _sha256(seal),
            "bytes": seal.stat().st_size,
        },
        "protected_test_accessed": False,
    }
    payload["identity_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _exclusive_json(args.receipt, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
