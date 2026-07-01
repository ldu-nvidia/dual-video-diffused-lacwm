#!/usr/bin/env python3
"""Create or validate immutable identity for a guarded full training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

MIN_SUPPORTED_GPU_MEMORY_MIB = 78_000
DATASET_NAMES = ("droid", "egodex", "agibot", "abc")
STRICT_DATA_POLICY = "strict"
FAST_DATA_POLICY = "files_only_user_waived_v1"
DATA_VALIDATION_POLICIES = (STRICT_DATA_POLICY, FAST_DATA_POLICY)
FAST_AUTHORIZATION_KIND = "lacwm_fast_training_authorization"
FAST_WAIVER_KIND = "lacwm_user_authorized_fast_mixed_overlay"
FAST_MIXED_REPORT_KIND = "lacwm_real_mixed_stateful_dataloader_smoke"


def canonical(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def absolute_executable(value: str) -> str:
    # Keep the venv entrypoint path rather than resolving its ``python`` symlink
    # to a shared managed/base interpreter.
    return str(Path(value).expanduser().absolute())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def supported_gpu_memory_mib(value: str) -> int:
    parsed = positive_int(value)
    if parsed < MIN_SUPPORTED_GPU_MEMORY_MIB:
        raise argparse.ArgumentTypeError(
            "minimum GPU memory must be at least "
            f"{MIN_SUPPORTED_GPU_MEMORY_MIB} MiB"
        )
    return parsed


def supported_world_size(value: str) -> int:
    parsed = positive_int(value)
    if parsed > 256 or parsed % 8 != 0:
        raise argparse.ArgumentTypeError(
            "world size must be a multiple of 8 between 8 and 256"
        )
    return parsed


def supported_node_count(value: str) -> int:
    parsed = positive_int(value)
    if not 1 <= parsed <= 32:
        raise argparse.ArgumentTypeError("node count must be between 1 and 32")
    return parsed


def eight_gpus_per_node(value: str) -> int:
    parsed = positive_int(value)
    if parsed != 8:
        raise argparse.ArgumentTypeError("this B200 profile requires 8 GPUs per node")
    return parsed


def dataset_stage(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9._-]+", value) is None:
        raise argparse.ArgumentTypeError(
            "dataset stage may contain only letters, digits, dot, underscore, and dash"
        )
    return value


def normalized_dataset_names(values: list[str]) -> list[str]:
    if len(values) != len(set(values)):
        raise RuntimeError("dataset names must not contain duplicates")
    selected = set(values)
    if not selected or not selected <= set(DATASET_NAMES):
        raise RuntimeError(
            "datasets must be a non-empty subset of " + ", ".join(DATASET_NAMES)
        )
    return [name for name in DATASET_NAMES if name in selected]


def expected_data_sources(data_root: str) -> dict[str, str]:
    root = Path(data_root).expanduser().resolve(strict=False)
    return {
        "droid": str((root / "droid_lerobot").resolve(strict=False)),
        "egodex": str(
            (root / "egodex_cdn" / "manifest.csv").resolve(strict=False)
        ),
        "agibot": str((root / "agibot" / "manifest.csv").resolve(strict=False)),
        "abc": str((root / "abc_pp" / "manifest.txt").resolve(strict=False)),
    }


def expected_payload(args: argparse.Namespace) -> dict:
    if args.world_size != args.node_count * args.gpus_per_node:
        raise RuntimeError(
            "world size does not match node topology: "
            f"{args.world_size} != {args.node_count} * {args.gpus_per_node}"
        )
    if args.warmup_steps >= args.max_iter:
        raise RuntimeError("warmup steps must be less than max iterations")
    data = json.loads(args.data_report.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime_report.read_text(encoding="utf-8"))
    smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    dataset_names = normalized_dataset_names(args.datasets)
    reports = data.get("reports", [])
    if not isinstance(reports, list) or not all(
        isinstance(report, dict) for report in reports
    ):
        raise RuntimeError("data report reports must be a list of objects")
    report_names = [str(report.get("name")) for report in reports]
    if len(report_names) != len(set(report_names)) or set(report_names) != set(
        dataset_names
    ):
        raise RuntimeError(
            "data report dataset names do not exactly match selected datasets: "
            f"report={report_names!r}, selected={dataset_names!r}"
        )
    expected_sources = expected_data_sources(args.data_root)
    fingerprints: dict[str, dict] = {}
    for report in reports:
        name = str(report["name"])
        source = report.get("source")
        if not isinstance(source, str) or canonical(source) != expected_sources[name]:
            raise RuntimeError(
                f"data report source for {name!r} is {source!r}, "
                f"expected {expected_sources[name]!r}"
            )
        fingerprint = report.get("fingerprint")
        if not isinstance(fingerprint, dict) or not fingerprint:
            raise RuntimeError(
                f"data report fingerprint for {name!r} must be a non-empty object"
            )
        fingerprints[name] = fingerprint

    invocation = data.get("invocation")
    if not isinstance(invocation, dict):
        raise RuntimeError("data report invocation must be an object")
    if invocation.get("datasets") != dataset_names:
        raise RuntimeError(
            "data report invocation datasets do not exactly match selected datasets"
        )
    invocation_sources = invocation.get("sources")
    if not isinstance(invocation_sources, dict) or any(
        canonical(str(invocation_sources.get(name, "/nonexistent")))
        != expected_sources[name]
        for name in dataset_names
    ):
        raise RuntimeError("data report invocation sources do not match selected data")
    selected_invocation = {
        "data_root": canonical(args.data_root),
        "datasets": dataset_names,
        "min_timesteps": invocation.get("min_timesteps"),
        "validate_all": invocation.get("validate_all"),
        "files_only": invocation.get("files_only"),
        "caps": {
            name: invocation.get("caps", {}).get(name)
            for name in dataset_names
        },
        "expected": {
            name: invocation.get("expected", {}).get(name)
            for name in dataset_names
        },
        "sources": {name: expected_sources[name] for name in dataset_names},
    }
    if "agibot" in dataset_names:
        selected_invocation["agibot_profile"] = invocation.get("agibot_profile")
        selected_invocation["agibot_roots"] = invocation.get("agibot_roots")
    fast_evidence: dict[str, object] | None = None
    if args.data_validation_policy == FAST_DATA_POLICY:
        if not args.fast_training_authorization or not args.mixed_loader_report:
            raise RuntimeError(
                "files-only user-waived identity requires authorization and mixed-loader reports"
            )
        fast_paths = {
            "authorization": args.fast_training_authorization,
            "files-only report": args.data_report,
            "mixed-loader report": args.mixed_loader_report,
            "gradient report": args.smoke_report,
        }
        for label, path in fast_paths.items():
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(
                    f"fast {label} must be a nonempty non-symlink regular file: {path}"
                )
        if invocation.get("files_only") is not True:
            raise RuntimeError("fast identity requires a files-only data report")
        allowed_external_roots = invocation.get("allowed_external_roots")
        canonical_data_parent = Path(canonical(args.data_root)).parent
        expected_external_roots = {
            "egodex": canonical(str(canonical_data_parent / "egodex_cdn")),
            "abc": canonical(str(canonical_data_parent / "abc_pp")),
        }
        if (
            not isinstance(allowed_external_roots, dict)
            or {
                name: canonical(
                    str(allowed_external_roots.get(name, "/nonexistent"))
                )
                for name in expected_external_roots
            }
            != expected_external_roots
        ):
            raise RuntimeError(
                "fast data report external manifest roots do not match the canonical corpus"
            )
        selected_invocation["allowed_external_roots"] = expected_external_roots
        authorization = json.loads(
            args.fast_training_authorization.read_text(encoding="utf-8")
        )
        mixed = json.loads(args.mixed_loader_report.read_text(encoding="utf-8"))
        waiver_path = Path(args.data_root) / "fast_validation_waiver.json"
        if waiver_path.is_symlink() or not waiver_path.is_file():
            raise RuntimeError(
                "original fast validation waiver must be a non-symlink regular file"
            )
        waiver = json.loads(waiver_path.read_text(encoding="utf-8"))
        if (
            not isinstance(authorization, dict)
            or authorization.get("schema_version") != 1
            or authorization.get("kind") != FAST_AUTHORIZATION_KIND
            or authorization.get("training_authorized") is not True
            or authorization.get("policy") != FAST_DATA_POLICY
            or authorization.get("branch") != "lora"
            or authorization.get("expected_commit") != args.git_commit
            or authorization.get("authorized_by") != "user"
            or not isinstance(authorization.get("authorization_basis"), str)
            or not str(authorization.get("authorization_basis", "")).strip()
            or authorization.get("authorization_scope")
            != "one_branch_one_commit_one_fast_overlay"
            or authorization.get("source_order")
            != ["Droid", "EgoDex", "Agibot", "ABC"]
            or authorization.get("source_lengths")
            != [10_000, 10_000, 5_671, 10_000]
            or authorization.get("total_episodes") != 35_671
            or canonical(str(authorization.get("data_root", "")))
            != canonical(args.data_root)
        ):
            raise RuntimeError("fast training authorization is invalid or ambiguously bound")
        if canonical(str(authorization.get("certificate_path", ""))) != canonical(
            str(args.fast_training_authorization)
        ):
            raise RuntimeError("fast training authorization was moved from its bound path")
        if args.fast_training_authorization.stat().st_mode & 0o222:
            raise RuntimeError("fast training authorization must be read-only")
        inputs = authorization.get("inputs")
        if not isinstance(inputs, dict) or set(inputs) != {
            "waiver",
            "files_only_report",
            "mixed_loader_report",
            "gradient_report",
        }:
            raise RuntimeError("fast training authorization input bindings are incomplete")
        expected_inputs = {
            "waiver": waiver_path,
            "files_only_report": args.data_report,
            "mixed_loader_report": args.mixed_loader_report,
            "gradient_report": args.smoke_report,
        }
        for name, expected_path in expected_inputs.items():
            record = inputs.get(name)
            recorded_path = Path(str(record.get("path", ""))).expanduser() if isinstance(record, dict) else Path()
            if (
                not isinstance(record, dict)
                or recorded_path.is_symlink()
                or not recorded_path.is_file()
                or canonical(str(record.get("path", "")))
                != canonical(str(expected_path))
                or record.get("sha256") != sha256(expected_path)
                or record.get("size") != expected_path.stat().st_size
            ):
                raise RuntimeError(
                    f"fast training authorization input binding changed: {name}"
                )
        if (
            not isinstance(waiver, dict)
            or waiver.get("schema_version") != 1
            or waiver.get("kind") != FAST_WAIVER_KIND
            or waiver.get("logical_read_skipped") is not True
            or waiver.get("strict_validated") is not False
            or waiver.get("training_authorized") is not False
            or waiver.get("selected_episodes") != 5_671
            or waiver.get("required_payloads") != 39_697
            or not isinstance(waiver.get("metadata_fingerprint_sha256"), str)
            or len(waiver["metadata_fingerprint_sha256"]) != 64
        ):
            raise RuntimeError("original fast validation waiver is invalid")
        authorized_agibot = authorization.get("agibot")
        if (
            not isinstance(authorized_agibot, dict)
            or authorized_agibot.get("metadata_fingerprint_sha256")
            != waiver.get("metadata_fingerprint_sha256")
            or authorized_agibot.get("required_payloads")
            != waiver.get("required_payloads")
            or authorized_agibot.get("required_payload_bytes")
            != waiver.get("required_payload_bytes")
        ):
            raise RuntimeError("authorization AgiBot seal differs from the original waiver")
        authorization_validation = authorization.get("validation")
        if not isinstance(authorization_validation, dict) or any(
            not isinstance(authorization_validation.get(name), dict)
            for name in (
                "original_waiver",
                "files_only",
                "mixed_loader",
                "real_gradient",
            )
        ):
            raise RuntimeError("authorization validation summaries are incomplete")
        if (
            authorization_validation["original_waiver"].get(
                "logical_read_skipped"
            )
            is not True
            or authorization_validation["original_waiver"].get(
                "strict_validated"
            )
            is not False
            or authorization_validation["original_waiver"].get(
                "training_authorized"
            )
            is not False
            or authorization_validation["files_only"].get("passed") is not True
            or authorization_validation["files_only"].get("files_only") is not True
            or authorization_validation["mixed_loader"].get("passed") is not True
            or authorization_validation["real_gradient"].get("passed") is not True
        ):
            raise RuntimeError("authorization validation summaries did not all pass")
        validation = mixed.get("validation", {}) if isinstance(mixed, dict) else {}
        mixed_data = validation.get("data", {}) if isinstance(validation, dict) else {}
        mixed_batches = validation.get("mix", {}) if isinstance(validation, dict) else {}
        resume = validation.get("resume", {}) if isinstance(validation, dict) else {}
        observed_sources = (
            mixed_batches.get("observed_source_counts")
            if isinstance(mixed_batches, dict)
            else None
        )
        if (
            not isinstance(mixed, dict)
            or mixed.get("schema_version") != 1
            or mixed.get("kind") != FAST_MIXED_REPORT_KIND
            or mixed.get("status") != "passed"
            or mixed.get("git_commit") != args.git_commit
            or mixed.get("git_status") != ""
            or canonical(str(mixed.get("requested_data_root", "")))
            != canonical(args.data_root)
            or canonical(str(mixed_data.get("root", ""))) != canonical(args.data_root)
            or mixed_data.get("source_order") != ["Droid", "EgoDex", "Agibot", "ABC"]
            or mixed_data.get("source_lengths") != [10_000, 10_000, 5_671, 10_000]
            or mixed_data.get("total_episodes") != 35_671
            or not isinstance(mixed_batches.get("batches_checked"), int)
            or mixed_batches.get("batches_checked", 0) <= 0
            or mixed_batches.get("mixed_batches")
            != mixed_batches.get("batches_checked")
            or not isinstance(observed_sources, dict)
            or set(observed_sources) != {"Droid", "EgoDex", "Agibot", "ABC"}
            or any(int(observed_sources.get(name, 0)) <= 0 for name in observed_sources)
            or resume.get("exact_continuation") is not True
            or resume.get("reference_signature") != resume.get("restored_signature")
        ):
            raise RuntimeError("mixed-loader evidence is not passing and identity-bound")
        state_path = Path(str(resume.get("state_path", ""))).expanduser()
        if (
            not state_path.is_file()
            or state_path.is_symlink()
            or state_path.stat().st_size != resume.get("state_size")
            or sha256(state_path) != resume.get("state_sha256")
        ):
            raise RuntimeError("mixed-loader state artifact changed after validation")
        fast_evidence = {
            "authorization": {
                "path": canonical(str(args.fast_training_authorization)),
                "sha256": sha256(args.fast_training_authorization),
            },
            "original_waiver": {
                "path": canonical(str(waiver_path)),
                "sha256": sha256(waiver_path),
                "metadata_fingerprint_sha256": waiver.get(
                    "metadata_fingerprint_sha256"
                ),
            },
            "mixed_loader": {
                "path": canonical(str(args.mixed_loader_report)),
                "sha256": sha256(args.mixed_loader_report),
                "state_sha256": resume.get("state_sha256"),
                "reference_signature": resume.get("reference_signature"),
            },
        }
    elif args.fast_training_authorization or args.mixed_loader_report:
        raise RuntimeError("strict identity forbids fast-policy evidence")
    elif invocation.get("files_only", False) is not False:
        raise RuntimeError("strict identity requires a content-validation report")
    effective_global_batch_size = (
        args.batch_size * args.gradient_accumulation_steps * args.world_size
    )
    payload = {
        "schema_version": 6,
        "dataset_stage": args.dataset_stage,
        "dataset_names": dataset_names,
        "variant": args.variant,
        "git_commit": args.git_commit,
        # ``batch_size`` is the physical per-rank microbatch.  Keep all three
        # batching values explicit so a resume cannot silently alter optimizer
        # semantics while retaining the same nominal update count.
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "node_count": args.node_count,
        "gpus_per_node": args.gpus_per_node,
        "world_size": args.world_size,
        "effective_global_batch_size": effective_global_batch_size,
        "schedule": {
            "max_iter": args.max_iter,
            "warmup_steps": args.warmup_steps,
            "log_every": args.log_every,
            "save_every": args.save_every,
            "val_every": args.val_every,
            "viz_every": args.viz_every,
        },
        "gpu_profile": {
            "model": "B200",
            "minimum_memory_mib": args.min_gpu_memory_mib,
        },
        "run_name": args.run_name,
        "paths": {
            "python": absolute_executable(args.python),
            "wan_dir": canonical(args.wan_dir),
            "videox_home": canonical(args.videox_home),
            "data_root": canonical(args.data_root),
            "run_root": canonical(args.run_root),
            "run_dir": canonical(args.run_dir),
        },
        "wandb": {
            "mode": args.wandb_mode,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "run_id": args.wandb_run_id,
        },
        "data": {
            "validation_policy": args.data_validation_policy,
            # Strict identities historically select only the requested dataset
            # rows from a validation report, so an unrelated source entry must
            # not invalidate an existing strict identity.  The fast policy is
            # different: its files-only report is one of the exact certificate
            # inputs and therefore must be bound byte-for-byte.
            "data_report_sha256": (
                sha256(args.data_report) if fast_evidence is not None else None
            ),
            "fingerprints": fingerprints,
            "invocation": selected_invocation,
            "validator_sha256": data.get("validator_sha256"),
            "fast_evidence": fast_evidence,
        },
        "runtime": {
            "python": runtime.get("python"),
            "packages": runtime.get("packages"),
            "distributions": runtime.get("distributions"),
            "environment": runtime.get("environment"),
            "videox_commit": runtime.get("videox_commit"),
            "videox_status": runtime.get("videox_status"),
            "weights": runtime.get("weights"),
        },
        "gradient_smoke": {
            "sha256": sha256(args.smoke_report),
            "kind": smoke.get("kind"),
            "variant": smoke.get("variant"),
        },
    }
    canonical_json = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    payload["identity_sha256"] = hashlib.sha256(canonical_json).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "validate"))
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--dataset-stage", type=dataset_stage, default="all-four")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
    )
    parser.add_argument("--variant", choices=("latent", "explicit"), required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--batch-size", type=positive_int, required=True)
    parser.add_argument(
        "--gradient-accumulation-steps", type=positive_int, required=True
    )
    parser.add_argument("--max-iter", type=positive_int, default=60_000)
    parser.add_argument("--warmup-steps", type=nonnegative_int, default=2_000)
    parser.add_argument("--log-every", type=positive_int, default=50)
    parser.add_argument("--save-every", type=positive_int, default=1_000)
    parser.add_argument("--val-every", type=positive_int, default=1_000)
    parser.add_argument("--viz-every", type=positive_int, default=1_000)
    parser.add_argument("--world-size", type=supported_world_size, default=8)
    parser.add_argument("--node-count", type=supported_node_count, default=1)
    parser.add_argument("--gpus-per-node", type=eight_gpus_per_node, default=8)
    parser.add_argument(
        "--min-gpu-memory-mib", type=supported_gpu_memory_mib, required=True
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--wan-dir", required=True)
    parser.add_argument("--videox-home", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wandb-mode", required=True)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-id", default="")
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument(
        "--data-validation-policy",
        choices=DATA_VALIDATION_POLICIES,
        default=STRICT_DATA_POLICY,
    )
    parser.add_argument("--fast-training-authorization", type=Path)
    parser.add_argument("--mixed-loader-report", type=Path)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    args = parser.parse_args(argv)

    expected = expected_payload(args)
    if args.action == "create":
        if args.identity.exists():
            raise RuntimeError(f"refusing to replace existing identity: {args.identity}")
        payload = dict(expected)
        payload["state"] = "prelaunch"
        temporary = args.identity.with_suffix(args.identity.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.identity)
        print(f"Created run identity: {args.identity}")
        return 0

    actual = json.loads(args.identity.read_text(encoding="utf-8"))
    problems = [
        key for key, value in expected.items() if actual.get(key) != value
    ]
    if problems:
        raise RuntimeError(f"run identity mismatch for fields: {problems}")
    print(f"Validated run identity: {args.identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
