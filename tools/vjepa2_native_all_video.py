#!/usr/bin/env python3
"""Post-study native-all-video evaluation for the faithful V-JEPA cascade.

The immutable training study uses a V-JEPA-first cascade at inference.  With a
50/50 split, a nominal ``2K``-call cascade advances the video state only ``K``
times.  This evaluator applies a runtime-only intervention:

* every one of ``K`` Wan calls advances the native video scheduler;
* VPM and A1 are evaluated with auxiliary state/clock injection disabled;
* J1 is evaluated both with aligned generated V-JEPA conditioning and with the
  same-checkpoint off intervention.

The preregistered K=4 screen is conjunctive.  ``A1/off`` versus ``VPM/off``
must show a material training-objective effect, ``J1/aligned`` versus
``J1/off`` must show a material generated-feature-use effect, and
``J1/aligned`` versus ``VPM/off`` must show a material end-to-end effect.  Each
contrast uses the same-NFE attribution gate: temporal-MSE relative-improvement
CI-low at least 3%, with video-NMSE and decoded-MSE CI-lows above -1%.

Validation is always evaluated first on the immutable 64 clips.  A lockbox
invocation fails before reading any lockbox registration/path unless the exact
validation report says the preregistered K=4 primary gate passed.  No NFE or
arm is selected using lockbox data.

For K in {1,2,4}, validation also runs cascade-off at 2K calls and requires the
float16 video-latent endpoint and uint8 decoded endpoint to hash-identically to
native-all-video off at K.  Runtime instrumentation records actual Wan calls,
video scheduler calls, auxiliary Euler calls, nonzero auxiliary transitions,
and the schedule nodes.  The evaluator never changes model or dataset code,
never trains, and never loads an online V-JEPA teacher.

LACWM convention: sigma=1 is Gaussian noise and sigma=0 is clean.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

try:
    from tools import evaluate_vjepa2_quality as quality
    from tools import vjepa2_frontier_lockbox as frontier_lockbox
    from tools import vjepa2_nfe_frontier as frontier
except ModuleNotFoundError:  # Direct ``python tools/...`` invocation.
    import evaluate_vjepa2_quality as quality
    import vjepa2_frontier_lockbox as frontier_lockbox
    import vjepa2_nfe_frontier as frontier


SCHEMA_VERSION = 1
KIND_ROW = "vjepa2_native_all_video_clip"
KIND_ENDPOINT_AUDIT = "vjepa2_native_all_video_endpoint_audit"
KIND_RANK = "vjepa2_native_all_video_rank"
KIND_REPORT = "vjepa2_native_all_video_report"

TRAINING_COMMIT = "656086686dae723c942a4209a9d71cdb17ed6ccc"
FINAL_UPDATE = 1000
EXPECTED_WORLD_SIZE = 8
DEFAULT_BATCH_SIZE_PER_RANK = 2
EXPECTED_SPLIT_CLIPS = {"validation": 64, "lockbox": 128}
SAMPLE_ID_OFFSETS = {"validation": 4_000_000, "lockbox": 5_000_000}

ARMS = ("VPM", "A1", "J1")
ARM_NAMES = {
    "VPM": "vpm_parameter_matched_video",
    "A1": "a1_auxiliary_objective_only",
    "J1": "j1_joint_auxiliary_leads",
}
RUNTIME_SOURCES = {
    "VPM": ("off",),
    "A1": ("off",),
    "J1": ("aligned", "off"),
}
SAMPLER_SOURCE = {"aligned": "autonomous", "off": "off"}
NFE_GRID = (1, 2, 4, 6, 8, 12, 20)
ENDPOINT_AUDIT_K = (1, 2, 4)
PRIMARY_CLAIM_NFE = 4
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 1234
COMPOSITE_COMPONENTS = {
    "training_objective": {
        "left": {"arm": "A1", "runtime_source": "off"},
        "reference": {"arm": "VPM", "runtime_source": "off"},
    },
    "generated_feature_use": {
        "left": {"arm": "J1", "runtime_source": "aligned"},
        "reference": {"arm": "J1", "runtime_source": "off"},
    },
    "end_to_end": {
        "left": {"arm": "J1", "runtime_source": "aligned"},
        "reference": {"arm": "VPM", "runtime_source": "off"},
    },
}

PRIMARY_METRIC = frontier.PRIMARY_METRIC
GUARDRAIL_METRICS = frontier.GUARDRAIL_METRICS
CLAIM_METRICS = frontier.CLAIM_METRICS


class NativeAllVideoError(RuntimeError):
    """Raised when post-study evidence is incomplete or incomparable."""


@dataclass(frozen=True)
class ArmInput:
    code: str
    run_dir: Path
    arm_manifest_path: Path
    arm_manifest: dict[str, Any]
    stage_manifest_path: Path
    stage_manifest: dict[str, Any]
    stage_outcome_path: Path
    stage_outcome: dict[str, Any]
    resolved_config: Path
    snapshot: Path
    snapshot_sha256: str


@dataclass(frozen=True)
class SplitInput:
    name: str
    expected_clips: int
    dataset_config_key: str
    manifest: Path
    cache_metadata: Path
    cache_arrays: dict[str, dict[str, Any]]
    descriptors: tuple[dict[str, Any], ...]
    lockbox_registration: dict[str, Any] | None
    validation_report: dict[str, Any] | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def _identity_valid(payload: Mapping[str, Any]) -> bool:
    recorded = payload.get("identity_sha256")
    if not isinstance(recorded, str) or frontier.SHA256_RE.fullmatch(recorded) is None:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest() == recorded


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise NativeAllVideoError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise NativeAllVideoError(
            f"{label} must be a non-symlink directory: {path}"
        )
    return path.resolve(strict=True)


def _canonical_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise NativeAllVideoError(f"{label} is missing: {path}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
    ):
        raise NativeAllVideoError(
            f"{label} must be a nonempty, non-symlink regular file: {path}"
        )
    return path.resolve(strict=True)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise NativeAllVideoError(
                    f"{label} contains duplicate key {key!r}: {path}"
                )
            payload[key] = value
        return payload

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {token}")
            ),
        )
    except NativeAllVideoError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise NativeAllVideoError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise NativeAllVideoError(f"{label} must contain one JSON object: {path}")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise NativeAllVideoError(
                        f"{label} has blank line {line_number}: {path}"
                    )
                value = json.loads(
                    line,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(f"non-finite constant {token}")
                    ),
                )
                if not isinstance(value, dict):
                    raise NativeAllVideoError(
                        f"{label} row {line_number} is not an object"
                    )
                rows.append(value)
    except NativeAllVideoError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise NativeAllVideoError(f"{label} is invalid JSONL: {path}") from exc
    if not rows:
        raise NativeAllVideoError(f"{label} contains no rows: {path}")
    return rows


def _exclusive_bytes(path: Path, content: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise NativeAllVideoError(f"refusing to overwrite output: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    _exclusive_bytes(path, _canonical_json(payload) + b"\n")


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise NativeAllVideoError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _validate_clean_evaluator(
    repo: Path,
    *,
    training_commit: str,
    evaluator_commit: str,
) -> dict[str, Any]:
    if training_commit != TRAINING_COMMIT:
        raise NativeAllVideoError(
            f"training commit must be the frozen faithful study commit "
            f"{TRAINING_COMMIT}"
        )
    if frontier.COMMIT_RE.fullmatch(evaluator_commit) is None:
        raise NativeAllVideoError("evaluator commit is not a full lowercase SHA")
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != evaluator_commit:
        raise NativeAllVideoError(
            f"evaluator checkout commit differs: {actual} != {evaluator_commit}"
        )
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise NativeAllVideoError("evaluator checkout must be clean")
    roots = {
        Path(module.__file__).resolve().parents[1]
        for module in (quality, frontier, frontier_lockbox)
    }
    roots.add(Path(__file__).resolve().parents[1])
    if roots != {repo}:
        raise NativeAllVideoError(
            "evaluator and imported scientific helpers are from different checkouts"
        )
    try:
        compatibility = frontier.git_inference_compatibility(
            repo,
            training_commit=training_commit,
            tool_commit=evaluator_commit,
        )
    except frontier.FrontierError as exc:
        raise NativeAllVideoError(str(exc)) from exc
    if compatibility.get("inference_critical_paths_unchanged") is not True:
        raise NativeAllVideoError("inference-critical source differs from training")
    return compatibility


def _validate_study(
    study_root: Path,
    *,
    training_commit: str,
) -> tuple[Path, dict[str, Any]]:
    study_path = _canonical_file(study_root / "study_manifest.json", "study manifest")
    study = _read_json(study_path, "study manifest")
    if (
        not _identity_valid(study)
        or study.get("kind") != "vjepa2_controlled_video_diffusion_study"
        or study.get("inputs", {}).get("repository", {}).get("git_commit")
        != training_commit
    ):
        raise NativeAllVideoError("study manifest identity differs")
    arms = study.get("arms")
    if not isinstance(arms, Mapping):
        raise NativeAllVideoError("study manifest lacks arm definitions")
    by_code = {
        str(value.get("code")): value
        for value in arms.values()
        if isinstance(value, Mapping)
    }
    for code in ARMS:
        arm = by_code.get(code)
        if not isinstance(arm, Mapping) or arm.get("name") != ARM_NAMES[code]:
            raise NativeAllVideoError(f"study arm definition differs for {code}")
    protocol = study.get("inference", {}).get("quality_protocol", {})
    if (
        protocol.get("deployable_sources_use_history_only_public_entrypoint")
        is not True
        or protocol.get("temporal_metric_includes_history_to_first_future_boundary")
        is not True
        or study.get("clock", {}).get("convention")
        != "sigma=1 noise, sigma=0 clean"
    ):
        raise NativeAllVideoError("study inference/clock contract differs")
    return study_path, study


def _validate_arm_input(
    study_root: Path,
    study: Mapping[str, Any],
    *,
    code: str,
    training_commit: str,
    verify_snapshot_sha256: bool,
) -> ArmInput:
    run_dir = _canonical_directory(study_root / ARM_NAMES[code], f"{code} run")
    arm_manifest_path = _canonical_file(
        run_dir / "arm_manifest.json", f"{code} arm manifest"
    )
    stage_manifest_path = _canonical_file(
        run_dir / f"stage_manifest_update_{FINAL_UPDATE:04d}.json",
        f"{code} final stage manifest",
    )
    stage_outcome_path = _canonical_file(
        run_dir / f"stage_outcome_update_{FINAL_UPDATE:04d}.json",
        f"{code} final stage outcome",
    )
    arm = _read_json(arm_manifest_path, f"{code} arm manifest")
    stage = _read_json(stage_manifest_path, f"{code} final stage manifest")
    outcome = _read_json(stage_outcome_path, f"{code} final stage outcome")
    if (
        not _identity_valid(arm)
        or arm.get("kind") != "vjepa2_controlled_study_arm"
        or arm.get("git_commit") != training_commit
        or arm.get("study_identity_sha256") != study.get("identity_sha256")
        or arm.get("run_dir") != str(run_dir)
        or arm.get("arm", {}).get("code") != code
        or arm.get("arm", {}).get("name") != ARM_NAMES[code]
    ):
        raise NativeAllVideoError(f"{code} arm manifest differs")
    if (
        not _identity_valid(stage)
        or stage.get("kind") != "vjepa2_controlled_study_stage"
        or stage.get("arm_code") != code
        or stage.get("arm_identity_sha256") != arm.get("identity_sha256")
        or stage.get("stage_endpoint_completed_updates") != FINAL_UPDATE
        or stage.get("trainer_terminal_iteration") != FINAL_UPDATE - 1
        or stage.get("primary_milestone") is not True
    ):
        raise NativeAllVideoError(f"{code} final stage manifest differs")
    if (
        not _identity_valid(outcome)
        or outcome.get("kind") != "vjepa2_controlled_study_stage_outcome"
        or outcome.get("arm_code") != code
        or outcome.get("arm_identity_sha256") != arm.get("identity_sha256")
        or outcome.get("stage_identity_sha256") != stage.get("identity_sha256")
        or outcome.get("completed_updates") != FINAL_UPDATE
        or outcome.get("primary_milestone") is not True
        or outcome.get("teacher_invocations_during_training") != 0
    ):
        raise NativeAllVideoError(f"{code} final stage outcome differs")
    resolved_record = stage.get("resolved_config")
    if not isinstance(resolved_record, Mapping):
        raise NativeAllVideoError(f"{code} stage lacks resolved config record")
    resolved_config = _canonical_file(
        resolved_record.get("path", ""), f"{code} final resolved config"
    )
    if (
        resolved_config != run_dir / f"resolved_update_{FINAL_UPDATE:04d}.yaml"
        or resolved_record.get("sha256") != _sha256(resolved_config)
        or resolved_record.get("bytes") != resolved_config.stat().st_size
    ):
        raise NativeAllVideoError(f"{code} resolved config record differs")
    snapshot = _canonical_file(stage.get("snapshot", ""), f"{code} final snapshot")
    observed = outcome.get("snapshot_observed_at_stage_end")
    if (
        snapshot != run_dir / "snapshot.pt"
        or not isinstance(observed, Mapping)
        or observed.get("path") != str(snapshot)
        or observed.get("bytes") != snapshot.stat().st_size
        or not isinstance(observed.get("sha256"), str)
        or frontier.SHA256_RE.fullmatch(str(observed.get("sha256"))) is None
        or (
            verify_snapshot_sha256
            and _sha256(snapshot) != observed.get("sha256")
        )
    ):
        raise NativeAllVideoError(f"{code} final snapshot record differs")
    completion_record = outcome.get("training_completion")
    if not isinstance(completion_record, Mapping):
        raise NativeAllVideoError(f"{code} final completion record is missing")
    completion = _canonical_file(
        completion_record.get("path", ""), f"{code} final completion"
    )
    if (
        completion != run_dir / "training_complete.json"
        or completion_record.get("sha256") != _sha256(completion)
        or completion_record.get("bytes") != completion.stat().st_size
    ):
        raise NativeAllVideoError(f"{code} final completion record differs")
    return ArmInput(
        code=code,
        run_dir=run_dir,
        arm_manifest_path=arm_manifest_path,
        arm_manifest=arm,
        stage_manifest_path=stage_manifest_path,
        stage_manifest=stage,
        stage_outcome_path=stage_outcome_path,
        stage_outcome=outcome,
        resolved_config=resolved_config,
        snapshot=snapshot,
        snapshot_sha256=str(observed["sha256"]),
    )


def _validate_passed_same_nfe_gate(
    gate: Mapping[str, Any],
    *,
    label: str,
) -> None:
    checks = gate.get("checks")
    lows = gate.get("ci_lows")
    expected_checks = {
        "temporal_ci_low_strictly_positive",
        "video_nmse_ci_low_above_minus_one_percent",
        "decoded_mse_ci_low_above_minus_one_percent",
        "temporal_ci_low_at_least_three_percent",
    }
    if (
        gate.get("passed") is not True
        or not isinstance(checks, Mapping)
        or set(checks) != expected_checks
        or not all(value is True for value in checks.values())
        or not isinstance(lows, Mapping)
        or set(lows) != set(CLAIM_METRICS)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in lows.values()
        )
        or float(lows[PRIMARY_METRIC]) < 0.03
        or float(lows["video_future_nmse"]) <= -0.01
        or float(lows["decoded_mse_unit_range"]) <= -0.01
    ):
        raise NativeAllVideoError(
            f"{label} does not pass the exact same-NFE attribution gate"
        )


def _validate_passed_composite_gate(
    composite: Mapping[str, Any],
) -> None:
    components = composite.get("components")
    if (
        composite.get("passed") is not True
        or composite.get("fixed_nfe") != PRIMARY_CLAIM_NFE
        or composite.get("all_components_required") is not True
        or composite.get("gate_implementation")
        != "tools.vjepa2_nfe_frontier.same_nfe_attribution_gate"
        or not isinstance(components, Mapping)
        or set(components) != set(COMPOSITE_COMPONENTS)
    ):
        raise NativeAllVideoError("composite K=4 gate structure differs")
    for name, expected in COMPOSITE_COMPONENTS.items():
        component = components[name]
        if (
            not isinstance(component, Mapping)
            or component.get("fixed_nfe") != PRIMARY_CLAIM_NFE
            or component.get("left") != expected["left"]
            or component.get("reference") != expected["reference"]
            or component.get("passed") is not True
            or not isinstance(
                component.get("same_nfe_attribution_gate"), Mapping
            )
        ):
            raise NativeAllVideoError(
                f"composite component {name} binding differs"
            )
        _validate_passed_same_nfe_gate(
            component["same_nfe_attribution_gate"],
            label=f"composite component {name}",
        )


def _validate_validation_report_before_lockbox(
    path_value: str | Path | None,
    *,
    study: Mapping[str, Any],
    training_commit: str,
    evaluator_commit: str,
) -> tuple[Path, dict[str, Any]]:
    """Read only validation evidence; lockbox arguments are untouched here."""

    if path_value is None:
        raise NativeAllVideoError(
            "lockbox evaluation requires an eligible validation report"
        )
    path = _canonical_file(path_value, "validation report")
    report = _read_json(path, "validation report")
    if (
        not _identity_valid(report)
        or report.get("kind") != KIND_REPORT
        or report.get("split") != "validation"
        or report.get("complete") is not True
        or report.get("lockbox_eligible") is not True
        or report.get("lockbox_inspected") is not False
        or report.get("study_identity_sha256") != study.get("identity_sha256")
        or report.get("training_git_commit") != training_commit
        or report.get("evaluator_git_commit") != evaluator_commit
        or tuple(report.get("nfe_grid", ())) != NFE_GRID
        or report.get("primary_claim_nfe") != PRIMARY_CLAIM_NFE
        or report.get("endpoint_equivalence_audit", {}).get("passed") is not True
    ):
        raise NativeAllVideoError(
            "validation report is not eligible native-all-video evidence"
        )
    composite = report.get("primary_composite_gate")
    if not isinstance(composite, Mapping):
        raise NativeAllVideoError(
            "validation report lacks the composite K=4 gate"
        )
    _validate_passed_composite_gate(composite)
    return path, report


def _validate_split_records(
    split_record: Mapping[str, Any],
    *,
    split_name: str,
    expected_clips: int,
    verify_cache_arrays: bool,
) -> tuple[Path, Path, dict[str, dict[str, Any]], tuple[dict[str, Any], ...]]:
    manifest_record = split_record.get("clip_manifest")
    cache_group = split_record.get("cache")
    cache_metadata_record = (
        cache_group.get("metadata") if isinstance(cache_group, Mapping) else None
    )
    if (
        not isinstance(manifest_record, Mapping)
        or manifest_record.get("entries") != expected_clips
        or not isinstance(cache_group, Mapping)
        or not isinstance(cache_metadata_record, Mapping)
    ):
        raise NativeAllVideoError(
            f"{split_name} must pin exactly {expected_clips} clips and cache"
        )
    manifest = _canonical_file(
        manifest_record.get("path", ""), f"{split_name} clip manifest"
    )
    metadata = _canonical_file(
        cache_metadata_record.get("path", ""), f"{split_name} cache metadata"
    )
    if (
        manifest_record.get("sha256") != _sha256(manifest)
        or cache_metadata_record.get("sha256") != _sha256(metadata)
    ):
        raise NativeAllVideoError(f"{split_name} manifest/cache digest differs")
    cache = _read_json(metadata, f"{split_name} cache metadata")
    arrays: dict[str, dict[str, Any]] = {}
    for name, file_key, sha_key in (
        ("target", "target_file", "target_sha256"),
        ("rgb", "rgb_file", "rgb_sha256"),
        ("actions", "actions_file", "actions_sha256"),
    ):
        record = cache_group.get(name)
        if not isinstance(record, Mapping):
            raise NativeAllVideoError(f"{split_name} lacks {name} cache record")
        path = _canonical_file(
            record.get("path", ""), f"{split_name} {name} cache array"
        )
        metadata_path = Path(str(cache.get(file_key, "")))
        if not metadata_path.is_absolute():
            metadata_path = metadata.parent / metadata_path
        metadata_path = _canonical_file(
            metadata_path, f"metadata {split_name} {name} array"
        )
        digest = record.get("sha256")
        if (
            path != metadata_path
            or not isinstance(digest, str)
            or frontier.SHA256_RE.fullmatch(digest) is None
            or record.get("bytes") != path.stat().st_size
            or cache.get(sha_key) != digest
            or (verify_cache_arrays and _sha256(path) != digest)
        ):
            raise NativeAllVideoError(f"{split_name} {name} cache differs")
        arrays[name] = {
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "full_sha256_verified_by_rank0": verify_cache_arrays,
        }
    descriptors: list[dict[str, Any]] = []
    try:
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise NativeAllVideoError(
                        f"{split_name} manifest has blank row {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise NativeAllVideoError(
                        f"{split_name} manifest row {line_number} is not an object"
                    )
                descriptors.append(value)
    except NativeAllVideoError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeAllVideoError(
            f"{split_name} clip manifest is invalid"
        ) from exc
    indexes = [int(row.get("auxiliary_index", -1)) for row in descriptors]
    clip_ids = [str(row.get("clip_id", "")) for row in descriptors]
    if (
        len(descriptors) != expected_clips
        or indexes != list(range(expected_clips))
        or len(set(clip_ids)) != expected_clips
        or any(not value for value in clip_ids)
    ):
        raise NativeAllVideoError(
            f"{split_name} descriptors are not dense, ordered, and unique"
        )
    return manifest, metadata, arrays, tuple(descriptors)


def _registered_lockbox_split_record(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    cache = registration.get("cache")
    arrays = cache.get("arrays") if isinstance(cache, Mapping) else None
    if (
        not isinstance(cache, Mapping)
        or not isinstance(cache.get("metadata"), Mapping)
        or not isinstance(arrays, Mapping)
        or set(arrays) != {"target", "rgb", "actions"}
        or any(not isinstance(arrays[name], Mapping) for name in arrays)
    ):
        raise NativeAllVideoError(
            "lockbox registration cache/array topology differs"
        )
    # Study split records place target/rgb/actions beside metadata, while the
    # registered-lockbox schema nests them under cache.arrays.  Normalize only
    # after the eligible validation gate and registration validator succeed.
    return {
        "clip_manifest": registration.get("manifest"),
        "cache": {
            "metadata": cache["metadata"],
            **{name: arrays[name] for name in ("target", "rgb", "actions")},
        },
    }


def _resolve_split(
    *,
    split_name: str,
    study: Mapping[str, Any],
    validation_report_value: str | Path | None,
    lockbox_registration_value: str | Path | None,
    training_commit: str,
    evaluator_commit: str,
    verify_cache_arrays: bool,
) -> SplitInput:
    if split_name == "validation":
        if validation_report_value is not None or lockbox_registration_value is not None:
            raise NativeAllVideoError(
                "validation evaluation forbids lockbox/report arguments"
            )
        split_record = study.get("inputs", {}).get("splits", {}).get(
            "validation", {}
        )
        manifest, metadata, arrays, descriptors = _validate_split_records(
            split_record,
            split_name="validation",
            expected_clips=EXPECTED_SPLIT_CLIPS["validation"],
            verify_cache_arrays=verify_cache_arrays,
        )
        return SplitInput(
            name="validation",
            expected_clips=EXPECTED_SPLIT_CLIPS["validation"],
            dataset_config_key="val_dataset",
            manifest=manifest,
            cache_metadata=metadata,
            cache_arrays=arrays,
            descriptors=descriptors,
            lockbox_registration=None,
            validation_report=None,
        )
    if split_name != "lockbox":
        raise NativeAllVideoError(f"unsupported split: {split_name!r}")

    # This call is deliberately first.  Do not canonicalize, stat, open, or
    # otherwise inspect the lockbox-registration argument before it succeeds.
    _validation_path, validation_report = (
        _validate_validation_report_before_lockbox(
            validation_report_value,
            study=study,
            training_commit=training_commit,
            evaluator_commit=evaluator_commit,
        )
    )
    if lockbox_registration_value is None:
        raise NativeAllVideoError(
            "eligible lockbox evaluation requires --lockbox-registration"
        )
    registration_path = _canonical_file(
        lockbox_registration_value, "lockbox registration"
    )
    registration = _read_json(registration_path, "lockbox registration")
    try:
        validated = frontier_lockbox.validate_registration(
            registration,
            study=study,
            rehash_arrays=verify_cache_arrays,
            verify_construction=verify_cache_arrays,
        )
    except frontier_lockbox.LockboxError as exc:
        raise NativeAllVideoError(str(exc)) from exc
    if (
        not _identity_valid(registration)
        or validated.get("episode_isolation_verified") is not True
        or (
            verify_cache_arrays
            and validated.get("deterministic_construction_reverified") is not True
        )
    ):
        raise NativeAllVideoError("lockbox registration integrity differs")
    split_record = _registered_lockbox_split_record(registration)
    manifest, metadata, arrays, descriptors = _validate_split_records(
        split_record,
        split_name="lockbox",
        expected_clips=EXPECTED_SPLIT_CLIPS["lockbox"],
        verify_cache_arrays=verify_cache_arrays,
    )
    validation_ids = {
        str(value.get("clip_id", ""))
        for value in study.get("inputs", {})
        .get("splits", {})
        .get("validation", {})
        .get("descriptors", [])
        if isinstance(value, Mapping)
    }
    # validate_registration is authoritative for episode isolation.  Clip-ID
    # overlap is an additional cheap guard when descriptors are embedded.
    lockbox_ids = {str(value["clip_id"]) for value in descriptors}
    if validation_ids and validation_ids.intersection(lockbox_ids):
        raise NativeAllVideoError("validation and lockbox clip IDs overlap")
    return SplitInput(
        name="lockbox",
        expected_clips=EXPECTED_SPLIT_CLIPS["lockbox"],
        dataset_config_key="viz_dataset",
        manifest=manifest,
        cache_metadata=metadata,
        cache_arrays=arrays,
        descriptors=descriptors,
        lockbox_registration=registration,
        validation_report=validation_report,
    )


def _load_resolved_configs(
    arm_inputs: Mapping[str, ArmInput],
    *,
    split: SplitInput,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    configs: dict[str, Any] = {}
    dataset_payloads: dict[str, Any] = {}
    for code in ARMS:
        config = OmegaConf.load(arm_inputs[code].resolved_config)
        target = str(config.model.get("_target_", ""))
        if not target.endswith(".DualExplicitActionDiTModel"):
            raise NativeAllVideoError(f"{code} model is not dual ExplicitActionDiT")
        dual = config.model.get("dual_diffusion")
        if dual is None or dual.get("enabled") is not True:
            raise NativeAllVideoError(f"{code} resolved config is not dual enabled")
        common_expected = {
            "tf_channels": 64,
            "auxiliary_history_mode": "diffuse_all",
            "head_condition_on_tf_clock": True,
        }
        for key, expected in common_expected.items():
            if dual.get(key) != expected:
                raise NativeAllVideoError(
                    f"{code} model.dual_diffusion.{key} differs"
                )
        if int(dual.get("evaluation_noise_seed", -1)) != int(
            configs.get("_noise_seed", dual.get("evaluation_noise_seed", -1))
        ):
            raise NativeAllVideoError("arm evaluation noise seeds differ")
        configs.setdefault("_noise_seed", int(dual.evaluation_noise_seed))
        expected_by_arm = {
            "VPM": {
                "schedule_mode": "aligned",
                "condition_mode": "off",
                "condition_on_tf": False,
                "condition_on_tf_clock": False,
                "parameter_matched_control": True,
                "tf_loss_weight": 0.0,
            },
            "A1": {
                "schedule_mode": "tf_first_cascaded",
                "condition_mode": "off",
                "condition_on_tf": False,
                "condition_on_tf_clock": False,
                "parameter_matched_control": False,
                "tf_loss_weight": 0.333,
            },
            "J1": {
                "schedule_mode": "tf_first_cascaded",
                "condition_mode": "matched",
                "condition_on_tf": True,
                "condition_on_tf_clock": True,
                "parameter_matched_control": False,
                "tf_loss_weight": 0.333,
            },
        }
        for key, expected in expected_by_arm[code].items():
            actual = dual.get(key)
            if isinstance(expected, float):
                matches = (
                    isinstance(actual, (int, float))
                    and not isinstance(actual, bool)
                    and math.isclose(
                        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                    )
                )
            else:
                matches = actual == expected
            if not matches:
                raise NativeAllVideoError(
                    f"{code} immutable training config {key} differs: "
                    f"{actual!r} != {expected!r}"
                )
        if code in {"A1", "J1"}:
            cascade_expected = {
                "cascade_tf_loss_probability": 0.4,
                "cascade_logit_mean": 1.2,
                "cascade_logit_std": 1.0,
                "cascade_tf_condition_max_sigma": 0.25,
                "cascade_inference_tf_fraction": 0.5,
            }
            for key, expected in cascade_expected.items():
                if not math.isclose(
                    float(dual.get(key, float("nan"))),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise NativeAllVideoError(
                        f"{code} immutable cascade config {key} differs"
                    )
        transform = config.model.get("time_frequency_transform")
        if transform is not None:
            raise NativeAllVideoError(
                f"{code} resolved model contains an online auxiliary transform"
            )
        if config.get("wandb", {}).get("enabled", False):
            config.wandb.enabled = False
        dataset_config = config[split.dataset_config_key]
        dataset_payloads[code] = OmegaConf.to_container(
            dataset_config, resolve=True
        )
        configs[code] = config
    configs.pop("_noise_seed", None)
    reference_payload = dataset_payloads["VPM"]
    for code in ("A1", "J1"):
        if dataset_payloads[code] != reference_payload:
            raise NativeAllVideoError(
                f"{code} dataset config differs from VPM for {split.name}"
            )
    return configs


def _instantiate_dataset(config: Any, *, split: SplitInput) -> Any:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    dataset_config = OmegaConf.create(
        OmegaConf.to_container(
            config[split.dataset_config_key], resolve=True
        )
    )
    if split.name == "lockbox":
        # Only these two paths are changed, after the validation gate and
        # registered lockbox have already passed.
        dataset_config.datasets.ABC.clip_manifest = str(split.manifest)
        dataset_config.datasets.ABC.cache_metadata = str(split.cache_metadata)
    dataset = instantiate(dataset_config)
    if len(dataset) != split.expected_clips:
        raise NativeAllVideoError(
            f"{split.name} dataset has {len(dataset)} != "
            f"{split.expected_clips} clips"
        )
    abc = getattr(dataset, "datasets", {}).get("ABC")
    if (
        abc is None
        or Path(str(getattr(abc, "clip_manifest", ""))) != split.manifest
        or Path(str(getattr(abc, "cache_metadata", ""))) != split.cache_metadata
    ):
        raise NativeAllVideoError(
            f"instantiated {split.name} dataset differs from pinned inputs"
        )
    return dataset


def _strict_load_model(
    arm: ArmInput,
    config: Any,
    *,
    device: Any,
) -> Any:
    import torch
    from hydra.utils import instantiate

    model = instantiate(config.model)
    snapshot = torch.load(
        arm.snapshot,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if (
        not isinstance(snapshot, Mapping)
        or "model" not in snapshot
        or snapshot.get("run_identity_sha256")
        != arm.arm_manifest.get("identity_sha256")
        or snapshot.get("_start_iter") != FINAL_UPDATE
    ):
        raise NativeAllVideoError(
            f"{arm.code} snapshot identity/update cursor differs"
        )
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise NativeAllVideoError(
            f"{arm.code} strict snapshot load failed: {incompatible}"
        )
    del snapshot
    model = model.to(device=device).eval()
    try:
        quality._assert_teacher_absent(model, config)
    except quality.QualityEvaluationError as exc:
        raise NativeAllVideoError(str(exc)) from exc
    if getattr(model, "time_frequency_transform", None) is not None:
        raise NativeAllVideoError(
            f"{arm.code} instantiated an online auxiliary transform"
        )
    if int(getattr(model, "evaluation_noise_seed", -1)) < 0:
        raise NativeAllVideoError(f"{arm.code} evaluation noise seed is invalid")
    return model


def _tensor_hash(value: Any) -> str:
    return quality._tensor_sha256(value)


def _tensor_slice_hashes(value: Any) -> list[str]:
    return quality._slice_hashes(value)


def _schedule_record(
    schedule: Any,
    model_timesteps: Any,
    tf_only_steps: int,
) -> dict[str, Any]:
    import torch

    video = schedule.video.detach().cpu().float().contiguous()
    auxiliary = schedule.time_frequency.detach().cpu().float().contiguous()
    timesteps = model_timesteps.detach().cpu().float().contiguous()
    if (
        video.ndim != 1
        or auxiliary.ndim != 1
        or timesteps.ndim != 1
        or video.shape != auxiliary.shape
        or video.numel() != timesteps.numel() + 1
        or not bool(torch.isfinite(video).all())
        or not bool(torch.isfinite(auxiliary).all())
        or not bool(torch.isfinite(timesteps).all())
    ):
        raise NativeAllVideoError("sampler exposed an invalid schedule")
    return {
        "video_sigma_nodes": [float(value) for value in video.tolist()],
        "auxiliary_sigma_nodes": [
            float(value) for value in auxiliary.tolist()
        ],
        "model_timesteps": [float(value) for value in timesteps.tolist()],
        "video_sigma_sha256": _tensor_hash(video),
        "auxiliary_sigma_sha256": _tensor_hash(auxiliary),
        "model_timesteps_sha256": _tensor_hash(timesteps),
        "tf_only_steps": int(tf_only_steps),
    }


class SamplingInstrumentation(AbstractContextManager["SamplingInstrumentation"]):
    """Count actual runtime calls without changing scientific model code."""

    def __init__(self, model: Any):
        self.model = model
        self.wan_calls = 0
        self.video_scheduler_calls = 0
        self.auxiliary_euler_calls = 0
        self.auxiliary_nonzero_sigma_transitions = 0
        self.schedule_calls = 0
        self.schedule: dict[str, Any] | None = None
        self._hook = None
        self._scheduler = model.sample_scheduler
        self._original_scheduler_step: Callable[..., Any] | None = None
        self._original_sampling_schedule: Callable[..., Any] | None = None
        self._model_module: ModuleType | None = None
        self._original_euler: Callable[..., Any] | None = None

    def __enter__(self) -> "SamplingInstrumentation":
        def count_wan(_module: Any, _inputs: Any, _output: Any) -> None:
            self.wan_calls += 1

        self._hook = self.model.forward_model.register_forward_hook(count_wan)
        self._original_scheduler_step = self._scheduler.step

        def counted_video_step(*args: Any, **kwargs: Any) -> Any:
            self.video_scheduler_calls += 1
            assert self._original_scheduler_step is not None
            return self._original_scheduler_step(*args, **kwargs)

        self._scheduler.step = counted_video_step
        self._original_sampling_schedule = self.model._sampling_schedule

        def counted_schedule(*args: Any, **kwargs: Any) -> Any:
            self.schedule_calls += 1
            assert self._original_sampling_schedule is not None
            result = self._original_sampling_schedule(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 3:
                raise NativeAllVideoError("sampling schedule return contract differs")
            self.schedule = _schedule_record(*result)
            return result

        self.model._sampling_schedule = counted_schedule
        module = inspect.getmodule(self.model._sample_future)
        if module is None or not hasattr(module, "euler_flow_step"):
            raise NativeAllVideoError(
                "cannot instrument model auxiliary Euler implementation"
            )
        self._model_module = module
        self._original_euler = getattr(module, "euler_flow_step")

        def counted_euler(
            state: Any,
            velocity: Any,
            sigma: Any,
            next_sigma: Any,
        ) -> Any:
            import torch

            self.auxiliary_euler_calls += 1
            current = torch.as_tensor(sigma).detach().cpu()
            following = torch.as_tensor(next_sigma).detach().cpu()
            if not torch.equal(current, following):
                self.auxiliary_nonzero_sigma_transitions += 1
            assert self._original_euler is not None
            return self._original_euler(state, velocity, sigma, next_sigma)

        setattr(module, "euler_flow_step", counted_euler)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._hook is not None:
            self._hook.remove()
        if self._original_scheduler_step is not None:
            self._scheduler.step = self._original_scheduler_step
        if self._original_sampling_schedule is not None:
            self.model._sampling_schedule = self._original_sampling_schedule
        if self._model_module is not None and self._original_euler is not None:
            setattr(self._model_module, "euler_flow_step", self._original_euler)
        return False

    def record(self) -> dict[str, Any]:
        if self.schedule_calls != 1 or self.schedule is None:
            raise NativeAllVideoError(
                f"expected one schedule construction, got {self.schedule_calls}"
            )
        return {
            "actual_wan_calls": self.wan_calls,
            "actual_video_scheduler_calls": self.video_scheduler_calls,
            "actual_auxiliary_euler_calls": self.auxiliary_euler_calls,
            "actual_auxiliary_nonzero_sigma_transitions": (
                self.auxiliary_nonzero_sigma_transitions
            ),
            "schedule_calls": self.schedule_calls,
            "schedule": self.schedule,
        }


def _validate_native_counter(
    counter: Mapping[str, Any],
    *,
    nfe: int,
) -> None:
    import numpy as np

    schedule = counter.get("schedule")
    if not isinstance(schedule, Mapping):
        raise NativeAllVideoError("native sampler lacks schedule evidence")
    video = np.asarray(schedule.get("video_sigma_nodes"), dtype=np.float64)
    auxiliary = np.asarray(
        schedule.get("auxiliary_sigma_nodes"), dtype=np.float64
    )
    if (
        counter.get("actual_wan_calls") != nfe
        or counter.get("actual_video_scheduler_calls") != nfe
        or counter.get("actual_auxiliary_euler_calls") != nfe
        or counter.get("actual_auxiliary_nonzero_sigma_transitions") != nfe
        or schedule.get("tf_only_steps") != 0
        or video.shape != (nfe + 1,)
        or auxiliary.shape != (nfe + 1,)
        or not np.array_equal(video, auxiliary)
        or video[0] != 1.0
        or video[-1] != 0.0
        or np.any(np.diff(video) > 0.0)
    ):
        raise NativeAllVideoError(
            f"native-all-video runtime counters/schedule differ at K={nfe}"
        )


def _validate_cascade_counter(
    counter: Mapping[str, Any],
    *,
    native_nfe: int,
) -> None:
    import numpy as np

    total_nfe = 2 * native_nfe
    schedule = counter.get("schedule")
    if not isinstance(schedule, Mapping):
        raise NativeAllVideoError("cascade audit lacks schedule evidence")
    video = np.asarray(schedule.get("video_sigma_nodes"), dtype=np.float64)
    auxiliary = np.asarray(
        schedule.get("auxiliary_sigma_nodes"), dtype=np.float64
    )
    if (
        counter.get("actual_wan_calls") != total_nfe
        or counter.get("actual_video_scheduler_calls") != native_nfe
        or counter.get("actual_auxiliary_euler_calls") != total_nfe
        or counter.get("actual_auxiliary_nonzero_sigma_transitions")
        != native_nfe
        or schedule.get("tf_only_steps") != native_nfe
        or video.shape != (total_nfe + 1,)
        or auxiliary.shape != (total_nfe + 1,)
        or not np.all(video[: native_nfe + 1] == 1.0)
        or video[-1] != 0.0
        or auxiliary[0] != 1.0
        or not np.all(auxiliary[native_nfe:] == 0.0)
    ):
        raise NativeAllVideoError(
            f"cascade-off 2K audit counters/schedule differ at K={native_nfe}"
        )


def _runtime_source_contract(arm: str, runtime_source: str) -> str:
    if arm not in ARMS or runtime_source not in RUNTIME_SOURCES[arm]:
        raise NativeAllVideoError(
            f"invalid native runtime source {arm}/{runtime_source}"
        )
    return SAMPLER_SOURCE[runtime_source]


def _set_runtime_grid(
    model: Any,
    *,
    schedule_mode: str,
    sampler_source: str,
    nfe: int,
) -> None:
    if schedule_mode not in {"aligned", "tf_first_cascaded"}:
        raise NativeAllVideoError(f"invalid runtime schedule {schedule_mode}")
    if sampler_source not in {"autonomous", "off"}:
        raise NativeAllVideoError(f"invalid deployable source {sampler_source}")
    if nfe < 1:
        raise NativeAllVideoError("NFE must be positive")
    model.tf_schedule_mode = schedule_mode
    model.evaluation_condition_sources = (sampler_source,)
    model.evaluation_nfe_steps = (nfe,)
    model.viz_num_steps = nfe
    model.capture_latent_trajectories = False
    model.artifact_batch_limit = None


def _invoke_deployable(
    model: Any,
    batch: Mapping[str, Any],
    *,
    sampling_ids: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    history_rgb = batch["rgb"][:, : model.num_history_frames]
    with SamplingInstrumentation(model) as instrumentation:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        ):
            model.sample_future_deployable(
                history_rgb,
                batch["actions"],
                morphology_index=batch["morphology_index"],
                collect_artifacts=True,
                sample_ids=sampling_ids,
            )
        counter = instrumentation.record()
    artifacts = model.pop_visualization_artifacts()
    if not isinstance(artifacts, Mapping):
        raise NativeAllVideoError("deployable sampler exposed no artifact mapping")
    if (
        int(artifacts.get("deployment_mode").reshape(-1)[0]) != 1
        or int(artifacts.get("auxiliary_clean_available").reshape(-1)[0]) != 0
        or int(artifacts.get("online_teacher_call_count").reshape(-1)[0]) != 0
    ):
        raise NativeAllVideoError(
            "deployable sampler used clean auxiliary/future or online teacher"
        )
    forbidden = {
        "video_clean",
        "tf_clean",
        "ground_truth_future_uint8",
    }
    leaked = forbidden.intersection(artifacts)
    if leaked:
        raise NativeAllVideoError(
            f"deployable sampler artifact leaked held-out targets: {sorted(leaked)}"
        )
    return dict(artifacts), counter


def _native_rows(
    *,
    artifacts: Mapping[str, Any],
    counter: Mapping[str, Any],
    scoring: Mapping[str, Any],
    arm: ArmInput,
    runtime_source: str,
    nfe: int,
    clip_indexes: Sequence[int],
    clip_ids: Sequence[str],
    sampling_ids: Sequence[int],
    split: SplitInput,
    training_commit: str,
    evaluator_commit: str,
    inference_compatibility_sha256: str,
    videox_runtime_identity_sha256: str,
    world_size: int,
    batch_size_per_rank: int,
) -> list[dict[str, Any]]:
    sampler_source = _runtime_source_contract(arm.code, runtime_source)
    _validate_native_counter(counter, nfe=nfe)
    try:
        base_rows = quality._artifact_rows(
            artifacts=artifacts,
            scoring_video_clean=scoring["video_clean"],
            scoring_auxiliary_clean=scoring["auxiliary_clean"],
            scoring_ground_truth=scoring["ground_truth"],
            scoring_vae_ground_truth=scoring["vae_ground_truth"],
            scoring_raw_history_last=scoring["raw_history_last"],
            scoring_vae_history_last=scoring["vae_history_last"],
            scoring_cached_rgb_input_sha256=scoring[
                "cached_rgb_input_sha256"
            ],
            scoring_cached_actions_input_sha256=scoring[
                "cached_actions_input_sha256"
            ],
            arm=arm.code,
            completed_updates=FINAL_UPDATE,
            source=sampler_source,
            nfe=nfe,
            clip_indexes=clip_indexes,
            clip_ids=clip_ids,
            observed_wan_calls=int(counter["actual_wan_calls"]),
            evaluation_split=split.name,
            frontier_selection_identity_sha256=None,
            lockbox_registration_identity_sha256=(
                None
                if split.lockbox_registration is None
                else split.lockbox_registration["identity_sha256"]
            ),
            study_identity_sha256=arm.arm_manifest["study_identity_sha256"],
            arm_identity_sha256=arm.arm_manifest["identity_sha256"],
            stage_identity_sha256=arm.stage_manifest["identity_sha256"],
            training_git_commit=training_commit,
            evaluator_git_commit=evaluator_commit,
            inference_code_compatibility_sha256=(
                inference_compatibility_sha256
            ),
            videox_runtime_identity_sha256=videox_runtime_identity_sha256,
            evaluation_world_size=world_size,
            evaluation_batch_size_per_rank=batch_size_per_rank,
            sampling_ids=sampling_ids,
        )
    except quality.QualityEvaluationError as exc:
        raise NativeAllVideoError(str(exc)) from exc
    reference_hashes = _tensor_slice_hashes(artifacts["reference_latents"])
    if len(reference_hashes) != len(base_rows):
        raise NativeAllVideoError("reference hash count differs from batch")
    rows: list[dict[str, Any]] = []
    for index, base in enumerate(base_rows):
        tensor_hashes = {
            **base["tensor_sha256"],
            "reference_latents_sha256": reference_hashes[index],
        }
        row = _identity_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": KIND_ROW,
                "created_at_utc": _now(),
                "split": split.name,
                "arm_code": arm.code,
                "runtime_source": runtime_source,
                "sampler_condition_source": sampler_source,
                "nfe": nfe,
                "clip_index": base["clip_index"],
                "clip_id": base["clip_id"],
                "sampling_id": base["sampling_id"],
                "completed_updates": FINAL_UPDATE,
                "study_identity_sha256": arm.arm_manifest[
                    "study_identity_sha256"
                ],
                "arm_identity_sha256": arm.arm_manifest["identity_sha256"],
                "stage_identity_sha256": arm.stage_manifest["identity_sha256"],
                "stage_outcome_identity_sha256": arm.stage_outcome[
                    "identity_sha256"
                ],
                "snapshot_sha256": arm.snapshot_sha256,
                "training_git_commit": training_commit,
                "evaluator_git_commit": evaluator_commit,
                "inference_code_compatibility_sha256": (
                    inference_compatibility_sha256
                ),
                "videox_runtime_identity_sha256": (
                    videox_runtime_identity_sha256
                ),
                "evaluation_world_size": world_size,
                "evaluation_batch_size_per_rank": batch_size_per_rank,
                "lockbox_registration_identity_sha256": (
                    None
                    if split.lockbox_registration is None
                    else split.lockbox_registration["identity_sha256"]
                ),
                "runtime_intervention": {
                    "schedule_mode": "aligned_native_all_video",
                    "all_wan_calls_advance_video": True,
                    "generated_auxiliary_state": True,
                    "auxiliary_state_injected_into_video": (
                        runtime_source == "aligned"
                    ),
                    "auxiliary_clock_injected_into_video": (
                        runtime_source == "aligned"
                    ),
                    "model_and_dataset_source_modified": False,
                },
                "actual_call_counts": {
                    "wan": counter["actual_wan_calls"],
                    "video_scheduler": counter[
                        "actual_video_scheduler_calls"
                    ],
                    "auxiliary_euler": counter[
                        "actual_auxiliary_euler_calls"
                    ],
                    "auxiliary_nonzero_sigma_transitions": counter[
                        "actual_auxiliary_nonzero_sigma_transitions"
                    ],
                    "online_teacher": 0,
                },
                "schedule": counter["schedule"],
                "sampler_entrypoint": (
                    "DualExplicitActionDiTModel.sample_future_deployable"
                ),
                "clean_future_or_auxiliary_passed_to_sampler": False,
                "oracle_leakage": False,
                "deployable_evidence": True,
                "effective_state_gate": base["effective_state_gate"],
                "effective_clock_gate": base["effective_clock_gate"],
                "metrics": base["metrics"],
                "diagnostic_metrics": base["diagnostic_metrics"],
                "tensor_sha256": tensor_hashes,
                "perceptual_metric": base["perceptual_metric"],
            }
        )
        rows.append(row)
    return rows


def _native_endpoint_record(
    rows: Sequence[Mapping[str, Any]],
    *,
    counter: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        clip_index = int(row["clip_index"])
        if clip_index in result:
            raise NativeAllVideoError("duplicate native endpoint clip")
        hashes = row["tensor_sha256"]
        result[clip_index] = {
            "clip_id": row["clip_id"],
            "sampling_id": row["sampling_id"],
            "hashes": {
                "video_initial_state_sha256": hashes[
                    "video_initial_state_sha256"
                ],
                "auxiliary_initial_noise_sha256": hashes[
                    "auxiliary_initial_noise_sha256"
                ],
                "reference_latents_sha256": hashes[
                    "reference_latents_sha256"
                ],
                "video_final_sha256": hashes["video_final_sha256"],
                "decoded_final_sha256": hashes["decoded_final_sha256"],
            },
            "schedule": counter["schedule"],
        }
    return result


def _cascade_endpoint_audit_rows(
    *,
    model: Any,
    batch: Mapping[str, Any],
    sampling_ids_tensor: Any,
    sampling_ids: Sequence[int],
    arm: ArmInput,
    native_nfe: int,
    native_records: Mapping[int, Mapping[str, Any]],
    clip_indexes: Sequence[int],
    clip_ids: Sequence[str],
    split: SplitInput,
    training_commit: str,
    evaluator_commit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    if split.name != "validation":
        raise NativeAllVideoError("endpoint equivalence audit is validation-only")
    if native_nfe not in ENDPOINT_AUDIT_K:
        raise NativeAllVideoError("endpoint audit K is outside the fixed set")
    total_nfe = 2 * native_nfe
    _set_runtime_grid(
        model,
        schedule_mode="tf_first_cascaded",
        sampler_source="off",
        nfe=total_nfe,
    )
    artifacts, counter = _invoke_deployable(
        model, batch, sampling_ids=sampling_ids_tensor
    )
    _validate_cascade_counter(counter, native_nfe=native_nfe)
    infix = "_off"
    if (
        int(
            artifacts[f"wan_call_count{infix}_nfe_{total_nfe}"].reshape(-1)[0]
        )
        != total_nfe
        or tuple(
            int(value)
            for value in artifacts["evaluation_nfe_steps"].tolist()
        )
        != (total_nfe,)
        or tuple(
            int(value)
            for value in artifacts["evaluation_condition_source_codes"].tolist()
        )
        != (quality.SOURCE_CODES["off"],)
    ):
        raise NativeAllVideoError("cascade endpoint audit artifact counters differ")
    cascade_hashes = {
        "video_initial_state_sha256": _tensor_slice_hashes(
            artifacts["video_initial_state"]
        ),
        "auxiliary_initial_noise_sha256": _tensor_slice_hashes(
            artifacts["tf_initial_noise"]
        ),
        "reference_latents_sha256": _tensor_slice_hashes(
            artifacts["reference_latents"]
        ),
        "video_final_sha256": _tensor_slice_hashes(
            artifacts[f"video_final{infix}_nfe_{total_nfe}"]
        ),
        "decoded_final_sha256": _tensor_slice_hashes(
            artifacts[f"decoded_future{infix}_nfe_{total_nfe}"]
        ),
    }
    cascade_schedule = counter["schedule"]
    cascade_video = np.asarray(
        cascade_schedule["video_sigma_nodes"], dtype=np.float64
    )
    rows: list[dict[str, Any]] = []
    for index, (clip_index, clip_id, sampling_id) in enumerate(
        zip(clip_indexes, clip_ids, sampling_ids)
    ):
        native = native_records.get(int(clip_index))
        if (
            not isinstance(native, Mapping)
            or native.get("clip_id") != clip_id
            or native.get("sampling_id") != sampling_id
        ):
            raise NativeAllVideoError("native endpoint record identity differs")
        native_hashes = native["hashes"]
        observed_cascade_hashes = {
            key: values[index] for key, values in cascade_hashes.items()
        }
        hash_checks = {
            key: observed_cascade_hashes[key] == native_hashes[key]
            for key in native_hashes
        }
        native_video = np.asarray(
            native["schedule"]["video_sigma_nodes"], dtype=np.float64
        )
        video_phase_schedule_equal = np.array_equal(
            cascade_video[native_nfe:], native_video
        )
        checks = {
            **hash_checks,
            "video_phase_schedule_equal": bool(video_phase_schedule_equal),
            "native_wan_calls_equal_k": True,
            "cascade_wan_calls_equal_2k": True,
            "native_and_cascade_video_updates_equal_k": True,
            "online_teacher_calls_zero": True,
            "clean_future_or_auxiliary_not_passed": True,
        }
        passed = all(checks.values())
        if not passed:
            failed = sorted(key for key, value in checks.items() if not value)
            raise NativeAllVideoError(
                f"{arm.code} K={native_nfe} endpoint equivalence failed for "
                f"{clip_id}: {failed}"
            )
        rows.append(
            _identity_payload(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": KIND_ENDPOINT_AUDIT,
                    "created_at_utc": _now(),
                    "split": "validation",
                    "arm_code": arm.code,
                    "native_nfe_k": native_nfe,
                    "cascade_nfe_2k": total_nfe,
                    "clip_index": int(clip_index),
                    "clip_id": clip_id,
                    "sampling_id": int(sampling_id),
                    "study_identity_sha256": arm.arm_manifest[
                        "study_identity_sha256"
                    ],
                    "arm_identity_sha256": arm.arm_manifest["identity_sha256"],
                    "stage_identity_sha256": arm.stage_manifest[
                        "identity_sha256"
                    ],
                    "snapshot_sha256": arm.snapshot_sha256,
                    "training_git_commit": training_commit,
                    "evaluator_git_commit": evaluator_commit,
                    "native_runtime_source": "off",
                    "native_schedule_mode": "aligned_native_all_video",
                    "cascade_schedule_mode": "tf_first_cascaded",
                    "native_actual_call_counts": {
                        "wan": native_nfe,
                        "video_scheduler": native_nfe,
                        "auxiliary_euler": native_nfe,
                        "auxiliary_nonzero_sigma_transitions": native_nfe,
                        "online_teacher": 0,
                    },
                    "cascade_actual_call_counts": {
                        "wan": counter["actual_wan_calls"],
                        "video_scheduler": counter[
                            "actual_video_scheduler_calls"
                        ],
                        "auxiliary_euler": counter[
                            "actual_auxiliary_euler_calls"
                        ],
                        "auxiliary_nonzero_sigma_transitions": counter[
                            "actual_auxiliary_nonzero_sigma_transitions"
                        ],
                        "online_teacher": 0,
                    },
                    "native_schedule": native["schedule"],
                    "cascade_schedule": cascade_schedule,
                    "native_tensor_sha256": native_hashes,
                    "cascade_tensor_sha256": observed_cascade_hashes,
                    "checks": checks,
                    "passed": True,
                    "hash_scope": {
                        "video_latent": (
                            "exact SHA256 of sampler-exposed CPU float16 tensor"
                        ),
                        "decoded_video": (
                            "exact SHA256 of sampler-exposed uint8 tensor"
                        ),
                    },
                }
            )
        )
    del artifacts
    return rows, counter


def _expected_row_keys(expected_clips: int) -> set[tuple[str, str, int, int]]:
    return {
        (arm, runtime_source, nfe, clip_index)
        for arm in ARMS
        for runtime_source in RUNTIME_SOURCES[arm]
        for nfe in NFE_GRID
        for clip_index in range(expected_clips)
    }


def _validate_global_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: SplitInput,
    arm_inputs: Mapping[str, ArmInput],
    study_identity_sha256: str,
    training_commit: str,
    evaluator_commit: str,
    inference_compatibility_sha256: str,
    videox_runtime_identity_sha256: str,
    world_size: int,
    batch_size_per_rank: int,
) -> dict[tuple[str, str, int, int], Mapping[str, Any]]:
    observed: dict[tuple[str, str, int, int], Mapping[str, Any]] = {}
    descriptor_ids = {
        index: str(value["clip_id"])
        for index, value in enumerate(split.descriptors)
    }
    for row in rows:
        if not _identity_valid(row) or row.get("kind") != KIND_ROW:
            raise NativeAllVideoError("native-all-video row identity differs")
        arm = row.get("arm_code")
        runtime_source = row.get("runtime_source")
        nfe = row.get("nfe")
        clip_index = row.get("clip_index")
        if (
            arm not in ARMS
            or runtime_source not in RUNTIME_SOURCES[arm]
            or nfe not in NFE_GRID
            or isinstance(clip_index, bool)
            or not isinstance(clip_index, int)
            or clip_index not in descriptor_ids
        ):
            raise NativeAllVideoError("native-all-video row grid identity differs")
        key = (str(arm), str(runtime_source), int(nfe), int(clip_index))
        if key in observed:
            raise NativeAllVideoError(f"duplicate native-all-video row: {key}")
        expected_sampler_source = SAMPLER_SOURCE[str(runtime_source)]
        expected_arm = arm_inputs[str(arm)]
        counts = row.get("actual_call_counts")
        if (
            row.get("split") != split.name
            or row.get("clip_id") != descriptor_ids[int(clip_index)]
            or row.get("sampling_id")
            != int(clip_index) + SAMPLE_ID_OFFSETS[split.name]
            or row.get("sampler_condition_source") != expected_sampler_source
            or row.get("completed_updates") != FINAL_UPDATE
            or row.get("study_identity_sha256") != study_identity_sha256
            or row.get("arm_identity_sha256")
            != expected_arm.arm_manifest["identity_sha256"]
            or row.get("stage_identity_sha256")
            != expected_arm.stage_manifest["identity_sha256"]
            or row.get("stage_outcome_identity_sha256")
            != expected_arm.stage_outcome["identity_sha256"]
            or row.get("snapshot_sha256") != expected_arm.snapshot_sha256
            or row.get("training_git_commit") != training_commit
            or row.get("evaluator_git_commit") != evaluator_commit
            or row.get("inference_code_compatibility_sha256")
            != inference_compatibility_sha256
            or row.get("videox_runtime_identity_sha256")
            != videox_runtime_identity_sha256
            or row.get("evaluation_world_size") != world_size
            or row.get("evaluation_batch_size_per_rank")
            != batch_size_per_rank
            or row.get("lockbox_registration_identity_sha256")
            != (
                None
                if split.lockbox_registration is None
                else split.lockbox_registration["identity_sha256"]
            )
            or row.get("clean_future_or_auxiliary_passed_to_sampler") is not False
            or row.get("oracle_leakage") is not False
            or row.get("deployable_evidence") is not True
            or not isinstance(counts, Mapping)
            or counts.get("wan") != nfe
            or counts.get("video_scheduler") != nfe
            or counts.get("auxiliary_euler") != nfe
            or counts.get("auxiliary_nonzero_sigma_transitions") != nfe
            or counts.get("online_teacher") != 0
        ):
            raise NativeAllVideoError(
                f"native-all-video row provenance/counters differ: {key}"
            )
        _validate_native_counter(
            {
                "actual_wan_calls": counts["wan"],
                "actual_video_scheduler_calls": counts["video_scheduler"],
                "actual_auxiliary_euler_calls": counts["auxiliary_euler"],
                "actual_auxiliary_nonzero_sigma_transitions": counts[
                    "auxiliary_nonzero_sigma_transitions"
                ],
                "schedule": row.get("schedule"),
            },
            nfe=int(nfe),
        )
        intervention = row.get("runtime_intervention")
        if (
            not isinstance(intervention, Mapping)
            or intervention.get("schedule_mode")
            != "aligned_native_all_video"
            or intervention.get("all_wan_calls_advance_video") is not True
            or intervention.get("generated_auxiliary_state") is not True
            or intervention.get("auxiliary_state_injected_into_video")
            is not (runtime_source == "aligned")
            or intervention.get("auxiliary_clock_injected_into_video")
            is not (runtime_source == "aligned")
            or intervention.get("model_and_dataset_source_modified") is not False
        ):
            raise NativeAllVideoError(
                f"runtime intervention differs: {key}"
            )
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise NativeAllVideoError(f"row lacks metrics: {key}")
        for metric in (
            *CLAIM_METRICS,
            "decoded_psnr_db",
            "auxiliary_future_nmse",
            "auxiliary_future_cosine_similarity",
        ):
            value = metrics.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise NativeAllVideoError(
                    f"row metric {metric} is invalid: {key}"
                )
        hashes = row.get("tensor_sha256")
        required_hashes = {
            "video_clean_sha256",
            "auxiliary_clean_sha256",
            "ground_truth_sha256",
            "vae_ground_truth_sha256",
            "raw_history_last_sha256",
            "vae_history_last_sha256",
            "cached_rgb_input_sha256",
            "cached_actions_input_sha256",
            "video_initial_state_sha256",
            "auxiliary_initial_state_sha256",
            "auxiliary_initial_noise_sha256",
            "video_final_sha256",
            "auxiliary_final_sha256",
            "decoded_final_sha256",
            "reference_latents_sha256",
        }
        if (
            not isinstance(hashes, Mapping)
            or set(hashes) != required_hashes
            or any(
                not isinstance(value, str)
                or frontier.SHA256_RE.fullmatch(value) is None
                for value in hashes.values()
            )
        ):
            raise NativeAllVideoError(f"row tensor hashes differ: {key}")
        if (
            hashes["auxiliary_initial_state_sha256"]
            != hashes["auxiliary_initial_noise_sha256"]
        ):
            raise NativeAllVideoError(
                f"zero-history auxiliary initial/noise differs: {key}"
            )
        if arm in {"VPM", "A1"} and (
            row.get("effective_state_gate") != 0.0
            or row.get("effective_clock_gate") != 0.0
        ):
            raise NativeAllVideoError(f"{arm} gate is nonzero: {key}")
        observed[key] = row
    expected = _expected_row_keys(split.expected_clips)
    if set(observed) != expected:
        missing = sorted(expected - set(observed))[:8]
        extra = sorted(set(observed) - expected)[:8]
        raise NativeAllVideoError(
            f"native row inventory differs: missing={missing}, extra={extra}"
        )

    immutable_hash_fields = (
        "video_clean_sha256",
        "auxiliary_clean_sha256",
        "ground_truth_sha256",
        "vae_ground_truth_sha256",
        "raw_history_last_sha256",
        "vae_history_last_sha256",
        "cached_rgb_input_sha256",
        "cached_actions_input_sha256",
        "video_initial_state_sha256",
        "auxiliary_initial_state_sha256",
        "auxiliary_initial_noise_sha256",
        "reference_latents_sha256",
    )
    for clip_index in range(split.expected_clips):
        clip_rows = [
            row
            for key, row in observed.items()
            if key[3] == clip_index
        ]
        reference_hashes = clip_rows[0]["tensor_sha256"]
        for row in clip_rows[1:]:
            for field in immutable_hash_fields:
                if row["tensor_sha256"][field] != reference_hashes[field]:
                    raise NativeAllVideoError(
                        f"paired immutable input hash {field} differs for "
                        f"clip {clip_index}"
                    )
    return observed


def _validate_endpoint_audits(
    audits: Sequence[Mapping[str, Any]],
    *,
    split: SplitInput,
    row_lookup: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
    arm_inputs: Mapping[str, ArmInput],
    study_identity_sha256: str,
    training_commit: str,
    evaluator_commit: str,
) -> dict[str, Any]:
    if split.name == "lockbox":
        if audits:
            raise NativeAllVideoError("lockbox must not rerun endpoint audit")
        validation_report = split.validation_report
        if (
            not isinstance(validation_report, Mapping)
            or validation_report.get("endpoint_equivalence_audit", {}).get(
                "passed"
            )
            is not True
        ):
            raise NativeAllVideoError(
                "lockbox lacks passed validation endpoint audit"
            )
        return {
            "passed": True,
            "run_on_split": "validation",
            "reused_validation_report_identity_sha256": validation_report[
                "identity_sha256"
            ],
            "rerun_on_lockbox": False,
        }
    observed: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for audit in audits:
        if (
            not _identity_valid(audit)
            or audit.get("kind") != KIND_ENDPOINT_AUDIT
            or audit.get("split") != "validation"
            or audit.get("passed") is not True
        ):
            raise NativeAllVideoError("endpoint audit identity/status differs")
        arm = audit.get("arm_code")
        native_nfe = audit.get("native_nfe_k")
        clip_index = audit.get("clip_index")
        key = (str(arm), int(native_nfe), int(clip_index))
        if (
            arm not in ARMS
            or native_nfe not in ENDPOINT_AUDIT_K
            or isinstance(clip_index, bool)
            or not isinstance(clip_index, int)
            or not 0 <= clip_index < split.expected_clips
            or key in observed
        ):
            raise NativeAllVideoError("endpoint audit grid identity differs")
        expected_arm = arm_inputs[str(arm)]
        native_row = row_lookup[(str(arm), "off", int(native_nfe), clip_index)]
        native_hashes = audit.get("native_tensor_sha256")
        cascade_hashes = audit.get("cascade_tensor_sha256")
        native_counts = audit.get("native_actual_call_counts")
        cascade_counts = audit.get("cascade_actual_call_counts")
        expected_native_counts = {
            "wan": native_nfe,
            "video_scheduler": native_nfe,
            "auxiliary_euler": native_nfe,
            "auxiliary_nonzero_sigma_transitions": native_nfe,
            "online_teacher": 0,
        }
        expected_check_keys = {
            "video_initial_state_sha256",
            "auxiliary_initial_noise_sha256",
            "reference_latents_sha256",
            "video_final_sha256",
            "decoded_final_sha256",
            "video_phase_schedule_equal",
            "native_wan_calls_equal_k",
            "cascade_wan_calls_equal_2k",
            "native_and_cascade_video_updates_equal_k",
            "online_teacher_calls_zero",
            "clean_future_or_auxiliary_not_passed",
        }
        if (
            audit.get("cascade_nfe_2k") != 2 * native_nfe
            or audit.get("clip_id") != native_row["clip_id"]
            or audit.get("sampling_id") != native_row["sampling_id"]
            or audit.get("study_identity_sha256") != study_identity_sha256
            or audit.get("arm_identity_sha256")
            != expected_arm.arm_manifest["identity_sha256"]
            or audit.get("stage_identity_sha256")
            != expected_arm.stage_manifest["identity_sha256"]
            or audit.get("snapshot_sha256") != expected_arm.snapshot_sha256
            or audit.get("training_git_commit") != training_commit
            or audit.get("evaluator_git_commit") != evaluator_commit
            or audit.get("native_runtime_source") != "off"
            or audit.get("native_schedule_mode") != "aligned_native_all_video"
            or audit.get("cascade_schedule_mode") != "tf_first_cascaded"
            or native_counts != expected_native_counts
            or audit.get("native_schedule") != native_row["schedule"]
            or not isinstance(native_hashes, Mapping)
            or not isinstance(cascade_hashes, Mapping)
            or set(native_hashes) != {
                "video_initial_state_sha256",
                "auxiliary_initial_noise_sha256",
                "reference_latents_sha256",
                "video_final_sha256",
                "decoded_final_sha256",
            }
            or set(cascade_hashes) != set(native_hashes)
            or any(
                cascade_hashes[field] != native_hashes[field]
                for field in native_hashes
            )
            or native_hashes.get("video_initial_state_sha256")
            != native_row["tensor_sha256"]["video_initial_state_sha256"]
            or native_hashes.get("auxiliary_initial_noise_sha256")
            != native_row["tensor_sha256"][
                "auxiliary_initial_noise_sha256"
            ]
            or native_hashes.get("reference_latents_sha256")
            != native_row["tensor_sha256"]["reference_latents_sha256"]
            or native_hashes.get("video_final_sha256")
            != native_row["tensor_sha256"]["video_final_sha256"]
            or native_hashes.get("decoded_final_sha256")
            != native_row["tensor_sha256"]["decoded_final_sha256"]
            or not isinstance(audit.get("checks"), Mapping)
            or set(audit["checks"]) != expected_check_keys
            or not all(audit["checks"].values())
        ):
            raise NativeAllVideoError(f"endpoint audit binding differs: {key}")
        if (
            not isinstance(cascade_counts, Mapping)
            or set(cascade_counts) != set(expected_native_counts)
            or cascade_counts.get("online_teacher") != 0
        ):
            raise NativeAllVideoError(
                f"endpoint audit lacks cascade counts: {key}"
            )
        _validate_cascade_counter(
            {
                "actual_wan_calls": cascade_counts.get("wan"),
                "actual_video_scheduler_calls": cascade_counts.get(
                    "video_scheduler"
                ),
                "actual_auxiliary_euler_calls": cascade_counts.get(
                    "auxiliary_euler"
                ),
                "actual_auxiliary_nonzero_sigma_transitions": (
                    cascade_counts.get(
                        "auxiliary_nonzero_sigma_transitions"
                    )
                ),
                "schedule": audit.get("cascade_schedule"),
            },
            native_nfe=int(native_nfe),
        )
        observed[key] = audit
    expected = {
        (arm, native_nfe, clip_index)
        for arm in ARMS
        for native_nfe in ENDPOINT_AUDIT_K
        for clip_index in range(split.expected_clips)
    }
    if set(observed) != expected:
        raise NativeAllVideoError("endpoint audit inventory is incomplete")
    return {
        "passed": True,
        "run_on_split": "validation",
        "native_k": list(ENDPOINT_AUDIT_K),
        "cascade_calls": [2 * value for value in ENDPOINT_AUDIT_K],
        "arms": list(ARMS),
        "clip_count": split.expected_clips,
        "record_count": len(observed),
        "all_video_latent_and_decoded_endpoint_hashes_exact": True,
        "all_schedule_and_call_count_checks_passed": True,
        "hash_scope": {
            "video_latent": "sampler-exposed CPU float16 tensor",
            "decoded_video": "sampler-exposed uint8 tensor",
        },
    }


def _paired_comparison(
    row_lookup: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
    *,
    split: str,
    expected_clips: int,
    left_arm: str,
    left_source: str,
    reference_arm: str,
    reference_source: str,
    nfe: int,
    label_prefix: str,
) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    for metric in CLAIM_METRICS:
        left: list[float] = []
        reference: list[float] = []
        for clip_index in range(expected_clips):
            left_row = row_lookup[
                (left_arm, left_source, nfe, clip_index)
            ]
            reference_row = row_lookup[
                (reference_arm, reference_source, nfe, clip_index)
            ]
            if (
                left_row["clip_id"] != reference_row["clip_id"]
                or left_row["sampling_id"] != reference_row["sampling_id"]
            ):
                raise NativeAllVideoError(
                    "paired comparison clip/noise identity differs"
                )
            left.append(float(left_row["metrics"][metric]))
            reference.append(float(reference_row["metrics"][metric]))
        try:
            effects[metric] = frontier.paired_effect(
                left,
                reference,
                bootstrap_samples=BOOTSTRAP_SAMPLES,
                confidence=0.95,
                seed=BOOTSTRAP_SEED,
                label=(
                    f"{label_prefix}:{split}:nfe={nfe}:metric={metric}:"
                    f"{left_arm}/{left_source}-vs-"
                    f"{reference_arm}/{reference_source}"
                ),
            )
        except frontier.FrontierError as exc:
            raise NativeAllVideoError(str(exc)) from exc
    comparison = {
        "split": split,
        "nfe": nfe,
        "left": {"arm": left_arm, "runtime_source": left_source},
        "reference": {
            "arm": reference_arm,
            "runtime_source": reference_source,
        },
        "metrics": effects,
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "confidence": 0.95,
            "seed": BOOTSTRAP_SEED,
            "unit": "paired immutable clip_id/clip_index",
            "positive_relative_improvement_favors_left": True,
        },
    }
    try:
        return {
            **comparison,
            "quality_gate": frontier.quality_gate(comparison),
            "same_nfe_attribution_gate": (
                frontier.same_nfe_attribution_gate(comparison)
            ),
        }
    except frontier.FrontierError as exc:
        raise NativeAllVideoError(str(exc)) from exc


def _comparison_grid(
    row_lookup: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
    *,
    split: SplitInput,
    left_arm: str,
    left_source: str,
    reference_arm: str,
    reference_source: str,
    label_prefix: str,
) -> dict[str, Any]:
    by_nfe = {
        str(nfe): _paired_comparison(
            row_lookup,
            split=split.name,
            expected_clips=split.expected_clips,
            left_arm=left_arm,
            left_source=left_source,
            reference_arm=reference_arm,
            reference_source=reference_source,
            nfe=nfe,
            label_prefix=label_prefix,
        )
        for nfe in NFE_GRID
    }
    return {
        "left": {"arm": left_arm, "runtime_source": left_source},
        "reference": {
            "arm": reference_arm,
            "runtime_source": reference_source,
        },
        "nfe_grid": list(NFE_GRID),
        "comparisons_by_nfe": by_nfe,
        "claim_nfe": PRIMARY_CLAIM_NFE,
        "metrics": by_nfe[str(PRIMARY_CLAIM_NFE)]["metrics"],
        "quality_gate": by_nfe[str(PRIMARY_CLAIM_NFE)]["quality_gate"],
        "same_nfe_attribution_gate": by_nfe[
            str(PRIMARY_CLAIM_NFE)
        ]["same_nfe_attribution_gate"],
    }


def _build_composite_gate(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(comparisons) != set(COMPOSITE_COMPONENTS):
        raise NativeAllVideoError("composite comparison inventory differs")
    components: dict[str, Any] = {}
    for name, expected in COMPOSITE_COMPONENTS.items():
        comparison = comparisons[name]
        gate = comparison.get("same_nfe_attribution_gate")
        if (
            comparison.get("left") != expected["left"]
            or comparison.get("reference") != expected["reference"]
            or comparison.get("claim_nfe") != PRIMARY_CLAIM_NFE
            or not isinstance(gate, Mapping)
        ):
            raise NativeAllVideoError(
                f"composite comparison {name} differs"
            )
        components[name] = {
            "fixed_nfe": PRIMARY_CLAIM_NFE,
            "left": expected["left"],
            "reference": expected["reference"],
            "metrics": comparison["metrics"],
            "same_nfe_attribution_gate": gate,
            "passed": gate.get("passed") is True,
        }
    composite = {
        "fixed_nfe": PRIMARY_CLAIM_NFE,
        "all_components_required": True,
        "gate_implementation": (
            "tools.vjepa2_nfe_frontier.same_nfe_attribution_gate"
        ),
        "components": components,
        "passed": all(value["passed"] for value in components.values()),
        "rule": (
            "At fixed K=4, A1/off vs VPM/off, J1/aligned vs J1/off, "
            "and J1/aligned vs VPM/off must each have temporal "
            "relative-improvement CI-low >= 3% and video-NMSE/decoded-MSE "
            "CI-lows > -1%."
        ),
    }
    if composite["passed"]:
        _validate_passed_composite_gate(composite)
    return composite


def _arm_provenance(arm: ArmInput) -> dict[str, Any]:
    return {
        "arm_identity_sha256": arm.arm_manifest["identity_sha256"],
        "stage_identity_sha256": arm.stage_manifest["identity_sha256"],
        "stage_outcome_identity_sha256": arm.stage_outcome["identity_sha256"],
        "completed_updates": FINAL_UPDATE,
        "resolved_config": {
            "path": str(arm.resolved_config),
            "sha256": _sha256(arm.resolved_config),
            "bytes": arm.resolved_config.stat().st_size,
        },
        "snapshot": {
            "path": str(arm.snapshot),
            "sha256": arm.snapshot_sha256,
            "bytes": arm.snapshot.stat().st_size,
            "strict_state_dict_load": True,
        },
    }


def _build_report(
    *,
    split: SplitInput,
    rows: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    arm_inputs: Mapping[str, ArmInput],
    study_path: Path,
    study: Mapping[str, Any],
    training_commit: str,
    evaluator_commit: str,
    inference_compatibility: Mapping[str, Any],
    videox_runtime: Mapping[str, Any],
    world_size: int,
    batch_size_per_rank: int,
    rank_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    compatibility_sha = hashlib.sha256(
        _canonical_json(inference_compatibility)
    ).hexdigest()
    videox_sha = hashlib.sha256(_canonical_json(videox_runtime)).hexdigest()
    row_lookup = _validate_global_rows(
        rows,
        split=split,
        arm_inputs=arm_inputs,
        study_identity_sha256=str(study["identity_sha256"]),
        training_commit=training_commit,
        evaluator_commit=evaluator_commit,
        inference_compatibility_sha256=compatibility_sha,
        videox_runtime_identity_sha256=videox_sha,
        world_size=world_size,
        batch_size_per_rank=batch_size_per_rank,
    )
    endpoint_summary = _validate_endpoint_audits(
        audits,
        split=split,
        row_lookup=row_lookup,
        arm_inputs=arm_inputs,
        study_identity_sha256=str(study["identity_sha256"]),
        training_commit=training_commit,
        evaluator_commit=evaluator_commit,
    )
    component_comparisons = {
        "training_objective": _comparison_grid(
            row_lookup,
            split=split,
            left_arm="A1",
            left_source="off",
            reference_arm="VPM",
            reference_source="off",
            label_prefix="component-training-objective",
        ),
        "generated_feature_use": _comparison_grid(
            row_lookup,
            split=split,
            left_arm="J1",
            left_source="aligned",
            reference_arm="J1",
            reference_source="off",
            label_prefix="component-generated-feature-use",
        ),
        "end_to_end": _comparison_grid(
            row_lookup,
            split=split,
            left_arm="J1",
            left_source="aligned",
            reference_arm="VPM",
            reference_source="off",
            label_prefix="component-end-to-end",
        ),
    }
    composite_gate = _build_composite_gate(component_comparisons)
    secondary = {
        "j1_aligned_vs_a1_off": _comparison_grid(
            row_lookup,
            split=split,
            left_arm="J1",
            left_source="aligned",
            reference_arm="A1",
            reference_source="off",
            label_prefix="secondary-j1-aligned-vs-a1-off",
        ),
        "j1_off_vs_a1_off": _comparison_grid(
            row_lookup,
            split=split,
            left_arm="J1",
            left_source="off",
            reference_arm="A1",
            reference_source="off",
            label_prefix="secondary-j1-off-vs-a1-off",
        ),
    }
    primary_gate_passed = composite_gate["passed"] is True
    endpoint_passed = endpoint_summary["passed"] is True
    lockbox_eligible = (
        split.name == "validation"
        and primary_gate_passed
        and endpoint_passed
    )
    confirmatory_lockbox_passed = (
        split.name == "lockbox" and primary_gate_passed and endpoint_passed
    )
    return _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_REPORT,
            "created_at_utc": _now(),
            "complete": True,
            "split": split.name,
            "study_identity_sha256": study["identity_sha256"],
            "training_git_commit": training_commit,
            "evaluator_git_commit": evaluator_commit,
            "inference_code_compatibility": dict(inference_compatibility),
            "inference_code_compatibility_sha256": compatibility_sha,
            "videox_runtime": dict(videox_runtime),
            "videox_runtime_identity_sha256": videox_sha,
            "world_size": world_size,
            "batch_size_per_rank": batch_size_per_rank,
            "clip_count": split.expected_clips,
            "nfe_grid": list(NFE_GRID),
            "runtime_sources": {
                arm: list(RUNTIME_SOURCES[arm]) for arm in ARMS
            },
            "primary_claim_nfe": PRIMARY_CLAIM_NFE,
            "primary_composite_gate": composite_gate,
            "component_comparisons": component_comparisons,
            "secondary_comparisons": secondary,
            "endpoint_equivalence_audit": endpoint_summary,
            "lockbox_eligible": lockbox_eligible,
            "confirmatory_lockbox_primary_gate_passed": (
                confirmatory_lockbox_passed
            ),
            "lockbox_inspected": split.name == "lockbox",
            "validation_gate_policy": {
                "fixed_nfe": PRIMARY_CLAIM_NFE,
                "all_three_components_required": True,
                "components": dict(COMPOSITE_COMPONENTS),
                "gate_implementation": (
                    "tools.vjepa2_nfe_frontier."
                    "same_nfe_attribution_gate"
                ),
                "gate": (
                    "each component requires temporal paired relative-"
                    "improvement CI-low >= 0.03; video-NMSE and decoded-MSE "
                    "CI-lows > -0.01"
                ),
                "endpoint_audit_must_pass": True,
                "nfe_reselection": False,
            },
            "validation_report_binding": (
                None
                if split.validation_report is None
                else {
                    "identity_sha256": split.validation_report[
                        "identity_sha256"
                    ],
                    "primary_gate_passed": True,
                    "endpoint_audit_passed": True,
                }
            ),
            "lockbox_registration": (
                None
                if split.lockbox_registration is None
                else {
                    "identity_sha256": split.lockbox_registration[
                        "identity_sha256"
                    ],
                    "episode_isolation": split.lockbox_registration[
                        "episode_isolation"
                    ],
                }
            ),
            "input_provenance": {
                "study_manifest": {
                    "path": str(study_path),
                    "sha256": _sha256(study_path),
                },
                "arms": {
                    code: _arm_provenance(arm_inputs[code]) for code in ARMS
                },
                "evaluation_manifest": {
                    "path": str(split.manifest),
                    "sha256": _sha256(split.manifest),
                },
                "evaluation_cache_metadata": {
                    "path": str(split.cache_metadata),
                    "sha256": _sha256(split.cache_metadata),
                },
                "evaluation_cache_arrays": {
                    name: {
                        "path": value["path"],
                        "sha256": value["sha256"],
                        "bytes": value["bytes"],
                        "full_sha256_verified_by_rank0": True,
                    }
                    for name, value in split.cache_arrays.items()
                },
            },
            "evidence": {
                "rank_files": dict(rank_evidence),
                "quality_row_count": len(rows),
                "endpoint_audit_row_count": len(audits),
                "expected_quality_row_count": (
                    split.expected_clips
                    * sum(
                        len(RUNTIME_SOURCES[arm]) * len(NFE_GRID)
                        for arm in ARMS
                    )
                ),
                "expected_endpoint_audit_row_count": (
                    0
                    if split.name == "lockbox"
                    else split.expected_clips
                    * len(ARMS)
                    * len(ENDPOINT_AUDIT_K)
                ),
            },
            "integrity": {
                "all_models_strict_loaded": True,
                "model_or_dataset_code_modified": False,
                "all_rows_identity_verified": True,
                "all_input_and_noise_hashes_paired": True,
                "actual_wan_video_auxiliary_calls_verified": True,
                "online_teacher_calls": 0,
                "clean_future_or_auxiliary_passed_to_sampler": False,
                "oracle_or_future_leakage": False,
                "all_cache_arrays_fully_rehashed_by_rank0": True,
                "sigma_convention": "sigma=1 noise, sigma=0 clean",
            },
            "metric_scope": {
                "primary_metrics": list(CLAIM_METRICS),
                "canonical_paired_bootstrap": {
                    "implementation": "tools.vjepa2_nfe_frontier.paired_effect",
                    "samples": BOOTSTRAP_SAMPLES,
                    "confidence": 0.95,
                    "seed": BOOTSTRAP_SEED,
                },
                "reconstruction_metrics_only": True,
                "perceptual_metric_available": False,
                "claim_restriction": (
                    "post-study warm-start ABC reconstruction evidence only; "
                    "not a general V-JEPA or perceptual-quality claim"
                ),
                "semantic_attribution_restriction": (
                    "J1 aligned versus off tests whether generated auxiliary "
                    "feedback helps; without an aligned-versus-shuffled "
                    "contrast it does not prove use of sample-specific "
                    "semantic content"
                ),
            },
        }
    )


def _rank_indexes(rank: int, world_size: int, expected_clips: int) -> list[int]:
    return list(range(rank, expected_clips, world_size))


def _add_counts(total: dict[str, int], counter: Mapping[str, Any]) -> None:
    total["wan"] += int(counter["actual_wan_calls"])
    total["video_scheduler"] += int(counter["actual_video_scheduler_calls"])
    total["auxiliary_euler"] += int(counter["actual_auxiliary_euler_calls"])
    total["auxiliary_nonzero_sigma_transitions"] += int(
        counter["actual_auxiliary_nonzero_sigma_transitions"]
    )


def _expected_rank_call_counts(
    *, split_name: str, batches: int
) -> dict[str, int]:
    native_calls = sum(
        len(RUNTIME_SOURCES[arm]) * sum(NFE_GRID) for arm in ARMS
    )
    audit_wan = (
        0
        if split_name == "lockbox"
        else len(ARMS) * sum(2 * value for value in ENDPOINT_AUDIT_K)
    )
    audit_updates = (
        0
        if split_name == "lockbox"
        else len(ARMS) * sum(ENDPOINT_AUDIT_K)
    )
    return {
        "wan": batches * (native_calls + audit_wan),
        "video_scheduler": batches * (native_calls + audit_updates),
        "auxiliary_euler": batches * (native_calls + audit_wan),
        "auxiliary_nonzero_sigma_transitions": batches
        * (native_calls + audit_updates),
        "online_teacher": 0,
    }


def _evaluate_rank(
    *,
    arm_inputs: Mapping[str, ArmInput],
    configs: Mapping[str, Any],
    dataset: Any,
    split: SplitInput,
    device: Any,
    assigned_indexes: Sequence[int],
    batch_size_per_rank: int,
    training_commit: str,
    evaluator_commit: str,
    inference_compatibility_sha256: str,
    videox_runtime_identity_sha256: str,
    world_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    import torch

    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    total_counts = {
        "wan": 0,
        "video_scheduler": 0,
        "auxiliary_euler": 0,
        "auxiliary_nonzero_sigma_transitions": 0,
        "online_teacher": 0,
    }
    for code in ARMS:
        arm = arm_inputs[code]
        model = _strict_load_model(arm, configs[code], device=device)
        try:
            for start in range(0, len(assigned_indexes), batch_size_per_rank):
                clip_indexes = list(
                    assigned_indexes[start : start + batch_size_per_rank]
                )
                samples = [dataset[index] for index in clip_indexes]
                observed_indexes = [
                    int(sample["clip_index"].item()) for sample in samples
                ]
                if observed_indexes != clip_indexes:
                    raise NativeAllVideoError(
                        f"{split.name} dataset substituted clips: "
                        f"{observed_indexes} != {clip_indexes}"
                    )
                try:
                    batch = quality._move_batch(samples, device)
                    scoring = quality._prepare_scoring_targets(model, batch)
                except quality.QualityEvaluationError as exc:
                    raise NativeAllVideoError(str(exc)) from exc
                clip_ids = [
                    str(split.descriptors[index]["clip_id"])
                    for index in clip_indexes
                ]
                sampling_ids_tensor = (
                    batch["clip_index"] + SAMPLE_ID_OFFSETS[split.name]
                )
                sampling_ids = [
                    int(value)
                    for value in sampling_ids_tensor.detach().cpu().tolist()
                ]
                native_endpoint_by_k: dict[
                    int, dict[int, dict[str, Any]]
                ] = {}
                for runtime_source in RUNTIME_SOURCES[code]:
                    sampler_source = _runtime_source_contract(
                        code, runtime_source
                    )
                    for nfe in NFE_GRID:
                        _set_runtime_grid(
                            model,
                            schedule_mode="aligned",
                            sampler_source=sampler_source,
                            nfe=nfe,
                        )
                        artifacts, counter = _invoke_deployable(
                            model,
                            batch,
                            sampling_ids=sampling_ids_tensor,
                        )
                        native = _native_rows(
                            artifacts=artifacts,
                            counter=counter,
                            scoring=scoring,
                            arm=arm,
                            runtime_source=runtime_source,
                            nfe=nfe,
                            clip_indexes=clip_indexes,
                            clip_ids=clip_ids,
                            sampling_ids=sampling_ids,
                            split=split,
                            training_commit=training_commit,
                            evaluator_commit=evaluator_commit,
                            inference_compatibility_sha256=(
                                inference_compatibility_sha256
                            ),
                            videox_runtime_identity_sha256=(
                                videox_runtime_identity_sha256
                            ),
                            world_size=world_size,
                            batch_size_per_rank=batch_size_per_rank,
                        )
                        rows.extend(native)
                        _add_counts(total_counts, counter)
                        if (
                            split.name == "validation"
                            and runtime_source == "off"
                            and nfe in ENDPOINT_AUDIT_K
                        ):
                            native_endpoint_by_k[nfe] = (
                                _native_endpoint_record(
                                    native, counter=counter
                                )
                            )
                        del artifacts
                if split.name == "validation":
                    if set(native_endpoint_by_k) != set(ENDPOINT_AUDIT_K):
                        raise NativeAllVideoError(
                            f"{code} native endpoint inventory is incomplete"
                        )
                    for native_nfe in ENDPOINT_AUDIT_K:
                        audit_rows, counter = _cascade_endpoint_audit_rows(
                            model=model,
                            batch=batch,
                            sampling_ids_tensor=sampling_ids_tensor,
                            sampling_ids=sampling_ids,
                            arm=arm,
                            native_nfe=native_nfe,
                            native_records=native_endpoint_by_k[native_nfe],
                            clip_indexes=clip_indexes,
                            clip_ids=clip_ids,
                            split=split,
                            training_commit=training_commit,
                            evaluator_commit=evaluator_commit,
                        )
                        audits.extend(audit_rows)
                        _add_counts(total_counts, counter)
                del scoring, batch, samples
        finally:
            del model
            torch.cuda.empty_cache()
    return rows, audits, total_counts


def _fresh_output_directory(
    value: str | Path,
    *,
    repo: Path,
    study_root: Path,
) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise NativeAllVideoError("output directory must be an absolute path")
    parent = _canonical_directory(raw.parent, "output parent directory")
    output = parent / raw.name
    if output.exists() or output.is_symlink():
        raise NativeAllVideoError(
            f"fresh output directory already exists: {output}"
        )
    if output.is_relative_to(repo) or output.is_relative_to(study_root):
        raise NativeAllVideoError(
            "evaluation output must be outside evaluator and immutable study trees"
        )
    return output


def _create_fresh_output_directory(
    value: str | Path,
    *,
    repo: Path,
    study_root: Path,
) -> None:
    output = _fresh_output_directory(
        value,
        repo=repo,
        study_root=study_root,
    )
    output.mkdir(mode=0o700)


def _canonical_evaluation_output_directory(
    value: str | Path,
    *,
    repo: Path,
    study_root: Path,
) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise NativeAllVideoError("output directory must be an absolute path")
    output = _canonical_directory(raw, "evaluation output")
    if output.is_relative_to(repo) or output.is_relative_to(study_root):
        raise NativeAllVideoError(
            "evaluation output must be outside evaluator and immutable study trees"
        )
    return output


def _validate_lockbox_arm_binding(
    split: SplitInput,
    arm_inputs: Mapping[str, ArmInput],
) -> None:
    if split.name != "lockbox":
        return
    report = split.validation_report
    if not isinstance(report, Mapping):
        raise NativeAllVideoError("lockbox lacks validation report")
    _validate_validation_report_arm_binding(report, arm_inputs)


def _validate_validation_report_arm_binding(
    report: Mapping[str, Any],
    arm_inputs: Mapping[str, ArmInput],
) -> None:
    recorded = report.get("input_provenance", {}).get("arms")
    if not isinstance(recorded, Mapping):
        raise NativeAllVideoError("validation report lacks arm provenance")
    for code in ARMS:
        value = recorded.get(code)
        arm = arm_inputs[code]
        if (
            not isinstance(value, Mapping)
            or value.get("arm_identity_sha256")
            != arm.arm_manifest["identity_sha256"]
            or value.get("stage_identity_sha256")
            != arm.stage_manifest["identity_sha256"]
            or value.get("stage_outcome_identity_sha256")
            != arm.stage_outcome["identity_sha256"]
            or value.get("snapshot", {}).get("sha256")
            != arm.snapshot_sha256
            or value.get("resolved_config", {}).get("sha256")
            != _sha256(arm.resolved_config)
        ):
            raise NativeAllVideoError(
                f"lockbox {code} artifacts differ from validation"
            )


def command_check_validation_gate(args: argparse.Namespace) -> int:
    repo = _canonical_directory(args.repo_root, "evaluator repository")
    study_root = _canonical_directory(args.study_root, "study root")
    _validate_clean_evaluator(
        repo,
        training_commit=args.training_commit,
        evaluator_commit=args.evaluator_commit,
    )
    _study_path, study = _validate_study(
        study_root,
        training_commit=args.training_commit,
    )
    if study.get("study_root") != str(study_root):
        raise NativeAllVideoError("study manifest root differs")
    arm_inputs = {
        code: _validate_arm_input(
            study_root,
            study,
            code=code,
            training_commit=args.training_commit,
            verify_snapshot_sha256=False,
        )
        for code in ARMS
    }
    _path, report = _validate_validation_report_before_lockbox(
        args.validation_report,
        study=study,
        training_commit=args.training_commit,
        evaluator_commit=args.evaluator_commit,
    )
    _validate_validation_report_arm_binding(report, arm_inputs)
    print(
        json.dumps(
            {
                "status": "passed",
                "validation_report_identity_sha256": report[
                    "identity_sha256"
                ],
                "primary_composite_gate_passed": True,
                "endpoint_equivalence_audit_passed": True,
                "lockbox_registration_inspected": False,
            },
            sort_keys=True,
        )
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    if args.batch_size_per_rank < 1:
        raise NativeAllVideoError("batch size per rank must be positive")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != EXPECTED_WORLD_SIZE:
        raise NativeAllVideoError(
            f"native-all-video evaluation requires {EXPECTED_WORLD_SIZE} ranks"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if "B200" not in properties.name.upper():
        raise NativeAllVideoError(
            f"native-all-video evaluation requires B200, found {properties.name}"
        )

    repo = _canonical_directory(args.repo_root, "evaluator repository")
    study_root = _canonical_directory(args.study_root, "study root")
    inference_compatibility = _validate_clean_evaluator(
        repo,
        training_commit=args.training_commit,
        evaluator_commit=args.evaluator_commit,
    )
    study_path, study = _validate_study(
        study_root,
        training_commit=args.training_commit,
    )
    if study.get("study_root") != str(study_root):
        raise NativeAllVideoError("study manifest root differs")
    videox_home = _canonical_directory(
        study.get("inputs", {}).get("runtime", {}).get("videox_home", ""),
        "pinned VideoX-Fun checkout",
    )
    try:
        videox_runtime = frontier.git_runtime_provenance(videox_home)
    except frontier.FrontierError as exc:
        raise NativeAllVideoError(str(exc)) from exc
    arm_inputs = {
        code: _validate_arm_input(
            study_root,
            study,
            code=code,
            training_commit=args.training_commit,
            verify_snapshot_sha256=(rank == 0),
        )
        for code in ARMS
    }
    split = _resolve_split(
        split_name=args.split,
        study=study,
        validation_report_value=args.validation_report,
        lockbox_registration_value=args.lockbox_registration,
        training_commit=args.training_commit,
        evaluator_commit=args.evaluator_commit,
        verify_cache_arrays=(rank == 0),
    )
    _validate_lockbox_arm_binding(split, arm_inputs)

    project_root = repo / "projects" / "latent_action_models"
    for root in (str(repo), str(project_root)):
        if root not in sys.path:
            sys.path.insert(0, root)
    configs = _load_resolved_configs(arm_inputs, split=split)
    dataset = _instantiate_dataset(configs["VPM"], split=split)
    assigned_indexes = _rank_indexes(rank, world_size, split.expected_clips)
    if len(assigned_indexes) % args.batch_size_per_rank:
        raise NativeAllVideoError(
            "each rank's clip count must divide batch size exactly"
        )
    if rank == 0:
        _create_fresh_output_directory(
            args.output_dir,
            repo=repo,
            study_root=study_root,
        )
    dist.barrier()
    output_dir = _canonical_evaluation_output_directory(
        args.output_dir,
        repo=repo,
        study_root=study_root,
    )

    compatibility_sha = hashlib.sha256(
        _canonical_json(inference_compatibility)
    ).hexdigest()
    videox_sha = hashlib.sha256(_canonical_json(videox_runtime)).hexdigest()
    rows, audits, total_counts = _evaluate_rank(
        arm_inputs=arm_inputs,
        configs=configs,
        dataset=dataset,
        split=split,
        device=device,
        assigned_indexes=assigned_indexes,
        batch_size_per_rank=args.batch_size_per_rank,
        training_commit=args.training_commit,
        evaluator_commit=args.evaluator_commit,
        inference_compatibility_sha256=compatibility_sha,
        videox_runtime_identity_sha256=videox_sha,
        world_size=world_size,
    )
    expected_call_counts = _expected_rank_call_counts(
        split_name=split.name,
        batches=len(assigned_indexes) // args.batch_size_per_rank,
    )
    if total_counts != expected_call_counts:
        raise NativeAllVideoError(
            f"rank {rank} total runtime calls differ: "
            f"{total_counts} != {expected_call_counts}"
        )
    expected_rows = len(assigned_indexes) * sum(
        len(RUNTIME_SOURCES[arm]) * len(NFE_GRID) for arm in ARMS
    )
    expected_audits = (
        0
        if split.name == "lockbox"
        else len(assigned_indexes) * len(ARMS) * len(ENDPOINT_AUDIT_K)
    )
    if len(rows) != expected_rows or len(audits) != expected_audits:
        raise NativeAllVideoError(
            f"rank {rank} evidence count differs: rows={len(rows)}/"
            f"{expected_rows}, audits={len(audits)}/{expected_audits}"
        )
    rows_path = output_dir / f"rank_{rank:03d}.jsonl"
    rows_bytes = b"".join(_canonical_json(row) + b"\n" for row in rows)
    _exclusive_bytes(rows_path, rows_bytes)
    audit_path: Path | None = None
    audit_bytes = b""
    if split.name == "validation":
        audit_path = output_dir / f"endpoint_audit_rank_{rank:03d}.jsonl"
        audit_bytes = b"".join(
            _canonical_json(row) + b"\n" for row in audits
        )
        _exclusive_bytes(audit_path, audit_bytes)
    rank_manifest = _identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_RANK,
            "created_at_utc": _now(),
            "split": split.name,
            "rank": rank,
            "world_size": world_size,
            "batch_size_per_rank": args.batch_size_per_rank,
            "assigned_clip_indexes": assigned_indexes,
            "study_identity_sha256": study["identity_sha256"],
            "training_git_commit": args.training_commit,
            "evaluator_git_commit": args.evaluator_commit,
            "inference_code_compatibility_sha256": compatibility_sha,
            "videox_runtime_identity_sha256": videox_sha,
            "arm_evidence": {
                code: {
                    "arm_identity_sha256": arm_inputs[code].arm_manifest[
                        "identity_sha256"
                    ],
                    "stage_identity_sha256": arm_inputs[code].stage_manifest[
                        "identity_sha256"
                    ],
                    "stage_outcome_identity_sha256": arm_inputs[
                        code
                    ].stage_outcome["identity_sha256"],
                    "snapshot_sha256": arm_inputs[code].snapshot_sha256,
                    "strict_state_dict_load": True,
                }
                for code in ARMS
            },
            "rows": {
                "path": str(rows_path),
                "sha256": hashlib.sha256(rows_bytes).hexdigest(),
                "bytes": len(rows_bytes),
                "count": len(rows),
            },
            "endpoint_audits": (
                None
                if audit_path is None
                else {
                    "path": str(audit_path),
                    "sha256": hashlib.sha256(audit_bytes).hexdigest(),
                    "bytes": len(audit_bytes),
                    "count": len(audits),
                }
            ),
            "actual_runtime_call_counts": total_counts,
            "online_teacher_call_count": 0,
            "clean_future_or_auxiliary_passed_to_sampler": False,
            "oracle_leakage": False,
            "evaluation_manifest_sha256": _sha256(split.manifest),
            "evaluation_cache_metadata_sha256": _sha256(
                split.cache_metadata
            ),
            "lockbox_registration_identity_sha256": (
                None
                if split.lockbox_registration is None
                else split.lockbox_registration["identity_sha256"]
            ),
            "device": {
                "name": properties.name,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
        }
    )
    rank_manifest_path = output_dir / f"rank_{rank:03d}_manifest.json"
    _exclusive_json(rank_manifest_path, rank_manifest)
    dist.barrier()

    if rank == 0:
        all_rows: list[dict[str, Any]] = []
        all_audits: list[dict[str, Any]] = []
        rank_evidence: dict[str, Any] = {}
        for other_rank in range(world_size):
            manifest_path = _canonical_file(
                output_dir / f"rank_{other_rank:03d}_manifest.json",
                f"rank {other_rank} manifest",
            )
            manifest = _read_json(manifest_path, f"rank {other_rank} manifest")
            other_rows_path = _canonical_file(
                output_dir / f"rank_{other_rank:03d}.jsonl",
                f"rank {other_rank} rows",
            )
            expected_indexes = _rank_indexes(
                other_rank, world_size, split.expected_clips
            )
            if (
                not _identity_valid(manifest)
                or manifest.get("kind") != KIND_RANK
                or manifest.get("rank") != other_rank
                or manifest.get("world_size") != world_size
                or manifest.get("split") != split.name
                or manifest.get("assigned_clip_indexes") != expected_indexes
                or manifest.get("study_identity_sha256")
                != study["identity_sha256"]
                or manifest.get("training_git_commit") != args.training_commit
                or manifest.get("evaluator_git_commit") != args.evaluator_commit
                or manifest.get("inference_code_compatibility_sha256")
                != compatibility_sha
                or manifest.get("videox_runtime_identity_sha256") != videox_sha
                or manifest.get("online_teacher_call_count") != 0
                or manifest.get("clean_future_or_auxiliary_passed_to_sampler")
                is not False
                or manifest.get("oracle_leakage") is not False
                or manifest.get("actual_runtime_call_counts")
                != _expected_rank_call_counts(
                    split_name=split.name,
                    batches=len(expected_indexes)
                    // args.batch_size_per_rank,
                )
                or manifest.get("rows", {}).get("path")
                != str(other_rows_path)
                or manifest.get("rows", {}).get("sha256")
                != _sha256(other_rows_path)
            ):
                raise NativeAllVideoError(
                    f"rank {other_rank} evidence manifest differs"
                )
            rank_rows = _read_jsonl(
                other_rows_path, f"rank {other_rank} rows"
            )
            if len(rank_rows) != manifest["rows"]["count"]:
                raise NativeAllVideoError(
                    f"rank {other_rank} row count differs"
                )
            all_rows.extend(rank_rows)
            audit_record = manifest.get("endpoint_audits")
            other_audits: list[dict[str, Any]] = []
            if split.name == "validation":
                other_audit_path = _canonical_file(
                    output_dir
                    / f"endpoint_audit_rank_{other_rank:03d}.jsonl",
                    f"rank {other_rank} endpoint audits",
                )
                if (
                    not isinstance(audit_record, Mapping)
                    or audit_record.get("path") != str(other_audit_path)
                    or audit_record.get("sha256") != _sha256(other_audit_path)
                ):
                    raise NativeAllVideoError(
                        f"rank {other_rank} endpoint-audit evidence differs"
                    )
                other_audits = _read_jsonl(
                    other_audit_path,
                    f"rank {other_rank} endpoint audits",
                )
                if len(other_audits) != audit_record["count"]:
                    raise NativeAllVideoError(
                        f"rank {other_rank} endpoint-audit count differs"
                    )
                all_audits.extend(other_audits)
            elif audit_record is not None:
                raise NativeAllVideoError(
                    "lockbox rank unexpectedly contains endpoint audits"
                )
            rank_evidence[str(other_rank)] = {
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": _sha256(manifest_path),
                    "identity_sha256": manifest["identity_sha256"],
                },
                "rows": dict(manifest["rows"]),
                "endpoint_audits": (
                    None if audit_record is None else dict(audit_record)
                ),
                "actual_runtime_call_counts": manifest[
                    "actual_runtime_call_counts"
                ],
            }
        report = _build_report(
            split=split,
            rows=all_rows,
            audits=all_audits,
            arm_inputs=arm_inputs,
            study_path=study_path,
            study=study,
            training_commit=args.training_commit,
            evaluator_commit=args.evaluator_commit,
            inference_compatibility=inference_compatibility,
            videox_runtime=videox_runtime,
            world_size=world_size,
            batch_size_per_rank=args.batch_size_per_rank,
            rank_evidence=rank_evidence,
        )
        report_path = output_dir / "report.json"
        _exclusive_json(report_path, report)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "split": split.name,
                    "report": str(report_path),
                    "report_identity_sha256": report["identity_sha256"],
                    "primary_composite_gate_passed": report[
                        "primary_composite_gate"
                    ]["passed"],
                    "lockbox_eligible": report["lockbox_eligible"],
                },
                sort_keys=True,
            )
        )
    dist.barrier()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_identity_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repo-root", required=True)
        command.add_argument("--study-root", required=True)
        command.add_argument(
            "--training-commit",
            default=TRAINING_COMMIT,
            help="immutable faithful-cascade training commit",
        )
        command.add_argument(
            "--evaluator-commit",
            required=True,
            help=(
                "clean descendant tool commit with unchanged "
                "inference-critical tree"
            ),
        )

    evaluate = commands.add_parser(
        "evaluate", help="run distributed validation or eligible lockbox scoring"
    )
    add_identity_arguments(evaluate)
    evaluate.add_argument(
        "--split", choices=("validation", "lockbox"), required=True
    )
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument(
        "--validation-report",
        default=None,
        help=(
            "required only for lockbox; must be this evaluator's eligible "
            "validation report"
        ),
    )
    evaluate.add_argument(
        "--lockbox-registration",
        default=None,
        help=(
            "required only for lockbox; not inspected until validation gate "
            "has passed"
        ),
    )
    evaluate.add_argument(
        "--batch-size-per-rank",
        type=int,
        default=DEFAULT_BATCH_SIZE_PER_RANK,
    )

    gate = commands.add_parser(
        "check-validation-gate",
        help=(
            "validate composite K=4/report/final-arm bindings without reading "
            "a lockbox registration"
        ),
    )
    add_identity_arguments(gate)
    gate.add_argument("--validation-report", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "check-validation-gate":
            return command_check_validation_gate(args)
        if args.command == "evaluate":
            return command_evaluate(args)
        raise NativeAllVideoError(f"unsupported command: {args.command}")
    except NativeAllVideoError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
