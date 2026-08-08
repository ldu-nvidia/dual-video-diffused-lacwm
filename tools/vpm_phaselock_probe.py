#!/usr/bin/env python3
"""Validation-only latent-delta guidance probe for the frozen VPM generator.

The probe adapts PhaseLock to the VPM checkpoint from the immutable V-JEPA 2
controlled study.  A cheap preliminary rollout produces a *generated* video
latent.  Its future-frame temporal deltas guide a second rollout that starts
from exactly the same Gaussian noise.  Neither rollout accepts clean future
RGB, a clean auxiliary target, nor an online teacher.

The implementation deliberately lives outside the historical model tree.  It
loads the checkpoint with the clean training checkout recorded by the study,
so later research changes cannot silently alter checkpoint inference.  Every
quality endpoint is rerun independently and a forward hook proves that its
actual Wan/transformer calls equal its declared total-call budget.

This file has two commands:

``register``
    Freeze code, checkpoint, validation data, and the predeclared endpoint
    grid before any candidate metric is computed.

``evaluate``
    Run the fixed grid on all 64 immutable validation clips with eight B200
    ranks.  The protected test split is not accepted by the CLI or opened by
    this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
KIND_REGISTRATION = "vpm_phaselock_probe_registration"
KIND_ROW = "vpm_phaselock_probe_clip"
KIND_RANK = "vpm_phaselock_probe_rank"
KIND_INVENTORY = "vpm_phaselock_probe_inventory"

TRAINING_COMMIT = "9cf8e6922f35a5d6645e3128545953723bf54da2"
EXPECTED_STUDY_KIND = "vjepa2_controlled_video_diffusion_study"
EXPECTED_STUDY_ID = "vjepa2-controlled-20260730-seed1234-9cf8e69-v3"
EXPECTED_STUDY_IDENTITY_SHA256 = (
    "dc720b8c1d417cb8cbef6f5bd9ab41b35a650c8b64115e15a61e84606b306ac4"
)
EXPECTED_STUDY_ROOT = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/"
    "lacwm_train/runs/dual_video_diffusion/vjepa2_controlled_study/"
    "vjepa2-controlled-20260730-seed1234-9cf8e69-v3"
)
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "8cb39c1f056855e28855c0b944c715d084b709e1421f4efeed3710e7099348c4"
)
EXPECTED_VALIDATION_METADATA_SHA256 = (
    "9bb873cf373aea4aa0e28319365a7e0492e63e110afb28dd8d358d5ee1cbb3f6"
)
EXPECTED_VALIDATION_RGB_SHA256 = (
    "ed82fc0f580baa90c4dc39c1608f97ad7092a4ddcb59c3612eed31711afda404"
)
EXPECTED_VALIDATION_ACTIONS_SHA256 = (
    "552a5cf0af156868d2866dfacabe102fc6b5cd24580bb377953e35a14625306a"
)
PHASELOCK_REFERENCE_COMMIT = "4ebb65ba1348e754723f08a190723cf82294db19"
PHASELOCK_REFERENCE_URL = "https://github.com/dnwjddl/phaselock"
PHASELOCK_PAPER_URL = "https://arxiv.org/abs/2606.06361"
VIDEOX_COMMIT = "1d6d9c3e1540968466937129fef4b288041e06de"
EXPECTED_WORLD_SIZE = 8
EXPECTED_BATCH_SIZE_PER_RANK = 2
EXPECTED_VALIDATION_CLIPS = 64
VALIDATION_SAMPLE_ID_OFFSET = 1_000_000
FUTURE_RGB_FRAMES = 8
GUIDANCE_STRENGTH = 0.05
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PhaseLockProbeError(RuntimeError):
    """Raised when an endpoint would be incomparable or non-deployable."""


@dataclass(frozen=True)
class Endpoint:
    code: str
    kind: str
    total_transformer_calls: int
    few_steps: int
    full_steps: int
    prior_alignment: str


def endpoint_grid() -> tuple[Endpoint, ...]:
    """Return the immutable validation endpoint grid.

    K=1 is the primary adaptation because the already frozen VPM validation
    frontier selected NFE=1.  K=2 is retained as the faithful PhaseLock prior
    extraction setting.  All guided endpoints are paired with a sample-shuffle
    control and an ordinary sampler at the exact same total transformer calls.
    """

    ordinary = tuple(
        Endpoint(
            code=f"ordinary_b{budget}",
            kind="ordinary",
            total_transformer_calls=budget,
            few_steps=0,
            full_steps=budget,
            prior_alignment="none",
        )
        for budget in (1, 3, 4, 6)
    )
    candidates = (
        ("k1_f2", 1, 2),
        ("k1_f3", 1, 3),
        ("k2_f2", 2, 2),
        ("k2_f4", 2, 4),
    )
    guided: list[Endpoint] = []
    for name, few_steps, full_steps in candidates:
        budget = few_steps + full_steps
        for alignment in ("aligned", "shuffled"):
            guided.append(
                Endpoint(
                    code=f"phaselock_{name}_{alignment}",
                    kind="phaselock",
                    total_transformer_calls=budget,
                    few_steps=few_steps,
                    full_steps=full_steps,
                    prior_alignment=alignment,
                )
            )
    return ordinary + tuple(guided)


ENDPOINTS = endpoint_grid()
ENDPOINT_BY_CODE = {endpoint.code: endpoint for endpoint in ENDPOINTS}


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


def identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: Mapping[str, Any]) -> bool:
    recorded = payload.get("identity_sha256")
    if not isinstance(recorded, str) or SHA256_RE.fullmatch(recorded) is None:
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


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _slice_hashes(tensor: Any) -> list[str]:
    return [_tensor_sha256(tensor[index : index + 1]) for index in range(len(tensor))]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PhaseLockProbeError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise PhaseLockProbeError(f"{label} must contain one JSON object: {path}")
    return payload


def _regular_file(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        raise PhaseLockProbeError(f"{label} must be absolute: {path}")
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise PhaseLockProbeError(
            f"{label} must be a non-empty, non-symlink file: {path}"
        )
    return path.resolve(strict=True)


def _directory(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise PhaseLockProbeError(
            f"{label} must be an absolute, non-symlink directory: {path}"
        )
    return path.resolve(strict=True)


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise PhaseLockProbeError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _assert_clean_commit(repo: Path, expected: str, label: str) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(expected) is None:
        raise PhaseLockProbeError(f"{label} expected commit is invalid")
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != expected:
        raise PhaseLockProbeError(f"{label} commit differs: {actual} != {expected}")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise PhaseLockProbeError(f"{label} repository must be clean")
    if _git(repo, "rev-parse", "--show-toplevel") != str(repo):
        raise PhaseLockProbeError(f"{label} path is not a Git worktree root")
    return {
        "path": str(repo),
        "git_commit": actual,
        "git_tree_sha": _git(repo, "rev-parse", "HEAD^{tree}"),
        "clean": True,
    }


def _file_record(path: Path, *, rehash: bool = True) -> dict[str, Any]:
    path = _regular_file(path, "registered input")
    record = {"path": str(path), "bytes": path.stat().st_size}
    if rehash:
        record["sha256"] = _sha256(path)
    return record


def _exclusive_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PhaseLockProbeError(f"refusing to overwrite output: {path}") from exc
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


def guidance_end_step(full_steps: int) -> int:
    """Faithful early-half guidance interval, with one valid low-NFE step."""

    if isinstance(full_steps, bool) or not isinstance(full_steps, int) or full_steps < 1:
        raise PhaseLockProbeError("full_steps must be a positive integer")
    return max(1, full_steps // 2)


def linear_guidance_strength(
    step_index: int,
    full_steps: int,
    *,
    initial_strength: float = GUIDANCE_STRENGTH,
) -> float:
    """Return PhaseLock's linearly decaying early-trajectory strength."""

    if (
        isinstance(step_index, bool)
        or not isinstance(step_index, int)
        or step_index < 0
        or not math.isfinite(initial_strength)
        or initial_strength < 0
    ):
        raise PhaseLockProbeError("invalid guidance schedule arguments")
    end = guidance_end_step(full_steps)
    if step_index >= end:
        return 0.0
    return initial_strength * (1.0 - step_index / end)


def extract_future_motion_prior(latents: Any, history_frames: int) -> Any:
    """Compute generated-future deltas for a ``[B,C,T,H,W]`` Wan latent.

    The first generated future latent is the anchor and is not modified.  This
    is the channel-first equivalent of PhaseLock's ``z[:,1:] - z[:,:-1]``.
    Observed history is excluded so no clean future or history-to-future oracle
    delta can enter the prior.
    """

    if getattr(latents, "ndim", None) != 5:
        raise PhaseLockProbeError("video latents must have shape [B,C,T,H,W]")
    temporal = int(latents.shape[2])
    if (
        isinstance(history_frames, bool)
        or not isinstance(history_frames, int)
        or history_frames < 1
        or temporal - history_frames < 2
    ):
        raise PhaseLockProbeError(
            "motion prior requires observed history and at least two future latents"
        )
    future = latents[:, :, history_frames:]
    return future[:, :, 1:] - future[:, :, :-1]


def apply_future_delta_guidance(
    latents: Any,
    motion_prior: Any,
    history_frames: int,
    strength: float,
) -> Any:
    """Apply latent-delta guidance without changing history/first future latent."""

    if not math.isfinite(float(strength)) or strength < 0:
        raise PhaseLockProbeError("guidance strength must be finite and nonnegative")
    current = extract_future_motion_prior(latents, history_frames)
    if tuple(current.shape) != tuple(motion_prior.shape):
        raise PhaseLockProbeError(
            "motion-prior shape differs from the current future-latent deltas"
        )
    if strength == 0:
        return latents
    guided = latents.clone()
    guided[:, :, history_frames + 1 :] = (
        latents[:, :, history_frames + 1 :]
        + float(strength) * (motion_prior.to(latents) - current)
    )
    return guided


def shuffled_motion_prior(motion_prior: Any, sampling_ids: Any) -> tuple[Any, Any]:
    """Roll priors within the fixed local pair and return donor sample IDs."""

    if motion_prior.shape[0] < 2 or int(sampling_ids.numel()) != motion_prior.shape[0]:
        raise PhaseLockProbeError(
            "shuffled control requires at least two paired batch elements"
        )
    return motion_prior.roll(1, dims=0), sampling_ids.reshape(-1).roll(1, dims=0)


def _study_paths(study_root: Path) -> dict[str, Path]:
    run_dir = study_root / "vpm_parameter_matched_video"
    return {
        "study": study_root / "study_manifest.json",
        "run_dir": run_dir,
        "arm": run_dir / "arm_manifest.json",
        "stage": run_dir / "stage_manifest_update_1000.json",
        "outcome": run_dir / "stage_outcome_update_1000.json",
        "config": run_dir / "resolved_update_1000.yaml",
        "snapshot": run_dir / "snapshot.pt",
    }


def _validate_expected_study_identity(
    study: Mapping[str, Any], study_root: Path
) -> None:
    """Pin the exact validation study named by the prospective protocol."""

    if (
        study.get("kind") != EXPECTED_STUDY_KIND
        or study.get("study_id") != EXPECTED_STUDY_ID
        or study.get("identity_sha256") != EXPECTED_STUDY_IDENTITY_SHA256
        or study_root != EXPECTED_STUDY_ROOT
        or Path(str(study.get("study_root", ""))) != EXPECTED_STUDY_ROOT
    ):
        raise PhaseLockProbeError(
            "controlled-study kind, ID, identity, or exact root differs"
        )


def _canonical_fresh_lustre_output(path: Path) -> Path:
    """Return a canonical nonexistent output whose existing parent is on Lustre."""

    path = path.expanduser()
    if not path.is_absolute() or not path.name or path.name in {".", ".."}:
        raise PhaseLockProbeError("probe output must be an absolute named path")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise PhaseLockProbeError("probe output parent must already exist") from exc
    canonical = parent / path.name
    lustre = Path("/lustre")
    if canonical != path or not (canonical == lustre or lustre in canonical.parents):
        raise PhaseLockProbeError(
            "probe output must be canonical, symlink-free, and under /lustre"
        )
    if canonical.exists() or canonical.is_symlink():
        raise PhaseLockProbeError(f"fresh probe output already exists: {canonical}")
    return canonical


def _validate_study_metadata(
    study_root: Path,
    model_repo: Path,
    *,
    rehash_snapshot: bool,
) -> dict[str, Any]:
    paths = _study_paths(study_root)
    for label, path in paths.items():
        if label != "run_dir":
            _regular_file(path, label)
    study = _read_json(paths["study"], "study manifest")
    arm = _read_json(paths["arm"], "VPM arm manifest")
    stage = _read_json(paths["stage"], "VPM stage manifest")
    outcome = _read_json(paths["outcome"], "VPM stage outcome")
    for label, payload in (
        ("study", study),
        ("arm", arm),
        ("stage", stage),
        ("outcome", outcome),
    ):
        if not identity_valid(payload):
            raise PhaseLockProbeError(f"{label} identity is invalid")
    _validate_expected_study_identity(study, study_root)
    if (
        arm.get("kind") != "vjepa2_controlled_study_arm"
        or arm.get("git_commit") != TRAINING_COMMIT
        or arm.get("arm", {}).get("code") != "VPM"
        or arm.get("arm", {}).get("parameter_matched_control") is not True
        or arm.get("arm", {}).get("condition_on_auxiliary_state") is not False
        or arm.get("arm", {}).get("condition_on_auxiliary_clock") is not False
        or arm.get("arm", {}).get("auxiliary_loss_weight") != 0.0
        or arm.get("study_identity_sha256") != study.get("identity_sha256")
        or Path(str(arm.get("run_dir", ""))).resolve() != paths["run_dir"]
    ):
        raise PhaseLockProbeError("VPM arm contract differs from the frozen study")
    if (
        stage.get("kind") != "vjepa2_controlled_study_stage"
        or stage.get("arm_code") != "VPM"
        or stage.get("arm_identity_sha256") != arm.get("identity_sha256")
        or stage.get("stage_endpoint_completed_updates") != 1000
        or stage.get("primary_milestone") is not True
        or outcome.get("kind") != "vjepa2_controlled_study_stage_outcome"
        or outcome.get("arm_identity_sha256") != arm.get("identity_sha256")
        or outcome.get("stage_identity_sha256") != stage.get("identity_sha256")
        or outcome.get("completed_updates") != 1000
    ):
        raise PhaseLockProbeError("VPM update-1000 stage provenance differs")
    resolved = stage.get("resolved_config")
    if (
        not isinstance(resolved, Mapping)
        or Path(str(resolved.get("path", ""))).resolve() != paths["config"]
        or resolved.get("sha256") != _sha256(paths["config"])
    ):
        raise PhaseLockProbeError("VPM resolved config differs from stage manifest")
    snapshot_record = outcome.get("snapshot_observed_at_stage_end")
    if (
        not isinstance(snapshot_record, Mapping)
        or Path(str(snapshot_record.get("path", ""))).resolve()
        != paths["snapshot"]
        or not isinstance(snapshot_record.get("sha256"), str)
        or SHA256_RE.fullmatch(str(snapshot_record["sha256"])) is None
        or (rehash_snapshot and _sha256(paths["snapshot"]) != snapshot_record["sha256"])
    ):
        raise PhaseLockProbeError("VPM snapshot differs from stage outcome")

    repository = study.get("inputs", {}).get("repository", {})
    if (
        repository.get("git_commit") != TRAINING_COMMIT
        or repository.get("clean") is not True
        or Path(str(repository.get("root", ""))).resolve() != model_repo
    ):
        raise PhaseLockProbeError("study/model-source checkout binding differs")
    validation = study.get("inputs", {}).get("splits", {}).get("validation")
    if not isinstance(validation, Mapping):
        raise PhaseLockProbeError("study lacks the immutable validation split")
    manifest_record = validation.get("clip_manifest")
    cache_record = validation.get("cache")
    if (
        not isinstance(manifest_record, Mapping)
        or manifest_record.get("entries") != EXPECTED_VALIDATION_CLIPS
        or manifest_record.get("sha256")
        != EXPECTED_VALIDATION_MANIFEST_SHA256
        or not isinstance(cache_record, Mapping)
        or cache_record.get("split") != "val"
        or cache_record.get("clip_count") != EXPECTED_VALIDATION_CLIPS
    ):
        raise PhaseLockProbeError("validation split does not contain 64 pinned clips")
    val_manifest = _regular_file(
        Path(str(manifest_record.get("path", ""))), "validation manifest"
    )
    metadata_record = cache_record.get("metadata")
    if not isinstance(metadata_record, Mapping):
        raise PhaseLockProbeError("validation cache metadata record is absent")
    val_metadata = _regular_file(
        Path(str(metadata_record.get("path", ""))), "validation cache metadata"
    )
    if (
        manifest_record.get("sha256") != _sha256(val_manifest)
        or metadata_record.get("sha256") != _sha256(val_metadata)
        or metadata_record.get("sha256")
        != EXPECTED_VALIDATION_METADATA_SHA256
    ):
        raise PhaseLockProbeError("validation manifest/cache digest differs")
    descriptors = [
        json.loads(line)
        for line in val_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(descriptors) != EXPECTED_VALIDATION_CLIPS
        or [int(row.get("auxiliary_index", -1)) for row in descriptors]
        != list(range(EXPECTED_VALIDATION_CLIPS))
        or len({str(row.get("clip_id", "")) for row in descriptors})
        != EXPECTED_VALIDATION_CLIPS
        or any(row.get("split") != "val" for row in descriptors)
    ):
        raise PhaseLockProbeError("validation descriptors are not dense and unique")
    array_records: dict[str, dict[str, Any]] = {}
    cache_metadata_payload = _read_json(
        val_metadata, "validation cache metadata"
    )
    if (
        cache_metadata_payload.get("split") != "val"
        or cache_metadata_payload.get("clip_count")
        != EXPECTED_VALIDATION_CLIPS
        or cache_metadata_payload.get("complete") is not True
        or cache_metadata_payload.get("clip_manifest_sha256")
        != EXPECTED_VALIDATION_MANIFEST_SHA256
    ):
        raise PhaseLockProbeError("validation cache metadata split changed")
    for name in ("rgb", "actions"):
        record = cache_record.get(name)
        if not isinstance(record, Mapping):
            raise PhaseLockProbeError(f"study lacks validation {name} cache")
        path = _regular_file(Path(str(record.get("path", ""))), f"validation {name}")
        if (
            not isinstance(record.get("sha256"), str)
            or SHA256_RE.fullmatch(str(record["sha256"])) is None
            or record.get("bytes") != path.stat().st_size
            or (rehash_snapshot and _sha256(path) != record["sha256"])
        ):
            raise PhaseLockProbeError(f"validation {name} cache record is invalid")
        metadata_path = Path(
            str(cache_metadata_payload.get(f"{name}_file", ""))
        )
        if not metadata_path.is_absolute():
            metadata_path = val_metadata.parent / metadata_path
        expected_shape = (
            [EXPECTED_VALIDATION_CLIPS, 13, 3, 180, 960]
            if name == "rgb"
            else [EXPECTED_VALIDATION_CLIPS, 13, 5, 23]
        )
        expected_dtype = "float16" if name == "rgb" else "float32"
        expected_sha256 = (
            EXPECTED_VALIDATION_RGB_SHA256
            if name == "rgb"
            else EXPECTED_VALIDATION_ACTIONS_SHA256
        )
        if (
            metadata_path.resolve() != path
            or cache_metadata_payload.get(f"{name}_sha256")
            != record["sha256"]
            or record.get("sha256") != expected_sha256
            or cache_metadata_payload.get(f"{name}_shape") != expected_shape
            or cache_metadata_payload.get(f"{name}_dtype") != expected_dtype
        ):
            raise PhaseLockProbeError(
                f"validation {name} array differs from cache metadata"
            )
        array_records[name] = {
            "path": str(path),
            "sha256": record["sha256"],
            "bytes": path.stat().st_size,
            "full_sha256_verified": bool(rehash_snapshot),
        }
    runtime = study.get("inputs", {}).get("runtime", {})
    videox = _directory(Path(str(runtime.get("videox_home", ""))), "VideoX checkout")
    if (
        _git(videox, "rev-parse", "HEAD") != VIDEOX_COMMIT
        or _git(videox, "status", "--porcelain", "--untracked-files=all")
    ):
        raise PhaseLockProbeError("VideoX runtime is not the frozen clean checkout")
    wan_dir = _directory(Path(str(runtime.get("wan_dir", ""))), "Wan directory")
    return {
        "paths": paths,
        "study": study,
        "arm": arm,
        "stage": stage,
        "outcome": outcome,
        "validation": {
            "manifest": _file_record(val_manifest),
            "cache_metadata": _file_record(val_metadata),
            "arrays": array_records,
            "clip_ids_sha256": hashlib.sha256(
                _canonical_json([str(row["clip_id"]) for row in descriptors])
            ).hexdigest(),
        },
        "descriptors": descriptors,
        "runtime": {
            "videox_home": str(videox),
            "videox_commit": VIDEOX_COMMIT,
            "wan_dir": str(wan_dir),
        },
        "snapshot_sha256": snapshot_record["sha256"],
    }


def command_register(args: argparse.Namespace) -> int:
    tool_repo = _directory(args.tool_repo, "tool repository")
    model_repo = _directory(args.model_repo, "historical model repository")
    if tool_repo == model_repo:
        raise PhaseLockProbeError("tool and historical model repositories must differ")
    tool_identity = _assert_clean_commit(
        tool_repo, args.tool_commit, "tool"
    )
    model_identity = _assert_clean_commit(
        model_repo, TRAINING_COMMIT, "historical model"
    )
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(tool_repo),
            "merge-base",
            "--is-ancestor",
            TRAINING_COMMIT,
            args.tool_commit,
        ],
        check=False,
    )
    if completed.returncode:
        raise PhaseLockProbeError("tool commit is not a descendant of training commit")
    study_root = _directory(args.study_root, "controlled-study root")
    validated = _validate_study_metadata(
        study_root, model_repo, rehash_snapshot=True
    )
    output_root = _canonical_fresh_lustre_output(args.output_root)
    output_root.mkdir(parents=True, mode=0o700)
    protocol_files = {}
    for relative in (
        "docs/experiments/VPM_PHASELOCK_PROBE_PROTOCOL.md",
        "tools/vpm_phaselock_probe.py",
        "tools/analyze_vpm_phaselock_probe.py",
        "tools/slurm/vpm_phaselock_probe.sbatch",
    ):
        path = _regular_file(tool_repo / relative, f"protocol implementation {relative}")
        protocol_files[relative] = _file_record(path)
    paths = validated["paths"]
    registration = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_REGISTRATION,
            "created_at_utc": _now(),
            "output_root": str(output_root),
            "tool_repository": tool_identity,
            "historical_model_repository": model_identity,
            "training_commit": TRAINING_COMMIT,
            "phaselock_reference": {
                "paper": PHASELOCK_PAPER_URL,
                "repository": PHASELOCK_REFERENCE_URL,
                "inspected_commit": PHASELOCK_REFERENCE_COMMIT,
                "adaptation": (
                    "direct channel-first Wan future-latent deltas; first future "
                    "latent anchored; no decode/re-encode in prior extraction"
                ),
            },
            "protocol_files": protocol_files,
            "fixed_protocol": {
                "split": "validation",
                "protected_test_access_allowed": False,
                "clip_count": EXPECTED_VALIDATION_CLIPS,
                "world_size": EXPECTED_WORLD_SIZE,
                "batch_size_per_rank": EXPECTED_BATCH_SIZE_PER_RANK,
                "sampling_id_offset": VALIDATION_SAMPLE_ID_OFFSET,
                "guidance_strength": GUIDANCE_STRENGTH,
                "guidance_interval": "steps [0,max(1,floor(K_full/2)))",
                "future_motion_operator": (
                    "D(z)=z[:,:,h+1:]-z[:,:,h:-1]; h=history latent count"
                ),
                "first_future_latent_is_anchor": True,
                "shuffled_control": (
                    "cyclic roll by one within each fixed two-sample local batch"
                ),
                "endpoint_grid": [asdict(endpoint) for endpoint in ENDPOINTS],
                "metric_owner": "evaluator only; targets never enter sampler",
                "online_teacher_calls": 0,
                "clean_future_or_auxiliary_passed_to_sampler": False,
                "clean_auxiliary_target_array_opened": False,
                "transformer_call_accounting": "forward hook, exact per endpoint",
            },
            "controlled_study": {
                "root": str(study_root),
                "study_manifest": _file_record(paths["study"]),
                "study_identity_sha256": validated["study"]["identity_sha256"],
                "arm_manifest": _file_record(paths["arm"]),
                "arm_identity_sha256": validated["arm"]["identity_sha256"],
                "stage_manifest": _file_record(paths["stage"]),
                "stage_identity_sha256": validated["stage"]["identity_sha256"],
                "stage_outcome": _file_record(paths["outcome"]),
                "stage_outcome_identity_sha256": validated["outcome"][
                    "identity_sha256"
                ],
                "resolved_config": _file_record(paths["config"]),
                "snapshot": {
                    "path": str(paths["snapshot"]),
                    "bytes": paths["snapshot"].stat().st_size,
                    "sha256": validated["snapshot_sha256"],
                    "full_sha256_verified_at_registration": True,
                },
            },
            "validation": validated["validation"],
            "runtime": validated["runtime"],
            "claim_scope": (
                "validation-only mechanistic evidence; no protected-test, FVD, "
                "latency, downstream DAgger, or generalization claim"
            ),
        }
    )
    _exclusive_json(output_root / "registration.json", registration)
    print(str(output_root / "registration.json"))
    return 0


def _validate_registration(
    path: Path,
    *,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _regular_file(path, "probe registration")
    registration = _read_json(path, "probe registration")
    if (
        not identity_valid(registration)
        or registration.get("kind") != KIND_REGISTRATION
        or registration.get("training_commit") != TRAINING_COMMIT
        or registration.get("fixed_protocol", {}).get("split") != "validation"
        or registration.get("fixed_protocol", {}).get(
            "protected_test_access_allowed"
        )
        is not False
        or registration.get("fixed_protocol", {}).get("endpoint_grid")
        != [asdict(endpoint) for endpoint in ENDPOINTS]
        or registration.get("fixed_protocol", {}).get("guidance_strength")
        != GUIDANCE_STRENGTH
    ):
        raise PhaseLockProbeError("registration identity/protocol differs")
    tool_record = registration.get("tool_repository", {})
    model_record = registration.get("historical_model_repository", {})
    tool_repo = _directory(Path(str(tool_record.get("path", ""))), "tool repository")
    model_repo = _directory(
        Path(str(model_record.get("path", ""))), "historical model repository"
    )
    _assert_clean_commit(tool_repo, str(tool_record.get("git_commit", "")), "tool")
    _assert_clean_commit(model_repo, TRAINING_COMMIT, "historical model")
    for relative, record in registration.get("protocol_files", {}).items():
        path_value = _regular_file(Path(str(record.get("path", ""))), relative)
        if (
            path_value != tool_repo / relative
            or record.get("bytes") != path_value.stat().st_size
            or record.get("sha256") != _sha256(path_value)
        ):
            raise PhaseLockProbeError(f"registered protocol file changed: {relative}")
    controlled = registration.get("controlled_study", {})
    study_root = _directory(Path(str(controlled.get("root", ""))), "study root")
    validated = _validate_study_metadata(
        study_root,
        model_repo,
        rehash_snapshot=(rank == 0),
    )
    for key, identity_key, payload_key in (
        ("study_manifest", "study_identity_sha256", "study"),
        ("arm_manifest", "arm_identity_sha256", "arm"),
        ("stage_manifest", "stage_identity_sha256", "stage"),
        ("stage_outcome", "stage_outcome_identity_sha256", "outcome"),
    ):
        record = controlled.get(key, {})
        file_path = _regular_file(Path(str(record.get("path", ""))), key)
        if (
            record.get("sha256") != _sha256(file_path)
            or record.get("bytes") != file_path.stat().st_size
            or controlled.get(identity_key)
            != validated[payload_key]["identity_sha256"]
        ):
            raise PhaseLockProbeError(f"registered controlled-study {key} changed")
    registered_validation = registration.get("validation", {})
    for key in ("manifest", "cache_metadata"):
        registered_record = registered_validation.get(key, {})
        observed_record = validated["validation"][key]
        if any(
            registered_record.get(field) != observed_record.get(field)
            for field in ("path", "sha256", "bytes")
        ):
            raise PhaseLockProbeError(
                f"registered validation {key} changed"
            )
    for key in ("rgb", "actions"):
        registered_record = registered_validation.get("arrays", {}).get(key, {})
        observed_record = validated["validation"]["arrays"][key]
        if any(
            registered_record.get(field) != observed_record.get(field)
            for field in ("path", "sha256", "bytes")
        ):
            raise PhaseLockProbeError(
                f"registered validation {key} array changed"
            )
    if (
        registered_validation.get("clip_ids_sha256")
        != validated["validation"]["clip_ids_sha256"]
    ):
        raise PhaseLockProbeError("registered validation clip identities changed")
    return registration, validated


def _load_model_and_dataset(
    registration: Mapping[str, Any],
    validated: Mapping[str, Any],
    *,
    device: Any,
) -> tuple[Any, Any, Any]:
    import torch

    model_repo = Path(registration["historical_model_repository"]["path"])
    project_root = model_repo / "projects" / "latent_action_models"
    historical_shim = model_repo / "tools" / "env" / "videox_shim"
    videox_root = Path(registration["runtime"]["videox_home"])
    # The script directory is the tool checkout, not its repository root. Put
    # the frozen historical model, project package, and VideoX shim first
    # before Hydra resolves model classes. This prevents the current research
    # checkout from changing the old checkpoint's import semantics.
    ordered_roots = (
        str(model_repo),
        str(project_root),
        str(historical_shim),
        str(videox_root),
    )
    for root in reversed(ordered_roots):
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    os.environ["WAN_DIR"] = str(registration["runtime"]["wan_dir"])
    os.environ["VIDEOX_HOME"] = str(registration["runtime"]["videox_home"])
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(validated["paths"]["config"])
    if config.get("wandb", {}).get("enabled", False):
        config.wandb.enabled = False
    if not str(config.model.get("_target_", "")).endswith(
        ".DualExplicitActionDiTModel"
    ):
        raise PhaseLockProbeError("VPM config has an unexpected model class")
    model = instantiate(config.model)
    model_module = Path(sys.modules[type(model).__module__].__file__).resolve()
    if model_repo not in model_module.parents:
        raise PhaseLockProbeError(
            f"model imported outside frozen historical checkout: {model_module}"
        )
    snapshot = torch.load(
        validated["paths"]["snapshot"],
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if (
        not isinstance(snapshot, Mapping)
        or "model" not in snapshot
        or snapshot.get("run_identity_sha256")
        != validated["arm"]["identity_sha256"]
        or snapshot.get("_start_iter") != 1000
    ):
        raise PhaseLockProbeError("VPM snapshot identity/update cursor differs")
    incompatible = model.load_state_dict(snapshot["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise PhaseLockProbeError(f"strict VPM checkpoint load failed: {incompatible}")
    del snapshot
    model = model.to(device=device).eval()
    model._ensure_video_only_runtime_contract()
    if (
        model.tf_condition_mode != "off"
        or model.condition_on_tf
        or bool(getattr(model, "condition_on_tf_clock", False))
        or model.tf_loss_weight != 0.0
        or not bool(getattr(model, "parameter_matched_control", False))
        or float(
            model.forward_model.tf_token_adapter.effective_gate().detach().float()
        )
        != 0.0
        or float(
            model.forward_model.tf_clock_embedding.effective_gate().detach().float()
        )
        != 0.0
    ):
        raise PhaseLockProbeError("loaded checkpoint is not the exact VPM no-op arm")
    suspicious = []
    for name, module in model.named_modules():
        identity = f"{name}:{type(module).__module__}.{type(module).__name__}".lower()
        if "vjepa" in identity or "jepa" in identity or "teacher" in identity:
            suspicious.append(identity)
    if suspicious or getattr(model, "time_frequency_transform", None) is not None:
        raise PhaseLockProbeError("an online feature/teacher module is registered")
    abc_config = config.val_dataset.datasets.ABC
    expected_manifest = Path(registration["validation"]["manifest"]["path"])
    expected_metadata = Path(registration["validation"]["cache_metadata"]["path"])
    if (
        Path(str(abc_config.clip_manifest)) != expected_manifest
        or Path(str(abc_config.cache_metadata)) != expected_metadata
        or int(config.val_dataset.padding_dim) != 157
        or bool(config.val_dataset.img_augment)
    ):
        raise PhaseLockProbeError("resolved dataset differs from registered validation")
    dataset = _RegisteredValidationInputs(
        rgb_path=Path(registration["validation"]["arrays"]["rgb"]["path"]),
        actions_path=Path(
            registration["validation"]["arrays"]["actions"]["path"]
        ),
        descriptors=validated["descriptors"],
        padding_dim=int(config.val_dataset.padding_dim),
    )
    return model, dataset, config


class _RegisteredValidationInputs:
    """Read only RGB/actions; the cached clean V-JEPA target is never opened."""

    def __init__(
        self,
        *,
        rgb_path: Path,
        actions_path: Path,
        descriptors: Sequence[Mapping[str, Any]],
        padding_dim: int,
    ) -> None:
        import numpy as np

        if padding_dim != 157 or len(descriptors) != EXPECTED_VALIDATION_CLIPS:
            raise PhaseLockProbeError("registered validation input contract differs")
        self._rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
        self._actions = np.load(actions_path, mmap_mode="r", allow_pickle=False)
        if (
            tuple(self._rgb.shape) != (EXPECTED_VALIDATION_CLIPS, 13, 3, 180, 960)
            or str(self._rgb.dtype) != "float16"
            or tuple(self._actions.shape) != (EXPECTED_VALIDATION_CLIPS, 13, 5, 23)
            or str(self._actions.dtype) != "float32"
        ):
            raise PhaseLockProbeError("registered RGB/action array shape changed")
        self._descriptors = tuple(descriptors)
        self._padding_dim = padding_dim

    def __len__(self) -> int:
        return EXPECTED_VALIDATION_CLIPS

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch

        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        rgb = torch.from_numpy(np.array(self._rgb[index], copy=True)).float()
        actions = torch.from_numpy(
            np.array(self._actions[index], copy=True)
        )
        if actions.shape[-1] > self._padding_dim:
            raise PhaseLockProbeError("cached action dimension exceeds model padding")
        if actions.shape[-1] < self._padding_dim:
            padding = torch.zeros(
                *actions.shape[:-1],
                self._padding_dim - actions.shape[-1],
                dtype=actions.dtype,
            )
            actions = torch.cat((actions, padding), dim=-1)
        return {
            "rgb": rgb,
            "actions": actions,
            # ABCDataset's frozen morphology mapping in the training source.
            "morphology_index": torch.tensor(9, dtype=torch.long),
            "clip_index": torch.tensor(index, dtype=torch.long),
        }


def _move_batch(samples: Sequence[Mapping[str, Any]], device: Any) -> dict[str, Any]:
    import torch
    from torch.utils.data import default_collate

    batch = default_collate(samples)
    for key, value in tuple(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device=device, non_blocking=False)
    required = {"rgb", "actions", "morphology_index", "clip_index"}
    missing = sorted(required - batch.keys())
    if missing:
        raise PhaseLockProbeError(f"validation batch lacks {missing}")
    return batch


def _prepare_deployable_rollout(
    model: Any,
    history_rgb: Any,
    actions: Any,
    morphology_index: Any,
    sampling_ids: Any,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    if history_rgb.ndim != 5 or history_rgb.shape[1] != model.num_history_frames:
        raise PhaseLockProbeError("sampler accepts exactly five observed RGB frames")
    batch_size = int(history_rgb.shape[0])
    history_latents = model._encode_clip(history_rgb).to(history_rgb.dtype)
    history_frames = int(history_latents.shape[2])
    if history_frames != model.num_history_latent:
        raise PhaseLockProbeError("history-only VAE latent count changed")
    latent_frames = int(
        model.rgb_tokenizer.latent_temporal_len(
            model.num_history_frames + model.num_future_frames
        )
    )
    if latent_frames - history_frames < 2:
        raise PhaseLockProbeError(
            "PhaseLock probe requires at least two generated future latents"
        )
    video_shape = (
        batch_size,
        int(history_latents.shape[1]),
        latent_frames,
        int(history_latents.shape[3]),
        int(history_latents.shape[4]),
    )
    reference = history_latents.new_zeros(video_shape)
    reference[:, :, :history_frames] = history_latents
    _, z_control, _ = model._latent_actions(
        history_rgb,
        actions,
        morphology_index,
        latent_frames,
        history_frames,
    )
    z_control = z_control.to(history_rgb.dtype)
    context = model._build_context(
        batch_size, history_rgb.device, history_rgb.dtype
    )
    clip_fea = model._build_clip(
        batch_size, history_rgb.device, history_rgb.dtype
    )
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    initial_video = model._evaluation_noise(
        video_shape,
        device=history_rgb.device,
        dtype=history_rgb.dtype,
        base_seed=model.evaluation_noise_seed,
        sample_ids=sampling_ids,
        stream=0,
        rank=rank,
    )
    auxiliary_shape = (
        batch_size,
        int(model.forward_model.tf_token_adapter.tf_channels),
        *video_shape[2:],
    )
    initial_tf = model._evaluation_noise(
        auxiliary_shape,
        device=history_rgb.device,
        dtype=history_rgb.dtype,
        base_seed=model.evaluation_noise_seed,
        sample_ids=sampling_ids,
        stream=1,
        rank=rank,
    )
    if model._auxiliary_history_frames(history_frames) != 0:
        raise PhaseLockProbeError("VPM deployable auxiliary state must diffuse all bins")
    return {
        "initial_video": initial_video,
        "initial_tf": initial_tf,
        "reference": reference,
        "history_frames": history_frames,
        "z_control": z_control,
        "context": context,
        "clip_fea": clip_fea,
        "batch_size": batch_size,
        "dtype": history_rgb.dtype,
        "device": history_rgb.device,
    }


def _run_trajectory(
    model: Any,
    prepared: Mapping[str, Any],
    *,
    steps: int,
    motion_prior: Any | None = None,
) -> dict[str, Any]:
    import torch
    from robot_wm.modeling.dual_diffusion.flow import (
        euler_flow_step,
        pair_video_sigma_schedule,
    )

    if steps < 1:
        raise PhaseLockProbeError("trajectory steps must be positive")
    video_state = prepared["initial_video"].clone()
    tf_state = prepared["initial_tf"].clone()
    reference = prepared["reference"]
    history_frames = int(prepared["history_frames"])
    batch_size = int(prepared["batch_size"])
    dtype = prepared["dtype"]
    device = prepared["device"]
    model.sample_scheduler.set_timesteps(steps, device=device)
    timesteps = tuple(model.sample_scheduler.timesteps)
    sigmas = model.sample_scheduler.sigmas.to(
        device=device, dtype=torch.float32
    )[: steps + 1]
    schedule = pair_video_sigma_schedule(
        sigmas,
        mode=model.tf_schedule_mode,
        tf_lead_logit=model.tf_lead_logit,
    )
    if len(timesteps) != steps or schedule.num_steps != steps:
        raise PhaseLockProbeError("scheduler did not construct the declared NFE")
    applied_strengths: list[float] = []
    for index, timestep in enumerate(timesteps):
        video_sigma = schedule.video[index]
        next_video_sigma = schedule.video[index + 1]
        tf_sigma = schedule.time_frequency[index]
        next_tf_sigma = schedule.time_frequency[index + 1]
        tf_batch_sigma = tf_sigma.expand(batch_size).to(dtype=dtype)
        prediction = model.forward_model(
            video_state,
            timestep.expand(batch_size).to(device),
            prepared["z_control"],
            reference,
            prepared["context"],
            prepared["clip_fea"],
            noisy_tf=tf_state,
            conditioning_tf=tf_state,
            tf_sigma=tf_batch_sigma,
            condition_on_tf=False,
            condition_on_tf_clock=False,
        )
        if not hasattr(prediction, "video_velocity") or not hasattr(
            prediction, "tf_velocity"
        ):
            raise PhaseLockProbeError("VPM forward did not return both flow velocities")
        video_state = model.sample_scheduler.step(
            prediction.video_velocity.float(),
            timestep,
            video_state.float(),
        ).prev_sample.to(dtype)
        history_sigma = next_video_sigma.to(device=device, dtype=dtype)
        video_state[:, :, :history_frames] = (
            (1.0 - history_sigma) * reference[:, :, :history_frames]
            + history_sigma
            * prepared["initial_video"][:, :, :history_frames]
        )
        tf_state = euler_flow_step(
            tf_state.float(),
            prediction.tf_velocity.float(),
            tf_sigma,
            next_tf_sigma,
        ).to(dtype)
        strength = (
            0.0
            if motion_prior is None
            else linear_guidance_strength(index, steps)
        )
        if strength:
            video_state = apply_future_delta_guidance(
                video_state,
                motion_prior,
                history_frames,
                strength,
            )
        applied_strengths.append(strength)
    if not bool(torch.isfinite(video_state).all()) or not bool(
        torch.isfinite(tf_state).all()
    ):
        raise PhaseLockProbeError("trajectory produced a non-finite latent")
    return {
        "video": video_state,
        "tf": tf_state,
        "calls": steps,
        "guidance_strengths": applied_strengths,
    }


def run_endpoint(
    model: Any,
    prepared: Mapping[str, Any],
    endpoint: Endpoint,
    sampling_ids: Any,
) -> dict[str, Any]:
    """Run one endpoint independently from the common initial noise."""

    if endpoint.kind == "ordinary":
        final = _run_trajectory(model, prepared, steps=endpoint.full_steps)
        return {
            **final,
            "extracted_motion_prior": None,
            "applied_motion_prior": None,
            "motion_prior_donor_sampling_ids": None,
            "preliminary_final_video": None,
        }
    preliminary = _run_trajectory(model, prepared, steps=endpoint.few_steps)
    prior = extract_future_motion_prior(
        preliminary["video"], int(prepared["history_frames"])
    )
    if endpoint.prior_alignment == "aligned":
        applied_prior = prior
        donor_ids = sampling_ids.reshape(-1)
    elif endpoint.prior_alignment == "shuffled":
        applied_prior, donor_ids = shuffled_motion_prior(prior, sampling_ids)
    else:
        raise PhaseLockProbeError("guided endpoint has invalid prior alignment")
    final = _run_trajectory(
        model,
        prepared,
        steps=endpoint.full_steps,
        motion_prior=applied_prior,
    )
    calls = int(preliminary["calls"]) + int(final["calls"])
    if calls != endpoint.total_transformer_calls:
        raise PhaseLockProbeError("guided endpoint call decomposition differs")
    return {
        **final,
        "calls": calls,
        "extracted_motion_prior": prior,
        "applied_motion_prior": applied_prior,
        "motion_prior_donor_sampling_ids": donor_ids,
        "preliminary_final_video": preliminary["video"],
    }


def _per_sample_nmse(prediction: Any, target: Any, history: int) -> list[float]:
    import torch

    prediction = prediction[:, :, history:].double().flatten(1)
    target = target[:, :, history:].double().flatten(1)
    denominator = target.square().sum(dim=1)
    if bool((denominator <= 0).any()) or not bool(torch.isfinite(denominator).all()):
        raise PhaseLockProbeError("video target has invalid future energy")
    values = (prediction - target).square().sum(dim=1) / denominator
    if not bool(torch.isfinite(values).all()):
        raise PhaseLockProbeError("video NMSE is non-finite")
    return [float(value) for value in values.tolist()]


def _per_sample_future_delta_nmse(
    prediction: Any, target: Any, history: int
) -> list[float]:
    import torch

    pred_delta = extract_future_motion_prior(prediction, history).double().flatten(1)
    target_delta = extract_future_motion_prior(target, history).double().flatten(1)
    denominator = target_delta.square().sum(dim=1)
    if bool((denominator <= 0).any()) or not bool(torch.isfinite(denominator).all()):
        raise PhaseLockProbeError("video temporal target has invalid energy")
    values = (pred_delta - target_delta).square().sum(dim=1) / denominator
    return [float(value) for value in values.tolist()]


def _per_sample_decoded(
    prediction_uint8: Any,
    target_uint8: Any,
    history_last_uint8: Any,
) -> dict[str, list[float]]:
    import torch

    prediction = prediction_uint8.double() / 255.0
    target = target_uint8.double() / 255.0
    history = history_last_uint8.double() / 255.0
    reduce_dims = tuple(range(1, prediction.ndim))
    mse = (prediction - target).square().mean(dim=reduce_dims)
    pred_temporal = torch.diff(torch.cat((history, prediction), dim=2), dim=2)
    target_temporal = torch.diff(torch.cat((history, target), dim=2), dim=2)
    temporal = (pred_temporal - target_temporal).square().mean(dim=reduce_dims)
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))
    if not all(bool(torch.isfinite(value).all()) for value in (mse, temporal, psnr)):
        raise PhaseLockProbeError("decoded metric is non-finite")
    return {
        "decoded_mse_unit_range": [float(value) for value in mse.tolist()],
        "decoded_psnr_db": [float(value) for value in psnr.tolist()],
        "decoded_temporal_difference_mse_unit_range": [
            float(value) for value in temporal.tolist()
        ],
    }


def _scoring_targets(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Build evaluator-owned targets; this function is never called by sampler."""

    import torch

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=True
    ):
        video_clean = model._encode_clip(batch["rgb"]).to(batch["rgb"].dtype)
    raw = batch["rgb"].permute(0, 2, 1, 3, 4)
    ground_truth = (
        (raw[:, :, -FUTURE_RGB_FRAMES:].float().clamp(-1.0, 1.0) + 1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .cpu()
    )
    history_last = (
        (
            raw[:, :, -(FUTURE_RGB_FRAMES + 1) : -FUTURE_RGB_FRAMES]
            .float()
            .clamp(-1.0, 1.0)
            + 1.0
        )
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .cpu()
    )
    return {
        "video_clean": video_clean.detach().cpu().to(torch.float16),
        "ground_truth": ground_truth,
        "history_last": history_last,
        "rgb_hashes": _slice_hashes(batch["rgb"]),
        "actions_hashes": _slice_hashes(batch["actions"]),
    }


def _endpoint_rows(
    *,
    endpoint: Endpoint,
    result: Mapping[str, Any],
    prepared: Mapping[str, Any],
    decoded_uint8: Any,
    scoring: Mapping[str, Any],
    clip_indexes: Sequence[int],
    clip_ids: Sequence[str],
    sampling_ids: Any,
    observed_calls: int,
    registration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch

    video_final = result["video"].detach().cpu().to(torch.float16)
    initial_video = prepared["initial_video"].detach().cpu().to(torch.float16)
    initial_tf = prepared["initial_tf"].detach().cpu().to(torch.float16)
    video_clean = scoring["video_clean"]
    history_frames = int(prepared["history_frames"])
    if video_final.shape != video_clean.shape:
        raise PhaseLockProbeError("generated and scoring video latent shapes differ")
    if observed_calls != endpoint.total_transformer_calls or result["calls"] != observed_calls:
        raise PhaseLockProbeError(
            f"{endpoint.code} actual calls {observed_calls} != declared "
            f"{endpoint.total_transformer_calls}"
        )
    video_nmse = _per_sample_nmse(video_final, video_clean, history_frames)
    latent_temporal_nmse = _per_sample_future_delta_nmse(
        video_final, video_clean, history_frames
    )
    decoded_metrics = _per_sample_decoded(
        decoded_uint8, scoring["ground_truth"], scoring["history_last"]
    )
    batch_size = len(clip_indexes)
    if batch_size != int(video_final.shape[0]):
        raise PhaseLockProbeError("row clip count differs from tensor batch")
    sampling_values = [int(value) for value in sampling_ids.detach().cpu().tolist()]
    donor_tensor = result["motion_prior_donor_sampling_ids"]
    donor_values = (
        [None] * batch_size
        if donor_tensor is None
        else [int(value) for value in donor_tensor.detach().cpu().tolist()]
    )
    extracted = result["extracted_motion_prior"]
    applied = result["applied_motion_prior"]
    preliminary = result["preliminary_final_video"]
    none_hashes: list[str | None] = [None] * batch_size
    extracted_hashes = none_hashes if extracted is None else _slice_hashes(extracted)
    applied_hashes = none_hashes if applied is None else _slice_hashes(applied)
    preliminary_hashes = (
        none_hashes if preliminary is None else _slice_hashes(preliminary)
    )
    hashes = {
        "cached_rgb_input_sha256": scoring["rgb_hashes"],
        "cached_actions_input_sha256": scoring["actions_hashes"],
        "video_clean_sha256": _slice_hashes(video_clean),
        "raw_ground_truth_sha256": _slice_hashes(scoring["ground_truth"]),
        "raw_history_last_sha256": _slice_hashes(scoring["history_last"]),
        "video_initial_noise_sha256": _slice_hashes(initial_video),
        "tf_initial_noise_sha256": _slice_hashes(initial_tf),
        "preliminary_final_video_sha256": preliminary_hashes,
        "extracted_motion_prior_sha256": extracted_hashes,
        "applied_motion_prior_sha256": applied_hashes,
        "video_final_sha256": _slice_hashes(video_final),
        "decoded_final_sha256": _slice_hashes(decoded_uint8),
    }
    rows = []
    for offset, (clip_index, clip_id) in enumerate(zip(clip_indexes, clip_ids)):
        row_hashes = {key: values[offset] for key, values in hashes.items()}
        rows.append(
            identity_payload(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": KIND_ROW,
                    "registration_identity_sha256": registration[
                        "identity_sha256"
                    ],
                    "tool_git_commit": registration["tool_repository"][
                        "git_commit"
                    ],
                    "training_git_commit": TRAINING_COMMIT,
                    "study_identity_sha256": registration[
                        "controlled_study"
                    ]["study_identity_sha256"],
                    "arm_identity_sha256": registration["controlled_study"][
                        "arm_identity_sha256"
                    ],
                    "stage_identity_sha256": registration[
                        "controlled_study"
                    ]["stage_identity_sha256"],
                    "evaluation_split": "validation",
                    "protected_test_accessed": False,
                    "clip_index": int(clip_index),
                    "clip_id": str(clip_id),
                    "sampling_id": sampling_values[offset],
                    "sampling_namespace": "validation",
                    "endpoint": asdict(endpoint),
                    "clean_future_rgb_passed_to_sampler": False,
                    "clean_auxiliary_passed_to_sampler": False,
                    "clean_auxiliary_target_array_opened": False,
                    "online_teacher_call_count": 0,
                    "video_history_latent_frames": history_frames,
                    "future_video_latent_frames": int(
                        video_final.shape[2] - history_frames
                    ),
                    "actual_transformer_call_count": observed_calls,
                    "initial_noise_reused_between_preliminary_and_full": (
                        endpoint.kind == "phaselock"
                    ),
                    "motion_prior_donor_sampling_id": donor_values[offset],
                    "guidance_strengths": [
                        float(value) for value in result["guidance_strengths"]
                    ],
                    "metrics": {
                        "video_future_nmse": video_nmse[offset],
                        "video_future_temporal_delta_nmse": (
                            latent_temporal_nmse[offset]
                        ),
                        "decoded_mse_unit_range": decoded_metrics[
                            "decoded_mse_unit_range"
                        ][offset],
                        "decoded_psnr_db": decoded_metrics["decoded_psnr_db"][
                            offset
                        ],
                        "decoded_temporal_difference_mse_unit_range": (
                            decoded_metrics[
                                "decoded_temporal_difference_mse_unit_range"
                            ][offset]
                        ),
                    },
                    "tensor_sha256": row_hashes,
                }
            )
        )
    return rows


def _expected_rank_indexes(rank: int, world_size: int) -> list[int]:
    return list(range(rank, EXPECTED_VALIDATION_CLIPS, world_size))


def _validate_global_rows(
    rows: Sequence[Mapping[str, Any]],
    registration: Mapping[str, Any],
    descriptors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {
        (clip_index, endpoint.code)
        for clip_index in range(EXPECTED_VALIDATION_CLIPS)
        for endpoint in ENDPOINTS
    }
    observed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        if (
            not identity_valid(row)
            or row.get("kind") != KIND_ROW
            or row.get("registration_identity_sha256")
            != registration["identity_sha256"]
            or row.get("evaluation_split") != "validation"
            or row.get("protected_test_accessed") is not False
            or row.get("clean_future_rgb_passed_to_sampler") is not False
            or row.get("clean_auxiliary_passed_to_sampler") is not False
            or row.get("clean_auxiliary_target_array_opened") is not False
            or row.get("online_teacher_call_count") != 0
        ):
            raise PhaseLockProbeError("quality row violates deployable provenance")
        clip_index = row.get("clip_index")
        endpoint_payload = row.get("endpoint")
        if (
            isinstance(clip_index, bool)
            or not isinstance(clip_index, int)
            or not 0 <= clip_index < EXPECTED_VALIDATION_CLIPS
            or not isinstance(endpoint_payload, Mapping)
        ):
            raise PhaseLockProbeError("quality row has invalid clip/endpoint")
        code = str(endpoint_payload.get("code", ""))
        endpoint = ENDPOINT_BY_CODE.get(code)
        if endpoint is None or dict(endpoint_payload) != asdict(endpoint):
            raise PhaseLockProbeError("quality row endpoint differs from fixed grid")
        if (
            row.get("clip_id") != descriptors[clip_index].get("clip_id")
            or row.get("sampling_id")
            != VALIDATION_SAMPLE_ID_OFFSET + clip_index
            or row.get("actual_transformer_call_count")
            != endpoint.total_transformer_calls
        ):
            raise PhaseLockProbeError("quality row clip/call identity differs")
        key = (clip_index, code)
        if key in observed:
            raise PhaseLockProbeError(f"duplicate quality row: {key}")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise PhaseLockProbeError("quality row lacks metrics")
        for metric in (
            "video_future_nmse",
            "video_future_temporal_delta_nmse",
            "decoded_mse_unit_range",
            "decoded_psnr_db",
            "decoded_temporal_difference_mse_unit_range",
        ):
            value = metrics.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (metric != "decoded_psnr_db" and float(value) < 0)
            ):
                raise PhaseLockProbeError(f"quality row has invalid {metric}")
        observed[key] = row
    if set(observed) != expected:
        raise PhaseLockProbeError(
            f"quality inventory differs: missing={len(expected-set(observed))}, "
            f"extra={len(set(observed)-expected)}"
        )
    pairing_fields = (
        "cached_rgb_input_sha256",
        "cached_actions_input_sha256",
        "video_clean_sha256",
        "raw_ground_truth_sha256",
        "raw_history_last_sha256",
        "video_initial_noise_sha256",
        "tf_initial_noise_sha256",
    )
    for clip_index in range(EXPECTED_VALIDATION_CLIPS):
        clip_rows = [observed[(clip_index, endpoint.code)] for endpoint in ENDPOINTS]
        reference_hashes = clip_rows[0]["tensor_sha256"]
        for row in clip_rows[1:]:
            if any(
                row["tensor_sha256"].get(field) != reference_hashes.get(field)
                for field in pairing_fields
            ):
                raise PhaseLockProbeError(
                    f"paired input/noise changed for validation clip {clip_index}"
                )
    # Each aligned/shuffled pair reruns the same deterministic preliminary
    # trajectory. The extracted (pre-permutation) prior must therefore match.
    for candidate in ("k1_f2", "k1_f3", "k2_f2", "k2_f4"):
        aligned_code = f"phaselock_{candidate}_aligned"
        shuffled_code = f"phaselock_{candidate}_shuffled"
        for clip_index in range(EXPECTED_VALIDATION_CLIPS):
            aligned = observed[(clip_index, aligned_code)]
            shuffled = observed[(clip_index, shuffled_code)]
            endpoint = ENDPOINT_BY_CODE[aligned_code]
            for field in (
                "preliminary_final_video_sha256",
                "extracted_motion_prior_sha256",
            ):
                if aligned["tensor_sha256"].get(field) != shuffled[
                    "tensor_sha256"
                ].get(field):
                    raise PhaseLockProbeError(
                        f"preliminary prior changed across control: {candidate}"
                    )
            if aligned["tensor_sha256"].get(
                "applied_motion_prior_sha256"
            ) != aligned["tensor_sha256"].get(
                "extracted_motion_prior_sha256"
            ):
                raise PhaseLockProbeError(
                    f"aligned prior is not the sample's extracted prior: {candidate}"
                )
            if (
                aligned.get("motion_prior_donor_sampling_id")
                != aligned.get("sampling_id")
                or shuffled.get("motion_prior_donor_sampling_id")
                == shuffled.get("sampling_id")
            ):
                raise PhaseLockProbeError(
                    f"motion-prior donor identity is invalid: {candidate}"
                )
            expected_strengths = [
                linear_guidance_strength(index, endpoint.full_steps)
                for index in range(endpoint.full_steps)
            ]
            if (
                aligned.get("guidance_strengths") != expected_strengths
                or shuffled.get("guidance_strengths") != expected_strengths
            ):
                raise PhaseLockProbeError(
                    f"guidance schedule changed for {candidate}"
                )
            donor_sampling_id = shuffled["motion_prior_donor_sampling_id"]
            donor_clip_index = int(donor_sampling_id) - VALIDATION_SAMPLE_ID_OFFSET
            if not 0 <= donor_clip_index < EXPECTED_VALIDATION_CLIPS:
                raise PhaseLockProbeError(
                    f"shuffled donor is outside validation: {candidate}"
                )
            donor = observed[(donor_clip_index, aligned_code)]
            if shuffled["tensor_sha256"].get(
                "applied_motion_prior_sha256"
            ) != donor["tensor_sha256"].get(
                "extracted_motion_prior_sha256"
            ):
                raise PhaseLockProbeError(
                    f"shuffled prior hash does not match its donor: {candidate}"
                )
    for clip_index in range(EXPECTED_VALIDATION_CLIPS):
        for budget in (1, 3, 4, 6):
            ordinary = observed[(clip_index, f"ordinary_b{budget}")]
            hashes = ordinary["tensor_sha256"]
            if (
                ordinary.get("motion_prior_donor_sampling_id") is not None
                or ordinary.get("guidance_strengths") != [0.0] * budget
                or any(
                    hashes.get(field) is not None
                    for field in (
                        "preliminary_final_video_sha256",
                        "extracted_motion_prior_sha256",
                        "applied_motion_prior_sha256",
                    )
                )
            ):
                raise PhaseLockProbeError(
                    f"ordinary endpoint contains a motion prior: budget={budget}"
                )
    return {
        "record_count": len(rows),
        "clip_count": EXPECTED_VALIDATION_CLIPS,
        "endpoint_count": len(ENDPOINTS),
        "all_clip_endpoint_keys_present_once": True,
        "paired_inputs_and_initial_noise_exact": True,
        "aligned_shuffled_preliminary_prior_exact": True,
        "protected_test_accessed": False,
    }


def command_evaluate(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != EXPECTED_WORLD_SIZE:
        raise PhaseLockProbeError(
            f"probe requires {EXPECTED_WORLD_SIZE} ranks, found {world_size}"
        )
    if args.batch_size_per_rank != EXPECTED_BATCH_SIZE_PER_RANK:
        raise PhaseLockProbeError(
            f"probe requires batch size {EXPECTED_BATCH_SIZE_PER_RANK} per rank"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if "B200" not in torch.cuda.get_device_properties(device).name.upper():
        raise PhaseLockProbeError("probe requires B200 GPUs")
    registration, validated = _validate_registration(args.registration, rank=rank)
    output_root = Path(registration["output_root"])
    expected_output = output_root / "evaluation"
    if args.output_dir.expanduser().absolute() != expected_output:
        raise PhaseLockProbeError(f"evaluation output must be exactly {expected_output}")
    assigned = _expected_rank_indexes(rank, world_size)
    if len(assigned) % EXPECTED_BATCH_SIZE_PER_RANK:
        raise PhaseLockProbeError("rank shard is not divisible by the fixed batch size")
    if rank == 0:
        if expected_output.exists():
            raise PhaseLockProbeError(f"fresh evaluation output exists: {expected_output}")
        expected_output.mkdir(parents=True, mode=0o700)
    dist.barrier()
    model, dataset, _config = _load_model_and_dataset(
        registration, validated, device=device
    )
    rows: list[dict[str, Any]] = []
    actual_calls_total = 0
    hook_calls = 0

    def count_transformer_calls(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal hook_calls
        hook_calls += 1

    hook = model.forward_model.register_forward_hook(count_transformer_calls)
    try:
        for start in range(0, len(assigned), EXPECTED_BATCH_SIZE_PER_RANK):
            clip_indexes = assigned[start : start + EXPECTED_BATCH_SIZE_PER_RANK]
            samples = [dataset[index] for index in clip_indexes]
            observed_indexes = [int(sample["clip_index"].item()) for sample in samples]
            if observed_indexes != clip_indexes:
                raise PhaseLockProbeError(
                    f"validation dataset substituted clips: {observed_indexes}"
                )
            batch = _move_batch(samples, device)
            scoring = _scoring_targets(model, batch)
            # Make the separation explicit: the sampler receives a five-frame
            # view and never the evaluator-owned full clip or auxiliary target.
            history_rgb = batch["rgb"][:, : model.num_history_frames].clone()
            sampling_ids = batch["clip_index"] + VALIDATION_SAMPLE_ID_OFFSET
            clip_ids = [
                str(validated["descriptors"][index]["clip_id"])
                for index in clip_indexes
            ]
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                prepared = _prepare_deployable_rollout(
                    model,
                    history_rgb,
                    batch["actions"],
                    batch["morphology_index"],
                    sampling_ids,
                )
                for endpoint in ENDPOINTS:
                    hook_calls = 0
                    result = run_endpoint(
                        model, prepared, endpoint, sampling_ids
                    )
                    observed_calls = hook_calls
                    decoded = model.rgb_tokenizer.decode_temporal(
                        result["video"],
                        out_hw=(history_rgb.shape[-2], history_rgb.shape[-1]),
                    )
                    decoded_uint8 = (
                        (
                            decoded[:, :, -FUTURE_RGB_FRAMES:]
                            .float()
                            .clamp(-1.0, 1.0)
                            + 1.0
                        )
                        .mul(127.5)
                        .round()
                        .to(torch.uint8)
                        .cpu()
                    )
                    rows.extend(
                        _endpoint_rows(
                            endpoint=endpoint,
                            result=result,
                            prepared=prepared,
                            decoded_uint8=decoded_uint8,
                            scoring=scoring,
                            clip_indexes=clip_indexes,
                            clip_ids=clip_ids,
                            sampling_ids=sampling_ids,
                            observed_calls=observed_calls,
                            registration=registration,
                        )
                    )
                    actual_calls_total += observed_calls
                    del result, decoded, decoded_uint8
            del scoring, prepared, batch, samples
    finally:
        hook.remove()
    expected_rows = len(assigned) * len(ENDPOINTS)
    expected_calls = (
        len(assigned)
        // EXPECTED_BATCH_SIZE_PER_RANK
        * sum(endpoint.total_transformer_calls for endpoint in ENDPOINTS)
    )
    if len(rows) != expected_rows or actual_calls_total != expected_calls:
        raise PhaseLockProbeError(
            f"rank {rank} output/calls differ: rows={len(rows)}/{expected_rows}, "
            f"calls={actual_calls_total}/{expected_calls}"
        )
    rows_path = expected_output / f"rank_{rank:03d}.jsonl"
    rows_bytes = b"".join(_canonical_json(row) + b"\n" for row in rows)
    _exclusive_bytes(rows_path, rows_bytes)
    rank_manifest = identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND_RANK,
            "created_at_utc": _now(),
            "registration_identity_sha256": registration["identity_sha256"],
            "rank": rank,
            "world_size": world_size,
            "batch_size_per_rank": EXPECTED_BATCH_SIZE_PER_RANK,
            "assigned_clip_indexes": assigned,
            "endpoint_grid": [asdict(endpoint) for endpoint in ENDPOINTS],
            "actual_transformer_call_count": actual_calls_total,
            "rows": {
                "path": str(rows_path),
                "bytes": len(rows_bytes),
                "sha256": hashlib.sha256(rows_bytes).hexdigest(),
                "count": len(rows),
            },
            "evaluation_split": "validation",
            "protected_test_accessed": False,
        }
    )
    _exclusive_json(expected_output / f"rank_{rank:03d}.json", rank_manifest)
    dist.barrier()
    if rank == 0:
        global_rows: list[dict[str, Any]] = []
        rank_manifests = []
        for source_rank in range(world_size):
            manifest_path = expected_output / f"rank_{source_rank:03d}.json"
            manifest = _read_json(manifest_path, f"rank {source_rank} manifest")
            if (
                not identity_valid(manifest)
                or manifest.get("kind") != KIND_RANK
                or manifest.get("rank") != source_rank
                or manifest.get("registration_identity_sha256")
                != registration["identity_sha256"]
                or manifest.get("protected_test_accessed") is not False
            ):
                raise PhaseLockProbeError(f"rank {source_rank} manifest is invalid")
            source_rows = Path(manifest["rows"]["path"])
            content = source_rows.read_bytes()
            if (
                len(content) != manifest["rows"]["bytes"]
                or hashlib.sha256(content).hexdigest()
                != manifest["rows"]["sha256"]
            ):
                raise PhaseLockProbeError(f"rank {source_rank} rows changed")
            for line in content.splitlines():
                global_rows.append(json.loads(line))
            rank_manifests.append(
                {
                    "path": str(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                    "sha256": _sha256(manifest_path),
                    "identity_sha256": manifest["identity_sha256"],
                }
            )
        checks = _validate_global_rows(
            global_rows, registration, validated["descriptors"]
        )
        inventory = identity_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": KIND_INVENTORY,
                "created_at_utc": _now(),
                "registration": {
                    "path": str(args.registration.resolve()),
                    "sha256": _sha256(args.registration.resolve()),
                    "identity_sha256": registration["identity_sha256"],
                },
                "evaluation_split": "validation",
                "protected_test_accessed": False,
                "endpoint_grid": [asdict(endpoint) for endpoint in ENDPOINTS],
                "rank_manifests": rank_manifests,
                "validation_checks": checks,
            }
        )
        _exclusive_json(expected_output / "inventory.json", inventory)
    dist.barrier()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register", help="freeze validation probe")
    register.add_argument("--tool-repo", type=Path, required=True)
    register.add_argument("--tool-commit", required=True)
    register.add_argument("--model-repo", type=Path, required=True)
    register.add_argument("--study-root", type=Path, required=True)
    register.add_argument("--output-root", type=Path, required=True)
    register.set_defaults(handler=command_register)
    evaluate = subparsers.add_parser("evaluate", help="run fixed validation grid")
    evaluate.add_argument("--registration", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument(
        "--batch-size-per-rank",
        type=int,
        default=EXPECTED_BATCH_SIZE_PER_RANK,
    )
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PhaseLockProbeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
