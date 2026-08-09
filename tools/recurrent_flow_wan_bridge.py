#!/usr/bin/env python3
"""Fail-closed bridge from the sealed recurrent-delta predictor to Wan flow.

The bridge deliberately reuses the raw-physics-flow representation and
renderer, but not its semantic label.  It produces four immutable compact
fields per split:

* ``recurrent_aligned``: recurrent-delta poses from observed states 0..4 and
  the registered candidate actions;
* ``recurrent_shuffled``: a complete field from an episode-disjoint donor in
  the already-frozen motion stratum;
* ``recurrent_wrong_time``: aligned transitions shifted +1 without wrapping;
* ``recurrent_hold``: the observed frame-4 pose held for all future steps.

``off`` is an exact-zero runtime tensor.  No command in this module submits a
job, opens future RGB, or reads measured state after the history boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import corrected_renderer_attribution as renderer_base  # noqa: E402
from tools import physics_flow_stage1 as raw_stage  # noqa: E402
from tools import recurrent_delta_spatial_confirmation as confirmation  # noqa: E402
from tools import trajectory_consistent_renderer_gate as trajectory  # noqa: E402
from tools.abc_d405_nominal_geometry_probe import (  # noqa: E402
    CAMERA_TYPE,
    _decode_top_calibration,
)


SCHEMA_VERSION = 1
CACHE_SCHEMA = "recurrent-physics-flow-cache-v1"
CACHE_KIND = "recurrent_physics_flow_cache_registration"
STUDY_KIND = "recurrent_physics_flow_wan_stage1_registration"

CONFIRMATION_SOURCE_COMMIT = "92cc9584298d2a395f3a79152768c9d0b88b3864"
CONFIRMATION_REGISTRATION_IDENTITY = (
    "db1c29e5a89c35b8b1b8c9a4ba8d1be2ef796c3793961fcd4204db46d363d42d"
)
CONFIRMATION_PREPARATION_IDENTITY = (
    "ea0c1383da10cb654aea3dd3d6682312fc00d1f68ace13b38f65587e6990397e"
)
CONFIRMATION_CLOSURE_IDENTITY = (
    "0b8f50bf3ac84e9932870b3c7cf430618bc49006f1306cf869328a5f65ebdcfc"
)
CONFIRMATION_ANALYSIS_IDENTITY = (
    "2426a1546a43403fc09f7d8bb60475526e4d342586727c880dd005aad99010f8"
)
CONFIRMATION_COMPLETION_IDENTITY = (
    "a94afb1e39ccee4b80298182f287431409fbce47ae260c94d4324fa23081e5ff"
)
FROZEN_PREDICTOR_SHA256 = (
    "8602fda20b3f0c8365c69efcc48e363a8e8f93666725e1af8665a18d03e80688"
)
POSITIVE_DECISION = "GO_RECURRENT_DELTA_WAN_SCREEN"
STRICT_AUDITOR_SOURCE_COMMIT = "ff3244912bfe62dfa0208a4a38cdbc1c1053f40f"
STRICT_AUDIT_IDENTITY = (
    "d7afdf0bddcd8dee32351b2356137503cf5485dcdaa5f8e0b699a63d1dc86b69"
)
STRICT_AUDIT_FILE_SHA256 = (
    "8e0475d40aabb7ed3d200ef900cf012e58f0096b27fa46642384869b92f545ea"
)
BASE_RAW_CACHE_REGISTRATION_IDENTITY = (
    "6a6653f14f8f01cb56e749f4430f6bdfbae86da758ed4e13363b4ef1b4be47c9"
)
BASE_RAW_CACHE_REGISTRATION_SHA256 = (
    "a1af8831819455207b5b9a47fe33b2da5a50cbd9d21c24b96646e553a4f8c0a8"
)
BASE_RAW_CACHE_REGISTRATION_PATH = Path(
    "/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/"
    "lacwm_train/artifacts/dual_video_diffusion/raw_physics_flow_cache/"
    "raw-physics-flow-cache-20260808-19717d3-v7/cache_registration.json"
)

CACHE_SOURCES = (
    "recurrent_aligned",
    "recurrent_shuffled",
    "recurrent_wrong_time",
    "recurrent_hold",
)
RUNTIME_SOURCES = (
    "recurrent_aligned",
    "off",
    "recurrent_shuffled",
    "recurrent_wrong_time",
    "recurrent_hold",
)
NFE_GRID = (1, 2, 4)
NOISE_SEEDS = (0, 1, 2, 3)
HISTORY_FRAMES = 5

# Only these arrays are deployable predictor state.  In particular, the source
# archive's target_measured_trajectory, score contexts, score indexes, and
# stored predictions are forbidden to the cache builder and Wan runtime.
INFERENCE_PREDICTOR_KEYS = (
    "recurrent_alpha",
    "recurrent_gain",
    "history_feature_mean",
    "history_feature_std",
    "history_feature_active",
    "history_pca_mean",
    "history_pca_components",
    "history_explained_variance_ratio",
    "action_feature_mean",
    "action_feature_std",
    "action_feature_active",
    "action_pca_mean",
    "action_pca_components",
    "action_explained_variance_ratio",
    "recurrent_input_mean",
    "recurrent_input_std",
    "recurrent_target_mean",
    "recurrent_target_std",
    "recurrent_coef",
    "recurrent_intercept",
)
FORBIDDEN_SOURCE_PREDICTOR_KEYS = (
    "fit_indices",
    "score_indices",
    "donor_indices",
    "donor_positions",
    "score_history_context",
    "score_action_context",
    "aligned_absolute_residual",
    "aligned_recurrent_trajectory",
    "shuffled_recurrent_trajectory",
    "target_measured_trajectory",
)


class RecurrentFlowBridgeError(RuntimeError):
    """An immutable lineage, causal-access, or endpoint contract changed."""


@dataclass(frozen=True)
class Endpoint:
    code: str
    arm: str
    nfe: int
    condition_source: str
    primary: bool


def endpoint_grid() -> tuple[Endpoint, ...]:
    """Return the frozen seven-endpoint NFE1 screen.

    NFE2/4 are intentionally absent until this NFE1 attribution screen passes.
    Every endpoint therefore executes exactly one Wan transformer call.
    """

    return (
        Endpoint("parent_vpm_off_nfe_1", "PARENT-VPM", 1, "off", True),
        Endpoint("matched_off_nfe_1", "FLOW-OFF", 1, "off", True),
        *(
            Endpoint(
                f"recurrent_checkpoint_{source}_nfe_1",
                "RECURRENT-FLOW",
                1,
                source,
                True,
            )
            for source in RUNTIME_SOURCES
        ),
    )


def _resolve_record(root: Path, record: Mapping[str, Any], fallback: str) -> Path:
    value = Path(str(record.get("path", "")))
    candidates = (value, root / fallback, root / value.name)
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            resolved = path.resolve(strict=True)
            if raw_stage.sha256_file(resolved) == record.get("sha256"):
                return resolved
    raise RecurrentFlowBridgeError(f"immutable artifact differs: {fallback}")


def _identity_document(root: Path, name: str, expected: str, kind: str) -> dict[str, Any]:
    path = root / name
    payload = raw_stage.read_json(path, name)
    if (
        not raw_stage.identity_valid(payload)
        or payload.get("identity_sha256") != expected
        or payload.get("kind") != kind
    ):
        raise RecurrentFlowBridgeError(f"sealed confirmation differs: {name}")
    return payload


def validate_strict_external_audit(path: Path, confirmation_root: Path) -> dict[str, Any]:
    """Require the independent post-run reconstruction before handoff."""

    path = path.resolve(strict=True)
    root = confirmation_root.resolve(strict=True)
    if path.is_symlink() or raw_stage.sha256_file(path) != STRICT_AUDIT_FILE_SHA256:
        raise RecurrentFlowBridgeError("strict external audit file hash differs")
    payload = raw_stage.read_json(path, "strict recurrent confirmation audit")
    canonical = payload.get("canonical_identities", {})
    decisions = payload.get("decisions", {})
    causal = payload.get("causal_reconstruction", {})
    built_in = payload.get("built_in_audit", {})
    if (
        not raw_stage.identity_valid(payload)
        or payload.get("kind") != "recurrent_delta_strict_external_postrun_audit"
        or payload.get("identity_sha256") != STRICT_AUDIT_IDENTITY
        or payload.get("status") != "strict_audit_passed"
        or payload.get("expected_source_commit") != CONFIRMATION_SOURCE_COMMIT
        or payload.get("auditor", {}).get("git_commit") != STRICT_AUDITOR_SOURCE_COMMIT
        or Path(str(payload.get("canonical_run_root", ""))).resolve(strict=True) != root
        or payload.get("canonical_root_mutated") is not False
        or payload.get("report_is_outside_canonical_root") is not True
        or canonical.get("registration.json") != CONFIRMATION_REGISTRATION_IDENTITY
        or canonical.get("preparation.json") != CONFIRMATION_PREPARATION_IDENTITY
        or canonical.get("causal_trajectory_closure.json") != CONFIRMATION_CLOSURE_IDENTITY
        or canonical.get("analysis.json") != CONFIRMATION_ANALYSIS_IDENTITY
        or canonical.get("run_complete.json") != CONFIRMATION_COMPLETION_IDENTITY
        or decisions.get("all_equal") is not True
        or {decisions.get(name) for name in ("analysis", "built_in_audit", "recomputed", "run_complete")}
        != {POSITIVE_DECISION}
        or causal.get("arrays_compared") != 25
        or causal.get("max_abs_error") != 0.0
        or built_in.get("artifact_hashes_verified") != 168
        or built_in.get("causal_replay_max_abs_error") != 0.0
        or built_in.get("status") != "audit_passed"
        or int(payload.get("explicit_false_access_flags", 0)) != 5513
    ):
        raise RecurrentFlowBridgeError("strict external recurrent audit differs")
    return raw_stage.file_record(path) | {
        "identity_sha256": STRICT_AUDIT_IDENTITY,
        "auditor_source_commit": STRICT_AUDITOR_SOURCE_COMMIT,
        "status": "strict_audit_passed",
    }


def validate_confirmation(root: Path, strict_audit_path: Path) -> dict[str, Any]:
    """Validate the exact passed renderer confirmation and frozen predictor."""

    root = root.resolve(strict=True)
    registration = _identity_document(
        root,
        "registration.json",
        CONFIRMATION_REGISTRATION_IDENTITY,
        "recurrent_delta_spatial_confirmation_registration",
    )
    preparation = _identity_document(
        root,
        "preparation.json",
        CONFIRMATION_PREPARATION_IDENTITY,
        "recurrent_delta_spatial_confirmation_preparation",
    )
    closure = _identity_document(
        root,
        "causal_trajectory_closure.json",
        CONFIRMATION_CLOSURE_IDENTITY,
        "recurrent_delta_causal_trajectory_closure",
    )
    analysis = _identity_document(
        root,
        "analysis.json",
        CONFIRMATION_ANALYSIS_IDENTITY,
        "recurrent_delta_spatial_confirmation_analysis",
    )
    complete = _identity_document(
        root,
        "run_complete.json",
        CONFIRMATION_COMPLETION_IDENTITY,
        "recurrent_delta_spatial_confirmation_complete",
    )
    gates = analysis.get("gates", {})
    if (
        registration.get("source", {}).get("registered_git_commit")
        != CONFIRMATION_SOURCE_COMMIT
        or preparation.get("registration_identity_sha256")
        != CONFIRMATION_REGISTRATION_IDENTITY
        or closure.get("preparation_identity_sha256")
        != CONFIRMATION_PREPARATION_IDENTITY
        or analysis.get("registration_identity_sha256")
        != CONFIRMATION_REGISTRATION_IDENTITY
        or complete.get("analysis_identity_sha256")
        != CONFIRMATION_ANALYSIS_IDENTITY
        or complete.get("registration_identity_sha256")
        != CONFIRMATION_REGISTRATION_IDENTITY
        or analysis.get("decision") != POSITIVE_DECISION
        or complete.get("decision") != POSITIVE_DECISION
        or complete.get("status") != "completed"
        or gates.get("recurrent_delta_spatial_confirmation_pass") is not True
        or not all(gates.get("flow_superiority", {}).values())
        or not all(gates.get("spatial_noninferiority", {}).values())
        or complete.get("protected_test_accessed") is not False
        or complete.get("validation_accessed") is not False
    ):
        raise RecurrentFlowBridgeError("recurrent confirmation did not pass exact GO contract")
    predictor_record = registration.get("prior", {}).get("artifacts", {}).get(
        "frozen_model", {}
    )
    predictor = _resolve_record(
        root, predictor_record, "prior/model_state_and_predictions.npz"
    )
    if raw_stage.sha256_file(predictor) != FROZEN_PREDICTOR_SHA256:
        raise RecurrentFlowBridgeError("frozen recurrent predictor differs")
    strict_audit = validate_strict_external_audit(strict_audit_path, root)
    return {
        "root": str(root),
        "registration": raw_stage.file_record(root / "registration.json"),
        "preparation": raw_stage.file_record(root / "preparation.json"),
        "closure": raw_stage.file_record(root / "causal_trajectory_closure.json"),
        "analysis": raw_stage.file_record(root / "analysis.json"),
        "completion": raw_stage.file_record(root / "run_complete.json"),
        "predictor": raw_stage.file_record(predictor),
        "strict_external_audit": strict_audit,
        "decision": POSITIVE_DECISION,
        "protected_test_accessed": False,
    }


def _strict_npz_members(
    path: Path,
    keys: Sequence[str],
    *,
    require_exact_inventory: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read only explicitly named NPY members from an NPZ archive.

    This does not call ``np.load`` on the container.  Central-directory
    metadata are inspected, then only whitelisted member streams are opened.
    """

    expected = tuple(keys)
    expected_members = {f"{name}.npy" for name in expected}
    arrays: dict[str, np.ndarray] = {}
    opened = []
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or not expected_members.issubset(names):
            raise RecurrentFlowBridgeError("predictor NPZ member inventory differs")
        if require_exact_inventory and set(names) != expected_members:
            raise RecurrentFlowBridgeError("inference predictor contains non-whitelisted keys")
        if any(
            name.startswith("/") or ".." in Path(name).parts or not name.endswith(".npy")
            for name in names
        ):
            raise RecurrentFlowBridgeError("unsafe predictor NPZ member name")
        by_name = {info.filename: info for info in infos}
        for key in expected:
            member = f"{key}.npy"
            info = by_name[member]
            with archive.open(info, "r") as handle:
                value = np.lib.format.read_array(handle, allow_pickle=False)
            if value.dtype.hasobject or not (
                value.dtype == np.bool_ or np.issubdtype(value.dtype, np.number)
            ):
                raise RecurrentFlowBridgeError(f"unsafe predictor dtype: {key}")
            if value.dtype != np.bool_ and not np.isfinite(value).all():
                raise RecurrentFlowBridgeError(f"nonfinite predictor tensor: {key}")
            arrays[key] = value
            opened.append(
                {
                    "member": member,
                    "crc32": int(info.CRC),
                    "compressed_bytes": int(info.compress_size),
                    "uncompressed_bytes": int(info.file_size),
                    "array_shape": list(value.shape),
                    "array_dtype": str(value.dtype),
                    "array_bytes_sha256": hashlib.sha256(
                        np.ascontiguousarray(value).view(np.uint8)
                    ).hexdigest(),
                }
            )
    return arrays, {
        "archive": raw_stage.file_record(path),
        "central_directory_members": names,
        "opened_members": opened,
        "opened_member_names": [item["member"] for item in opened],
        "member_streams_not_opened_by_extractor": sorted(set(names) - expected_members),
        "np_load_container_used": False,
        "allow_pickle": False,
    }


def extract_inference_predictor(source: Path, destination: Path) -> dict[str, Any]:
    """Derive target-free runtime weights from the sealed privileged archive."""

    source = source.resolve(strict=True)
    if raw_stage.sha256_file(source) != FROZEN_PREDICTOR_SHA256:
        raise RecurrentFlowBridgeError("privileged predictor source hash differs")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    arrays, access = _strict_npz_members(
        source, INFERENCE_PREDICTOR_KEYS, require_exact_inventory=False
    )
    source_inventory = set(access["central_directory_members"])
    missing_forbidden = {
        f"{name}.npy" for name in FORBIDDEN_SOURCE_PREDICTOR_KEYS
    } - source_inventory
    if missing_forbidden:
        raise RecurrentFlowBridgeError(
            f"sealed source forbidden-key inventory changed: {sorted(missing_forbidden)}"
        )
    opened = set(access["opened_member_names"])
    forbidden = {f"{name}.npy" for name in FORBIDDEN_SOURCE_PREDICTOR_KEYS}
    if opened & forbidden:
        raise RecurrentFlowBridgeError("privileged target/score member was deserialized")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed NPZ makes subsequent per-member access simple and auditable;
    # it contains only deployable weights, never target or score arrays.
    np.savez(destination, **arrays)
    replay, replay_access = _strict_npz_members(
        destination, INFERENCE_PREDICTOR_KEYS, require_exact_inventory=True
    )
    if any(not np.array_equal(arrays[name], replay[name]) for name in arrays):
        raise RecurrentFlowBridgeError("inference-only predictor replay differs")
    return raw_stage.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_delta_inference_predictor_extraction",
            "status": "target_free_inference_weights_only",
            "source_archive": raw_stage.file_record(source),
            "source_archive_bytes_hashed": True,
            "opaque_hash_traversed_all_source_archive_bytes": True,
            "output_archive": raw_stage.file_record(destination),
            "whitelisted_keys": list(INFERENCE_PREDICTOR_KEYS),
            "forbidden_source_keys": list(FORBIDDEN_SOURCE_PREDICTOR_KEYS),
            "source_access": access,
            "output_access": replay_access,
            "forbidden_members_deserialized": [],
            "target_measured_trajectory_deserialized": False,
            "stored_score_context_or_outcome_deserialized": False,
            "predictor_refit_or_tuning": False,
            "future_rgb_opened": False,
            "future_measured_state_opened": False,
            "protected_test_accessed": False,
        }
    )


def _load_predictor(path: Path) -> tuple[Any, Any, Any, float]:
    payload, access = _strict_npz_members(
        path, INFERENCE_PREDICTOR_KEYS, require_exact_inventory=True
    )
    if access["member_streams_not_opened_by_extractor"]:
        raise RecurrentFlowBridgeError("runtime predictor has unexpected members")
    history_projection = confirmation._load_projection(payload, "history")
    action_projection = confirmation._load_projection(payload, "action")
    recurrent_model = confirmation._load_ridge(payload, "recurrent")
    alpha = float(payload["recurrent_alpha"])
    gain = float(payload["recurrent_gain"])
    if alpha != confirmation.PRIOR_RECURRENT_ALPHA or gain != confirmation.PRIOR_RECURRENT_GAIN:
        raise RecurrentFlowBridgeError("frozen recurrent hyperparameters differ")
    return history_projection, action_projection, recurrent_model, gain


def build_predictor_features(
    measured_history: np.ndarray, raw_actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the exact causal feature tensors used by the sealed model."""

    measured = np.asarray(measured_history, dtype=np.float32)
    actions = np.asarray(raw_actions, dtype=np.float32)
    if measured.ndim != 3 or measured.shape[1:] != (HISTORY_FRAMES, raw_stage.ACTION_DIM):
        raise RecurrentFlowBridgeError(f"measured-history geometry differs: {measured.shape}")
    if actions.shape != (
        len(measured),
        raw_stage.SAMPLE_SIZE,
        raw_stage.CHUNK_SIZE,
        raw_stage.ACTION_DIM,
    ):
        raise RecurrentFlowBridgeError(f"candidate-action geometry differs: {actions.shape}")
    observed_residual = measured[:, 1:] - actions[:, :4, -1]
    history = np.concatenate(
        (
            measured.reshape(len(measured), -1),
            actions[:, :4].reshape(len(measured), -1),
            observed_residual.reshape(len(measured), -1),
        ),
        axis=1,
    ).astype(np.float32)
    future = actions[:, 4:12].astype(np.float32)
    q4 = measured[:, 4].astype(np.float32)
    if history.shape[1] != 406 or future.shape[1:] != (8, 5, 14):
        raise RecurrentFlowBridgeError("sealed predictor input width changed")
    return history, future, q4


def predict_recurrent_trajectory(
    measured_history: np.ndarray,
    raw_actions: np.ndarray,
    predictor_path: Path,
) -> np.ndarray:
    """Roll out the frozen recurrent predictor; no future target is accepted."""

    history, future, q4 = build_predictor_features(measured_history, raw_actions)
    history_projection, action_projection, model, gain = _load_predictor(predictor_path)
    result = trajectory.rollout_recurrent(
        model,
        history_projection.transform(history),
        action_projection.transform(future),
        future,
        q4,
        gain,
    )
    result = trajectory._render_pose(result)
    if result.shape != (len(history), 9, raw_stage.ACTION_DIM) or not np.isfinite(result).all():
        raise RecurrentFlowBridgeError("recurrent trajectory is invalid")
    return result


def derive_interventions(
    aligned: np.ndarray,
    donor_indexes: Sequence[int | None],
    eligible: Sequence[bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Derive complete-field shuffle and nonwrapping wrong-time controls."""

    value = np.asarray(aligned)
    raw_stage.validate_compact_flow(value, count=len(donor_indexes))
    if len(eligible) != len(value):
        raise RecurrentFlowBridgeError("eligibility length differs")
    shuffled = np.zeros_like(value)
    for index, (donor, is_eligible) in enumerate(zip(donor_indexes, eligible, strict=True)):
        if is_eligible:
            if donor is None or int(donor) == index or not eligible[int(donor)]:
                raise RecurrentFlowBridgeError("invalid episode-disjoint recurrent donor")
            shuffled[index] = value[int(donor)]
    wrong_time = raw_stage.nonwrapping_timeshift(value)
    return shuffled, wrong_time


def _source_records() -> dict[str, Any]:
    return {
        name: raw_stage.file_record(REPO_ROOT / relative)
        for name, relative in {
            "bridge": "tools/recurrent_flow_wan_bridge.py",
            "raw_stage": "tools/physics_flow_stage1.py",
            "confirmation": "tools/recurrent_delta_spatial_confirmation.py",
            "trajectory": "tools/trajectory_consistent_renderer_gate.py",
            "model": "projects/latent_action_models/lam/physics_flow_model.py",
            "model_config": "projects/latent_action_models/configs/models/physics_flow_model.yaml",
            "dataset": "robot_wm/datasets/abc/recurrent_physics_flow_dataset.py",
            "protocol": "docs/experiments/RECURRENT_FLOW_WAN_SCREEN_PROTOCOL.md",
        }.items()
    }


def _raw_input_revalidation(base_registration: Mapping[str, Any]) -> dict[str, Any]:
    """Rehash every bridge-used manifest/RGB/action input and bind state layouts."""

    splits = {}
    for split, input_key in (("train", "train"), ("val", "validation")):
        registered = base_registration["inputs"][input_key]
        manifest_path = Path(registered["manifest"]["path"]).resolve(strict=True)
        cache_metadata_path = Path(registered["cache"]["metadata"]["path"]).resolve(
            strict=True
        )
        manifest = raw_stage.file_record(manifest_path)
        cache_metadata = raw_stage.file_record(cache_metadata_path)
        if manifest != registered["manifest"] or cache_metadata != registered["cache"]["metadata"]:
            raise RecurrentFlowBridgeError(f"{split} manifest/cache metadata changed")
        arrays = {}
        for name in ("rgb", "actions"):
            frozen = registered["cache"]["arrays"][name]
            path = Path(frozen["path"]).resolve(strict=True)
            current = raw_stage.file_record(path)
            if current.get("sha256") != frozen.get("sha256"):
                raise RecurrentFlowBridgeError(f"{split} {name} cache changed")
            arrays[name] = current
        rows = raw_stage.read_jsonl(manifest_path)
        descriptors = base_registration["splits"][split]["descriptors"]
        if len(rows) != len(descriptors):
            raise RecurrentFlowBridgeError(f"{split} manifest/descriptor count differs")
        action_values = np.load(
            registered["cache"]["arrays"]["actions"]["path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        eligibility = [bool(item["d405_eligible"]) for item in descriptors]
        donors, strata, motion = raw_stage.deterministic_donors(
            rows, action_values, eligibility
        )
        descriptor_bindings = []
        for index, (row, descriptor) in enumerate(zip(rows, descriptors, strict=True)):
            frozen_fields = {
                "clip_index": index,
                "clip_id": row["clip_id"],
                "episode_dir": row["episode_dir"],
                "frame_indices": row["frame_indices"],
                "start": row["start"],
            }
            if any(descriptor.get(name) != value for name, value in frozen_fields.items()):
                raise RecurrentFlowBridgeError(f"{split} row/descriptor binding differs")
            expected_donor_clip = (
                rows[int(donors[index])]["clip_id"]
                if donors[index] is not None
                else None
            )
            if (
                descriptor.get("episode_shuffled_donor_index") != donors[index]
                or descriptor.get("episode_shuffled_donor_clip_id")
                != expected_donor_clip
                or descriptor.get("motion_stratum") != strata[index]
                or descriptor.get("planned_motion_rms") != motion[index]
            ):
                raise RecurrentFlowBridgeError(
                    f"{split} deterministic donor/stratum replay differs"
                )
            descriptor_bindings.append(
                {
                    **frozen_fields,
                    "action_row_sha256": descriptor["action_row_sha256"],
                    "d405_eligible": descriptor["d405_eligible"],
                    "episode_shuffled_donor_index": descriptor[
                        "episode_shuffled_donor_index"
                    ],
                }
            )
        split_binding_sha = hashlib.sha256(
            raw_stage.canonical_json(descriptor_bindings)
        ).hexdigest()
        state_preflight = base_registration["state_access_preflight"][split]
        if (
            state_preflight.get("row_count") != len(rows)
            or state_preflight.get("future_measured_state_opened") is not False
            or not raw_stage.identity_valid(state_preflight)
        ):
            raise RecurrentFlowBridgeError(f"{split} state preflight differs")
        splits[split] = {
            "manifest": manifest,
            "cache_metadata": cache_metadata,
            "arrays": arrays,
            "rows": len(rows),
            "descriptor_binding_sha256": split_binding_sha,
            "state_archive_layout_preflight_identity_sha256": state_preflight[
                "identity_sha256"
            ],
            "all_state_archive_layouts_revalidated": True,
            "selected_state_values_read_at_registration": False,
            "raw_flow_arrays_opened": False,
            "rgb_cache_bytes_hashed": True,
            "rgb_array_loaded": False,
            "future_rgb_values_decoded_or_indexed": False,
            "deterministic_donors_and_strata_recomputed": True,
        }
    return raw_stage.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_flow_raw_input_revalidation",
            "splits": splits,
            "manifest_rgb_action_bytes_rehashed": True,
            "state_archive_layouts_replayed": True,
            "raw_flow_numeric_equivalence_inherited": False,
            "rgb_cache_bytes_hashed": True,
            "future_rgb_values_decoded_or_indexed": False,
            "rgb_array_loaded": False,
            "future_measured_state_opened": False,
            "protected_test_accessed": False,
        }
    )


def _freeze_causal_prefix_values(
    base_registration: Mapping[str, Any]
) -> dict[str, Any]:
    """Seal exactly the observed 0:5 states and candidate actions for all rows."""

    frozen_splits = {}
    for split in ("train", "val"):
        rows, actions, descriptors = raw_stage._load_split_registration(
            base_registration, split
        )
        frozen_rows = []
        for index, (row, descriptor) in enumerate(zip(rows, descriptors, strict=True)):
            measured, raw_actions, receipt = _read_history_prefix_and_actions_once(
                Path(str(row["episode_dir"])) / "states.npz",
                history_rows=row["frame_indices"][:HISTORY_FRAMES],
                action_start=int(row["start"]),
                action_stop=int(row["start"])
                + raw_stage.SAMPLE_SIZE * raw_stage.CHUNK_SIZE,
            )
            candidate = raw_actions.reshape(
                raw_stage.SAMPLE_SIZE,
                raw_stage.CHUNK_SIZE,
                raw_stage.ACTION_DIM,
            )
            cached = np.asarray(actions[index])[..., : raw_stage.ACTION_DIM]
            if (
                not np.array_equal(candidate, cached)
                or raw_stage.tensor_sha256(np.asarray(actions[index]))
                != descriptor["action_row_sha256"]
            ):
                raise RecurrentFlowBridgeError(
                    f"{split} causal-prefix action reconciliation differs: {index}"
                )
            frozen_rows.append(
                raw_stage.identity_payload(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "recurrent_flow_registered_causal_prefix_row",
                        "split": split,
                        "clip_index": index,
                        "clip_id": row["clip_id"],
                        "history_state_indexes": list(row["frame_indices"][:5]),
                        "measured_history_sha256": raw_stage.tensor_sha256(measured),
                        "candidate_actions_14d_sha256": raw_stage.tensor_sha256(
                            candidate
                        ),
                        "cached_actions_23d_sha256": descriptor[
                            "action_row_sha256"
                        ],
                        "state_access_receipt_identity_sha256": receipt[
                            "identity_sha256"
                        ],
                        "observed_state_array_data_bytes_returned": 280,
                        "registered_action_array_data_bytes_returned": 3_640,
                        "future_measured_state_array_data_bytes_returned": 0,
                        "future_measured_state_opened": False,
                        "future_rgb_opened": False,
                        "protected_test_accessed": False,
                    }
                )
            )
        frozen_splits[split] = raw_stage.identity_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "recurrent_flow_registered_causal_prefix_split",
                "split": split,
                "row_count": len(frozen_rows),
                "rows": frozen_rows,
                "observed_state_bytes_per_row": 280,
                "registered_action_bytes_per_row": 3_640,
                "future_measured_state_array_data_bytes_returned": 0,
                "future_measured_state_opened": False,
                "future_rgb_opened": False,
                "protected_test_accessed": False,
            }
        )
    return raw_stage.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_flow_registered_causal_prefix_all_splits",
            "splits": frozen_splits,
            "total_rows": raw_stage.TRAIN_COUNT + raw_stage.VAL_COUNT,
            "state_value_hashes_frozen_before_cache_materialization": True,
            "candidate_action_value_hashes_frozen_before_cache_materialization": True,
            "future_measured_state_array_data_bytes_returned": 0,
            "future_measured_state_opened": False,
            "future_rgb_opened": False,
            "protected_test_accessed": False,
        }
    )


def _validate_source(registration: Mapping[str, Any]) -> None:
    source = registration.get("source", {})
    observed = raw_stage.clean_repository(
        Path(str(source.get("repository", ""))),
        str(source.get("git_commit", "")),
        "recurrent-flow bridge source",
    )
    if observed["git_commit"] != source.get("git_commit"):
        raise RecurrentFlowBridgeError("bridge source commit changed")
    current = _source_records()
    if any(
        current.get(name, {}).get("sha256") != record.get("sha256")
        for name, record in source.get("files", {}).items()
    ):
        raise RecurrentFlowBridgeError("bridge source file changed after registration")


def command_verify_confirmation(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            validate_confirmation(args.confirmation_root, args.strict_audit),
            sort_keys=True,
        )
    )
    return 0


def command_register_cache(args: argparse.Namespace) -> int:
    output = raw_stage.fresh_lustre_root(args.output)
    source = raw_stage.clean_repository(
        args.source_repo, args.expected_commit, "recurrent-flow bridge source"
    )
    canonical_raw_path = args.raw_cache_registration.resolve(strict=True)
    if (
        canonical_raw_path != BASE_RAW_CACHE_REGISTRATION_PATH
        or raw_stage.sha256_file(canonical_raw_path)
        != BASE_RAW_CACHE_REGISTRATION_SHA256
    ):
        raise RecurrentFlowBridgeError("only the canonical v7 raw input registration is allowed")
    base_registration = raw_stage.validate_cache_registration(canonical_raw_path)
    if base_registration.get("identity_sha256") != BASE_RAW_CACHE_REGISTRATION_IDENTITY:
        raise RecurrentFlowBridgeError("canonical v7 raw registration identity differs")
    input_revalidation = _raw_input_revalidation(base_registration)
    causal_prefix_freeze = _freeze_causal_prefix_values(base_registration)
    sealed = validate_confirmation(args.confirmation_root, args.strict_audit)
    output.mkdir(mode=0o700)
    lineage = output / "lineage"
    lineage.mkdir(mode=0o700)
    copied = {}
    for name, record in sealed.items():
        if not isinstance(record, Mapping) or "sha256" not in record:
            continue
        if name == "predictor":
            # Never copy the privileged archive into the cache/runtime tree.
            continue
        source_path = _resolve_record(Path(sealed["root"]), record, Path(record["path"]).name)
        destination = lineage / Path(record["path"]).name
        shutil.copyfile(source_path, destination)
        copied[name] = raw_stage.file_record(destination)
    privileged_predictor = _resolve_record(
        Path(sealed["root"]), sealed["predictor"], "prior/model_state_and_predictions.npz"
    )
    inference_predictor = lineage / "recurrent_predictor_inference_only.npz"
    extraction = extract_inference_predictor(privileged_predictor, inference_predictor)
    extraction_path = lineage / "inference_predictor_extraction.json"
    raw_stage.exclusive_json(extraction_path, extraction)
    registration = raw_stage.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": CACHE_KIND,
            "status": "registered_before_recurrent_rendering_or_wan_outcomes",
            "created_at_utc": raw_stage.now(),
            "output_root": str(output),
            "source": {
                "repository": source["path"],
                "git_commit": source["git_commit"],
                "files": _source_records(),
            },
            "raw_stage_input_contract": {
                "registration": raw_stage.file_record(args.raw_cache_registration),
                "identity_sha256": base_registration["identity_sha256"],
                "reuse": "manifests_rgb_actions_descriptors_calibration_renderer_only",
                "raw_flow_arrays_opened": False,
                "input_revalidation": input_revalidation,
                "canonical_registration_identity_sha256": (
                    BASE_RAW_CACHE_REGISTRATION_IDENTITY
                ),
                "canonical_registration_file_sha256": (
                    BASE_RAW_CACHE_REGISTRATION_SHA256
                ),
            },
            "causal_prefix_freeze": causal_prefix_freeze,
            "confirmation": {
                "source_commit": CONFIRMATION_SOURCE_COMMIT,
                "registration_identity_sha256": CONFIRMATION_REGISTRATION_IDENTITY,
                "preparation_identity_sha256": CONFIRMATION_PREPARATION_IDENTITY,
                "closure_identity_sha256": CONFIRMATION_CLOSURE_IDENTITY,
                "analysis_identity_sha256": CONFIRMATION_ANALYSIS_IDENTITY,
                "completion_identity_sha256": CONFIRMATION_COMPLETION_IDENTITY,
                "strict_external_audit_identity_sha256": STRICT_AUDIT_IDENTITY,
                "strict_external_auditor_source_commit": STRICT_AUDITOR_SOURCE_COMMIT,
                "decision": POSITIVE_DECISION,
                "copied_artifacts": copied,
                "privileged_predictor_source": sealed["predictor"],
                "privileged_predictor_copied_to_runtime": False,
                "inference_predictor": raw_stage.file_record(inference_predictor),
                "inference_predictor_extraction": raw_stage.file_record(extraction_path),
                "privileged_source_archive_bytes_hashed": True,
                "forbidden_members_deserialized": [],
                "target_measured_trajectory_deserialized": False,
                "predictor_refit_or_tuning": False,
            },
            "splits": {
                split: {
                    "clip_count": raw_stage.TRAIN_COUNT if split == "train" else raw_stage.VAL_COUNT,
                    "d405_count": base_registration["splits"][split]["d405_count"],
                }
                for split in ("train", "val")
            },
            "representation": {
                "compact_shape": [8, 4, 24, 40],
                "packed_shape": [16, 4, 24, 120],
                "components": [
                    "dx_over_width",
                    "dy_over_height",
                    "visibility",
                    "log_depth_ratio",
                ],
                "predictor_inputs": "measured states at observed frames 0..4 plus registered candidate actions",
                "predictor_calls_per_clip": 1,
                "online_predictor_calls_during_wan_sampling": 0,
                "v5_raw_cache_numeric_equivalence": "not_applicable_not_inherited",
            },
            "controls": {
                "recurrent_aligned": "sealed recurrent-delta trajectory for the same episode",
                "off": "runtime exact zeros",
                "recurrent_shuffled": "complete aligned field from frozen episode-disjoint same-stratum donor",
                "recurrent_wrong_time": "aligned transition t+1 at t, no wrap, final zero",
                "recurrent_hold": "observed frame-4 pose held for all nine poses",
            },
            "causal_inputs_only": True,
            "future_rgb_opened": False,
            "rgb_cache_bytes_hashed": True,
            "future_rgb_values_decoded_or_indexed": False,
            "future_measured_state_opened": False,
            "generator_outcome_opened": False,
            "protected_test_accessed": False,
        }
    )
    raw_stage.exclusive_json(output / "cache_registration.json", registration)
    print(json.dumps(registration, sort_keys=True))
    return 0


def validate_cache_registration(path: Path) -> dict[str, Any]:
    payload = raw_stage.read_json(path, "recurrent cache registration")
    if (
        not raw_stage.identity_valid(payload)
        or payload.get("kind") != CACHE_KIND
        or payload.get("status") != "registered_before_recurrent_rendering_or_wan_outcomes"
        or payload.get("confirmation", {}).get("decision") != POSITIVE_DECISION
        or payload.get("confirmation", {}).get("analysis_identity_sha256")
        != CONFIRMATION_ANALYSIS_IDENTITY
        or payload.get("confirmation", {}).get("completion_identity_sha256")
        != CONFIRMATION_COMPLETION_IDENTITY
        or payload.get("confirmation", {}).get("strict_external_audit_identity_sha256")
        != STRICT_AUDIT_IDENTITY
        or payload.get("confirmation", {}).get("strict_external_auditor_source_commit")
        != STRICT_AUDITOR_SOURCE_COMMIT
        or payload.get("confirmation", {}).get("predictor_refit_or_tuning") is not False
        or payload.get("confirmation", {}).get("privileged_predictor_copied_to_runtime") is not False
        or payload.get("confirmation", {}).get("privileged_source_archive_bytes_hashed") is not True
        or payload.get("confirmation", {}).get("forbidden_members_deserialized") != []
        or payload.get("confirmation", {}).get("target_measured_trajectory_deserialized") is not False
        or payload.get("causal_inputs_only") is not True
        or payload.get("future_rgb_opened") is not False
        or payload.get("future_measured_state_opened") is not False
        or payload.get("generator_outcome_opened") is not False
        or payload.get("protected_test_accessed") is not False
    ):
        raise RecurrentFlowBridgeError("recurrent cache registration differs")
    _validate_source(payload)
    root = Path(str(payload["output_root"])).resolve(strict=True)
    if path.resolve(strict=True) != root / "cache_registration.json":
        raise RecurrentFlowBridgeError("recurrent cache registration path is noncanonical")
    base_record = payload["raw_stage_input_contract"]["registration"]
    base_path = _resolve_record(root, base_record, "cache_registration.json")
    if (
        base_path != BASE_RAW_CACHE_REGISTRATION_PATH
        or raw_stage.sha256_file(base_path) != BASE_RAW_CACHE_REGISTRATION_SHA256
    ):
        raise RecurrentFlowBridgeError("canonical v7 raw registration path/hash differs")
    base_registration = raw_stage.validate_cache_registration(base_path)
    if (
        base_registration["identity_sha256"]
        != BASE_RAW_CACHE_REGISTRATION_IDENTITY
        or base_registration["identity_sha256"]
        != payload["raw_stage_input_contract"]["identity_sha256"]
    ):
        raise RecurrentFlowBridgeError("raw-stage input registration changed")
    if _raw_input_revalidation(base_registration) != payload[
        "raw_stage_input_contract"
    ].get("input_revalidation"):
        raise RecurrentFlowBridgeError("bridge-used raw inputs changed after registration")
    prefix = payload.get("causal_prefix_freeze", {})
    if (
        not raw_stage.identity_valid(prefix)
        or prefix.get("total_rows") != raw_stage.TRAIN_COUNT + raw_stage.VAL_COUNT
        or prefix.get("state_value_hashes_frozen_before_cache_materialization")
        is not True
        or prefix.get("candidate_action_value_hashes_frozen_before_cache_materialization")
        is not True
        or prefix.get("future_measured_state_opened") is not False
    ):
        raise RecurrentFlowBridgeError("registered causal-prefix freeze differs")
    return payload


def _read_history_prefix_and_actions_once(
    state_path: Path,
    *,
    history_rows: Sequence[int],
    action_start: int,
    action_stop: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Direct-read five observed rows and one candidate-action slice.

    State rows may be non-contiguous.  Each selected state row is read once;
    each of the joint/gripper action members is read once for the whole slice.
    No unselected NPY payload byte may intersect the audited pread ledger.
    """

    rows = tuple(int(value) for value in history_rows)
    if len(rows) != HISTORY_FRAMES or len(set(rows)) != HISTORY_FRAMES:
        raise RecurrentFlowBridgeError("history state indexes must be five unique rows")
    if min(rows) < 0 or action_start < 0 or action_stop <= action_start:
        raise RecurrentFlowBridgeError("selected history/action indexes are invalid")
    path = state_path.expanduser()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RecurrentFlowBridgeError(f"regular absolute states archive required: {path}")
    path = path.resolve(strict=True)
    archive_bytes = path.stat().st_size
    selected_ranges: dict[str, list[tuple[int, int]]] = {}
    selected_bytes: dict[str, list[bytes]] = {}
    with raw_stage._AuditedArchiveFile(path) as archive:
        layouts, directory_offset, directory_bytes = raw_stage._scan_stored_npz_layout(
            archive, archive_bytes=archive_bytes
        )
        headers = {
            name: raw_stage._parse_float32_c_npy_header(
                archive,
                layouts[name],
                expected_width=raw_stage.STATE_ARRAY_WIDTHS[name],
            )
            for name in raw_stage.STATE_ARRAY_WIDTHS
        }
        frame_counts = {int(header["shape"][0]) for header in headers.values()}
        if len(frame_counts) != 1:
            raise RecurrentFlowBridgeError("state/action NPY frame counts differ")
        frame_count = next(iter(frame_counts))
        if max(rows) >= frame_count or action_stop > frame_count:
            raise RecurrentFlowBridgeError("selected history/action range exceeds archive")
        selections = {
            "joint_states.npy": [(row, row + 1) for row in rows],
            "gripper_states.npy": [(row, row + 1) for row in rows],
            "joint_actions.npy": [(action_start, action_stop)],
            "gripper_actions.npy": [(action_start, action_stop)],
        }
        for name, row_ranges in selections.items():
            width = raw_stage.STATE_ARRAY_WIDTHS[name]
            data_start = int(headers[name]["array_data_archive_byte_start"])
            selected_ranges[name] = []
            selected_bytes[name] = []
            for row_start, row_stop in row_ranges:
                byte_start = data_start + row_start * width * 4
                byte_stop = data_start + row_stop * width * 4
                selected_ranges[name].append((byte_start, byte_stop))
                selected_bytes[name].append(
                    raw_stage._read_exact_at(
                        archive,
                        offset=byte_start,
                        size=byte_stop - byte_start,
                        label=f"selected_array_data:{name}",
                    )
                )
        records = tuple(dict(record) for record in archive.records)
        # Equivalent to the raw Stage-1 audit, generalized from one contiguous
        # state range to five explicitly selected row ranges.
        for record in records:
            read_start = int(record["archive_byte_start"])
            read_stop = int(record["archive_byte_stop"])
            for name, member in layouts.items():
                if name in headers:
                    data_start = int(headers[name]["array_data_archive_byte_start"])
                    data_stop = int(headers[name]["array_data_archive_byte_stop"])
                else:
                    data_start = member.payload_offset
                    data_stop = member.payload_offset + member.payload_bytes
                overlap_start = max(read_start, data_start)
                overlap_stop = min(read_stop, data_stop)
                if overlap_stop <= overlap_start:
                    continue
                allowed = selected_ranges.get(name, [])
                if not raw_stage._range_is_contained(overlap_start, overlap_stop, allowed):
                    raise RecurrentFlowBridgeError(
                        f"unselected array-data bytes read from {name}"
                    )

    def decode(name: str) -> np.ndarray:
        width = raw_stage.STATE_ARRAY_WIDTHS[name]
        return np.concatenate(
            [
                np.frombuffer(value, dtype=np.dtype("<f4")).copy().reshape(-1, width)
                for value in selected_bytes[name]
            ],
            axis=0,
        )

    measured = np.concatenate(
        (decode("joint_states.npy"), decode("gripper_states.npy")), axis=1
    )
    actions = np.concatenate(
        (decode("joint_actions.npy"), decode("gripper_actions.npy")), axis=1
    )
    expected_action_bytes = (action_stop - action_start) * raw_stage.ACTION_DIM * 4
    expected_state_bytes = HISTORY_FRAMES * raw_stage.ACTION_DIM * 4
    state_ledger = [
        (name, interval, payload)
        for name in ("joint_states.npy", "gripper_states.npy")
        for interval, payload in zip(selected_ranges[name], selected_bytes[name], strict=True)
    ]
    action_ledger = [
        (name, selected_ranges[name][0], selected_bytes[name][0])
        for name in ("joint_actions.npy", "gripper_actions.npy")
    ]
    if sum(len(value) for _, _, value in state_ledger) != expected_state_bytes:
        raise RecurrentFlowBridgeError("observed-state byte accounting differs")
    if sum(len(value) for _, _, value in action_ledger) != expected_action_bytes:
        raise RecurrentFlowBridgeError("candidate-action byte accounting differs")
    receipt = raw_stage.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_flow_history_prefix_state_access",
            "path": str(path),
            "archive_bytes": archive_bytes,
            "history_state_indexes": list(rows),
            "latest_measured_state_index": rows[-1],
            "action_slice": [action_start, action_stop],
            "observed_state_array_data_bytes_returned": expected_state_bytes,
            "registered_action_array_data_bytes_returned": expected_action_bytes,
            "state_member_pread_count": len(state_ledger),
            "action_member_pread_count": len(action_ledger),
            "logical_candidate_action_slice_read_count": 1,
            "selected_array_data_pread_ledger": [
                {
                    "member": name[:-4],
                    "archive_byte_start": interval[0],
                    "archive_byte_stop": interval[1],
                    "bytes": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(),
                }
                for name, interval, value in (*state_ledger, *action_ledger)
            ],
            "zip_directory_archive_byte_start": directory_offset,
            "zip_directory_bytes": directory_bytes,
            "all_array_data_reads_within_selected_ranges": True,
            "future_measured_state_array_data_bytes_returned": 0,
            "unregistered_action_array_data_bytes_returned": 0,
            "unused_member_payload_bytes_returned": 0,
            "whole_archive_content_digest_computed": False,
            "future_measured_state_opened": False,
            "future_rgb_opened": False,
            "protected_test_accessed": False,
        }
    )
    return measured.astype(np.float32), actions.astype(np.float32), receipt


def _read_causal_row(
    row: Mapping[str, Any], descriptor: Mapping[str, Any], cached_action: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    state_path = Path(str(row["episode_dir"])) / "states.npz"
    start = int(row["start"])
    measured, raw_actions, receipt = _read_history_prefix_and_actions_once(
        state_path,
        history_rows=row["frame_indices"][:HISTORY_FRAMES],
        action_start=start,
        action_stop=start + raw_stage.SAMPLE_SIZE * raw_stage.CHUNK_SIZE,
    )
    actions = raw_actions.reshape(
        raw_stage.SAMPLE_SIZE, raw_stage.CHUNK_SIZE, raw_stage.ACTION_DIM
    )
    if not np.array_equal(actions, np.asarray(cached_action)[..., : raw_stage.ACTION_DIM]):
        raise RecurrentFlowBridgeError("cached/raw candidate actions differ")
    if raw_stage.tensor_sha256(np.asarray(cached_action)) != descriptor["action_row_sha256"]:
        raise RecurrentFlowBridgeError("registered candidate-action row changed")
    return measured, actions.astype(np.float32), receipt


def _render_one(
    poses: np.ndarray,
    *,
    renderer_state: tuple[Any, ...],
    fy: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    model, data, mujoco, renderer, option, addresses, camera_id, moving = renderer_state
    return raw_stage._render_trajectory(
        poses=poses,
        model=model,
        data=data,
        mujoco=mujoco,
        renderer=renderer,
        option=option,
        addresses=addresses,
        camera_id=camera_id,
        moving_geoms=moving,
        fy=fy,
    )


def command_build_cache(args: argparse.Namespace) -> int:
    registration = validate_cache_registration(args.registration)
    base_record = registration["raw_stage_input_contract"]["registration"]
    base_registration_path = _resolve_record(
        Path(registration["output_root"]), base_record, "cache_registration.json"
    )
    base_registration = raw_stage.validate_cache_registration(base_registration_path)
    raw_stage.enforce_current_cache_renderer_runtime(base_registration)
    split = args.split
    count = raw_stage.TRAIN_COUNT if split == "train" else raw_stage.VAL_COUNT
    split_root = Path(registration["output_root"]) / split
    if split_root.exists() or split_root.is_symlink():
        raise RecurrentFlowBridgeError(f"fresh recurrent split already exists: {split_root}")
    split_root.mkdir(mode=0o700)
    rows, actions, descriptors = raw_stage._load_split_registration(base_registration, split)
    arrays = {
        source: np.lib.format.open_memmap(
            split_root / f"{source}.float16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(count, 8, 4, 24, 40),
        )
        for source in CACHE_SOURCES
    }
    for value in arrays.values():
        value[:] = 0
        value.flush()
    predictor_path = _resolve_record(
        Path(registration["output_root"]),
        registration["confirmation"]["inference_predictor"],
        "lineage/recurrent_predictor_inference_only.npz",
    )
    renderer_state = raw_stage._create_renderer(base_registration)
    lineage_rows = []
    eligible = []
    started = time.perf_counter()
    try:
        frozen_prefix_rows = registration["causal_prefix_freeze"]["splits"][split][
            "rows"
        ]
        for index, (row, descriptor) in enumerate(zip(rows, descriptors, strict=True)):
            if any(
                descriptor.get(name) != value
                for name, value in {
                    "clip_index": index,
                    "clip_id": row.get("clip_id"),
                    "episode_dir": row.get("episode_dir"),
                    "frame_indices": row.get("frame_indices"),
                    "start": row.get("start"),
                }.items()
            ):
                raise RecurrentFlowBridgeError("registered descriptor order changed")
            history, candidate_actions, access = _read_causal_row(
                row, descriptor, np.asarray(actions[index])
            )
            frozen_prefix = frozen_prefix_rows[index]
            if (
                not raw_stage.identity_valid(frozen_prefix)
                or frozen_prefix.get("clip_index") != index
                or frozen_prefix.get("clip_id") != row["clip_id"]
                or frozen_prefix.get("measured_history_sha256")
                != raw_stage.tensor_sha256(history)
                or frozen_prefix.get("candidate_actions_14d_sha256")
                != raw_stage.tensor_sha256(candidate_actions)
                or frozen_prefix.get("state_access_receipt_identity_sha256")
                != access["identity_sha256"]
            ):
                raise RecurrentFlowBridgeError(
                    f"registered causal-prefix values changed: {split}/{index}"
                )
            is_eligible = bool(descriptor["d405_eligible"])
            eligible.append(is_eligible)
            diagnostic: dict[str, Any] = {}
            if is_eligible:
                mcap = Path(descriptor["raw_mcap"])
                calibration = _decode_top_calibration(mcap)
                if calibration.get("camera_type") != CAMERA_TYPE:
                    raise RecurrentFlowBridgeError("D405 eligibility changed")
                identity = hashlib.sha256(raw_stage.canonical_json(calibration)).hexdigest()
                if identity != descriptor["calibration_identity_sha256"]:
                    raise RecurrentFlowBridgeError("registered D405 calibration changed")
                native_height = int(calibration["camera_height"])
                K = np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
                fy = float(K[1, 1] * raw_stage.OUTPUT_HEIGHT / native_height)
                if not math.isfinite(fy) or fy <= 0:
                    raise RecurrentFlowBridgeError("scaled D405 focal length is invalid")
                renderer_state[0].cam_fovy[renderer_state[6]] = math.degrees(
                    2.0 * math.atan(raw_stage.OUTPUT_HEIGHT / (2.0 * fy))
                )
                recurrent_pose = predict_recurrent_trajectory(
                    history[None], candidate_actions[None], predictor_path
                )[0]
                hold_pose = np.broadcast_to(history[4], (9, raw_stage.ACTION_DIM)).copy()
                hold_pose = trajectory._render_pose(hold_pose)
                aligned, aligned_diag = _render_one(
                    recurrent_pose, renderer_state=renderer_state, fy=fy
                )
                hold, hold_diag = _render_one(
                    hold_pose, renderer_state=renderer_state, fy=fy
                )
                hold[:, (0, 1, 3)] = 0
                arrays["recurrent_aligned"][index] = aligned
                arrays["recurrent_hold"][index] = hold
                diagnostic = {
                    "aligned": aligned_diag,
                    "hold": hold_diag,
                    "recurrent_pose_sha256": raw_stage.tensor_sha256(recurrent_pose),
                }
            lineage_rows.append(
                raw_stage.identity_payload(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "recurrent_physics_flow_cache_row",
                        "split": split,
                        "clip_index": index,
                        "clip_id": row["clip_id"],
                        "d405_eligible": is_eligible,
                        "history_state_indexes": list(row["frame_indices"][:5]),
                        "latest_measured_state_boundary": 4,
                        "state_access_receipt": access,
                        "diagnostic": diagnostic,
                        "predictor_refit_or_tuning": False,
                        "future_rgb_opened": False,
                        "future_measured_state_opened": False,
                        "protected_test_accessed": False,
                    }
                )
            )
        donors = [item["episode_shuffled_donor_index"] for item in descriptors]
        shuffled, wrong = derive_interventions(
            np.asarray(arrays["recurrent_aligned"]), donors, eligible
        )
        arrays["recurrent_shuffled"][:] = shuffled
        arrays["recurrent_wrong_time"][:] = wrong
        for value in arrays.values():
            value.flush()
    finally:
        renderer_state[3].close()
    del arrays
    lineage_path = split_root / "row_lineage.jsonl"
    raw_stage.exclusive_jsonl(lineage_path, lineage_rows)
    records = {
        source: raw_stage.file_record(split_root / f"{source}.float16.npy")
        for source in CACHE_SOURCES
    }
    metadata = raw_stage.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_physics_flow_cache_metadata",
            "schema": CACHE_SCHEMA,
            "complete": True,
            "split": split,
            "clip_count": count,
            "compact_transition_shape": [count, 8, 4, 24, 40],
            "flow_dtype": "float16",
            "condition_family": "sealed_recurrent_delta_geometry",
            "arrays": records,
            "aligned_flow_file": records["recurrent_aligned"]["path"],
            "aligned_flow_sha256": records["recurrent_aligned"]["sha256"],
            "cache_registration": raw_stage.file_record(args.registration),
            "row_lineage": raw_stage.file_record(lineage_path),
            "confirmation_decision": POSITIVE_DECISION,
            "confirmation_analysis_identity_sha256": CONFIRMATION_ANALYSIS_IDENTITY,
            "confirmation_completion_identity_sha256": CONFIRMATION_COMPLETION_IDENTITY,
            "privileged_predictor_source_sha256": FROZEN_PREDICTOR_SHA256,
            "inference_predictor_sha256": registration["confirmation"][
                "inference_predictor"
            ]["sha256"],
            "predictor_refit_or_tuning": False,
            "render_wall_seconds": time.perf_counter() - started,
            "causal_inputs_only": True,
            "future_rgb_opened": False,
            "future_measured_state_opened": False,
            "generator_outcome_opened": False,
            "protected_test_accessed": False,
        }
    )
    raw_stage.exclusive_json(split_root / "metadata.json", metadata)
    print(json.dumps(metadata, sort_keys=True))
    return 0


def load_cache_metadata(path: Path, split: str) -> dict[str, Any]:
    payload = raw_stage.read_json(path, f"recurrent {split} metadata")
    expected_count = raw_stage.TRAIN_COUNT if split == "train" else raw_stage.VAL_COUNT
    if (
        not raw_stage.identity_valid(payload)
        or payload.get("schema") != CACHE_SCHEMA
        or payload.get("complete") is not True
        or payload.get("split") != split
        or payload.get("clip_count") != expected_count
        or payload.get("condition_family") != "sealed_recurrent_delta_geometry"
        or payload.get("confirmation_decision") != POSITIVE_DECISION
        or payload.get("confirmation_analysis_identity_sha256")
        != CONFIRMATION_ANALYSIS_IDENTITY
        or payload.get("confirmation_completion_identity_sha256")
        != CONFIRMATION_COMPLETION_IDENTITY
        or payload.get("predictor_refit_or_tuning") is not False
        or payload.get("causal_inputs_only") is not True
        or payload.get("future_rgb_opened") is not False
        or payload.get("future_measured_state_opened") is not False
        or payload.get("generator_outcome_opened") is not False
        or payload.get("protected_test_accessed") is not False
    ):
        raise RecurrentFlowBridgeError(f"recurrent {split} metadata differs")
    return payload


def _full_replay_cache(
    *,
    registration: Mapping[str, Any],
    base_registration: Mapping[str, Any],
    split: str,
    values: Mapping[str, np.ndarray],
    lineage_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute every aligned/hold row from causal inputs and render exactly."""

    raw_stage.enforce_current_cache_renderer_runtime(base_registration)
    rows, actions, descriptors = raw_stage._load_split_registration(
        base_registration, split
    )
    predictor_path = _resolve_record(
        Path(registration["output_root"]),
        registration["confirmation"]["inference_predictor"],
        "lineage/recurrent_predictor_inference_only.npz",
    )
    renderer_state = raw_stage._create_renderer(base_registration)
    compared = 0
    max_abs_error = 0.0
    try:
        for index, (row, descriptor, lineage) in enumerate(
            zip(rows, descriptors, lineage_rows, strict=True)
        ):
            history, candidate_actions, access = _read_causal_row(
                row, descriptor, np.asarray(actions[index])
            )
            if access["identity_sha256"] != lineage["state_access_receipt"][
                "identity_sha256"
            ]:
                raise RecurrentFlowBridgeError("causal state/action replay differs")
            if not bool(descriptor["d405_eligible"]):
                continue
            calibration = _decode_top_calibration(Path(descriptor["raw_mcap"]))
            identity = hashlib.sha256(raw_stage.canonical_json(calibration)).hexdigest()
            if (
                calibration.get("camera_type") != CAMERA_TYPE
                or identity != descriptor["calibration_identity_sha256"]
            ):
                raise RecurrentFlowBridgeError("replay D405 calibration differs")
            native_height = int(calibration["camera_height"])
            K = np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
            fy = float(K[1, 1] * raw_stage.OUTPUT_HEIGHT / native_height)
            renderer_state[0].cam_fovy[renderer_state[6]] = math.degrees(
                2.0 * math.atan(raw_stage.OUTPUT_HEIGHT / (2.0 * fy))
            )
            recurrent_pose = predict_recurrent_trajectory(
                history[None], candidate_actions[None], predictor_path
            )[0]
            if raw_stage.tensor_sha256(recurrent_pose) != lineage["diagnostic"].get(
                "recurrent_pose_sha256"
            ):
                raise RecurrentFlowBridgeError("recurrent pose replay hash differs")
            hold_pose = trajectory._render_pose(
                np.broadcast_to(history[4], (9, raw_stage.ACTION_DIM)).copy()
            )
            aligned, _ = _render_one(recurrent_pose, renderer_state=renderer_state, fy=fy)
            hold, _ = _render_one(hold_pose, renderer_state=renderer_state, fy=fy)
            hold[:, (0, 1, 3)] = 0
            for source, expected in (
                ("recurrent_aligned", aligned),
                ("recurrent_hold", hold),
            ):
                observed = np.asarray(values[source][index])
                error = float(
                    np.max(np.abs(observed.astype(np.float32) - expected.astype(np.float32)))
                )
                max_abs_error = max(max_abs_error, error)
                if not np.array_equal(observed, expected):
                    raise RecurrentFlowBridgeError(
                        f"full predictor/render replay differs: {split}/{index}/{source}"
                    )
                compared += 1
    finally:
        renderer_state[3].close()
    return {
        "rows": len(rows),
        "eligible_rows": sum(bool(item["d405_eligible"]) for item in descriptors),
        "rendered_arrays_compared": compared,
        "max_abs_error": max_abs_error,
        "byte_exact": True,
        "all_causal_state_action_receipts_replayed": True,
        "future_rgb_opened": False,
        "future_measured_state_opened": False,
        "protected_test_accessed": False,
    }


def command_audit_cache(args: argparse.Namespace) -> int:
    if not args.full_replay:
        raise RecurrentFlowBridgeError(
            "canonical cache audit requires --full-replay on its first and only write"
        )
    metadata = load_cache_metadata(args.metadata, args.split)
    count = raw_stage.TRAIN_COUNT if args.split == "train" else raw_stage.VAL_COUNT
    registration_path = Path(metadata["cache_registration"]["path"])
    if raw_stage.file_record(registration_path.resolve(strict=True)) != metadata[
        "cache_registration"
    ]:
        raise RecurrentFlowBridgeError("metadata/cache-registration record differs")
    registration = validate_cache_registration(registration_path)
    base_path = _resolve_record(
        Path(registration["output_root"]),
        registration["raw_stage_input_contract"]["registration"],
        "cache_registration.json",
    )
    base_registration = raw_stage.validate_cache_registration(base_path)
    descriptors = base_registration["splits"][args.split]["descriptors"]
    split_root = args.metadata.parent.resolve(strict=True)
    if args.metadata.resolve(strict=True) != split_root / "metadata.json":
        raise RecurrentFlowBridgeError("recurrent metadata path is noncanonical")
    lineage_record = metadata.get("row_lineage", {})
    lineage_path = Path(str(lineage_record.get("path", ""))).resolve(strict=True)
    if (
        lineage_path != split_root / "row_lineage.jsonl"
        or raw_stage.file_record(lineage_path) != lineage_record
    ):
        raise RecurrentFlowBridgeError("recurrent row-lineage artifact differs")
    lineage_rows = raw_stage.read_jsonl(lineage_path)
    if len(lineage_rows) != count:
        raise RecurrentFlowBridgeError("recurrent row-lineage count differs")
    for index, (lineage, descriptor) in enumerate(
        zip(lineage_rows, descriptors, strict=True)
    ):
        access = lineage.get("state_access_receipt", {})
        if (
            not raw_stage.identity_valid(lineage)
            or lineage.get("kind") != "recurrent_physics_flow_cache_row"
            or lineage.get("split") != args.split
            or lineage.get("clip_index") != index
            or lineage.get("clip_id") != descriptor["clip_id"]
            or lineage.get("d405_eligible") is not bool(descriptor["d405_eligible"])
            or lineage.get("history_state_indexes") != descriptor["frame_indices"][:5]
            or lineage.get("latest_measured_state_boundary") != 4
            or not raw_stage.identity_valid(access)
            or access.get("observed_state_array_data_bytes_returned") != 280
            or access.get("registered_action_array_data_bytes_returned") != 3_640
            or access.get("logical_candidate_action_slice_read_count") != 1
            or access.get("future_measured_state_opened") is not False
            or lineage.get("future_rgb_opened") is not False
            or lineage.get("future_measured_state_opened") is not False
            or lineage.get("protected_test_accessed") is not False
        ):
            raise RecurrentFlowBridgeError(f"recurrent row lineage differs: {index}")
    values = {}
    for source in CACHE_SOURCES:
        record = metadata["arrays"][source]
        path = Path(record["path"]).resolve(strict=True)
        expected_path = split_root / f"{source}.float16.npy"
        if path != expected_path or raw_stage.file_record(path) != record:
            raise RecurrentFlowBridgeError(f"recurrent cache changed: {source}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        raw_stage.validate_compact_flow(array, count=count)
        values[source] = array
    if (
        Path(metadata["aligned_flow_file"]).resolve(strict=True)
        != split_root / "recurrent_aligned.float16.npy"
        or metadata["aligned_flow_sha256"]
        != metadata["arrays"]["recurrent_aligned"]["sha256"]
    ):
        raise RecurrentFlowBridgeError("canonical recurrent aligned path differs")
    eligible = [bool(item["d405_eligible"]) for item in descriptors]
    donors = [item["episode_shuffled_donor_index"] for item in descriptors]
    expected_shuffle, expected_wrong = derive_interventions(
        np.asarray(values["recurrent_aligned"]), donors, eligible
    )
    if not np.array_equal(expected_shuffle, values["recurrent_shuffled"]):
        raise RecurrentFlowBridgeError("recurrent shuffled intervention differs")
    if not np.array_equal(expected_wrong, values["recurrent_wrong_time"]):
        raise RecurrentFlowBridgeError("recurrent wrong-time intervention differs")
    if any(
        np.any(values[source][index] != 0)
        for index, is_eligible in enumerate(eligible)
        if not is_eligible
        for source in CACHE_SOURCES
    ):
        raise RecurrentFlowBridgeError("ineligible recurrent cache row is nonzero")
    for index, is_eligible in enumerate(eligible):
        if not is_eligible:
            continue
        aligned = np.asarray(values["recurrent_aligned"][index])
        hold = np.asarray(values["recurrent_hold"][index])
        if any(not np.any(aligned[transition, 2] > 0) for transition in range(8)):
            raise RecurrentFlowBridgeError("eligible aligned row lacks rendered support")
        if any(not np.any(hold[transition, 2] > 0) for transition in range(8)):
            raise RecurrentFlowBridgeError("eligible hold row lacks rendered support")
        if np.any(hold[:, (0, 1, 3)] != 0):
            raise RecurrentFlowBridgeError("hold displacement/depth components are nonzero")
        if not all(np.array_equal(hold[0, 2], hold[t, 2]) for t in range(1, 8)):
            raise RecurrentFlowBridgeError("hold visibility is not transition-invariant")
    replay = (
        _full_replay_cache(
            registration=registration,
            base_registration=base_registration,
            split=args.split,
            values=values,
            lineage_rows=lineage_rows,
        )
        if args.full_replay
        else None
    )
    trusted = replay is not None and replay.get("byte_exact") is True
    audit = raw_stage.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_physics_flow_cache_audit",
            "status": (
                "passed_full_all_row_predictor_render_replay"
                if trusted
                else "structural_pass_untrusted_for_launch"
            ),
            "split": args.split,
            "metadata": raw_stage.file_record(args.metadata),
            "cache_registration_identity_sha256": registration["identity_sha256"],
            "rows": count,
            "all_array_hashes_verified": True,
            "episode_shuffled_exact": True,
            "nonwrapping_wrong_time_exact": True,
            "non_d405_exact_zero": True,
            "eligible_aligned_and_hold_support_nonzero": True,
            "hold_semantics_exact": True,
            "row_lineage": raw_stage.file_record(lineage_path),
            "full_all_row_predictor_render_replay": replay,
            "launch_eligible": trusted,
            "future_rgb_opened": False,
            "future_measured_state_opened": False,
            "generator_outcome_opened": False,
            "protected_test_accessed": False,
        }
    )
    output = args.metadata.parent / "audit.json"
    raw_stage.exclusive_json(output, audit)
    print(json.dumps(audit, sort_keys=True))
    return 0


def command_plan(args: argparse.Namespace) -> int:
    registration = validate_cache_registration(args.registration)
    root = Path(registration["output_root"])
    cache_records = {}
    for split in ("train", "val"):
        metadata_path = root / split / "metadata.json"
        metadata = load_cache_metadata(metadata_path, split)
        audit_path = root / split / "audit.json"
        audit = raw_stage.read_json(audit_path, f"{split} recurrent audit")
        if (
            not raw_stage.identity_valid(audit)
            or audit.get("status") != "passed_full_all_row_predictor_render_replay"
            or audit.get("launch_eligible") is not True
            or audit.get("metadata", {}).get("sha256")
            != raw_stage.sha256_file(metadata_path)
        ):
            raise RecurrentFlowBridgeError(f"{split} cache is not audit-complete")
        cache_records[split] = {
            "metadata": raw_stage.file_record(metadata_path),
            "audit": raw_stage.file_record(audit_path),
            "aligned_sha256": metadata["aligned_flow_sha256"],
        }
    endpoints = endpoint_grid()
    if args.authorize_screen and args.authorization_token != (
        "AUTHORIZE_RECURRENT_FLOW_WAN_NFE1"
    ):
        raise RecurrentFlowBridgeError("explicit NFE1 screen authorization token differs")
    plan = raw_stage.identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "recurrent_physics_flow_wan_dry_run_plan",
            "mode": "no_commands_executed_no_jobs_submitted",
            "cache_registration_identity_sha256": registration["identity_sha256"],
            "cache_records": cache_records,
            "training": {
                "arms": [
                    {"code": "FLOW-OFF", "fuse_flow": False, "updates": 200},
                    {"code": "RECURRENT-FLOW", "fuse_flow": True, "updates": 200},
                ],
                "same_parent_seed_clip_order_noise_optimizer": True,
                "wan_calls_per_example": 1,
                "online_predictor_calls": 0,
                "predictor_is_cache_time_only": True,
            },
            "evaluation": {
                "endpoints": [asdict(value) for value in endpoints],
                "noise_seeds": list(NOISE_SEEDS),
                "all_endpoints_materialized_before_future_rgb_open": True,
                "wan_calls_equal_within_nfe": True,
                "online_predictor_calls": 0,
                "primary_nfe": 1,
                "controls_required": list(RUNTIME_SOURCES),
            },
            "launch_ready": False,
            "operator_authorized_for_future_screen": bool(args.authorize_screen),
            "submission_performed": False,
            "submission_command": None,
            "remaining_blockers": [
                "freeze study registration and arm run identities",
                "bind recurrent trainer contract without raw-family masquerading",
                "bind thin equal-call evaluator to the recurrent endpoint grid",
                "independent source/cache/parent readiness audit",
            ],
            "protected_test_accessed": False,
        }
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-confirmation")
    verify.add_argument("--confirmation-root", type=Path, required=True)
    verify.add_argument("--strict-audit", type=Path, required=True)
    verify.set_defaults(handler=command_verify_confirmation)
    register = commands.add_parser("register-cache")
    register.add_argument("--source-repo", type=Path, required=True)
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--raw-cache-registration", type=Path, required=True)
    register.add_argument("--confirmation-root", type=Path, required=True)
    register.add_argument("--strict-audit", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)
    register.set_defaults(handler=command_register_cache)
    build = commands.add_parser("build-cache")
    build.add_argument("--registration", type=Path, required=True)
    build.add_argument("--split", choices=("train", "val"), required=True)
    build.set_defaults(handler=command_build_cache)
    audit = commands.add_parser("audit-cache")
    audit.add_argument("--metadata", type=Path, required=True)
    audit.add_argument("--split", choices=("train", "val"), required=True)
    audit.add_argument("--full-replay", action="store_true")
    audit.set_defaults(handler=command_audit_cache)
    plan = commands.add_parser("plan")
    plan.add_argument("--registration", type=Path, required=True)
    plan.add_argument("--authorize-screen", action="store_true")
    plan.add_argument("--authorization-token")
    plan.set_defaults(handler=command_plan)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (RecurrentFlowBridgeError, raw_stage.PhysicsFlowStage1Error) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
