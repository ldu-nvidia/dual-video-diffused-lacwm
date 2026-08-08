#!/usr/bin/env python3
"""Fit immutable CAMP per-channel moments on planner-fit448 only.

This is a pre-training data-reduction job, not a model-training or validation
job.  It reads the registered train RGB cache, encodes each clip twice (the
full 13-frame clip and the independent five-frame observed prefix), verifies
causal equality of the observed Wan tokens, and writes one content-addressable
JSON artifact.  No calibration, validation, or protected-test row is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.modeling.dual_diffusion.causal_motion_plan import (  # noqa: E402
    FROZEN_TRAIN_MANIFEST_SHA256,
    FROZEN_TRAIN_METADATA_SHA256,
    FROZEN_TRAIN_RGB_SHA256,
    FUTURE_TOKENS,
    HISTORY_TOKENS,
    NORMALIZATION_KIND,
    NORMALIZATION_SCHEMA_VERSION,
    PLAN_CHANNELS,
    PLAN_HEIGHT,
    PLAN_WIDTH,
    finalize_channel_moments,
    motion_plan_target,
    planner_partition_indexes,
)


EXPECTED_WORLD_SIZE = 8
EXPECTED_TRAIN_CLIPS = 512
EXPECTED_FIT_CLIPS = 448
EXPECTED_CALIBRATION_CLIPS = 64
EXPECTED_RGB_SHAPE = (512, 13, 3, 180, 960)
EXPECTED_RGB_DTYPE = "float16"
CAUSAL_TOLERANCE = 1e-4
# This job has no distributed model state: each rank independently runs one
# Wan VAE on its assigned GPU and exchanges only small Python records plus
# 16-channel scalar moments. Keep that control plane on CPU/Gloo.
COLLECTIVE_BACKEND = "gloo"
ACCUMULATOR_DEVICE = "cpu"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class MotionPlanStatsError(RuntimeError):
    """The fit-only normalization boundary or source identity was violated."""


def _canonical_json(value: Any) -> bytes:
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


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise MotionPlanStatsError(f"{label} must be a regular absolute file")
    return path.resolve(strict=True)


def _file_record(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MotionPlanStatsError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise MotionPlanStatsError(f"{label} must contain one object")
    return value


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise MotionPlanStatsError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _clean_source(repo: Path, expected_commit: str) -> dict[str, Any]:
    repo = repo.expanduser().resolve(strict=True)
    if repo != REPO_ROOT or COMMIT_RE.fullmatch(expected_commit) is None:
        raise MotionPlanStatsError("tool repository or expected commit differs")
    if _git(repo, "rev-parse", "HEAD") != expected_commit:
        raise MotionPlanStatsError("tool checkout is not the registered commit")
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise MotionPlanStatsError("tool checkout is dirty")
    return {"path": str(repo), "git_commit": expected_commit, "clean": True}


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError("manifest row is not an object")
                    rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise MotionPlanStatsError("train manifest is invalid") from exc
    if len(rows) != EXPECTED_TRAIN_CLIPS:
        raise MotionPlanStatsError("train manifest row count differs")
    for index, row in enumerate(rows):
        if (
            row.get("split") != "train"
            or row.get("auxiliary_index") != index
            or row.get("sample_size") != 13
            or row.get("chunk_size") != 5
            or not isinstance(row.get("clip_id"), str)
            or SHA256_RE.fullmatch(str(row.get("clip_id"))) is None
        ):
            raise MotionPlanStatsError(f"train manifest row {index} differs")
    return rows


def _resolve_rgb(metadata_path: Path, metadata: dict[str, Any]) -> Path:
    value = metadata.get("rgb_file")
    if not isinstance(value, str) or not value:
        raise MotionPlanStatsError("train metadata lacks rgb_file")
    path = Path(value)
    if not path.is_absolute():
        path = metadata_path.parent / path
    return _regular_file(path, "train RGB array")


def _broadcast_rank_zero(value: Any, error: str | None = None) -> Any:
    import torch.distributed as dist

    payload = [value if dist.get_rank() == 0 else None, error]
    dist.broadcast_object_list(payload, src=0)
    if payload[1] is not None:
        raise MotionPlanStatsError(f"rank-zero input registration failed: {payload[1]}")
    return payload[0]


def _rank_zero_inputs(args: argparse.Namespace) -> dict[str, Any]:
    source = _clean_source(args.tool_repo, args.expected_commit)
    manifest = _file_record(args.train_manifest, "train manifest")
    metadata_record = _file_record(args.train_cache_metadata, "train metadata")
    if manifest["sha256"] != FROZEN_TRAIN_MANIFEST_SHA256:
        raise MotionPlanStatsError("train manifest SHA-256 differs")
    if metadata_record["sha256"] != FROZEN_TRAIN_METADATA_SHA256:
        raise MotionPlanStatsError("train metadata SHA-256 differs")
    metadata_path = Path(metadata_record["path"])
    metadata = _read_json(metadata_path, "train metadata")
    if (
        metadata.get("complete") is not True
        or metadata.get("split") != "train"
        or metadata.get("clip_count") != EXPECTED_TRAIN_CLIPS
        or metadata.get("clip_manifest_sha256") != FROZEN_TRAIN_MANIFEST_SHA256
        or metadata.get("rgb_sha256") != FROZEN_TRAIN_RGB_SHA256
        or tuple(metadata.get("rgb_shape", ())) != EXPECTED_RGB_SHAPE
    ):
        raise MotionPlanStatsError("train metadata contract differs")
    rows = _manifest_rows(Path(manifest["path"]))
    rgb = _file_record(_resolve_rgb(metadata_path, metadata), "train RGB array")
    if rgb["sha256"] != FROZEN_TRAIN_RGB_SHA256:
        raise MotionPlanStatsError("train RGB SHA-256 differs")
    fit = planner_partition_indexes(EXPECTED_TRAIN_CLIPS, "planner_fit")
    calibration = planner_partition_indexes(
        EXPECTED_TRAIN_CLIPS, "planner_calibration"
    )
    if len(fit) != EXPECTED_FIT_CLIPS or len(calibration) != EXPECTED_CALIBRATION_CLIPS:
        raise MotionPlanStatsError("planner partition differs")
    return {
        "source": source,
        "manifest": manifest,
        "metadata": metadata_record,
        "rgb": rgb,
        "fit_indexes": list(fit),
        "fit_clip_ids_sha256": _sha256_json([rows[index]["clip_id"] for index in fit]),
    }


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser()
    if not path.is_absolute() or not path.name or path.name in {".", ".."}:
        raise MotionPlanStatsError("output must be an absolute named path")
    parent = path.parent.resolve(strict=True)
    canonical = parent / path.name
    if canonical != path or path.exists() or path.is_symlink():
        raise MotionPlanStatsError("normalization output must be canonical and fresh")
    content = _canonical_json(payload) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise MotionPlanStatsError("refusing to overwrite normalization output") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _configure_runtime(videox_home: Path, wan_dir: Path) -> tuple[Path, Path]:
    videox_home = videox_home.expanduser().resolve(strict=True)
    wan_dir = wan_dir.expanduser().resolve(strict=True)
    shim = REPO_ROOT / "tools" / "env" / "videox_shim"
    project = REPO_ROOT / "projects" / "latent_action_models"
    for root in reversed((str(REPO_ROOT), str(project), str(shim), str(videox_home))):
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    os.environ["VIDEOX_HOME"] = str(videox_home)
    os.environ["WAN_DIR"] = str(wan_dir)
    return videox_home, wan_dir


def _runtime_record(videox_home: Path, wan_dir: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    config_path = videox_home / "config" / "wan2.1" / "wan_civitai.yaml"
    config = _file_record(config_path, "Wan VAE configuration")
    value = OmegaConf.load(config_path)
    vae_subpath = str(
        value.get("vae_kwargs", {}).get("vae_subpath", "Wan2.1_VAE.pth")
    )
    weights_path = (wan_dir / vae_subpath).resolve(strict=True)
    if wan_dir not in weights_path.parents:
        raise MotionPlanStatsError("Wan VAE weights escape the runtime root")
    commit = _git(videox_home, "rev-parse", "HEAD")
    if COMMIT_RE.fullmatch(commit) is None or _git(
        videox_home, "status", "--porcelain", "--untracked-files=all"
    ):
        raise MotionPlanStatsError("VideoX runtime must be a clean commit")
    return {
        "wan_dir": str(wan_dir),
        "videox_home": str(videox_home),
        "videox_git_commit": commit,
        "wan_config": config,
        "wan_vae": _file_record(weights_path, "Wan VAE weights"),
    }


def command_fit(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise MotionPlanStatsError("CAMP statistics require CUDA Wan VAE encoding")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise MotionPlanStatsError("launch CAMP statistics with torchrun")
    torch.cuda.set_device(local_rank)
    # GPU tensors never cross ranks in this fit-only reduction. Initializing
    # NCCL before rank zero hashes the 6.9 GB input adds device-state risk but
    # no useful work, so small records and moments deliberately use Gloo.
    dist.init_process_group(backend=COLLECTIVE_BACKEND)
    if dist.get_world_size() != EXPECTED_WORLD_SIZE:
        raise MotionPlanStatsError("CAMP statistics require exactly eight ranks")
    rank = dist.get_rank()
    rank_zero_inputs = None
    rank_zero_error = None
    if rank == 0:
        try:
            rank_zero_inputs = _rank_zero_inputs(args)
        except BaseException as exc:
            rank_zero_error = f"{type(exc).__name__}: {exc}"
    inputs = _broadcast_rank_zero(rank_zero_inputs, rank_zero_error)
    if not isinstance(inputs, dict):
        raise MotionPlanStatsError("rank-zero input registration failed")
    videox_home, wan_dir = _configure_runtime(args.videox_home, args.wan_dir)
    runtime_record = None
    runtime_error = None
    if rank == 0:
        try:
            runtime_record = _runtime_record(videox_home, wan_dir)
        except BaseException as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
    runtime_record = _broadcast_rank_zero(runtime_record, runtime_error)
    if not isinstance(runtime_record, dict):
        raise MotionPlanStatsError("runtime registration failed")
    from robot_wm.modeling.tokenizers.rgb.wan_vae import WanVAETokenizer

    tokenizer = WanVAETokenizer(
        model_path=str(wan_dir),
        config_path=str(videox_home / "config" / "wan2.1" / "wan_civitai.yaml"),
    ).to(device=torch.device("cuda", local_rank)).eval()
    rgb = np.load(Path(inputs["rgb"]["path"]), mmap_mode="r", allow_pickle=False)
    if tuple(rgb.shape) != EXPECTED_RGB_SHAPE or str(rgb.dtype) != EXPECTED_RGB_DTYPE:
        raise MotionPlanStatsError("train RGB array shape/dtype differs")
    assigned = list(inputs["fit_indexes"])[rank::EXPECTED_WORLD_SIZE]
    if len(assigned) != EXPECTED_FIT_CLIPS // EXPECTED_WORLD_SIZE:
        raise MotionPlanStatsError("rank planner-fit assignment differs")
    device = torch.device("cuda", local_rank)
    channel_sum = torch.zeros(
        PLAN_CHANNELS, dtype=torch.float64, device=ACCUMULATOR_DEVICE
    )
    channel_square_sum = torch.zeros_like(channel_sum)
    local_elements = torch.zeros((), dtype=torch.int64, device=ACCUMULATOR_DEVICE)
    causal_max = torch.zeros((), dtype=torch.float64, device=ACCUMULATOR_DEVICE)
    with torch.inference_mode():
        for index in assigned:
            clip = torch.from_numpy(np.array(rgb[index], copy=True)).unsqueeze(0)
            clip = clip.to(device=device, dtype=torch.float32, non_blocking=False)
            full_latents = tokenizer.encode_temporal(clip.permute(0, 2, 1, 3, 4))
            history_latents = tokenizer.encode_temporal(
                clip[:, :5].permute(0, 2, 1, 3, 4)
            )
            if (
                tuple(full_latents.shape) != (1, 16, 4, 24, 120)
                or tuple(history_latents.shape) != (1, 16, 2, 24, 120)
            ):
                raise MotionPlanStatsError("Wan temporal latent geometry differs")
            difference = (
                full_latents[:, :, :HISTORY_TOKENS] - history_latents
            ).abs().max().double()
            difference_value = float(difference.item())
            causal_max = torch.maximum(
                causal_max,
                torch.tensor(
                    difference_value,
                    dtype=torch.float64,
                    device=ACCUMULATOR_DEVICE,
                ),
            )
            if difference_value > CAUSAL_TOLERANCE:
                raise MotionPlanStatsError(
                    "full-clip observed Wan tokens differ from history-only encoding"
                )
            target = motion_plan_target(full_latents, history_latents).double()
            reduce_dims = (0, 2, 3, 4)
            channel_sum += target.sum(dim=reduce_dims).cpu()
            channel_square_sum += target.square().sum(dim=reduce_dims).cpu()
            local_elements += target.shape[0] * target.shape[2] * target.shape[3] * target.shape[4]
    dist.all_reduce(channel_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(channel_square_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_elements, op=dist.ReduceOp.SUM)
    dist.all_reduce(causal_max, op=dist.ReduceOp.MAX)
    expected_elements = EXPECTED_FIT_CLIPS * FUTURE_TOKENS * PLAN_HEIGHT * PLAN_WIDTH
    if int(local_elements.item()) != expected_elements:
        raise MotionPlanStatsError("global motion-plan element count differs")
    mean, std = finalize_channel_moments(
        channel_sum.cpu(), channel_square_sum.cpu(), expected_elements
    )
    if rank == 0:
        unsigned = {
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "kind": NORMALIZATION_KIND,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete_before_planner_training",
            "tool_repository": inputs["source"],
            "train_manifest": inputs["manifest"],
            "train_cache_metadata": inputs["metadata"],
            "train_rgb": inputs["rgb"],
            "train_manifest_sha256": FROZEN_TRAIN_MANIFEST_SHA256,
            "train_cache_metadata_sha256": FROZEN_TRAIN_METADATA_SHA256,
            "train_rgb_sha256": FROZEN_TRAIN_RGB_SHA256,
            "split_rule": "auxiliary_index_mod_8_nonzero",
            "fit_clips": EXPECTED_FIT_CLIPS,
            "calibration_clips_excluded": EXPECTED_CALIBRATION_CLIPS,
            "validation_clips_read": 0,
            "protected_test_clips_read": 0,
            "fit_indexes_sha256": _sha256_json(inputs["fit_indexes"]),
            "fit_clip_ids_sha256": inputs["fit_clip_ids_sha256"],
            "history_encoding": "independent_five_frame_observed_only",
            "future_tensor_used_for": "statistics_target_only",
            "causal_history_check": "full_clip_first_two_tokens_equals_history_only",
            "causal_history_max_abs_tolerance": CAUSAL_TOLERANCE,
            "causal_history_max_abs_observed": float(causal_max.item()),
            "world_size": EXPECTED_WORLD_SIZE,
            "elements_per_channel": expected_elements,
            "estimator": "population_mean_and_population_standard_deviation",
            "accumulator_dtype": "float64",
            "mean": [float(value) for value in mean.tolist()],
            "std": [float(value) for value in std.tolist()],
            "runtime": {
                **runtime_record,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        }
        payload = {**unsigned, "identity_sha256": _sha256_json(unsigned)}
        _exclusive_json(args.output, payload)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "identity_sha256": payload["identity_sha256"],
                    "file_sha256": _sha256(args.output),
                    "fit_clips": EXPECTED_FIT_CLIPS,
                    "calibration_clips_read": 0,
                    "validation_clips_read": 0,
                    "protected_test_clips_read": 0,
                    "causal_history_max_abs_observed": float(causal_max.item()),
                },
                sort_keys=True,
            )
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-cache-metadata", type=Path, required=True)
    parser.add_argument("--wan-dir", type=Path, required=True)
    parser.add_argument("--videox-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    return command_fit(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
