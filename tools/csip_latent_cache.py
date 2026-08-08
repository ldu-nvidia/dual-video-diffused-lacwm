#!/usr/bin/env python3
"""Extract or verify content-bound clean Wan latents for CSIP Phase-0.

``extract`` must run under exactly eight local ``torchrun`` ranks.  Each rank
writes disjoint immutable shards.  The validation split additionally requires
the fixed checkpoint seal, so validation RGB cannot be opened during fitting.
No CLI option accepts a test manifest, test cache, or arbitrary RGB path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import csip_contract as contract  # noqa: E402


def _broadcast_rank_zero(value: Any, error: str | None = None) -> Any:
    import torch.distributed as dist

    payload = [value if dist.get_rank() == 0 else None, error]
    dist.broadcast_object_list(payload, src=0)
    if payload[1] is not None:
        raise contract.CSIPContractError(
            f"rank-zero latent-cache registration failed: {payload[1]}"
        )
    return payload[0]


def _configure_runtime(registration: dict[str, Any]) -> tuple[Path, Path]:
    runtime = registration["runtime"]
    videox = Path(runtime["videox_home"]).resolve(strict=True)
    wan = Path(runtime["wan_dir"]).resolve(strict=True)
    shim = REPO_ROOT / "tools" / "env" / "videox_shim"
    project = REPO_ROOT / "projects" / "latent_action_models"
    for root in reversed((str(REPO_ROOT), str(project), str(shim), str(videox))):
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    os.environ["VIDEOX_HOME"] = str(videox)
    os.environ["WAN_DIR"] = str(wan)
    return videox, wan


def _atomic_npy(path: Path, value: "Any") -> None:
    import numpy as np

    if path.exists() or path.is_symlink():
        raise contract.CSIPContractError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise contract.CSIPContractError("latent shard temporary path exists")
    with temporary.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rank_zero_prepare(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, dict[str, Any] | None]:
    seal_record = None
    if args.split == "validation":
        if args.seal is None:
            raise contract.CSIPContractError(
                "validation latent extraction requires --seal"
            )
        from tools.csip_workflow import validate_seal

        # The seal validator reads only registered train records.  Validation
        # source records may be opened only after this fixed endpoint passes.
        seal, sealed_registration = validate_seal(args.seal)
        registration = contract.validate_registration(
            args.registration, open_validation=True
        )
        if sealed_registration["identity_sha256"] != registration["identity_sha256"]:
            raise contract.CSIPContractError("validation seal belongs to another study")
        if seal["registration"] != contract.registration_file_record(
            args.registration
        ):
            raise contract.CSIPContractError(
                "validation seal belongs to another registration file"
            )
        seal_record = {
            **contract.file_record(args.seal, "checkpoint seal"),
            "identity_sha256": seal["identity_sha256"],
        }
    else:
        if args.seal is not None:
            raise contract.CSIPContractError("train extraction must not accept a seal")
        registration = contract.validate_registration(
            args.registration, open_validation=False
        )
    output = Path(registration["planned_paths"][f"{args.split}_latent_root"])
    expected_metadata = Path(
        registration["planned_paths"][f"{args.split}_latent_metadata"]
    )
    if expected_metadata != output / "metadata.json":
        raise contract.CSIPContractError("registered latent paths are inconsistent")
    if output.exists() or output.is_symlink():
        raise contract.CSIPContractError("latent output root must be fresh")
    output.mkdir(parents=True, mode=0o700)
    return registration, output, seal_record


def command_extract(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise contract.CSIPContractError("Wan latent extraction requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise contract.CSIPContractError("launch extraction with torchrun")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    if dist.get_world_size() != contract.EXPECTED_WORLD_SIZE:
        raise contract.CSIPContractError("CSIP extraction requires exactly eight ranks")
    rank = dist.get_rank()
    prepared = None
    error = None
    if rank == 0:
        try:
            registration, output, seal_record = _rank_zero_prepare(args)
            prepared = {
                "registration": registration,
                "output": str(output),
                "seal_record": seal_record,
            }
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
    prepared = _broadcast_rank_zero(prepared, error)
    if not isinstance(prepared, dict):
        raise contract.CSIPContractError("latent extraction registration is absent")
    registration = prepared["registration"]
    output = Path(prepared["output"])
    contract.current_python_matches(registration["runtime"]["python"])
    videox, wan = _configure_runtime(registration)

    from robot_wm.modeling.tokenizers.rgb.wan_vae import WanVAETokenizer

    tokenizer = (
        WanVAETokenizer(
            model_path=str(wan),
            config_path=str(videox / "config" / "wan2.1" / "wan_civitai.yaml"),
        )
        .to(device=torch.device("cuda", local_rank))
        .eval()
    )
    source_key = "train" if args.split == "train" else "validation"
    source = registration["datasets"][source_key]
    rgb = np.load(source["arrays"]["rgb"]["path"], mmap_mode="r", allow_pickle=False)
    count = (
        contract.EXPECTED_TRAIN_CLIPS
        if args.split == "train"
        else contract.EXPECTED_VALIDATION_CLIPS
    )
    if tuple(rgb.shape) != (count, 13, 3, 180, 960) or str(rgb.dtype) != "float16":
        raise contract.CSIPContractError("source RGB array geometry differs")
    indexes = np.asarray(
        list(range(rank, count, contract.EXPECTED_WORLD_SIZE)), dtype=np.int64
    )
    expected_per_rank = count // contract.EXPECTED_WORLD_SIZE
    if indexes.shape != (expected_per_rank,):
        raise contract.CSIPContractError("rank latent assignment differs")
    full_values = np.empty((expected_per_rank, 16, 4, 24, 120), dtype=np.float16)
    history_values = np.empty((expected_per_rank, 16, 2, 24, 120), dtype=np.float16)
    causal_max = 0.0
    device = torch.device("cuda", local_rank)
    with torch.inference_mode():
        for local_index, source_index in enumerate(indexes.tolist()):
            clip = torch.from_numpy(np.array(rgb[source_index], copy=True)).unsqueeze(0)
            clip = clip.to(device=device, dtype=torch.float32, non_blocking=False)
            full = tokenizer.encode_temporal(clip.permute(0, 2, 1, 3, 4))
            history = tokenizer.encode_temporal(clip[:, :5].permute(0, 2, 1, 3, 4))
            if tuple(full.shape) != (1, 16, 4, 24, 120) or tuple(history.shape) != (
                1,
                16,
                2,
                24,
                120,
            ):
                raise contract.CSIPContractError("Wan latent geometry differs")
            difference = float((full[:, :, :2] - history).abs().max().item())
            causal_max = max(causal_max, difference)
            if difference > contract.CAUSAL_TOLERANCE:
                raise contract.CSIPContractError(
                    "full-clip history differs from independent history encoding"
                )
            full_values[local_index] = full[0].detach().half().cpu().numpy()
            history_values[local_index] = history[0].detach().half().cpu().numpy()

    paths = {
        "indexes": output / f"rank-{rank:02d}-indexes.npy",
        "full_latents": output / f"rank-{rank:02d}-full.npy",
        "history_latents": output / f"rank-{rank:02d}-history.npy",
    }
    _atomic_npy(paths["indexes"], indexes)
    _atomic_npy(paths["full_latents"], full_values)
    _atomic_npy(paths["history_latents"], history_values)
    shard = {
        "rank": rank,
        "source_indexes": indexes.tolist(),
        "causal_history_max_abs_observed": causal_max,
        **{
            name: contract.file_record(path, f"rank {rank} {name}")
            for name, path in paths.items()
        },
    }
    gathered: list[Any] = [None] * contract.EXPECTED_WORLD_SIZE
    dist.all_gather_object(gathered, shard)
    if rank == 0:
        shards = sorted(gathered, key=lambda value: int(value["rank"]))
        all_indexes = [
            int(index)
            for shard_value in shards
            for index in shard_value["source_indexes"]
        ]
        if sorted(all_indexes) != list(range(count)) or len(set(all_indexes)) != count:
            raise contract.CSIPContractError("latent shard coverage differs")
        maximum = max(
            float(value["causal_history_max_abs_observed"]) for value in shards
        )
        metadata = contract.with_identity(
            {
                "schema_version": contract.SCHEMA_VERSION,
                "kind": contract.CACHE_KIND,
                "created_at_utc": contract.now_utc(),
                "status": "complete",
                "split": "train" if args.split == "train" else "val",
                "clips": count,
                "world_size": contract.EXPECTED_WORLD_SIZE,
                "dtype": "float16",
                "full_latent_shape": [count, 16, 4, 24, 120],
                "history_latent_shape": [count, 16, 2, 24, 120],
                "source_manifest_sha256": source["manifest"]["sha256"],
                "source_rgb_sha256": source["arrays"]["rgb"]["sha256"],
                "source_rgb_full_sha256_verified_at_registration": source["arrays"][
                    "rgb"
                ]["full_sha256_verified"],
                "registration_identity_sha256": registration["identity_sha256"],
                "runtime": registration["runtime"],
                "history_encoding": "independent_five_frame_observed_only",
                "causal_history_check": "full_clip_first_two_tokens_equals_history_only",
                "causal_history_tolerance": contract.CAUSAL_TOLERANCE,
                "causal_history_max_abs_observed": maximum,
                "checkpoint_seal": prepared["seal_record"],
                "shards": shards,
                "protected_test_paths_accepted": False,
                "protected_test_clips_read": 0,
            }
        )
        metadata_path = output / "metadata.json"
        contract.exclusive_json(metadata_path, metadata)
        print(
            json.dumps(
                {
                    "metadata": str(metadata_path),
                    "identity_sha256": metadata["identity_sha256"],
                    "clips": count,
                    "causal_history_max_abs_observed": maximum,
                },
                sort_keys=True,
            )
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


def command_verify(args: argparse.Namespace) -> int:
    registration = contract.validate_registration(args.registration)
    expected_path = Path(registration["planned_paths"][f"{args.split}_latent_metadata"])
    if args.metadata.resolve() != expected_path:
        raise contract.CSIPContractError(
            "latent metadata path differs from registration"
        )
    payload = contract.validate_latent_cache(
        args.metadata, registration=registration, split=args.split
    )
    print(
        json.dumps(
            {
                "valid": True,
                "split": args.split,
                "clips": payload["clips"],
                "identity_sha256": payload["identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--registration", type=Path, required=True)
    extract.add_argument("--split", choices=("train", "validation"), required=True)
    extract.add_argument("--seal", type=Path)
    extract.set_defaults(func=command_extract)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--registration", type=Path, required=True)
    verify.add_argument("--split", choices=("train", "validation"), required=True)
    verify.add_argument("--metadata", type=Path, required=True)
    verify.set_defaults(func=command_verify)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
