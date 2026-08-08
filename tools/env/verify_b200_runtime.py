#!/usr/bin/env python3
"""Fail-fast verification for the pinned lacwm B200 runtime.

The default mode verifies package and VideoX API compatibility without touching
CUDA. Pass --expected-gpus/--require-b200 on the training server to include GPU
and peer-access checks.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from packaging.utils import canonicalize_name


VIDEOX_COMMIT = "1d6d9c3e1540968466937129fef4b288041e06de"
VIDEOX_CONFIG_SHA256 = "21fe4409b664385a1c1cc5c23d92506ffb05ef3c374a18de9df67b715dca07e9"
WAN_MODEL_ID = "alibaba-pai/Wan2.1-Fun-1.3B-Control"
WAN_MODEL_REVISION = "ce96ebd52b1134d2c8a903ceb491ab27aa1e5b7c"
OFFICIAL_T5_SHA256 = "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"
TOKENIZER_SHA256 = {
    "special_tokens_map.json": "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d",
    "spiece.model": "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",
    "tokenizer.json": "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b",
    "tokenizer_config.json": "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05",
}
EXPECTED = {
    "torch": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "diffusers": "0.38.0",
    "accelerate": "1.6.0",
    "peft": "0.19.1",
    "safetensors": "0.8.0",
    "huggingface_hub": "0.36.0",
    "transformers": "4.57.6",
    "decord": "0.6.0",
}
EXPECTED_DISTRIBUTIONS = {
    "hydra-core": "1.3.2",
    "omegaconf": "2.3.0",
    "colorlog": "6.9.0",
    "einops": "0.8.1",
    "kornia": "0.8.2",
    "torchdata": "0.11.0",
    "wandb": "0.27.0",
    "numpy": "2.0.1",
    "scipy": "1.15.3",
    "pandas": "2.2.3",
    "pyarrow": "24.0.0",
    "h5py": "3.12.1",
    "opencv-python": "4.11.0.86",
    "av": "17.1.0",
    "imageio": "2.37.0",
    "imageio-ffmpeg": "0.6.0",
    "matplotlib": "3.10.3",
    "mcap": "1.4.0",
    "mcap-protobuf-support": "0.5.4",
    "lpips": "0.1.4",
    "pytest": "8.4.1",
    "timm": "1.0.19",
    "ftfy": "6.3.1",
    "librosa": "0.11.0",
    "sentencepiece": "0.2.1",
    "tokenizers": "0.22.2",
}
EXPECTED_WEIGHTS = {
    "config.json": {
        "bytes": 249,
        "sha256": "55779e6882d7c0918d8c289d61c4ba4693c1014b53c593f49362dd4f3baead49",
    },
    "diffusion_pytorch_model.safetensors": {
        "bytes": 3_129_105_448,
        "sha256": "9ff6289322b41bf187206eac2a57e85ce85c9ee5bfe8bc44eabeaeb86b44129a",
    },
    "Wan2.1_VAE.pth": {
        "bytes": 507_609_880,
        "sha256": "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
    },
}
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "requirements-b200-lock.txt"


def _version(module) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _git_status(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip()


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap_cfg_skip_forward(method):
    """Recover the decorated forward function from VideoX-Fun's closure."""
    required = {"x", "t", "context", "seq_len", "clip_fea", "y"}
    candidate = inspect.unwrap(method)
    if required.issubset(inspect.signature(candidate).parameters):
        return candidate
    for cell in method.__closure__ or ():
        value = cell.cell_contents
        if callable(value) and required.issubset(inspect.signature(value).parameters):
            return value
    return candidate


def _verify_full_environment() -> dict[str, object]:
    expected: dict[str, str] = {}
    for raw in LOCK_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if "==" not in line:
            raise RuntimeError(f"unresolved requirement in {LOCK_PATH}: {line}")
        name, version = line.split("==", 1)
        expected[canonicalize_name(name)] = version

    installed = {
        canonicalize_name(dist.metadata["Name"]): dist.version
        for dist in metadata.distributions()
        if dist.metadata.get("Name")
    }
    # The repository itself is installed editable and is bound separately by
    # its clean Git commit. Every third-party distribution must match the lock.
    installed.pop(canonicalize_name("robot_wm"), None)
    missing = sorted(set(expected) - set(installed))
    extra = sorted(set(installed) - set(expected))
    mismatched = {
        name: {"expected": expected[name], "actual": installed[name]}
        for name in sorted(set(expected) & set(installed))
        if expected[name] != installed[name]
    }
    if missing or extra or mismatched:
        raise RuntimeError(
            "full environment differs from resolved lock: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )
    canonical_inventory = json.dumps(
        installed, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "lock_path": str(LOCK_PATH),
        "lock_sha256": _sha256(LOCK_PATH),
        "inventory_sha256": hashlib.sha256(canonical_inventory).hexdigest(),
        "distribution_count": len(installed),
        "sys_executable": sys.executable,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-gpus", type=int, default=0)
    parser.add_argument("--require-b200", action="store_true")
    parser.add_argument("--wan-dir", type=Path)
    args = parser.parse_args()

    if sys.version_info[:3] != (3, 10, 20):
        raise RuntimeError(
            f"expected Python 3.10.20, got {sys.version.split()[0]}"
        )
    expected_python = os.environ.get("LACWM_PYTHON")
    if (
        not expected_python
        or Path(expected_python).resolve(strict=True)
        != Path(sys.executable).resolve(strict=True)
    ):
        raise RuntimeError(
            "LACWM_PYTHON does not resolve to the running interpreter: "
            f"configured={expected_python!r}, running={sys.executable!r}"
        )
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE must be exactly 1")

    import accelerate
    import decord
    import diffusers
    import huggingface_hub
    import peft
    import safetensors
    import torch
    import torchvision
    import transformers

    modules = {
        "torch": torch,
        "torchvision": torchvision,
        "diffusers": diffusers,
        "accelerate": accelerate,
        "peft": peft,
        "safetensors": safetensors,
        "huggingface_hub": huggingface_hub,
        "transformers": transformers,
        "decord": decord,
    }
    versions = {name: _version(module) for name, module in modules.items()}
    mismatches = {
        name: {"expected": EXPECTED[name], "actual": actual}
        for name, actual in versions.items()
        if actual != EXPECTED[name]
    }
    if mismatches:
        raise RuntimeError(f"package version mismatch: {mismatches}")
    distribution_versions = {
        name: metadata.version(name) for name in EXPECTED_DISTRIBUTIONS
    }
    distribution_mismatches = {
        name: {"expected": EXPECTED_DISTRIBUTIONS[name], "actual": actual}
        for name, actual in distribution_versions.items()
        if actual != EXPECTED_DISTRIBUTIONS[name]
    }
    if distribution_mismatches:
        raise RuntimeError(
            f"package distribution version mismatch: {distribution_mismatches}"
        )
    environment_status = _verify_full_environment()
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"expected CUDA runtime 12.8, got {torch.version.cuda}")

    videox_value = os.environ.get("VIDEOX_HOME")
    if not videox_value:
        raise RuntimeError("VIDEOX_HOME is unset")
    videox_home = Path(videox_value)
    if not videox_home.is_dir():
        raise RuntimeError("VIDEOX_HOME does not identify the pinned checkout")
    actual_commit = _git_head(videox_home)
    if actual_commit != VIDEOX_COMMIT:
        raise RuntimeError(
            f"VideoX-Fun commit mismatch: {actual_commit} != {VIDEOX_COMMIT}"
        )
    videox_status = _git_status(videox_home)
    if videox_status:
        raise RuntimeError(f"pinned VideoX-Fun checkout is dirty: {videox_status}")

    expected_pythonpath = [
        REPO_ROOT / "tools/env/videox_shim",
        videox_home,
        REPO_ROOT / "projects/latent_action_models",
        REPO_ROOT,
    ]
    observed_pythonpath = [
        Path(value).resolve(strict=True)
        for value in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if value
    ]
    if observed_pythonpath != [path.resolve(strict=True) for path in expected_pythonpath]:
        raise RuntimeError(
            "PYTHONPATH is not the exact source-bound runtime prefix: "
            f"observed={observed_pythonpath}, expected={expected_pythonpath}"
        )

    from videox_fun.models.wan_transformer3d import WanTransformer3DModel
    from videox_fun.models.wan_vae import AutoencoderKLWan
    import lam
    import robot_wm

    source_paths = {
        "robot_wm": str(Path(robot_wm.__file__).resolve(strict=True)),
        "lam": str(Path(lam.__file__).resolve(strict=True)),
        "wan_transformer": str(
            Path(inspect.getfile(WanTransformer3DModel)).resolve(strict=True)
        ),
        "wan_vae": str(Path(inspect.getfile(AutoencoderKLWan)).resolve(strict=True)),
    }
    expected_source_roots = {
        "robot_wm": REPO_ROOT,
        "lam": REPO_ROOT / "projects/latent_action_models",
        "wan_transformer": videox_home,
        "wan_vae": videox_home,
    }
    for name, value in source_paths.items():
        path = Path(value)
        root = expected_source_roots[name].resolve(strict=True)
        if path != root and root not in path.parents:
            raise RuntimeError(f"{name} imported outside its bound source root: {path}")

    load_sig = inspect.signature(WanTransformer3DModel.from_pretrained)
    required_load = {
        "pretrained_model_path",
        "transformer_additional_kwargs",
        "low_cpu_mem_usage",
        "torch_dtype",
    }
    if not required_load.issubset(load_sig.parameters):
        raise RuntimeError(f"unexpected Wan loader API: {load_sig}")

    forward = _unwrap_cfg_skip_forward(WanTransformer3DModel.forward)
    forward_sig = inspect.signature(forward)
    required_forward = {"x", "t", "context", "seq_len", "clip_fea", "y"}
    if not required_forward.issubset(forward_sig.parameters):
        raise RuntimeError(f"unexpected Wan forward API: {forward_sig}")

    vae_load_sig = inspect.signature(AutoencoderKLWan.from_pretrained)
    if "additional_kwargs" not in vae_load_sig.parameters:
        raise RuntimeError(f"unexpected Wan VAE loader API: {vae_load_sig}")

    # Checkpoint validation is opt-in. activate_b200.sh exports the planned
    # WAN_DIR even before weights are downloaded, while API-only verification
    # must remain usable during environment construction.
    wan_dir = args.wan_dir
    weight_status = None
    if wan_dir is not None:
        required_weights = [wan_dir / "null_prompt_umt5.pt"]
        required_weights.extend(wan_dir / name for name in EXPECTED_WEIGHTS)
        missing = [str(path) for path in required_weights if not path.is_file()]
        if missing:
            raise RuntimeError(f"missing Wan files: {missing}")

        verified = {}
        for name, expected in EXPECTED_WEIGHTS.items():
            path = wan_dir / name
            size = path.stat().st_size
            if size != expected["bytes"]:
                raise RuntimeError(
                    f"unexpected size for {path}: {size} != {expected['bytes']}"
                )
            actual_hash = _sha256(path)
            if actual_hash != expected["sha256"]:
                raise RuntimeError(
                    f"unexpected SHA-256 for {path}: {actual_hash} != {expected['sha256']}"
                )
            verified[name] = {"bytes": size, "sha256": actual_hash}

        null_prompt_path = wan_dir / "null_prompt_umt5.pt"
        null_prompt = torch.load(null_prompt_path, map_location="cpu", weights_only=True)
        if not isinstance(null_prompt, torch.Tensor):
            raise RuntimeError(f"null prompt is not a tensor: {null_prompt_path}")
        if tuple(null_prompt.shape) != (1, 4096) or null_prompt.dtype != torch.float32:
            raise RuntimeError(
                "unexpected null-prompt tensor: "
                f"shape={tuple(null_prompt.shape)}, dtype={null_prompt.dtype}"
            )
        if not torch.isfinite(null_prompt).all():
            raise RuntimeError("null-prompt tensor contains non-finite values")
        null_prompt_hash = _sha256(null_prompt_path)
        metadata_path = null_prompt_path.with_suffix(null_prompt_path.suffix + ".json")
        try:
            null_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"missing or invalid null-prompt provenance {metadata_path}: {exc}"
            ) from exc
        expected_metadata = {
            "schema_version": 1,
            "model_id": WAN_MODEL_ID,
            "model_revision": WAN_MODEL_REVISION,
            "output_sha256": null_prompt_hash,
            "shape": [1, 4096],
            "dtype": "torch.float32",
            "valid_tokens": 1,
            "text_encoder_sha256": OFFICIAL_T5_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "videox_commit": VIDEOX_COMMIT,
            "videox_config_sha256": VIDEOX_CONFIG_SHA256,
        }
        metadata_mismatches = {
            key: {"expected": value, "actual": null_metadata.get(key)}
            for key, value in expected_metadata.items()
            if null_metadata.get(key) != value
        }
        if metadata_mismatches:
            raise RuntimeError(
                f"null-prompt provenance mismatch in {metadata_path}: "
                f"{metadata_mismatches}"
            )
        weight_status = {
            "root": str(wan_dir),
            "verified": verified,
            "null_prompt": {
                "shape": list(null_prompt.shape),
                "dtype": str(null_prompt.dtype),
                "finite": True,
                "sha256": null_prompt_hash,
                "provenance": str(metadata_path),
            },
        }

    gpu_status = None
    if args.expected_gpus or args.require_b200:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        count = torch.cuda.device_count()
        if args.expected_gpus and count != args.expected_gpus:
            raise RuntimeError(f"expected {args.expected_gpus} GPUs, found {count}")
        devices = []
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            if args.require_b200 and capability != (10, 0):
                raise RuntimeError(
                    f"GPU {index} is {props.name} capability {capability}, not B200 sm_100"
                )
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(f"GPU {index} does not report BF16 support")
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "capability": capability,
                    "memory_bytes": props.total_memory,
                }
            )
        inaccessible = [
            [src, dst]
            for src in range(count)
            for dst in range(count)
            if src != dst and not torch.cuda.can_device_access_peer(src, dst)
        ]
        gpu_status = {
            "count": count,
            "devices": devices,
            "nccl_available": torch.distributed.is_nccl_available(),
            "inaccessible_peer_pairs": inaccessible,
        }
        if not gpu_status["nccl_available"]:
            raise RuntimeError("PyTorch NCCL backend is unavailable")
        if inaccessible:
            raise RuntimeError(f"GPU peer access is unavailable for pairs: {inaccessible}")

    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "packages": versions,
                "distributions": distribution_versions,
                "environment": environment_status,
                "python_no_user_site": True,
                "source_paths": source_paths,
                "videox_commit": actual_commit,
                "videox_status": "clean",
                "wan_loader": str(load_sig),
                "wan_forward": str(forward_sig),
                "vae_loader": str(vae_load_sig),
                "weights": weight_status,
                "gpus": gpu_status,
            },
            indent=2,
            default=list,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
