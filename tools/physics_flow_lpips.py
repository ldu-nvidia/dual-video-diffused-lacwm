#!/usr/bin/env python3
"""Offline, content-bound AlexNet LPIPS instrument for physics-flow Stage-1.

The torchvision AlexNet checkpoint and LPIPS linear weights must already be
present on local storage.  This module never downloads an evaluation asset.
It returns a stable receipt over the installed implementation, every packaged
LPIPS Python/weight file, the cached AlexNet checkpoint, and the exact loaded
state dictionary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import sys
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class OfflineLPIPSError(RuntimeError):
    """The registered LPIPS implementation or offline asset is unavailable."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise OfflineLPIPSError(f"regular absolute LPIPS asset required: {path}")
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _identity(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return {
        **unsigned,
        "identity_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def identity_valid(payload: dict[str, Any]) -> bool:
    observed = payload.get("identity_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        return False
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest() == observed


def _state_dict_sha256(state: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    import torch

    digest = hashlib.sha256()
    schema = []
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise OfflineLPIPSError(f"LPIPS state contains a non-tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        record = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        schema.append(record)
        digest.update(_canonical_json(record))
        digest.update(b"\0")
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest(), schema


def _implementation_inventory(package_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(package_root.rglob("*")):
        if (
            path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".pth"}
        ):
            records.append(
                {
                    "relative_path": str(path.relative_to(package_root)),
                    **_file_record(path),
                }
            )
    if not records or not any(record["relative_path"].endswith("alex.pth") for record in records):
        raise OfflineLPIPSError("installed LPIPS package lacks packaged AlexNet weights")
    return records


def _offline_assets() -> tuple[Any, dict[str, Any]]:
    import lpips
    import torch
    import torchvision
    from torchvision.models import AlexNet_Weights

    package_file = Path(lpips.__file__).resolve(strict=True)
    package_root = package_file.parent
    weights = AlexNet_Weights.IMAGENET1K_V1
    filename = Path(urlparse(weights.url).path).name
    if not filename:
        raise OfflineLPIPSError("torchvision AlexNet weight URL has no filename")
    checkpoint = Path(torch.hub.get_dir()).expanduser().resolve(strict=False) / "checkpoints" / filename
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise OfflineLPIPSError(
            "offline AlexNet checkpoint is absent; pre-stage it before registration: "
            f"{checkpoint}"
        )
    checkpoint_before = _file_record(checkpoint)
    implementation = _implementation_inventory(package_root)

    # Guard the Python network entry points used by torch/torchvision download
    # helpers.  A locally cached load must not call either function.
    original_urlopen = urllib.request.urlopen
    original_create_connection = socket.create_connection
    original_download = torch.hub.download_url_to_file

    def deny_network(*_args: Any, **_kwargs: Any) -> Any:
        raise OfflineLPIPSError("network access is forbidden for LPIPS evaluation")

    urllib.request.urlopen = deny_network
    socket.create_connection = deny_network
    torch.hub.download_url_to_file = deny_network
    try:
        model = lpips.LPIPS(net="alex", pretrained=True, pnet_rand=False, verbose=False)
    finally:
        urllib.request.urlopen = original_urlopen
        socket.create_connection = original_create_connection
        torch.hub.download_url_to_file = original_download

    checkpoint_after = _file_record(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise OfflineLPIPSError("AlexNet checkpoint changed during offline construction")
    state_digest, state_schema = _state_dict_sha256(dict(model.state_dict()))
    receipt = _identity(
        {
            "kind": "raw_physics_flow_offline_lpips_alex_v1",
            "network": "alex",
            "lpips_constructor": {
                "pretrained": True,
                "pnet_rand": False,
                "spatial": False,
                "version": "0.1",
            },
            "versions": {
                "python": sys.version,
                "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__),
                "lpips_distribution": importlib.metadata.version("lpips"),
            },
            "module": _file_record(package_file),
            "implementation_and_packaged_weights": implementation,
            "torchvision_alexnet": {
                "weights_enum": "AlexNet_Weights.IMAGENET1K_V1",
                "url_filename": filename,
                "checkpoint": checkpoint_after,
            },
            "loaded_state_dict_sha256": state_digest,
            "loaded_state_schema": state_schema,
            "loaded_state_tensor_count": len(state_schema),
            "network_access_permitted": False,
            "download_performed": False,
            "protected_test_accessed": False,
        }
    )
    return model, receipt


def load_offline_lpips(*, device: Any | None = None) -> tuple[Any, dict[str, Any]]:
    """Construct the frozen scorer and optionally move it to ``device``."""

    model, receipt = _offline_assets()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if device is not None:
        model = model.to(device=device)
    return model, receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-only", action="store_true", required=True)
    parser.parse_args(argv)
    try:
        _model, receipt = load_offline_lpips()
    except OfflineLPIPSError as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
