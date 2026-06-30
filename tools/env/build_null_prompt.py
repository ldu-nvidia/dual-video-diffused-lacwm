#!/usr/bin/env python3
"""Build the cached empty-prompt umT5 embedding used by lacwm training.

The official Wan repository does not ship ``null_prompt_umt5.pt``.  This tool
computes it once with the official tokenizer and text-encoder checkpoint, then
writes it atomically.  Training subsequently loads only the small cached tensor
and does not keep the 11 GB text encoder in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer


OFFICIAL_T5_SHA256 = "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"
MODEL_ID = "alibaba-pai/Wan2.1-Fun-1.3B-Control"
MODEL_REVISION = "ce96ebd52b1134d2c8a903ceb491ab27aa1e5b7c"
VIDEOX_COMMIT = "1d6d9c3e1540968466937129fef4b288041e06de"
VIDEOX_CONFIG_SHA256 = "21fe4409b664385a1c1cc5c23d92506ffb05ef3c374a18de9df67b715dca07e9"
TOKENIZER_SHA256 = {
    "special_tokens_map.json": "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d",
    "spiece.model": "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",
    "tokenizer.json": "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b",
    "tokenizer_config.json": "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05",
}


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-dir", type=Path, required=True)
    parser.add_argument("--videox-home", type=Path, required=True)
    parser.add_argument(
        "--text-encoder",
        type=Path,
        help="Defaults to WAN_DIR/models_t5_umt5-xxl-enc-bf16.pth",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    text_encoder_path = args.text_encoder or (
        args.wan_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    )
    tokenizer_path = args.wan_dir / "google" / "umt5-xxl"
    config_path = args.videox_home / "config" / "wan2.1" / "wan_civitai.yaml"
    output_path = args.output or (args.wan_dir / "null_prompt_umt5.pt")

    for path in (text_encoder_path, tokenizer_path, config_path):
        if not path.exists():
            raise FileNotFoundError(path)
    actual_config_hash = sha256(config_path)
    if actual_config_hash != VIDEOX_CONFIG_SHA256:
        raise RuntimeError(
            f"unexpected VideoX Wan config SHA-256: {actual_config_hash}; "
            f"expected {VIDEOX_CONFIG_SHA256}"
        )

    actual_hash = sha256(text_encoder_path)
    if actual_hash != OFFICIAL_T5_SHA256:
        raise RuntimeError(
            f"unexpected text-encoder SHA-256: {actual_hash}; "
            f"expected {OFFICIAL_T5_SHA256}"
        )
    tokenizer_hashes = {
        name: sha256(tokenizer_path / name) for name in TOKENIZER_SHA256
    }
    if tokenizer_hashes != TOKENIZER_SHA256:
        raise RuntimeError(
            f"unexpected tokenizer SHA-256 values: {tokenizer_hashes}; "
            f"expected {TOKENIZER_SHA256}"
        )

    from videox_fun.models.wan_text_encoder import WanT5EncoderModel

    config = OmegaConf.load(config_path)
    kwargs = OmegaConf.to_container(config["text_encoder_kwargs"], resolve=True)
    max_length = int(kwargs["text_length"])
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokens = tokenizer(
        [""],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    valid_tokens = int(tokens.attention_mask[0].sum().item())
    if valid_tokens <= 0:
        raise RuntimeError("empty prompt unexpectedly produced no valid tokens")

    model = WanT5EncoderModel.from_pretrained(
        str(text_encoder_path),
        additional_kwargs=kwargs,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    device = torch.device(args.device)
    model.to(device)
    with torch.inference_mode():
        encoded = model(
            tokens.input_ids.to(device),
            attention_mask=tokens.attention_mask.to(device),
        )[0]
    null_prompt = encoded[0, :valid_tokens].float().cpu().contiguous()
    if null_prompt.ndim != 2 or null_prompt.shape[1] != int(kwargs["dim"]):
        raise RuntimeError(f"unexpected null-prompt shape: {tuple(null_prompt.shape)}")
    if not torch.isfinite(null_prompt).all():
        raise RuntimeError("null-prompt embedding contains non-finite values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(null_prompt, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    metadata = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "shape": list(null_prompt.shape),
        "dtype": str(null_prompt.dtype),
        "valid_tokens": valid_tokens,
        "text_encoder": str(text_encoder_path),
        "text_encoder_sha256": OFFICIAL_T5_SHA256,
        "tokenizer_sha256": tokenizer_hashes,
        "videox_commit": VIDEOX_COMMIT,
        "videox_config_sha256": actual_config_hash,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with metadata_temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(metadata_temporary, metadata_path)
    directory_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
