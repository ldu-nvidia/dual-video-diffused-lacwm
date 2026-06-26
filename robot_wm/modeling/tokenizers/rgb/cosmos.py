# Wrapper for the cosmos tokenizer.

# References:
#     https://github.com/NVIDIA/Cosmos/tree/b867572b99d08f450ddb8bcd6661d8c35bf6b967/cosmos1/models/tokenizer

import os
import sys
import warnings
from pathlib import Path
from typing import Sequence, Union

import torch
from einops import rearrange

from robot_wm.modeling.tokenizers.rgb.base import RGBTokenizer
from robot_wm.modeling.tokenizers.rgb.transforms import DynamicPad

COSMOS_HOME = os.environ.get("COSMOS_HOME", None)
if COSMOS_HOME is None:
    msg = "could not load cosmos; set COSMOS_HOME to the root of the cosmos repo"
    raise ImportError(msg)
sys.path.insert(1, COSMOS_HOME)
from cosmos1.models.tokenizer.inference.image_lib import ImageTokenizer  # type: ignore
from cosmos1.models.tokenizer.inference.video_lib import (  # type: ignore
    CausalVideoTokenizer,
)

COSMOS_IMAGE_TOKENIZER_NAMES = [
    "Cosmos-0.1-Tokenizer-CI8x8",
    "Cosmos-0.1-Tokenizer-CI16x16",
    "Cosmos-0.1-Tokenizer-DI8x8",
    "Cosmos-0.1-Tokenizer-DI16x16",
]
COSMOS_IMAGE_TOKENIZER_SCALING_FACTORS = {
    "Cosmos-0.1-Tokenizer-CI8x8": (8, 8),
    "Cosmos-0.1-Tokenizer-CI16x16": (16, 16),
    "Cosmos-0.1-Tokenizer-DI8x8": (8, 8),
    "Cosmos-0.1-Tokenizer-DI16x16": (16, 16),
}
COSMOS_VIDEO_TOKENIZER_NAMES = [
    "Cosmos-0.1-Tokenizer-CV4x8x8",
    "Cosmos-0.1-Tokenizer-CV8x8x8",
    "Cosmos-0.1-Tokenizer-CV8x16x16",
    "Cosmos-0.1-Tokenizer-DV4x8x8",
    "Cosmos-0.1-Tokenizer-DV8x8x8",
    "Cosmos-0.1-Tokenizer-DV8x16x16",
    "Cosmos-1.0-Tokenizer-CV8x8x8",
    "Cosmos-1.0-Tokenizer-DV8x16x16",
]
COSMOS_VIDEO_TOKENIZER_SCALING_FACTORS = {
    "Cosmos-0.1-Tokenizer-CV4x8x8": (4, 8, 8),
    "Cosmos-0.1-Tokenizer-CV8x8x8": (8, 8, 8),
    "Cosmos-0.1-Tokenizer-CV8x16x16": (8, 16, 16),
    "Cosmos-0.1-Tokenizer-DV4x8x8": (4, 8, 8),
    "Cosmos-0.1-Tokenizer-DV8x8x8": (8, 8, 8),
    "Cosmos-0.1-Tokenizer-DV8x16x16": (8, 16, 16),
    "Cosmos-1.0-Tokenizer-CV8x8x8": (8, 8, 8),
    "Cosmos-1.0-Tokenizer-DV8x16x16": (8, 16, 16),
}
COSMOS_TOKENIZER_TYPES = ["autoencoder", "encoder", "decoder", "both"]


def _get_checkpoint_path(tokenizer_name: str, tokenizer_type: str) -> Path:
    return Path(COSMOS_HOME) / "checkpoints" / tokenizer_name / f"{tokenizer_type}.jit"


def _get_kwargs(tokenizer_name: str, tokenizer_type: str) -> dict[str, str]:
    assert tokenizer_type in COSMOS_TOKENIZER_TYPES
    kwargs = {}
    if tokenizer_type == "autoencoder":
        kwargs["checkpoint"] = _get_checkpoint_path(tokenizer_name, tokenizer_type)
    elif tokenizer_type == "encoder":
        kwargs["checkpoint_enc"] = _get_checkpoint_path(tokenizer_name, tokenizer_type)
    elif tokenizer_type == "decoder":
        kwargs["checkpoint_dec"] = _get_checkpoint_path(tokenizer_name, tokenizer_type)
    elif tokenizer_type == "both":
        kwargs["checkpoint_enc"] = _get_checkpoint_path(tokenizer_name, "encoder")
        kwargs["checkpoint_dec"] = _get_checkpoint_path(tokenizer_name, "decoder")
    else:
        raise ValueError(f"invalid tokenizer_type: {tokenizer_type}")
    return kwargs


def _load_image_tokenizer(
    tokenizer_name: str, tokenizer_type: str = "both"
) -> Union[ImageTokenizer, CausalVideoTokenizer]:
    assert tokenizer_name in COSMOS_IMAGE_TOKENIZER_NAMES
    kwargs = _get_kwargs(tokenizer_name, tokenizer_type)
    return ImageTokenizer(**kwargs)


def _load_video_tokenizer(
    tokenizer_name: str, tokenizer_type: str = "both"
) -> Union[ImageTokenizer, CausalVideoTokenizer]:
    assert tokenizer_name in COSMOS_VIDEO_TOKENIZER_NAMES
    kwargs = _get_kwargs(tokenizer_name, tokenizer_type)
    return CausalVideoTokenizer(**kwargs)


class CosmosImageTokenizer(RGBTokenizer):
    def __init__(
        self,
        tokenizer_name: str,
        tokenizer_type: str,
        output_dim: int,
        input_range: str = "normal",
        input_mean: Sequence[float] = [0.485, 0.456, 0.406],
        input_std: Sequence[float] = [0.229, 0.224, 0.225],
        chunk_decoded_shape: Sequence[int] = (3, -1, -1),
        chunk_encoded_shape: Sequence[int] = (-1, -1, -1),
    ):
        super().__init__(
            output_dim=output_dim,
            input_range=input_range,
            input_mean=input_mean,
            input_std=input_std,
            chunk_decoded_shape=chunk_decoded_shape,
            chunk_encoded_shape=chunk_encoded_shape,
        )
        assert output_dim == (16 if "CI" in tokenizer_name else 6)
        self.tokenizer_name = tokenizer_name
        self.tokenizer_type = tokenizer_type

        self.model = _load_image_tokenizer(self.tokenizer_name, self.tokenizer_type)
        self.padding = DynamicPad(factors=self.scale_factors)

    @property
    def scale_factors(self) -> Sequence[int]:
        return COSMOS_IMAGE_TOKENIZER_SCALING_FACTORS[self.tokenizer_name]

    @property
    def _encode_idx(self) -> int:
        return 0 if "CI" in self.tokenizer_name else 1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [N, *, C, H, W]
        Returns:
            torch.Tensor [N, *, d, h, w]
        """
        x = self.padding(x)
        orig_shape = x.shape
        if len(orig_shape) == 4:
            x = x.unsqueeze(1)
        N, T, C, H, W = x.shape
        x = x.reshape(N * T, C, H, W)
        z = self.model.encode(x)[self._encode_idx]
        _, d, h, w = z.shape
        z = z.view(N, T, d, h, w)
        if len(orig_shape) == 4:
            z = z.squeeze(1)
        return z

    def decode(self, z: torch.Tensor, out_shape: Sequence[int]) -> torch.Tensor:
        """
        Args:
            z: torch.Tensor [N, *, d, h, w]
            out_shape: Sequence[int] [N, *, C, H, W]
        Returns:
            torch.Tensor [N, *, C, H, W]
        """
        orig_shape = z.shape
        if len(orig_shape) == 4:
            z = z.unsqueeze(1)
        N, T, d, h, w = z.shape
        z = z.reshape(N * T, d, h, w)
        if "CI" in self.tokenizer_name:
            xhat = self.model.decode(z)
        else:
            warnings.warn("Unclear if cosmos quantizer is being called correctly.")
            indices = self.model._enc_model.quantizer(z)[0]
            xhat = self.model.decode(indices)
        _, C, H, W = xhat.shape
        xhat = xhat.view(N, T, C, H, W)
        if len(orig_shape) == 4:
            xhat = xhat.squeeze(1)
        xhat = self.padding.reverse(xhat=xhat, out_shape=out_shape)
        return xhat


class CosmosVideoTokenizer(RGBTokenizer):
    def __init__(self, tokenizer_name: str, tokenizer_type: str, output_dim: int):
        super().__init__(output_dim=output_dim)
        assert output_dim == (16 if "CV" in tokenizer_name else 6)
        self.tokenizer_name = tokenizer_name
        self.tokenizer_type = tokenizer_type

        self.model = _load_video_tokenizer(self.tokenizer_name, self.tokenizer_type)
        self.padding = DynamicPad(factors=self.scale_factors)

    @property
    def scale_factors(self) -> Sequence[int]:
        return COSMOS_VIDEO_TOKENIZER_SCALING_FACTORS[self.tokenizer_name]

    @property
    def _encode_idx(self) -> int:
        return 0 if "CV" in self.tokenizer_name else 1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [N, *, C, H, W]
        Returns:
            torch.Tensor [N, *, d, h, w]
        """
        x = self.padding(x)
        orig_shape = x.shape
        if len(orig_shape) == 4:
            x = x.unsqueeze(1)
        x = rearrange(x, "N T C H W -> N C T H W")
        z = self.model.encode(x)[self._encode_idx]
        z = rearrange(z, "N d t h w -> N t d h w")
        if len(orig_shape) == 4:
            z = z.squeeze(1)
        return z

    def decode(self, z: torch.Tensor, out_shape: Sequence[int]) -> torch.Tensor:
        """
        Args:
            z: torch.Tensor [N, *, d, h, w]
            out_shape: Sequence[int] [N, *, C, H, W]
        Returns:
            torch.Tensor [N, *, C, H, W]
        """
        orig_shape = z.shape
        if len(orig_shape) == 4:
            z = z.unsqueeze(1)
        z = rearrange(z, "N t d h w -> N d t h w")
        if "CV" in self.tokenizer_name:
            xhat = self.model.decode(z)
        else:
            warnings.warn("Unclear if cosmos quantizer is being called correctly.")
            indices = self.model._enc_model.quantizer(z)[0]
            xhat = self.model.decode(indices)
        xhat = rearrange(xhat, "N C T H W -> N T C H W")
        if len(orig_shape) == 4:
            xhat = xhat.squeeze(1)
        xhat = self.padding.reverse(xhat=xhat, out_shape=out_shape)
        return xhat


# usage example
if __name__ == "__main__":
    N, C, T, H, W = 1, 3, 10, 180, 320
    rgb = torch.randn(N, T, C, H, W).to("cuda").to(torch.bfloat16)
    print(f"{rgb.shape = }")

    name = "Cosmos-0.1-Tokenizer-CI8x8"
    model = CosmosImageTokenizer(name).to("cuda").to(torch.bfloat16)
    # name = "Cosmos-1.0-Tokenizer-CV8x8x8"
    # model = CosmosVideoTokenizer(name).to("cuda").to(torch.bfloat16)
    model_size = sum([p.numel() for p in model.parameters()])
    print(f"{model_size = :,}")

    z = model.encode(rgb)
    print(f"{z.shape = }")

    rgb_hat = model.decode(z, rgb.shape)
    print(f"{rgb_hat.shape = }")
