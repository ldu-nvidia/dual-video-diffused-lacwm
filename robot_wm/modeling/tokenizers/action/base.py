from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ActionTokenizer(ABC, nn.Module):
    def __init__(self, chunk_decoded_shape: tuple[int, ...] = (-1, -1), chunk_encoded_shape: tuple[int, ...] = (-1, -1)):
        super().__init__()
        self._chunk_decoded_shape = chunk_decoded_shape
        self._chunk_encoded_shape = chunk_encoded_shape

    @property
    @abstractmethod
    def dim(self) -> int:
        """Returns the output channel size."""

    @property
    def chunk_decoded_shape(self) -> tuple[int, ...]:
        """Returns the shape of the decoded action tokens. (e.g. (5, 7) for a proprioception chunk) """
        return self._chunk_decoded_shape

    @property
    def chunk_encoded_shape(self) -> tuple[int, ...]:
        """Returns the shape of the encoded action tokens. (e.g. (5, D) for a proprioception chunk) """
        return self._chunk_encoded_shape

    @abstractmethod
    def init_weights(self):
        """Initializes tokenizer weights"""

    @abstractmethod
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: torch.Tensor [N, T, *, D]
        Returns:
            torch.Tensor [N, T, *, d]
        """
