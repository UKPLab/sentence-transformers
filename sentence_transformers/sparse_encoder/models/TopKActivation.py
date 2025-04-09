from __future__ import annotations

import torch
import torch.nn as nn


class TopKActivation(nn.Module):
    """
    TopK activation function for Sparse AutoEncoder.

    This module implements the TopK activation function.

    The TopK activation function keeps only the k largest values and sets the rest to zero.
    """

    def __init__(self, k: int = 100):
        """
        Initialize the TopK activation function.

        Args:
            k: Number of top values to keep
        """
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TopK activation function.

        Args:
            x: Input tensor

        Returns:
            Tensor with only the k largest values, rest set to zero
        """
        # Get the k largest values and their indices
        values, indices = torch.topk(x, k=self.k, dim=-1)

        # Create a mask of zeros with the same shape as x
        mask = torch.zeros_like(x)

        # Set the k largest values to 1 in the mask
        for i in range(self.k):
            mask.scatter_(-1, indices[..., i : i + 1], 1)

        # Apply the mask to the input tensor
        return x * mask
