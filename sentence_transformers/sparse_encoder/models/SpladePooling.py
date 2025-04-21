from __future__ import annotations

import json
import os
from typing import Any

import torch
from torch import nn


class SpladePooling(nn.Module):
    """SPLADE pooling layer that aggregates MLM logits using max or sum pooling.

    This pooling layer takes MLM logits (shape: batch_size, seq_length, vocab_size)
    and applies SPLADE transformation (ReLU + log) followed by pooling across the
    sequence length dimension.

    Args:
        word_embedding_dimension: Dimension of the word embeddings (vocab size)
        pooling_strategy: Either 'max' or 'sum' for SPLADE pooling

    """

    SPLADE_POOLING_MODES = ("sum", "max")

    def __init__(self, pooling_strategy: str = "max") -> None:
        super().__init__()
        self.pooling_strategy = pooling_strategy
        if pooling_strategy not in self.SPLADE_POOLING_MODES:
            raise ValueError("pooling_strategy must be either 'max' or 'sum'")
        self.config_keys = ["pooling_strategy"]

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """FForward pass of the model.
        Args:
            features: Dictionary containing input features with 'mlm_logits' key
        Returns:
            Dictionary containing SPLADE pooled embeddings
        """
        # Get the MLM head logits (shape: batch_size, seq_length, vocab_size)
        mlm_logits = features["mlm_logits"]

        # Apply ReLU and log transformation for SPLADE
        splade_scores = torch.log1p(torch.relu(mlm_logits))

        # Pool across sequence length dimension
        if self.pooling_strategy == "max":
            pooled_scores = torch.max(splade_scores, dim=1)[0]  # shape: batch_size, vocab_size
        else:  # sum
            pooled_scores = torch.sum(splade_scores, dim=1)  # shape: batch_size, vocab_size

        return {"sentence_embedding": pooled_scores}

    def get_config_dict(self) -> dict[str, Any]:
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path) -> None:
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path) -> SpladePooling:
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        return SpladePooling(**config)

    def __repr__(self) -> str:
        return f"SpladePooling({self.get_config_dict()})"
