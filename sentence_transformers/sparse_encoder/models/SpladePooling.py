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
        pooling_strategy: Either 'max' or 'sum' for SPLADE pooling

    """

    SPLADE_POOLING_MODES = ("sum", "max")

    def __init__(self, pooling_strategy: str = "max", word_embedding_dimension: int = None) -> None:
        super().__init__()
        self.pooling_strategy = pooling_strategy
        if pooling_strategy not in self.SPLADE_POOLING_MODES:
            raise ValueError("pooling_strategy must be either 'max' or 'sum'")
        self.config_keys = ["pooling_strategy", "word_embedding_dimension"]
        self.word_embedding_dimension = word_embedding_dimension  # This will be set in the forward method

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the model.
        Args:
            features: Dictionary containing input features with 'mlm_logits' key
        Returns:
            Dictionary containing SPLADE pooled embeddings
        """
        # Get the MLM head logits (shape: batch_size, seq_length, vocab_size)
        mlm_logits = features["token_embeddings"]

        # Apply ReLU and log transformation for SPLADE
        splade_scores = torch.log1p(torch.relu(mlm_logits))

        # Pool across sequence length dimension
        if self.pooling_strategy == "max":
            pooled_scores = torch.max(splade_scores, dim=1)[0]  # shape: batch_size, vocab_size
        else:  # sum
            pooled_scores = torch.sum(splade_scores, dim=1)  # shape: batch_size, vocab_size

        # Set the word embedding dimension
        if self.word_embedding_dimension is None:
            self.word_embedding_dimension = pooled_scores.shape[1]
        features["sentence_embedding"] = pooled_scores
        return features

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

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of the sentence embedding.

        Returns:
            int: Dimension of the sentence embedding
        """
        return self.word_embedding_dimension
