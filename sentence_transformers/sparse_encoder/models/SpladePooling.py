from __future__ import annotations

import torch
from torch import nn

# TODO: SAVING LOADING with config.json


class SpladePooling(nn.Module):
    """SPLADE pooling layer that aggregates MLM logits using max or sum pooling.

    This pooling layer takes MLM logits (shape: batch_size, seq_length, vocab_size)
    and applies SPLADE transformation (ReLU + log) followed by pooling across the
    sequence length dimension.

    Args:
        pooling_strategy: Either 'max' or 'sum' for SPLADE pooling

    """

    def __init__(
        self,
        pooling_strategy: str = "max",
    ) -> None:
        super().__init__()
        self.pooling_strategy = pooling_strategy
        if pooling_strategy not in ["max", "sum"]:
            raise ValueError("pooling_strategy must be either 'max' or 'sum'")

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the mofrom ...models.Pooling import Pooling
        del.

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

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of the SPLADE embeddings (vocabulary size)"""
        # This will be set by the MLMTransformer
        return self.auto_model.config.vocab_size if hasattr(self, "auto_model") else None
