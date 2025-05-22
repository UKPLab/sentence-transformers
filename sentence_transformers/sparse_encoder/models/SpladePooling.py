from __future__ import annotations

import torch

from sentence_transformers.models.Module import Module


class SpladePooling(Module):
    """
    SPLADE Pooling module for creating the sparse embeddings.

    This module implements the SPLADE pooling mechanism that:

    1. Takes token logits from a masked language model (MLM).
    2. Applies a sparse transformation using an activation function followed by log1p (i.e., log(1 + activation(MLM_logits))).
    3. Applies a pooling strategy `max` or `sum` to produce sparse embeddings.

    The resulting embeddings are highly sparse and capture lexical information,
    making them suitable for efficient information retrieval.

    Args:
        pooling_strategy (str): Pooling method across token dimensions.
            Choices:
                - `sum`: Sum pooling (used in original SPLADE see https://arxiv.org/pdf/2107.05720).
                - `max`: Max pooling (used in SPLADEv2 and later models see https://arxiv.org/pdf/2109.10086 or https://arxiv.org/pdf/2205.04733).
        activation_function (str): Activation function applied before log1p transformation.
            Choices:
                - `relu`: ReLU activation (standard in all Splade models).
                - `log1p_relu`: log(1 + ReLU(x)) variant used in Opensearch Splade models see arxiv.org/pdf/2504.14839.
        word_embedding_dimension (int, optional): Dimensionality of the output embeddings (if needed).
    """

    SPLADE_POOLING_MODES = ("sum", "max")
    SPLADE_ACTIVATION = ["relu", "log1p_relu"]
    config_keys: list[str] = ["pooling_strategy", "activation_function", "word_embedding_dimension"]
    forward_kwargs = {"memory_efficient"}

    def __init__(
        self, pooling_strategy: str = "max", activation_function="relu", word_embedding_dimension: int = None
    ) -> None:
        super().__init__()
        self.pooling_strategy = pooling_strategy
        if pooling_strategy not in self.SPLADE_POOLING_MODES:
            raise ValueError("pooling_strategy must be either 'max' or 'sum'")
        self.activation_function = activation_function
        if activation_function not in self.SPLADE_ACTIVATION:
            raise ValueError("activation_function must be either 'relu' or 'log1p_relu'")
        self.word_embedding_dimension = word_embedding_dimension  # This will be set in the forward method

    def forward(self, features: dict[str, torch.Tensor], memory_efficient: bool = False) -> dict[str, torch.Tensor]:
        """Forward pass of the model.
        Args:
            features: Dictionary containing input features with 'token_embeddings' key as MLM logits.
        Returns:
            Dictionary containing SPLADE pooled embeddings
        """
        # Get the MLM head logits (shape: batch_size, seq_length, vocab_size)
        mlm_logits = features["token_embeddings"]
        if memory_efficient:
            batch_size, seq_len, vocab_s = mlm_logits.shape

            effective_computation_dtype = mlm_logits.dtype

            if self.pooling_strategy == "max":
                pooled_scores = torch.full(
                    (batch_size, vocab_s), float("-inf"), dtype=effective_computation_dtype, device=mlm_logits.device
                )
            elif self.pooling_strategy == "sum":
                pooled_scores = torch.zeros(
                    (batch_size, vocab_s), dtype=effective_computation_dtype, device=mlm_logits.device
                )
            else:
                raise ValueError(f"Unsupported pooling_strategy: {self.pooling_strategy}")

            chunk_size = 32

            for i in range(0, seq_len, chunk_size):
                current_chunk_logits = mlm_logits[:, i : i + chunk_size, :]

                if self.activation_function == "relu":
                    current_chunk_transformed = torch.log1p(torch.relu(current_chunk_logits))
                elif self.activation_function == "log1p_relu":
                    current_chunk_transformed = torch.log1p(torch.log1p(torch.relu(current_chunk_logits)))
                # current_chunk_transformed now has the dtype that log1p naturally produced.

                # Ensure it's consistent with our determined effective_computation_dtype for pooling/accumulation.
                current_chunk_transformed = current_chunk_transformed.to(effective_computation_dtype)

                if self.pooling_strategy == "max":
                    chunk_pooled = torch.max(current_chunk_transformed, dim=1)[0]
                    # Avoid in-place update with out= for torch.maximum when grads are required
                    pooled_scores = torch.maximum(pooled_scores, chunk_pooled)
                else:  # sum
                    chunk_pooled = torch.sum(current_chunk_transformed, dim=1)
                    pooled_scores += chunk_pooled
        else:
            # Apply ReLU and log transformation for SPLADE
            if self.activation_function == "relu":
                splade_scores = torch.log1p(torch.relu(mlm_logits))
            elif self.activation_function == "log1p_relu":
                splade_scores = torch.log1p(torch.log1p(torch.relu(mlm_logits)))
            else:
                raise ValueError("activation_function must be either 'relu' or 'log1p_relu'")

            # Pool across sequence length dimension
            if self.pooling_strategy == "max":
                pooled_scores = torch.max(splade_scores, dim=1)[0]  # shape: batch_size, vocab_size
            else:  # sum
                pooled_scores = torch.sum(splade_scores, dim=1)  # shape: batch_size, vocab_size

        del features["token_embeddings"]

        if self.word_embedding_dimension is None:
            self.word_embedding_dimension = pooled_scores.shape[1]
        features["sentence_embedding"] = pooled_scores
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)

    def __repr__(self) -> str:
        return f"SpladePooling({self.get_config_dict()})"

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of the sentence embedding.

        Returns:
            int: Dimension of the sentence embedding
        """
        return self.word_embedding_dimension
