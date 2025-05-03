from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers.sparse_encoder import SparseEncoder


def normalized_mean_squared_error(reconstruction: torch.Tensor, original_input: torch.Tensor) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized mean squared error (shape: [1])
    """
    return (((reconstruction - original_input) ** 2).mean(dim=1) / (original_input**2).mean(dim=1)).mean()


class CSRReconstructionLoss(nn.Module):
    """
    CSRReconstructionLoss implements the reconstruction loss component for Contrastive Sparse Representation (CSR) models.

    This loss ensures that the sparse encoding can accurately reconstruct the original model embeddings through
    three components:

    1. A primary reconstruction loss (L_k) that measures the error between the original embedding and its
       reconstruction using the top-k sparse components.
    2. A secondary reconstruction loss (L_4k) that measures the error using the top-4k sparse components.
    3. An auxiliary loss (L_aux) that helps to learn residual information.


    Args:
        model: SparseEncoder model with autoencoder components
        beta: Weight for the auxiliary loss component (L_aux)

    References:
        - For more details, see the paper "Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation"
          https://arxiv.org/abs/2503.01776

    Requirements:
        1. The model must be configured to output the necessary reconstruction components
        2. Used with SparseEncoder models that implement compositional sparse autoencoding

    Relations:
        - Used as a component within :class:`CSRLoss` combined with a contrastive loss

    Example:
        ::
            This loss is typically used within the :class:`CSRLoss` class, which combines it with other loss components.

    """

    def __init__(self, model: SparseEncoder, beta: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.beta = beta

    def forward(self, sentence_features: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the CSRReconstruction Loss module.
        This method is used when the loss is computed as part of the model's forward pass.

        Args:
            sentence_features: Iterable of dictionaries containing sentence embeddings and their sparse representations

        Returns:
            Dictionary containing the total loss and individual loss components
        """
        # Compute embeddings using the model
        outputs = [self.model(sentence_feature) for sentence_feature in sentence_features]
        return self.compute_loss_from_embeddings(outputs)

    def compute_loss_from_embeddings(self, outputs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Compute the CSRReconstruction loss from embeddings.

        Args:
            outputs: List of dictionaries containing sentence embeddings and their sparse representations

        Returns:
            Dictionary containing the total loss and individual loss components
        """
        # Initialize loss components
        total_L_k = 0.0
        total_L_4k = 0.0
        total_L_aux = 0.0

        # Process each sentence feature
        for features in outputs:
            x = features["sentence_embedding_backbone"]
            recons_k = features["decoded_embedding_k"]
            recons_4k = features["decoded_embedding_4k"]
            recons_aux = features["decoded_embedding_aux"]
            reconsk_pre_bias = features["decoded_embedding_k_pre_bias"]

            # L(k) = ||f(x) - f(dx)_k||₂²
            L_k = F.mse_loss(x, recons_k)

            # L(4k) = ||f(x) - f(dx)_4k||₂²
            L_4k = F.mse_loss(x, recons_4k)

            # L_aux = ||e - ê||₂²
            L_aux = normalized_mean_squared_error(recons_aux, x - reconsk_pre_bias)

            # Accumulate losses
            total_L_k += L_k
            total_L_4k += L_4k
            total_L_aux += L_aux

        # Average losses over batch
        batch_size = len(outputs)
        if batch_size > 0:
            total_L_k /= batch_size
            total_L_4k /= batch_size
            total_L_aux /= batch_size

        # Total loss: L_recon = L(k) + L(4k)/8 + β*L_aux
        total_loss = total_L_k + total_L_4k / 8 + self.beta * total_L_aux

        return total_loss

    def get_config_dict(self):
        """
        Get the configuration dictionary.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {"beta": self.beta}
