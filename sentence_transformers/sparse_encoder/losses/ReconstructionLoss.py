from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers.sparse_encoder import SparseEncoder


class ReconstructionLoss(nn.Module):
    """
    Reconstruction Loss module for Sparse AutoEncoder.

    This module computes the reconstruction loss according to the formula:
    L_recon = L(k) + L(4k)/8 + β*L_aux

    where:
    - L(k) = ||f(x) - f(dx)_k||₂²
    - L(4k) = ||f(x) - f(dx)_4k||₂²
    - L_aux = ||e - ê||₂², e = f(x) - f(dx), ê = W_dec*z
    """

    def __init__(self, model: SparseEncoder, beta: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.beta = beta

    def forward(self, sentence_features: Iterable[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the Reconstruction Loss module.
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
        Compute the reconstruction loss from embeddings.

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
            f_x = features["sentence_embedding_backbone"]
            x_hat_k = features["decoded_embedding_k"]
            x_hat_4k = features["decoded_embedding_4k"]
            e = features["error"]
            e_hat = features["error_hat"]

            # L(k) = ||f(x) - f(dx)_k||₂²
            L_k = F.mse_loss(f_x, x_hat_k)

            # L(4k) = ||f(x) - f(dx)_4k||₂²
            L_4k = F.mse_loss(f_x, x_hat_4k)

            # L_aux = ||e - ê||₂²
            L_aux = F.mse_loss(e, e_hat)

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
        return {
            "beta": self.beta,
        }
