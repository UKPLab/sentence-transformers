from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from sentence_transformers.sparse_encoder.losses.CSRReconstructionLoss import CSRReconstructionLoss
from sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss import (
    SparseMultipleNegativesRankingLoss,
)
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class CSRLoss(nn.Module):
    """
    CSR Loss module that combines the CSRReconstruction Loss and Sparse Multiple Negatives Ranking Loss MRL (InfoNCE).
    Based on the paper:
    Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation, https://arxiv.org/abs/2503.01776

    This module computes the combined loss according to the formula:
    L_CSR = L_recon + γ * L_MRL

    where:
    - L_recon = L(k) + L(4k)/8 + β*L_aux
    - L_MRL is the Multiple Negatives Ranking Loss
    """

    def __init__(self, model: SparseEncoder, beta: float = 0.1, gamma: float = 1.0, scale: float = 20.0):
        super().__init__()
        self.model = model
        self.beta = beta
        self.gamma = gamma
        self.scale = scale

        # Initialize the component losses
        self.reconstruction_loss = CSRReconstructionLoss(model, beta)
        self.ranking_loss = SparseMultipleNegativesRankingLoss(model, scale)

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the CSR Loss module.
        This method is used when the loss is computed as part of the model's forward pass.

        Args:
            sentence_features: Iterable of dictionaries containing sentence embeddings
            labels: Optional tensor of labels (not used in this implementation)

        Returns:
            Dictionary containing the total loss and individual loss components
        """
        # Compute embeddings using the model
        outputs = [self.model(sentence_feature) for sentence_feature in sentence_features]
        sentence_embedding = [output["sentence_embedding"] for output in outputs]

        recon_loss = self.reconstruction_loss.compute_loss_from_embeddings(outputs)

        ranking_loss = self.ranking_loss.compute_loss_from_embeddings(sentence_embedding)

        # Compute total loss: L_CSR = L_recon + γ * L_MRL
        total_loss = recon_loss + self.gamma * ranking_loss

        return total_loss

    def get_config_dict(self):
        """
        Get the configuration dictionary.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {"beta": self.beta, "gamma": self.gamma, "scale": self.scale}
