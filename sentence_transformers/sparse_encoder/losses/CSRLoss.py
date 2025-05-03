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
    CSRLoss implements a combined loss function for Contrastive Sparse Representation (CSR) models.

    This loss combines two components:
    1. A reconstruction loss :class:`CSRReconstructionLoss` that ensures the sparse representation can faithfully
       reconstruct the original embedding.
    2. A contrastive learning component :class:`SparseMultipleNegativesRankingLoss` that ensures semantically
       similar sentences have similar representations.

    The total loss is linear combination of the two losses.

    Args:
        model: SparseEncoder model
        beta: Weight for the L_aux component in the reconstruction loss
        gamma: Weight for the contrastive MRL loss component
        scale: Scale factor for the similarity scores in the MRL loss

    References:
        - For more details, see the paper "Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation"
          https://arxiv.org/abs/2503.01776

    Requirements:
        1. Sentence pairs or triplets for the MRL component
        2. Uses autoencoder components of the SparseEncoder model

    Relations:
        - Uses :class:`CSRReconstructionLoss` for the reconstruction component
        - Uses :class:`SparseMultipleNegativesRankingLoss` for the contrastive component

    Example:
        ::

            from datasets import Dataset

            from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

            model = SparseEncoder("sentence-transformers/all-MiniLM-L6-v2")
            train_dataset = Dataset.from_dict(
                {
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                    "negative": ["It's quite rainy, sadly.", "She walked to the store."],
                }
            )
            loss = losses.CSRLoss(model, beta=0.1, gamma=1.0, scale=20.0)

            trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
            trainer.train()
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

        ranking_loss = self.ranking_loss.compute_loss_from_embeddings(sentence_embedding, labels)

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

    @property
    def citation(self) -> str:
        return """
@misc{wen2025matryoshkarevisitingsparsecoding,
      title={Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation},
      author={Tiansheng Wen and Yifei Wang and Zequn Zeng and Zhong Peng and Yudi Su and Xinyang Liu and Bo Chen and Hongwei Liu and Stefanie Jegelka and Chenyu You},
      year={2025},
      eprint={2503.01776},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.01776},
}
"""
