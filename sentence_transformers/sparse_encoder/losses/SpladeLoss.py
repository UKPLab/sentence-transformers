from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import torch
import torch.nn as nn

from sentence_transformers.sparse_encoder.losses import (
    FlopsLoss,
    SparseDistillKLDivLoss,
    SparseMarginMSELoss,
    SparseMultipleNegativesRankingLoss,
)
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class PrincipalLoss(Enum):
    """The principal loss types for the model"""

    MMSE = SparseMarginMSELoss
    KL = SparseDistillKLDivLoss
    MRL = SparseMultipleNegativesRankingLoss


class SpladeLoss(nn.Module):
    def __init__(
        self, model: SparseEncoder, main_loss: PrincipalLoss, lamda_corpus: float = 0.1, lamda_query: float = 0.1
    ):
        super().__init__()
        self.model = model
        self.lamda_corpus = lamda_corpus
        self.lamda_query = lamda_query
        self.main_loss = main_loss(model, similarity_fct=model.similarity_pairwise)
        self.flops_loss = FlopsLoss(model)

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        # Compute embeddings using the model
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        main_loss_value = self.main_loss.compute_loss_from_embeddings(embeddings, labels)

        flops_query = self.flops_loss.compute_loss_from_embeddings(embeddings, "query")
        flops_corpus = self.flops_loss.compute_loss_from_embeddings(embeddings, "corpus")

        # Compute the total loss
        total_loss = main_loss_value + self.lamda_query * flops_query + self.lamda_corpus * flops_corpus
        return total_loss

    def get_config_dict(self):
        """
        Get the configuration dictionary.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {"lamda_corpus": self.lamda_corpus, "lamda_query": self.lamda_query, "main_loss": self.main_loss}

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
