from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class DebiasedMultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.cos_sim, tau_plus: float = 0.01) -> None:
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.tau_plus = tau_plus
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Compute the embeddings and distribute them to anchor and candidates (positive and optionally negatives)
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        anchors = embeddings[0]  # (batch_size, embedding_dim)
        candidates = torch.cat(embeddings[1:])  # (batch_size * (1 + num_negatives), embedding_dim)

        # For every anchor, we compute the similarity to all other candidates (positives and negatives),
        # also from other anchors. This gives us a lot of in-batch negatives.
        scores: Tensor = self.similarity_fct(anchors, candidates) * self.scale
        # (batch_size, batch_size * (1 + num_negatives))

        mask = torch.ones_like(scores, dtype=torch.bool)
        for i in range(scores.size(0)):
            mask[i, i] = False

        neg_exp = torch.exp(scores.masked_fill(mask, float("-inf"))).sum(dim=-1)

        pos_exp = torch.exp(torch.gather(scores, -1,
                                         torch.arange(scores.size(0),
                                                      device=scores.device).unsqueeze(1)).squeeze())
        
        N = scores.size(1) - 1

    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}

    @property
    def citation(self) -> str:
        return """
TODO: Add citation
"""
