from __future__ import annotations

from sentence_transformers import util
from sentence_transformers.losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseMultipleNegativesRankingLoss(MultipleNegativesRankingLoss):
    def __init__(self, model: SparseEncoder, scale: float = 20.0, similarity_fct=util.dot_score) -> None:
        return super().__init__(model, scale=scale, similarity_fct=similarity_fct)
