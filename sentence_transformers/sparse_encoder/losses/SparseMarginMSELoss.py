from __future__ import annotations

from sentence_transformers import util
from sentence_transformers.losses.MarginMSELoss import MarginMSELoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseMarginMSELoss(MarginMSELoss):
    def __init__(self, model: SparseEncoder, similarity_fct=util.pairwise_dot_score) -> None:
        return super().__init__(model, similarity_fct=similarity_fct)
