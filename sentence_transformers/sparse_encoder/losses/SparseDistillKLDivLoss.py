from __future__ import annotations

from sentence_transformers import util
from sentence_transformers.losses.DistillKLDivLoss import DistillKLDivLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseDistillKLDivLoss(DistillKLDivLoss):
    def __init__(self, model: SparseEncoder, similarity_fct=util.pairwise_dot_score) -> None:
        super().__init__(model, similarity_fct=similarity_fct)
