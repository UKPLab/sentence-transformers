from __future__ import annotations

from sentence_transformers import util
from sentence_transformers.sparse_encoder.losses.SparseCoSENTLoss import SparseCoSENTLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseAnglELoss(SparseCoSENTLoss):
    def __init__(self, model: SparseEncoder, scale: float = 20.0) -> None:
        return super().__init__(model, scale, similarity_fct=util.pairwise_angle_sim)
