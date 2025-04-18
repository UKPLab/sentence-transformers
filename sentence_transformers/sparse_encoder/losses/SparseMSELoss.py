from __future__ import annotations

from sentence_transformers.losses.MSELoss import MSELoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseMSELoss(MSELoss):
    def __init__(self, model: SparseEncoder) -> None:
        return super().__init__(model)
