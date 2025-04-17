from __future__ import annotations

from sentence_transformers.losses.GISTEmbedLoss import GISTEmbedLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseGISTEmbedLoss(GISTEmbedLoss):
    def __init__(
        self,
        model: SparseEncoder,
        guide: SparseEncoder,
        temperature: float = 0.01,
    ) -> None:
        return super().__init__(model, guide=guide, temperature=temperature)
