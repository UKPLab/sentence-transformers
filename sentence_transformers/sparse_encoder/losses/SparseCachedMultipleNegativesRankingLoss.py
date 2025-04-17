from __future__ import annotations

from torch import Tensor

from sentence_transformers import util
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import CachedMultipleNegativesRankingLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseCachedMultipleNegativesRankingLoss(CachedMultipleNegativesRankingLoss):
    def __init__(
        self,
        model: SparseEncoder,
        scale: float = 20.0,
        similarity_fct: callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        super().__init__(
            model,
            scale=scale,
            similarity_fct=similarity_fct,
            mini_batch_size=mini_batch_size,
            show_progress_bar=show_progress_bar,
        )
