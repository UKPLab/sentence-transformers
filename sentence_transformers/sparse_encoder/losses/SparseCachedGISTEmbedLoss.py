from __future__ import annotations

from sentence_transformers.losses.CachedGISTEmbedLoss import CachedGISTEmbedLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseCachedGISTEmbedLoss(CachedGISTEmbedLoss):
    def __init__(
        self,
        model: SparseEncoder,
        guide: SparseEncoder,
        temperature: float = 0.01,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        return super().__init__(
            model=model,
            guide=guide,
            temperature=temperature,
            mini_batch_size=mini_batch_size,
            show_progress_bar=show_progress_bar,
        )
