from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from sentence_transformers.similarity_functions import SimilarityFunction
    from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


logger = logging.getLogger(__name__)


class SparseEmbeddingSimilarityEvaluator(EmbeddingSimilarityEvaluator):
    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        scores: list[float],
        batch_size: int = 16,
        main_similarity: str | SimilarityFunction | None = None,
        similarity_fn_names: list[Literal["cosine", "euclidean", "manhattan", "dot"]] | None = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] | None = None,
        truncate_dim: int | None = None,
    ):
        return super().__init__(
            sentences1=sentences1,
            sentences2=sentences2,
            scores=scores,
            batch_size=batch_size,
            main_similarity=main_similarity,
            similarity_fn_names=similarity_fn_names,
            name=name,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
            precision=precision,
            truncate_dim=truncate_dim,
        )

    def __call__(
        self, model: SparseEncoder, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        return super().__call__(model=model, output_path=output_path, epoch=epoch, steps=steps)

    def embed_inputs(
        self,
        model: SparseEncoder,
        sentences: str | list[str] | np.ndarray,
        **kwargs,
    ) -> Tensor:
        return model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_sparse_tensor=True,
            precision=self.precision,
            normalize_embeddings=bool(self.precision),
            **kwargs,
        )

    def store_metrics_in_model_card_data(
        self, model: SparseEncoder, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self.name, metrics, epoch=epoch, step=step)
