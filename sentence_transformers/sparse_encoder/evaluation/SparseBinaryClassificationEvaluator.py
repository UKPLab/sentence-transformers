from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from sentence_transformers.evaluation import BinaryClassificationEvaluator

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


logger = logging.getLogger(__name__)


class SparseBinaryClassificationEvaluator(BinaryClassificationEvaluator):
    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        labels: list[int],
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        truncate_dim: int | None = None,
        similarity_fn_names: list[Literal["cosine", "dot", "euclidean", "manhattan"]] | None = None,
    ):
        super().__init__(
            sentences1=sentences1,
            sentences2=sentences2,
            labels=labels,
            name=name,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
            truncate_dim=truncate_dim,
            similarity_fn_names=similarity_fn_names,
        )

    def __call__(
        self, model: SparseEncoder, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        return super().__call__(model, output_path, epoch, steps)

    def compute_metrices(self, model: SparseEncoder) -> dict[str, dict[str, float]]:
        return super().compute_metrices(model)

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
            **kwargs,
        )

    def store_metrics_in_model_card_data(
        self, model: SparseEncoder, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch, step)
