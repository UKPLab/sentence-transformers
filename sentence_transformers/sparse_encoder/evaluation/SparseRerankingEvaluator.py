from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers.util import cos_sim

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder

logger = logging.getLogger(__name__)


class SparseRerankingEvaluator(RerankingEvaluator):
    def __init__(
        self,
        samples: list[dict[str, str | list[str]]],
        at_k: int = 10,
        name: str = "",
        write_csv: bool = True,
        similarity_fct: Callable[[Tensor, Tensor], Tensor] = cos_sim,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        use_batched_encoding: bool = True,
        truncate_dim: int | None = None,
        mrr_at_k: int | None = None,
    ):
        return super().__init__(
            samples=samples,
            at_k=at_k,
            name=name,
            write_csv=write_csv,
            similarity_fct=similarity_fct,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            use_batched_encoding=use_batched_encoding,
            truncate_dim=truncate_dim,
            mrr_at_k=mrr_at_k,
        )

    def __call__(
        self, model: SparseEncoder, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        return super().__call__(model=model, output_path=output_path, epoch=epoch, steps=steps)

    def compute_metrices(self, model: SparseEncoder):
        return super().compute_metrices(model)

    def compute_metrices_batched(self, model: SparseEncoder):
        return super().compute_metrices_batched(model)

    def compute_metrices_individual(self, model: SparseEncoder):
        return super().compute_metrices_individual(model)

    def embed_inputs(
        self,
        model: SparseEncoder,
        sentences: str | list[str] | np.ndarray,
        show_progress_bar: bool | None = None,
        **kwargs,
    ) -> Tensor:
        kwargs["truncate_dim"] = self.truncate_dim
        return model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_sparse_tensor=True,
            convert_to_tensor=False,  # as we are using slicing on sparse tensors that is not supported so we want to keep a list of sparse tensors
            **kwargs,
        )

    def store_metrics_in_model_card_data(
        self, model: SparseEncoder, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch=epoch, step=step)
