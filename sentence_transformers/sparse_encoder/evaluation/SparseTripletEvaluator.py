from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from sentence_transformers.evaluation import TripletEvaluator

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from sentence_transformers.similarity_functions import SimilarityFunction
    from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


logger = logging.getLogger(__name__)


class SparseTripletEvaluator(TripletEvaluator):
    def __init__(
        self,
        anchors: list[str],
        positives: list[str],
        negatives: list[str],
        main_similarity_function: str | SimilarityFunction | None = None,
        margin: float | dict[str, float] | None = None,
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        truncate_dim: int | None = None,
        similarity_fn_names: list[Literal["cosine", "dot", "euclidean", "manhattan"]] | None = None,
        main_distance_function: str | SimilarityFunction | None = "deprecated",
    ):
        return super().__init__(
            anchors=anchors,
            positives=positives,
            negatives=negatives,
            main_similarity_function=main_similarity_function,
            margin=margin,
            name=name,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
            truncate_dim=truncate_dim,
            similarity_fn_names=similarity_fn_names,
            main_distance_function=main_distance_function,
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
        kwargs["truncate_dim"] = self.truncate_dim
        return model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_sparse_tensor=True,
            save_on_cpu=True,
            **kwargs,
        )

    def store_metrics_in_model_card_data(
        self, model: SparseEncoder, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch=epoch, step=step)
