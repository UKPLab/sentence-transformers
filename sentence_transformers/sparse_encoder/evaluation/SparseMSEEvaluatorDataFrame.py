from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sentence_transformers.evaluation import MSEEvaluatorFromDataFrame

if TYPE_CHECKING:
    import numpy as np

    from sentence_transformers.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)


class SparseMSEEvaluatorDataFrame(MSEEvaluatorFromDataFrame):
    def __init__(
        self,
        dataframe: list[dict[str, str]],
        teacher_model: SparseEncoder,
        combinations: list[tuple[str, str]],
        batch_size: int = 8,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
    ):
        super.__init__(dataframe, teacher_model, combinations, batch_size, name, write_csv, truncate_dim)

    def __call__(
        self, model: SparseEncoder, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        return super.__call__(model, output_path, epoch, steps)

    def embed_inputs(
        self,
        model: SparseEncoder,
        sentences: str | list[str] | np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return model.encode(
            sentences,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            **kwargs,
        )

    def store_metrics_in_model_card_data(
        self, model: SparseEncoder, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch, step)
