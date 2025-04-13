from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sentence_transformers.evaluation import MSEEvaluator

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from sentence_transformers.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)


class SparseMSEEvaluator(MSEEvaluator):
    def __init__(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        teacher_model=None,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
    ):
        # Set attributes before calling super().__init__() because SparseMSEEvaluator.embed_inputs()
        # is called in the superclass constructor.
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.name = name
        self.write_csv = write_csv
        self.truncate_dim = truncate_dim

        super().__init__(
            source_sentences=source_sentences,
            target_sentences=target_sentences,
            teacher_model=teacher_model,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            name=name,
            write_csv=write_csv,
            truncate_dim=truncate_dim,
        )

    def __call__(
        self,
        model: SparseEncoder,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        return super().__call__(model, output_path, epoch, steps)

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
        self,
        model: SparseEncoder,
        metrics: dict[str, Any],
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch, step)
