from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sentence_transformers.evaluation import TranslationEvaluator

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


logger = logging.getLogger(__name__)


class SparseTranslationEvaluator(TranslationEvaluator):
    def __init__(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        show_progress_bar: bool = False,
        batch_size: int = 16,
        name: str = "",
        print_wrong_matches: bool = False,
        write_csv: bool = True,
        truncate_dim: int | None = None,
    ):
        super().__init__(
            source_sentences,
            target_sentences,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            name=name,
            print_wrong_matches=print_wrong_matches,
            write_csv=write_csv,
            truncate_dim=truncate_dim,
        )

        assert len(self.source_sentences) == len(self.target_sentences)

        if name:
            name = "_" + name

        self.csv_file = "translation_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "src2trg", "trg2src"]
        self.primary_metric = "mean_accuracy"

    def __call__(
        self, model: SparseEncoder, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        return super().__call__(model, output_path, epoch, steps)

    def embed_inputs(
        self,
        model: SparseEncoder,
        sentences: str | list[str] | np.ndarray,
        **kwargs,
    ) -> list[Tensor]:
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
