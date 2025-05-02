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
    """
    This evaluator extends TranslationEvaluator but is specifically designed for sparse encoder models.

    models but doesn't take benefit of the sparse tensor torch representation yet, Memory issues may occur.

    Computes the mean squared error (x100) between the computed sentence embedding
    and some target sentence embedding.

    The MSE is computed between ||teacher.encode(source_sentences) - student.encode(target_sentences)||.

    For multilingual knowledge distillation (https://arxiv.org/abs/2004.09813), source_sentences are in English
    and target_sentences are in a different language like German, Chinese, Spanish...

    Args:
        source_sentences (List[str]): Source sentences to embed with the teacher model.
        target_sentences (List[str]): Target sentences to embed with the student model.
        teacher_model (SparseEncoder, optional): The teacher model to compute the source sentence embeddings.
        show_progress_bar (bool, optional): Show progress bar when computing embeddings. Defaults to False.
        batch_size (int, optional): Batch size to compute sentence embeddings. Defaults to 32.
        name (str, optional): Name of the evaluator. Defaults to "".
        write_csv (bool, optional): Write results to CSV file. Defaults to True.
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to. `None` uses the model's current truncation
            dimension. Defaults to None.

    Example:
        ::

            import logging

            from datasets import load_dataset

            from sentence_transformers.sparse_encoder import (
                SparseEncoder,
                SparseMSEEvaluator,
            )

            logging.basicConfig(format="%(message)s", level=logging.INFO)

            # Load a model
            student_model = SparseEncoder("prithivida/Splade_PP_en_v1")
            teacher_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

            # Load any dataset with some texts
            dataset = load_dataset("sentence-transformers/stsb", split="validation")
            sentences = dataset["sentence1"] + dataset["sentence2"]

            # Given queries, a corpus and a mapping with relevant documents, the SparseMSEEvaluator computes different MSE metrics.
            mse_evaluator = SparseMSEEvaluator(
                source_sentences=sentences,
                target_sentences=sentences,
                teacher_model=teacher_model,
                name="stsb-dev",
            )
            results = mse_evaluator(student_model)
            '''
            MSE evaluation (lower = better) on the stsb-dev dataset:
            MSE (*100):	0.035540
            '''
            # Print the results
            print(f"Primary metric: {mse_evaluator.primary_metric}")
            # => Primary metric: stsb-dev_negative_mse
            print(f"Primary metric value: {results[mse_evaluator.primary_metric]:.4f}")
            # => Primary metric value: -0.0355
    """

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
        logger.warning(
            "The SparseMSEEvaluator is not handling the mse compute with sparse tensors yet. Memory issues may occur."
        )

    def __call__(
        self,
        model: SparseEncoder,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
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
            convert_to_sparse_tensor=False,
            save_on_cpu=True,
            **kwargs,
        )

    def store_metrics_in_model_card_data(
        self,
        model: SparseEncoder,
        metrics: dict[str, Any],
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch=epoch, step=step)
