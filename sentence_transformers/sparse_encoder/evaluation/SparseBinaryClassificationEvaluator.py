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
    """
    This evaluator extends :class:`BinaryClassificationEvaluator` but is specifically designed for sparse encoder models.

    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity, dot score, Euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    Args:
        sentences1 (List[str]): The first column of sentences.
        sentences2 (List[str]): The second column of sentences.
        labels (List[int]): labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1.
        name (str, optional): Name for the output. Defaults to "".
        batch_size (int, optional): Batch size used to compute embeddings. Defaults to 32.
        show_progress_bar (bool, optional): If true, prints a progress bar. Defaults to False.
        write_csv (bool, optional): Write results to a CSV file. Defaults to True.
        truncate_dim (Optional[int], optional): The dimension to truncate sentence embeddings to. `None` uses the model's current truncation dimension. Defaults to None.
        similarity_fn_names (Optional[List[Literal["cosine", "dot", "euclidean", "manhattan"]]], optional): The similarity functions to use. If not specified, defaults to the ``similarity_fn_name`` attribute of the model. Defaults to None.

    Example:
        ::
            import logging

            from datasets import load_dataset

            from sentence_transformers.sparse_encoder import (
                SparseBinaryClassificationEvaluator,
                SparseEncoder,
            )

            logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

            # Initialize the SPLADE model
            model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

            # Load a dataset with two text columns and a class label column (https://huggingface.co/datasets/sentence-transformers/quora-duplicates)
            eval_dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train[-1000:]")

            # Initialize the evaluator
            binary_acc_evaluator = SparseBinaryClassificationEvaluator(
                sentences1=eval_dataset["sentence1"],
                sentences2=eval_dataset["sentence2"],
                labels=eval_dataset["label"],
                name="quora_duplicates_dev",
                show_progress_bar=True,
                similarity_fn_names=["cosine", "dot", "euclidean", "manhattan"],
            )
            results = binary_acc_evaluator(model)
            '''
            Accuracy with Cosine-Similarity:             74.90      (Threshold: 0.8668)
            F1 with Cosine-Similarity:                   67.37      (Threshold: 0.5959)
            Precision with Cosine-Similarity:            54.15
            Recall with Cosine-Similarity:               89.13
            Average Precision with Cosine-Similarity:    67.81
            Matthews Correlation with Cosine-Similarity: 49.89

            Accuracy with Dot-Product:             76.50    (Threshold: 24.3460)
            F1 with Dot-Product:                   66.93    (Threshold: 20.0762)
            Precision with Dot-Product:            57.62
            Recall with Dot-Product:               79.81
            Average Precision with Dot-Product:    65.94
            Matthews Correlation with Dot-Product: 48.82

            Accuracy with Euclidean-Distance:             67.70     (Threshold: -10.0062)
            F1 with Euclidean-Distance:                   48.60     (Threshold: -0.2346)
            Precision with Euclidean-Distance:            32.13
            Recall with Euclidean-Distance:               99.69
            Average Precision with Euclidean-Distance:    20.52
            Matthews Correlation with Euclidean-Distance: -4.59

            Accuracy with Manhattan-Distance:             67.70     (Threshold: -103.1993)
            F1 with Manhattan-Distance:                   48.60     (Threshold: -1.1565)
            Precision with Manhattan-Distance:            32.13
            Recall with Manhattan-Distance:               99.69
            Average Precision with Manhattan-Distance:    21.05
            Matthews Correlation with Manhattan-Distance: -4.59
            '''
            # Print the results
            print(f"Primary metric: {binary_acc_evaluator.primary_metric}")
            # => Primary metric: quora_duplicates_dev_max_ap
            print(f"Primary metric value: {results[binary_acc_evaluator.primary_metric]:.4f}")
            # => Primary metric value: 0.6781
    """

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
        return super().__init__(
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
        return super().__call__(model, output_path=output_path, epoch=epoch, steps=steps)

    def compute_metrices(self, model: SparseEncoder) -> dict[str, dict[str, float]]:
        return super().compute_metrices(model=model)

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
