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
    """
    This evaluator extends :class:`~sentence_transformers.evaluation.TripletEvaluator` but is specifically designed for sparse encoder models.

    Evaluate a model based on a triplet: (sentence, positive_example, negative_example).
    Checks if ``similarity(sentence, positive_example) < similarity(sentence, negative_example) + margin``.

    Args:
        anchors (List[str]): Sentences to check similarity to. (e.g. a query)
        positives (List[str]): List of positive sentences
        negatives (List[str]): List of negative sentences
        main_similarity_function (Union[str, SimilarityFunction], optional):
            The similarity function to use. If not specified, use cosine similarity,
            dot product, Euclidean, and Manhattan similarity. Defaults to None.
        margin (Union[float, Dict[str, float]], optional): Margins for various similarity metrics.
            If a float is provided, it will be used as the margin for all similarity metrics.
            If a dictionary is provided, the keys should be 'cosine', 'dot', 'manhattan', and 'euclidean'.
            The value specifies the minimum margin by which the negative sample should be further from
            the anchor than the positive sample. Defaults to None.
        name (str): Name for the output. Defaults to "".
        batch_size (int): Batch size used to compute embeddings. Defaults to 16.
        show_progress_bar (bool): If true, prints a progress bar. Defaults to False.
        write_csv (bool): Write results to a CSV file. Defaults to True.
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to.
            `None` uses the model's current truncation dimension. Defaults to None.
        similarity_fn_names (List[str], optional): List of similarity function names to evaluate.
            If not specified, evaluate using the ``model.similarity_fn_name``.
            Defaults to None.

    Example:
        ::

            import logging

            from datasets import load_dataset

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseTripletEvaluator

            logging.basicConfig(format="%(message)s", level=logging.INFO)

            # Load a model
            model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

            # Load triplets from the AllNLI dataset
            # The dataset contains triplets of (anchor, positive, negative) sentences
            dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev[:1000]")

            # Initialize the SparseTripletEvaluator
            evaluator = SparseTripletEvaluator(
                anchors=dataset[:1000]["anchor"],
                positives=dataset[:1000]["positive"],
                negatives=dataset[:1000]["negative"],
                name="all_nli_dev",
                batch_size=32,
                show_progress_bar=True,
            )

            # Run the evaluation
            results = evaluator(model)
            '''
            TripletEvaluator: Evaluating the model on the all_nli_dev dataset:
            Accuracy Dot Similarity:	85.10%
            '''
            # Print the results
            print(f"Primary metric: {evaluator.primary_metric}")
            # => Primary metric: all_nli_dev_dot_accuracy
            print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
            # => Primary metric value: 0.8510

    """

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
            save_to_cpu=True,
            **kwargs,
        )

    def store_metrics_in_model_card_data(
        self, model: SparseEncoder, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch=epoch, step=step)
