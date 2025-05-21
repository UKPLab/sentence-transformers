from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Literal

import torch

from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.util import append_to_last_row

if TYPE_CHECKING:
    import numpy as np

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
        max_active_dims (Optional[int], optional): The maximum number of active dimensions to use.
            `None` uses the model's current `max_active_dims`. Defaults to None.
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
            Model Sparsity Stats  Query : Row Non-Zero Mean: 105.4530029296875, Row Sparsity Mean: 0.9965449571609497
            Model Sparsity Stats  Corpus : Row Non-Zero Mean: 69.18349838256836, Row Sparsity Mean: 0.9977333247661591
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
        max_active_dims: int | None = None,
        similarity_fn_names: list[Literal["cosine", "dot", "euclidean", "manhattan"]] | None = None,
        main_distance_function: str | SimilarityFunction | None = "deprecated",
    ):
        self.max_active_dims = max_active_dims
        self.sparsity_stats = {
            "row_non_zero_mean_query": 0,
            "row_sparsity_mean_query": 0,
            "row_non_zero_mean_corpus": 0,
            "row_sparsity_mean_corpus": 0,
        }
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
            similarity_fn_names=similarity_fn_names,
            main_distance_function=main_distance_function,
        )

    def _append_csv_headers(self, similarity_fn_names):
        super()._append_csv_headers(similarity_fn_names)
        for sparsity_stat in self.sparsity_stats.keys():
            self.csv_headers.append(f"{sparsity_stat}")

    def __call__(
        self, model: SparseEncoder, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        self.sparsity_stats = {
            "row_non_zero_mean_query": 0,
            "row_sparsity_mean_query": 0,
            "row_non_zero_mean_corpus": 0,
            "row_sparsity_mean_corpus": 0,
        }
        metrics = super().__call__(model=model, output_path=output_path, epoch=epoch, steps=steps)

        metrics.update(self.prefix_name_to_metrics(self.sparsity_stats, self.name))
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        logger.info(
            f"Model Sparsity Stats Query : Row Non-Zero Mean: {self.sparsity_stats['row_non_zero_mean_query']}, Row Sparsity Mean: {self.sparsity_stats['row_sparsity_mean_query']}"
        )
        logger.info(
            f"Model Sparsity Stats Corpus : Row Non-Zero Mean: {self.sparsity_stats['row_non_zero_mean_corpus']}, Row Sparsity Mean: {self.sparsity_stats['row_sparsity_mean_corpus']}"
        )
        if output_path is not None and self.write_csv:
            append_to_last_row(
                os.path.join(output_path, self.csv_file),
                self.sparsity_stats.values(),
            )

        return metrics

    def embed_inputs(
        self,
        model: SparseEncoder,
        sentences: str | list[str] | np.ndarray,
        **kwargs,
    ) -> torch.Tensor:
        embeddings = model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
            max_active_dims=self.max_active_dims,
            **kwargs,
        )
        stat = model.get_sparsity_stats(embeddings)
        if len(self.anchors) == len(sentences) and self.sparsity_stats["row_non_zero_mean_query"] == 0:
            self.sparsity_stats["row_non_zero_mean_query"] = stat["row_non_zero_mean"]
            self.sparsity_stats["row_sparsity_mean_query"] = stat["row_sparsity_mean"]
        else:
            if self.sparsity_stats["row_non_zero_mean_corpus"] == 0:
                self.sparsity_stats["row_non_zero_mean_corpus"] = stat["row_non_zero_mean"]
                self.sparsity_stats["row_sparsity_mean_corpus"] = stat["row_sparsity_mean"]
            else:
                self.sparsity_stats["row_non_zero_mean_corpus"] = (
                    self.sparsity_stats["row_non_zero_mean_corpus"] + stat["row_non_zero_mean"]
                ) / 2
                self.sparsity_stats["row_sparsity_mean_corpus"] = (
                    self.sparsity_stats["row_sparsity_mean_corpus"] + stat["row_sparsity_mean"]
                ) / 2

        return embeddings

    def store_metrics_in_model_card_data(
        self, model: SparseEncoder, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch=epoch, step=step)

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = super().get_config_dict()
        if self.max_active_dims is not None:
            config_dict["max_active_dims"] = self.max_active_dims
        return config_dict
