from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Callable

import torch

from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers.util import append_to_last_row, cos_sim

if TYPE_CHECKING:
    import numpy as np

    from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder

logger = logging.getLogger(__name__)


class SparseRerankingEvaluator(RerankingEvaluator):
    """
    This evaluator extends :class:`~sentence_transformers.evaluation.RerankingEvaluator' but is specifically designed for sparse encoder models.

    This class evaluates a SparseEncoder model for the task of re-ranking.

    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10, NDCG@10 and MAP is compute to measure the quality of the ranking.

    Args:
        samples (list): A list of dictionaries, where each dictionary represents a sample and has the following keys:

            - 'query': The search query.
            - 'positive': A list of positive (relevant) documents.
            - 'negative': A list of negative (irrelevant) documents.
        at_k (int, optional): Only consider the top k most similar documents to each query for the evaluation. Defaults to 10.
        name (str, optional): Name of the evaluator. Defaults to "".
        write_csv (bool, optional): Write results to CSV file. Defaults to True.
        similarity_fct (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): Similarity function between sentence embeddings. By default, cosine similarity. Defaults to cos_sim.
        batch_size (int, optional): Batch size to compute sentence embeddings. Defaults to 64.
        show_progress_bar (bool, optional): Show progress bar when computing embeddings. Defaults to False.
        use_batched_encoding (bool, optional): Whether or not to encode queries and documents in batches for greater speed, or 1-by-1 to save memory. Defaults to True.
        max_active_dims (Optional[int], optional): The maximum number of active dimensions to use.
            `None` uses the model's current `max_active_dims`. Defaults to None.
        mrr_at_k (Optional[int], optional): Deprecated parameter. Please use `at_k` instead. Defaults to None.

    Example:
        ::

            import logging

            from datasets import load_dataset

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseRerankingEvaluator

            logging.basicConfig(format="%(message)s", level=logging.INFO)

            # Load a model
            model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

            # Load a dataset with queries, positives, and negatives
            eval_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation").select(range(1000))

            samples = [
                {
                    "query": sample["query"],
                    "positive": [
                        text
                        for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"])
                        if is_selected
                    ],
                    "negative": [
                        text
                        for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"])
                        if not is_selected
                    ],
                }
                for sample in eval_dataset
            ]


            # Now evaluate using only the documents from the 1000 samples
            reranking_evaluator = SparseRerankingEvaluator(
                samples=samples,
                name="ms-marco-dev-small",
                show_progress_bar=True,
                batch_size=32,
            )

            results = reranking_evaluator(model)
            '''
            RerankingEvaluator: Evaluating the model on the ms-marco-dev-small dataset:
            Queries: 967 	 Positives: Min 1.0, Mean 1.1, Max 3.0 	 Negatives: Min 1.0, Mean 7.1, Max 9.0
            MAP: 53.46
            MRR@10: 54.18
            NDCG@10: 65.10
            Model Sparsity Stats  Query : Row Non-Zero Mean: 43.89658737182617, Row Sparsity Mean: 0.9985617995262146
            Model Sparsity Stats  Corpus : Row Non-Zero Mean: 128.37216186523438, Row Sparsity Mean: 0.9957940578460693
            '''
            # Print the results
            print(f"Primary metric: {reranking_evaluator.primary_metric}")
            # => Primary metric: ms-marco-dev-small_ndcg@10
            print(f"Primary metric value: {results[reranking_evaluator.primary_metric]:.4f}")
            # => Primary metric value: 0.6510

    """

    def __init__(
        self,
        samples: list[dict[str, str | list[str]]],
        at_k: int = 10,
        name: str = "",
        write_csv: bool = True,
        similarity_fct: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cos_sim,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        use_batched_encoding: bool = True,
        max_active_dims: int | None = None,
        mrr_at_k: int | None = None,
    ):
        self.max_active_dims = max_active_dims
        self.sparsity_stats = {
            "row_non_zero_mean_query": 0,
            "row_sparsity_mean_query": 0,
            "row_non_zero_mean_corpus": 0,
            "row_sparsity_mean_corpus": 0,
        }
        super().__init__(
            samples=samples,
            at_k=at_k,
            name=name,
            write_csv=write_csv,
            similarity_fct=similarity_fct,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            use_batched_encoding=use_batched_encoding,
            mrr_at_k=mrr_at_k,
        )
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
    ) -> torch.Tensor:
        embeddings = model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_sparse_tensor=True,
            convert_to_tensor=False,  # as we are using slicing on sparse tensors that is not supported so we want to keep a list of sparse tensors
            save_to_cpu=True,
            max_active_dims=self.max_active_dims,
            **kwargs,
        )
        stat = model.get_sparsity_stats(torch.stack(embeddings))
        if len(self.samples) == len(sentences):
            self.sparsity_stats["row_non_zero_mean_query"] = stat["row_non_zero_mean"]
            self.sparsity_stats["row_sparsity_mean_query"] = stat["row_sparsity_mean"]
        else:
            self.sparsity_stats["row_non_zero_mean_corpus"] = stat["row_non_zero_mean"]
            self.sparsity_stats["row_sparsity_mean_corpus"] = stat["row_sparsity_mean"]
        return embeddings

    def store_metrics_in_model_card_data(
        self, model: SparseEncoder, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch=epoch, step=step)
