from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from sentence_transformers.evaluation.NanoBEIREvaluator import NanoBEIREvaluator
from sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator import (
    SparseInformationRetrievalEvaluator,
)
from sentence_transformers.util import append_to_last_row

if TYPE_CHECKING:
    from torch import Tensor

    from sentence_transformers.evaluation import SimilarityFunction
    from sentence_transformers.evaluation.NanoBEIREvaluator import DatasetNameType
    from sentence_transformers.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)


class SparseNanoBEIREvaluator(NanoBEIREvaluator):
    """
    This evaluator extends :class:`~sentence_transformers.evaluation.NanoBEIREvaluator` but is specifically designed for sparse encoder models.

    This class evaluates the performance of a SparseEncoder Model on the NanoBEIR collection of Information Retrieval datasets.

    The collection is a set of datasets based on the BEIR collection, but with a significantly smaller size, so it can
    be used for quickly evaluating the retrieval performance of a model before committing to a full evaluation.
    The datasets are available on Hugging Face in the `NanoBEIR collection <https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6>`_.
    This evaluator will return the same metrics as the InformationRetrievalEvaluator (i.e., MRR, nDCG, Recall@k), for each dataset and on average.

    Args:
        dataset_names (List[str]): The names of the datasets to evaluate on. Defaults to all datasets.
        mrr_at_k (List[int]): A list of integers representing the values of k for MRR calculation. Defaults to [10].
        ndcg_at_k (List[int]): A list of integers representing the values of k for NDCG calculation. Defaults to [10].
        accuracy_at_k (List[int]): A list of integers representing the values of k for accuracy calculation. Defaults to [1, 3, 5, 10].
        precision_recall_at_k (List[int]): A list of integers representing the values of k for precision and recall calculation. Defaults to [1, 3, 5, 10].
        map_at_k (List[int]): A list of integers representing the values of k for MAP calculation. Defaults to [100].
        show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.
        batch_size (int): The batch size for evaluation. Defaults to 32.
        write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
        max_active_dims (Optional[int], optional): The maximum number of active dimensions to use.
            `None` uses the model's current `max_active_dims`. Defaults to None.
        score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to {SimilarityFunction.COSINE.value: cos_sim, SimilarityFunction.DOT_PRODUCT.value: dot_score}.
        main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.
        aggregate_fn (Callable[[list[float]], float]): The function to aggregate the scores. Defaults to np.mean.
        aggregate_key (str): The key to use for the aggregated score. Defaults to "mean".
        query_prompts (str | dict[str, str], optional): The prompts to add to the queries. If a string, will add the same prompt to all queries. If a dict, expects that all datasets in dataset_names are keys.
        corpus_prompts (str | dict[str, str], optional): The prompts to add to the corpus. If a string, will add the same prompt to all corpus. If a dict, expects that all datasets in dataset_names are keys.
        write_predictions (bool): Whether to write the predictions to a JSONL file. Defaults to False.
            This can be useful for downstream evaluation as it can be used as input to the :class:`~sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator` that accept precomputed predictions.

    Example:
        ::

            import logging

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

            logging.basicConfig(format="%(message)s", level=logging.INFO)

            # Load a model
            model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

            datasets = ["QuoraRetrieval", "MSMARCO"]

            evaluator = SparseNanoBEIREvaluator(
                dataset_names=datasets,
                show_progress_bar=True,
                batch_size=32,
            )

            # Run evaluation
            results = evaluator(model)
            '''
            Evaluating NanoQuoraRetrieval
            Information Retrieval Evaluation of the model on the NanoQuoraRetrieval dataset:
            Queries: 50
            Corpus: 5046

            Score-Function: dot
            Accuracy@1: 92.00%
            Accuracy@3: 96.00%
            Accuracy@5: 98.00%
            Accuracy@10: 100.00%
            Precision@1: 92.00%
            Precision@3: 40.00%
            Precision@5: 24.80%
            Precision@10: 13.20%
            Recall@1: 79.73%
            Recall@3: 92.53%
            Recall@5: 94.93%
            Recall@10: 98.27%
            MRR@10: 0.9439
            NDCG@10: 0.9339
            MAP@100: 0.9070
            Model Query Sparsity: Active Dimensions: 59.4, Sparsity Ratio: 0.9981
            Model Corpus Sparsity: Active Dimensions: 61.9, Sparsity Ratio: 0.9980

            Information Retrieval Evaluation of the model on the NanoMSMARCO dataset:
            Queries: 50
            Corpus: 5043

            Score-Function: dot
            Accuracy@1: 48.00%
            Accuracy@3: 74.00%
            Accuracy@5: 76.00%
            Accuracy@10: 86.00%
            Precision@1: 48.00%
            Precision@3: 24.67%
            Precision@5: 15.20%
            Precision@10: 8.60%
            Recall@1: 48.00%
            Recall@3: 74.00%
            Recall@5: 76.00%
            Recall@10: 86.00%
            MRR@10: 0.6191
            NDCG@10: 0.6780
            MAP@100: 0.6277
            Model Query Sparsity: Active Dimensions: 45.4, Sparsity Ratio: 0.9985
            Model Corpus Sparsity: Active Dimensions: 122.6, Sparsity Ratio: 0.9960

            Average Queries: 50.0
            Average Corpus: 5044.5
            Aggregated for Score Function: dot
            Accuracy@1: 70.00%
            Accuracy@3: 85.00%
            Accuracy@5: 87.00%
            Accuracy@10: 93.00%
            Precision@1: 70.00%
            Recall@1: 63.87%
            Precision@3: 32.33%
            Recall@3: 83.27%
            Precision@5: 20.00%
            Recall@5: 85.47%
            Precision@10: 10.90%
            Recall@10: 92.13%
            MRR@10: 0.7815
            NDCG@10: 0.8060
            Model Query Sparsity: Active Dimensions: 52.4, Sparsity Ratio: 0.9983
            Model Corpus Sparsity: Active Dimensions: 92.2, Sparsity Ratio: 0.9970
            '''
            # Print the results
            print(f"Primary metric: {evaluator.primary_metric}")
            # => Primary metric: NanoBEIR_mean_dot_ndcg@10
            print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
            # => Primary metric value: 0.8060

    """

    information_retrieval_class = SparseInformationRetrievalEvaluator

    def __init__(
        self,
        dataset_names: list[DatasetNameType] | None = None,
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        write_csv: bool = True,
        max_active_dims: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        main_score_function: str | SimilarityFunction | None = None,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
        query_prompts: str | dict[str, str] | None = None,
        corpus_prompts: str | dict[str, str] | None = None,
        write_predictions: bool = False,
    ):
        self.max_active_dims = max_active_dims
        self.sparsity_stats = defaultdict(list)
        super().__init__(
            dataset_names=dataset_names,
            mrr_at_k=mrr_at_k,
            ndcg_at_k=ndcg_at_k,
            accuracy_at_k=accuracy_at_k,
            precision_recall_at_k=precision_recall_at_k,
            map_at_k=map_at_k,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            write_csv=write_csv,
            score_functions=score_functions,
            main_score_function=main_score_function,
            aggregate_fn=aggregate_fn,
            aggregate_key=aggregate_key,
            query_prompts=query_prompts,
            corpus_prompts=corpus_prompts,
            write_predictions=write_predictions,
        )
        if self.max_active_dims is not None:
            self.name += f"_{self.max_active_dims}"

    def _get_human_readable_name(self, dataset_name: DatasetNameType) -> str:
        human_readable_name = super()._get_human_readable_name(dataset_name)
        if self.max_active_dims is not None:
            human_readable_name += f"_{self.max_active_dims}"
        return human_readable_name

    def _append_csv_headers(self, similarity_fn_names):
        super()._append_csv_headers(similarity_fn_names)
        self.csv_headers.extend(
            ["query_active_dims", "query_sparsity_ratio", "corpus_active_dims", "corpus_sparsity_ratio"]
        )

    def __call__(
        self, model: SparseEncoder, output_path: str | None = None, epoch: int = -1, steps: int = -1, *args, **kwargs
    ) -> dict[str, float]:
        self.sparsity_stats = defaultdict(list)
        self.lengths = defaultdict(list)
        per_dataset_results = super().__call__(
            model, output_path=output_path, epoch=epoch, steps=steps, *args, **kwargs
        )
        for evaluator in self.evaluators:
            self.lengths["query"].append(len(evaluator.queries))
            self.lengths["corpus"].append(len(evaluator.corpus))
            for key, value in evaluator.sparsity_stats.items():
                self.sparsity_stats[key].append(value)
        for key, value in self.sparsity_stats.items():
            self.sparsity_stats[key] = sum(
                val * length for val, length in zip(value, self.lengths[key.split("_")[0]])
            ) / sum(self.lengths[key.split("_")[0]])

        per_dataset_results.update(self.prefix_name_to_metrics(self.sparsity_stats, self.name))
        aggregated_results = {
            key: value for key, value in per_dataset_results.items() if key.startswith(self.name) and key != self.name
        }
        self.store_metrics_in_model_card_data(model, aggregated_results, epoch, steps)
        logger.info(
            f"Model Query Sparsity: Active Dimensions: {self.sparsity_stats['query_active_dims']:.1f}, Sparsity Ratio: {self.sparsity_stats['query_sparsity_ratio']:.4f}"
        )
        logger.info(
            f"Model Corpus Sparsity: Active Dimensions: {self.sparsity_stats['corpus_active_dims']:.1f}, Sparsity Ratio: {self.sparsity_stats['corpus_sparsity_ratio']:.4f}"
        )
        if output_path is not None and self.write_csv:
            append_to_last_row(
                os.path.join(output_path, self.csv_file),
                self.sparsity_stats.values(),
            )

        return per_dataset_results

    def _load_dataset(
        self, dataset_name: DatasetNameType, **ir_evaluator_kwargs
    ) -> SparseInformationRetrievalEvaluator:
        ir_evaluator_kwargs["max_active_dims"] = self.max_active_dims
        ir_evaluator_kwargs.pop("truncate_dim", None)
        return super()._load_dataset(dataset_name, **ir_evaluator_kwargs)

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = super().get_config_dict()
        if self.max_active_dims is not None:
            config_dict["max_active_dims"] = self.max_active_dims
        return config_dict
