from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from sentence_transformers.evaluation.NanoBEIREvaluator import NanoBEIREvaluator
from sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator import (
    SparseInformationRetrievalEvaluator,
)

if TYPE_CHECKING:
    from torch import Tensor

    from sentence_transformers.evaluation import SimilarityFunction
    from sentence_transformers.evaluation.NanoBEIREvaluator import DatasetNameType
    from sentence_transformers.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)


class SparseNanoBEIREvaluator(NanoBEIREvaluator):
    """
    This evaluator extends :class:`NanoBEIREvaluator' but is specifically designed for sparse encoder models.

    This class evaluates the performance of a SparseEncoder Model on the NanoBEIR collection of Information Retrieval datasets.

    The collection is a set of datasets based on the BEIR collection, but with a significantly smaller size, so it can
    be used for quickly evaluating the retrieval performance of a model before commiting to a full evaluation.
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
        truncate_dim (int, optional): The dimension to truncate the embeddings to. Defaults to None.
        score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to {SimilarityFunction.COSINE.value: cos_sim, SimilarityFunction.DOT_PRODUCT.value: dot_score}.
        main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.
        aggregate_fn (Callable[[list[float]], float]): The function to aggregate the scores. Defaults to np.mean.
        aggregate_key (str): The key to use for the aggregated score. Defaults to "mean".
        query_prompts (str | dict[str, str], optional): The prompts to add to the queries. If a string, will add the same prompt to all queries. If a dict, expects that all datasets in dataset_names are keys.
        corpus_prompts (str | dict[str, str], optional): The prompts to add to the corpus. If a string, will add the same prompt to all corpus. If a dict, expects that all datasets in dataset_names are keys.

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
            Query info: num_rows: 50, num_cols: 30522, row_non_zero_mean: 62.97999954223633, row_sparsity_mean: 0.9979365468025208 1/1 [00:04<00:00,  4.12s/it]
            Corpus info: num_rows: 5046, num_cols: 30522, row_non_zero_mean: 63.394371032714844, row_sparsity_mean: 0.9979230165481567
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
            MAP@100: 0.9072

            Evaluating NanoMSMARCO
            Information Retrieval Evaluation of the model on the NanoMSMARCO dataset:
            Query info: num_rows: 50, num_cols: 30522, row_non_zero_mean: 48.099998474121094, row_sparsity_mean: 0.99842399358749391/1 [00:19<00:00, 19.40s/it]
            Corpus info: num_rows: 5043, num_cols: 30522, row_non_zero_mean: 125.38131713867188, row_sparsity_mean: 0.9958921670913696
            Score-Function: dot
            Accuracy@1: 48.00%
            Accuracy@3: 74.00%
            Accuracy@5: 76.00%
            Accuracy@10: 88.00%
            Precision@1: 48.00%
            Precision@3: 24.67%
            Precision@5: 15.20%
            Precision@10: 8.80%
            Recall@1: 48.00%
            Recall@3: 74.00%
            Recall@5: 76.00%
            Recall@10: 88.00%
            MRR@10: 0.6211
            NDCG@10: 0.6838
            MAP@100: 0.6277

            Average Querie: num_rows: 50.0, num_cols: 30522.0, row_non_zero_mean: 55.53999900817871, row_sparsity_mean: 0.9981802701950073
            Average Corpus: num_rows: 5044.5, num_cols: 30522.0, row_non_zero_mean: 94.38784408569336, row_sparsity_mean: 0.9969075918197632
            Aggregated for Score Function: dot
            Accuracy@1: 70.00%
            Accuracy@3: 85.00%
            Accuracy@5: 87.00%
            Accuracy@10: 94.00%
            Precision@1: 70.00%
            Recall@1: 63.87%
            Precision@3: 32.33%
            Recall@3: 83.27%
            Precision@5: 20.00%
            Recall@5: 85.47%
            Precision@10: 11.00%
            Recall@10: 93.13%
            MRR@10: 0.7825
            NDCG@10: 0.8089
            '''
            # Print the results
            print(f"Primary metric: {evaluator.primary_metric}")
            # => Primary metric: NanoBEIR_mean_dot_ndcg@10
            print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
            # => Primary metric value: 0.8089

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
        truncate_dim: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] = None,
        main_score_function: str | SimilarityFunction | None = None,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
        query_prompts: str | dict[str, str] | None = None,
        corpus_prompts: str | dict[str, str] | None = None,
    ):
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
            truncate_dim=truncate_dim,
            score_functions=score_functions,
            main_score_function=main_score_function,
            aggregate_fn=aggregate_fn,
            aggregate_key=aggregate_key,
            query_prompts=query_prompts,
            corpus_prompts=corpus_prompts,
        )

    def __call__(
        self, model: SparseEncoder, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs
    ) -> dict[str, float]:
        return super().__call__(model, output_path=output_path, epoch=epoch, steps=steps, *args, **kwargs)

    def _load_dataset(
        self, dataset_name: DatasetNameType, **ir_evaluator_kwargs
    ) -> SparseInformationRetrievalEvaluator:
        return super()._load_dataset(dataset_name, **ir_evaluator_kwargs)
