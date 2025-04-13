from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from sentence_transformers.evaluation.NanoBEIREvaluator import NanoBEIREvaluator, dataset_name_to_id
from sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator import (
    SparseInformationRetrievalEvaluator,
)
from sentence_transformers.util import is_datasets_available

if TYPE_CHECKING:
    from torch import Tensor

    from sentence_transformers.evaluation import SimilarityFunction
    from sentence_transformers.evaluation.NanoBEIREvaluator import (
        DatasetNameType,
    )
    from sentence_transformers.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)


class SparseNanoBEIREvaluator(NanoBEIREvaluator):
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
        return super().__call__(model, output_path, epoch, steps, *args, **kwargs)

    def _load_dataset(
        self, dataset_name: DatasetNameType, **ir_evaluator_kwargs
    ) -> SparseInformationRetrievalEvaluator:
        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Please install it to use the SparseNanoBEIREvaluator via `pip install datasets`."
            )
        from datasets import load_dataset

        dataset_path = dataset_name_to_id[dataset_name.lower()]
        corpus = load_dataset(dataset_path, "corpus", split="train")
        queries = load_dataset(dataset_path, "queries", split="train")
        qrels = load_dataset(dataset_path, "qrels", split="train")
        corpus_dict = {sample["_id"]: sample["text"] for sample in corpus if len(sample["text"]) > 0}
        queries_dict = {sample["_id"]: sample["text"] for sample in queries if len(sample["text"]) > 0}
        qrels_dict = {}
        for sample in qrels:
            if sample["query-id"] not in qrels_dict:
                qrels_dict[sample["query-id"]] = set()
            qrels_dict[sample["query-id"]].add(sample["corpus-id"])

        if self.query_prompts is not None:
            ir_evaluator_kwargs["query_prompt"] = self.query_prompts.get(dataset_name, None)
        if self.corpus_prompts is not None:
            ir_evaluator_kwargs["corpus_prompt"] = self.corpus_prompts.get(dataset_name, None)
        human_readable_name = self._get_human_readable_name(dataset_name)
        return SparseInformationRetrievalEvaluator(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=qrels_dict,
            name=human_readable_name,
            **ir_evaluator_kwargs,
        )

    def store_metrics_in_model_card_data(
        self, model: SparseEncoder, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch, step)
