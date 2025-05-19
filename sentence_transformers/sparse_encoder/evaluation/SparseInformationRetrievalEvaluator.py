from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Callable

import torch

from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import append_to_last_row

if TYPE_CHECKING:
    import numpy as np

    from sentence_transformers.similarity_functions import SimilarityFunction
    from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


logger = logging.getLogger(__name__)


class SparseInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """
    This evaluator extends :class:`~sentence_transformers.evaluation.InformationRetrievalEvaluator` but is specifically designed for sparse encoder models.

    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)

    Args:
        queries (Dict[str, str]): A dictionary mapping query IDs to queries.
        corpus (Dict[str, str]): A dictionary mapping document IDs to documents.
        relevant_docs (Dict[str, Set[str]]): A dictionary mapping query IDs to a set of relevant document IDs.
        corpus_chunk_size (int): The size of each chunk of the corpus. Defaults to 50000.
        mrr_at_k (List[int]): A list of integers representing the values of k for MRR calculation. Defaults to [10].
        ndcg_at_k (List[int]): A list of integers representing the values of k for NDCG calculation. Defaults to [10].
        accuracy_at_k (List[int]): A list of integers representing the values of k for accuracy calculation. Defaults to [1, 3, 5, 10].
        precision_recall_at_k (List[int]): A list of integers representing the values of k for precision and recall calculation. Defaults to [1, 3, 5, 10].
        map_at_k (List[int]): A list of integers representing the values of k for MAP calculation. Defaults to [100].
        show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.
        batch_size (int): The batch size for evaluation. Defaults to 32.
        name (str): A name for the evaluation. Defaults to "".
        write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
        max_active_dims (Optional[int], optional): The maximum number of active dimensions to use.
            `None` uses the model's current `max_active_dims`. Defaults to None.
        score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to the ``similarity`` function from the ``model``.
        main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.
        query_prompt (str, optional): The prompt to be used when encoding the corpus. Defaults to None.
        query_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus. Defaults to None.
        corpus_prompt (str, optional): The prompt to be used when encoding the corpus. Defaults to None.
        corpus_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus. Defaults to None.

    Example:
        ::

            import logging
            import random

            from datasets import load_dataset

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator

            logging.basicConfig(format="%(message)s", level=logging.INFO)

            # Load a model
            model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

            # Load the NFcorpus IR dataset (https://huggingface.co/datasets/BeIR/nfcorpus, https://huggingface.co/datasets/BeIR/nfcorpus-qrels)
            corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
            queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
            relevant_docs_data = load_dataset("BeIR/nfcorpus-qrels", split="test")

            # For this dataset, we want to concatenate the title and texts for the corpus
            corpus = corpus.map(lambda x: {"text": x["title"] + " " + x["text"]}, remove_columns=["title"])

            # Shrink the corpus size heavily to only the relevant documents + 1,000 random documents
            required_corpus_ids = set(map(str, relevant_docs_data["corpus-id"]))
            required_corpus_ids |= set(random.sample(corpus["_id"], k=1000))
            corpus = corpus.filter(lambda x: x["_id"] in required_corpus_ids)

            # Convert the datasets to dictionaries
            corpus = dict(zip(corpus["_id"], corpus["text"]))  # Our corpus (cid => document)
            queries = dict(zip(queries["_id"], queries["text"]))  # Our queries (qid => question)
            relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
            for qid, corpus_ids in zip(relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]):
                qid = str(qid)
                corpus_ids = str(corpus_ids)
                if qid not in relevant_docs:
                    relevant_docs[qid] = set()
                relevant_docs[qid].add(corpus_ids)

            # Given queries, a corpus and a mapping with relevant documents, the SparseInformationRetrievalEvaluator computes different IR metrics.
            ir_evaluator = SparseInformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name="BeIR-nfcorpus-subset-test",
                show_progress_bar=True,
                batch_size=16,
            )

            # Run evaluation
            results = ir_evaluator(model)
            '''
            Queries: 323
            Corpus: 3269

            Score-Function: dot
            Accuracy@1: 50.46%
            Accuracy@3: 64.40%
            Accuracy@5: 67.49%
            Accuracy@10: 72.14%
            Precision@1: 50.46%
            Precision@3: 40.87%
            Precision@5: 34.12%
            Precision@10: 26.10%
            Recall@1: 6.11%
            Recall@3: 11.73%
            Recall@5: 13.64%
            Recall@10: 17.24%
            MRR@10: 0.5801
            NDCG@10: 0.3626
            MAP@100: 0.1832
            Model Sparsity Stats  Query : Row Non-Zero Mean: 43.08049392700195, Row Sparsity Mean: 0.9985886216163635
            Model Sparsity Stats  Corpus : Row Non-Zero Mean: 206.8623504638672, Row Sparsity Mean: 0.9932224750518799
            '''
            # Print the results
            print(f"Primary metric: {ir_evaluator.primary_metric}")
            # => Primary metric: BeIR-nfcorpus-subset-test_dot_ndcg@10
            print(f"Primary metric value: {results[ir_evaluator.primary_metric]:.4f}")
            # => Primary metric value: 0.3626

    """

    def __init__(
        self,
        queries: dict[str, str],  # qid => query
        corpus: dict[str, str],  # cid => doc
        relevant_docs: dict[str, set[str]],  # qid => Set[cid]
        corpus_chunk_size: int = 50000,
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        max_active_dims: int | None = None,
        score_functions: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None = None,
        main_score_function: str | SimilarityFunction | None = None,
        query_prompt: str | None = None,
        query_prompt_name: str | None = None,
        corpus_prompt: str | None = None,
        corpus_prompt_name: str | None = None,
    ) -> None:
        self.max_active_dims = max_active_dims
        self.sparsity_stats = {
            "row_non_zero_mean_query": 0,
            "row_sparsity_mean_query": 0,
            "row_non_zero_mean_corpus": 0,
            "row_sparsity_mean_corpus": 0,
        }
        return super().__init__(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            corpus_chunk_size=corpus_chunk_size,
            mrr_at_k=mrr_at_k,
            ndcg_at_k=ndcg_at_k,
            accuracy_at_k=accuracy_at_k,
            precision_recall_at_k=precision_recall_at_k,
            map_at_k=map_at_k,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            name=name,
            write_csv=write_csv,
            score_functions=score_functions,
            main_score_function=main_score_function,
            query_prompt=query_prompt,
            query_prompt_name=query_prompt_name,
            corpus_prompt=corpus_prompt,
            corpus_prompt_name=corpus_prompt_name,
        )

    def _append_csv_headers(self, similarity_fn_names):
        super()._append_csv_headers(similarity_fn_names)
        for sparsity_stat in self.sparsity_stats.keys():
            self.csv_headers.append(f"{sparsity_stat}")

    def __call__(
        self, model: SparseEncoder, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs
    ) -> dict[str, float]:
        self.sparsity_stats = {
            "row_non_zero_mean_query": 0,
            "row_sparsity_mean_query": 0,
            "row_non_zero_mean_corpus": 0,
            "row_sparsity_mean_corpus": 0,
        }
        self.count_corpus_seen = 0
        metrics = super().__call__(model=model, output_path=output_path, epoch=epoch, steps=steps)

        metrics.update(self.prefix_name_to_metrics(self.sparsity_stats, self.name))
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        logger.info(
            f"Model Sparsity Stats  Query : Row Non-Zero Mean: {self.sparsity_stats['row_non_zero_mean_query']}, Row Sparsity Mean: {self.sparsity_stats['row_sparsity_mean_query']}"
        )
        logger.info(
            f"Model Sparsity Stats  Corpus : Row Non-Zero Mean: {self.sparsity_stats['row_non_zero_mean_corpus']}, Row Sparsity Mean: {self.sparsity_stats['row_sparsity_mean_corpus']}"
        )
        if output_path is not None and self.write_csv:
            append_to_last_row(
                os.path.join(output_path, self.csv_file),
                self.sparsity_stats.values(),
            )

        return metrics

    def compute_metrices(
        self, model: SparseEncoder, corpus_model=None, corpus_embeddings: torch.Tensor | None = None
    ) -> dict[str, float]:
        return super().compute_metrices(model=model, corpus_model=corpus_model, corpus_embeddings=corpus_embeddings)

    def embed_inputs(
        self,
        model: SparseEncoder,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = None,
        prompt: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        embeddings = model.encode(
            sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
            max_active_dims=self.max_active_dims,
            **kwargs,
        )
        stat = model.get_sparsity_stats(embeddings)
        if len(self.queries) == len(sentences) and self.sparsity_stats["row_non_zero_mean_query"] == 0:
            self.sparsity_stats["row_non_zero_mean_query"] = stat["row_non_zero_mean"]
            self.sparsity_stats["row_sparsity_mean_query"] = stat["row_sparsity_mean"]
        else:
            if self.sparsity_stats["row_non_zero_mean_corpus"] == 0:
                self.sparsity_stats["row_non_zero_mean_corpus"] = stat["row_non_zero_mean"]
                self.sparsity_stats["row_sparsity_mean_corpus"] = stat["row_sparsity_mean"]
                self.count_corpus_seen = len(sentences)
            else:
                # do the weighted average
                self.sparsity_stats["row_non_zero_mean_corpus"] = (
                    self.sparsity_stats["row_non_zero_mean_corpus"] * self.count_corpus_seen
                    + stat["row_non_zero_mean"] * len(sentences)
                ) / (self.count_corpus_seen + len(sentences))
                self.sparsity_stats["row_sparsity_mean_corpus"] = (
                    self.sparsity_stats["row_sparsity_mean_corpus"] * self.count_corpus_seen
                    + stat["row_sparsity_mean"] * len(sentences)
                ) / (self.count_corpus_seen + len(sentences))
                self.count_corpus_seen += len(sentences)
        return embeddings

    def store_metrics_in_model_card_data(
        self, model: SparseEncoder, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch=epoch, step=step)
