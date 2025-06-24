from __future__ import annotations

import logging
import os
from collections import defaultdict
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
        write_predictions (bool): Whether to write the predictions to a JSONL file. Defaults to False.
            This can be useful for downstream evaluation as it can be used as input to the :class:`~sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator` that accept precomputed predictions.

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
            Accuracy@1: 50.77%
            Accuracy@3: 64.40%
            Accuracy@5: 66.87%
            Accuracy@10: 71.83%
            Precision@1: 50.77%
            Precision@3: 40.45%
            Precision@5: 34.06%
            Precision@10: 25.98%
            Recall@1: 6.27%
            Recall@3: 11.69%
            Recall@5: 13.74%
            Recall@10: 17.23%
            MRR@10: 0.5814
            NDCG@10: 0.3621
            MAP@100: 0.1838
            Model Query Sparsity: Active Dimensions: 40.0, Sparsity Ratio: 0.9987
            Model Corpus Sparsity: Active Dimensions: 206.2, Sparsity Ratio: 0.9932
            '''
            # Print the results
            print(f"Primary metric: {ir_evaluator.primary_metric}")
            # => Primary metric: BeIR-nfcorpus-subset-test_dot_ndcg@10
            print(f"Primary metric value: {results[ir_evaluator.primary_metric]:.4f}")
            # => Primary metric value: 0.3621

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
        write_predictions: bool = False,
    ) -> None:
        self.max_active_dims = max_active_dims
        self.sparsity_stats = {"query": defaultdict(list), "corpus": defaultdict(list)}
        self.corpus_lengths = []
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
            write_predictions=write_predictions,
        )

    def _append_csv_headers(self, similarity_fn_names):
        super()._append_csv_headers(similarity_fn_names)
        self.csv_headers.extend(
            ["query_active_dims", "query_sparsity_ratio", "corpus_active_dims", "corpus_sparsity_ratio"]
        )

    def __call__(
        self, model: SparseEncoder, output_path: str | None = None, epoch: int = -1, steps: int = -1, *args, **kwargs
    ) -> dict[str, float]:
        self.sparsity_stats = {"query": defaultdict(list), "corpus": defaultdict(list)}
        self.corpus_lengths = []
        metrics = super().__call__(model=model, output_path=output_path, epoch=epoch, steps=steps)
        for prefix in ["query", "corpus"]:
            for key, value in self.sparsity_stats[prefix].items():
                if prefix == "query":
                    self.sparsity_stats[prefix][key] = sum(value) / len(value)
                else:
                    self.sparsity_stats[prefix][key] = sum(
                        val * length for val, length in zip(value, self.corpus_lengths)
                    ) / sum(self.corpus_lengths)
        self.sparsity_stats = {
            f"{prefix}_{key}": value for prefix, values in self.sparsity_stats.items() for key, value in values.items()
        }
        metrics.update(self.prefix_name_to_metrics(self.sparsity_stats, self.name))
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        logger.info(
            f"Model Query Sparsity: Active Dimensions: {self.sparsity_stats['query_active_dims']:.1f}, Sparsity Ratio: {self.sparsity_stats['query_sparsity_ratio']:.4f}"
        )
        logger.info(
            f"Model Corpus Sparsity: Active Dimensions: {self.sparsity_stats['corpus_active_dims']:.1f}, Sparsity Ratio: {self.sparsity_stats['corpus_sparsity_ratio']:.4f}"
        )
        if output_path is not None and self.write_csv:
            append_to_last_row(os.path.join(output_path, self.csv_file), self.sparsity_stats.values())

        return metrics

    def compute_metrices(
        self,
        model: SparseEncoder,
        corpus_model=None,
        corpus_embeddings: torch.Tensor | None = None,
        output_path: str | None = None,
    ) -> dict[str, float]:
        return super().compute_metrices(
            model=model, corpus_model=corpus_model, corpus_embeddings=corpus_embeddings, output_path=output_path
        )

    def embed_inputs(
        self,
        model: SparseEncoder,
        sentences: str | list[str] | np.ndarray,
        encode_fn_name: str | None = None,
        prompt_name: str | None = None,
        prompt: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encode_fn_name is None:
            encode_fn = model.encode
        elif encode_fn_name == "query":
            encode_fn = model.encode_query
        elif encode_fn_name == "document":
            encode_fn = model.encode_document
        embeddings = encode_fn(
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
        stat = model.sparsity(embeddings)
        prefix = "query" if encode_fn_name in ["query", None] else "corpus"
        for key, value in stat.items():
            self.sparsity_stats[prefix][key].append(value)
        if prefix == "corpus":
            self.corpus_lengths.append(len(sentences))
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
