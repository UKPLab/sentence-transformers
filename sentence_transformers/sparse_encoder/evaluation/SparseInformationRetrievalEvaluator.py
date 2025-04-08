from __future__ import annotations

import heapq
import logging
from contextlib import nullcontext
from typing import Callable

import torch
from torch import Tensor
from tqdm import trange

from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)


class SparseInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """

    This class evaluates an Information Retrieval (IR) setting. on sparse embeddings.

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
        truncate_dim (int, optional): The dimension to truncate the embeddings to. Defaults to None.
        score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to the ``similarity`` function from the ``model``.
        main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.
        query_prompt (str, optional): The prompt to be used when encoding the corpus. Defaults to None.
        query_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus. Defaults to None.
        corpus_prompt (str, optional): The prompt to be used when encoding the corpus. Defaults to None.
        corpus_prompt_name (str, optional): The name of the prompt to be used when encoding the corpus. Defaults to None.

    Example:
        ::

            import random
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import InformationRetrievalEvaluator
            from datasets import load_dataset

            # Load a model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Load the Touche-2020 IR dataset (https://huggingface.co/datasets/BeIR/webis-touche2020, https://huggingface.co/datasets/BeIR/webis-touche2020-qrels)
            corpus = load_dataset("BeIR/webis-touche2020", "corpus", split="corpus")
            queries = load_dataset("BeIR/webis-touche2020", "queries", split="queries")
            relevant_docs_data = load_dataset("BeIR/webis-touche2020-qrels", split="test")

            # For this dataset, we want to concatenate the title and texts for the corpus
            corpus = corpus.map(lambda x: {'text': x['title'] + " " + x['text']}, remove_columns=['title'])

            # Shrink the corpus size heavily to only the relevant documents + 30,000 random documents
            required_corpus_ids = set(map(str, relevant_docs_data["corpus-id"]))
            required_corpus_ids |= set(random.sample(corpus["_id"], k=30_000))
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

            # Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR metrics.
            ir_evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name="BeIR-touche2020-subset-test",
            )
            results = ir_evaluator(model)
            '''
            Information Retrieval Evaluation of the model on the BeIR-touche2020-test dataset:
            Queries: 49
            Corpus: 31923

            Score-Function: cosine
            Accuracy@1: 77.55%
            Accuracy@3: 93.88%
            Accuracy@5: 97.96%
            Accuracy@10: 100.00%
            Precision@1: 77.55%
            Precision@3: 72.11%
            Precision@5: 71.43%
            Precision@10: 62.65%
            Recall@1: 1.72%
            Recall@3: 4.78%
            Recall@5: 7.90%
            Recall@10: 13.86%
            MRR@10: 0.8580
            NDCG@10: 0.6606
            MAP@100: 0.2934
            '''
            print(ir_evaluator.primary_metric)
            # => "BeIR-touche2020-test_cosine_map@100"
            print(results[ir_evaluator.primary_metric])
            # => 0.29335196224364596
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
        truncate_dim: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        main_score_function: str | SimilarityFunction | None = None,
        query_prompt: str | None = None,
        query_prompt_name: str | None = None,
        corpus_prompt: str | None = None,
        corpus_prompt_name: str | None = None,
    ) -> None:
        """
        Initialize the SparseInformationRetrievalEvaluator.

        Args:
            queries: List of queries
            corpus: List of documents in the corpus
            relevant_docs: List of lists of relevant document indices for each query
            name: Name of the evaluator
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show a progress bar
            write_csv: Whether to write results to CSV
            score_function: Function to compute similarity scores
            main_score_function: Main similarity metric to use
            limit: Limit the number of examples to evaluate
        """
        super().__init__(
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
            truncate_dim=truncate_dim,
            score_functions=score_functions,
            main_score_function=main_score_function,
            query_prompt=query_prompt,
            query_prompt_name=query_prompt_name,
            corpus_prompt=corpus_prompt,
            corpus_prompt_name=corpus_prompt_name,
        )

    def compute_metrices(
        self,
        model: SparseEncoder,
        corpus_model=None,
        corpus_embeddings: Tensor | None = None,
    ) -> dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        # Compute embedding for the queries
        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            query_embeddings = model.encode(
                self.queries,
                prompt_name=self.query_prompt_name,
                prompt=self.query_prompt,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_sparse_tensor=True,
            )

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        # Iterate over chunks of the corpus
        for corpus_start_idx in trange(
            0,
            len(self.corpus),
            self.corpus_chunk_size,
            desc="Corpus Chunks",
            disable=not self.show_progress_bar,
        ):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            # Encode chunk of corpus
            if corpus_embeddings is None:
                with (
                    nullcontext()
                    if self.truncate_dim is None
                    else corpus_model.truncate_sentence_embeddings(self.truncate_dim)
                ):
                    sub_corpus_embeddings = corpus_model.encode(
                        self.corpus[corpus_start_idx:corpus_end_idx],
                        prompt_name=self.corpus_prompt_name,
                        prompt=self.corpus_prompt,
                        batch_size=self.batch_size,
                        show_progress_bar=self.show_progress_bar,
                        convert_to_sparse_tensor=True,
                    )
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            # Compute cosine similarites
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                # Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                    pair_scores,
                    min(max_k, len(pair_scores[0])),
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(
                        pair_scores_top_k_idx[query_itr],
                        pair_scores_top_k_values[query_itr],
                    ):
                        corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                        # NOTE: TREC/BEIR/MTEB skips cases where the corpus_id is the same as the query_id, e.g.:
                        # if corpus_id == self.queries_ids[query_itr]:
                        #     continue
                        # This is not done here, as this might be unexpected behaviour if the user just uses
                        # sets of integers from 0 as query_ids and corpus_ids.
                        if len(queries_result_list[name][query_itr]) < max_k:
                            # heaqp tracks the quantity of the first element in the tuple
                            heapq.heappush(queries_result_list[name][query_itr], (score, corpus_id))
                        else:
                            heapq.heappushpop(queries_result_list[name][query_itr], (score, corpus_id))

        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {
                        "corpus_id": corpus_id,
                        "score": score,
                    }

        logger.info(f"Queries: {len(self.queries)}")
        logger.info(f"Corpus: {len(self.corpus)}\n")

        # Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        # Output
        for name in self.score_function_names:
            logger.info(f"Score-Function: {name}")
            self.output_scores(scores[name])

        return scores
