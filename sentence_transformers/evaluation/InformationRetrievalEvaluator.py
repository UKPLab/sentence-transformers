from __future__ import annotations

import heapq
import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from torch import Tensor
from tqdm import trange

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import cos_sim, dot_score

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class InformationRetrievalEvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)

    Example:
        ::

            import random
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import InformationRetrievalEvaluator
            from datasets import load_dataset

            # Load a model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Load the Quora IR dataset (https://huggingface.co/datasets/BeIR/quora, https://huggingface.co/datasets/BeIR/quora-qrels)
            corpus = load_dataset("BeIR/quora", "corpus", split="corpus")
            queries = load_dataset("BeIR/quora", "queries", split="queries")
            relevant_docs_data = load_dataset("BeIR/quora-qrels", split="validation")

            # Shrink the corpus size heavily to only the relevant documents + 10,000 random documents
            required_corpus_ids = list(map(str, relevant_docs_data["corpus-id"]))
            required_corpus_ids += random.sample(corpus["_id"], k=10_000)
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
                name="BeIR-quora-dev",
            )
            results = ir_evaluator(model)
            '''
            Information Retrieval Evaluation of the model on the BeIR-quora-dev dataset:
            Queries: 5000
            Corpus: 17476

            Score-Function: cosine
            Accuracy@1: 96.26%
            Accuracy@3: 99.38%
            Accuracy@5: 99.74%
            Accuracy@10: 99.94%
            Precision@1: 96.26%
            Precision@3: 43.01%
            Precision@5: 27.66%
            Precision@10: 14.58%
            Recall@1: 82.93%
            Recall@3: 96.28%
            Recall@5: 98.38%
            Recall@10: 99.55%
            MRR@10: 0.9782
            NDCG@10: 0.9807
            MAP@100: 0.9732
            Score-Function: dot
            Accuracy@1: 96.26%
            Accuracy@3: 99.38%
            Accuracy@5: 99.74%
            Accuracy@10: 99.94%
            Precision@1: 96.26%
            Precision@3: 43.01%
            Precision@5: 27.66%
            Precision@10: 14.58%
            Recall@1: 82.93%
            Recall@3: 96.28%
            Recall@5: 98.38%
            Recall@10: 99.55%
            MRR@10: 0.9782
            NDCG@10: 0.9807
            MAP@100: 0.9732
            '''
            print(ir_evaluator.primary_metric)
            # => "BeIR-quora-dev_cosine_map@100"
            print(results[ir_evaluator.primary_metric])
            # => 0.9732046108457585
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
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
            SimilarityFunction.COSINE.value: cos_sim,
            SimilarityFunction.DOT_PRODUCT.value: dot_score,
        },  # Score function, higher=more similar
        main_score_function: str | SimilarityFunction | None = None,
    ) -> None:
        """
        Initializes the InformationRetrievalEvaluator.

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
            score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to {SimilarityFunction.COSINE.value: cos_sim, SimilarityFunction.DOT_PRODUCT.value: dot_score}.
            main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.
        """
        super().__init__()
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = SimilarityFunction(main_score_function) if main_score_function else None
        self.truncate_dim = truncate_dim

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in accuracy_at_k:
                self.csv_headers.append(f"{score_name}-Accuracy@{k}")

            for k in precision_recall_at_k:
                self.csv_headers.append(f"{score_name}-Precision@{k}")
                self.csv_headers.append(f"{score_name}-Recall@{k}")

            for k in mrr_at_k:
                self.csv_headers.append(f"{score_name}-MRR@{k}")

            for k in ndcg_at_k:
                self.csv_headers.append(f"{score_name}-NDCG@{k}")

            for k in map_at_k:
                self.csv_headers.append(f"{score_name}-MAP@{k}")

    def __call__(
        self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"Information Retrieval Evaluation of the model on the {self.name} dataset{out_txt}:")

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]["accuracy@k"][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]["precision@k"][k])
                    output_data.append(scores[name]["recall@k"][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]["mrr@k"][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]["ndcg@k"][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]["map@k"][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [(name, scores[name]["map@k"][max(self.map_at_k)]) for name in self.score_function_names],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_map@{max(self.map_at_k)}"
            else:
                self.primary_metric = f"{self.main_score_function.value}_map@{max(self.map_at_k)}"

        metrics = {
            f"{score_function}_{metric_name.replace('@k', '@' + str(k))}": value
            for score_function, values_dict in scores.items()
            for metric_name, values in values_dict.items()
            for k, value in values.items()
        }
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics)
        return metrics

    def compute_metrices(
        self, model: SentenceTransformer, corpus_model=None, corpus_embeddings: Tensor = None
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
                show_progress_bar=self.show_progress_bar,
                batch_size=self.batch_size,
                convert_to_tensor=True,
            )

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        # Iterate over chunks of the corpus
        for corpus_start_idx in trange(
            0, len(self.corpus), self.corpus_chunk_size, desc="Corpus Chunks", disable=not self.show_progress_bar
        ):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            # Encode chunk of corpus
            if corpus_embeddings is None:
                with nullcontext() if self.truncate_dim is None else corpus_model.truncate_sentence_embeddings(
                    self.truncate_dim
                ):
                    sub_corpus_embeddings = corpus_model.encode(
                        self.corpus[corpus_start_idx:corpus_end_idx],
                        show_progress_bar=False,
                        batch_size=self.batch_size,
                        convert_to_tensor=True,
                    )
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            # Compute cosine similarites
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                # Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                    pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False
                )
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(
                        pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]
                    ):
                        corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                        if len(queries_result_list[name][query_itr]) < max_k:
                            heapq.heappush(
                                queries_result_list[name][query_itr], (score, corpus_id)
                            )  # heaqp tracks the quantity of the first element in the tuple
                        else:
                            heapq.heappushpop(queries_result_list[name][query_itr], (score, corpus_id))

        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {"corpus_id": corpus_id, "score": score}

        logger.info(f"Queries: {len(self.queries)}")
        logger.info(f"Corpus: {len(self.corpus)}\n")

        # Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        # Output
        for name in self.score_function_names:
            logger.info(f"Score-Function: {name}")
            self.output_scores(scores[name])

        return scores

    def compute_metrics(self, queries_result_list: list[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x["score"], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is across the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
        }

    def output_scores(self, scores):
        for k in scores["accuracy@k"]:
            logger.info("Accuracy@{}: {:.2f}%".format(k, scores["accuracy@k"][k] * 100))

        for k in scores["precision@k"]:
            logger.info("Precision@{}: {:.2f}%".format(k, scores["precision@k"][k] * 100))

        for k in scores["recall@k"]:
            logger.info("Recall@{}: {:.2f}%".format(k, scores["recall@k"][k] * 100))

        for k in scores["mrr@k"]:
            logger.info("MRR@{}: {:.4f}".format(k, scores["mrr@k"][k]))

        for k in scores["ndcg@k"]:
            logger.info("NDCG@{}: {:.4f}".format(k, scores["ndcg@k"][k]))

        for k in scores["map@k"]:
            logger.info("MAP@{}: {:.4f}".format(k, scores["map@k"][k]))

    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg
