import csv
import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Dict, Optional

import numpy as np
import torch
import tqdm
from sklearn.metrics import average_precision_score, ndcg_score

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.util import cos_sim

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class RerankingEvaluator(SentenceEvaluator):
    """
    This class evaluates a SentenceTransformer model for the task of re-ranking.

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
        truncate_dim (Optional[int], optional): The dimension to truncate sentence embeddings to. `None` uses the model's current truncation dimension. Defaults to None.
        mrr_at_k (Optional[int], optional): Deprecated parameter. Please use `at_k` instead. Defaults to None.
    """

    def __init__(
        self,
        samples,
        at_k: int = 10,
        name: str = "",
        write_csv: bool = True,
        similarity_fct: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cos_sim,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        use_batched_encoding: bool = True,
        truncate_dim: Optional[int] = None,
        mrr_at_k: Optional[int] = None,
    ):
        super().__init__()
        self.samples = samples
        self.name = name

        if mrr_at_k is not None:
            logger.warning(f"The `mrr_at_k` parameter has been deprecated; please use `at_k={mrr_at_k}` instead.")
            self.at_k = mrr_at_k
        else:
            self.at_k = at_k

        self.similarity_fct = similarity_fct
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.use_batched_encoding = use_batched_encoding
        self.truncate_dim = truncate_dim

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        ### Remove sample with empty positive / negative set
        self.samples = [
            sample for sample in self.samples if len(sample["positive"]) > 0 and len(sample["negative"]) > 0
        ]

        self.csv_file = "RerankingEvaluator" + ("_" + name if name else "") + f"_results_@{self.at_k}.csv"
        self.csv_headers = [
            "epoch",
            "steps",
            "MAP",
            "MRR@{}".format(self.at_k),
            "NDCG@{}".format(self.at_k),
        ]
        self.write_csv = write_csv
        self.primary_metric = "map"

    def __call__(
        self, model: "SentenceTransformer", output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> Dict[str, float]:
        """
        Evaluates the model on the dataset and returns the evaluation metrics.

        Args:
            model (SentenceTransformer): The SentenceTransformer model to evaluate.
            output_path (str, optional): The output path to write the results. Defaults to None.
            epoch (int, optional): The current epoch number. Defaults to -1.
            steps (int, optional): The current step number. Defaults to -1.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"RerankingEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        scores = self.compute_metrices(model)
        mean_ap = scores["map"]
        mean_mrr = scores["mrr"]
        mean_ndcg = scores["ndcg"]

        #### Some stats about the dataset
        num_positives = [len(sample["positive"]) for sample in self.samples]
        num_negatives = [len(sample["negative"]) for sample in self.samples]

        logger.info(
            "Queries: {} \t Positives: Min {:.1f}, Mean {:.1f}, Max {:.1f} \t Negatives: Min {:.1f}, Mean {:.1f}, Max {:.1f}".format(
                len(self.samples),
                np.min(num_positives),
                np.mean(num_positives),
                np.max(num_positives),
                np.min(num_negatives),
                np.mean(num_negatives),
                np.max(num_negatives),
            )
        )
        logger.info("MAP: {:.2f}".format(mean_ap * 100))
        logger.info("MRR@{}: {:.2f}".format(self.at_k, mean_mrr * 100))
        logger.info("NDCG@{}: {:.2f}".format(self.at_k, mean_ndcg * 100))

        #### Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mean_ap, mean_mrr, mean_ndcg])

        metrics = {
            "map": mean_ap,
            f"mrr@{self.at_k}": mean_mrr,
            f"ndcg@{self.at_k}": mean_ndcg,
        }
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics)
        return metrics

    def compute_metrices(self, model):
        """
        Computes the evaluation metrics for the given model.

        Args:
            model (SentenceTransformer): The SentenceTransformer model to compute metrics for.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        return (
            self.compute_metrices_batched(model)
            if self.use_batched_encoding
            else self.compute_metrices_individual(model)
        )

    def compute_metrices_batched(self, model):
        """
        Computes the evaluation metrics in a batched way, by batching all queries and all documents together.

        Args:
            model (SentenceTransformer): The SentenceTransformer model to compute metrics for.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        all_mrr_scores = []
        all_ndcg_scores = []
        all_ap_scores = []

        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            all_query_embs = model.encode(
                [sample["query"] for sample in self.samples],
                convert_to_tensor=True,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
            )

            all_docs = []

            for sample in self.samples:
                all_docs.extend(sample["positive"])
                all_docs.extend(sample["negative"])

            all_docs_embs = model.encode(
                all_docs, convert_to_tensor=True, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar
            )

        # Compute scores
        query_idx, docs_idx = 0, 0
        for instance in self.samples:
            query_emb = all_query_embs[query_idx]
            query_idx += 1

            num_pos = len(instance["positive"])
            num_neg = len(instance["negative"])
            docs_emb = all_docs_embs[docs_idx : docs_idx + num_pos + num_neg]
            docs_idx += num_pos + num_neg

            if num_pos == 0 or num_neg == 0:
                continue

            pred_scores = self.similarity_fct(query_emb, docs_emb)
            if len(pred_scores.shape) > 1:
                pred_scores = pred_scores[0]

            pred_scores_argsort = torch.argsort(-pred_scores)  # Sort in decreasing order
            pred_scores = pred_scores.cpu().tolist()

            # Compute MRR score
            is_relevant = [1] * num_pos + [0] * num_neg
            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0 : self.at_k]):
                if is_relevant[index]:
                    mrr_score = 1 / (rank + 1)
                    break
            all_mrr_scores.append(mrr_score)

            # Compute NDCG score
            all_ndcg_scores.append(ndcg_score([is_relevant], [pred_scores], k=self.at_k))

            # Compute AP
            all_ap_scores.append(average_precision_score(is_relevant, pred_scores))

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)
        mean_ndcg = np.mean(all_ndcg_scores)

        return {"map": mean_ap, "mrr": mean_mrr, "ndcg": mean_ndcg}

    def compute_metrices_individual(self, model):
        """
        Computes the evaluation metrics individually by embedding every (query, positive, negative) tuple individually.

        Args:
            model (SentenceTransformer): The SentenceTransformer model to compute metrics for.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        all_mrr_scores = []
        all_ndcg_scores = []
        all_ap_scores = []

        for instance in tqdm.tqdm(self.samples, disable=not self.show_progress_bar, desc="Samples"):
            query = instance["query"]
            positive = list(instance["positive"])
            negative = list(instance["negative"])

            if len(positive) == 0 or len(negative) == 0:
                continue

            docs = positive + negative
            is_relevant = [1] * len(positive) + [0] * len(negative)

            with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
                query_emb = model.encode(
                    [query], convert_to_tensor=True, batch_size=self.batch_size, show_progress_bar=False
                )
                docs_emb = model.encode(
                    docs, convert_to_tensor=True, batch_size=self.batch_size, show_progress_bar=False
                )

            pred_scores = self.similarity_fct(query_emb, docs_emb)
            if len(pred_scores.shape) > 1:
                pred_scores = pred_scores[0]

            pred_scores_argsort = torch.argsort(-pred_scores)  # Sort in decreasing order
            pred_scores = pred_scores.cpu().tolist()

            # Compute MRR score
            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0 : self.at_k]):
                if is_relevant[index]:
                    mrr_score = 1 / (rank + 1)
                    break
            all_mrr_scores.append(mrr_score)

            # Compute NDCG score
            all_ndcg_scores.append(ndcg_score([is_relevant], [pred_scores], k=self.at_k))

            # Compute AP
            all_ap_scores.append(average_precision_score(is_relevant, pred_scores))

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)
        mean_ndcg = np.mean(all_ndcg_scores)

        return {"map": mean_ap, "mrr": mean_mrr, "ndcg": mean_ndcg}
