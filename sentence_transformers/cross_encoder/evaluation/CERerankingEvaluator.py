from __future__ import annotations

import csv
import logging
import os

import numpy as np
from sklearn.metrics import ndcg_score

logger = logging.getLogger(__name__)


class CERerankingEvaluator:
    """
    This class evaluates a CrossEncoder model for the task of re-ranking.

    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 and NDCG@10 are computed to measure the quality of the ranking.

    Args:
        samples (List[Dict, str, Union[str, List[str]]): Must be a list and each element is of the form:
            {'query': '', 'positive': [], 'negative': []}. Query is the search query, positive is a list
            of positive (relevant) documents, negative is a list of negative (irrelevant) documents.
    """

    def __init__(self, samples, at_k: int = 10, name: str = "", write_csv: bool = True, mrr_at_k: int | None = None):
        self.samples = samples
        self.name = name
        if mrr_at_k is not None:
            logger.warning(f"The `mrr_at_k` parameter has been deprecated; please use `at_k={mrr_at_k}` instead.")
            self.at_k = mrr_at_k
        else:
            self.at_k = at_k

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        self.csv_file = "CERerankingEvaluator" + ("_" + name if name else "") + f"_results_@{self.at_k}.csv"
        self.csv_headers = [
            "epoch",
            "steps",
            f"MRR@{self.at_k}",
            f"NDCG@{self.at_k}",
        ]
        self.write_csv = write_csv

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("CERerankingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        all_mrr_scores = []
        all_ndcg_scores = []
        num_queries = 0
        num_positives = []
        num_negatives = []
        for instance in self.samples:
            query = instance["query"]
            positive = list(instance["positive"])
            negative = list(instance["negative"])
            docs = positive + negative
            is_relevant = [1] * len(positive) + [0] * len(negative)

            if len(positive) == 0 or len(negative) == 0:
                continue

            num_queries += 1
            num_positives.append(len(positive))
            num_negatives.append(len(negative))

            model_input = [[query, doc] for doc in docs]
            pred_scores = model.predict(model_input, convert_to_numpy=True, show_progress_bar=False)
            pred_scores_argsort = np.argsort(-pred_scores)  # Sort in decreasing order

            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0 : self.at_k]):
                if is_relevant[index]:
                    mrr_score = 1 / (rank + 1)
                    break

            all_mrr_scores.append(mrr_score)
            all_ndcg_scores.append(ndcg_score([is_relevant], [pred_scores], k=self.at_k))

        mean_mrr = np.mean(all_mrr_scores)
        mean_ndcg = np.mean(all_ndcg_scores)

        logger.info(
            f"Queries: {num_queries} \t Positives: Min {np.min(num_positives):.1f}, Mean {np.mean(num_positives):.1f}, Max {np.max(num_positives):.1f} \t Negatives: Min {np.min(num_negatives):.1f}, Mean {np.mean(num_negatives):.1f}, Max {np.max(num_negatives):.1f}"
        )
        logger.info(f"MRR@{self.at_k}: {mean_mrr * 100:.2f}")
        logger.info(f"NDCG@{self.at_k}: {mean_ndcg * 100:.2f}")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mean_mrr, mean_ndcg])

        return mean_mrr
