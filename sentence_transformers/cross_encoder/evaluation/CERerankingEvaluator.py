from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder

logger = logging.getLogger(__name__)


class CERerankingEvaluator(SentenceEvaluator):
    """
    This class evaluates a CrossEncoder model for the task of re-ranking.

    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10, NDCG@10 and MAP are computed to measure the quality of the ranking.

    TODO: Mention that 100 is max because all positives are included.
    TODO: Clarify what happens with negatives_are_ranked

    Args:
        samples (list): A list of dictionaries, where each dictionary represents a sample and has the following keys:
            - 'query': The search query.
            - 'positive': A list of positive (relevant) documents.
            - 'negative': A list of negative (irrelevant) documents.
        at_k (int, optional): Only consider the top k most similar documents to each query for the evaluation. Defaults to 10.
        negatives_are_ranked (bool, optional): Whether the negative documents are already ranked. If True, the negative
            documents are assumed to be all documents, including the positive ones. If True, evaluate both the reranked
            and the original ranking, and log the results side by side. Defaults to False.
        name (str, optional): Name of the evaluator, used for logging, saving in a CSV, and the model card. Defaults to "".
        batch_size (int, optional): Batch size to compute sentence embeddings. Defaults to 64.
        show_progress_bar (bool, optional): Show progress bar when computing embeddings. Defaults to False.
        write_csv (bool, optional): Write results to CSV file. Defaults to True.
        mrr_at_k (Optional[int], optional): Deprecated parameter. Please use `at_k` instead. Defaults to None.

    Example:
        ::

            from sentence_transformers import CrossEncoder
            from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
            from datasets import load_dataset

            # Load a model
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            # Load a dataset with queries, positives, and negatives
            eval_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

            samples = [
                {
                    "query": sample["query"],
                    "positive": [text for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"]) if is_selected],
                    "negative": [text for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"]) if not is_selected],
                }
                for sample in eval_dataset
            ]

            # Initialize the evaluator
            reranking_evaluator = CERerankingEvaluator(
                samples=samples,
                name="ms-marco-dev",
            )
            results = reranking_evaluator(model)
            '''
            CERerankingEvaluator: Evaluating the model on the ms-marco-dev dataset:
            Queries: 9706      Positives: Min 1.0, Mean 1.1, Max 5.0   Negatives: Min 1.0, Mean 7.1, Max 9.0
            MAP: 64.46
            MRR@10: 65.18
            NDCG@10: 73.51
            '''
            print(reranking_evaluator.primary_metric)
            # => ms-marco-dev_ndcg@10
            print(results[reranking_evaluator.primary_metric])
            # => 0.735088966736088
    """

    def __init__(
        self,
        samples: list[dict[str, str | list[str]]],
        at_k: int = 10,
        negatives_are_ranked: bool = False,
        name: str = "",
        batch_size: int = 64,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        mrr_at_k: int | None = None,
    ):
        super().__init__()
        self.samples = samples
        if mrr_at_k is not None:
            logger.warning(f"The `mrr_at_k` parameter has been deprecated; please use `at_k={mrr_at_k}` instead.")
            self.at_k = mrr_at_k
        else:
            self.at_k = at_k
        self.negatives_are_ranked = negatives_are_ranked
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        self.csv_file = "CERerankingEvaluator" + ("_" + name if name else "") + f"_results_@{self.at_k}.csv"
        self.csv_headers = [
            "epoch",
            "steps",
            "MAP",
            f"MRR@{self.at_k}",
            f"NDCG@{self.at_k}",
        ]
        self.write_csv = write_csv
        self.primary_metric = f"ndcg@{self.at_k}"  # TODO: Is this the best default metric to use?

    def __call__(self, model: CrossEncoder, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(f"CERerankingEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        base_mrr_scores = []
        base_ndcg_scores = []
        base_ap_scores = []
        all_mrr_scores = []
        all_ndcg_scores = []
        all_ap_scores = []
        num_queries = 0
        num_positives = []
        num_negatives = []
        for instance in self.samples:
            query = instance["query"]
            positive = list(instance["positive"])
            negative = list(instance["negative"])

            if self.negatives_are_ranked:
                # This means that 'negative' is all documents, including the positive ones
                base_is_relevant = [int(sample in positive) for sample in negative]
                # If not all positives are in negative, we need to add them at the end
                base_is_relevant += [1] * (len(positive) - sum(base_is_relevant))
                base_pred_scores = np.array(range(len(base_is_relevant), 0, -1))
                base_mrr, base_ndcg, base_ap = self.compute_metrics(base_is_relevant, base_pred_scores)
                base_mrr_scores.append(base_mrr)
                base_ndcg_scores.append(base_ndcg)
                base_ap_scores.append(base_ap)

                negative = [sample for sample in negative if sample not in positive]

            docs = positive + negative
            is_relevant = [1] * len(positive) + [0] * len(negative)

            if len(positive) == 0 or len(negative) == 0:
                continue

            num_queries += 1
            num_positives.append(len(positive))
            num_negatives.append(len(negative))

            model_input = [[query, doc] for doc in docs]
            pred_scores = model.predict(model_input, convert_to_numpy=True, show_progress_bar=False)

            mrr, ndcg, ap = self.compute_metrics(is_relevant, pred_scores)

            all_mrr_scores.append(mrr)
            all_ndcg_scores.append(ndcg)
            all_ap_scores.append(ap)

        mean_mrr = np.mean(all_mrr_scores)
        mean_ndcg = np.mean(all_ndcg_scores)
        mean_ap = np.mean(all_ap_scores)
        metrics = {
            "map": mean_ap,
            f"mrr@{self.at_k}": mean_mrr,
            f"ndcg@{self.at_k}": mean_ndcg,
        }
        # metrics = self.prefix_name_to_metrics(metrics, self.name)
        # self.store_metrics_in_model_card_data(model, metrics, epoch, steps)

        logger.info(
            f"Queries:\t{num_queries} \t Positives: Min {np.min(num_positives):.1f}, Mean {np.mean(num_positives):.1f}, Max {np.max(num_positives):.1f} \t Negatives: Min {np.min(num_negatives):.1f}, Mean {np.mean(num_negatives):.1f}, Max {np.max(num_negatives):.1f}"
        )
        if self.negatives_are_ranked:
            mean_base_mrr = np.mean(base_mrr_scores)
            mean_base_ndcg = np.mean(base_ndcg_scores)
            mean_base_ap = np.mean(base_ap_scores)
            base_metrics = {
                "base_map": mean_base_ap,
                f"base_mrr@{self.at_k}": mean_base_mrr,
                f"base_ndcg@{self.at_k}": mean_base_ndcg,
            }
            logger.info("\t\tBase  -> Reranked")
            logger.info(f"MAP:\t{mean_base_ap * 100:.2f} -> {mean_ap * 100:.2f}")
            logger.info(f"MRR@{self.at_k}:\t{mean_base_mrr * 100:.2f} -> {mean_mrr * 100:.2f}")
            logger.info(f"NDCG@{self.at_k}:\t{mean_base_ndcg * 100:.2f} -> {mean_ndcg * 100:.2f}")

            model_card_metrics = {
                "map": f"{mean_ap:.4f} ({mean_ap - mean_base_ap:+.4f})",
                f"mrr@{self.at_k}": f"{mean_mrr:.4f} ({mean_mrr - mean_base_mrr:+.4f})",
                f"ndcg@{self.at_k}": f"{mean_ndcg:.4f} ({mean_ndcg - mean_base_ndcg:+.4f})",
            }
            model_card_metrics = self.prefix_name_to_metrics(model_card_metrics, self.name)
            self.store_metrics_in_model_card_data(model, model_card_metrics, epoch, steps)

            metrics.update(base_metrics)
            metrics = self.prefix_name_to_metrics(metrics, self.name)
        else:
            logger.info(f"MAP:\t{mean_ap * 100:.2f}")
            logger.info(f"MRR@{self.at_k}:\t{mean_mrr * 100:.2f}")
            logger.info(f"NDCG@{self.at_k}:\t{mean_ndcg * 100:.2f}")

            metrics = self.prefix_name_to_metrics(metrics, self.name)
            self.store_metrics_in_model_card_data(model, metrics, epoch, steps)

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mean_mrr, mean_ndcg])

        return metrics

    def compute_metrics(self, y_true, y_pred):
        ranking = np.argsort(y_pred)[::-1]

        mrr = 0
        for rank, index in enumerate(ranking[0 : self.at_k]):
            if y_true[index]:
                mrr = 1 / (rank + 1)
                break

        ndcg = ndcg_score([y_true], [y_pred], k=self.at_k)
        ap = average_precision_score(y_true, y_pred)
        return mrr, ndcg, ap
