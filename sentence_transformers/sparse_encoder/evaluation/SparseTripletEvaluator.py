from __future__ import annotations

import csv
import logging
import os
from contextlib import nullcontext

from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder
from sentence_transformers.util import (
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
)

logger = logging.getLogger(__name__)


class SparseTripletEvaluator(TripletEvaluator):
    """
    # TODO
    """

    def __call__(
        self,
        model: SparseEncoder,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
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

        logger.info(f"TripletEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            embeddings_anchors = model.encode(
                self.anchors,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_sparse_tensor=True,
            )
            embeddings_positives = model.encode(
                self.positives,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_sparse_tensor=True,
            )
            embeddings_negatives = model.encode(
                self.negatives,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_sparse_tensor=True,
            )

        if not self.similarity_fn_names:
            self.similarity_fn_names = [model.similarity_fn_name]
            self._append_csv_headers(self.similarity_fn_names)

        similarity_functions = {
            "cosine": lambda anchors, positives, negatives: (
                pairwise_cos_sim(anchors, positives),
                pairwise_cos_sim(anchors, negatives),
            ),
            "dot": lambda anchors, positives, negatives: (
                pairwise_dot_score(anchors, positives),
                pairwise_dot_score(anchors, negatives),
            ),
            "manhattan": lambda anchors, positives, negatives: (
                pairwise_manhattan_sim(anchors, positives),
                pairwise_manhattan_sim(anchors, negatives),
            ),
            "euclidean": lambda anchors, positives, negatives: (
                pairwise_euclidean_sim(anchors, positives),
                pairwise_euclidean_sim(anchors, negatives),
            ),
        }

        metrics = {}
        for fn_name in self.similarity_fn_names:
            if fn_name in similarity_functions:
                positive_scores, negative_scores = similarity_functions[fn_name](
                    embeddings_anchors, embeddings_positives, embeddings_negatives
                )
                accuracy = (positive_scores > negative_scores + self.margin[fn_name]).float().mean().item()
                metrics[f"{fn_name}_accuracy"] = accuracy
                logger.info(f"Accuracy {fn_name.capitalize()} Similarity:\t{accuracy:.2%}")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps] + list(metrics.values()))

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps] + list(metrics.values()))

        if len(self.similarity_fn_names) > 1:
            metrics["max_accuracy"] = max(metrics.values())

        if self.main_similarity_function:
            self.primary_metric = {
                SimilarityFunction.COSINE: "cosine_accuracy",
                SimilarityFunction.DOT_PRODUCT: "dot_accuracy",
                SimilarityFunction.EUCLIDEAN: "euclidean_accuracy",
                SimilarityFunction.MANHATTAN: "manhattan_accuracy",
            }.get(self.main_similarity_function)
        else:
            if len(self.similarity_fn_names) > 1:
                self.primary_metric = "max_accuracy"
            else:
                self.primary_metric = f"{self.similarity_fn_names[0]}_accuracy"

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics
