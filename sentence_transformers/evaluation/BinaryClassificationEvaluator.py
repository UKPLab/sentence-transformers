from __future__ import annotations

import csv
import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.similarity_functions import SimilarityFunction

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class BinaryClassificationEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity, dot score, Euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    Args:
        sentences1 (List[str]): The first column of sentences.
        sentences2 (List[str]): The second column of sentences.
        labels (List[int]): labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1.
        name (str, optional): Name for the output. Defaults to "".
        batch_size (int, optional): Batch size used to compute embeddings. Defaults to 32.
        show_progress_bar (bool, optional): If true, prints a progress bar. Defaults to False.
        write_csv (bool, optional): Write results to a CSV file. Defaults to True.
        truncate_dim (Optional[int], optional): The dimension to truncate sentence embeddings to. `None` uses the model's current truncation dimension. Defaults to None.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import BinaryClassificationEvaluator
            from datasets import load_dataset

            # Load a model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Load a dataset with two text columns and a class label column (https://huggingface.co/datasets/sentence-transformers/quora-duplicates)
            eval_dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train[-1000:]")

            # Initialize the evaluator
            binary_acc_evaluator = BinaryClassificationEvaluator(
                sentences1=eval_dataset["sentence1"],
                sentences2=eval_dataset["sentence2"],
                labels=eval_dataset["label"],
                name="quora-duplicates-dev",
            )
            results = binary_acc_evaluator(model)
            '''
            Binary Accuracy Evaluation of the model on the quora-duplicates-dev dataset:
            Accuracy with Cosine-Similarity:           81.60    (Threshold: 0.8352)
            F1 with Cosine-Similarity:                 75.27    (Threshold: 0.7715)
            Precision with Cosine-Similarity:          65.81
            Recall with Cosine-Similarity:             87.89
            Average Precision with Cosine-Similarity:  76.03

            Accuracy with Dot-Product:           81.60  (Threshold: 0.8352)
            F1 with Dot-Product:                 75.27  (Threshold: 0.7715)
            Precision with Dot-Product:          65.81
            Recall with Dot-Product:             87.89
            Average Precision with Dot-Product:  76.03

            Accuracy with Manhattan-Distance:           81.50   (Threshold: 12.0727)
            F1 with Manhattan-Distance:                 74.97   (Threshold: 15.2269)
            Precision with Manhattan-Distance:          63.89
            Recall with Manhattan-Distance:             90.68
            Average Precision with Manhattan-Distance:  75.66

            Accuracy with Euclidean-Distance:           81.60   (Threshold: 0.5741)
            F1 with Euclidean-Distance:                 75.27   (Threshold: 0.6760)
            Precision with Euclidean-Distance:          65.81
            Recall with Euclidean-Distance:             87.89
            Average Precision with Euclidean-Distance:  76.03
            '''
            print(binary_acc_evaluator.primary_metric)
            # => "quora-duplicates-dev_max_ap"
            print(results[binary_acc_evaluator.primary_metric])
            # => 0.760277070888393
    """

    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        labels: list[int],
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        truncate_dim: int | None = None,
    ):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.truncate_dim = truncate_dim

        self.primary_metric = "max_ap"

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert label == 0 or label == 1

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = "binary_classification_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps"]
        metrics = [
            "accuracy",
            "accuracy_threshold",
            "f1",
            "precision",
            "recall",
            "f1_threshold",
            "ap",
        ]
        for v in SimilarityFunction.possible_values():
            for m in metrics:
                self.csv_headers.append(f"{v}_{m}")

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def __call__(
        self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        """
        Compute the evaluation metrics for the given model.

        Args:
            model (SentenceTransformer): The model to evaluate.
            output_path (str, optional): Path to save the evaluation results CSV file. Defaults to None.
            epoch (int, optional): The epoch number. Defaults to -1.
            steps (int, optional): The number of steps. Defaults to -1.

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

        logger.info(f"Binary Accuracy Evaluation of the model on the {self.name} dataset{out_txt}:")

        scores = self.compute_metrices(model)

        file_output_data = [epoch, steps]

        for header_name in self.csv_headers:
            if "_" in header_name:
                sim_fct, metric = header_name.split("_", maxsplit=1)
                file_output_data.append(scores[sim_fct][metric])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)

        metrics = {
            f"{short_name}_{metric}": value
            for short_name, values in scores.items()
            for metric, value in values.items()
        }
        metrics.update(
            {f"max_{metric}": max(scores[short_name][metric] for short_name in scores) for metric in scores["cosine"]}
        )
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics)
        return metrics

    def compute_metrices(self, model):
        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            try:
                # If the sentences are hashable, then we can use a set to avoid embedding the same sentences multiple
                # times
                sentences = list(set(self.sentences1 + self.sentences2))
            except TypeError:
                # Otherwise we just embed everything, e.g. if the sentences are images for evaluating a CLIP model
                embeddings = model.encode(
                    self.sentences1 + self.sentences2,
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_numpy=True,
                )
                embeddings1 = embeddings[: len(self.sentences1)]
                embeddings2 = embeddings[len(self.sentences1) :]
            else:
                embeddings = model.encode(
                    sentences,
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_numpy=True,
                )
                emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
                embeddings1 = [emb_dict[sent] for sent in self.sentences1]
                embeddings2 = [emb_dict[sent] for sent in self.sentences2]

        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        embeddings1_np = np.asarray(embeddings1)
        embeddings2_np = np.asarray(embeddings2)
        dot_scores = np.sum(embeddings1_np * embeddings2_np, axis=-1)

        labels = np.asarray(self.labels)
        output_scores = {}
        for short_name, name, scores, reverse in [
            [SimilarityFunction.COSINE.value, "Cosine-Similarity", cosine_scores, True],
            [SimilarityFunction.DOT_PRODUCT.value, "Dot-Product", dot_scores, True],
            [SimilarityFunction.MANHATTAN.value, "Manhattan-Distance", manhattan_distances, False],
            [SimilarityFunction.EUCLIDEAN.value, "Euclidean-Distance", euclidean_distances, False],
        ]:
            acc, acc_threshold = self.find_best_acc_and_threshold(scores, labels, reverse)
            f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(scores, labels, reverse)
            ap = average_precision_score(labels, scores * (1 if reverse else -1))

            logger.info(f"Accuracy with {name}:           {acc * 100:.2f}\t(Threshold: {acc_threshold:.4f})")
            logger.info(f"F1 with {name}:                 {f1 * 100:.2f}\t(Threshold: {f1_threshold:.4f})")
            logger.info(f"Precision with {name}:          {precision * 100:.2f}")
            logger.info(f"Recall with {name}:             {recall * 100:.2f}")
            logger.info(f"Average Precision with {name}:  {ap * 100:.2f}\n")

            output_scores[short_name] = {
                "accuracy": acc,
                "accuracy_threshold": acc_threshold,
                "f1": f1,
                "f1_threshold": f1_threshold,
                "precision": precision,
                "recall": recall,
                "ap": ap,
            }

        return output_scores

    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(labels == 0)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return max_acc, best_threshold

    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows) - 1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold
