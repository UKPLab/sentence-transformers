from . import SentenceEvaluator, SimilarityFunction
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from sentence_transformers.util import batch_to_device
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
import numpy as np
from typing import List
from ..readers import InputExample


class BinaryEmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    Same Usage with BinaryEmbeddingSimilarityEvaluator , but no need assumes that the dataset is split 50-50.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, sentences1: List[str], sentences2: List[str], labels: List[int],
                 main_similarity: SimilarityFunction = SimilarityFunction.COSINE, name: str = '',
                 batch_size: int = 16, show_progress_bar: bool = False):
        """
        Constructs an evaluator based for the dataset

        The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.main_similarity = main_similarity
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "binary_similarity_evaluation" + ("_"+name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_acc", "cosine_threshold", "manhattan_acc", "manhattan_threshold", "euclidean_acc", "euclidean_threshold", "dot-product_acc", "dot-product_threshold"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info("Binary Accuracy Evaluation of the model on " + self.name + " dataset" + out_txt)
        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size,
                                   show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size,
                                   show_progress_bar=self.show_progress_bar, convert_to_numpy=True)

        cosine_scores = 1-paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        labels = np.asarray(self.labels)
        cosine_acc, cosine_threshold = self.find_best_acc_and_threshold(cosine_scores, labels, True)
        manhattan_acc, manhatten_threshold = self.find_best_acc_and_threshold(manhattan_distances, labels, False)
        euclidean_acc, euclidean_threshold = self.find_best_acc_and_threshold(euclidean_distances, labels, False)
        dot_acc, dot_threshold = self.find_best_acc_and_threshold(dot_products, labels, False)

        logging.info("Accuracy with Cosine-Similarity:\t{:.2f}\t(Threshold: {:.4f})".format(
            cosine_acc*100, cosine_threshold))
        logging.info("Accuracy with Manhattan-Distance:\t{:.2f}\t(Threshold: {:.4f})".format(
            manhattan_acc*100, manhatten_threshold))
        logging.info("Accuracy with Euclidean-Distance:\t{:.2f}\t(Threshold: {:.4f})".format(
            euclidean_acc*100, euclidean_threshold))
        logging.info("Accuracy with Dot-Product:\t{:.2f}\t(Threshold: {:.4f})\n".format(
            dot_acc * 100, dot_threshold))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, cosine_acc, cosine_threshold, manhattan_acc, manhatten_threshold, euclidean_acc, euclidean_threshold, dot_acc, dot_threshold])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, cosine_acc, cosine_threshold, manhattan_acc, manhatten_threshold, euclidean_acc, euclidean_threshold, dot_acc, dot_threshold])

        if self.main_similarity == SimilarityFunction.COSINE:
            return cosine_acc
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return euclidean_acc
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return manhattan_acc
        else:
            raise ValueError("Unknown main_similarity value")

    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(labels == 0)

        for i in range(len(rows)-1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i+1][0]) / 2

        return max_acc, best_threshold

