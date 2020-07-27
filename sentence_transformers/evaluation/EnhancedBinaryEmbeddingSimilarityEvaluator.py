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


class EnhancedBinaryEmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    Same Usage with BinaryEmbeddingSimilarityEvaluator , but no need assumes that the dataset is split 50-50.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader,
                 main_similarity: SimilarityFunction = SimilarityFunction.COSINE, name: str = ''):
        """
        Constructs an evaluator based for the dataset

        The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.dataloader = dataloader
        self.main_similarity = main_similarity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        if name:
            name = "_" + name

        self.csv_file: str = "binary_similarity_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_acc", "euclidean_acc", "manhattan_acc"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        embeddings1 = []
        embeddings2 = []
        labels = []

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info("Evaluation the model on " + self.name + " dataset" + out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
            features, label_ids = batch_to_device(batch, self.device)
            with torch.no_grad():
                emb1, emb2 = [model(sent_features)['sentence_embedding'].to("cpu").numpy() for sent_features in
                              features]

            labels.extend(label_ids.to("cpu").numpy())
            embeddings1.extend(emb1)
            embeddings2.extend(emb2)
        cosine_scores = 1-paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        # Ensure labels are just 0 or 1
        for label in labels:
            assert (label == 0 or label == 1)

        labels = np.asarray(labels)
        cosine_acc, cosine_threshold = self.find_best_acc_and_threshold(cosine_scores, labels, True)
        manhattan_acc, manhatten_threshold = self.find_best_acc_and_threshold(manhattan_distances, labels, False)
        euclidean_acc, euclidean_threshold = self.find_best_acc_and_threshold(euclidean_distances, labels, False)

        logging.info("Accuracy with Cosine-Similarity:\t{:.2f}\t(Threshold: {:.4f})".format(
            cosine_acc*100, cosine_threshold))
        logging.info("Accuracy with Manhattan-Distance:\t{:.2f}\t(Threshold: {:.4f})".format(
            manhattan_acc*100, manhatten_threshold))
        logging.info("Accuracy with Euclidean-Distance:\t{:.2f}\t(Threshold: {:.4f})\n".format(
            euclidean_acc*100, euclidean_threshold))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, cosine_acc, euclidean_acc, manhattan_acc])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, cosine_acc, euclidean_acc, manhattan_acc])

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

        """pos_dists = scores[labels == 1]
        neg_dists = scores[labels == 0]

        pos_mean, neg_mean = np.mean(pos_dists), np.mean(neg_dists)
        total_num = len(pos_dists) + len(neg_dists)
        max_acc = 0
        best_threshold = 0
        for threshold in range(int(pos_mean * 10), int(neg_mean * 10)):
            threshold = threshold / 10

            correct_neg_num = neg_dists[neg_dists > threshold].shape[0]
            correct_pos_num = pos_dists[pos_dists < threshold].shape[0]

            acc = (correct_neg_num + correct_pos_num) / total_num
            if acc > max_acc:
                max_acc = acc
                best_threshold = threshold
        return max_acc, best_threshold"""
