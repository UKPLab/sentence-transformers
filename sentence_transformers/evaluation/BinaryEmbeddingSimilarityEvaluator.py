from . import SentenceEvaluator, SimilarityFunction
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from ..util import batch_to_device
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
import numpy as np


class BinaryEmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    This is done by taking the metrics and checking if sentence pairs with a label of 0 are in the top 50% and pairs
    with label 1 in the bottom 50%.
    This assumes that the dataset is split 50-50.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    def __init__(self, dataloader: DataLoader,
                 main_similarity: SimilarityFunction = SimilarityFunction.COSINE, name:str =''):
        """
        Constructs an evaluator based for the dataset

        The labels need to be 0 for dissimilar pairs and 1 for similar pairs.
        The dataset needs to be split 50-50 with the labels.

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
            name = "_"+name

        self.csv_file: str = "binary_similarity_evaluation"+name+"_results.csv"
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

        logging.info("Evaluation the model on "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
            features, label_ids = batch_to_device(batch, self.device)
            with torch.no_grad():
                emb1, emb2 = [model(sent_features)['sentence_embedding'].to("cpu").numpy() for sent_features in features]

            labels.extend(label_ids.to("cpu").numpy())
            embeddings1.extend(emb1)
            embeddings2.extend(emb2)
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)

        #Ensure labels are just 0 or 1
        for label in labels:
            assert (label == 0 or label == 1)

        cosine_middle = np.median(cosine_scores)
        cosine_acc = 0
        for label, score in zip(labels, cosine_scores):
            if (label == 1 and score > cosine_middle) or (label == 0 and score <= cosine_middle):
                cosine_acc += 1
        cosine_acc /= len(labels)

        manhattan_middle = np.median(manhattan_distances)
        manhattan_acc = 0
        for label, score in zip(labels, manhattan_distances):
            if (label == 1 and score > manhattan_middle) or (label == 0 and score <= manhattan_middle):
                manhattan_acc += 1
        manhattan_acc /= len(labels)

        euclidean_middle = np.median(euclidean_distances)
        euclidean_acc = 0
        for label, score in zip(labels, euclidean_distances):
            if (label == 1 and score > euclidean_middle) or (label == 0 and score <= euclidean_middle):
                euclidean_acc += 1
        euclidean_acc /= len(labels)

        logging.info("Cosine-Classification:\t{:4f}".format(
            cosine_acc))
        logging.info("Manhattan-Classification:\t{:4f}".format(
            manhattan_acc))
        logging.info("Euclidean-Classification:\t{:4f}\n".format(
            euclidean_acc))

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