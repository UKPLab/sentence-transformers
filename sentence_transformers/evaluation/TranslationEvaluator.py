from . import SentenceEvaluator, SimilarityFunction
from torch.utils.data import DataLoader

import torch
import logging
from tqdm import tqdm
from ..util import batch_to_device
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
import scipy.spatial

class TranslationEvaluator(SentenceEvaluator):
    """
    Given two sets of sentences in different languages, e.g. (en_1, en_2, en_3...) and (fr_1, fr_2, fr_3, ...),
    and assuming that en_i = fr_i.
    Checks if vec(en_i) has the highest similarity to vec(fr_i). Computes the accurarcy in both directions
    """
    def __init__(self, dataloader: DataLoader, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = None):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.dataloader = dataloader
        self.main_similarity = main_similarity
        self.name = name
        if name:
            name = "_"+name

        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.csv_file = "translation_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "src2trg", "trg2src"]


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        embeddings1 = []
        embeddings2 = []
        labels = []

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Evaluation the model on "+self.name+" dataset"+out_txt)

        self.dataloader.collate_fn = model.smart_batching_collate

        iterator = self.dataloader
        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Convert Evaluating")

        for step, batch in enumerate(iterator):
            features, label_ids = batch_to_device(batch, self.device)
            with torch.no_grad():
                emb1, emb2 = [model(sent_features)['sentence_embedding'].to("cpu").numpy() for sent_features in features]

            labels.extend(label_ids.to("cpu").numpy())
            embeddings1.extend(emb1)
            embeddings2.extend(emb2)

        distances = -scipy.spatial.distance.cdist(embeddings1, embeddings2, "cosine")


        correct_src2trg = 0
        correct_trg2src = 0
        for i in range(len(distances)):
            max_idx = np.argmax(distances[i])
            if i == max_idx:
                correct_src2trg += 1

        distances = distances.T
        for i in range(len(distances)):
            max_idx = np.argmax(distances[i])
            if i == max_idx:
                correct_trg2src += 1

        acc_src2trg = correct_src2trg / len(distances)
        acc_trg2src = correct_trg2src / len(distances)

        logging.info("Accuracy src2trg: {:.2f}".format(acc_src2trg*100))
        logging.info("Accuracy trg2src: {:.2f}".format(acc_trg2src*100))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc_src2trg, acc_trg2src])

        return (acc_src2trg+acc_trg2src)/2
