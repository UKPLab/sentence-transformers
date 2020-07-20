from . import SentenceEvaluator, SimilarityFunction
from torch.utils.data import DataLoader

import torch
import logging
from tqdm import tqdm
from ..util import batch_to_device
import os
import csv
import numpy as np
import scipy.spatial
from typing import List, Tuple

class TranslationEvaluator(SentenceEvaluator):
    """
    Given two sets of sentences in different languages, e.g. (en_1, en_2, en_3...) and (fr_1, fr_2, fr_3, ...),
    and assuming that en_i = fr_i.
    Checks if vec(en_i) has the highest similarity to vec(fr_i). Computes the accurarcy in both directions
    """
    def __init__(self, source_sentences: List[str], target_sentences: List[str],  show_progress_bar: bool = False, batch_size: int = 8, name: str = ''):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param source_sentences:
            List of sentences in source language
        :param target_sentences:
            List of sentences in target language
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        assert len(self.source_sentences) == len(self.target_sentences)

        if name:
            name = "_"+name

        self.csv_file = "translation_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "src2trg", "trg2src"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Evaluating translation matching Accuracy on "+self.name+" dataset"+out_txt)

        embeddings1 = np.asarray(model.encode(self.source_sentences, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size))
        embeddings2 = np.asarray(model.encode(self.target_sentences, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size))

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
