from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import torch
import numpy as np
import logging
import os
import csv


class MSEEvaluatorFromDataFrame(SentenceEvaluator):
    """
    Computes the mean squared error (x100) between the computed sentence embedding
    and some target sentence embedding.
    :param dataframe:
        It must have the following format. Rows contains different, parallel sentences. Columns are the respective language codes
        [{'en': 'My sentence', 'es': 'Sentence in Spanisch', 'fr': 'Sentence in French'...},
         {'en': 'My second sentence', ....]
    :param combinations:
        Must be of the format [('en', 'es'), ('en', 'fr'), ...]
        First entry in a tuple is the source language. The sentence in the respective language will be fetched from the dataframe and passed to the teacher model.
        Second entry in a tuple the the target language. Sentence will be fetched from the dataframe and passed to the student model
    """
    def __init__(self, dataframe: List[Dict[str, str]], teacher_model: SentenceTransformer, combinations: List[Tuple[str, str]], batch_size: int = 8, name=''):

        self.combinations = combinations
        self.name = name
        self.batch_size = batch_size


        if name:
            name = "_"+name

        self.csv_file = "mse_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]
        self.data = {}

        logging.info("Compute teacher embeddings")
        all_source_sentences = set()
        for src_lang, trg_lang in self.combinations:
            src_sentences = []
            trg_sentences = []

            for row in dataframe:
                if row[src_lang].strip() != "" and row[trg_lang].strip() != "":
                    all_source_sentences.add(row[src_lang])
                    src_sentences.append(row[src_lang])
                    trg_sentences.append(row[trg_lang])

            self.data[(src_lang, trg_lang)] = (src_sentences, trg_sentences)
            self.csv_headers.append("{}-{}".format(src_lang, trg_lang))

        all_source_sentences = list(all_source_sentences)
        all_src_embeddings = teacher_model.encode(all_source_sentences, batch_size=self.batch_size)
        self.teacher_embeddings = {sent: emb for sent, emb in zip(all_source_sentences, all_src_embeddings)}

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int  = -1):
        model.eval()

        mse_scores = []
        for src_lang, trg_lang in self.combinations:
            src_sentences, trg_sentences = self.data[(src_lang, trg_lang)]

            src_embeddings = np.asarray([self.teacher_embeddings[sent] for sent in src_sentences])
            trg_embeddings = np.asarray(model.encode(trg_sentences, batch_size=self.batch_size))

            mse = ((src_embeddings - trg_embeddings) ** 2).mean()
            mse *= 100
            mse_scores.append(mse)

            logging.info("MSE evaluation on {} dataset - {}-{}:".format(self.name, src_lang, trg_lang))
            logging.info("MSE (*100):\t{:4f}".format(mse))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps]+mse_scores)

        return -np.mean(mse_scores) #Return negative score as SentenceTransformers maximizes the performance

