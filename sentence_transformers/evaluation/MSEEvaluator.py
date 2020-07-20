from sentence_transformers.evaluation import SentenceEvaluator
import numpy as np
import logging
import os
import csv
from typing import List

class MSEEvaluator(SentenceEvaluator):
    """
    Computes the mean squared error (x100) between the computed sentence embedding
    and some target sentence embedding
    """
    def __init__(self, source_sentences: List[str], target_sentences: List[str], teacher_model = None, show_progress_bar: bool = False, batch_size:int = 8, name: str = ''):
        self.source_sentences = source_sentences
        self.source_embeddings = np.asarray(teacher_model.encode(source_sentences, show_progress_bar=show_progress_bar, batch_size=batch_size))

        self.target_sentences = target_sentences
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name

        self.csv_file = "mse_evaluation_" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "MSE"]

    def __call__(self, model, output_path, epoch  = -1, steps = -1):
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        target_embeddings = np.asarray(model.encode(self.source_sentences, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size))

        mse = ((self.source_embeddings - target_embeddings)**2).mean()
        mse *= 100

        logging.info("MSE evaluation (lower = better) on "+self.name+" dataset"+out_txt)
        logging.info("MSE (*100):\t{:4f}".format(mse))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mse])

        return -mse #Return negative score as SentenceTransformers maximizes the performance