from . import SentenceEvaluator
import logging
from ..util import pytorch_cos_sim
import os
import csv
import numpy as np
import scipy.spatial
from typing import List
import torch


logger = logging.getLogger(__name__)

class TranslationEvaluator(SentenceEvaluator):
    """
    Given two sets of sentences in different languages, e.g. (en_1, en_2, en_3...) and (fr_1, fr_2, fr_3, ...),
    and assuming that fr_i is the translation of en_i.
    Checks if vec(en_i) has the highest similarity to vec(fr_i). Computes the accurarcy in both directions
    """
    def __init__(self, source_sentences: List[str], target_sentences: List[str],  show_progress_bar: bool = False, batch_size: int = 16, name: str = '', print_wrong_matches: bool = False, write_csv: bool = True):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param source_sentences:
            List of sentences in source language
        :param target_sentences:
            List of sentences in target language
        :param print_wrong_matches:
            Prints incorrect matches
        :param write_csv:
            Write results to CSV file
        """
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.print_wrong_matches = print_wrong_matches

        assert len(self.source_sentences) == len(self.target_sentences)

        if name:
            name = "_"+name

        self.csv_file = "translation_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "src2trg", "trg2src"]
        self.write_csv = write_csv

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluating translation matching Accuracy on "+self.name+" dataset"+out_txt)

        embeddings1 = torch.stack(model.encode(self.source_sentences, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_numpy=False))
        embeddings2 = torch.stack(model.encode(self.target_sentences, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_numpy=False))


        cos_sims = pytorch_cos_sim(embeddings1, embeddings2).detach().cpu().numpy()

        correct_src2trg = 0
        correct_trg2src = 0

        for i in range(len(cos_sims)):
            max_idx = np.argmax(cos_sims[i])

            if i == max_idx:
                correct_src2trg += 1
            elif self.print_wrong_matches:
                print("i:", i, "j:", max_idx, "INCORRECT" if i != max_idx else "CORRECT")
                print("Src:", self.source_sentences[i])
                print("Trg:", self.target_sentences[max_idx])
                print("Argmax score:", cos_sims[i][max_idx], "vs. correct score:", cos_sims[i][i])

                results = zip(range(len(cos_sims[i])), cos_sims[i])
                results = sorted(results, key=lambda x: x[1], reverse=True)
                for idx, score in results[0:5]:
                    print("\t", idx, "(Score: %.4f)" % (score), self.target_sentences[idx])



        cos_sims = cos_sims.T
        for i in range(len(cos_sims)):
            max_idx = np.argmax(cos_sims[i])
            if i == max_idx:
                correct_trg2src += 1

        acc_src2trg = correct_src2trg / len(cos_sims)
        acc_trg2src = correct_trg2src / len(cos_sims)

        logger.info("Accuracy src2trg: {:.2f}".format(acc_src2trg*100))
        logger.info("Accuracy trg2src: {:.2f}".format(acc_trg2src*100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc_src2trg, acc_trg2src])

        return (acc_src2trg+acc_trg2src)/2
