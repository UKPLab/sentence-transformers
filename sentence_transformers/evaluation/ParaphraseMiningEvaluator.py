from . import SentenceEvaluator
import logging
from sentence_transformers.util import paraphrase_mining
import os
import csv
from sklearn.metrics import average_precision_score

from typing import List, Tuple, Dict
from collections import defaultdict


class ParaphraseMiningEvaluator(SentenceEvaluator):
    """
    Given a large set of sentences, this evaluator performs paraphrase (duplicate) mining and
    identifies the pairs with the highest similarity. It compare the extracted paraphrase pairs
     with a set of gold labels and computes the F1 score.
    """

    def __init__(self, sentences_map: Dict[str, str], duplicates_list: List[Tuple[str, str]] = None, query_chunk_size:int = 5000, corpus_chunk_size:int = 100000, max_pairs: int = 500000, top_k: int = 100, show_progress_bar: bool = False, batch_size:int = 16, name: str = ''):
        """

        :param sentences_map: A dictionary that maps sentence-ids to sentences, i.e. sentences_map[id] => sentence.
        :param duplicates_list: Duplicates_list is a list with id pairs [(id1, id2), (id1, id5)] that identifies the duplicates / paraphrases in the sentences_map
        :param query_chunk_size: To identify the paraphrases, the cosine-similarity between all sentence-pairs will be computed. As this might require a lot of memory, we perform a batched computation.  #query_batch_size sentences will be compared against up to #corpus_batch_size sentences. In the default setting, 5000 sentences will be grouped together and compared up-to against 100k other sentences.
        :param corpus_chunk_size: The corpus will be batched, to reduce the memory requirement
        :param max_pairs: We will only extract up to #max_pairs potential paraphrase candidates.
        :param top_k: For each query, we extract the top_k most similar pairs and add it to a sorted list. I.e., for one sentence we cannot find more than top_k paraphrases
        :param show_progress_bar: Output a progress bar
        :param batch_size: Batch size for computing sentence embeddings
        :param name: Name of the experiment
        """
        self.sentences = []
        self.ids = []

        for id, sentence in sentences_map.items():
            self.sentences.append(sentence)
            self.ids.append(id)

        self.name = name
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.query_chunk_size = query_chunk_size
        self.corpus_chunk_size = corpus_chunk_size
        self.max_pairs = max_pairs
        self.top_k = top_k

        self.duplicates = defaultdict(lambda: defaultdict(bool))
        self.total_num_duplicates = 0

        if duplicates_list is not None:
            for id1, id2 in duplicates_list:
                if id1 in sentences_map and id2 in sentences_map and not self.duplicates[id1][id2]:
                    self.duplicates[id1][id2] = True
                    self.duplicates[id2][id1] = True
                    self.total_num_duplicates += 1

        if name:
            name = "_" + name

        self.csv_file: str = "paraphrase_mining_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "precision", "recall", "f1", "threshold", "average_precision"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            out_txt = f" after epoch {epoch}:" if steps == -1 else f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info("Paraphrase Mining Evaluation on " + self.name + " dataset" + out_txt)

        #Compute embedding for the sentences
        pairs_list = paraphrase_mining(model, self.sentences, self.show_progress_bar, self.batch_size,  self.query_chunk_size,  self.corpus_chunk_size, self.max_pairs, self.top_k )


        logging.info("Number of candidate pairs: " + str(len(pairs_list)))

        #Compute F1 score and Average Precision
        nextract = ncorrect = 0
        threshold = 0
        best_f1 = best_recall = best_precision = 0

        y_scores = []
        y_true = []

        for i in range(len(pairs_list)):
            score, i, j = pairs_list[i]
            id1 = self.ids[i]
            id2 = self.ids[j]

            # Get y_scores and y_true List for Average Precision
            y_scores.append(score)
            y_true.append(1 if self.duplicates[id1][id2] or self.duplicates[id2][id1] else 0)


            #Compute optimal threshold and F1-score
            nextract += 1
            if self.duplicates[id1][id2] or self.duplicates[id2][id1]:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / self.total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (pairs_list[i][0] + pairs_list[i + 1][0]) / 2

        average_precision = average_precision_score(y_true, y_scores)

        logging.info("Average Precision: {:.2f}".format(average_precision * 100))
        logging.info("Optimal threshold: {:.4f}".format(threshold))
        logging.info("Precision: {:.2f}".format(best_precision * 100))
        logging.info("Recall: {:.2f}".format(best_recall * 100))
        logging.info("F1: {:.2f}\n".format(best_f1 * 100))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, best_precision, best_recall, best_f1, threshold, average_precision])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, best_precision, best_recall, best_f1, threshold, average_precision])

        return average_precision


