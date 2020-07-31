from . import SentenceEvaluator, SimilarityFunction
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from sentence_transformers.util import batch_to_device, pytorch_cos_sim
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import queue

class ParaphraseMiningEvaluator(SentenceEvaluator):
    def __init__(self, sentences_map: Dict[str, str], duplicates_list: List[Tuple[str, str]] = None, query_batch_size:int = 100000, corpus_batch_size:int = 250000, max_pairs: int = 500000, show_progress_bar: bool = False,  batch_size:int = 16, name: str = ''):
        self.sentences = []
        self.ids = []

        for id, sentence in sentences_map.items():
            self.sentences.append(sentence)
            self.ids.append(id)

        self.name = name
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.query_batch_size = query_batch_size
        self.corpus_batch_size = corpus_batch_size
        self.max_pairs = max_pairs

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
        self.csv_headers = ["epoch", "steps", "precision", "recall", "f1", "threshold"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            out_txt = f" after epoch {epoch}:" if steps == -1 else f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info("Paraphrase Mining Evaluation on " + self.name + " dataset" + out_txt)

        #Compute embedding for the sentences
        embeddings = model.encode(self.sentences, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)

        #Mine for duplicates
        pairs = queue.PriorityQueue()
        min_score = -1
        num_added = 0

        for corpus_start_idx in range(0, len(embeddings), self.corpus_batch_size):
            corpus_end_idx = min(corpus_start_idx + self.corpus_batch_size, len(embeddings))
            for query_start_idx in range(0, len(embeddings), self.query_batch_size):
                query_end_idx = min(query_start_idx + self.query_batch_size, len(embeddings))

                # logging.info("Compute cosine similarities")
                cos_scores = pytorch_cos_sim(embeddings[query_start_idx:query_end_idx], embeddings[corpus_start_idx:corpus_end_idx]).cpu().numpy()
                cos_scores = np.nan_to_num(cos_scores)

                # logging.info("Sort scores")
                cos_score_argsort = np.argpartition(-cos_scores, 100)

                # logging.info("Find most similar pairs out of {} queries".format(len(cos_scores)))
                for query_itr in range(len(cos_scores)):
                    for corpus_itr in cos_score_argsort[query_itr][0:100]:
                        i = query_start_idx + query_itr
                        j = corpus_start_idx + corpus_itr

                        if i != j and cos_scores[query_itr][corpus_itr] > min_score:
                            pairs.put((cos_scores[query_itr][corpus_itr], i, j))
                            num_added += 1

                            if num_added >= self.max_pairs:
                                entry = pairs.get()
                                min_score = entry[0]

        # Get the pairs
        added_pairs = set()  # Used for duplicate detection
        pairs_list = []
        while not pairs.empty():
            score, i, j = pairs.get()
            id1, id2 = sorted([self.ids[i], self.ids[j]])

            if id1 != id2 and (id1, id2) not in added_pairs:
                added_pairs.add((id1, id2))
                pairs_list.append([score, id1, id2])

        # Highest scores first
        pairs_list = sorted(pairs_list, key=lambda x: x[0], reverse=True)

        logging.info("Number of candidate pairs: " + str(len(pairs_list)))

        #Compute F1 score
        nextract = ncorrect = 0
        threshold = 0
        best_f1 = best_recall = best_precision = 0
        for i in range(len(pairs_list)):
            score, id1, id2 = pairs_list[i]
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
                    writer.writerow([epoch, steps, best_precision, best_recall, best_f1, threshold])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, best_precision, best_recall, best_f1, threshold])

        return best_f1
