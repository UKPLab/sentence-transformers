from . import SentenceEvaluator
import logging
import numpy as np
import os
import csv
from ..util import cos_sim, dot_score
import torch
from sklearn.metrics import average_precision_score
import tqdm

logger = logging.getLogger(__name__)

class RerankingEvaluator(SentenceEvaluator):
    """
    This class evaluates a SentenceTransformer model for the task of re-ranking.

    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 and MAP is compute to measure the quality of the ranking.

    :param samples: Must be a list and each element is of the form: {'query': '', 'positive': [], 'negative': []}. Query is the search query,
     positive is a list of positive (relevant) documents, negative is a list of negative (irrelevant) documents.
    """
    def __init__(self, samples, mrr_at_k: int = 10, name: str = '', write_csv: bool = True, similarity_fct=cos_sim, batch_size: int = 64, show_progress_bar: bool = False, use_batched_encoding: bool = True):
        self.samples = samples
        self.name = name
        self.mrr_at_k = mrr_at_k
        self.similarity_fct = similarity_fct
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.use_batched_encoding = use_batched_encoding

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        ### Remove sample with empty positive / negative set
        self.samples = [sample for sample in self.samples if len(sample['positive']) > 0 and len(sample['negative']) > 0]


        self.csv_file = "RerankingEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "MAP", "MRR@{}".format(mrr_at_k)]
        self.write_csv = write_csv

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("RerankingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)


        scores = self.compute_metrices(model)
        mean_ap = scores['map']
        mean_mrr = scores['mrr']

        #### Some stats about the dataset
        num_positives = [len(sample['positive']) for sample in self.samples]
        num_negatives = [len(sample['negative']) for sample in self.samples]

        logger.info("Queries: {} \t Positives: Min {:.1f}, Mean {:.1f}, Max {:.1f} \t Negatives: Min {:.1f}, Mean {:.1f}, Max {:.1f}".format(len(self.samples), np.min(num_positives), np.mean(num_positives),
                                                                                                                                             np.max(num_positives), np.min(num_negatives),
                                                                                                                                             np.mean(num_negatives), np.max(num_negatives)))
        logger.info("MAP: {:.2f}".format(mean_ap * 100))
        logger.info("MRR@{}: {:.2f}".format(self.mrr_at_k, mean_mrr * 100))

        #### Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mean_ap, mean_mrr])

        return mean_ap

    def compute_metrices(self, model):
        return self.compute_metrices_batched(model) if self.use_batched_encoding else self.compute_metrices_individual(model)

    def compute_metrices_batched(self, model):
        """
        Computes the metrices in a batched way, by batching all queries and
        all documents together
        """
        all_mrr_scores = []
        all_ap_scores = []

        all_query_embs = model.encode([sample['query'] for sample in self.samples],
                                  convert_to_tensor=True,
                                  batch_size=self.batch_size,
                                  show_progress_bar=True) #self.show_progress_bar)

        all_docs = []

        for sample in self.samples:
            all_docs.extend(sample['positive'])
            all_docs.extend(sample['negative'])

        all_docs_embs = model.encode(all_docs,
                                    convert_to_tensor=True,
                                    batch_size=self.batch_size,
                                    show_progress_bar=self.show_progress_bar)

        #Compute scores
        query_idx, docs_idx = 0,0
        for instance in self.samples:
            query_emb = all_query_embs[query_idx]
            query_idx += 1

            num_pos = len(instance['positive'])
            num_neg = len(instance['negative'])
            docs_emb = all_docs_embs[docs_idx:docs_idx+num_pos+num_neg]
            docs_idx += num_pos+num_neg

            if num_pos == 0 or num_neg == 0:
                continue

            pred_scores = self.similarity_fct(query_emb, docs_emb)
            if len(pred_scores.shape) > 1:
                pred_scores = pred_scores[0]

            pred_scores_argsort = torch.argsort(-pred_scores)  #Sort in decreasing order

            #Compute MRR score
            is_relevant = [True]*num_pos + [False]*num_neg
            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0:self.mrr_at_k]):
                if is_relevant[index]:
                    mrr_score = 1 / (rank+1)
                    break
            all_mrr_scores.append(mrr_score)

            # Compute AP
            all_ap_scores.append(average_precision_score(is_relevant, pred_scores.cpu().tolist()))

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        return {'map': mean_ap, 'mrr': mean_mrr}


    def compute_metrices_individual(self, model):
        """
        Embeds every (query, positive, negative) tuple individually.
        Is slower than the batched version, but saves memory as only the
        embeddings for one tuple are needed. Useful when you have
        a really large test set
        """
        all_mrr_scores = []
        all_ap_scores = []


        for instance in tqdm.tqdm(self.samples, disable=not self.show_progress_bar, desc="Samples"):
            query = instance['query']
            positive = list(instance['positive'])
            negative = list(instance['negative'])

            if len(positive) == 0 or len(negative) == 0:
                continue

            docs = positive + negative
            is_relevant = [True]*len(positive) + [False]*len(negative)

            query_emb = model.encode([query], convert_to_tensor=True, batch_size=self.batch_size, show_progress_bar=False)
            docs_emb = model.encode(docs, convert_to_tensor=True, batch_size=self.batch_size, show_progress_bar=False)

            pred_scores = self.similarity_fct(query_emb, docs_emb)
            if len(pred_scores.shape) > 1:
                pred_scores = pred_scores[0]

            pred_scores_argsort = torch.argsort(-pred_scores)  #Sort in decreasing order

            #Compute MRR score
            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0:self.mrr_at_k]):
                if is_relevant[index]:
                    mrr_score = 1 / (rank+1)
                    break
            all_mrr_scores.append(mrr_score)

            # Compute AP
            all_ap_scores.append(average_precision_score(is_relevant, pred_scores.cpu().tolist()))

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        return {'map': mean_ap, 'mrr': mean_mrr}

