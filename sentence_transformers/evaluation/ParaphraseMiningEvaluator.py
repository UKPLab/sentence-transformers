from . import SentenceEvaluator
import logging
from sentence_transformers.util import paraphrase_mining
import os
import csv

from typing import List, Tuple, Dict
from collections import defaultdict


logger = logging.getLogger(__name__)


class ParaphraseMiningEvaluator(SentenceEvaluator):
    """
    Given a large set of sentences, this evaluator performs paraphrase (duplicate) mining and
    identifies the pairs with the highest similarity. It compare the extracted paraphrase pairs
     with a set of gold labels and computes the F1 score.
    """

    def __init__(self, sentences_map: Dict[str, str], duplicates_list: List[Tuple[str, str]] = None, duplicates_dict: Dict[str, Dict[str, bool]] = None, add_transitive_closure: bool = False, query_chunk_size:int = 5000, corpus_chunk_size:int = 100000, max_pairs: int = 500000, top_k: int = 100, show_progress_bar: bool = False, batch_size: int = 16, name: str = '', write_csv: bool = True):
        """

        :param sentences_map: A dictionary that maps sentence-ids to sentences, i.e. sentences_map[id] => sentence.
        :param duplicates_list: Duplicates_list is a list with id pairs [(id1, id2), (id1, id5)] that identifies the duplicates / paraphrases in the sentences_map
        :param duplicates_dict: A default dictionary mapping [id1][id2] to true if id1 and id2 are duplicates. Must be symmetric, i.e., if [id1][id2] => True, then [id2][id1] => True.
        :param add_transitive_closure: If true, it adds a transitive closure, i.e. if dup[a][b] and dup[b][c], then dup[a][c]
        :param query_chunk_size: To identify the paraphrases, the cosine-similarity between all sentence-pairs will be computed. As this might require a lot of memory, we perform a batched computation.  #query_batch_size sentences will be compared against up to #corpus_batch_size sentences. In the default setting, 5000 sentences will be grouped together and compared up-to against 100k other sentences.
        :param corpus_chunk_size: The corpus will be batched, to reduce the memory requirement
        :param max_pairs: We will only extract up to #max_pairs potential paraphrase candidates.
        :param top_k: For each query, we extract the top_k most similar pairs and add it to a sorted list. I.e., for one sentence we cannot find more than top_k paraphrases
        :param show_progress_bar: Output a progress bar
        :param batch_size: Batch size for computing sentence embeddings
        :param name: Name of the experiment
        :param write_csv: Write results to CSV file
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

        self.duplicates = duplicates_dict if duplicates_dict is not None else defaultdict(lambda: defaultdict(bool))
        if duplicates_list is not None:
            for id1, id2 in duplicates_list:
                if id1 in sentences_map and id2 in sentences_map:
                    self.duplicates[id1][id2] = True
                    self.duplicates[id2][id1] = True


        #Add transitive closure
        if add_transitive_closure:
            self.duplicates = self.add_transitive_closure(self.duplicates)


        positive_key_pairs = set()
        for key1 in self.duplicates:
            for key2 in self.duplicates[key1]:
                if key1 in sentences_map and key2 in sentences_map and (self.duplicates[key1][key2] or self.duplicates[key2][key1]):
                    positive_key_pairs.add(tuple(sorted([key1, key2])))

        self.total_num_duplicates = len(positive_key_pairs)

        if name:
            name = "_" + name

        self.csv_file: str = "paraphrase_mining_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "precision", "recall", "f1", "threshold", "average_precision"]
        self.write_csv = write_csv

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            out_txt = f" after epoch {epoch}:" if steps == -1 else f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("Paraphrase Mining Evaluation on " + self.name + " dataset" + out_txt)

        #Compute embedding for the sentences
        pairs_list = paraphrase_mining(model, self.sentences, self.show_progress_bar, self.batch_size,  self.query_chunk_size,  self.corpus_chunk_size, self.max_pairs, self.top_k )


        logger.info("Number of candidate pairs: " + str(len(pairs_list)))

        #Compute F1 score and Average Precision
        n_extract = n_correct = 0
        threshold = 0
        best_f1 = best_recall = best_precision = 0

        average_precision = 0

        for idx in range(len(pairs_list)):
            score, i, j = pairs_list[idx]
            id1 = self.ids[i]
            id2 = self.ids[j]

            #Compute optimal threshold and F1-score
            n_extract += 1
            if self.duplicates[id1][id2] or self.duplicates[id2][id1]:
                n_correct += 1
                precision = n_correct / n_extract
                recall = n_correct / self.total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                average_precision += precision
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (pairs_list[idx][0] + pairs_list[min(idx + 1, len(pairs_list)-1)][0]) / 2

        average_precision = average_precision / self.total_num_duplicates

        logger.info("Average Precision: {:.2f}".format(average_precision * 100))
        logger.info("Optimal threshold: {:.4f}".format(threshold))
        logger.info("Precision: {:.2f}".format(best_precision * 100))
        logger.info("Recall: {:.2f}".format(best_recall * 100))
        logger.info("F1: {:.2f}\n".format(best_f1 * 100))

        if output_path is not None and self.write_csv:
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


    @staticmethod
    def add_transitive_closure(graph):
        nodes_visited = set()
        for a in list(graph.keys()):
            if a not in nodes_visited:
                connected_subgraph_nodes = set()
                connected_subgraph_nodes.add(a)

                # Add all nodes in the connected graph
                neighbor_nodes_queue = list(graph[a])
                while len(neighbor_nodes_queue) > 0:
                    node = neighbor_nodes_queue.pop(0)
                    if node not in connected_subgraph_nodes:
                        connected_subgraph_nodes.add(node)
                        neighbor_nodes_queue.extend(graph[node])

                # Ensure transitivity between all nodes in the graph
                connected_subgraph_nodes = list(connected_subgraph_nodes)
                for i in range(len(connected_subgraph_nodes) - 1):
                    for j in range(i + 1, len(connected_subgraph_nodes)):
                        graph[connected_subgraph_nodes[i]][connected_subgraph_nodes[j]] = True
                        graph[connected_subgraph_nodes[j]][connected_subgraph_nodes[i]] = True

                        nodes_visited.add(connected_subgraph_nodes[i])
                        nodes_visited.add(connected_subgraph_nodes[j])
        return graph


