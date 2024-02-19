from . import SentenceEvaluator, SimilarityFunction
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from typing import List
from ..readers import InputExample
import numpy as np


logger = logging.getLogger(__name__)


class TripletEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example).
        Checks if distance(sentence, positive_example) < distance(sentence, negative_example).
    """

    def __init__(
        self,
        anchors: List[str],
        positives: List[str],
        negatives: List[str],
        score_function: SimilarityFunction = None,
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        """
        :param anchors: Sentences to check similarity to. (e.g. a query)
        :param positives: List of positive sentences
        :param negatives: List of negative sentences
        :param score_function: One of "cos_sim" (Cosine), "euclidean_sim" (Euclidean) or "manhattan_sim" (Manhattan). Defaults to None, returning all 3.
        :param name: Name for the output
        :param batch_size: Batch size used to compute embeddings
        :param show_progress_bar: If true, prints a progress bar
        :param write_csv: Write results to a CSV file
        """
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.name = name

        assert len(self.anchors) == len(self.positives)
        assert len(self.anchors) == len(self.negatives)

        self.best_scoring_function = score_function

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "triplet_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy_cosinus", "accuracy_manhattan", "accuracy_euclidean"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        anchors = []
        positives = []
        negatives = []

        for example in examples:
            anchors.append(example.texts[0])
            positives.append(example.texts[1])
            negatives.append(example.texts[2])
        return cls(anchors, positives, negatives, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("TripletEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        num_triplets = 0
        num_correct_cos_triplets, num_correct_manhattan_triplets, num_correct_euclidean_triplets = 0, 0, 0

        embeddings_anchors = model.encode(
            self.anchors, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        embeddings_positives = model.encode(
            self.positives, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )
        embeddings_negatives = model.encode(
            self.negatives, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True
        )

        # Cosine distance
        pos_cos_sim = 1-paired_cosine_distances(embeddings_anchors, embeddings_positives)
        neg_cos_sims = 1-paired_cosine_distances(embeddings_anchors, embeddings_negatives)

        # Manhattan
        pos_manhattan_sim = -paired_manhattan_distances(embeddings_anchors, embeddings_positives)
        neg_manhattan_sims = -paired_manhattan_distances(embeddings_anchors, embeddings_negatives)

        # Euclidean
        pos_euclidean_sim = -paired_euclidean_distances(embeddings_anchors, embeddings_positives)
        neg_euclidean_sims = -paired_euclidean_distances(embeddings_anchors, embeddings_negatives)

        for idx in range(len(pos_cos_sim)):
            num_triplets += 1

            if pos_cos_sim[idx] > neg_cos_sims[idx]:
                num_correct_cos_triplets += 1

            if pos_manhattan_sim[idx] > neg_manhattan_sims[idx]:
                num_correct_manhattan_triplets += 1

            if pos_euclidean_sim[idx] > neg_euclidean_sims[idx]:
                num_correct_euclidean_triplets += 1

        accuracy_cos = num_correct_cos_triplets / num_triplets
        accuracy_manhattan = num_correct_manhattan_triplets / num_triplets
        accuracy_euclidean = num_correct_euclidean_triplets / num_triplets
        accs =  {
            SimilarityFunction.COSINE.value: accuracy_cos,
            SimilarityFunction.MANHATTAN.value: accuracy_manhattan,
            SimilarityFunction.EUCLIDEAN.value: accuracy_euclidean,
        }

        logger.info("Accuracy Cosine Similarity:   \t{:.2f}".format(accuracy_cos * 100))
        logger.info("Accuracy Manhattan Similarity:\t{:.2f}".format(accuracy_manhattan * 100))
        logger.info("Accuracy Euclidean Similarity:\t{:.2f}\n".format(accuracy_euclidean * 100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhattan, accuracy_euclidean])

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhattan, accuracy_euclidean])

        if self.best_scoring_function == SimilarityFunction.COSINE:
            return accuracy_cos
        if self.best_scoring_function == SimilarityFunction.MANHATTAN:
            return accuracy_manhattan
        if self.best_scoring_function == SimilarityFunction.EUCLIDEAN:
            return accuracy_euclidean

        key_acc_max = max(accs, key=lambda x: accs[x]) 
        self.best_scoring_function = key_acc_max 
        return accs[key_acc_max]
