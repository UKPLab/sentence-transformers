from __future__ import annotations

import csv
import logging
import os

import numpy as np
from sklearn.metrics import average_precision_score

from sentence_transformers import InputExample
from sentence_transformers.evaluation import BinaryClassificationEvaluator

logger = logging.getLogger(__name__)


class CEBinaryClassificationEvaluator:
    """
    This evaluator can be used with the CrossEncoder class. Given sentence pairs and binary labels (0 and 1),
    it compute the average precision and the best possible f1 score
    """

    def __init__(
        self,
        sentence_pairs: list[list[str]],
        labels: list[int],
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        assert len(sentence_pairs) == len(labels)
        for label in labels:
            assert label == 0 or label == 1

        self.sentence_pairs = sentence_pairs
        self.labels = np.asarray(labels)
        self.name = name

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = "CEBinaryClassificationEvaluator" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = [
            "epoch",
            "steps",
            "Accuracy",
            "Accuracy_Threshold",
            "F1",
            "F1_Threshold",
            "Precision",
            "Recall",
            "Average_Precision",
        ]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("CEBinaryClassificationEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(
            self.sentence_pairs, convert_to_numpy=True, show_progress_bar=self.show_progress_bar
        )

        acc, acc_threshold = BinaryClassificationEvaluator.find_best_acc_and_threshold(pred_scores, self.labels, True)
        f1, precision, recall, f1_threshold = BinaryClassificationEvaluator.find_best_f1_and_threshold(
            pred_scores, self.labels, True
        )
        ap = average_precision_score(self.labels, pred_scores)

        logger.info(f"Accuracy:           {acc * 100:.2f}\t(Threshold: {acc_threshold:.4f})")
        logger.info(f"F1:                 {f1 * 100:.2f}\t(Threshold: {f1_threshold:.4f})")
        logger.info(f"Precision:          {precision * 100:.2f}")
        logger.info(f"Recall:             {recall * 100:.2f}")
        logger.info(f"Average Precision:  {ap * 100:.2f}\n")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc, acc_threshold, f1, f1_threshold, precision, recall, ap])

        return ap
