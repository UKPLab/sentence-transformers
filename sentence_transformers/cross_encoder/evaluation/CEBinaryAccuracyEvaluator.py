import logging
import os
import csv
from typing import List
from ... import InputExample
import numpy as np


logger = logging.getLogger(__name__)


class CEBinaryAccuracyEvaluator:
    """
    This evaluator can be used with the CrossEncoder class.

    It is designed for CrossEncoders with 1 outputs. It measure the
    accuracy of the predict class vs. the gold labels. It uses a fixed threshold to determine the label (0 vs 1).

    See CEBinaryClassificationEvaluator for an evaluator that determines automatically the optimal threshold.
    """

    def __init__(
        self,
        sentence_pairs: List[List[str]],
        labels: List[int],
        name: str = "",
        threshold: float = 0.5,
        write_csv: bool = True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.name = name
        self.threshold = threshold

        self.csv_file = "CEBinaryAccuracyEvaluator" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Accuracy"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CEBinaryAccuracyEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
        pred_labels = pred_scores > self.threshold

        assert len(pred_labels) == len(self.labels)

        acc = np.sum(pred_labels == self.labels) / len(self.labels)

        logger.info("Accuracy: {:.2f}".format(acc * 100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc])

        return acc
