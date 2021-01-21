import logging
from sklearn.metrics import average_precision_score
from typing import List
import numpy as np
import os
import csv

from ... import InputExample
from ...evaluation import BinaryClassificationEvaluator


logger = logging.getLogger(__name__)

class CEBinaryClassificationEvaluator:
    """
    This evaluator can be used with the CrossEncoder class. Given sentence pairs and binary labels (0 and 1),
    it compute the average precision and the best possible f1 score
    """
    def __init__(self, sentence_pairs: List[List[str]], labels: List[int], name: str='', write_csv: bool = True):
        assert len(sentence_pairs) == len(labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.sentence_pairs = sentence_pairs
        self.labels = np.asarray(labels)
        self.name = name

        self.csv_file = "CEBinaryClassificationEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Accuracy", "Accuracy_Threshold", "F1", "F1_Threshold", "Precision", "Recall", "Average_Precision"]
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

        logger.info("CEBinaryClassificationEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)

        acc, acc_threshold = BinaryClassificationEvaluator.find_best_acc_and_threshold(pred_scores, self.labels, True)
        f1, precision, recall, f1_threshold = BinaryClassificationEvaluator.find_best_f1_and_threshold(pred_scores, self.labels, True)
        ap = average_precision_score(self.labels, pred_scores)

        logger.info("Accuracy:           {:.2f}\t(Threshold: {:.4f})".format(acc * 100, acc_threshold))
        logger.info("F1:                 {:.2f}\t(Threshold: {:.4f})".format(f1 * 100, f1_threshold))
        logger.info("Precision:          {:.2f}".format(precision * 100))
        logger.info("Recall:             {:.2f}".format(recall * 100))
        logger.info("Average Precision:  {:.2f}\n".format(ap * 100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc, acc_threshold, f1, f1_threshold, precision, recall, ap])


        return ap
