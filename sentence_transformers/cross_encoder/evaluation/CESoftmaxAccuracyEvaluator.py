import logging
from scipy.stats import pearsonr, spearmanr
from typing import List
from ... import InputExample
import numpy as np

class CESoftmaxAccuracyEvaluator:
    """
    This evaluator can be used with the CrossEncoder class.

    It is designed for CrossEncoders with 2 or more outputs. It measure the
    accuracy of the predict class vs. the gold labels.
    """
    def __init__(self, sentence_pairs: List[List[str]], labels: List[int], name: str=''):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.name = name

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

        logging.info("CESoftmaxAccuracyEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
        pred_labels = np.argmax(pred_scores, axis=1)

        assert len(pred_labels) == len(self.labels)

        acc = np.sum(pred_labels == self.labels) / len(self.labels)

        logging.info("Accuracy: {:.2f}".format(acc*100))

        return acc