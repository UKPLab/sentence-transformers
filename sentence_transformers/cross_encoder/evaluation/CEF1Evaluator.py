import csv
import logging
import os
from typing import List

import numpy as np
from sklearn.metrics import f1_score

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.readers.InputExample import InputExample

logger = logging.getLogger(__name__)


class CEF1Evaluator:
    """
    CrossEncoder F1 score based evaluator for binary and multiclass tasks.

    The task type (binary or multiclass) is determined from the labels array. For
    binary tasks the returned metric is binary F1 score. For the multiclass tasks
    the returned metric is macro F1 score.

    Args:
        sentence_pairs (List[List[str]]): A list of sentence pairs, where each pair is a list of two strings.
        labels (List[int]): A list of integer labels corresponding to each sentence pair.
        batch_size (int, optional): Batch size for prediction. Defaults to 32.
        show_progress_bar (bool, optional): Show tqdm progress bar.
        name (str, optional): An optional name for the CSV file with stored results. Defaults to an empty string.
        write_csv (bool, optional): Flag to determine if the data should be saved to a CSV file. Defaults to True.
    """

    def __init__(
        self,
        sentence_pairs: List[List[str]],
        labels: List[int],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        name: str = "",
        write_csv: bool = True,
    ) -> None:
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.name = name
        self.write_csv = write_csv

        n_unique = np.unique(labels).size

        if n_unique == 2:
            self.f1_callables = [
                ("Binary F1 score", lambda x, y: f1_score(x, y, average="binary")),
            ]
        elif n_unique > 2:
            self.f1_callables = [
                ("Macro F1 score", lambda x, y: f1_score(x, y, average="macro")),
                ("Micro F1 score", lambda x, y: f1_score(x, y, average="micro")),
                ("Weighted F1 score", lambda x, y: f1_score(x, y, average="weighted")),
            ]
        else:
            raise ValueError(
                "Got only one distinct label. Please make sure there are at least two labels in the `labels` array."
            )

        self.csv_file = "CEF1Evaluator" + (f"_{name}" if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps"] + [metric_name for metric_name, _ in self.f1_callables]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        """
        Create an instance of CEF1Evaluator from a list of InputExample objects.

        Args:
            examples (List[InputExample]): A list of InputExample objects.
            **kwargs: Additional keyword arguments to pass to the CEF1Evaluator constructor.

        Returns:
            CEF1Evaluator: An instance of CEF1Evaluator.
        """
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)

        return cls(sentence_pairs, labels, **kwargs)

    def __call__(self, model: CrossEncoder, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        Evaluate the model using the CEF1Evaluator.

        Args:
            model (CrossEncoder): The cross-encoder model to evaluate.
            output_path (str, optional): The path to save the evaluation results. Defaults to None.
            epoch (int, optional): The epoch number. Defaults to -1.
            steps (int, optional): The number of steps. Defaults to -1.

        Returns:
            float: The F1 score.
        """
        if epoch != -1:
            if steps == -1:
                out_txt = f"after epoch {epoch}:"
            else:
                out_txt = f"in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info(f"CEF1Evaluator: Evaluating the model on {self.name} dataset {out_txt}")
        pred_scores = model.predict(
            self.sentence_pairs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        pred_labels = np.argmax(pred_scores, axis=1)

        assert len(pred_labels) == len(self.labels)

        save_f1 = []
        for f1_name, f1_fn in self.f1_callables:
            f1_val = f1_fn(pred_labels, self.labels)
            save_f1.append(f1_val)
            logger.info(f"{f1_name:20s}: {f1_val * 100:.2f}")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            mode = "a" if output_file_exists else "w"
            with open(csv_path, mode=mode, encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, *save_f1])

        return save_f1[0]
