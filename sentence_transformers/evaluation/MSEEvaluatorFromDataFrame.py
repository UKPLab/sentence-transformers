from __future__ import annotations

import csv
import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class MSEEvaluatorFromDataFrame(SentenceEvaluator):
    """
    Computes the mean squared error (x100) between the computed sentence embedding and some target sentence embedding.

    Args:
        dataframe (List[Dict[str, str]]): It must have the following format. Rows contains different, parallel sentences.
            Columns are the respective language codes::

            [{'en': 'My sentence in English', 'es': 'Oración en español', 'fr': 'Phrase en français'...},
             {'en': 'My second sentence', ...}]
        teacher_model (SentenceTransformer): The teacher model used to compute the sentence embeddings.
        combinations (List[Tuple[str, str]]): Must be of the format ``[('en', 'es'), ('en', 'fr'), ...]``.
            First entry in a tuple is the source language. The sentence in the respective language will be fetched from
            the dataframe and passed to the teacher model. Second entry in a tuple the the target language. Sentence
            will be fetched from the dataframe and passed to the student model
        batch_size (int, optional): The batch size to compute sentence embeddings. Defaults to 8.
        name (str, optional): The name of the evaluator. Defaults to "".
        write_csv (bool, optional): Whether to write the results to a CSV file. Defaults to True.
        truncate_dim (Optional[int], optional): The dimension to truncate sentence embeddings to. If None, uses the model's
            current truncation dimension. Defaults to None.
    """

    def __init__(
        self,
        dataframe: list[dict[str, str]],
        teacher_model: SentenceTransformer,
        combinations: list[tuple[str, str]],
        batch_size: int = 8,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
    ):
        super().__init__()
        self.combinations = combinations
        self.name = name
        self.batch_size = batch_size

        if name:
            name = "_" + name

        self.csv_file = "mse_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]
        self.primary_metric = "negative_mse"
        self.write_csv = write_csv
        self.truncate_dim = truncate_dim
        self.data = {}

        logger.info("Compute teacher embeddings")
        all_source_sentences = set()
        for src_lang, trg_lang in self.combinations:
            src_sentences = []
            trg_sentences = []

            for row in dataframe:
                if row[src_lang].strip() != "" and row[trg_lang].strip() != "":
                    all_source_sentences.add(row[src_lang])
                    src_sentences.append(row[src_lang])
                    trg_sentences.append(row[trg_lang])

            self.data[(src_lang, trg_lang)] = (src_sentences, trg_sentences)
            self.csv_headers.append(f"{src_lang}-{trg_lang}")

        all_source_sentences = list(all_source_sentences)
        with (
            nullcontext()
            if self.truncate_dim is None
            else teacher_model.truncate_sentence_embeddings(self.truncate_dim)
        ):
            all_src_embeddings = teacher_model.encode(all_source_sentences, batch_size=self.batch_size)
        self.teacher_embeddings = {sent: emb for sent, emb in zip(all_source_sentences, all_src_embeddings)}

    def __call__(
        self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        model.eval()

        mse_scores = []
        for src_lang, trg_lang in self.combinations:
            src_sentences, trg_sentences = self.data[(src_lang, trg_lang)]

            src_embeddings = np.asarray([self.teacher_embeddings[sent] for sent in src_sentences])
            with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
                trg_embeddings = np.asarray(model.encode(trg_sentences, batch_size=self.batch_size))

            mse = ((src_embeddings - trg_embeddings) ** 2).mean()
            mse *= 100
            mse_scores.append(mse)

            logger.info(f"MSE evaluation on {self.name} dataset - {src_lang}-{trg_lang}:")
            logger.info(f"MSE (*100):\t{mse:4f}")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps] + mse_scores)

        # Return negative score as SentenceTransformers maximizes the performance
        metrics = {"negative_mse": -np.mean(mse_scores).item()}
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    @property
    def description(self) -> str:
        return "Knowledge Distillation"
