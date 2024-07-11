from __future__ import annotations

import csv
import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.similarity_functions import SimilarityFunction

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    Example:
        ::

            from datasets import load_dataset
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

            # Load a model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
            eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

            # Initialize the evaluator
            dev_evaluator = EmbeddingSimilarityEvaluator(
                sentences1=eval_dataset["sentence1"],
                sentences2=eval_dataset["sentence2"],
                scores=eval_dataset["score"],
                main_similarity=SimilarityFunction.COSINE,
                name="sts-dev",
            )
            dev_evaluator(model)
            '''
            EmbeddingSimilarityEvaluator: Evaluating the model on the sts-dev dataset:
            Cosine-Similarity :       Pearson: 0.7874 Spearman: 0.8004
            Manhattan-Distance:       Pearson: 0.7823 Spearman: 0.7827
            Euclidean-Distance:       Pearson: 0.7824 Spearman: 0.7827
            Dot-Product-Similarity:   Pearson: 0.7192 Spearman: 0.7126
            '''
            # => {'sts-dev_pearson_cosine': 0.880607226102985, 'sts-dev_spearman_cosine': 0.881019449484294, ...}
    """

    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        scores: list[float],
        batch_size: int = 16,
        main_similarity: str | SimilarityFunction | None = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] | None = None,
        truncate_dim: int | None = None,
    ):
        """
        Constructs an evaluator based for the dataset.

        Args:
            sentences1 (List[str]): List with the first sentence in a pair.
            sentences2 (List[str]): List with the second sentence in a pair.
            scores (List[float]): Similarity score between sentences1[i] and sentences2[i].
            batch_size (int, optional): The batch size for processing the sentences. Defaults to 16.
            main_similarity (Optional[Union[str, SimilarityFunction]], optional): The main similarity function to use.
                Can be a string (e.g. "cosine", "dot") or a SimilarityFunction object. Defaults to None.
            name (str, optional): The name of the evaluator. Defaults to "".
            show_progress_bar (bool, optional): Whether to show a progress bar during evaluation. Defaults to False.
            write_csv (bool, optional): Whether to write the evaluation results to a CSV file. Defaults to True.
            precision (Optional[Literal["float32", "int8", "uint8", "binary", "ubinary"]], optional): The precision
                to use for the embeddings. Can be "float32", "int8", "uint8", "binary", or "ubinary". Defaults to None.
            truncate_dim (Optional[int], optional): The dimension to truncate sentence embeddings to. `None` uses the
                model's current truncation dimension. Defaults to None.
        """
        super().__init__()
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.write_csv = write_csv
        self.precision = precision
        self.truncate_dim = truncate_dim

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = SimilarityFunction(main_similarity) if main_similarity else None
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = (
            "similarity_evaluation"
            + ("_" + name if name else "")
            + ("_" + precision if precision else "")
            + "_results.csv"
        )
        self.csv_headers = [
            "epoch",
            "steps",
            "cosine_pearson",
            "cosine_spearman",
            "euclidean_pearson",
            "euclidean_spearman",
            "manhattan_pearson",
            "manhattan_spearman",
            "dot_pearson",
            "dot_spearman",
        ]

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def __call__(
        self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"EmbeddingSimilarityEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            embeddings1 = model.encode(
                self.sentences1,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                precision=self.precision,
                normalize_embeddings=bool(self.precision),
            )
            embeddings2 = model.encode(
                self.sentences2,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                precision=self.precision,
                normalize_embeddings=bool(self.precision),
            )
        # Binary and ubinary embeddings are packed, so we need to unpack them for the distance metrics
        if self.precision == "binary":
            embeddings1 = (embeddings1 + 128).astype(np.uint8)
            embeddings2 = (embeddings2 + 128).astype(np.uint8)
        if self.precision in ("ubinary", "binary"):
            embeddings1 = np.unpackbits(embeddings1, axis=1)
            embeddings2 = np.unpackbits(embeddings2, axis=1)

        labels = self.scores

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        logger.info(f"Cosine-Similarity :\tPearson: {eval_pearson_cosine:.4f}\tSpearman: {eval_spearman_cosine:.4f}")
        logger.info(
            f"Manhattan-Distance:\tPearson: {eval_pearson_manhattan:.4f}\tSpearman: {eval_spearman_manhattan:.4f}"
        )
        logger.info(
            f"Euclidean-Distance:\tPearson: {eval_pearson_euclidean:.4f}\tSpearman: {eval_spearman_euclidean:.4f}"
        )
        logger.info(f"Dot-Product-Similarity:\tPearson: {eval_pearson_dot:.4f}\tSpearman: {eval_spearman_dot:.4f}")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                        eval_pearson_cosine,
                        eval_spearman_cosine,
                        eval_pearson_euclidean,
                        eval_spearman_euclidean,
                        eval_pearson_manhattan,
                        eval_spearman_manhattan,
                        eval_pearson_dot,
                        eval_spearman_dot,
                    ]
                )

        self.primary_metric = {
            SimilarityFunction.COSINE: "spearman_cosine",
            SimilarityFunction.EUCLIDEAN: "spearman_euclidean",
            SimilarityFunction.MANHATTAN: "spearman_manhattan",
            SimilarityFunction.DOT_PRODUCT: "spearman_dot",
        }.get(self.main_similarity, "spearman_max")
        metrics = {
            "pearson_cosine": eval_pearson_cosine,
            "spearman_cosine": eval_spearman_cosine,
            "pearson_manhattan": eval_pearson_manhattan,
            "spearman_manhattan": eval_spearman_manhattan,
            "pearson_euclidean": eval_pearson_euclidean,
            "spearman_euclidean": eval_spearman_euclidean,
            "pearson_dot": eval_pearson_dot,
            "spearman_dot": eval_spearman_dot,
            "pearson_max": max(eval_pearson_cosine, eval_pearson_manhattan, eval_pearson_euclidean, eval_pearson_dot),
            "spearman_max": max(
                eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot
            ),
        }
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics)
        return metrics

    @property
    def description(self) -> str:
        return "Semantic Similarity"
