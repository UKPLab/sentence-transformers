from __future__ import annotations

import logging
from contextlib import nullcontext

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef

from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder
from sentence_transformers.util import (
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
)

logger = logging.getLogger(__name__)


class SparseBinaryClassificationEvaluator(BinaryClassificationEvaluator):
    """
    Evaluate a sparse model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity, dot score, Euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    Args:
        sentences1 (List[str]): The first column of sentences.
        sentences2 (List[str]): The second column of sentences.
        labels (List[int]): labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1.
        name (str, optional): Name for the output. Defaults to "".
        batch_size (int, optional): Batch size used to compute embeddings. Defaults to 32.
        show_progress_bar (bool, optional): If true, prints a progress bar. Defaults to False.
        write_csv (bool, optional): Write results to a CSV file. Defaults to True.
        truncate_dim (Optional[int], optional): The dimension to truncate sentence embeddings to. `None` uses the model's current truncation dimension. Defaults to None.
        similarity_fn_names (Optional[List[Literal["cosine", "dot", "euclidean", "manhattan"]]], optional): The similarity functions to use. If not specified, defaults to the ``similarity_fn_name`` attribute of the model. Defaults to None.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import BinaryClassificationEvaluator
            from datasets import load_dataset

            # Load a model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Load a dataset with two text columns and a class label column (https://huggingface.co/datasets/sentence-transformers/quora-duplicates)
            eval_dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train[-1000:]")

            # Initialize the evaluator
            binary_acc_evaluator = BinaryClassificationEvaluator(
                sentences1=eval_dataset["sentence1"],
                sentences2=eval_dataset["sentence2"],
                labels=eval_dataset["label"],
                name="quora_duplicates_dev",
            )
            results = binary_acc_evaluator(model)
            '''
            Binary Accuracy Evaluation of the model on the quora_duplicates_dev dataset:
            Accuracy with Cosine-Similarity:             81.60  (Threshold: 0.8352)
            F1 with Cosine-Similarity:                   75.27  (Threshold: 0.7715)
            Precision with Cosine-Similarity:            65.81
            Recall with Cosine-Similarity:               87.89
            Average Precision with Cosine-Similarity:    76.03
            Matthews Correlation with Cosine-Similarity: 62.48
            '''
            print(binary_acc_evaluator.primary_metric)
            # => "quora_duplicates_dev_cosine_ap"
            print(results[binary_acc_evaluator.primary_metric])
            # => 0.760277070888393
    """

    def __call__(
        self,
        model: SparseEncoder,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        if not isinstance(model, SparseEncoder):
            raise ValueError("This evaluator is designed for SparseEncoder models only")

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"Binary Accuracy Evaluation of the model on the {self.name} dataset{out_txt}:")

        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(self.truncate_dim):
            embeddings1 = model.encode(
                self.sentences1,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor_sparse=True,
            )
            embeddings2 = model.encode(
                self.sentences2,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor_sparse=True,
            )

        if not self.similarity_fn_names:
            self.similarity_fn_names = [model.similarity_fn_name]
            self._append_csv_headers(self.similarity_fn_names)

        similarity_fns = {
            SimilarityFunction.COSINE.value: {
                "score_fn": lambda x, y: pairwise_cos_sim(x, y),
                "name": "Cosine-Similarity",
                "greater_is_better": True,
            },
            SimilarityFunction.DOT_PRODUCT.value: {
                "score_fn": lambda x, y: pairwise_dot_score(x, y),
                "name": "Dot-Product",
                "greater_is_better": True,
            },
            SimilarityFunction.MANHATTAN.value: {
                "score_fn": lambda x, y: pairwise_manhattan_sim(x, y),
                "name": "Manhattan-Distance",
                "greater_is_better": False,
            },
            SimilarityFunction.EUCLIDEAN.value: {
                "score_fn": lambda x, y: pairwise_euclidean_sim(x, y),
                "name": "Euclidean-Distance",
                "greater_is_better": False,
            },
        }

        labels = np.asarray(self.labels)
        output_scores = {}
        for similarity_fn_name in self.similarity_fn_names:
            similarity_fn = similarity_fns[similarity_fn_name]
            scores = similarity_fn["score_fn"](embeddings1, embeddings2).cpu().numpy()
            greater_is_better = similarity_fn["greater_is_better"]
            name = similarity_fn["name"]

            acc, acc_threshold = self.find_best_acc_and_threshold(scores, labels, greater_is_better)
            f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(scores, labels, greater_is_better)
            ap = average_precision_score(labels, scores * (1 if greater_is_better else -1))

            predicted_labels = (scores >= f1_threshold) if greater_is_better else (scores <= f1_threshold)
            mcc = matthews_corrcoef(labels, predicted_labels)

            logger.info(f"Accuracy with {name}:             {acc * 100:.2f}\t(Threshold: {acc_threshold:.4f})")
            logger.info(f"F1 with {name}:                   {f1 * 100:.2f}\t(Threshold: {f1_threshold:.4f})")
            logger.info(f"Precision with {name}:            {precision * 100:.2f}")
            logger.info(f"Recall with {name}:               {recall * 100:.2f}")
            logger.info(f"Average Precision with {name}:    {ap * 100:.2f}")
            logger.info(f"Matthews Correlation with {name}: {mcc * 100:.2f}\n")

            output_scores[similarity_fn_name] = {
                "accuracy": acc,
                "accuracy_threshold": acc_threshold,
                "f1": f1,
                "f1_threshold": f1_threshold,
                "precision": precision,
                "recall": recall,
                "ap": ap,
                "mcc": mcc,
            }

        metrics = {
            f"{short_name}_{metric}": value
            for short_name, values in output_scores.items()
            for metric, value in values.items()
        }
        if len(self.similarity_fn_names) > 1:
            metrics.update(
                {
                    f"max_{metric}": max(output_scores[short_name][metric] for short_name in output_scores)
                    for metric in output_scores["cosine"]
                }
            )
            self.primary_metric = "max_ap"
        else:
            self.primary_metric = f"{self.similarity_fn_names[0]}_ap"

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics
