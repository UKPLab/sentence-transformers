"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""

from __future__ import annotations

import csv
import gzip
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    evaluation,
    losses,
    util,
)


def test_BinaryClassificationEvaluator_find_best_f1_and_threshold() -> None:
    """Tests that the F1 score for the computed threshold is correct"""
    y_true = np.random.randint(0, 2, 1000)
    y_pred_cosine = np.random.randn(1000)
    (
        best_f1,
        best_precision,
        best_recall,
        threshold,
    ) = evaluation.BinaryClassificationEvaluator.find_best_f1_and_threshold(
        y_pred_cosine, y_true, high_score_more_similar=True
    )
    y_pred_labels = [1 if pred >= threshold else 0 for pred in y_pred_cosine]
    sklearn_f1score = f1_score(y_true, y_pred_labels)
    assert np.abs(best_f1 - sklearn_f1score) < 1e-6


def test_BinaryClassificationEvaluator_find_best_accuracy_and_threshold() -> None:
    """Tests that the Acc score for the computed threshold is correct"""
    y_true = np.random.randint(0, 2, 1000)
    y_pred_cosine = np.random.randn(1000)
    (
        max_acc,
        threshold,
    ) = evaluation.BinaryClassificationEvaluator.find_best_acc_and_threshold(
        y_pred_cosine, y_true, high_score_more_similar=True
    )
    y_pred_labels = [1 if pred >= threshold else 0 for pred in y_pred_cosine]
    sklearn_acc = accuracy_score(y_true, y_pred_labels)
    assert np.abs(max_acc - sklearn_acc) < 1e-6


def test_LabelAccuracyEvaluator(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    """Tests that the LabelAccuracyEvaluator can be loaded correctly"""
    model = paraphrase_distilroberta_base_v1_model
    nli_dataset_path = "datasets/AllNLI.tsv.gz"
    if not os.path.exists(nli_dataset_path):
        util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    dev_samples = []
    with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if row["split"] == "train":
                label_id = label2int[row["label"]]
                dev_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=label_id))
                if len(dev_samples) >= 100:
                    break

    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=len(label2int),
    )

    dev_dataloader = DataLoader(dev_samples, shuffle=False, batch_size=16)
    evaluator = evaluation.LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss)
    metrics = evaluator(model)
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.2


def test_ParaphraseMiningEvaluator(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    """Tests that the ParaphraseMiningEvaluator can be loaded"""
    model = paraphrase_distilroberta_base_v1_model
    sentences = {
        0: "Hello World",
        1: "Hello World!",
        2: "The cat is on the table",
        3: "On the table the cat is",
    }
    data_eval = evaluation.ParaphraseMiningEvaluator(sentences, [(0, 1), (2, 3)])
    metrics = data_eval(model)
    assert metrics[data_eval.primary_metric] > 0.99
