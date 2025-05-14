"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""

from __future__ import annotations

import csv
import gzip
import os
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    evaluation,
    losses,
    util,
)


@pytest.mark.skip(reason="This test is rather slow, and the LabelAccuracyEvaluator is not commonly used.")
def test_LabelAccuracyEvaluator(paraphrase_distilroberta_base_v1_model: SentenceTransformer, tmp_path: Path) -> None:
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
    metrics = evaluator(model, output_path=str(tmp_path))
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.2
