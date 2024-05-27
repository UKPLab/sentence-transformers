"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""

import csv
import gzip
import os
from typing import Generator, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

from sentence_transformers import (
    SentencesDataset,
    SentenceTransformer,
    losses,
    util,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.util import is_training_available


@pytest.fixture()
def sts_resource() -> Generator[Tuple[List[InputExample], List[InputExample]], None, None]:
    sts_dataset_path = "datasets/stsbenchmark.tsv.gz"
    if not os.path.exists(sts_dataset_path):
        util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)

    stsb_train_samples = []
    stsb_test_samples = []
    with gzip.open(sts_dataset_path, "rt", encoding="utf8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)

            if row["split"] == "test":
                stsb_test_samples.append(inp_example)
            elif row["split"] == "train":
                stsb_train_samples.append(inp_example)
    yield stsb_train_samples, stsb_test_samples


@pytest.fixture()
def nli_resource() -> Generator[List[InputExample], None, None]:
    nli_dataset_path = "datasets/AllNLI.tsv.gz"
    if not os.path.exists(nli_dataset_path):
        util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    nli_train_samples = []
    max_train_samples = 10000
    with gzip.open(nli_dataset_path, "rt", encoding="utf8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if row["split"] == "train":
                label_id = label2int[row["label"]]
                nli_train_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=label_id))
                if len(nli_train_samples) >= max_train_samples:
                    break
    yield nli_train_samples


def evaluate_stsb_test(model, expected_score, test_samples) -> None:
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")
    scores = model.evaluate(evaluator)
    score = scores[evaluator.primary_metric] * 100
    print("STS-Test Performance: {:.2f} vs. exp: {:.2f}".format(score, expected_score))
    assert score > expected_score or abs(score - expected_score) < 0.1


@pytest.mark.slow
@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_train_stsb_slow(
    distilbert_base_uncased_model: SentenceTransformer, sts_resource: Tuple[List[InputExample], List[InputExample]]
) -> None:
    model = distilbert_base_uncased_model
    sts_train_samples, sts_test_samples = sts_resource
    train_dataset = SentencesDataset(sts_train_samples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model=model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,
        epochs=1,
        evaluation_steps=1000,
        warmup_steps=int(len(train_dataloader) * 0.1),
        use_amp=torch.cuda.is_available(),
    )

    evaluate_stsb_test(model, 80.0, sts_test_samples)


@pytest.mark.skipif("CI" in os.environ, reason="This test is too slow for the CI (~8 minutes)")
@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_train_stsb(
    distilbert_base_uncased_model: SentenceTransformer, sts_resource: Tuple[List[InputExample], List[InputExample]]
) -> None:
    model = distilbert_base_uncased_model
    sts_train_samples, sts_test_samples = sts_resource
    train_dataset = SentencesDataset(sts_train_samples[:100], model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model=model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,
        epochs=1,
        evaluation_steps=1000,
        warmup_steps=int(len(train_dataloader) * 0.1),
        use_amp=torch.cuda.is_available(),
    )

    evaluate_stsb_test(model, 60.0, sts_test_samples)


@pytest.mark.slow
@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_train_nli_slow(
    distilbert_base_uncased_model: SentenceTransformer,
    nli_resource: List[InputExample],
    sts_resource: Tuple[List[InputExample], List[InputExample]],
):
    model = distilbert_base_uncased_model
    _, sts_test_samples = sts_resource
    train_dataset = SentencesDataset(nli_resource, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=3,
    )
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,
        epochs=1,
        warmup_steps=int(len(train_dataloader) * 0.1),
        use_amp=torch.cuda.is_available(),
    )

    evaluate_stsb_test(model, 50.0, sts_test_samples)


@pytest.mark.skipif("CI" in os.environ, reason="This test is too slow for the CI (~25 minutes)")
@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_train_nli(
    distilbert_base_uncased_model: SentenceTransformer,
    nli_resource: List[InputExample],
    sts_resource: Tuple[List[InputExample], List[InputExample]],
):
    model = distilbert_base_uncased_model
    _, sts_test_samples = sts_resource
    train_dataset = SentencesDataset(nli_resource[:100], model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=3,
    )
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,
        epochs=1,
        warmup_steps=int(len(train_dataloader) * 0.1),
        use_amp=torch.cuda.is_available(),
    )

    evaluate_stsb_test(model, 50.0, sts_test_samples)
