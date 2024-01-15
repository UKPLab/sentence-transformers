"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""
import csv
import gzip
import os

import pytest
from torch.utils.data import DataLoader

from sentence_transformers import (
    SentencesDataset,
    SentenceTransformer,
    losses,
    models,
    util,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample


@pytest.fixture()
def sts_resource():
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
def nli_resource():
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


@pytest.fixture()
def model():
    word_embedding_model = models.Transformer("distilbert-base-uncased")
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


def evaluate_stsb_test(model, expected_score, test_samples):
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")
    score = model.evaluate(evaluator) * 100
    print("STS-Test Performance: {:.2f} vs. exp: {:.2f}".format(score, expected_score))
    assert score > expected_score or abs(score - expected_score) < 0.1


@pytest.mark.slow
def test_train_stsb_slow(sts_resource, model):
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
        use_amp=True,
    )

    evaluate_stsb_test(model, 80.0, sts_test_samples)


def test_train_stsb(model, sts_resource):
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
        use_amp=True,
    )

    evaluate_stsb_test(model, 60.0, sts_test_samples)


@pytest.mark.slow
def test_train_nli_slow(model, nli_resource, sts_resource):
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
        use_amp=True,
    )

    evaluate_stsb_test(model, 50.0, sts_test_samples)


def test_train_nli(model, nli_resource, sts_resource):
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
        use_amp=True,
    )

    evaluate_stsb_test(model, 50.0, sts_test_samples)
