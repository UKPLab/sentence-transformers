"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""

import csv
import gzip
import os
from pathlib import Path
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader

from sentence_transformers import CrossEncoder, util
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.readers import InputExample
from typing import Generator, List, Tuple


@pytest.fixture()
def sts_resource() -> Generator[Tuple[List[InputExample], List[InputExample]], None, None]:
    sts_dataset_path = "datasets/stsbenchmark.tsv.gz"
    if not os.path.exists(sts_dataset_path):
        util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)

    stsb_train_samples = []
    stsb_test_samples = []
    with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)

            if row["split"] == "test":
                stsb_test_samples.append(inp_example)
            elif row["split"] == "train":
                stsb_train_samples.append(inp_example)
    yield stsb_train_samples, stsb_test_samples


def evaluate_stsb_test(
    distilroberta_base_ce_model: CrossEncoder,
    expected_score: float,
    test_samples: List[InputExample],
    num_test_samples: int = -1,
) -> None:
    model = distilroberta_base_ce_model
    evaluator = CECorrelationEvaluator.from_input_examples(test_samples[:num_test_samples], name="sts-test")
    score = evaluator(model) * 100
    print("STS-Test Performance: {:.2f} vs. exp: {:.2f}".format(score, expected_score))
    assert score > expected_score or abs(score - expected_score) < 0.1


def test_pretrained_stsb(sts_resource: Tuple[List[InputExample], List[InputExample]]):
    _, sts_test_samples = sts_resource
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    evaluate_stsb_test(model, 87.92, sts_test_samples)


@pytest.mark.slow
def test_train_stsb_slow(
    distilroberta_base_ce_model: CrossEncoder, sts_resource: Tuple[List[InputExample], List[InputExample]]
) -> None:
    model = distilroberta_base_ce_model
    sts_train_samples, sts_test_samples = sts_resource
    train_dataloader = DataLoader(sts_train_samples, shuffle=True, batch_size=16)
    model.fit(
        train_dataloader=train_dataloader,
        epochs=1,
        warmup_steps=int(len(train_dataloader) * 0.1),
    )
    evaluate_stsb_test(model, 75, sts_test_samples)


def test_train_stsb(
    distilroberta_base_ce_model: CrossEncoder, sts_resource: Tuple[List[InputExample], List[InputExample]]
) -> None:
    model = distilroberta_base_ce_model
    sts_train_samples, sts_test_samples = sts_resource
    train_dataloader = DataLoader(sts_train_samples[:500], shuffle=True, batch_size=16)
    model.fit(
        train_dataloader=train_dataloader,
        epochs=1,
        warmup_steps=int(len(train_dataloader) * 0.1),
    )
    evaluate_stsb_test(model, 50, sts_test_samples, num_test_samples=100)


def test_classifier_dropout_is_set() -> None:
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base", classifier_dropout=0.1234)
    assert model.config.classifier_dropout == 0.1234
    assert model.model.config.classifier_dropout == 0.1234


def test_classifier_dropout_default_value() -> None:
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    assert model.config.classifier_dropout is None
    assert model.model.config.classifier_dropout is None


def test_load_with_revision() -> None:
    model_name = "sentence-transformers-testing/stsb-bert-tiny-safetensors"

    main_model = CrossEncoder(model_name, num_labels=1, revision="main")
    latest_model = CrossEncoder(
        model_name,
        num_labels=1,
        revision="f3cb857cba53019a20df283396bcca179cf051a4",
    )
    older_model = CrossEncoder(
        model_name,
        num_labels=1,
        revision="ba33022fdf0b0fc2643263f0726f44d0a07d0e24",
    )

    # Set the classifier.bias and classifier.weight equal among models. This
    # is needed because the AutoModelForSequenceClassification randomly initializes
    # the classifier.bias and classifier.weight for each (model) initialization.
    # The test is only possible if all models have the same classifier.bias
    # and classifier.weight parameters.
    latest_model.model.classifier.bias = main_model.model.classifier.bias
    latest_model.model.classifier.weight = main_model.model.classifier.weight
    older_model.model.classifier.bias = main_model.model.classifier.bias
    older_model.model.classifier.weight = main_model.model.classifier.weight

    test_sentences = [["Hello there!", "Hello, World!"]]
    main_prob = main_model.predict(test_sentences, convert_to_tensor=True)
    assert torch.equal(main_prob, latest_model.predict(test_sentences, convert_to_tensor=True))
    assert not torch.equal(main_prob, older_model.predict(test_sentences, convert_to_tensor=True))


def test_rank() -> None:
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    # We want to compute the similarity between the query sentence
    query = "A man is eating pasta."

    # With all sentences in the corpus
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ]
    expected_ranking = [0, 1, 3, 6, 2, 5, 7, 4, 8]

    # 1. We rank all sentences in the corpus for the query
    ranks = model.rank(query, corpus)
    pred_ranking = [rank["corpus_id"] for rank in ranks]
    assert pred_ranking == expected_ranking


@pytest.mark.parametrize("safe_serialization", [True, False, None])
def test_safe_serialization(safe_serialization: bool) -> None:
    with tempfile.TemporaryDirectory() as cache_folder:
        model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
        if safe_serialization:
            model.save(cache_folder, safe_serialization=safe_serialization)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        elif safe_serialization is None:
            model.save(cache_folder)
            model_files = list(Path(cache_folder).glob("**/model.safetensors"))
            assert 1 == len(model_files)
        else:
            model.save(cache_folder, safe_serialization=safe_serialization)
            model_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
            assert 1 == len(model_files)
