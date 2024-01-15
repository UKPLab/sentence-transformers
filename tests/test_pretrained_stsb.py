"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""
import csv
import gzip
import os
from functools import partial

import pytest

from sentence_transformers import InputExample, SentenceTransformer, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


def pretrained_model_score(model_name, expected_score: float, max_test_samples: int = 100) -> None:
    model = SentenceTransformer(model_name)
    sts_dataset_path = "datasets/stsbenchmark.tsv.gz"

    if not os.path.exists(sts_dataset_path):
        util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)

    test_samples = []
    with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)

            if row["split"] == "test":
                test_samples.append(inp_example)
            if max_test_samples != -1 and len(test_samples) >= max_test_samples:
                break

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")

    score = model.evaluate(evaluator) * 100
    print(model_name, "{:.2f} vs. exp: {:.2f}".format(score, expected_score))
    assert score > expected_score or abs(score - expected_score) < 0.1


pretrained_model_score = partial(pretrained_model_score, max_test_samples=100)
pretrained_model_score_slow = partial(pretrained_model_score, max_test_samples=-1)


@pytest.mark.slow
def test_bert_base_slow() -> None:
    pretrained_model_score_slow("bert-base-nli-mean-tokens", 77.12)
    pretrained_model_score_slow("bert-base-nli-max-tokens", 77.21)
    pretrained_model_score_slow("bert-base-nli-cls-token", 76.30)
    pretrained_model_score_slow("bert-base-nli-stsb-mean-tokens", 85.14)


@pytest.mark.slow
def test_bert_large_slow() -> None:
    pretrained_model_score_slow("bert-large-nli-mean-tokens", 79.19)
    pretrained_model_score_slow("bert-large-nli-max-tokens", 78.41)
    pretrained_model_score_slow("bert-large-nli-cls-token", 78.29)
    pretrained_model_score_slow("bert-large-nli-stsb-mean-tokens", 85.29)


@pytest.mark.slow
def test_roberta_slow() -> None:
    pretrained_model_score_slow("roberta-base-nli-mean-tokens", 77.49)
    pretrained_model_score_slow("roberta-large-nli-mean-tokens", 78.69)
    pretrained_model_score_slow("roberta-base-nli-stsb-mean-tokens", 85.30)
    pretrained_model_score_slow("roberta-large-nli-stsb-mean-tokens", 86.39)


@pytest.mark.slow
def test_distilbert_slow() -> None:
    pretrained_model_score_slow("distilbert-base-nli-mean-tokens", 78.69)
    pretrained_model_score_slow("distilbert-base-nli-stsb-mean-tokens", 85.16)
    pretrained_model_score_slow("paraphrase-distilroberta-base-v1", 81.81)


@pytest.mark.slow
def test_multiling_slow() -> None:
    pretrained_model_score_slow("distiluse-base-multilingual-cased", 80.75)
    pretrained_model_score_slow("paraphrase-xlm-r-multilingual-v1", 83.50)
    pretrained_model_score_slow("paraphrase-multilingual-MiniLM-L12-v2", 84.42)


@pytest.mark.slow
def test_mpnet_slow() -> None:
    pretrained_model_score_slow("paraphrase-mpnet-base-v2", 86.99)


@pytest.mark.slow
def test_other_models_slow() -> None:
    pretrained_model_score_slow("average_word_embeddings_komninos", 61.56)


@pytest.mark.slow
def test_msmarco_slow() -> None:
    pretrained_model_score_slow("msmarco-roberta-base-ance-firstp", 77.0)
    pretrained_model_score_slow("msmarco-distilbert-base-v3", 78.85)


@pytest.mark.slow
def test_sentence_t5_slow() -> None:
    pretrained_model_score_slow("sentence-t5-base", 85.52)


def test_bert_base() -> None:
    pretrained_model_score("bert-base-nli-mean-tokens", 86.53)
    pretrained_model_score("bert-base-nli-max-tokens", 87.00)
    pretrained_model_score("bert-base-nli-cls-token", 85.93)
    pretrained_model_score("bert-base-nli-stsb-mean-tokens", 89.26)


def test_bert_large() -> None:
    pretrained_model_score("bert-large-nli-mean-tokens", 90.06)
    pretrained_model_score("bert-large-nli-max-tokens", 90.15)
    pretrained_model_score("bert-large-nli-cls-token", 89.51)
    pretrained_model_score("bert-large-nli-stsb-mean-tokens", 92.27)


def test_roberta() -> None:
    pretrained_model_score("roberta-base-nli-mean-tokens", 87.91)
    pretrained_model_score("roberta-large-nli-mean-tokens", 89.41)
    pretrained_model_score("roberta-base-nli-stsb-mean-tokens", 93.39)
    pretrained_model_score("roberta-large-nli-stsb-mean-tokens", 91.26)


def test_distilbert() -> None:
    pretrained_model_score("distilbert-base-nli-mean-tokens", 88.83)
    pretrained_model_score("distilbert-base-nli-stsb-mean-tokens", 91.01)
    pretrained_model_score("paraphrase-distilroberta-base-v1", 90.89)


def test_multiling() -> None:
    pretrained_model_score("distiluse-base-multilingual-cased", 88.79)
    pretrained_model_score("paraphrase-xlm-r-multilingual-v1", 92.76)
    pretrained_model_score("paraphrase-multilingual-MiniLM-L12-v2", 92.64)


def test_mpnet() -> None:
    pretrained_model_score("paraphrase-mpnet-base-v2", 92.83)


def test_other_models() -> None:
    pretrained_model_score("average_word_embeddings_komninos", 68.97)


def test_msmarco() -> None:
    pretrained_model_score("msmarco-roberta-base-ance-firstp", 83.61)
    pretrained_model_score("msmarco-distilbert-base-v3", 87.96)


def test_sentence_t5() -> None:
    pretrained_model_score("sentence-t5-base", 92.75)
