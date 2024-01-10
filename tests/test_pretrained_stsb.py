"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""
from functools import partial
from sentence_transformers import SentenceTransformer, InputExample, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import os
import gzip
import csv
import pytest


def pretrained_model_score(model_name, expected_score, max_test_samples: int = 100):
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


@pytest.mark.slow
class TestPretrainedSTSbSlow:
    pretrained_model_score = partial(pretrained_model_score, max_test_samples=-1)

    def test_bert_base(self):
        self.pretrained_model_score("bert-base-nli-mean-tokens", 77.12)
        self.pretrained_model_score("bert-base-nli-max-tokens", 77.21)
        self.pretrained_model_score("bert-base-nli-cls-token", 76.30)
        self.pretrained_model_score("bert-base-nli-stsb-mean-tokens", 85.14)

    def test_bert_large(self):
        self.pretrained_model_score("bert-large-nli-mean-tokens", 79.19)
        self.pretrained_model_score("bert-large-nli-max-tokens", 78.41)
        self.pretrained_model_score("bert-large-nli-cls-token", 78.29)
        self.pretrained_model_score("bert-large-nli-stsb-mean-tokens", 85.29)

    def test_roberta(self):
        self.pretrained_model_score("roberta-base-nli-mean-tokens", 77.49)
        self.pretrained_model_score("roberta-large-nli-mean-tokens", 78.69)
        self.pretrained_model_score("roberta-base-nli-stsb-mean-tokens", 85.30)
        self.pretrained_model_score("roberta-large-nli-stsb-mean-tokens", 86.39)

    def test_distilbert(self):
        self.pretrained_model_score("distilbert-base-nli-mean-tokens", 78.69)
        self.pretrained_model_score("distilbert-base-nli-stsb-mean-tokens", 85.16)
        self.pretrained_model_score("paraphrase-distilroberta-base-v1", 81.81)

    def test_multiling(self):
        self.pretrained_model_score("distiluse-base-multilingual-cased", 80.75)
        self.pretrained_model_score("paraphrase-xlm-r-multilingual-v1", 83.50)
        self.pretrained_model_score("paraphrase-multilingual-MiniLM-L12-v2", 84.42)

    def test_mpnet(self):
        self.pretrained_model_score("paraphrase-mpnet-base-v2", 86.99)

    def test_other_models(self):
        self.pretrained_model_score("average_word_embeddings_komninos", 61.56)

    def test_msmarco(self):
        self.pretrained_model_score("msmarco-roberta-base-ance-firstp", 77.0)
        self.pretrained_model_score("msmarco-distilbert-base-v3", 78.85)

    def test_sentence_t5(self):
        self.pretrained_model_score("sentence-t5-base", 85.52)


class TestPretrainedSTSbFast:
    pretrained_model_score = partial(pretrained_model_score, max_test_samples=100)

    def test_bert_base(self):
        self.pretrained_model_score("bert-base-nli-mean-tokens", 86.53)
        self.pretrained_model_score("bert-base-nli-max-tokens", 87.00)
        self.pretrained_model_score("bert-base-nli-cls-token", 85.93)
        self.pretrained_model_score("bert-base-nli-stsb-mean-tokens", 89.26)

    def test_bert_large(self):
        self.pretrained_model_score("bert-large-nli-mean-tokens", 90.06)
        self.pretrained_model_score("bert-large-nli-max-tokens", 90.15)
        self.pretrained_model_score("bert-large-nli-cls-token", 89.51)
        self.pretrained_model_score("bert-large-nli-stsb-mean-tokens", 92.27)

    def test_roberta(self):
        self.pretrained_model_score("roberta-base-nli-mean-tokens", 87.91)
        self.pretrained_model_score("roberta-large-nli-mean-tokens", 89.41)
        self.pretrained_model_score("roberta-base-nli-stsb-mean-tokens", 93.39)
        self.pretrained_model_score("roberta-large-nli-stsb-mean-tokens", 91.26)

    def test_distilbert(self):
        self.pretrained_model_score("distilbert-base-nli-mean-tokens", 88.83)
        self.pretrained_model_score("distilbert-base-nli-stsb-mean-tokens", 91.01)
        self.pretrained_model_score("paraphrase-distilroberta-base-v1", 90.89)

    def test_multiling(self):
        self.pretrained_model_score("distiluse-base-multilingual-cased", 88.79)
        self.pretrained_model_score("paraphrase-xlm-r-multilingual-v1", 92.76)
        self.pretrained_model_score("paraphrase-multilingual-MiniLM-L12-v2", 92.64)

    def test_mpnet(self):
        self.pretrained_model_score("paraphrase-mpnet-base-v2", 92.83)

    def test_other_models(self):
        self.pretrained_model_score("average_word_embeddings_komninos", 68.97)

    def test_msmarco(self):
        self.pretrained_model_score("msmarco-roberta-base-ance-firstp", 83.61)
        self.pretrained_model_score("msmarco-distilbert-base-v3", 87.96)

    def test_sentence_t5(self):
        self.pretrained_model_score("sentence-t5-base", 92.75)
