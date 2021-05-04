"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""
from sentence_transformers import SentenceTransformer,  InputExample, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import unittest
import os
import gzip
import csv

class PretrainedSTSbTest(unittest.TestCase):

    def pretrained_model_score(self, model_name, expected_score):
        model = SentenceTransformer(model_name)
        sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

        if not os.path.exists(sts_dataset_path):
            util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

        train_samples = []
        dev_samples = []
        test_samples = []
        with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

                if row['split'] == 'dev':
                    dev_samples.append(inp_example)
                elif row['split'] == 'test':
                    test_samples.append(inp_example)
                else:
                    train_samples.append(inp_example)

        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')

        score = model.evaluate(evaluator)*100
        print(model_name, "{:.2f} vs. exp: {:.2f}".format(score, expected_score))
        assert score > expected_score or abs(score-expected_score) < 0.1

    def test_bert_base(self):
        self.pretrained_model_score('bert-base-nli-mean-tokens', 77.12)
        self.pretrained_model_score('bert-base-nli-max-tokens', 77.21)
        self.pretrained_model_score('bert-base-nli-cls-token', 76.30)
        self.pretrained_model_score('bert-base-nli-stsb-mean-tokens', 85.14)


    def test_bert_large(self):
        self.pretrained_model_score('bert-large-nli-mean-tokens', 79.19)
        self.pretrained_model_score('bert-large-nli-max-tokens', 78.41)
        self.pretrained_model_score('bert-large-nli-cls-token', 78.29)
        self.pretrained_model_score('bert-large-nli-stsb-mean-tokens', 85.29)

    def test_roberta(self):
        self.pretrained_model_score('roberta-base-nli-mean-tokens', 77.49)
        self.pretrained_model_score('roberta-large-nli-mean-tokens', 78.69)
        self.pretrained_model_score('roberta-base-nli-stsb-mean-tokens', 85.30)
        self.pretrained_model_score('roberta-large-nli-stsb-mean-tokens', 86.39)

    def test_distilbert(self):
        self.pretrained_model_score('distilbert-base-nli-mean-tokens', 78.69)
        self.pretrained_model_score('distilbert-base-nli-stsb-mean-tokens', 85.16)
        self.pretrained_model_score('paraphrase-distilroberta-base-v1', 81.81)

    def test_multiling(self):
        self.pretrained_model_score('distiluse-base-multilingual-cased', 80.75)
        self.pretrained_model_score('paraphrase-xlm-r-multilingual-v1', 83.50)

    def test_other_models(self):
        self.pretrained_model_score('average_word_embeddings_komninos', 61.56)

    def test_msmarco(self):
        self.pretrained_model_score('msmarco-roberta-base-ance-fristp', 77.0)
        self.pretrained_model_score('msmarco-distilbert-base-v3', 78.85)


if "__main__" == __name__:
    unittest.main()