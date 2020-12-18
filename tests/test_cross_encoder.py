"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""
import csv
import gzip
import os
import unittest

from torch.utils.data import DataLoader
import logging
from sentence_transformers import CrossEncoder, util, LoggingHandler
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator



class CrossEncoderTest(unittest.TestCase):
    def setUp(self):
        sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
        if not os.path.exists(sts_dataset_path):
            util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

        #Read STSB
        self.stsb_train_samples = []
        self.dev_samples = []
        self.test_samples = []
        with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
                inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

                if row['split'] == 'dev':
                    self.dev_samples.append(inp_example)
                elif row['split'] == 'test':
                    self.test_samples.append(inp_example)
                else:
                    self.stsb_train_samples.append(inp_example)

    def evaluate_stsb_test(self, model, expected_score):
        evaluator = CECorrelationEvaluator.from_input_examples(self.test_samples, name='sts-test')
        score = evaluator(model)*100
        print("STS-Test Performance: {:.2f} vs. exp: {:.2f}".format(score, expected_score))
        assert score > expected_score or abs(score-expected_score) < 0.1

    def test_pretrained_stsb(self):
        model = CrossEncoder("sentence-transformers/ce-distilroberta-base-stsb")
        self.evaluate_stsb_test(model, 87.92)

    def test_train_stsb(self):
        model = CrossEncoder('distilroberta-base', num_labels=1)
        train_dataloader = DataLoader(self.stsb_train_samples, shuffle=True, batch_size=16)
        model.fit(train_dataloader=train_dataloader,
                  epochs=1,
                  warmup_steps=int(len(train_dataloader)*0.1))
        self.evaluate_stsb_test(model, 80)




if "__main__" == __name__:
    unittest.main()