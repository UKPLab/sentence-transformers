"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""
import csv
import gzip
import os
import unittest

from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample


class PretrainedSTSbTest(unittest.TestCase):
    def setUp(self):
        sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
        if not os.path.exists(sts_dataset_path):
            util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

        nli_dataset_path = 'datasets/AllNLI.tsv.gz'
        if not os.path.exists(nli_dataset_path):
            util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

        #Read NLI
        label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        self.nli_train_samples = []
        max_train_samples = 10000
        with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'train':
                    label_id = label2int[row['label']]
                    self.nli_train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
                    if len(self.nli_train_samples) >= max_train_samples:
                        break

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
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(self.test_samples, name='sts-test')
        score = model.evaluate(evaluator)*100
        print("STS-Test Performance: {:.2f} vs. exp: {:.2f}".format(score, expected_score))
        assert score > expected_score or abs(score-expected_score) < 0.1

    def test_train_stsb(self):
        word_embedding_model = models.Transformer('distilbert-base-uncased')
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        train_dataset = SentencesDataset(self.stsb_train_samples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model=model)
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=None,
                  epochs=1,
                  evaluation_steps=1000,
                  warmup_steps=int(len(train_dataloader)*0.1),
                  use_amp=True)

        self.evaluate_stsb_test(model, 80.0)

    def test_train_nli(self):
        word_embedding_model = models.Transformer('distilbert-base-uncased')
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        train_dataset = SentencesDataset(self.nli_train_samples, model=model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
        train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=3)
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=None,
                  epochs=1,
                  warmup_steps=int(len(train_dataloader) * 0.1),
                  use_amp=True)

        self.evaluate_stsb_test(model, 50.0)



if "__main__" == __name__:
    unittest.main()