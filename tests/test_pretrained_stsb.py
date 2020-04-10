"""
Tests that the pretrained models produce the correct scores on the STSb dataset
"""
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSDataReader
import unittest


class PretrainedSTSbTest(unittest.TestCase):
    def pretrained_model_score(self, model_name, expected_score):
        model = SentenceTransformer(model_name)
        sts_reader = STSDataReader('../examples/datasets/stsbenchmark')

        test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=8)
        evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

        score = model.evaluate(evaluator)*100
        print(model_name, "{:.2f} vs. exp: {:.2f}".format(score, expected_score))
        assert abs(score-expected_score) < 0.1

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
        self.pretrained_model_score('roberta-base-nli-stsb-mean-tokens', 85.44)
        self.pretrained_model_score('roberta-large-nli-stsb-mean-tokens', 86.39)

    def test_distilbert(self):
        self.pretrained_model_score('distilbert-base-nli-mean-tokens', 76.97)
        self.pretrained_model_score('distilbert-base-nli-stsb-mean-tokens', 84.38)

    def test_multiling(self):
        self.pretrained_model_score('distiluse-base-multilingual-cased', 80.62)

if "__main__" == __name__:
    unittest.main()