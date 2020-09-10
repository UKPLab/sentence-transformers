"""
Tests the WKPooling model
"""
import unittest
from sentence_transformers import models, SentenceTransformer
import scipy

class WKPoolingTest(unittest.TestCase):
    sentence_pairs = [
        ('Can you please. Send me the attachment.', 'I dont know. Where is it?'),
        ('My name is Paul', 'My name is Lisa'),
        ('The cat sits on the mat while the dog is barking', 'London is the capital of England'),
        ('BERT (Devlin et al., 2018) and RoBERTa (Liu et al., 2019) has set a new state-of-the-art performance on sentence-pair regression tasks like semantic textual similarity (STS)', 'However, it requires that both sentences are fed into the network, which causes a massive computational overhead: Finding the most similar pair in a collection of 10,000 sentences requires about 50 million inference computations (~65 hours) with BERT.'),
        ('In deep learning, each level learns to transform its input data into a slightly more abstract and composite representation.', 'London is considered to be one of the world\'s most important global cities.')
    ]

    def test_bert_wkpooling(self):
        word_embedding_model = models.BERT('bert-base-uncased', model_args={'output_hidden_states': True})
        pooling_model = models.WKPooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        scores = [0.6906377742193329,
                  0.9910573945907297,
                  0.8395676755959804,
                  0.7569234597143,
                  0.8324509121875274]

        for sentences, score in zip(WKPoolingTest.sentence_pairs, scores):
            embedding = model.encode(sentences, convert_to_numpy=True)

            similarity = 1-scipy.spatial.distance.cosine(embedding[0], embedding[1])
            assert abs(similarity-score) < 0.01

    def test_roberta_wkpooling(self):
        word_embedding_model = models.Transformer('roberta-base', model_args={'output_hidden_states': True})
        pooling_model = models.WKPooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        scores = [0.9594874382019043,
                  0.9928674697875977,
                  0.9241214990615845,
                  0.9309519529342651,
                  0.9506515264511108]

        for sentences, score in zip(WKPoolingTest.sentence_pairs, scores):
            embedding = model.encode(sentences, convert_to_numpy=True)

            similarity = 1-scipy.spatial.distance.cosine(embedding[0], embedding[1])
            assert abs(similarity-score) < 0.01


if "__main__" == __name__:
    unittest.main()