"""
Computes embeddings
"""

import csv
import gzip
import os
import unittest

from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import numpy as np

class ComputeEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    def test_encode_single_sentences(self):
        #Single sentence
        emb = self.model.encode("Hello Word, a test sentence")
        assert emb.shape == (768,)
        assert abs(np.sum(emb) - 7.9811716) < 0.001

        # Single sentence as list
        emb = self.model.encode(["Hello Word, a test sentence"])
        assert emb.shape == (1, 768)
        assert abs(np.sum(emb) - 7.9811716) < 0.001

        # Sentence list
        emb = self.model.encode(["Hello Word, a test sentence", "Here comes another sentence", "My final sentence"])
        assert emb.shape == (3, 768)
        print(np.sum(emb))
        assert abs(np.sum(emb) - 22.968266) < 0.001

    def test_encode_tuple_sentences(self):
        # Input a sentence tuple
        emb = self.model.encode([("Hello Word, a test sentence", "Second input for model")])
        assert emb.shape == (1, 768)
        assert abs(np.sum(emb) - 9.503508) < 0.001

        # List of sentence tuples
        emb = self.model.encode([("Hello Word, a test sentence", "Second input for model"), ("My second tuple", "With two inputs"), ("Final tuple", "final test")])
        assert emb.shape == (3, 768)
        assert abs(np.sum(emb) - 32.14627) < 0.001


