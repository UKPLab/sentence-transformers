"""
Compute image embeddings
"""

import unittest
from sentence_transformers import SentenceTransformer, util
import numpy as np
from PIL import Image
import os

class ComputeEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        self.model = SentenceTransformer('clip-ViT-B-32')

    def test_simple_encode(self):
        # Encode an image:
        image_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../examples/applications/image-search/two_dogs_in_snow.jpg")
        print(image_filepath)
        img_emb = self.model.encode(Image.open(image_filepath))

        # Encode text descriptions
        text_emb = self.model.encode(['Two dogs in the snow', 'A cat on a table', 'A picture of London at night'])

        # Compute cosine similarities
        cos_scores = util.cos_sim(img_emb, text_emb)[0]
        assert abs(cos_scores[0] - 0.3069) < 0.01
        assert abs(cos_scores[1] - 0.1010) < 0.01
        assert abs(cos_scores[2] - 0.1086) < 0.01
