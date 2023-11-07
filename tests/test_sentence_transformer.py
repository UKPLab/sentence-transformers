"""
Tests general behaviour of the SentenceTransformer class
"""


import torch
from sentence_transformers import SentenceTransformer
import unittest

class TestSentenceTransformer(unittest.TestCase):
    def test_to(self):
        model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu")

        test_device = torch.device("cpu", index=1)
        self.assertIsNot(test_device, model._target_device, msg="Both should be CPU, but different device instances")

        model.to(test_device)
        self.assertIs(test_device, model._target_device, msg="The model device should update")

        model.encode("Test sentence")
        self.assertIs(test_device, model._target_device, msg="Encoding shouldn't change the device")
