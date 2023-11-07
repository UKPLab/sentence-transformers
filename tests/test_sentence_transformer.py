"""
Tests general behaviour of the SentenceTransformer class
"""


import torch
from sentence_transformers import SentenceTransformer
import unittest

class TestSentenceTransformer(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
    def test_to(self):
        model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu")

        test_device = torch.device("cuda")
        self.assertEqual(model.device.type, "cpu")
        self.assertEqual(test_device.type, "cuda")

        model.to(test_device)
        self.assertEqual(model.device.type, "cuda", msg="The model device should have updated")

        model.encode("Test sentence")
        self.assertEqual(model.device.type, "cuda", msg="Encoding shouldn't change the device")

        self.assertEqual(model._target_device, model.device, msg="Prevent backwards compatibility failure for _target_device")
        model._target_device = "cpu"
        self.assertEqual(model.device.type, "cpu", msg="Ensure that setting `_target_device` still works.")