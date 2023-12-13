"""
Tests general behaviour of the SentenceTransformer class
"""


from pathlib import Path
import tempfile

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
import unittest


class TestSentenceTransformer(unittest.TestCase):
    def test_load_with_safetensors(self):
        with tempfile.TemporaryDirectory() as cache_folder:
            safetensors_model = SentenceTransformer(
                "sentence-transformers-testing/stsb-bert-tiny-safetensors",
                cache_folder=cache_folder,
            )

            # Only the safetensors file must be loaded
            pytorch_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
            self.assertEqual(0, len(pytorch_files), msg="PyTorch model file must not be downloaded.")
            safetensors_files = list(Path(cache_folder).glob("**/model.safetensors"))
            self.assertEqual(1, len(safetensors_files), msg="Safetensors model file must be downloaded.")

        with tempfile.TemporaryDirectory() as cache_folder:
            transformer = Transformer(
                "sentence-transformers-testing/stsb-bert-tiny-safetensors",
                cache_dir=cache_folder,
                model_args={"use_safetensors": False},
            )
            pooling = Pooling(transformer.get_word_embedding_dimension())
            pytorch_model = SentenceTransformer(modules=[transformer, pooling])

            # Only the pytorch file must be loaded
            pytorch_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
            self.assertEqual(1, len(pytorch_files), msg="PyTorch model file must be downloaded.")
            safetensors_files = list(Path(cache_folder).glob("**/model.safetensors"))
            self.assertEqual(0, len(safetensors_files), msg="Safetensors model file must not be downloaded.")

        sentences = ["This is a test sentence", "This is another test sentence"]
        self.assertTrue(
            torch.equal(safetensors_model.encode(sentences, convert_to_tensor=True), pytorch_model.encode(sentences, convert_to_tensor=True)),
            msg="Ensure that Safetensors and PyTorch loaded models result in identical embeddings",
        )

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
        self.assertEqual(model.device.type, "cpu", msg="Ensure that setting `_target_device` doesn't crash.")