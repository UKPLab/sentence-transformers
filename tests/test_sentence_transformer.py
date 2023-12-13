"""
Tests general behaviour of the SentenceTransformer class
"""


from pathlib import Path
import tempfile
import pytest

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling


def test_load_with_safetensors() -> None:
    with tempfile.TemporaryDirectory() as cache_folder:
        safetensors_model = SentenceTransformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            cache_folder=cache_folder,
        )

        # Only the safetensors file must be loaded
        pytorch_files = list(Path(cache_folder).glob("**/pytorch_model.bin"))
        assert 0 == len(pytorch_files), "PyTorch model file must not be downloaded."
        safetensors_files = list(Path(cache_folder).glob("**/model.safetensors"))
        assert 1 == len(safetensors_files), "Safetensors model file must be downloaded."

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
        assert 1 == len(pytorch_files), "PyTorch model file must be downloaded."
        safetensors_files = list(Path(cache_folder).glob("**/model.safetensors"))
        assert 0 == len(safetensors_files), "Safetensors model file must not be downloaded."

    sentences = ["This is a test sentence", "This is another test sentence"]
    assert torch.equal(
        safetensors_model.encode(sentences, convert_to_tensor=True),
        pytorch_model.encode(sentences, convert_to_tensor=True),
    ), "Ensure that Safetensors and PyTorch loaded models result in identical embeddings"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA must be available to test moving devices effectively.")
def test_to() -> None:
    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu")

    test_device = torch.device("cuda")
    assert model.device.type == "cpu"
    assert test_device.type == "cuda"

    model.to(test_device)
    assert model.device.type == "cuda", "The model device should have updated"

    model.encode("Test sentence")
    assert model.device.type == "cuda", "Encoding shouldn't change the device"

    assert model._target_device == model.device, "Prevent backwards compatibility failure for _target_device"
    model._target_device = "cpu"
    assert model.device.type == "cpu", "Ensure that setting `_target_device` doesn't crash."
