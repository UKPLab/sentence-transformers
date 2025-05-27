from __future__ import annotations

import os

import numpy as np
import pytest
from tokenizers import Tokenizer

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, StaticEmbedding, Transformer
from sentence_transformers.util import is_datasets_available
from tests.utils import SafeTemporaryDirectory

if is_datasets_available():
    from datasets import DatasetDict, load_dataset


@pytest.fixture()
def stsb_bert_tiny_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")


@pytest.fixture(scope="session")
def stsb_bert_tiny_model_reused() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")


@pytest.fixture()
def stsb_bert_tiny_model_onnx() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-onnx")


@pytest.fixture()
def stsb_bert_tiny_model_openvino() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-openvino")


@pytest.fixture()
def paraphrase_distilroberta_base_v1_model() -> SentenceTransformer:
    return SentenceTransformer("paraphrase-distilroberta-base-v1")


@pytest.fixture()
def clip_vit_b_32_model() -> SentenceTransformer:
    return SentenceTransformer("clip-ViT-B-32")


@pytest.fixture(scope="session")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def embedding_weights():
    return np.random.rand(30522, 768)


@pytest.fixture
def static_embedding_model(tokenizer: Tokenizer, embedding_weights) -> StaticEmbedding:
    return StaticEmbedding(tokenizer, embedding_weights=embedding_weights)


@pytest.fixture()
def distilbert_base_uncased_model() -> SentenceTransformer:
    word_embedding_model = Transformer("distilbert-base-uncased")
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


@pytest.fixture(scope="session")
def stsb_dataset_dict() -> DatasetDict:
    return load_dataset("sentence-transformers/stsb")


@pytest.fixture()
def cache_dir():
    """
    In the CI environment, we use a temporary directory as `cache_dir`
    to avoid keeping the downloaded models on disk after the test.
    """
    if os.environ.get("CI", None):
        # Note: `ignore_cleanup_errors=True` is used to avoid NotADirectoryError in Windows on GitHub Actions.
        # See https://github.com/python/cpython/issues/107408, https://www.scivision.dev/python-tempfile-permission-error-windows/
        with SafeTemporaryDirectory() as tmp_dir:
            yield tmp_dir
    else:
        yield None
