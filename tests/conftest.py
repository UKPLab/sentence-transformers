import os
import platform
import tempfile
import pytest

from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.models import Transformer, Pooling


@pytest.fixture()
def stsb_bert_tiny_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")


@pytest.fixture(scope="session")
def stsb_bert_tiny_model_reused() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")


@pytest.fixture()
def paraphrase_distilroberta_base_v1_model() -> SentenceTransformer:
    return SentenceTransformer("paraphrase-distilroberta-base-v1")


@pytest.fixture()
def distilroberta_base_ce_model() -> CrossEncoder:
    return CrossEncoder("distilroberta-base", num_labels=1)


@pytest.fixture()
def clip_vit_b_32_model() -> SentenceTransformer:
    return SentenceTransformer("clip-ViT-B-32")


@pytest.fixture()
def distilbert_base_uncased_model() -> SentenceTransformer:
    word_embedding_model = Transformer("distilbert-base-uncased")
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


@pytest.fixture()
def cache_dir():
    """
    In the CI environment, we use a temporary directory as `cache_dir`
    to avoid keeping the downloaded models on disk after the test.

    This is only required for Ubuntu, as we otherwise have disk space issues there.
    """
    if os.environ.get("CI", None) and platform.system() == "Linux":
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    else:
        yield None
