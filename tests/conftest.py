import pytest

from sentence_transformers import SentenceTransformer


@pytest.fixture()
def model() -> SentenceTransformer:
    return SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    )
