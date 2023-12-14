
from sentence_transformers import SentenceTransformer
import pytest


@pytest.fixture()
def model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
