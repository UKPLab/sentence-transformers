import pytest

from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.models import Transformer, Pooling


@pytest.fixture()
def stsb_bert_tiny_model() -> SentenceTransformer:
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
