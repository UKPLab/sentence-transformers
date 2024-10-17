from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from sentence_transformers.models.StaticEmbedding import StaticEmbedding

try:
    import model2vec
except ImportError:
    model2vec = None

skip_if_no_model2vec = pytest.mark.skipif(model2vec is None, reason="The model2vec library is not installed.")


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def embedding_weights():
    return np.random.rand(30522, 768)


@pytest.fixture
def static_embedding(tokenizer: Tokenizer, embedding_weights) -> StaticEmbedding:
    return StaticEmbedding(tokenizer, embedding_weights=embedding_weights)


def test_initialization_with_embedding_weights(tokenizer: Tokenizer, embedding_weights) -> None:
    model = StaticEmbedding(tokenizer, embedding_weights=embedding_weights)
    assert model.embedding.weight.shape == (30522, 768)


def test_initialization_with_embedding_dim(tokenizer: Tokenizer) -> None:
    model = StaticEmbedding(tokenizer, embedding_dim=768)
    assert model.embedding.weight.shape == (30522, 768)


def test_tokenize(static_embedding: StaticEmbedding) -> None:
    texts = ["Hello world!", "How are you?"]
    tokens = static_embedding.tokenize(texts)
    assert "input_ids" in tokens
    assert "offsets" in tokens


def test_forward(static_embedding: StaticEmbedding) -> None:
    texts = ["Hello world!", "How are you?"]
    tokens = static_embedding.tokenize(texts)
    output = static_embedding(tokens)
    assert "sentence_embedding" in output


def test_save_and_load(tmp_path: Path, static_embedding: StaticEmbedding) -> None:
    save_dir = tmp_path / "model"
    save_dir.mkdir()
    static_embedding.save(str(save_dir))

    loaded_model = StaticEmbedding.load(str(save_dir))
    assert loaded_model.embedding.weight.shape == static_embedding.embedding.weight.shape


@skip_if_no_model2vec()
def test_from_distillation() -> None:
    model = StaticEmbedding.from_distillation("sentence-transformers-testing/stsb-bert-tiny-safetensors", pca_dims=32)
    assert model.embedding.weight.shape == (29528, 32)


@skip_if_no_model2vec()
def test_from_model2vec() -> None:
    model = StaticEmbedding.from_model2vec("minishlab/M2V_base_output")
    assert model.embedding.weight.shape == (29528, 256)
