"""
Computes embeddings
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from sentence_transformers import SentenceTransformer


def test_encode_token_embeddings(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    """
    Test that encode(output_value='token_embeddings') works
    """
    model = paraphrase_distilroberta_base_v1_model
    sent = [
        "Hello Word, a test sentence",
        "Here comes another sentence",
        "My final sentence",
        "Sentences",
        "Sentence five five five five five five five",
    ]
    emb = model.encode(sent, output_value="token_embeddings", batch_size=2)
    assert len(emb) == len(sent)

    for s, e in zip(sent, emb):
        assert len(model.tokenize([s])["input_ids"][0]) == e.shape[0]


def test_encode_single_sentences(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    model = paraphrase_distilroberta_base_v1_model
    # Single sentence
    emb = model.encode("Hello Word, a test sentence")
    assert emb.shape == (768,)
    assert abs(np.sum(emb) - 7.9811716) < 0.002

    # Single sentence as list
    emb = model.encode(["Hello Word, a test sentence"])
    assert emb.shape == (1, 768)
    assert abs(np.sum(emb) - 7.9811716) < 0.002

    # Sentence list
    emb = model.encode(
        [
            "Hello Word, a test sentence",
            "Here comes another sentence",
            "My final sentence",
        ]
    )
    assert emb.shape == (3, 768)
    assert abs(np.sum(emb) - 22.968266) < 0.007


def test_encode_normalize(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    model = paraphrase_distilroberta_base_v1_model
    emb = model.encode(
        [
            "Hello Word, a test sentence",
            "Here comes another sentence",
            "My final sentence",
        ],
        normalize_embeddings=True,
    )
    assert emb.shape == (3, 768)
    for norm in np.linalg.norm(emb, axis=1):
        assert abs(norm - 1) < 0.001


def test_encode_tuple_sentences(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    model = paraphrase_distilroberta_base_v1_model
    # Input a sentence tuple
    emb = model.encode([("Hello Word, a test sentence", "Second input for model")])
    assert emb.shape == (1, 768)
    assert abs(np.sum(emb) - 9.503508) < 0.002

    # List of sentence tuples
    emb = model.encode(
        [
            ("Hello Word, a test sentence", "Second input for model"),
            ("My second tuple", "With two inputs"),
            ("Final tuple", "final test"),
        ]
    )
    assert emb.shape == (3, 768)
    assert abs(np.sum(emb) - 32.14627) < 0.002


@pytest.mark.parametrize("precision", ("int8", "uint8"))
def test_encode_sentence_embedding_int_precision(
    paraphrase_distilroberta_base_v1_model: SentenceTransformer,
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"]
) -> None:
    model = paraphrase_distilroberta_base_v1_model
    # Single sentence
    emb = model.encode("Hello Word, a test sentence", output_value="sentence_embedding", precision=precision)
    assert emb.shape == (768, )
    assert emb.dtype == np.dtype(precision)

    # Single sentence as list
    emb = model.encode(["Hello Word, a test sentence"], output_value="sentence_embedding", precision=precision)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 768)
    assert emb.dtype == np.dtype(precision)

    # Sentence list
    emb = model.encode(
        [
            "Hello Word, a test sentence",
            "Here comes another sentence",
            "My final sentence",
        ],
        output_value="sentence_embedding",
        precision=precision,
    )
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (3, 768)
    assert emb.dtype == np.dtype(precision)


@pytest.mark.parametrize("precision", ("int8", "uint8"))
def test_encode_token_embeddings_int_precision(
    paraphrase_distilroberta_base_v1_model: SentenceTransformer,
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"]
) -> None:
    model = paraphrase_distilroberta_base_v1_model
    # Single sentence
    emb = model.encode("Hello Word, a test sentence", output_value="token_embeddings", precision=precision)
    assert emb.shape == (8, 768)
    assert emb.dtype == np.dtype(precision)

    # Single sentence as list
    emb = model.encode(["Hello Word, a test sentence"], output_value="token_embeddings", precision=precision)
    assert isinstance(emb, list)
    assert emb[0].shape == (8, 768)
    assert emb[0].dtype == np.dtype(precision)

    # Sentence list
    emb = model.encode(
        [
            "Hello Word, a test sentence",
            "Here comes another sentence",
            "My final sentence",
        ],
        output_value="token_embeddings",
        precision=precision,
    )
    assert isinstance(emb, list)
    assert emb[0].shape == (8, 768)
    assert emb[0].dtype == np.dtype(precision)
    assert emb[1].shape == (6, 768)
    assert emb[1].dtype == np.dtype(precision)
    assert emb[2].shape == (5, 768)
    assert emb[2].dtype == np.dtype(precision)


@pytest.mark.parametrize("precision", ("int8", "uint8"))
def test_encode_output_value_none_int_precision(
    paraphrase_distilroberta_base_v1_model: SentenceTransformer,
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"]
) -> None:
    model = paraphrase_distilroberta_base_v1_model
    # Single sentence
    emb = model.encode("Hello Word, a test sentence", output_value=None, precision=precision)
    assert isinstance(emb, dict)
    assert emb["sentence_embedding"].shape == (768,)
    assert emb["sentence_embedding"].dtype == np.dtype(precision)
    assert emb["token_embeddings"].shape == (8, 768)
    assert emb["token_embeddings"].dtype == np.dtype(precision)

    # Single sentence as list
    emb = model.encode(["Hello Word, a test sentence"], output_value=None, precision=precision)
    assert isinstance(emb, list)
    assert isinstance(emb[0], dict)
    assert emb[0]["sentence_embedding"].shape == (768,)
    assert emb[0]["sentence_embedding"].dtype == np.dtype(precision)
    assert emb[0]["token_embeddings"].shape == (8, 768)
    assert emb[0]["token_embeddings"].dtype == np.dtype(precision)

    # Sentence list
    emb = model.encode(
        [
            "Hello Word, a test sentence",
            "Here comes another sentence",
            "My final sentence",
        ],
        output_value=None,
        precision=precision,
    )
    assert isinstance(emb, list)
    assert all(isinstance(e, dict) for e in emb)

    assert emb[0]["sentence_embedding"].shape == (768,)
    assert emb[0]["sentence_embedding"].dtype == np.dtype(precision)
    assert emb[0]["token_embeddings"].shape == (8, 768)
    assert emb[0]["token_embeddings"].dtype == np.dtype(precision)

    assert emb[1]["sentence_embedding"].shape == (768,)
    assert emb[1]["sentence_embedding"].dtype == np.dtype(precision)
    assert emb[1]["token_embeddings"].shape == (8, 768)
    assert emb[1]["token_embeddings"].dtype == np.dtype(precision)

    assert emb[2]["sentence_embedding"].shape == (768,)
    assert emb[2]["sentence_embedding"].dtype == np.dtype(precision)
    assert emb[2]["token_embeddings"].shape == (8, 768)
    assert emb[2]["token_embeddings"].dtype == np.dtype(precision)