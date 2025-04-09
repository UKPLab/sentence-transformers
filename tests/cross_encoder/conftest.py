from __future__ import annotations

import pytest

from sentence_transformers import CrossEncoder


@pytest.fixture()
def distilroberta_base_ce_model() -> CrossEncoder:
    return CrossEncoder("distilroberta-base", num_labels=1)


@pytest.fixture()
def reranker_bert_tiny_model() -> CrossEncoder:
    return CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce")


@pytest.fixture(scope="session")
def reranker_bert_tiny_model_reused() -> CrossEncoder:
    return CrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce")
