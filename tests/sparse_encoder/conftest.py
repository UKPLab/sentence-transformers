from __future__ import annotations

import pytest

from sentence_transformers import SparseEncoder


@pytest.fixture()
def splade_bert_tiny_model() -> SparseEncoder:
    return SparseEncoder("sparse-encoder-testing/splade-bert-tiny-nq")


@pytest.fixture()
def csr_bert_tiny_model() -> SparseEncoder:
    return SparseEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
