from __future__ import annotations

from copy import deepcopy

import pytest

from sentence_transformers import SparseEncoder


@pytest.fixture(scope="session")
def _splade_bert_tiny_model() -> SparseEncoder:
    return SparseEncoder("sparse-encoder-testing/splade-bert-tiny-nq")


@pytest.fixture()
def splade_bert_tiny_model(_splade_bert_tiny_model: SparseEncoder) -> SparseEncoder:
    return deepcopy(_splade_bert_tiny_model)


@pytest.fixture()
def inference_free_splade_bert_tiny_model() -> SparseEncoder:
    return SparseEncoder("sparse-encoder-testing/inference-free-splade-bert-tiny-nq")


@pytest.fixture(scope="session")
def inference_free_splade_bert_tiny_model_reused() -> SparseEncoder:
    return SparseEncoder("sparse-encoder-testing/inference-free-splade-bert-tiny-nq")


@pytest.fixture()
def csr_bert_tiny_model() -> SparseEncoder:
    return SparseEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")


@pytest.fixture(scope="session")
def csr_bert_tiny_model_reused() -> SparseEncoder:
    return SparseEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
