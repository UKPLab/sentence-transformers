from __future__ import annotations

import numpy as np
import pytest
import torch

from sentence_transformers import SparseEncoder

from .utils import sparse_allclose


@pytest.mark.skip(
    "This test fails if optimum.intel.openvino is imported, because openvinotoolkit/nncf "
    "patches torch._C._nn.gelu in a way that breaks pickling."
)
@pytest.mark.parametrize("normalize_embeddings", (False, True))
@pytest.mark.parametrize("prompt_name", (None, "retrieval"))
def test_encode_multi_process(
    splade_bert_tiny_model: SparseEncoder, normalize_embeddings: bool, prompt_name: str | None
) -> None:
    model = splade_bert_tiny_model
    model.prompts = {"retrieval": "Represent this sentence for searching relevant passages: "}
    sentences = [f"This is sentence {i}" for i in range(40)]

    # Start the multi-process pool on e.g. two CPU devices & compute the embeddings using the pool
    pool = model.start_multi_process_pool(["cpu", "cpu"])
    emb = model.encode(
        sentences, normalize_embeddings=normalize_embeddings, prompt_name=prompt_name, pool=pool, chunk_size=10
    )
    model.stop_multi_process_pool(pool)
    assert emb.shape == (len(sentences), 128)

    # Make sure the embeddings aren't just all 0
    assert emb.sum() != 0.0

    # Compare against normal embeddings
    emb_normal = model.encode(sentences, normalize_embeddings=normalize_embeddings, prompt_name=prompt_name)
    diff = np.max(np.abs(emb - emb_normal))
    assert diff < 0.001

    # Ensure that after normalizing, the means are all almost 0, and otherwise not
    assert np.all(np.abs(emb.mean(1)) < 0.01) == normalize_embeddings


@pytest.mark.slow
def test_multi_process_encode_same_as_standard_encode(splade_bert_tiny_model_reused: SparseEncoder):
    model = splade_bert_tiny_model_reused
    # Test that multi-process encoding gives the same result as standard encoding
    texts = ["First sentence.", "Second sentence.", "Third sentence."] * 5

    # Standard encode
    embeddings_standard = model.encode(texts).cpu()

    # Multi-process encode with device=["cpu"] * 2
    embeddings_multi = model.encode(texts, device=["cpu"] * 2)

    # Should produce the same embeddings
    assert sparse_allclose(embeddings_standard, embeddings_multi, atol=1e-5)


@pytest.mark.slow
def test_multi_process_pool(splade_bert_tiny_model_reused: SparseEncoder):
    # Test the start_multi_process_pool and stop_multi_process_pool functions
    model = splade_bert_tiny_model_reused
    texts = ["First sentence.", "Second sentence.", "Third sentence."] * 5

    # Standard encode
    embeddings_standard = model.encode(texts).cpu()

    pool = model.start_multi_process_pool(["cpu"] * 2)
    try:
        # Encode using the pool
        embeddings_multi = model.encode(texts, pool=pool)

    finally:
        model.stop_multi_process_pool(pool)

    # Should be numpy array with correct shape and the same embeddings
    assert isinstance(embeddings_multi, torch.Tensor)
    assert embeddings_multi.is_sparse
    assert embeddings_multi.shape == (len(texts), model.get_sentence_embedding_dimension())
    assert sparse_allclose(embeddings_standard, embeddings_multi, atol=1e-5)


@pytest.mark.slow
def test_multi_process_with_args(splade_bert_tiny_model_reused: SparseEncoder):
    # Test multi-process encoding with various arguments
    model = splade_bert_tiny_model_reused
    texts = ["First sentence.", "Second sentence."]

    # Create a pool
    pool = model.start_multi_process_pool(["cpu"] * 2)

    try:
        # Test with normalize_embeddings and convert_to_tensor
        embeddings_maxed = model.encode(texts, pool=pool, max_active_dims=16)

        # Should be a tensor with normalized vectors
        assert isinstance(embeddings_maxed, torch.Tensor)
        assert embeddings_maxed.is_sparse
        assert torch.equal(embeddings_maxed.to_dense().nonzero(as_tuple=True)[0], torch.tensor([0] * 16 + [1] * 16))

        # Test with precision options
        embeddings_non_sparse = model.encode(texts, pool=pool, convert_to_sparse_tensor=False)
        assert isinstance(embeddings_maxed, torch.Tensor)
        assert not embeddings_non_sparse.is_sparse
        assert embeddings_non_sparse.shape == (len(texts), model.get_sentence_embedding_dimension())
    finally:
        model.stop_multi_process_pool(pool)


@pytest.mark.slow
def test_multi_process_chunk_size(splade_bert_tiny_model_reused: SparseEncoder):
    # Test explicit chunk_size parameter
    model = splade_bert_tiny_model_reused
    texts = ["First sentence.", "Second sentence.", "Third sentence."] * 10

    # Test with explicit chunk size
    embeddings = model.encode(texts, device=["cpu"] * 2, chunk_size=5)

    # Should produce correct embeddings
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.is_sparse
    assert embeddings.shape == (len(texts), model.get_sentence_embedding_dimension())


@pytest.mark.slow
def test_multi_process_with_prompt(splade_bert_tiny_model: SparseEncoder):
    # Test multi-process encoding with prompts
    model = splade_bert_tiny_model
    model.prompts = {"retrieval": "Represent this sentence for searching relevant passages: "}
    texts = ["First sentence.", "Second sentence."] * 5

    standard_embeddings = model.encode(texts, prompt_name="retrieval").cpu()

    assert isinstance(standard_embeddings, torch.Tensor)
    assert standard_embeddings.is_sparse
    assert standard_embeddings.shape == (len(texts), model.get_sentence_embedding_dimension())

    # Create a pool
    pool = model.start_multi_process_pool(["cpu"] * 2)

    try:
        # Encode with prompt
        multi_embeddings = model.encode(texts, pool=pool, prompt_name="retrieval")
    finally:
        model.stop_multi_process_pool(pool)

    assert isinstance(multi_embeddings, torch.Tensor)
    assert multi_embeddings.is_sparse
    assert multi_embeddings.shape == (len(texts), model.get_sentence_embedding_dimension())

    assert sparse_allclose(standard_embeddings, multi_embeddings, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_sparse_tensor", [True, False])
def test_multi_process_with_empty_texts(
    splade_bert_tiny_model_reused: SparseEncoder,
    convert_to_tensor: bool,
    convert_to_sparse_tensor: bool,
):
    # Test encoding with empty texts
    model = splade_bert_tiny_model_reused
    texts = []

    # Encode with empty texts
    standard_embeddings = model.encode(
        texts, convert_to_tensor=convert_to_tensor, convert_to_sparse_tensor=convert_to_sparse_tensor
    )
    multi_embeddings = model.encode(
        texts,
        device=["cpu"] * 2,
        convert_to_tensor=convert_to_tensor,
        convert_to_sparse_tensor=convert_to_sparse_tensor,
    )

    # Should return empty arrays, identical types as without multi-processing
    assert type(standard_embeddings) is type(multi_embeddings)
    assert len(standard_embeddings) == 0
    assert len(multi_embeddings) == 0


@pytest.mark.slow
@pytest.mark.parametrize("convert_to_tensor", [True, False])
@pytest.mark.parametrize("convert_to_sparse_tensor", [True, False])
def test_multi_process_with_single_string(
    splade_bert_tiny_model_reused: SparseEncoder,
    convert_to_tensor: bool,
    convert_to_sparse_tensor: bool,
):
    # Test encoding with a single text
    model = splade_bert_tiny_model_reused
    texts = "This is a single sentence."

    # Encode with single text
    standard_embeddings = model.encode(
        texts, convert_to_tensor=convert_to_tensor, convert_to_sparse_tensor=convert_to_sparse_tensor
    )
    multi_embeddings = model.encode(
        texts,
        device=["cpu"] * 2,
        convert_to_tensor=convert_to_tensor,
        convert_to_sparse_tensor=convert_to_sparse_tensor,
    )

    # Assert that the embeddings are the same type and shape
    assert type(standard_embeddings) is type(multi_embeddings)
    if isinstance(standard_embeddings, (np.ndarray, torch.Tensor)):
        assert standard_embeddings.shape == multi_embeddings.shape
    else:
        assert len(standard_embeddings) == len(multi_embeddings)
        # Check that dictionary items are the same
        if isinstance(standard_embeddings, dict):
            assert standard_embeddings.keys() == multi_embeddings.keys()
            for key in standard_embeddings:
                if isinstance(standard_embeddings[key], torch.Tensor):
                    assert torch.allclose(standard_embeddings[key].cpu(), multi_embeddings[key], atol=1e-5)
                elif isinstance(standard_embeddings[key], np.ndarray):
                    assert np.allclose(standard_embeddings[key], multi_embeddings[key], atol=1e-5)
                else:
                    assert standard_embeddings[key] == multi_embeddings[key]
        elif isinstance(standard_embeddings, list) and len(standard_embeddings) > 0:
            for std_item, multi_item in zip(standard_embeddings, multi_embeddings):
                assert set(std_item.keys()) == set(multi_item.keys())
                for key in std_item:
                    if isinstance(std_item[key], torch.Tensor):
                        assert torch.allclose(std_item[key].cpu(), multi_item[key], atol=1e-5)
                    elif isinstance(std_item[key], np.ndarray):
                        assert np.allclose(std_item[key], multi_item[key], atol=1e-5)
                    else:
                        assert std_item[key] == multi_item[key]


@pytest.mark.slow
def test_multi_process_more_workers_than_texts(splade_bert_tiny_model_reused: SparseEncoder):
    # Test with more workers than texts
    model = splade_bert_tiny_model_reused
    texts = ["First sentence.", "Second sentence."]

    embeddings = model.encode(texts, device=["cpu"] * 3)

    # Should be numpy array with correct shape
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (len(texts), model.get_sentence_embedding_dimension())


@pytest.mark.slow
def test_multi_process_with_large_chunk_size(splade_bert_tiny_model_reused: SparseEncoder):
    # Test with a large chunk size
    model = splade_bert_tiny_model_reused
    texts = ["First sentence.", "Second sentence."] * 10  # 20 sentences

    # Use a large chunk size
    embeddings = model.encode(texts, device=["cpu"] * 2, chunk_size=30)

    # Should produce correct embeddings
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (len(texts), model.get_sentence_embedding_dimension())
