from __future__ import annotations

from pathlib import Path

import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.models import SparseStaticEmbedding
from tests.sparse_encoder.utils import sparse_allclose


def test_sparse_static_embedding_padding_ignored(inference_free_splade_bert_tiny_model: SparseEncoder) -> None:
    model = inference_free_splade_bert_tiny_model

    input_texts = ["This is a test input", "This is a considerably longer test input to check padding behavior."]

    # Encode the input texts
    batch_embeddings = model.encode_query(input_texts, save_to_cpu=True)

    single_embeddings = [model.encode_query(text, save_to_cpu=True) for text in input_texts]
    single_embeddings = torch.stack(single_embeddings)

    # Check that the batch embeddings match the single embeddings
    assert sparse_allclose(
        batch_embeddings, single_embeddings, atol=1e-6
    ), "Batch encoding does not match single encoding."


def test_sparse_static_embedding_save_load(
    inference_free_splade_bert_tiny_model: SparseEncoder, tmp_path: Path
) -> None:
    model = inference_free_splade_bert_tiny_model

    # Define test inputs
    test_inputs = ["This is a simple test.", "Another example text for testing."]

    # Get embeddings before saving
    original_embeddings = model.encode_query(test_inputs, save_to_cpu=True)

    # Save the model
    save_path = tmp_path / "test_sparse_static_embedding_model"
    model.save_pretrained(save_path)

    # Load the model
    loaded_model = SparseEncoder(str(save_path))

    # Get embeddings after loading
    loaded_embeddings = loaded_model.encode_query(test_inputs, save_to_cpu=True)

    # Check if embeddings are the same before and after save/load
    assert sparse_allclose(original_embeddings, loaded_embeddings, atol=1e-6), "Embeddings changed after save and load"

    # Check if SparseStaticEmbedding weights are maintained after loading
    assert isinstance(
        loaded_model[0].query_0_SparseStaticEmbedding, SparseStaticEmbedding
    ), "SparseStaticEmbedding component missing after loading"
    assert torch.allclose(
        model[0].query_0_SparseStaticEmbedding.weight, loaded_model[0].query_0_SparseStaticEmbedding.weight
    ), "SparseStaticEmbedding weights changed after save and load"
