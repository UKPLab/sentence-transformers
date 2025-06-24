from __future__ import annotations

import torch

from sentence_transformers import SparseEncoder
from tests.sparse_encoder.utils import sparse_allclose


def test_idf_padding_ignored(inference_free_splade_bert_tiny_model: SparseEncoder):
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
