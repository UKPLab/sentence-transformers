from __future__ import annotations

import numpy as np
import torch

from sentence_transformers.util.tensor import normalize_embeddings


def test_normalize_embeddings() -> None:
    """Tests the correct computation of util.normalize_embeddings"""
    embedding_size = 100
    a = torch.tensor(np.random.randn(50, embedding_size))
    a_norm = normalize_embeddings(a)

    for embedding in a_norm:
        assert len(embedding) == embedding_size
        emb_norm = torch.norm(embedding)
        assert abs(emb_norm.item() - 1) < 0.0001
