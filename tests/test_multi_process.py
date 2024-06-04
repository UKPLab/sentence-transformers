"""
Computes embeddings
"""

from typing import Optional

import numpy as np
import pytest

from sentence_transformers import SentenceTransformer


@pytest.mark.parametrize("normalize_embeddings", (False, True))
@pytest.mark.parametrize("prompt_name", (None, "retrieval"))
def test_encode_multi_process(
    stsb_bert_tiny_model: SentenceTransformer, normalize_embeddings: bool, prompt_name: Optional[str]
) -> None:
    model = stsb_bert_tiny_model
    model.prompts = {"retrieval": "Represent this sentence for searching relevant passages: "}
    sentences = ["This is sentence {}".format(i) for i in range(40)]

    # Start the multi-process pool on e.g. two CPU devices & compute the embeddings using the pool
    pool = model.start_multi_process_pool(["cpu", "cpu"])
    emb = model.encode_multi_process(
        sentences, pool, chunk_size=10, normalize_embeddings=normalize_embeddings, prompt_name=prompt_name
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
