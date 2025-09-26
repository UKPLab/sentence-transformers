from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense, StaticEmbedding


def test_dense_load_and_save_in_other_precisions(static_embedding_model: StaticEmbedding, tmp_path: Path) -> None:
    base_model = SentenceTransformer(modules=[static_embedding_model, Dense(768, 256, activation_function=nn.Tanh())])
    test_text = ["This is a test"]
    base_embedding = base_model.encode(test_text, convert_to_tensor=True)

    base_path = str(tmp_path / "model")
    base_model.save_pretrained(base_path)

    loaded_model = SentenceTransformer(base_path)
    assert loaded_model[1].linear.weight.dtype == torch.float32
    assert loaded_model[1].linear.weight.shape == (256, 768)
    loaded_embedding = loaded_model.encode(test_text, convert_to_tensor=True)
    assert torch.allclose(base_embedding, loaded_embedding, atol=1e-6)

    fp64_model = deepcopy(base_model).to(torch.float64)
    fp64_path = str(tmp_path / "fp64")
    fp64_model.save_pretrained(fp64_path)
    loaded_fp64_model = SentenceTransformer(fp64_path)
    assert loaded_fp64_model[1].linear.weight.dtype == torch.float64
    assert loaded_fp64_model[1].linear.weight.shape == (256, 768)
    loaded_fp64_embedding = loaded_fp64_model.encode(test_text, convert_to_tensor=True)
    assert torch.allclose(base_embedding, loaded_fp64_embedding.to(torch.float32), atol=1e-6)

    fp16_model = deepcopy(base_model).to(torch.float16)
    fp16_path = str(tmp_path / "fp16")
    fp16_model.save_pretrained(fp16_path)
    loaded_fp16_model = SentenceTransformer(fp16_path)
    assert loaded_fp16_model[1].linear.weight.dtype == torch.float16
    assert loaded_fp16_model[1].linear.weight.shape == (256, 768)
    loaded_fp16_embedding = loaded_fp16_model.encode(test_text, convert_to_tensor=True)
    assert torch.allclose(base_embedding, loaded_fp16_embedding.to(torch.float32), atol=1e-3)

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        bf16_model = deepcopy(base_model).to(torch.bfloat16)
        bf16_path = str(tmp_path / "bf16")
        bf16_model.save_pretrained(bf16_path)
        loaded_bf16_model = SentenceTransformer(bf16_path)
        assert loaded_bf16_model[1].linear.weight.dtype == torch.bfloat16
        assert loaded_bf16_model[1].linear.weight.shape == (256, 768)
        loaded_bf16_embedding = loaded_bf16_model.encode(test_text, convert_to_tensor=True)
        assert torch.allclose(base_embedding, loaded_bf16_embedding.to(torch.float32), atol=1e-2)
