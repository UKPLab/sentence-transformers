from __future__ import annotations

import json
import os

import torch
from torch import Tensor, nn


class CSRSparsity(nn.Module):
    def __init__(self, sparsity_threshold: float | None = 0.0, topk: int | None = 0):
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        self.topk = topk

    def forward(self, features: dict[str, Tensor], sparsity_threshold: float | None = None, topk: int = None):
        embeddings = features["sentence_embedding"]

        # Apply sparsity threshold
        threshold = sparsity_threshold if sparsity_threshold is not None else self.sparsity_threshold
        if threshold > 0:
            embeddings = torch.where(
                torch.abs(embeddings) > threshold,
                embeddings,
                torch.zeros_like(embeddings),
            )

        # Apply top-k sparsity
        topk = topk if topk is not None else self.topk
        if topk > 0:
            values, indices = torch.topk(embeddings.abs(), topk, dim=-1)
            embeddings = torch.zeros_like(embeddings)
            embeddings.scatter_(1, indices, values)

        features.update({"sparse_embedding": embeddings})
        return features

    def get_config_dict(self):
        return {
            "sparsity_threshold": self.sparsity_threshold,
            "topk": self.topk,
        }

    def save(self, output_path, safe_serialization: bool = True) -> None:
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)
        return CSRSparsity(**config)

    def __repr__(self):
        return f"CSRSparsity({self.get_config_dict()})"
