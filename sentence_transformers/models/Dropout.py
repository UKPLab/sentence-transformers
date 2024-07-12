from __future__ import annotations

import json
import os

from torch import Tensor, nn


class Dropout(nn.Module):
    """Dropout layer.

    Args:
        dropout: Sets a dropout value for dense layer.
    """

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, features: dict[str, Tensor]):
        features.update({"sentence_embedding": self.dropout_layer(features["sentence_embedding"])})
        return features

    def save(self, output_path):
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump({"dropout": self.dropout}, fOut)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        model = Dropout(**config)
        return model
