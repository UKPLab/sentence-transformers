from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from torch import nn


class LSTM(nn.Module):
    """Bidirectional LSTM running over word embeddings."""

    def __init__(
        self,
        word_embedding_dimension: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0,
        bidirectional: bool = True,
    ):
        nn.Module.__init__(self)
        self.config_keys = ["word_embedding_dimension", "hidden_dim", "num_layers", "dropout", "bidirectional"]
        self.word_embedding_dimension = word_embedding_dimension
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embeddings_dimension = hidden_dim
        if self.bidirectional:
            self.embeddings_dimension *= 2

        self.encoder = nn.LSTM(
            word_embedding_dimension,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, features):
        token_embeddings = features["token_embeddings"]
        sentence_lengths = torch.clamp(features["sentence_lengths"], min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            token_embeddings, sentence_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed = self.encoder(packed)
        unpack = nn.utils.rnn.pad_packed_sequence(packed[0], batch_first=True)[0]
        features.update({"token_embeddings": unpack})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def tokenize(self, text: str, **kwargs) -> list[int]:
        raise NotImplementedError()

    def save(self, output_path: str, safe_serialization: bool = True):
        with open(os.path.join(output_path, "lstm_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        device = next(self.parameters()).device
        if safe_serialization:
            save_safetensors_model(self.cpu(), os.path.join(output_path, "model.safetensors"))
            self.to(device)
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, "lstm_config.json")) as fIn:
            config = json.load(fIn)

        model = LSTM(**config)
        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
        else:
            model.load_state_dict(
                torch.load(
                    os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"), weights_only=True
                )
            )
        return model
