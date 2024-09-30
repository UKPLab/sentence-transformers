from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from torch import nn


class CNN(nn.Module):
    """CNN-layer with multiple kernel-sizes over the word embeddings"""

    def __init__(
        self,
        in_word_embedding_dimension: int,
        out_channels: int = 256,
        kernel_sizes: list[int] = [1, 3, 5],
        stride_sizes: list[int] = None,
    ):
        nn.Module.__init__(self)
        self.config_keys = ["in_word_embedding_dimension", "out_channels", "kernel_sizes"]
        self.in_word_embedding_dimension = in_word_embedding_dimension
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.embeddings_dimension = out_channels * len(kernel_sizes)
        self.convs = nn.ModuleList()

        in_channels = in_word_embedding_dimension
        if stride_sizes is None:
            stride_sizes = [1] * len(kernel_sizes)

        for kernel_size, stride in zip(kernel_sizes, stride_sizes):
            padding_size = int((kernel_size - 1) / 2)
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding_size,
            )
            self.convs.append(conv)

    def forward(self, features):
        token_embeddings = features["token_embeddings"]

        token_embeddings = token_embeddings.transpose(1, -1)
        vectors = [conv(token_embeddings) for conv in self.convs]
        out = torch.cat(vectors, 1).transpose(1, -1)

        features.update({"token_embeddings": out})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def tokenize(self, text: str, **kwargs) -> list[int]:
        raise NotImplementedError()

    def save(self, output_path: str, safe_serialization: bool = True):
        with open(os.path.join(output_path, "cnn_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, "cnn_config.json")) as fIn:
            config = json.load(fIn)

        model = CNN(**config)
        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
        else:
            model.load_state_dict(
                torch.load(
                    os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"), weights_only=True
                )
            )
        return model
