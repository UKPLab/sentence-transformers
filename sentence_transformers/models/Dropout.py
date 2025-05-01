from __future__ import annotations

from typing import Self

from torch import Tensor, nn

from sentence_transformers.SentenceTransformer import Module


class Dropout(Module):
    """Dropout layer.

    Args:
        dropout: Sets a dropout value for dense layer.
    """

    config_keys: list[str] = ["dropout"]

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, features: dict[str, Tensor]):
        features.update({"sentence_embedding": self.dropout_layer(features["sentence_embedding"])})
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        directory: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        config = cls.load_config(
            model_name_or_path,
            directory=directory,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(**config)
