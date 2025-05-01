from __future__ import annotations

import os
from pathlib import Path
from typing import Self

import torch
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from torch import Tensor, nn

from sentence_transformers.models.Module import Module


class LayerNorm(Module):
    config_keys: list[str] = ["dimension"]

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.norm = nn.LayerNorm(dimension)

    def forward(self, features: dict[str, Tensor]):
        features["sentence_embedding"] = self.norm(features["sentence_embedding"])
        return features

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def save(self, output_path, safe_serialization: bool = True) -> None:
        self.save_config(output_path)

        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

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
        hub_kwargs = {
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = cls.load_config(
            model_name_or_path=model_name_or_path,
            directory=directory,
            **hub_kwargs,
        )
        model = cls(**config)

        safetensors_path = cls.load_file_path(
            model_name_or_path,
            filename=Path(directory, "model.safetensors"),
            **hub_kwargs,
        )
        if safetensors_path is not None:
            load_safetensors_model(model, safetensors_path)
        else:
            pytorch_model_path = cls.load_file_path(
                model_name_or_path,
                filename=Path(directory, "pytorch_model.bin"),
                **hub_kwargs,
            )
            if pytorch_model_path is None:
                raise ValueError(f"Could not find 'model.safetensors' or 'pytorch_model.bin' in {model_name_or_path}.")

            model.load_state_dict(torch.load(pytorch_model_path, map_location=torch.device("cpu"), weights_only=True))
        return model
