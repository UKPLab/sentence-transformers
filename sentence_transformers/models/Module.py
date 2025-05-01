from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Self

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model

from sentence_transformers.util import load_dir_path, load_file_path


# BaseModule? ModuleBase?
class Module(ABC, torch.nn.Module):
    config_file_name: str = "config.json"
    config_keys: list[str] = []
    save_in_root: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, features: dict[str, torch.Tensor | Any], **kwargs) -> dict[str, torch.Tensor | Any]: ...

    def get_config_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.config_keys}

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        config = cls.load_config(
            model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(**config)

    @classmethod
    def load_config(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        config_filename: str | None = None,
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> dict[str, Any]:
        """
        Load the config file from the model subfolder. The config file is expected to be in JSON format.

        Args:
            model_name_or_path (str): The path to the model subfolder or the name of the model.
        """
        config_path = load_file_path(
            model_name_or_path=model_name_or_path,
            filename=config_filename or cls.config_file_name,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if config_path is None:
            return {}

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        return config

    @staticmethod
    def load_file_path(
        model_name_or_path: str,
        filename: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> str | None:
        return load_file_path(
            model_name_or_path=model_name_or_path,
            filename=filename,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

    @staticmethod
    def load_dir_path(
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> str:
        return load_dir_path(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

    @classmethod
    def load_torch_weights(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        model: torch.nn.Module | None = None,
    ):
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        # 1. Attempt to load a safetensors file from the local or remote directory
        safetensors_path = cls.load_file_path(model_name_or_path, filename="model.safetensors", **hub_kwargs)
        if safetensors_path is not None:
            # Either load the weights into the model or return the weights
            if model is not None:
                load_safetensors_model(model, safetensors_path)
                return model
            else:
                weights = load_safetensors_file(safetensors_path)
                return weights

        # 2. If safetensors file is not found, attempt to load a pytorch model file
        # from the local or remote directory
        pytorch_model_path = cls.load_file_path(model_name_or_path, filename="pytorch_model.bin", **hub_kwargs)
        if pytorch_model_path is None:
            raise ValueError(f"Could not find 'model.safetensors' or 'pytorch_model.bin' in {model_name_or_path}.")

        weights = torch.load(pytorch_model_path, map_location=torch.device("cpu"), weights_only=True)
        if model is not None:
            model.load_state_dict(weights)
            return model
        return weights

    @abstractmethod
    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None: ...

    def save_config(self, output_path: str, filename: str | None = None) -> None:
        config = self.get_config_dict()
        config_output_path = os.path.join(output_path, filename or self.config_file_name)
        with open(config_output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    def save_torch_weights(self, output_path: str, safe_serialization: bool = True) -> None:
        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
