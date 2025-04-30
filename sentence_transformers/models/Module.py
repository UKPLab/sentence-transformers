from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import torch

from sentence_transformers.util import load_dir_path, load_file_path


# BaseModule? ModuleBase?
class Module(ABC, torch.nn.Module):
    config_file_name: str = "config.json"
    save_in_root: bool = False
    config_keys: list[str] = []

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
        return cls(model_name_or_path, **config)

    @classmethod
    def load_config(
        cls,
        model_name_or_path: str,
        directory: str = "",
        config_filename: str | None = None,
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> dict[str, Any]:
        """
        Load the config file from the model directory. The config file is expected to be in JSON format.

        Args:
            model_name_or_path (str): The path to the model directory or the name of the model.
        """
        config_filename = config_filename or cls.config_file_name
        config_file_path = Path(directory, config_filename)
        config_path = load_file_path(
            model_name_or_path=model_name_or_path,
            filename=config_file_path,
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
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> str | None:
        return load_file_path(
            model_name_or_path=model_name_or_path,
            filename=filename,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

    @staticmethod
    def load_dir_path(
        model_name_or_path: str,
        directory: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> str:
        return load_dir_path(
            model_name_or_path=model_name_or_path,
            directory=directory,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

    @abstractmethod
    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None: ...

    def save_config(self, output_path: str, filename: str | None = None) -> None:
        config = self.get_config_dict()
        config_output_path = os.path.join(output_path, filename or self.config_file_name)
        with open(config_output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
