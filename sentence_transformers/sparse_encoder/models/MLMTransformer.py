from __future__ import annotations

import json
import os
from typing import Any

import torch
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from torch import nn
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer


class MLMTransformer(nn.Module):
    """A minimal Transformer model that uses MLM (Masked Language Modeling).

    This model implements only the essential functionality needed for MLM,
    without inheriting from the base Transformer class.

    Args:
        model_name_or_path: Hugging Face models name
        max_seq_length: Truncate any inputs longer than max_seq_length
        model_args: Keyword arguments passed to the Hugging Face Transformers model
        tokenizer_args: Keyword arguments passed to the Hugging Face Transformers tokenizer
        config_args: Keyword arguments passed to the Hugging Face Transformers config
        cache_dir: Cache dir for Hugging Face Transformers to store/load models
        do_lower_case: If true, lowercases the input
        tokenizer_name_or_path: Name or path of the tokenizer
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int | None = None,
        model_args: dict[str, Any] | None = None,
        tokenizer_args: dict[str, Any] | None = None,
        config_args: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        do_lower_case: bool = False,
        tokenizer_name_or_path: str | None = None,
    ) -> None:
        super().__init__()

        # Set default values for optional arguments
        if model_args is None:
            model_args = {}
        if tokenizer_args is None:
            tokenizer_args = {}
        if config_args is None:
            config_args = {}

        # Load config
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir, **config_args)

        # Load tokenizer
        if max_seq_length is not None and "model_max_length" not in tokenizer_args:
            tokenizer_args["model_max_length"] = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            (tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path),
            cache_dir=cache_dir,
            **tokenizer_args,
        )

        # Set max_seq_length
        self.max_seq_length = max_seq_length
        if max_seq_length is None:
            if hasattr(self.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                self.max_seq_length = min(self.config.max_position_embeddings, self.tokenizer.model_max_length)

        # Load MLM model
        self.auto_model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, config=self.config, cache_dir=cache_dir, **model_args
        )

        # Store initialization parameters
        self.model_name_or_path = model_name_or_path
        self.model_args = model_args
        self.tokenizer_args = tokenizer_args
        self.config_args = config_args
        self.cache_dir = cache_dir
        self.do_lower_case = do_lower_case
        self.tokenizer_name_or_path = tokenizer_name_or_path

    def get_config_dict(self):
        """
        Get the configuration dictionary.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {
            "model_name_or_path": self.model_name_or_path,
            "max_seq_length": self.max_seq_length,
            "model_args": self.model_args,
            "tokenizer_args": self.tokenizer_args,
            "config_args": self.config_args,
            "cache_dir": self.cache_dir,
            "do_lower_case": self.do_lower_case,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
        }

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        """
        Save the model to the specified path.

        Args:
            output_path: Path to save the model to
            safe_serialization: Whether to use safetensors for serialization
        """
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)

        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path: str):
        """
        Load the model from the specified path.

        Args:
            input_path: Path to load the model from

        Returns:
            Loaded MLMTransformer model
        """
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        model = MLMTransformer(**config)
        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
        else:
            model.load_state_dict(
                torch.load(
                    os.path.join(input_path, "pytorch_model.bin"),
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
            )
        return model

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            features: Dictionary containing input features

        Returns:
            Dictionary containing token embeddings and MLM logits
        """
        # Get MLM outputs
        mlm_outputs = self.auto_model(**features)

        # Get the MLM head logits (shape: batch_size, seq_length, vocab_size)
        mlm_logits = mlm_outputs.logits

        return {"mlm_logits": mlm_logits}

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of the token embeddings"""
        return self.auto_model.config.hidden_size

    def tokenize(self, texts: list[str], padding: bool = True) -> dict[str, torch.Tensor]:
        """Tokenize the input texts.

        Args:
            texts: List of texts to tokenize
            padding: Whether to pad the sequences

        Returns:
            Dictionary containing tokenized inputs
        """
        # Check if the model is a DistilBERT model
        is_distilbert = "distilbert" in self.auto_model.config.model_type.lower()

        # For DistilBERT models, we need to exclude token_type_ids
        if is_distilbert:
            return self.tokenizer(
                texts,
                padding=padding,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
                return_token_type_ids=False,  # Exclude token_type_ids for DistilBERT
            )
        else:
            return self.tokenizer(
                texts,
                padding=padding,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
