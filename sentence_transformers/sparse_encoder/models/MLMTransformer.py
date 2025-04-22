from __future__ import annotations

import json
import os
from typing import Any

import torch
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

    save_in_root: bool = True

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
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.config_keys = ["max_seq_length", "do_lower_case"]
        self.do_lower_case = do_lower_case
        self.backend = backend

        if model_args is None:
            model_args = {}
        if tokenizer_args is None:
            tokenizer_args = {}
        if config_args is None:
            config_args = {}

        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir, **config_args)

        self.auto_model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, config=self.config, cache_dir=cache_dir, **model_args
        )

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

    def get_config_dict(self) -> dict[str, Any]:
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        with open(os.path.join(output_path, "sentence_bert_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        self.auto_model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(output_path)

    @classmethod
    def load(cls, input_path: str) -> MLMTransformer:
        """
        Load the model from the specified path.

        Args:
            input_path: Path to load the model from

        Returns:
            Loaded MLMTransformer model
        """
        with open(os.path.join(input_path, "sentence_bert_config.json")) as fIn:
            config = json.load(fIn)
        print(config)
        return cls(model_name_or_path=input_path, **config)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            features: Dictionary containing input features

        Returns:
            Dictionary containing token embeddings
        """
        # Get MLM outputs
        mlm_outputs = self.auto_model(**features)

        # Get the MLM head logits (shape: batch_size, seq_length, vocab_size)
        mlm_logits = mlm_outputs.logits

        features["token_embeddings"] = mlm_logits
        return features

    def tokenize(
        self, texts: list[str] | list[dict] | list[tuple[str, str]], padding: str | bool = True
    ) -> dict[str, torch.Tensor]:
        """Tokenizes a text and maps tokens to token-ids"""
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=padding,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length,
            )
        )
        return output

    def get_sentence_embedding_dimension(self) -> int:
        return self.auto_model.vocab_projector.out_features

    def __repr__(self) -> str:
        return f"MLMTransformer({self.get_config_dict()}) with MLMTransformer model: {self.auto_model.__class__.__name__} "
