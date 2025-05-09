from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
from transformers import AutoTokenizer

from sentence_transformers.models.InputModule import InputModule

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class IDF(InputModule):
    """
    IDF (Inverse Document Frequency) module for efficient sparse representations.

    This lightweight module applies IDF weights to token representations, emphasizing
    rare terms while downweighting common ones. It creates sparse vectors where non-zero
    values correspond to the IDF weights of tokens in the text.

    Key benefits:
    - Computationally efficient for both encoding and search
    - Compatible with traditional inverted indices
    - Can be used for inference-free query encoding when paired with pre-computed document embeddings

    Args:
        weight: IDF weights for vocabulary tokens
        tokenizer:  PreTrainedTokenizer to convert tokens to IDs
        frozen: If True, weights are fixed. Defaults to False.
    """

    config_keys: list[str] = ["frozen"]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        weight: torch.Tensor | None,
        frozen: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        if weight is not None:
            self.weight = torch.nn.Parameter(weight, requires_grad=not frozen)
        else:
            self.weight = torch.nn.Parameter(torch.ones(len(self.tokenizer.get_vocab())), requires_grad=not frozen)
        self.frozen = frozen
        self.word_embedding_dimension = self.weight.size(0)
        self.max_seq_length = self.tokenizer.model_max_length

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = features["input_ids"]
        batch_size = input_ids.shape[0]

        multi_hot = torch.zeros(
            batch_size, self.word_embedding_dimension, dtype=torch.float32, device=input_ids.device
        )
        batch_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(-1)
        multi_hot[batch_indices, input_ids] = 1

        sentence_embedding = multi_hot * self.weight

        features["sentence_embedding"] = sentence_embedding
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_tokenizer(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)
        self.save_config(output_path)

    @classmethod
    def from_json(cls, json_path: str, tokenizer: PreTrainedTokenizer, **config):
        """
        Create an IDF model from a JSON file containing token to IDF weight mappings.

        Args:
            json_path (str): Path to the JSON file containing token to IDF weight mappings.
            tokenizer (PreTrainedTokenizer): Tokenizer to use for converting tokens to IDs.
            **config: Additional configuration options for the IDF model.

        Returns:
            IDF: An initialized IDF model.
        """
        if not os.path.exists(json_path):
            try:
                from huggingface_hub import hf_hub_download

                json_path = hf_hub_download(repo_id=json_path, filename="idf.json")
            except ValueError:
                raise ValueError(f"IDF JSON file not found at {json_path}. Please provide a valid path.")

        with open(json_path) as fIn:
            idf = json.load(fIn)

        tokens, weights = zip(*idf.items())
        token_ids = tokenizer.convert_tokens_to_ids(list(tokens))
        weights = torch.tensor(weights, dtype=torch.float32)

        max_token_id = max(token_ids) + 1
        weight = torch.zeros(max_token_id, dtype=torch.float32)
        for token_id, w in zip(token_ids, weights):
            weight[token_id] = w

        return cls(weight=weight, tokenizer=tokenizer, **config)

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
        """
        Load the IDF model with its tokenizer.

        Args:
            model_name_or_path (str): Path to the directory containing the saved model.
            subfolder (str): Subfolder within the model directory
            token (bool | str | None): Token for Hugging Face authentication
            cache_folder (str | None): Cache folder for Hugging Face
            revision (str | None): Model revision
            local_files_only (bool): Whether to only load local files
            **kwargs: Additional keyword arguments

        Returns:
            IDF: The loaded IDF model.
        """
        config = cls.load_config(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

        # Check if we have a JSON path in config
        path = config.pop("path", None)
        if path is not None and path.endswith(".json"):
            return cls.from_json(path, tokenizer, **config)

        # Load model weights
        model = cls(weight=None, tokenizer=tokenizer, **config)
        model = cls.load_torch_weights(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
            model=model,
        )
        return model

    def __repr__(self) -> str:
        tokenizer_info = f", tokenizer: {self.tokenizer.__class__.__name__}"
        return f"IDF ({self.get_config_dict()}, dim:{self.word_embedding_dimension}{tokenizer_info})"

    def get_sentence_embedding_dimension(self) -> int:
        return self.word_embedding_dimension

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
        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=padding,
                truncation="longest_first",
                return_tensors="pt",
            )
        )
        return output
