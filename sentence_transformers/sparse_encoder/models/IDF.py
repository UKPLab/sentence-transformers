from __future__ import annotations

import json
import os
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_model as save_safetensors_model
from transformers import AutoTokenizer


class IDF(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        frozen: bool = False,
        tokenizer=None,
        do_lower_case: bool = False,
        max_seq_length: int | None = None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=not frozen)
        self.frozen = frozen
        self.word_embedding_dimension = self.weight.size(0)
        self.tokenizer = tokenizer
        self.config_keys = ["frozen", "max_seq_length", "do_lower_case"]
        self.do_lower_case = do_lower_case

        # Set max_seq_length
        self.max_seq_length = max_seq_length
        if max_seq_length is None:
            if self.tokenizer is not None and hasattr(self.tokenizer, "model_max_length"):
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

    def save(self, output_path, safe_serialization: bool = True) -> None:
        """Save both the IDF model and its tokenizer if available"""
        os.makedirs(output_path, exist_ok=True)

        # Save the model weights
        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

        # Save config with tokenizer info
        config_dict = self.get_config_dict()

        # Save tokenizer if available
        if self.tokenizer is not None:
            config_dict["has_tokenizer"] = True
            self.tokenizer.save_pretrained(os.path.join(output_path, "tokenizer"))
        else:
            config_dict["has_tokenizer"] = False

        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(config_dict, fOut)

    @classmethod
    def from_json(cls, json_path: str, tokenizer, **config):
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
    def load(cls, input_path: str):
        """Load the IDF model with its tokenizer if available"""
        with open(os.path.join(input_path, "config.json")) as fIn:
            config: dict = json.load(fIn)

        # Load tokenizer if it was saved
        tokenizer = None
        if config.pop("has_tokenizer", False) and os.path.exists(os.path.join(input_path, "tokenizer")):
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(input_path, "tokenizer"))

        path = config.pop("path", None)

        if path is not None and path.endswith(".json"):
            if tokenizer is None:
                raise ValueError("Tokenizer is required for loading from JSON")
            return cls.from_json(path, tokenizer, **config)

        # Load model weights
        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            weight = load_safetensors_file(os.path.join(input_path, "model.safetensors"))["weight"]
        else:
            weight = torch.load(os.path.join(input_path, "pytorch_model.bin"))["weight"]

        model = cls(weight=weight, tokenizer=tokenizer, **config)
        return model

    def __repr__(self) -> str:
        tokenizer_info = f", tokenizer: {self.tokenizer.__class__.__name__}" if self.tokenizer else ""
        return f"IDF ({self.get_config_dict()}, dim:{self.word_embedding_dimension}, {tokenizer_info})"

    def get_sentence_embedding_dimension(self) -> int:
        return self.word_embedding_dimension

    def get_config_dict(self) -> dict[str, Any]:
        return {key: self.__dict__[key] for key in self.config_keys}

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


# TODO: Remember to remove this before merge/release
if __name__ == "__main__":
    from sentence_transformers.sparse_encoder import MLMTransformer

    # Example usage
    doc_encoder = MLMTransformer("opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill")

    # Create IDF model with tokenizer
    idf_model = IDF.from_json(
        "runs/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill/idf.json",
        tokenizer=doc_encoder.tokenizer,
        frozen=True,
    )

    # Save model with tokenizer
    idf_model.save("runs/idf_complete")

    # Load model with tokenizer
    loaded_model = IDF.load("runs/idf_complete")
    print(f"Model loaded: {loaded_model}")

    # Use the loaded model's tokenizer
    if loaded_model.tokenizer:
        tokens = loaded_model.tokenizer("Example text to encode")
        print(f"Tokenized: {tokens}")
