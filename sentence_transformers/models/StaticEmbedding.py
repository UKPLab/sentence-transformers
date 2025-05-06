from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from tokenizers import Tokenizer
from torch import nn
from transformers import PreTrainedTokenizerFast

from sentence_transformers.util import get_device_name


class StaticEmbedding(nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer | PreTrainedTokenizerFast,
        embedding_weights: np.ndarray | torch.Tensor | None = None,
        embedding_dim: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initializes the StaticEmbedding model given a tokenizer. The model is a simple embedding bag model that
        takes the mean of trained per-token embeddings to compute text embeddings.

        Args:
            tokenizer (Tokenizer | PreTrainedTokenizerFast): The tokenizer to be used. Must be a fast tokenizer
                from ``transformers`` or ``tokenizers``.
            embedding_weights (np.ndarray | torch.Tensor | None, optional): Pre-trained embedding weights.
                Defaults to None.
            embedding_dim (int | None, optional): Dimension of the embeddings. Required if embedding_weights
                is not provided. Defaults to None.

        Example::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.models import StaticEmbedding
            from tokenizers import Tokenizer

            # Pre-distilled embeddings:
            static_embedding = StaticEmbedding.from_model2vec("minishlab/potion-base-8M")
            # or distill your own embeddings:
            static_embedding = StaticEmbedding.from_distillation("BAAI/bge-base-en-v1.5", device="cuda")
            # or start with randomized embeddings:
            tokenizer = Tokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
            static_embedding = StaticEmbedding(tokenizer, embedding_dim=512)

            model = SentenceTransformer(modules=[static_embedding])

            embeddings = model.encode(["What are Pandas?", "The giant panda, also known as the panda bear or simply the panda, is a bear native to south central China."])
            similarity = model.similarity(embeddings[0], embeddings[1])
            # tensor([[0.8093]]) (If you use potion-base-8M)
            # tensor([[0.6234]]) (If you use the distillation method)
            # tensor([[-0.0693]]) (For example, if you use randomized embeddings)

        Raises:
            ValueError: If the tokenizer is not a fast tokenizer.
            ValueError: If neither `embedding_weights` nor `embedding_dim` is provided.
        """
        super().__init__()

        if isinstance(tokenizer, PreTrainedTokenizerFast):
            tokenizer = tokenizer._tokenizer
        elif not isinstance(tokenizer, Tokenizer):
            raise ValueError(
                "The tokenizer must be fast (i.e. Rust-backed) to use this class. "
                "Use Tokenizer.from_pretrained() from `tokenizers` to load a fast tokenizer."
            )

        if embedding_weights is not None:
            if isinstance(embedding_weights, np.ndarray):
                embedding_weights = torch.from_numpy(embedding_weights)

            self.embedding = nn.EmbeddingBag.from_pretrained(embedding_weights, freeze=False)
        elif embedding_dim is not None:
            self.embedding = nn.EmbeddingBag(tokenizer.get_vocab_size(), embedding_dim)
        else:
            raise ValueError("Either `embedding_weights` or `embedding_dim` must be provided.")

        self.num_embeddings = self.embedding.num_embeddings
        self.embedding_dim = self.embedding.embedding_dim

        self.tokenizer: Tokenizer = tokenizer
        self.tokenizer.no_padding()

        # For the model card
        self.base_model = kwargs.get("base_model", None)

    def tokenize(self, texts: list[str], **kwargs) -> dict[str, torch.Tensor]:
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        encodings_ids = [encoding.ids for encoding in encodings]

        offsets = torch.from_numpy(np.cumsum([0] + [len(token_ids) for token_ids in encodings_ids[:-1]]))
        input_ids = torch.tensor([token_id for token_ids in encodings_ids for token_id in token_ids], dtype=torch.long)
        return {"input_ids": input_ids, "offsets": offsets}

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        features["sentence_embedding"] = self.embedding(features["input_ids"], features["offsets"])
        return features

    def get_config_dict(self) -> dict[str, float]:
        return {}

    @property
    def max_seq_length(self) -> int:
        return math.inf

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dim

    def save(self, save_dir: str, safe_serialization: bool = True, **kwargs) -> None:
        if safe_serialization:
            save_safetensors_file(self.state_dict(), os.path.join(save_dir, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        self.tokenizer.save(str(Path(save_dir) / "tokenizer.json"))

    def load(load_dir: str, **kwargs) -> StaticEmbedding:
        tokenizer = Tokenizer.from_file(str(Path(load_dir) / "tokenizer.json"))
        if os.path.exists(os.path.join(load_dir, "model.safetensors")):
            weights = load_safetensors_file(os.path.join(load_dir, "model.safetensors"))
        else:
            weights = torch.load(
                os.path.join(load_dir, "pytorch_model.bin"), map_location=torch.device("cpu"), weights_only=True
            )
        try:
            weights = weights["embedding.weight"]
        except KeyError:
            # For compatibility with model2vec models, which are saved with just an "embeddings" key
            weights = weights["embeddings"]
        return StaticEmbedding(tokenizer, embedding_weights=weights)

    @classmethod
    def from_distillation(
        cls,
        model_name: str,
        vocabulary: list[str] | None = None,
        device: str | None = None,
        pca_dims: int | None = 256,
        apply_zipf: bool = True,
        use_subword: bool = True,
    ) -> StaticEmbedding:
        """
        Creates a StaticEmbedding instance from a distillation process using the `model2vec` package.

        Args:
            model_name (str): The name of the model to distill.
            vocabulary (list[str] | None, optional): A list of vocabulary words to use. Defaults to None.
            device (str): The device to run the distillation on (e.g., 'cpu', 'cuda'). If not specified,
                the strongest device is automatically detected. Defaults to None.
            pca_dims (int | None, optional): The number of dimensions for PCA reduction. Defaults to 256.
            apply_zipf (bool): Whether to apply Zipf's law during distillation. Defaults to True.
            use_subword (bool): Whether to use subword tokenization. Defaults to True.

        Returns:
            StaticEmbedding: An instance of StaticEmbedding initialized with the distilled model's
                tokenizer and embedding weights.

        Raises:
            ImportError: If the `model2vec` package is not installed.
        """

        try:
            from model2vec.distill import distill
        except ImportError:
            raise ImportError(
                "To use this method, please install the `model2vec` package: `pip install model2vec[distill]`"
            )

        device = get_device_name()
        static_model = distill(
            model_name,
            vocabulary=vocabulary,
            device=device,
            pca_dims=pca_dims,
            apply_zipf=apply_zipf,
            use_subword=use_subword,
        )
        if isinstance(static_model.embedding, np.ndarray):
            embedding_weights = torch.from_numpy(static_model.embedding)
        else:
            embedding_weights = static_model.embedding.weight
        tokenizer: Tokenizer = static_model.tokenizer

        return cls(tokenizer, embedding_weights=embedding_weights, base_model=model_name)

    @classmethod
    def from_model2vec(cls, model_id_or_path: str) -> StaticEmbedding:
        """
        Create a StaticEmbedding instance from a model2vec model. This method loads a pre-trained model2vec model
        and extracts the embedding weights and tokenizer to create a StaticEmbedding instance.

        Args:
            model_id_or_path (str): The identifier or path to the pre-trained model2vec model.

        Returns:
            StaticEmbedding: An instance of StaticEmbedding initialized with the tokenizer and embedding weights
                 the model2vec model.

        Raises:
            ImportError: If the `model2vec` package is not installed.
        """

        try:
            from model2vec import StaticModel
        except ImportError:
            raise ImportError("To use this method, please install the `model2vec` package: `pip install model2vec`")

        static_model = StaticModel.from_pretrained(model_id_or_path)
        if isinstance(static_model.embedding, np.ndarray):
            embedding_weights = torch.from_numpy(static_model.embedding)
        else:
            embedding_weights = static_model.embedding.weight
        tokenizer: Tokenizer = static_model.tokenizer

        return cls(tokenizer, embedding_weights=embedding_weights, base_model=model_id_or_path)
