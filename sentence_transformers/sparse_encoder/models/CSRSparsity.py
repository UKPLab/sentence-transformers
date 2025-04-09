from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model

from sentence_transformers.sparse_encoder.models.TopKActivation import TopKActivation


class CSRSparsity(nn.Module):
    """
    CSR (Compressed Sparse Row) Sparsity module.

    This module implements the Sparse AutoEncoder architecture based on the paper:
    Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation, https://arxiv.org/abs/2503.01776
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int = 100,
        k_aux: int = 50,
    ):
        """
        Initialize the CSR Sparsity module.

        Args:
            input_dim: Dimension of the input embeddings
            hidden_dim: Dimension of the hidden layer
            k: Number of top values to keep in TopK activation
            k_aux: Number of top values to keep for auxiliary loss
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.k_aux = k_aux

        # Encoder parameters
        self.W_enc = nn.Parameter(torch.randn(hidden_dim, input_dim) / input_dim**0.5)
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))
        self.b_pre = nn.Parameter(torch.zeros(input_dim))

        # Decoder parameters
        self.W_dec = nn.Parameter(torch.randn(input_dim, hidden_dim) / hidden_dim**0.5)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        # TopK activation functions
        self.topk = TopKActivation(k=k)
        self.topk_aux = TopKActivation(k=k_aux)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input embeddings.

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            Encoded embeddings of shape (batch_size, hidden_dim)
        """
        # Compute z = TopK(W_enc * (x - b_pre) + b_enc)
        z = self.topk(F.linear(x - self.b_pre, self.W_enc, self.b_enc))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the encoded embeddings.

        Args:
            z: Encoded embeddings of shape (batch_size, hidden_dim)

        Returns:
            Decoded embeddings of shape (batch_size, input_dim)
        """
        # Compute x̂ = W_dec * z + b_pre
        x_hat = F.linear(z, self.W_dec, self.b_pre)
        return x_hat

    def forward(self, features: dict) -> dict:
        """
        Forward pass of the CSR Sparsity module.

        Args:
            features: Dictionary containing sentence embeddings

        Returns:
            Dictionary containing the original embeddings and sparse embeddings
        """
        # Get the sentence embeddings
        x = features["sentence_embedding"]

        # Compute z = TopK(W_enc * (x - b_pre) + b_enc)
        z = self.encode(x)

        # Compute z_4k = TopK(W_enc * (x - b_pre) + b_enc) with 4k values
        z_4k = self.topk_aux(F.linear(x - self.b_pre, self.W_enc, self.b_enc))

        # Compute z_aux = TopK(W_enc * (x - b_pre) + b_enc) with k_aux values
        z_aux = self.topk_aux(F.linear(x - self.b_pre, self.W_enc, self.b_enc))

        # Compute x̂ = W_dec * z + b_pre
        x_hat = self.decode(z)

        # Compute x̂_4k = W_dec * z_4k + b_pre
        x_hat_4k = self.decode(z_4k)

        # Compute x̂_aux = W_dec * z_aux + b_pre
        x_hat_aux = self.decode(z_aux)

        # Compute f(x) - f(dx) for auxiliary loss
        e = x - x_hat
        e_hat = F.linear(z, self.W_dec, self.b_dec)

        # Update the features dictionary
        features.update(
            {
                "sparse_embedding": z,
                "sparse_embedding_4k": z_4k,
                "auxiliary_embedding": z_aux,
                "decoded_embedding": x_hat,
                "decoded_embedding_4k": x_hat_4k,
                "decoded_embedding_aux": x_hat_aux,
                "error": e,
                "error_hat": e_hat,
            }
        )

        return features

    def get_config_dict(self):
        """
        Get the configuration dictionary.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "k": self.k,
            "k_aux": self.k_aux,
        }

    def save(self, output_path, safe_serialization: bool = True) -> None:
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)
        
        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)
        module = CSRSparsity(**config)
        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(module, os.path.join(input_path, "model.safetensors"))
        else:
            module.load_state_dict(
                torch.load(
                    os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"), weights_only=True
                )
            )
        return module

    def __repr__(self):
        return f"CSRSparsity({self.get_config_dict()})"
