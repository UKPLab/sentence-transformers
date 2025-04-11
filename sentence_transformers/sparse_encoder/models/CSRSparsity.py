from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model

from sentence_transformers.sparse_encoder.models.TopKActivation import TopKActivation


class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


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

        # Pre-computed bias
        self.b_pre = nn.Parameter(torch.zeros(input_dim))

        # Encoder parameters
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)

        # latent bias
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Decoder parameters
        self.decoder: TiedTranspose = TiedTranspose(self.encoder)

        # TopK activation functions
        self.topk = TopKActivation(k=k)
        self.top4k = TopKActivation(k=4 * k)
        self.topk_aux = TopKActivation(k=k_aux)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input embeddings.

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            Encoded embeddings of shape (batch_size, hidden_dim)
        """
        # Compute z = W_enc * (x - b_pre) + b_enc
        z = F.linear(x - self.b_pre, self.encoder.weight, self.latent_bias)
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
        x_hat = F.linear(z, self.decoder.weight, self.b_pre)
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
        f_x = features["sentence_embedding"]

        # Compute z = (W_enc * (f_x - b_pre) + b_enc)
        z = self.encode(f_x)

        # Compute z_k = TopK(W_enc * (f_x - b_pre) + b_enc) with k values
        z_k = self.topk(z)

        # Compute z_4k = TopK(W_enc * (f_x - b_pre) + b_enc) with 4k values
        z_4k = self.top4k(z)

        # Compute z_aux = TopK(W_enc * (f_x - b_pre) + b_enc) with k_aux values
        z_aux = self.topk_aux(z)

        # Compute x̂_k = W_dec * z_k + b_pre
        x_hat_k = self.decode(z_k)

        # Compute x̂_4k = W_dec * z_4k + b_pre
        x_hat_4k = self.decode(z_4k)

        # Compute x̂_aux = W_dec * z_aux + b_pre
        x_hat_aux = self.decode(z_aux)

        e = f_x - x_hat_aux
        e_hat = x_hat_k + self.b_pre

        # Update the features dictionary
        features.update(
            {
                "sentence_embedding_backbone": f_x,
                "sentence_embedding_encoded": z,
                "sentence_embedding_encoded_4k": z_4k,
                "auxiliary_embedding": z_aux,
                "decoded_embedding_k": x_hat_k,
                "decoded_embedding_4k": x_hat_4k,
                "decoded_embedding_aux": x_hat_aux,
                "error": e,
                "error_hat": e_hat,
            }
        )
        features["sentence_embedding"] = z_k
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
