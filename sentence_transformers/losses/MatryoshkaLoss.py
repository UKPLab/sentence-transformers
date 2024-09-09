from __future__ import annotations

import random
import warnings
from typing import Any, Iterable

import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses.CachedGISTEmbedLoss import CachedGISTEmbedLoss
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import CachedMultipleNegativesRankingLoss


class ForwardDecorator:
    def __init__(self, fn) -> None:
        self.fn = fn

        self.dim = None
        self.cache = []
        self.cache_dim = None
        self.idx = 0

    def set_dim(self, dim) -> None:
        self.dim = dim
        self.idx = 0

    def shrink(self, tensor: Tensor) -> Tensor:
        tensor_dim = tensor.shape[-1]
        if self.dim > tensor_dim:
            raise ValueError(
                f"Dimension {self.dim} in matryoshka_dims cannot be greater than the model's embedding dimension: {tensor_dim}"
            )
        tensor = tensor[..., : self.dim]
        tensor = F.normalize(tensor, p=2, dim=-1)
        return tensor

    def __call__(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        # Growing cache:
        if self.cache_dim is None or self.cache_dim == self.dim:
            output = self.fn(features)
            self.cache.append(output)
            self.cache_dim = self.dim
        # Using cache:
        else:
            output = self.cache[self.idx]
        output["token_embeddings"] = self.shrink(output["token_embeddings"])
        output["sentence_embedding"] = self.shrink(output["sentence_embedding"])
        self.idx += 1
        return output


class MatryoshkaLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        loss: nn.Module,
        matryoshka_dims: list[int],
        matryoshka_weights: list[float | int] | None = None,
        n_dims_per_step: int = -1,
    ) -> None:
        """
        The MatryoshkaLoss can be seen as a loss *modifier* that allows you to use other loss functions at various
        different embedding dimensions. This is useful for when you want to train a model where users have the option
        to lower the embedding dimension to improve their embedding comparison speed and costs.

        Args:
            model: SentenceTransformer model
            loss: The loss function to be used, e.g.
                :class:`MultipleNegativesRankingLoss`,
                :class:`CoSENTLoss`, etc.
            matryoshka_dims: A list of embedding dimensions to be used
                for the loss function, e.g. [768, 512, 256, 128, 64].
            matryoshka_weights: A list of weights to be used for the
                loss function, e.g. [1, 1, 1, 1, 1]. If None, then the
                weights will be set to 1 for all dimensions.
            n_dims_per_step: The number of dimensions to use per step.
                If -1, then all dimensions are used. If > 0, then a
                random sample of n_dims_per_step dimensions are used per
                step. The default value is -1.

        References:
            - The concept was introduced in this paper: https://arxiv.org/abs/2205.13147
            - `Matryoshka Embeddings <../../examples/training/matryoshka/README.html>`_

        Requirements:
            1. The base loss cannot be :class:`CachedMultipleNegativesRankingLoss` or :class:`CachedGISTEmbedLoss`.

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | any                                   | any    |
            +---------------------------------------+--------+

        Relations:
            - :class:`Matryoshka2dLoss` uses this loss in combination with :class:`AdaptiveLayerLoss` which allows for
                layer reduction for faster inference.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses, InputExample
                from torch.utils.data import DataLoader

                model = SentenceTransformer("microsoft/mpnet-base")
                train_examples = [
                    InputExample(texts=['Anchor 1', 'Positive 1']),
                    InputExample(texts=['Anchor 2', 'Positive 2']),
                ]
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
                train_loss = losses.MultipleNegativesRankingLoss(model=model)
                train_loss = losses.MatryoshkaLoss(model, train_loss, [768, 512, 256, 128, 64])
                model.fit(
                    [(train_dataloader, train_loss)],
                    epochs=10,
                )
        """
        super().__init__()
        self.model = model
        self.loss = loss
        if isinstance(loss, CachedMultipleNegativesRankingLoss):
            warnings.warn("MatryoshkaLoss is not compatible with CachedMultipleNegativesRankingLoss.", stacklevel=2)
        if isinstance(loss, CachedGISTEmbedLoss):
            warnings.warn("MatryoshkaLoss is not compatible with CachedGISTEmbedLoss.", stacklevel=2)

        if matryoshka_weights is None:
            matryoshka_weights = [1] * len(matryoshka_dims)
        # Sort the dimensions and weights in descending order
        dims_weights = zip(matryoshka_dims, matryoshka_weights)
        self.matryoshka_dims, self.matryoshka_weights = zip(*sorted(dims_weights, key=lambda x: x[0], reverse=True))
        self.n_dims_per_step = n_dims_per_step

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        original_forward = self.model.forward
        try:
            decorated_forward = ForwardDecorator(original_forward)
            self.model.forward = decorated_forward

            dim_indices = range(len(self.matryoshka_dims))
            if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
                dim_indices = random.sample(dim_indices, self.n_dims_per_step)

            loss = 0.0
            for idx in dim_indices:
                dim = self.matryoshka_dims[idx]
                weight = self.matryoshka_weights[idx]
                decorated_forward.set_dim(dim)
                loss += weight * self.loss(sentence_features, labels)
        finally:
            self.model.forward = original_forward
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "matryoshka_dims": self.matryoshka_dims,
            "matryoshka_weights": self.matryoshka_weights,
            "n_dims_per_step": self.n_dims_per_step,
        }

    @property
    def citation(self) -> str:
        return """
@misc{kusupati2024matryoshka,
    title={Matryoshka Representation Learning},
    author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    year={2024},
    eprint={2205.13147},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""
