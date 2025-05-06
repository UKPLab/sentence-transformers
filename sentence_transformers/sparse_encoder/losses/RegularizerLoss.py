from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from sentence_transformers.models.Asym import Asym
from sentence_transformers.sparse_encoder.models.IDF import IDF
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class FlopsLoss(nn.Module):
    def __init__(self, model: SparseEncoder) -> None:
        """
        FlopsLoss implements a regularization technique to promote sparsity in sparse encoder models.
        It calculates the squared L2 norm of the mean embedding vector, which helps reduce the number of floating-point
        operations (FLOPs) required during inference by encouraging more zero values in the embeddings.

        This loss is used as a regularization component within other losses like :class:`SpladeLoss` rather than
        being used as a standalone loss function.

        Args:
            model: SparseEncoder model to be regularized

        References:
            - For further details, see: https://arxiv.org/abs/2004.05665

        Relations:
            - Used as a component within :class:`SpladeLoss` to regularize both query and document embeddings

        Example:
            - This loss is typically used within the :class:`SpladeLoss` class, which combines it with other loss components.

        """
        super().__init__()
        self.model = model

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Compute the embeddings and distribute them to anchor and candidates (positive and optionally negatives)
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        return self.compute_loss_from_embeddings(embeddings)

    def compute_loss_from_embeddings(self, embeddings: list[torch.Tensor], embeddings_type: str) -> torch.Tensor:
        anchors = embeddings[0]  # (batch_size, embedding_dim)
        candidates = torch.cat(embeddings[1:])  # (batch_size * (1 + num_negatives), embedding_dim)

        if embeddings_type == "query":
            return torch.sum(torch.mean(anchors, dim=0) ** 2)
        else:
            return torch.sum(torch.mean(candidates, dim=0) ** 2)

    @property
    def citation(self) -> str:
        return """
            @article{paria2020minimizing,
    title={Minimizing flops to learn efficient sparse representations},
    author={Paria, Biswajit and Yeh, Chih-Kuan and Yen, Ian EH and Xu, Ning and Ravikumar, Pradeep and P{\'o}czos, Barnab{\'a}s},
    journal={arXiv preprint arXiv:2004.05665},
    year={2020}
    }
    """


class IDFFlopsLoss(nn.Module):
    def __init__(self, model: SparseEncoder, treshold: float = 0) -> None:
        """
        IDFFlopsLoss implements a regularization technique to promote sparsity in document embeddings by incorporating
        inverse document frequency (IDF) information. It extends the basic FLOPs loss by weighting each dimension of
        the embeddings according to its IDF score, encouraging sparsity especially on less informative features.

        The loss is computed as the squared L2 norm of the mean masked and IDF-normalized candidate embeddings.
        Only embeddings with sufficient non-zero elements (above the specified threshold) contribute to the loss.

        This loss is designed to be used as a regularization component within other losses like :class:`SpladeLoss`
        and is specifically applied to document embeddings rather than query embeddings.

        Args:
            model: SparseEncoder model to be regularized. Must contain an :class:`Asym` module with an :class:`IDF` submodule.
            threshold: Minimum number of non-zero elements required for an embedding to be considered in the loss computation.

        References:
            - For further details, see: https://arxiv.org/pdf/2411.04403v1 and https://arxiv.org/pdf/2504.14839

        Relations:
            - Used as a component within :class:`SpladeLoss` to regularize document embeddings based on the IDF-weighted sparsity.

        Example:
            - This loss is typically used within the :class:`SpladeLoss` class, which combines it with other loss components.

        """
        super().__init__()
        self.model = model
        self.threshold = treshold
        self.idf_weights = None
        # Check if it's the good format so if in modules of the model we have an Asym with an IDF
        for module in model.modules():
            if isinstance(module, Asym):
                for submodule in module.modules():
                    if isinstance(submodule, IDF):
                        self.idf_weights = submodule.weight

            if self.idf_weights is not None:
                break
        if self.idf_weights is None:
            raise ValueError(
                "The model must contain an Asym module with an IDF submodule to use IDFFlopsLoss. "
                "Please check the model architecture."
            )

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Compute the embeddings and distribute them to anchor and candidates (positive and optionally negatives)
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        return self.compute_loss_from_embeddings(embeddings)

    def compute_loss_from_embeddings(self, embeddings: list[torch.Tensor], embeddings_type: str) -> torch.Tensor:
        candidates = torch.cat(embeddings[1:])  # (batch_size * (1 + num_negatives), embedding_dim)

        l0_norm = (candidates != 0).sum(dim=1)

        mask = (l0_norm > self.threshold).float()

        weighted_candidates = candidates / self.idf_weights

        masked_candidates = weighted_candidates * mask.unsqueeze(1)

        return torch.sum(masked_candidates.mean(dim=0) ** 2)

    @property
    def citation(self) -> str:
        return """
@article{geng2024towards,
  title={Towards Competitive Search Relevance For Inference-Free Learned Sparse Retrievers},
  author={Geng, Zhichao and Ru, Dongyu and Yang, Yang},
  journal={arXiv preprint arXiv:2411.04403},
  year={2024}
}
"""
