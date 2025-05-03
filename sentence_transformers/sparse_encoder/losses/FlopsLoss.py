from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class FlopsLoss(nn.Module):
    def __init__(self, model: SparseEncoder) -> None:
        """
        FlopsLoss implements a regularization technique to promote sparsity in sparse encoder models.
        It calculates the squared L2 norm of the mean embedding vector, which helps reduce the number of floating-point
        operations (FLOPs) required during inference by encouraging more zero values in the embeddings.

        This loss is used as a regularization component within other losses like it's done in SpladeLoss rather than
        as a standalone loss function.

        Args:
            model: SparseEncoder model to be regularized

        References:
            - For further details, see: https://arxiv.org/abs/2004.05665

        Relations:
            - Used as a component within :class:`SpladeLoss` to regularize both query and document embeddings

        Example:
            This loss is typically used within the :class:`SpladeLoss` class, which combines it with other loss components.

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
