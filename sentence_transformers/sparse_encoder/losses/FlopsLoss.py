from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class FlopsLoss(nn.Module):
    def __init__(self, model: SparseEncoder, threshold: float = None) -> None:
        """
        FlopsLoss implements a regularization technique to promote sparsity in sparse encoder models.
        It calculates the squared L2 norm of the mean embedding vector, which helps reduce the number of floating-point
        operations (FLOPs) required during inference by encouraging more zero values in the embeddings.
        It can use a threshold to ignore embeddings with too few non-zero elements.

        This loss is used as a regularization component within other losses like :class:`SpladeLoss` rather than
        being used as a standalone loss function.

        Args:
            model: SparseEncoder model to be regularized
            threshold: Optional threshold for the number of non-zero elements in the embeddings.
                If specified, only embeddings with more than this number of non-zero elements will be considered.
                This can help to ignore embeddings that are too sparse and may not contribute meaningfully to the loss.

        References:
            - For further details, see: https://arxiv.org/pdf/2004.05665 for the general FLOPS loss and https://arxiv.org/pdf/2504.14839 for FLOPS with thresholds, a.k.a. FLOPS with l0 masking.

        Relations:
            - Used as a component within :class:`SpladeLoss` to regularize both query and document embeddings

        Example:
            - This loss is typically used within the :class:`SpladeLoss` class, which combines it with other loss components.

        """
        super().__init__()
        self.model = model
        self.threshold = threshold

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Compute the embeddings and distribute them to anchor and candidates (positive and optionally negatives)
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        return self.compute_loss_from_embeddings(embeddings)

    def compute_loss_from_embeddings(self, embeddings: list[torch.Tensor], embeddings_type: str) -> torch.Tensor:
        if embeddings_type == "query":
            embeddings_to_use = embeddings[0]  # (batch_size, embedding_dim)
        else:
            embeddings_to_use = torch.cat(embeddings[1:])  # (batch_size * (1 + num_negatives), embedding_dim)

        if self.threshold is not None:
            l0_norm = (embeddings_to_use != 0).sum(dim=1)
            mask = (l0_norm > self.threshold).float()
            embeddings_to_use = embeddings_to_use * mask.unsqueeze(1)

        return torch.sum(torch.mean(embeddings_to_use, dim=0) ** 2)

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
