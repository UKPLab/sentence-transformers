from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import torch
import torch.nn as nn

from sentence_transformers.sparse_encoder.losses import (
    FlopsLoss,
    SparseDistillKLDivLoss,
    SparseMarginMSELoss,
    SparseMultipleNegativesRankingLoss,
)
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class PrincipalLoss(Enum):
    """The principal loss types for the model"""

    MMSE = SparseMarginMSELoss
    KL = SparseDistillKLDivLoss
    MRL = SparseMultipleNegativesRankingLoss


class SpladeLoss(nn.Module):
    def __init__(
        self, model: SparseEncoder, main_loss: PrincipalLoss, lambda_corpus: float = 0.1, lambda_query: float = 0.1
    ):
        super().__init__()
        self.model = model
        self.lambda_corpus = lambda_corpus
        self.lambda_query = lambda_query
        self.main_loss = main_loss
        self.flops_loss = FlopsLoss(model)

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        # Compute embeddings using the model
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        main_loss_value = self.main_loss.compute_loss_from_embeddings(embeddings, labels)

        flops_query = self.flops_loss.compute_loss_from_embeddings(embeddings, "query")
        flops_corpus = self.flops_loss.compute_loss_from_embeddings(embeddings, "corpus")

        # Compute the total loss
        total_loss = main_loss_value + self.lambda_query * flops_query + self.lambda_corpus * flops_corpus
        return total_loss

    def get_config_dict(self):
        """
        Get the configuration dictionary.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {"lambda_corpus": self.lambda_corpus, "lambda_query": self.lambda_query, "main_loss": self.main_loss}

    @property
    def citation(self) -> str:
        return """
@inproceedings{10.1145/3477495.3531857,
author = {Formal, Thibault and Lassance, Carlos and Piwowarski, Benjamin and Clinchant, St\'{e}phane},
title = {From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective},
year = {2022},
isbn = {9781450387323},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3531857},
doi = {10.1145/3477495.3531857},
abstract = {Neural retrievers based on dense representations combined with Approximate Nearest Neighbors search have recently received a lot of attention, owing their success to distillation and/or better sampling of examples for training -- while still relying on the same backbone architecture. In the meantime, sparse representation learning fueled by traditional inverted indexing techniques has seen a growing interest, inheriting from desirable IR priors such as explicit lexical matching. While some architectural variants have been proposed, a lesser effort has been put in the training of such models. In this work, we build on SPLADE -- a sparse expansion-based retriever -- and show to which extent it is able to benefit from the same training improvements as dense models, by studying the effect of distillation, hard-negative mining as well as the Pre-trained Language Model initialization. We furthermore study the link between effectiveness and efficiency, on in-domain and zero-shot settings, leading to state-of-the-art results in both scenarios for sufficiently expressive models.},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2353–2359},
numpages = {7},
keywords = {neural networks, indexing, sparse representations, regularization},
location = {Madrid, Spain},
series = {SIGIR '22}
}
}
"""
