from __future__ import annotations

import torch.nn as nn

from sentence_transformers.losses.CosineSimilarityLoss import CosineSimilarityLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseCosineSimilarityLoss(CosineSimilarityLoss):
    def __init__(
        self,
        model: SparseEncoder,
        loss_fct: nn.Module = nn.MSELoss(),
        cos_score_transformation: nn.Module = nn.Identity(),
    ) -> None:
        return super().__init__(model, loss_fct=loss_fct, cos_score_transformation=cos_score_transformation)
