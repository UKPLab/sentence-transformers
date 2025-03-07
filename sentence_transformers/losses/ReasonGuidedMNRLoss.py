from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class ReasoningGuidedRankingLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct=util.cos_sim,
    ) -> None:
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # Transformation layer for reasoning influence
        self.reasoning_transform = nn.Linear(
            model.get_sentence_embedding_dimension(),
            model.get_sentence_embedding_dimension(),
        )

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        anchors = embeddings[0]  # (B, D)
        positives = embeddings[1]  # (B, D)

        # Apply reasoning transformation during training
        if len(embeddings) > 2:  # Reasoning is provided
            reasoning = embeddings[2]  # (B, D)
            reasoning_embed = self.reasoning_transform(reasoning)
            anchors = anchors + reasoning_embed  # Modify anchors with reasoning context

        candidates = torch.cat([positives] + embeddings[3:], dim=0)  # Include negatives if available

        # Compute similarity scores
        scores = self.similarity_fct(anchors, candidates) * self.scale
        range_labels = torch.arange(0, scores.size(0), device=scores.device)

        return self.cross_entropy_loss(scores, range_labels)

    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}
