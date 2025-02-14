from __future__ import annotations

from collections.abc import Generator

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import fullname


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        num_negatives: int | None = 4,
        scale: int = 20.0,
        activation_fct: nn.Module | None = nn.Tanh(),
    ) -> None:
        super().__init__()
        self.model = model
        self.num_negatives = num_negatives
        self.scale = scale
        self.activation_fct = activation_fct

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def call_model_with_columns(self, anchors: list[str], candidates: list[str]) -> Tensor:
        pairs = list(zip(anchors, candidates))
        return self.call_model_with_pairs(pairs)

    def call_model_with_pairs(self, pairs: list[list[str]]) -> Tensor:
        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens.to(self.model.device)
        logits = self.model(**tokens)[0]
        return logits.squeeze(1)

    def get_in_batch_negatives(
        self, anchors: list[str], candidates: list[list[str]]
    ) -> Generator[list[str], None, None]:
        batch_size = len(anchors)
        num_candidates = len(candidates)

        # Given N anchors, we want to select num_negatives negatives for each anchor
        candidates_flattened = [candidate for sublist in candidates for candidate in sublist]

        if self.num_negatives is not None:
            # Create a mask for each anchor to each candidate index, where the matching positive
            # and hard negatives are masked out. From the remaining options, we randomly select
            # num_negatives indices.
            mask = ~torch.eye(batch_size, dtype=torch.bool).repeat(1, num_candidates)
            negative_indices = torch.multinomial(mask.float(), self.num_negatives)
        else:
            # If num_negatives is None, we select all negatives
            negative_indices = torch.arange(len(candidates[0])).repeat(len(candidates), 1)

        for negative_indices_row in negative_indices.T:
            yield [candidates_flattened[negative_idx] for negative_idx in negative_indices_row]

    def calculate_loss(self, logits: Tensor, batch_size: int) -> Tensor:
        # (bsz, 1 + num_rand_negatives + num_hard_negatives)
        logits = torch.cat(logits, dim=0).reshape(-1, batch_size).T

        # Apply the post-processing on the logits
        if self.activation_fct:
            logits = self.activation_fct(logits)
        if self.scale:
            logits = logits * self.scale

        # For each sample in the batch, the first label is the positive, the rest are negatives
        labels = torch.zeros(batch_size, device=logits.device, dtype=torch.long)

        loss = self.cross_entropy_loss(logits, labels)
        return loss

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        anchors = inputs[0]
        positives = inputs[1]
        batch_size = len(anchors)

        scores = [self.call_model_with_columns(anchors, positives)]

        # In-batch negatives:
        for negatives in self.get_in_batch_negatives(anchors, inputs[1:]):
            scores.append(self.call_model_with_columns(anchors, negatives))

        # Hard negatives:
        for negatives in inputs[2:]:
            scores.append(self.call_model_with_columns(anchors, negatives))

        return self.calculate_loss(scores, batch_size)

    def get_config_dict(self) -> dict[str, float]:
        return {
            "scale": self.scale,
            "num_negatives": self.num_negatives,
            "activation_fct": fullname(self.activation_fct),
        }
