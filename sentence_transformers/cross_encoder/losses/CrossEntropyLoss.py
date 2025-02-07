from __future__ import annotations

from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder


# TODO: Consider the naming of this class
class CrossEntropyLoss(nn.Module):
    def __init__(self, model: CrossEncoder) -> None:
        super().__init__()
        self.model = model
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(
                f"CrossEntropyLoss expects a dataset with two non-label columns, but got a dataset with {len(inputs)} columns."
            )

        pairs = list(zip(inputs[0], inputs[1]))
        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens.to(self.model.device)
        logits = self.model(**tokens)[0]
        loss = self.ce_loss(logits, labels)
        return loss
