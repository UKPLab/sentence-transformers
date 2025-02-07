from __future__ import annotations

import time
from contextlib import ContextDecorator

from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder


class timer(ContextDecorator):
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> None:
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        print(f"{self.name} took {time.time() - self.start:.4f} seconds")


# TODO: Bad name, don't 1-1 copy the name from PyTorch
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

        # with timer("making pairs"):
        pairs = list(zip(inputs[0], inputs[1]))
        # with timer("tokenizing"):
        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # with timer("moving to device"):
        tokens.to(self.model.device)
        # with timer(f"inference (shape: {tokens['input_ids'].shape})"):
        logits = self.model(**tokens)[0]
        # with timer("calculating loss"):
        loss = self.ce_loss(logits, labels)
        return loss
