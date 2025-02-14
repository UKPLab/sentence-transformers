from __future__ import annotations

from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import fullname


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        activation_fct: nn.Module = nn.Identity(),
        pos_weight: Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.activation_fct = activation_fct
        self.pos_weight = pos_weight
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, **kwargs)

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(
                f"BinaryCrossEntropyLoss expects a dataset with two non-label columns, but got a dataset with {len(inputs)} columns."
            )

        pairs = list(zip(inputs[0], inputs[1]))
        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens.to(self.model.device)
        logits = self.model(**tokens)[0].view(-1)
        logits = self.activation_fct(logits)
        loss = self.bce_with_logits_loss(logits, labels.float())
        return loss

    def get_config_dict(self):
        return {
            "activation_fct": fullname(self.activation_fct),
            "pos_weight": self.pos_weight if self.pos_weight is None else self.pos_weight.item(),
        }
