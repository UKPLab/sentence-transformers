from __future__ import annotations

from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import fullname


class MSELoss(nn.Module):
    def __init__(self, model: CrossEncoder, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.loss_fct = nn.MSELoss(**kwargs)

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(
                f"MSELoss expects a dataset with two non-label columns, but got a dataset with {len(inputs)} columns."
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
        loss = self.loss_fct(logits, labels.float())
        return loss

    def get_config_dict(self):
        return {
            "activation_fct": fullname(self.activation_fct),
        }
