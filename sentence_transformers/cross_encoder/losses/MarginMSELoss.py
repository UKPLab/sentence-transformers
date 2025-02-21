from __future__ import annotations

from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import fullname


class MarginMSELoss(nn.Module):
    def __init__(self, model: CrossEncoder, activation_fct: nn.Module = nn.Identity(), **kwargs) -> None:
        super().__init__()
        self.model = model
        self.activation_fct = activation_fct
        self.loss_fct = nn.MSELoss(**kwargs)

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        if len(inputs) != 3:
            raise ValueError(
                f"MSELoss expects a dataset with three non-label columns, but got a dataset with {len(inputs)} columns."
            )

        positive_pairs = list(zip(inputs[0], inputs[1]))
        tokens = self.model.tokenizer(
            positive_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens.to(self.model.device)
        positive_logits = self.model(**tokens)[0].view(-1)
        positive_logits = self.activation_fct(positive_logits)

        negative_pairs = list(zip(inputs[0], inputs[2]))
        tokens = self.model.tokenizer(
            negative_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens.to(self.model.device)
        negative_logits = self.model(**tokens)[0].view(-1)
        negative_logits = self.activation_fct(negative_logits)

        margin_logits = positive_logits - negative_logits
        loss = self.loss_fct(margin_logits, labels.float())
        return loss

    def get_config_dict(self):
        return {
            "activation_fct": fullname(self.activation_fct),
        }

    @property
    def citation(self) -> str:
        return """
@misc{hofstätter2021improving,
    title={Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation},
    author={Sebastian Hofstätter and Sophia Althammer and Michael Schröder and Mete Sertkan and Allan Hanbury},
    year={2021},
    eprint={2010.02666},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
"""
