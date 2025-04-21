from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from sentence_transformers import SentenceTransformer, util


class DistillKLDivLoss(nn.Module):
    # TODO

    def __init__(self, model: SentenceTransformer, similarity_fct=util.pairwise_dot_score) -> None:
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.KLDivLoss(reduction="none")

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        embeddings_query = embeddings[0]
        embeddings_pos = embeddings[1]
        embeddings_neg = embeddings[2]

        # Compute student scores
        student_scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
        student_scores_neg = self.similarity_fct(embeddings_query, embeddings_neg)

        # Pack into one tensor and apply log_softmax
        student_scores = torch.stack([student_scores_pos, student_scores_neg], dim=1)
        student_log_probs = torch.log_softmax(student_scores, dim=1)

        # Labels contain teacher similarity scores (already computed before training)
        # We expect labels to contain the teacher_pos_score and teacher_neg_score
        teacher_pos_scores = labels[:, 0]
        teacher_neg_scores = labels[:, 1]
        teacher_scores = torch.stack([teacher_pos_scores, teacher_neg_scores], dim=1)
        teacher_probs = torch.softmax(teacher_scores, dim=1)

        # KL Divergence
        loss = self.loss_fct(student_log_probs, teacher_probs).sum(dim=1).mean()
        return loss

    @property
    def citation(self) -> str:
        return """
@misc{lin2020distillingdenserepresentationsranking,
      title={Distilling Dense Representations for Ranking using Tightly-Coupled Teachers},
      author={Sheng-Chieh Lin and Jheng-Hong Yang and Jimmy Lin},
      year={2020},
      eprint={2010.11386},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2010.11386},
}
"""
