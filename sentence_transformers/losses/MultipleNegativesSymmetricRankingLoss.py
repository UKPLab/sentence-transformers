from typing import Dict, Iterable

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class MultipleNegativesSymmetricRankingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.cos_sim):
        """
        This loss is an adaptation of MultipleNegativesRankingLoss. MultipleNegativesRankingLoss computes the following loss:
        For a given anchor and a list of candidates, find the positive candidate.

        In MultipleNegativesSymmetricRankingLoss, we add another loss term: Given the positive and a list of all anchors,
        find the correct (matching) anchor.

        For the example of question-answering: You have (question, answer)-pairs. MultipleNegativesRankingLoss just computes
        the loss to find the answer for a given question. MultipleNegativesSymmetricRankingLoss additionally computes the
        loss to find the question for a given answer.

        Note: If you pass triplets, the negative entry will be ignored. A anchor is just searched for the positive.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale
                value
            similarity_fct: similarity function between sentence
                embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)

        Requirements:
            1. (anchor, positive) pairs

        Relations:
            - Like :class:`MultipleNegativesRankingLoss`, but with an additional loss term.

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesSymmetricRankingLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super(MultipleNegativesSymmetricRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        anchor = reps[0]
        candidates = torch.cat(reps[1:])

        scores = self.similarity_fct(anchor, candidates) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Example a[i] should match with b[i]

        anchor_positive_scores = scores[:, 0 : len(reps[1])]
        forward_loss = self.cross_entropy_loss(scores, labels)
        backward_loss = self.cross_entropy_loss(anchor_positive_scores.transpose(0, 1), labels)
        return (forward_loss + backward_loss) / 2

    def get_config_dict(self):
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}
