from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from sentence_transformers import SentenceTransformer, util


class DistillKLDivLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, similarity_fct=util.pairwise_dot_score) -> None:
        """
        Compute the KL divergence loss between probability distributions derived from student and teacher models' similarity scores.
        By default, similarity is calculated using the dot-product. This loss is designed for knowledge distillation
        where a smaller student model learns from a more powerful teacher model.

        The loss computes softmax probabilities from the teacher similarity scores and log-softmax probabilities
        from the student model, then calculates the KL divergence between these distributions.

        Args:
            model: SentenceTransformer model (student model)
            similarity_fct: Which similarity function to use for the student model

        References:
            - For more details, please refer to https://arxiv.org/abs/2010.11386

        Requirements:
            1. (query, positive, negative_1, ..., negative_n) examples
            2. Labels containing teacher model's scores between query-positive and query-negative pairs

        Inputs:
            +------------------------------------------------+------------------------------------------------------------+
            | Texts                                          | Labels                                                     |
            +================================================+============================================================+
            | (query, positive, negative)                    | [Teacher(query, positive), Teacher(query, negative)]       |
            +------------------------------------------------+------------------------------------------------------------+
            | (query, positive, negative_1, ..., negative_n) | [Teacher(query, positive), Teacher(query, negative_i)...]  |
            +------------------------------------------------+------------------------------------------------------------+

        Relations:
            - Similar to :class:`~sentence_transformers.losses.MarginMSELoss` but uses KL divergence instead of MSE
            - More suited for distillation tasks where preserving ranking is important

        Example:

            Using a teacher model to compute similarity scores for distillation:

            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                student_model = SentenceTransformer("microsoft/mpnet-base")
                teacher_model = SentenceTransformer("all-mpnet-base-v2")
                train_dataset = Dataset.from_dict({
                    "query": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to work."],
                    "negative": ["It's very cold.", "She walked to the store."],
                })

                def compute_labels(batch):
                    import torch

                    emb_queries = teacher_model.encode(batch["query"])
                    emb_positives = teacher_model.encode(batch["positive"])
                    emb_negatives = teacher_model.encode(batch["negative"])

                    pos_scores = teacher_model.similarity_pairwise(emb_queries, emb_positives)
                    neg_scores = teacher_model.similarity_pairwise(emb_queries, emb_negatives)

                    # Stack the scores for positive and negative pairs
                    return {
                        "label": torch.stack([pos_scores, neg_scores], dim=1)
                    }

                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.DistillKLDivLoss(student_model)

                trainer = SentenceTransformerTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
                trainer.train()

            With multiple negatives:

            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                student_model = SentenceTransformer("microsoft/mpnet-base")
                teacher_model = SentenceTransformer("all-mpnet-base-v2")

                train_dataset = Dataset.from_dict(
                    {
                        "query": ["It's nice weather outside today.", "He drove to work."],
                        "positive": ["It's so sunny.", "He took the car to work."],
                        "negative1": ["It's very cold.", "She walked to the store."],
                        "negative2": ["Its rainy", "She took the bus"],
                    }
                )


                def compute_labels(batch):
                    import torch

                    emb_queries = teacher_model.encode(batch["query"])
                    emb_positives = teacher_model.encode(batch["positive"])
                    emb_negatives1 = teacher_model.encode(batch["negative1"])
                    emb_negatives2 = teacher_model.encode(batch["negative2"])

                    pos_scores = teacher_model.similarity_pairwise(emb_queries, emb_positives)
                    neg_scores1 = teacher_model.similarity_pairwise(emb_queries, emb_negatives1)
                    neg_scores2 = teacher_model.similarity_pairwise(emb_queries, emb_negatives2)

                    # Stack the scores for positive and multiple negative pairs
                    return {
                        "label": torch.stack([pos_scores, neg_scores1, neg_scores2], dim=1)
                    }

                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.DistillKLDivLoss(student_model)

                trainer = SentenceTransformerTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
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
        embeddings_negs = embeddings[2:]

        # Compute student scores
        student_scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
        if len(embeddings_negs) == 1:
            # Single negative case
            student_scores_neg = self.similarity_fct(embeddings_query, embeddings_negs[0])
        else:
            # Multiple negatives case
            student_scores_neg = torch.stack(
                [self.similarity_fct(embeddings_query, neg) for neg in embeddings_negs],
                dim=1,
            )

        # Teacher scores
        teacher_pos_scores = labels[:, 0].unsqueeze(1)
        teacher_neg_scores = labels[:, 1:]

        # Prepare student scores to match teacher scores shape
        student_scores_pos = student_scores_pos.unsqueeze(1)

        # Create log probabilities for student scores
        if len(embeddings_negs) == 1:
            student_scores_neg = student_scores_neg.unsqueeze(1)
            student_scores = torch.cat([student_scores_pos, student_scores_neg], dim=1)
        else:
            student_scores = torch.cat([student_scores_pos, student_scores_neg], dim=1)

        student_log_probs = torch.log_softmax(student_scores, dim=1)

        # Create probabilities for teacher scores
        teacher_scores = torch.cat([teacher_pos_scores, teacher_neg_scores], dim=1)
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
