from typing import Any, Dict, Iterable

import torch
from torch import Tensor, nn

from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import fullname


class CosineSimilarityLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, loss_fct=nn.MSELoss(), cos_score_transformation=nn.Identity()):
        """
        CosineSimilarityLoss expects that the InputExamples consists of two texts and a float label. It computes the
        vectors ``u = model(sentence_A)`` and ``v = model(sentence_B)`` and measures the cosine-similarity between the two.
        By default, it minimizes the following loss: ``||input_label - cos_score_transformation(cosine_sim(u,v))||_2``.

        Args:
            model: SentenceTransformer model
            loss_fct: Which pytorch loss function should be used to
                compare the ``cosine_similarity(u, v)`` with the
                input_label? By default, MSE is used: ``||input_label -
                cosine_sim(u, v)||_2``
            cos_score_transformation: The cos_score_transformation
                function is applied on top of cosine_similarity. By
                default, the identify function is used (i.e. no change).

        References:
            - `Training Examples > Semantic Textual Similarity <../../examples/training/sts/README.html>`_

        Requirements:
            1. Sentence pairs with corresponding similarity scores in range `[0, 1]`

        Relations:
            - :class:`CoSENTLoss` seems to produce a stronger training signal than CosineSimilarityLoss. In our experiments, CoSENTLoss is recommended.
            - :class:`AnglELoss` is :class:`CoSENTLoss` with ``pairwise_angle_sim`` as the metric, rather than ``pairwise_cos_sim``. It also produces a stronger training signal than CosineSimilarityLoss.

        Inputs:
            +--------------------------------+------------------------+
            | Texts                          | Labels                 |
            +================================+========================+
            | (sentence_A, sentence_B) pairs | float similarity score |
            +--------------------------------+------------------------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "sentence1": ["It's nice weather outside today.", "He drove to work."],
                    "sentence2": ["It's so sunny.", "She walked to the store."],
                    "score": [1.0, 0.3],
                })
                loss = losses.CosineSimilarityLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        return self.loss_fct(output, labels.float().view(-1))

    def get_config_dict(self) -> Dict[str, Any]:
        return {"loss_fct": fullname(self.loss_fct)}
