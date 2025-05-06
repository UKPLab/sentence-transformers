from __future__ import annotations

import torch.nn as nn

from sentence_transformers.losses.CosineSimilarityLoss import CosineSimilarityLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseCosineSimilarityLoss(CosineSimilarityLoss):
    def __init__(
        self,
        model: SparseEncoder,
        loss_fct: nn.Module = nn.MSELoss(),
        cos_score_transformation: nn.Module = nn.Identity(),
    ) -> None:
        """
        SparseCosineSimilarityLoss expects that the InputExamples consists of two texts and a float label. It computes the
        vectors ``u = model(sentence_A)`` and ``v = model(sentence_B)`` and measures the cosine-similarity between the two.
        By default, it minimizes the following loss: ``||input_label - cos_score_transformation(cosine_sim(u,v))||_2``.

        Args:
            model: SparseEncoder model
            loss_fct: Which pytorch loss function should be used to
                compare the ``cosine_similarity(u, v)`` with the
                input_label? By default, MSE is used: ``||input_label -
                cosine_sim(u, v)||_2``
            cos_score_transformation: The cos_score_transformation
                function is applied on top of cosine_similarity. By
                default, the identify function is used (i.e. no change).

        References:
        # TODO: Not yet for sparse might want it ?
            - `Training Examples > Semantic Textual Similarity <../../../examples/sentence_transformer/training/sts/README.html>`_

        Requirements:
            1. Sentence pairs with corresponding similarity scores in range `[0, 1]`

        Inputs:
            +--------------------------------+------------------------+
            | Texts                          | Labels                 |
            +================================+========================+
            | (sentence_A, sentence_B) pairs | float similarity score |
            +--------------------------------+------------------------+

        Relations:
            - :class:`SparseCoSENTLoss` seems to produce a stronger training signal than CosineSimilarityLoss. In our experiments, SparseCoSENTLoss is recommended.
            - :class:`SparseAnglELoss` is :class:`SparseCoSENTLoss` with ``pairwise_angle_sim`` as the metric, rather than ``pairwise_cos_sim``. It also produces a stronger training signal than SparseCosineSimilarityLoss.

        Example:
            ::

                from datasets import Dataset
                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                model = SparseEncoder("distilbert/distilbert-base-uncased")
                train_dataset = Dataset.from_dict(
                    {
                        "sentence1": ["It's nice weather outside today.", "He drove to work."],
                        "sentence2": ["It's so sunny.", "She walked to the store."],
                        "score": [1.0, 0.3],
                    }
                )
                loss = losses.SparseCosineSimilarityLoss(model)

                trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        return super().__init__(model, loss_fct=loss_fct, cos_score_transformation=cos_score_transformation)
