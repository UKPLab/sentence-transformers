from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from sentence_transformers import util
from sentence_transformers.losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseMultipleNegativesRankingLoss(MultipleNegativesRankingLoss):
    def __init__(self, model: SparseEncoder, scale: float = 1.0, similarity_fct=util.dot_score) -> None:
        """
        Given a list of (anchor, positive) pairs or (anchor, positive, negative) triplets, this loss optimizes the following:

        1. Given an anchor (e.g. a question), assign the highest similarity to the corresponding positive (i.e. answer)
           out of every single positive and negative (e.g. all answers) in the batch.

        If you provide the optional negatives, they will all be used as extra options from which the model must pick the
        correct positive. Within reason, the harder this "picking" is, the stronger the model will become. Because of
        this, a higher batch size results in more in-batch negatives, which then increases performance (to a point).

        This loss function works great to train embeddings for retrieval setups where you have positive pairs
        (e.g. (query, answer)) as it will sample in each batch ``n-1`` negative docs randomly.

        This loss is also known as InfoNCE loss, SimCSE loss, Cross-Entropy Loss with in-batch negatives, or simply
        in-batch negatives loss.

        Args:
            model: SparseEncoder model
            scale: Output of similarity function is multiplied by scale
                value
            similarity_fct: similarity function between sentence
                embeddings. By default, dot product. Can also be set to cosine
                similarity (and then set scale to 20)

        Requirements:
            1. Need to be used in SpladeLoss or CSRLoss as a loss function.
            2. (anchor, positive) pairs or (anchor, positive, negative) triplets

        Inputs:
            +-------------------------------------------------+--------+
            | Texts                                           | Labels |
            +=================================================+========+
            | (anchor, positive) pairs                        | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative) triplets           | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative_1, ..., negative_n) | none   |
            +-------------------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - :class:`SparseCachedMultipleNegativesRankingLoss` is equivalent to this loss, but it uses caching that allows for
              much higher batch sizes (and thus better performance) without extra memory usage. However, it is slightly
              slower.
            - :class:`SparseGISTEmbedLoss` is equivalent to this loss, but uses a guide model to guide the in-batch negative
              sample selection. `SparseGISTEmbedLoss` yields a stronger training signal at the cost of some training overhead.

        Example:
            ::

                from datasets import Dataset

                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                model = SparseEncoder("distilbert/distilbert-base-uncased")
                train_dataset = Dataset.from_dict(
                    {
                        "anchor": ["It's nice weather outside today.", "He drove to work."],
                        "positive": ["It's so sunny.", "He took the car to the office."],
                    }
                )
                loss = losses.SpladeLoss(
                    model=model, loss=losses.SparseMultipleNegativesRankingLoss(model), document_regularizer_weight=3e-5, query_regularizer_weight=5e-5
                )

                trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        return super().__init__(model, scale=scale, similarity_fct=similarity_fct)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        raise AttributeError(
            "SparseMultipleNegativesRankingLoss should not be used alone. Use it with SpladeLoss or CSRLoss."
        )
