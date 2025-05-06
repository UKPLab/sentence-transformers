from __future__ import annotations

from typing import Literal

from sentence_transformers.losses.GISTEmbedLoss import GISTEmbedLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseGISTEmbedLoss(GISTEmbedLoss):
    def __init__(
        self,
        model: SparseEncoder,
        guide: SparseEncoder,
        temperature: float = 0.1,
        margin_strategy: Literal["absolute", "relative"] = "absolute",
        margin: float = 0.0,
    ) -> None:
        """
        This loss is used to train a SparseEncoder model using the GISTEmbed algorithm.
        It takes a model and a guide model as input, and uses the guide model to guide the
        in-batch negative sample selection. The cosine similarity is used to compute the loss
        and the temperature parameter is used to scale the cosine similarities.

        You can apply different false-negative filtering strategies to discard hard negatives that are too similar to
        the positive. Two strategies are supported:

            - "absolute": Discards negatives whose similarity score is greater than or equal to ``positive_score - margin``.
            - "relative": Discards negatives whose similarity score is greater than or equal to ``positive_score * (1 - margin)``.

        Args:
            model: SparseEncoder model based on a `transformers` model.
            guide: SparseEncoder model to guide the in-batch negative sample selection.
            temperature: Temperature parameter to scale the cosine similarities. Defaults to 0.1, adapted for Sparse embeddings. 
                Experimentation is recommended.
            margin_strategy: Strategy used for false negative filtering. One of {"absolute", "relative"}.
            margin: The margin value for filtering negatives. Defaults to 0.0, together with the "absolute" strategy,
                this only removes negatives that are more similar to the query than the positive is to the query.

        References:
            - For further details, see: https://arxiv.org/abs/2402.16829

        Requirements:
            1. (anchor, positive, negative) triplets
            2. (anchor, positive) pairs

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - :class:`SparseMultipleNegativesRankingLoss` is similar to this loss, but it does not use
              a guide model to guide the in-batch negative sample selection. :class:`SparseGISTEmbedLoss` yields
              a stronger training signal at the cost of some training overhead.

        Example:
            ::

                from datasets import Dataset
                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                # Initialize the SPLADE model
                model = SparseEncoder("distilbert/distilbert-base-uncased")
                guide = SparseEncoder("naver/splade-cocondenser-ensembledistil")

                train_dataset = Dataset.from_dict(
                    {
                        "anchor": ["It's nice weather outside today.", "He drove to work."],
                        "positive": ["It's so sunny.", "He took the car to the office."],
                    }
                )
                loss = losses.SparseGISTEmbedLoss(model, guide=guide)

                trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """

        return super().__init__(
            model, guide=guide, temperature=temperature, margin_strategy=margin_strategy, margin=margin
        )
