from __future__ import annotations

from typing import Literal

from sentence_transformers.losses.CachedGISTEmbedLoss import CachedGISTEmbedLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseCachedGISTEmbedLoss(CachedGISTEmbedLoss):
    def __init__(
        self,
        model: SparseEncoder,
        guide: SparseEncoder,
        temperature: float = 0.1,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
        margin_strategy: Literal["absolute", "relative"] = "absolute",
        margin: float = 0.0,
    ) -> None:
        """
        This loss is a combination of :class:`SparseGISTEmbedLoss` and :class:`SparseCachedMultipleNegativesRankingLoss`.
        Typically, :class:`SparseCachedMultipleNegativesRankingLoss` requires a larger batch size for better performance.
        :class:`SparseGISTEmbedLoss` yields stronger training signals than :class:`SparseMultipleNegativesRankingLoss` due to the
        use of a guide model for in-batch negative sample selection. Meanwhile, :class:`SparseCachedMultipleNegativesRankingLoss`
        allows for scaling of the batch size by dividing the computation into two stages of embedding and loss
        calculation, which both can be scaled by mini-batches (https://arxiv.org/pdf/2101.06983.pdf).

        By combining the guided selection from :class:`SparseGISTEmbedLoss` and Gradient Cache from
        :class:`SparseCachedMultipleNegativesRankingLoss`, it is possible to reduce memory usage while maintaining performance
        levels comparable to those of :class:`SparseGISTEmbedLoss`.

        You can apply different false-negative filtering strategies to discard hard negatives that are too similar to
        the positive. Two strategies are supported:

            - "absolute": Discards negatives whose similarity score is greater than or equal to ``positive_score - margin``.
            - "relative": Discards negatives whose similarity score is greater than or equal to ``positive_score * (1 - margin)``.

        Args:
            model: SparseEncoder model
            guide: SparseEncoder model to guide the in-batch negative sample selection.
            temperature: Temperature parameter to scale the cosine similarities, default is 0.1 here adapted for Sparse embeddings,
                might need some adaptations.
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the more memory efficient the training is, but
                the slower the training will be. It's recommended to set it as high as your GPU memory allows. The default
                value is 32.
            show_progress_bar: If True, a progress bar for the mini-batches is shown during training. The default is False.
            margin_strategy: Strategy used for false negative filtering. One of {"absolute", "relative"}.
            margin: The margin value for filtering negatives. Defaults to 0.0, together with the "absolute" strategy,
                this only removes negatives that are more similar to the query than the positive is to the query.

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://arxiv.org/pdf/2101.06983.pdf
            - GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning https://arxiv.org/abs/2402.16829

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative pairs)
            2. Should be used with large batch sizes for superior performance, but has slower training time than :class:`SparseMultipleNegativesRankingLoss`

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
            - Equivalent to :class:`SparseGISTEmbedLoss`, but with caching that allows for much higher batch sizes

        Example:
            ::

                from datasets import Dataset

                from sentence_transformers.sparse_encoder import (
                    SparseEncoder,
                    SparseEncoderTrainer,
                    SparseEncoderTrainingArguments,
                    losses,
                )

                model = SparseEncoder("distilbert/distilbert-base-uncased")
                guide = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                train_dataset = Dataset.from_dict(
                    {
                        "anchor": ["It's nice weather outside today.", "He drove to work."] * 20,
                        "positive": ["It's so sunny.", "He took the car to the office."] * 20,
                    }
                )
                loss = losses.SparseCachedGISTEmbedLoss(
                    model,
                    guide,
                    mini_batch_size=8,
                    show_progress_bar=True,
                )

                trainer = SparseEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                    args=SparseEncoderTrainingArguments(per_device_train_batch_size=32),
                )
                trainer.train()
        """
        return super().__init__(
            model=model,
            guide=guide,
            temperature=temperature,
            mini_batch_size=mini_batch_size,
            show_progress_bar=show_progress_bar,
            margin_strategy=margin_strategy,
            margin=margin,
        )
