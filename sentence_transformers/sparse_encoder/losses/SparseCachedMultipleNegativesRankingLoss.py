from __future__ import annotations

from torch import Tensor

from sentence_transformers import util
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import CachedMultipleNegativesRankingLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseCachedMultipleNegativesRankingLoss(CachedMultipleNegativesRankingLoss):
    def __init__(
        self,
        model: SparseEncoder,
        scale: float = 1.0,
        similarity_fct: callable[[Tensor, Tensor], Tensor] = util.dot_score,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        """
        Boosted version of MultipleNegativesRankingLoss (https://arxiv.org/pdf/1705.00652.pdf) by GradCache (https://arxiv.org/pdf/2101.06983.pdf).

        Constrastive learning (here our MNRL loss) with in-batch negatives is usually hard to work with large batch sizes due to (GPU) memory limitation.
        Even with batch-scaling methods like gradient-scaling, it cannot work either. This is because the in-batch negatives make the data points within
        the same batch non-independent and thus the batch cannot be broke down into mini-batches. GradCache is a smart way to solve this problem.
        It achieves the goal by dividing the computation into two stages of embedding and loss calculation, which both can be scaled by mini-batches.
        As a result, memory of constant size (e.g. that works with batch size = 32) can now process much larger batches (e.g. 65536).

        In detail:

            (1) It first does a quick embedding step without gradients/computation graphs to get all the embeddings;
            (2) Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings;
            (3) A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain.

        Notes: All steps are done with mini-batches. In the original implementation of GradCache, (2) is not done in mini-batches and
        requires a lot memory when the batch size is large. One drawback is about the speed. Gradient caching will sacrifice
        around 20% computation time according to the paper.

        Args:
            model: SparseEncoder model
            scale: Output of similarity function is multiplied by scale value
            similarity_fct: similarity function between sentence embeddings.
                By default, dot product. Can also be set to cosine similarity (and then set scale to 20)
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the more memory efficient the training is, but
                the slower the training will be. It's recommended to set it as high as your GPU memory allows. The default
                value is 32.
            show_progress_bar: If True, a progress bar for the mini-batches is shown during training. The default is False.

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://arxiv.org/pdf/2101.06983.pdf

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative pairs)
            2. Should be used with large `per_device_train_batch_size` and low `mini_batch_size` for superior performance, but slower training time than :class:`SparseMultipleNegativesRankingLoss`.

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
            - Equivalent to :class:`SparseMultipleNegativesRankingLoss`, but with caching that allows for much higher batch sizes
              (and thus better performance) without extra memory usage. This loss also trains roughly 2x to 2.4x slower than
              :class:`SparseMultipleNegativesRankingLoss`.

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
                train_dataset = Dataset.from_dict(
                    {
                        "anchor": ["It's nice weather outside today.", "He drove to work."] * 20,
                        "positive": ["It's so sunny.", "He took the car to the office."] * 20,
                    }
                )
                loss = losses.SparseCachedMultipleNegativesRankingLoss(model, mini_batch_size=8, show_progress_bar=True)

                trainer = SparseEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                    args=SparseEncoderTrainingArguments(per_device_train_batch_size=32),
                )
                trainer.train()
        """
        super().__init__(
            model,
            scale=scale,
            similarity_fct=similarity_fct,
            mini_batch_size=mini_batch_size,
            show_progress_bar=show_progress_bar,
        )
