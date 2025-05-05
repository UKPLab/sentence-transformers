from __future__ import annotations

from sentence_transformers import util
from sentence_transformers.losses.CoSENTLoss import CoSENTLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseCoSENTLoss(CoSENTLoss):
    def __init__(self, model: SparseEncoder, scale: float = 20.0, similarity_fct=util.cos_sim) -> None:
        """
        This class implements CoSENT (Cosine Sentence).
        It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair.

        It computes the following loss function:

        ``loss = logsum(1+exp(s(i,j)-s(k,l))+exp...)``, where ``(i,j)`` and ``(k,l)`` are any of the input pairs in the
        batch such that the expected similarity of ``(i,j)`` is greater than ``(k,l)``. The summation is over all possible
        pairs of input pairs in the batch that match this condition.

        Anecdotal experiments show that this loss function produces a more powerful training signal than :class:`SparseCosineSimilarityLoss`,
        resulting in faster convergence and a final model with superior performance. Consequently, SparseCoSENTLoss may be used
        as a drop-in replacement for :class:`SparseCosineSimilarityLoss` in any training script.

        Args:
            model: SparseEncoder
            similarity_fct: Function to compute the PAIRWISE similarity
                between embeddings. Default is
                ``util.pairwise_cos_sim``.
            scale: Output of similarity function is multiplied by scale
                value. Represents the inverse temperature.

        References:
            - For further details, see: https://kexue.fm/archives/8847

        Requirements:
            - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Inputs:
            +--------------------------------+------------------------+
            | Texts                          | Labels                 |
            +================================+========================+
            | (sentence_A, sentence_B) pairs | float similarity score |
            +--------------------------------+------------------------+

        Relations:
            - :class:`SparseAnglELoss` is SparseCoSENTLoss with ``pairwise_angle_sim`` as the metric, rather than ``pairwise_cos_sim``.
            - :class:`SparseCosineSimilarityLoss` seems to produce a weaker training signal than SparseCoSENTLoss. In our experiments, SparseCoSENTLoss is recommended.

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
                loss = losses.SparseCoSENTLoss(model)

                trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        return super().__init__(model, scale=scale, similarity_fct=similarity_fct)
