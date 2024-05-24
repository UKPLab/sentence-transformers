from typing import Dict, Iterable

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class CoSENTLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.pairwise_cos_sim):
        """
        This class implements CoSENT (Cosine Sentence) loss.
        It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair.

        It computes the following loss function:

        ``loss = logsum(1+exp(s(k,l)-s(i,j))+exp...)``, where ``(i,j)`` and ``(k,l)`` are any of the input pairs in the
        batch such that the expected similarity of ``(i,j)`` is greater than ``(k,l)``. The summation is over all possible
        pairs of input pairs in the batch that match this condition.

        Anecdotal experiments show that this loss function produces a more powerful training signal than :class:`CosineSimilarityLoss`,
        resulting in faster convergence and a final model with superior performance. Consequently, CoSENTLoss may be used
        as a drop-in replacement for :class:`CosineSimilarityLoss` in any training script.

        Args:
            model: SentenceTransformerModel
            similarity_fct: Function to compute the PAIRWISE similarity
                between embeddings. Default is
                ``util.pairwise_cos_sim``.
            scale: Output of similarity function is multiplied by scale
                value. Represents the inverse temperature.

        References:
            - For further details, see: https://kexue.fm/archives/8847

        Requirements:
            - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Relations:
            - :class:`AnglELoss` is CoSENTLoss with ``pairwise_angle_sim`` as the metric, rather than ``pairwise_cos_sim``.
            - :class:`CosineSimilarityLoss` seems to produce a weaker training signal than CoSENTLoss. In our experiments, CoSENTLoss is recommended.

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
                loss = losses.CoSENTLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super(CoSENTLoss, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.scale = scale

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        scores = self.similarity_fct(embeddings[0], embeddings[1])
        scores = scores * self.scale
        scores = scores[:, None] - scores[None, :]

        # label matrix indicating which pairs are relevant
        labels = labels[:, None] < labels[None, :]
        labels = labels.float()

        # mask out irrelevant pairs so they are negligible after exp()
        scores = scores - (1 - labels) * 1e12

        # append a zero as e^0 = 1
        scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
        loss = torch.logsumexp(scores, dim=0)

        return loss

    def get_config_dict(self):
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}

    @property
    def citation(self) -> str:
        return """
@online{kexuefm-8847,
    title={CoSENT: A more efficient sentence vector scheme than Sentence-BERT},
    author={Su Jianlin},
    year={2022},
    month={Jan},
    url={https://kexue.fm/archives/8847},
}
"""
