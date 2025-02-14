from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers.models import StaticEmbedding, Transformer
from sentence_transformers.SentenceTransformer import SentenceTransformer


class GISTEmbedLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        guide: SentenceTransformer,
        temperature: float = 0.01,
    ) -> None:
        """
        This loss is used to train a SentenceTransformer model using the GISTEmbed algorithm.
        It takes a model and a guide model as input, and uses the guide model to guide the
        in-batch negative sample selection. The cosine similarity is used to compute the loss
        and the temperature parameter is used to scale the cosine similarities.

        Args:
            model: SentenceTransformer model based on a `transformers`
                model.
            guide: SentenceTransformer model to guide the in-batch
                negative sample selection.
            temperature: Temperature parameter to scale the cosine
                similarities.

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
            - :class:`MultipleNegativesRankingLoss` is similar to this loss, but it does not use
              a guide model to guide the in-batch negative sample selection. `GISTEmbedLoss` yields
              a stronger training signal at the cost of some training overhead.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                guide = SentenceTransformer("all-MiniLM-L6-v2")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.GISTEmbedLoss(model, guide)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.guide = guide
        self.temperature = temperature
        self.similarity_fct = nn.CosineSimilarity(dim=-1)
        if not isinstance(model[0], Transformer) or not isinstance(guide[0], Transformer):
            raise ValueError(
                "Both the training model and the guiding model must be based on the `transformers` architecture."
            )
        self.must_retokenize = (
            model.tokenizer.get_vocab() != guide.tokenizer.get_vocab() or guide.max_seq_length < model.max_seq_length
        )
        if self.must_retokenize:
            self.tokenizer = self.model.tokenizer

            if isinstance(self.model[0], StaticEmbedding):
                raise ValueError(
                    "If we must retokenize because the guide model has a different tokenizer, "
                    "then the Sentence Transformer model must not be based on a StaticEmbedding."
                )

    def sim_matrix(self, embed1: Tensor, embed2: Tensor) -> Tensor:
        return self.similarity_fct(embed1.unsqueeze(1), embed2.unsqueeze(0))

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        with torch.no_grad():
            if self.must_retokenize:
                decoded = [
                    self.tokenizer.batch_decode(sentence_feature["input_ids"], skip_special_tokens=True)
                    for sentence_feature in sentence_features
                ]
                sentence_features = [self.guide.tokenize(sentences) for sentences in decoded]
                sentence_features = [
                    {key: value.to(self.guide.device) for key, value in sentence_feature.items()}
                    for sentence_feature in sentence_features
                ]

            guide_embeddings = [
                self.guide(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features
            ]

        negative = None
        negative_guide = None

        if len(embeddings) == 2:
            anchor, positive = embeddings
            anchor_guide, positive_guide = guide_embeddings
        elif len(embeddings) == 3:
            anchor, positive, negative = embeddings
            anchor_guide, positive_guide, negative_guide = guide_embeddings
        else:
            raise ValueError(f"Expected 2 or 3 embeddings, got {len(embeddings)}")

        # Compute the model's similarities
        ap_sim = self.sim_matrix(anchor, positive)
        aa_sim = self.sim_matrix(anchor, anchor)
        pp_sim = self.sim_matrix(positive, positive)

        # Let's compute the similarity matrices for the combinations of anchor and positive samples.
        guided_ap_sim = self.sim_matrix(anchor_guide, positive_guide)
        guided_aa_sim = self.sim_matrix(anchor_guide, anchor_guide)
        guided_pp_sim = self.sim_matrix(positive_guide, positive_guide)

        # Define the anchor threshold
        guided_sim = guided_ap_sim.diagonal().view(-1, 1)

        # Find which samples cannot be used as negatives because they are
        # more similar to the query than the assigned positive as deemed by the guide model.
        # For these samples, we mask them with -inf to basically ignore their contribution to
        # the loss.
        ap_sim[guided_ap_sim > guided_sim] = -torch.inf
        aa_sim[guided_aa_sim > guided_sim] = -torch.inf
        pp_sim[guided_pp_sim > guided_sim] = -torch.inf

        scores = [ap_sim, aa_sim, pp_sim]

        # Handle the case where we have a negative sample
        if negative is not None:
            an_sim = self.sim_matrix(anchor, negative)
            guided_an_sim = self.sim_matrix(anchor_guide, negative_guide)
            an_sim[guided_an_sim > guided_sim] = -torch.inf

            scores.append(an_sim)

        scores = torch.cat(scores, dim=1) / self.temperature

        # NOTE: We use arange here since the ap_sim matrix contains the anchor-positive
        # similarities along the diagonal.
        labels = torch.arange(scores.size(0)).long().to(scores.device)

        return nn.CrossEntropyLoss()(scores, labels)

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "guide": self.guide,
            "temperature": self.temperature,
        }

    @property
    def citation(self) -> str:
        return """
@misc{solatorio2024gistembed,
    title={GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning},
    author={Aivin V. Solatorio},
    year={2024},
    eprint={2402.16829},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""
