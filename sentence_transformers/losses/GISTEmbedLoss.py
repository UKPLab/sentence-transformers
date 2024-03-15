from typing import Any, Iterable, Dict
import torch
from torch import nn, Tensor
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.models import Transformer


class GISTEmbedLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        guide: SentenceTransformer,
        temperature: float = 0.01,
    ):
        """
        This loss is used to train a SentenceTransformer model using the GISTEmbed algorithm.
        It takes a model and a guide model as input, and uses the guide model to guide the
        in-batch negative sample selection. The cosine similarity is used to compute the loss
        and the temperature parameter is used to scale the cosine similarities.

        :param model: SentenceTransformer model based on a `transformers` model.
        :param guide: SentenceTransformer model to guide the in-batch negative sample selection.
        :param temperature: Temperature parameter to scale the cosine similarities.

        References:
            - For further details, see: https://arxiv.org/abs/2402.16829

        Requirements:
            1. (anchor, positive, negative) triplets
            2. (anchor, positive) pairs

        Relations:
            - :class:`MultipleNegativesRankingLoss` is similar to this loss, but it does not use
              a guide model to guide the in-batch negative sample selection. `GISTEmbedLoss` yields
              a stronger training signal at the cost of some training overhead.

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses, InputExample
                from torch.utils.data import DataLoader

                model = SentenceTransformer('all-MiniLM-L6-v2')
                guide = SentenceTransformer('avsolatorio/GIST-small-Embedding-v0')
                train_examples = [
                    InputExample(texts=['The first query',  'The first positive passage',  'The first negative passage']),
                    InputExample(texts=['The second query', 'The second positive passage', 'The second negative passage']),
                ]
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
                train_loss = losses.GISTEmbedLoss(model=model, guide=guide)
                model.fit(
                    [(train_dataloader, train_loss)],
                    epochs=10,
                )
        """
        super(GISTEmbedLoss, self).__init__()
        self.model = model
        self.guide = guide
        self.temperature = temperature
        self.similarity_fct = nn.CosineSimilarity(dim=-1)
        if not isinstance(model[0], Transformer) or not isinstance(guide[0], Transformer):
            raise ValueError(
                "Both the training model and the guiding model must be based on the `transformers` architecture."
            )
        self.must_retokenize = (
            model.tokenizer.vocab != guide.tokenizer.vocab or guide.max_seq_length < model.max_seq_length
        )

    def sim_matrix(self, embed1, embed2):
        return self.similarity_fct(embed1.unsqueeze(1), embed2.unsqueeze(0))

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        with torch.no_grad():
            if self.must_retokenize:
                decoded = [
                    self.model.tokenizer.batch_decode(sentence_feature["input_ids"], skip_special_tokens=True)
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
            raise ValueError("Expected 2 or 3 embeddings, got {}".format(len(embeddings)))

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

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "guide": self.guide,
            "temperature": self.temperature,
        }
