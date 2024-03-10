from typing import Iterable, Dict, Callable
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .ContrastiveLoss import SiameseDistanceMetric
from sentence_transformers.SentenceTransformer import SentenceTransformer


class GISTEmbedLoss(nn.Module):
    def __init__(
        self, model: SentenceTransformer, guide: SentenceTransformer,
        distance_metric: Callable[[Tensor, Tensor]] = SiameseDistanceMetric.COSINE_DISTANCE,
        temperature: float = 0.01,
    ):
        """
        This loss is used to train a SentenceTransformer model using the GISTEmbed algorithm.
        It takes a model and a guide model as input, and uses the guide model to guide the
        in-batch negative sample selection.

        :param model: SentenceTransformer model
        :param guide: SentenceTransformer model to guide the in-batch negative sample selection.

        References:
            - For further details, see: https://arxiv.org/abs/2402.16829

        Requirements:
            1. (anchor, positive, negative) triplets
            2. (anchor, positive/negative) pairs

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+
            | (anchor, positive/negative) pairs     | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses, InputExample
                from torch.utils.data import DataLoader

                model = SentenceTransformer('all-MiniLM-L6-v2')
                guide = SentenceTransformer('all-MiniLM-L6-v2')
                train_examples = [
                    InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
                    InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0),
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
        self.distance_metric = distance_metric
        self.temperature = temperature

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        guide_embeddings = [self.guide(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

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

        # This is different from the original implementation, which uses a similarity metric instead of a distance metric.
        # We just negate the distance metric to get the similarity metric.

        # Compute the model's similarities
        ap_sim = -self.distance_metric(anchor, positive)
        an_sim = -self.distance_metric(anchor, negative)
        aa_sim = -self.distance_metric(anchor, anchor)
        pp_sim = -self.distance_metric(positive, positive)

        # Let's compute the similarity matrices for the combinations of anchor, positive, and negative samples.
        guided_ap_sim = -self.distance_metric(anchor_guide, positive_guide)
        guided_aa_sim = -self.distance_metric(anchor_guide, anchor_guide)
        guided_an_sim = -self.distance_metric(anchor_guide, negative_guide)
        guided_pp_sim = -self.distance_metric(positive_guide, positive_guide)


        # Define the anchor threshold
        guided_sim = guided_ap_sim.diagonal().view(-1, 1)

        # Find which samples cannot be used as negatives because they are
        # more similar to the query than the assigned positive as deemed by the guide model.
        # For this samples, we mask them with -inf to basically ignore their contribution to
        # the loss.

        ap_mask = guided_ap_sim > guided_sim
        an_mask = guided_an_sim > guided_sim
        aa_mask = guided_aa_sim > guided_sim
        pp_mask = guided_pp_sim > guided_sim

        ap_sim[ap_mask] = -torch.inf
        an_sim[an_mask] = -torch.inf
        aa_sim[aa_mask] = -torch.inf
        pp_sim[pp_mask] = -torch.inf


        scores = torch.cat([ap_sim, an_sim, aa_sim, pp_sim], dim=1) / self.temperature
        labels = torch.arange(scores.size(0)).long().to(anchor.device)
        return nn.CrossEntropyLoss()(scores, labels)

