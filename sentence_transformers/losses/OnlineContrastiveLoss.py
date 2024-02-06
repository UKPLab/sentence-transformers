from typing import Iterable, Dict
import torch.nn.functional as F
from torch import nn, Tensor
from .ContrastiveLoss import SiameseDistanceMetric
from sentence_transformers.SentenceTransformer import SentenceTransformer


class OnlineContrastiveLoss(nn.Module):
    def __init__(
        self, model: SentenceTransformer, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.5
    ):
        """
        This Online Contrastive loss is similar to :class:`ConstrativeLoss`, but it selects hard positive (positives that
        are far apart) and hard negative pairs (negatives that are close) and computes the loss only for these pairs.
        This loss often yields better performances than ContrastiveLoss.

        :param model: SentenceTransformer model
        :param distance_metric: Function that returns a distance between two embeddings. The class SiameseDistanceMetric contains pre-defined metrics that can be used
        :param margin: Negative samples (label == 0) should have a distance of at least the margin value.

        References:
            - `Training Examples > Quora Duplicate Questions <../../examples/training/quora_duplicate_questions/README.html>`_

        Requirements:
            1. (anchor, positive/negative) pairs
            2. Data should include hard positives and hard negatives

        Relations:
            - :class:`ContrastiveLoss` is similar, but does not use hard positive and hard negative pairs.
            :class:`OnlineContrastiveLoss` often yields better results.

        Inputs:
            +-----------------------------------------------+------------------------------+
            | Texts                                         | Labels                       |
            +===============================================+==============================+
            | (anchor, positive/negative) pairs             | 1 if positive, 0 if negative |
            +-----------------------------------------------+------------------------------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses, InputExample
                from torch.utils.data import DataLoader

                model = SentenceTransformer('all-MiniLM-L6-v2')
                train_examples = [
                    InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
                    InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0),
                ]

                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
                train_loss = losses.OnlineContrastiveLoss(model=model)
                model.fit(
                    [(train_dataloader, train_loss)],
                    epochs=10,
                )
        """
        super(OnlineContrastiveLoss, self).__init__()
        self.model = model
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, size_average=False):
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        distance_matrix = self.distance_metric(embeddings[0], embeddings[1])
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss
