from typing import Iterable, Dict
import torch.nn.functional as F
from torch import nn, Tensor
from .ContrastiveLoss import SiameseDistanceMetric
from sentence_transformers.SentenceTransformer import SentenceTransformer


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    """

    def __init__(self, model: SentenceTransformer, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin:float = 0.5):
        super(OnlineContrastiveLoss, self).__init__()
        self.model = model
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, size_average=False):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

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