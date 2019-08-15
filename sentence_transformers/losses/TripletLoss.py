import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from enum import Enum
from ..SentenceTransformer import SentenceTransformer

class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

class TripletLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin=1):
        super(TripletLoss, self).__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        rep_anchor, rep_pos, rep_neg = reps
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()