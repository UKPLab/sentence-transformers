import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict


class BatchSemiHardTripletLoss(nn.Module):
    def __init__(self, sentence_embedder, triplet_margin: float = 1):
        super(BatchSemiHardTripletLoss, self).__init__()
        self.sentence_embedder = sentence_embedder
        self.triplet_margin = triplet_margin

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.sentence_embedder(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        return BatchSemiHardTripletLoss.batch_semi_hard_triplet_loss(labels, reps[0], margin=self.triplet_margin)


    # Semi-Hard Triplet Loss
    # Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    @staticmethod
    def batch_semi_hard_triplet_loss(labels: Tensor, embeddings: Tensor, margin: float, squared: bool = False) -> Tensor:
        pass