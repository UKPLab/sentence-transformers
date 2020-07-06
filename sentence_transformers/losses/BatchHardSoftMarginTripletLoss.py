import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict

from .BatchHardTripletLoss import BatchHardTripletLoss

class BatchHardSoftMarginTripletLoss(BatchHardTripletLoss):
    def __init__(self, sentence_embedder):
        super(BatchHardSoftMarginTripletLoss, self).__init__(sentence_embedder)
        self.sentence_embedder = sentence_embedder

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.sentence_embedder(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        return BatchHardSoftMarginTripletLoss.batch_hard_triplet_soft_margin_loss(labels, reps[0])


    # Hard Triplet Loss with Soft Margin
    # Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    @staticmethod
    def batch_hard_triplet_soft_margin_loss(labels: Tensor, embeddings: Tensor, squared: bool = False) -> Tensor:
        pass