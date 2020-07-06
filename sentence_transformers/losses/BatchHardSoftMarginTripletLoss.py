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
        # Get the pairwise distance matrix
        pairwise_dist = BatchHardTripletLoss._pairwise_distances(embeddings, squared=squared)
        #pairwise_dist = BatchHardTripletLoss._cosine_distance(embeddings)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = BatchHardTripletLoss._get_anchor_positive_triplet_mask(labels).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = BatchHardTripletLoss._get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss with soft margin
        #tl = hardest_positive_dist - hardest_negative_dist + margin
        #tl[tl < 0] = 0
        tl = torch.log1p(torch.exp(hardest_positive_dist - hardest_negative_dist))
        triplet_loss = tl.mean()

        return triplet_loss