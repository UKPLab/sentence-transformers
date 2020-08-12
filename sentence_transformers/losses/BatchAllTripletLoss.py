import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from .BatchHardTripletLoss import BatchHardTripletLoss

class BatchAllTripletLoss(nn.Module):
    def __init__(self, sentence_embedder, triplet_margin: float = 1):
        super(BatchAllTripletLoss, self).__init__()
        self.sentence_embedder = sentence_embedder
        self.triplet_margin = triplet_margin

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.sentence_embedder(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        return BatchAllTripletLoss.batch_all_triplet_loss(labels, reps[0], margin=self.triplet_margin)


    # Hard Triplet Loss
    # Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    # Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    # Blog post: https://omoindrot.github.io/triplet-loss
    @staticmethod
    def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
        """Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            Label_Sentence_Triplet: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = BatchHardTripletLoss._pairwise_distances(embeddings, squared=squared)

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = BatchHardTripletLoss._get_triplet_mask(labels)
        triplet_loss = mask.float() * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss[triplet_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        num_valid_triplets = mask.sum()

        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets

