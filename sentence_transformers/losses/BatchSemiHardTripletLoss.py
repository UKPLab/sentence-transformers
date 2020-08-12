import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from .BatchHardTripletLoss import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction

class BatchSemiHardTripletLoss(nn.Module):
    def __init__(self, sentence_embedder, triplet_margin: float = 5, distance_function = BatchHardTripletLossDistanceFunction.eucledian_distance):
        super(BatchSemiHardTripletLoss, self).__init__()
        self.sentence_embedder = sentence_embedder
        self.triplet_margin = triplet_margin
        self.distance_function = distance_function

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.sentence_embedder(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        return self.batch_semi_hard_triplet_loss(labels, reps[0])


    # Semi-Hard Triplet Loss
    # Based on: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/triplet.py#L71
    # Paper: FaceNet: A Unified Embedding for Face Recognition and Clustering: https://arxiv.org/pdf/1503.03832.pdf
    def batch_semi_hard_triplet_loss(self, labels: Tensor, embeddings: Tensor) -> Tensor:
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
        labels = labels.unsqueeze(1)

        pdist_matrix = self.distance_function(embeddings)


        adjacency = labels == labels.t()
        adjacency_not = ~adjacency

        batch_size = torch.numel(labels)
        pdist_matrix_tile = pdist_matrix.repeat([batch_size, 1])

        mask = adjacency_not.repeat([batch_size, 1]) & (pdist_matrix_tile > torch.reshape(pdist_matrix.t(), [-1, 1]))

        mask_final = torch.reshape(torch.sum(mask, 1, keepdims=True) > 0.0, [batch_size, batch_size])
        mask_final = mask_final.t()

        negatives_outside = torch.reshape(BatchSemiHardTripletLoss._masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
        negatives_outside = negatives_outside.t()

        negatives_inside = BatchSemiHardTripletLoss._masked_maximum(pdist_matrix, adjacency_not)
        negatives_inside = negatives_inside.repeat([1, batch_size])

        semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

        loss_mat = (pdist_matrix - semi_hard_negatives) + self.triplet_margin

        mask_positives = adjacency.float().to(labels.device) - torch.eye(batch_size, device=labels.device)
        mask_positives = mask_positives.to(labels.device)
        num_positives = torch.sum(mask_positives)

        triplet_loss = torch.sum(torch.max(loss_mat * mask_positives, torch.tensor([0.0], device=labels.device))) / num_positives

        return triplet_loss

    @staticmethod
    def _masked_minimum(data, mask, dim=1):
        axis_maximums, _ = data.max(dim, keepdims=True)
        masked_minimums = (data - axis_maximums) * mask
        masked_minimums, _ = masked_minimums.min(dim, keepdims=True)
        masked_minimums += axis_maximums

        return masked_minimums

    @staticmethod
    def _masked_maximum(data, mask, dim=1):
        axis_minimums, _ = data.min(dim, keepdims=True)
        masked_maximums = (data - axis_minimums) * mask
        masked_maximums, _ = masked_maximums.max(dim, keepdims=True)
        masked_maximums += axis_minimums

        return masked_maximums

