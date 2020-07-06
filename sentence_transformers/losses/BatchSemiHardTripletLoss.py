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
    # https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/triplet.py#L71
    # Paper: FaceNet: A Unified Embedding for Face Recognition and Clustering: https://arxiv.org/pdf/1503.03832.pdf
    @staticmethod
    def batch_semi_hard_triplet_loss(labels: Tensor, embeddings: Tensor, margin: float, squared: bool = False) -> Tensor:
        labels = labels.unsqueeze(1)

        pdist_matrix = BatchSemiHardTripletLoss._pairwise_distances(embeddings, squared)
        #pdist_matrix = BatchSemiHardTripletLoss._cosine_distance(embeddings)

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

        loss_mat = (pdist_matrix - semi_hard_negatives) + margin

        mask_positives = adjacency.float().cuda() - torch.eye(batch_size).cuda()
        mask_positives = mask_positives.cuda()
        num_positives = torch.sum(mask_positives)

        triplet_loss = torch.sum(torch.max(loss_mat * mask_positives, torch.tensor([0.0]).cuda())) / num_positives

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

    @staticmethod
    def _pairwise_distances(embeddings, squared=False):
        """Compute the 2D matrix of distances between all the embeddings.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        dot_product = torch.matmul(embeddings, embeddings.t())

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 - mask) * torch.sqrt(distances)

        return distances