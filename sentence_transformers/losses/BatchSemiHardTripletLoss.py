import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from .BatchHardTripletLoss import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction
from sentence_transformers.SentenceTransformer import SentenceTransformer


class BatchSemiHardTripletLoss(nn.Module):
    """
    BatchSemiHardTripletLoss takes a batch with (label, sentence) pairs and computes the loss for all possible, valid
    triplets, i.e., anchor and positive must have the same label, anchor and negative a different label. It then looks
    for the semi hard positives and negatives.
    The labels must be integers, with same label indicating sentences from the same class. You train dataset
    must contain at least 2 examples per label class. The margin is computed automatically.

    Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    Blog post: https://omoindrot.github.io/triplet-loss

    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two emeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used


    Example::

       from sentence_transformers import SentenceTransformer, SentencesDataset, losses
       from sentence_transformers.readers import InputExample

       model = SentenceTransformer('distilbert-base-nli-mean-tokens')
       train_examples = [InputExample(texts=['Sentence from class 0'], label=0), InputExample(texts=['Another sentence from class 0'], label=0),
           InputExample(texts=['Sentence from class 1'], label=1), InputExample(texts=['Sentence from class 2'], label=2)]
       train_dataset = SentencesDataset(train_examples, model)
       train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
       train_loss = losses.BatchSemiHardTripletLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, distance_metric = BatchHardTripletLossDistanceFunction.eucledian_distance, margin: float = 5):
        super(BatchSemiHardTripletLoss, self).__init__()
        self.sentence_embedder = model
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep = self.sentence_embedder(sentence_features[0])['sentence_embedding']
        return self.batch_semi_hard_triplet_loss(labels, rep)


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

        pdist_matrix = self.distance_metric(embeddings)

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

        loss_mat = (pdist_matrix - semi_hard_negatives) + self.margin

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

