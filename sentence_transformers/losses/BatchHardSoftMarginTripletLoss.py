import torch
from torch import Tensor
from typing import Iterable, Dict
from .BatchHardTripletLoss import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction
from sentence_transformers.SentenceTransformer import SentenceTransformer


class BatchHardSoftMarginTripletLoss(BatchHardTripletLoss):
    """
    BatchHardSoftMarginTripletLoss takes a batch with (label, sentence) pairs and computes the loss for all possible, valid
    triplets, i.e., anchor and positive must have the same label, anchor and negative a different label. The labels
    must be integers, with same label indicating sentences from the same class. You train dataset
    must contain at least 2 examples per label class. The margin is computed automatically.

    Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    Blog post: https://omoindrot.github.io/triplet-loss

    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two emeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used


    Example::

       from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
       from sentence_transformers.readers import InputExample

       model = SentenceTransformer('distilbert-base-nli-mean-tokens')
       train_examples = [InputExample(texts=['Sentence from class 0'], label=0), InputExample(texts=['Another sentence from class 0'], label=0),
           InputExample(texts=['Sentence from class 1'], label=1), InputExample(texts=['Sentence from class 2'], label=2)]
       train_dataset = SentencesDataset(train_examples, model)
       train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
       train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
    """

    def __init__(
        self, model: SentenceTransformer, distance_metric=BatchHardTripletLossDistanceFunction.eucledian_distance
    ):
        super(BatchHardSoftMarginTripletLoss, self).__init__(model)
        self.sentence_embedder = model
        self.distance_metric = distance_metric

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep = self.sentence_embedder(sentence_features[0])["sentence_embedding"]
        return self.batch_hard_triplet_soft_margin_loss(labels, rep)

    # Hard Triplet Loss with Soft Margin
    # Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    def batch_hard_triplet_soft_margin_loss(self, labels: Tensor, embeddings: Tensor) -> Tensor:
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            Label_Sentence_Triplet: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self.distance_metric(embeddings)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss with soft margin
        # tl = hardest_positive_dist - hardest_negative_dist + margin
        # tl[tl < 0] = 0
        tl = torch.log1p(torch.exp(hardest_positive_dist - hardest_negative_dist))
        triplet_loss = tl.mean()

        return triplet_loss
