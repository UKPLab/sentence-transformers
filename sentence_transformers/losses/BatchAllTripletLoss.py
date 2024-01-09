from torch import nn, Tensor
from typing import Iterable, Dict
from .BatchHardTripletLoss import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction
from sentence_transformers.SentenceTransformer import SentenceTransformer


class BatchAllTripletLoss(nn.Module):
    """
    BatchAllTripletLoss takes a batch with (label, sentence) pairs and computes the loss for all possible, valid
    triplets, i.e., anchor and positive must have the same label, anchor and negative a different label. The labels
    must be integers, with same label indicating sentences from the same class. You train dataset
    must contain at least 2 examples per label class.

    | Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    | Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    | Blog post: https://omoindrot.github.io/triplet-loss

    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two emeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples should be at least margin further apart from the anchor than the positive.

    Example::

        from sentence_transformers import SentenceTransformer, SentencesDataset, losses
        from sentence_transformers.readers import InputExample

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['Sentence from class 0'], label=0), InputExample(texts=['Another sentence from class 0'], label=0),
            InputExample(texts=['Sentence from class 1'], label=1), InputExample(texts=['Sentence from class 2'], label=2)]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.BatchAllTripletLoss(model=model)
    """

    def __init__(
        self,
        model: SentenceTransformer,
        distance_metric=BatchHardTripletLossDistanceFunction.eucledian_distance,
        margin: float = 5,
    ):
        super(BatchAllTripletLoss, self).__init__()
        self.sentence_embedder = model
        self.triplet_margin = margin
        self.distance_metric = distance_metric

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep = self.sentence_embedder(sentence_features[0])["sentence_embedding"]
        return self.batch_all_triplet_loss(labels, rep)

    def batch_all_triplet_loss(self, labels, embeddings):
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
        pairwise_dist = self.distance_metric(embeddings)

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.triplet_margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = BatchHardTripletLoss.get_triplet_mask(labels)
        triplet_loss = mask.float() * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss[triplet_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        # num_valid_triplets = mask.sum()
        # fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss
