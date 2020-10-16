from .. import util
import torch
from torch import nn, Tensor
from typing import Iterable, Dict
import torch.nn.functional as F

class MegaBatchMarginLoss(nn.Module):
    """
    Loss function inspired from ParaNMT paper:
    https://www.aclweb.org/anthology/P18-1042/

    Given a large batch (like 500 or more examples) of (anchor_i, positive_i) pairs,
    find for each pair in the batch the hardest negative, i.e. find j != i such that cos_sim(anchor_i, positive_j)
    is maximal. Then create from this a triplet (anchor_i, positive_i, positive_j) where positive_j
    serves as the negative for this triplet.

    Train than as with the triplet loss
    """

    def __init__(self, model, positive_margin: float = 0.8, negative_margin: float = 0.3, use_mini_batched_version: bool = True, mini_batch_size: bool = 50):
        """
        :param model: SentenceTransformerModel
        :param positive_margin: Positive margin, cos(anchor, positive) should be > positive_margin
        :param negative_margin: Negative margin, cos(anchor, negative) should be < negative_margin
        :param use_mini_batched_version: As large batch sizes require a lot of memory, we can use a mini-batched version. We break down the large batch with 500 examples to smaller batches with fewer examples.
        :param mini_batch_size: Size for the mini-batches. Should be a devisor for the batch size in your data loader.
        """
        super(MegaBatchMarginLoss, self).__init__()
        self.model = model
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.mini_batch_size = mini_batch_size
        self.forward = self.forward_mini_batched if use_mini_batched_version else self.forward_non_mini_batched


    def forward_mini_batched(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        anchor, positive = sentence_features
        feature_names = list(anchor.keys())

        with torch.no_grad():
            self.model.eval()
            all_positive_emb = self.model(positive)['sentence_embedding'].detach()
            self.model.train()

        diagonal_matrix = torch.eye(len(all_positive_emb), len(all_positive_emb), device=all_positive_emb.device)

        #Iterate over the triplets (anchor, positive, hardest_negative) in smaller mini_batch sizes
        for start_idx in range(0, len(all_positive_emb), self.mini_batch_size):
            end_idx = start_idx + self.mini_batch_size
            anchor_emb = self.model({key: anchor[key][start_idx:end_idx] for key in feature_names})['sentence_embedding']

            # Find hard negatives. For each anchor, find the hardest negative
            # Store them in the triplets (anchor, positive, hardest_negative)
            hard_negative_features = {key: [] for key in feature_names}
            with torch.no_grad():
                cos_scores = util.pytorch_cos_sim(anchor_emb, all_positive_emb)
                negative_scores = cos_scores - 2 * diagonal_matrix[start_idx:end_idx]  # Remove positive scores along the diagonal, set them to -1 so that they are not selected by the max() operation
                negatives_max, negatives_ids = torch.max(negative_scores, dim=1)

            for hard_negative_id in negatives_ids:
                for key in feature_names:
                    hard_negative_features[key].append(positive[key][hard_negative_id])

            for key in feature_names:
                hard_negative_features[key] = torch.stack(hard_negative_features[key])


            #Compute differentiable negative and positive embeddings
            positive_emb = self.model({key: positive[key][start_idx:end_idx] for key in feature_names})['sentence_embedding']
            negative_emb = self.model(hard_negative_features)['sentence_embedding']

            assert anchor_emb.shape == positive_emb.shape
            assert anchor_emb.shape == negative_emb.shape

            #Compute loss
            pos_cosine = F.cosine_similarity(anchor_emb, positive_emb)
            neg_cosine = F.cosine_similarity(anchor_emb, negative_emb)
            losses = F.relu(self.positive_margin - pos_cosine) + F.relu(neg_cosine - self.negative_margin)
            losses = losses.mean()

            #Backpropagate unless it is the last mini batch. The last mini-batch will be back propagated by the outside train loop
            if end_idx < len(cos_scores):
                losses.backward()

        return losses


    ##### Non mini-batched version ###
    def forward_non_mini_batched(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a, embeddings_b = reps

        cos_scores = util.pytorch_cos_sim(embeddings_a, embeddings_b)
        positive_scores = torch.diagonal(cos_scores)
        negative_scores = cos_scores - (2*torch.eye(*cos_scores.shape, device=cos_scores.device))  # Remove positive scores along the diagonal
        negatives_max, _ = torch.max(negative_scores, dim=1)
        losses = F.relu(self.positive_margin - positive_scores) + F.relu(negatives_max - self.negative_margin)
        return losses.mean()
