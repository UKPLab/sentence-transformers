import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, sentence_embedder):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.sentence_embedder = sentence_embedder

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.sentence_embedder(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        reps_a, reps_b = reps
        return self.multiple_negatives_ranking_loss(reps_a, reps_b)

    # Multiple Negatives Ranking Loss
    # Paper: https://arxiv.org/pdf/1705.00652.pdf
    #   Efficient Natural Language Response Suggestion for Smart Reply
    #   Section 4.4
    def multiple_negatives_ranking_loss(self, embeddings_a: Tensor, embeddings_b: Tensor):
        """
        Compute the loss over a batch with two embeddings per example.

        Each pair is a positive example. The negative examples are all other embeddings in embeddings_b with each embedding
        in embedding_a.

        See the paper for more information: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        :param embeddings_a:
            Tensor of shape (batch_size, embedding_dim)
        :param embeddings_b:
            Tensor of shape (batch_size, embedding_dim)
        :return:
            The scalar loss
        """
        scores = torch.matmul(embeddings_a, embeddings_b.t())
        diagonal_mean = torch.mean(torch.diag(scores))
        mean_log_row_sum_exp = torch.mean(torch.logsumexp(scores, dim=1))
        return -diagonal_mean + mean_log_row_sum_exp
