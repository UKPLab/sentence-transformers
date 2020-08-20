import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer

class MultipleNegativesRankingLoss(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, b_1), (a_2, b_2)..., (a_n, b_n)
        where we assume that (a_i, b_i) are a positive pair and (a_i, b_j) for i!=j a negative pair.

        For each a_i, it uses all other b_j as negative samples, i.e., for a_i, we have 1 positive example (b_i) and
        n-1 negative examples (b_j). It then minimizes the negative log-likehood for softmax normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.

        The performance usually increases with increasing batch sizes.

        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        The error function is equivalent to::

            scores = torch.matmul(embeddings_a, embeddings_b.t())
            labels = torch.tensor(range(len(scores)), dtype=torch.long).to(self.model.device) #Example a[i] should match with b[i]
            cross_entropy_loss = nn.CrossEntropyLoss()
            return cross_entropy_loss(scores, labels)

        Example::

            from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
            from sentence_transformers.readers import InputExample

            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.model = model


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        reps_a, reps_b = reps
        return self.multiple_negatives_ranking_loss(reps_a, reps_b)


    def multiple_negatives_ranking_loss(self, embeddings_a: Tensor, embeddings_b: Tensor):
        """
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



