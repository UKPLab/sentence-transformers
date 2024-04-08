from .. import util
from torch import nn, Tensor
from typing import Iterable, Dict


class MarginMSELoss(nn.Module):
    def __init__(self, model, similarity_fct=util.pairwise_dot_score):
        """
        Compute the MSE loss between the ``|sim(Query, Pos) - sim(Query, Neg)|`` and ``|gold_sim(Query, Pos) - gold_sim(Query, Neg)|``.
        By default, sim() is the dot-product. The gold_sim is often the similarity score from a teacher model.

        In contrast to :class:`MultipleNegativesRankingLoss`, the two passages do not have to be strictly positive and negative,
        both can be relevant or not relevant for a given query. This can be an advantage of MarginMSELoss over
        MultipleNegativesRankingLoss, but note that the MarginMSELoss is much slower to train. With MultipleNegativesRankingLoss,
        with a batch size of 64, we compare one query against 128 passages. With MarginMSELoss, we compare a query only
        against two passages.

        :param model: SentenceTransformerModel
        :param similarity_fct: Which similarity function to use.

        References:
            - For more details, please refer to https://arxiv.org/abs/2010.02666.
            - `Training Examples > MS MARCO <../../examples/training/ms_marco/README.html>`_
            - `Unsupervised Learning > Domain Adaptation <../../examples/domain_adaptation/README.html>`_

        Requirements:
            1. (query, passage_one, passage_two) triplets
            2. Usually used with a finetuned teacher M in a knowledge distillation setup

        Relations:
            - :class:`MSELoss` is equivalent to this loss, but without a margin through the negative pair.

        Inputs:
            +-----------------------------------------------+-----------------------------------------------+
            | Texts                                         | Labels                                        |
            +===============================================+===============================================+
            | (query, passage_one, passage_two) triplets    | M(query, passage_one) - M(query, passage_two) |
            +-----------------------------------------------+-----------------------------------------------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, InputExample, losses
                from sentence_transformers.util import pairwise_dot_score
                from torch.utils.data import DataLoader
                import torch

                student_model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')
                teacher_model = SentenceTransformer('sentence-transformers/bert-base-nli-stsb-mean-tokens')

                train_examples = [
                    ['The first query',  'The first positive passage',  'The first negative passage'],
                    ['The second query', 'The second positive passage', 'The second negative passage'],
                    ['The third query',  'The third positive passage',  'The third negative passage'],
                ]
                train_batch_size = 1
                encoded = torch.tensor([teacher_model.encode(x).tolist() for x in train_examples])
                labels = pairwise_dot_score(encoded[:, 0], encoded[:, 1]) - pairwise_dot_score(encoded[:, 0], encoded[:, 2])

                train_input_examples = [InputExample(texts=x, label=labels[i]) for i, x in enumerate(train_examples)]
                train_dataloader = DataLoader(train_input_examples, shuffle=True, batch_size=train_batch_size)
                train_loss = losses.MarginMSELoss(model=student_model)

                student_model.fit(
                    [(train_dataloader, train_loss)],
                    epochs=10,
                )
        """
        super(MarginMSELoss, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]

        scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
        scores_neg = self.similarity_fct(embeddings_query, embeddings_neg)
        margin_pred = scores_pos - scores_neg

        return self.loss_fct(margin_pred, labels)
