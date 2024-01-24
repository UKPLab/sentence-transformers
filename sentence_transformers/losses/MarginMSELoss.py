from .. import util
from torch import nn, Tensor
from typing import Iterable, Dict


class MarginMSELoss(nn.Module):
    """
    Compute the MSE loss between the ``|sim(Query, Pos) - sim(Query, Neg)|`` and ``|gold_sim(Query, Pos) - gold_sim(Query, Neg)|``.
    By default, sim() is the dot-product.
    For more details, please refer to https://arxiv.org/abs/2010.02666.
    
    Requirements:
        1. (query, positive, negative) triplets
        2. Usually used with a finetuned teacher M in a knowledge distillation setup

    Relations:
        \ \
        - equivalent to `MSELoss` but with a margin
    
    Inputs:

    +-----------------------------------------------+-----------------------------------------+
    | Texts                                         | Labels                                  |
    +===============================================+=========================================+
    | (query, positive, negative) triplets          | M(query, positive) - M(query, negative) |
    +-----------------------------------------------+-----------------------------------------+

    Example::

        from sentence_transformers import SentenceTransformer, InputExample, losses
        from sentence_transformers.util import pairwise_dot_score
        from torch.utils.data import DataLoader
        import torch

        model1 = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')
        model2 = SentenceTransformer('sentence-transformers/bert-base-nli-stsb-mean-tokens')

        train_examples = [
            ['The first query',  'The first positive passage',  'The first negative passage'],
            ['The second query', 'The second positive passage', 'The second negative passage'],
            ['The third query',  'The third positive passage',  'The third negative passage'],
        ]
        train_batch_size = 1
        encoded = torch.tensor([model2.encode(x).tolist() for x in train_examples])
        labels = pairwise_dot_score(encoded[:, 0], encoded[:, 1]) - pairwise_dot_score(encoded[:, 0], encoded[:, 2])

        train_input_examples = [InputExample(texts=x, label=labels[i]) for i, x in enumerate(train_examples)]
        train_dataloader = DataLoader(train_input_examples, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.MarginMSELoss(model=model1)

        model1.fit(
            [(train_dataloader, train_loss)],
            epochs=10,
        )
    """

    def __init__(self, model, similarity_fct=util.pairwise_dot_score):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use.
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
