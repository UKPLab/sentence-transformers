import torch
from torch import nn, Tensor
from typing import Iterable, Dict, Callable
from ..SentenceTransformer import SentenceTransformer
import logging


logger = logging.getLogger(__name__)


class SoftmaxLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        sentence_embedding_dimension: int,
        num_labels: int,
        concatenation_sent_rep: bool = True,
        concatenation_sent_difference: bool = True,
        concatenation_sent_multiplication: bool = False,
        loss_fct: Callable = nn.CrossEntropyLoss(),
    ):
        """
        This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
        model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

        :class:`MultipleNegativesRankingLoss` is an alternative loss function that often yields better results,
        as per https://arxiv.org/abs/2004.09813.

        :param model: SentenceTransformer model
        :param sentence_embedding_dimension: Dimension of your sentence embeddings
        :param num_labels: Number of different labels
        :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
        :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
        :param concatenation_sent_multiplication: Add u*v for the softmax classifier?
        :param loss_fct: Optional: Custom pytorch loss function. If not set, uses nn.CrossEntropyLoss()

        References:
            - Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks: https://arxiv.org/abs/1908.10084
            - `Training Examples > Natural Language Inference <../../examples/training/nli/README.html>`_

        Requirements:
            1. sentence pairs with a class label

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (sentence_A, sentence_B) pairs        | class  |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentencesDataset, losses
                from sentence_transformers.readers import InputExample
                from torch.utils.data import DataLoader

                model = SentenceTransformer('distilbert-base-nli-mean-tokens')
                train_examples = [
                    InputExample(texts=['First pair, sent A',  'First pair, sent B'], label=0),
                    InputExample(texts=['Second pair, sent A', 'Second pair, sent B'], label=1),
                    InputExample(texts=['Third pair, sent A',  'Third pair, sent B'], label=0),
                    InputExample(texts=['Fourth pair, sent A', 'Fourth pair, sent B'], label=2),
                ]
                train_batch_size = 2
                train_dataset = SentencesDataset(train_examples, model)
                train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
                train_loss = losses.SoftmaxLoss(
                    model=model,
                    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                    num_labels=len(set(x.label for x in train_examples))
                )
                model.fit(
                    [(train_dataloader, train_loss)],
                    epochs=10,
                )
        """
        super(SoftmaxLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        logger.info("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        self.classifier = nn.Linear(
            num_vectors_concatenated * sentence_embedding_dimension, num_labels, device=model.device
        )
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output
