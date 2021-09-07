import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict


class MSELoss(nn.Module):
    """
    Computes the MSE loss between the computed sentence embedding and a target sentence embedding. This loss
    is used when extending sentence embeddings to new languages as described in our publication
    Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation: https://arxiv.org/abs/2004.09813

    For an example, see the documentation on extending language models to new languages.
    """
    def __init__(self, model):
        """
        :param model: SentenceTransformerModel
        """
        super(MSELoss, self).__init__()
        self.model = model
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep = self.model(sentence_features[0])['sentence_embedding']
        return self.loss_fct(rep, labels)
