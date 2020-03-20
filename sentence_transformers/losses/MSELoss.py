import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict


class MSELoss(nn.Module):
    """
    Computes the MSE loss between the computed sentence embedding and a target sentence embedding
    """
    def __init__(self, model):
        super(MSELoss, self).__init__()
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep = self.model(sentence_features[0])['sentence_embedding']
        loss_fct = nn.MSELoss()
        loss = loss_fct(rep, labels)
        return loss
