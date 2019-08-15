import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from ..SentenceTransformer import SentenceTransformer

class CosineSimilarityLoss(nn.Module):
    def __init__(self, model: SentenceTransformer):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        output = torch.cosine_similarity(rep_a, rep_b)
        loss_fct = nn.MSELoss()

        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output