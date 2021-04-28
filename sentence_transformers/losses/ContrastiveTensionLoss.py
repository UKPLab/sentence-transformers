import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
import copy


class ContrastiveTensionLoss(nn.Module):
    """
        This loss expects as input a batch consisting of multiple mini-batches of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_{K+1}, p_{K+1})
        where p_1 = a_1 = a_2 = ... a_{K+1} and p_2, p_3, ..., p_{K+1} are expected to be different from p_1 (this is done via random sampling).
        The corresponding labels y_1, y_2, ..., y_{K+1} for each mini-batch are assigned as: y_i = 1 if i == 1 and y_i = 0 otherwise.
        In other words, K represent the number of negative pairs and the positive pair is actually made of two identical sentences. The data generation
        process has already been implemented in readers/ContrastiveTensionReader.py
        For tractable optimization, two independent encoders ('model1' and 'model2') are created for encoding a_i and p_i, respectively. For inference,
        only model2 are used, which gives better performance. The training objective is binary cross entropy.
        For more information, see: https://openreview.net/pdf?id=Ov_sMNau-PF

    """
    def __init__(self, model: SentenceTransformer):
        """
        :param model: SentenceTransformer model
        """
        super(ContrastiveTensionLoss, self).__init__()
        self.model2 = model  # This will be the final model used during the inference time.
        self.model1 = copy.deepcopy(model)
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        sentence_features1, sentence_features2 = tuple(sentence_features)
        reps_1 = self.model1(sentence_features1)['sentence_embedding']  # (bsz, hdim)
        reps_2 = self.model2(sentence_features2)['sentence_embedding']

        sim_scores = torch.matmul(reps_1[:,None], reps_2[:,:,None]).squeeze(-1).squeeze(-1)  # (bsz,) dot product, i.e. S1S2^T

        loss = self.criterion(sim_scores, labels.type_as(sim_scores))
        return loss
