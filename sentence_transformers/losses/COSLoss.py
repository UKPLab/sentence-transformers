import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict


class COSLoss(nn.Module):
    """
    Computes the cosine similarity between the computed sentence embedding and a target sentence embedding. 
    The final loss is then computed with MSE between the returned similarity and label of 1.0.
    This loss should work similar to the MSELoss, but instead of MSE minimization between sentence embedding and target, 
    cosinus similarity is used.
    For an example, see the documentation on extending language models to new languages.
    """
    def __init__(self, model, loss_fct = nn.MSELoss()):
        super(COSLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep = self.model(sentence_features[0])['sentence_embedding']
        cos_sim = torch.cosine_similarity(rep, labels)
        float_labs = torch.Tensor([1.0 for x in cos_sim]).cuda()

        return self.loss_fct(cos_sim, float_labs)
