from torch import nn, Tensor
from typing import Iterable, Dict


class MSELoss(nn.Module):
    """
    Computes the MSE loss between the computed sentence embedding and a target sentence embedding. This loss
    is used when extending sentence embeddings to new languages as described in our publication
    Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation: https://arxiv.org/abs/2004.09813

    For an example, see the documentation on extending language models to new languages.
    
    Requirements:
        - Usually uses a finetuned teacher M in a knowledge distillation setup

    Input:

    | Texts                       | Labels                      |
    | --------------------------- | --------------------------- |
    | model_1_sentence_embeddings | model_2_sentence_embeddings |
    
    """

    def __init__(self, model):
        """
        :param model: SentenceTransformerModel
        """
        super(MSELoss, self).__init__()
        self.model = model
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep = self.model(sentence_features[0])["sentence_embedding"]
        return self.loss_fct(rep, labels)
