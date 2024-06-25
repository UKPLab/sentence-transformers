from typing import Dict

import torch.nn.functional as F
from torch import Tensor, nn


class Normalize(nn.Module):
    """This layer normalizes embeddings to unit length"""

    def __init__(self) -> None:
        super(Normalize, self).__init__()

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        features.update({"sentence_embedding": F.normalize(features["sentence_embedding"], p=2, dim=1)})
        return features

    def save(self, output_path) -> None:
        pass

    @staticmethod
    def load(input_path) -> "Normalize":
        return Normalize()
