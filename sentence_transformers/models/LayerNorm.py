import torch
from torch import Tensor
from torch import nn
from typing import Dict
import os
import json


class LayerNorm(nn.Module):
    def __init__(self, dimension: int):
        super(LayerNorm, self).__init__()
        self.dimension = dimension
        self.norm = nn.LayerNorm(dimension)

    def forward(self, features: Dict[str, Tensor]):
        features["sentence_embedding"] = self.norm(features["sentence_embedding"])
        return features

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def save(self, output_path):
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump({"dimension": self.dimension}, fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        model = LayerNorm(**config)
        model.load_state_dict(
            torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
        )
        return model
