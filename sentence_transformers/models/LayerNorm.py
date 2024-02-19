import torch
from torch import Tensor
from torch import nn
from typing import Dict
import os
import json


class LayerNorm(nn.Module):
    """
    Layer normalization module for normalizing sentence embeddings.

    Parameters
    ----------
    dimension : int
        Dimension of the input embeddings.

    Methods
    -------
    forward(features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        Forward pass of the LayerNorm module.

    get_sentence_embedding_dimension() -> int:
        Get the dimension of the sentence embeddings.

    save(output_path: str):
        Save the LayerNorm model to the specified output path.

    @staticmethod
    load(input_path: str) -> 'LayerNorm':
        Load the LayerNorm model from the specified input path.
    """
    def __init__(self, dimension: int):
        """
        Initialize the LayerNorm module.

        Parameters
        ----------
        dimension : int
            Dimension of the input embeddings.
        """
        super(LayerNorm, self).__init__()
        self.dimension = dimension
        self.norm = nn.LayerNorm(dimension)

    def forward(self, features: Dict[str, Tensor]):
        """
        Forward pass of the LayerNorm module.

        Parameters
        ----------
        features : Dict[str, Tensor]
            Dictionary containing input features with key 'sentence_embedding'.

        Returns
        -------
        Dict[str, Tensor]
            Dictionary containing output features with key 'sentence_embedding'.
        """
        features["sentence_embedding"] = self.norm(features["sentence_embedding"])
        return features

    def get_sentence_embedding_dimension(self):
        """
        Get the dimension of the sentence embeddings.

        Returns
        -------
        int
            Dimension of the sentence embeddings.
        """
        return self.dimension

    def save(self, output_path):
        """
        Save the LayerNorm model to the specified output path.

        Parameters
        ----------
        output_path : str
            Path to save the model.
        """
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump({"dimension": self.dimension}, fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path):
        """
        Load the LayerNorm model from the specified input path.

        Parameters
        ----------
        input_path : str
            Path from which to load the model.

        Returns
        -------
        LayerNorm
            Loaded LayerNorm model.
        """
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        model = LayerNorm(**config)
        model.load_state_dict(
            torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
        )
        return model
