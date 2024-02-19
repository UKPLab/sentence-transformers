from torch import Tensor
from torch import nn
from typing import Dict
import torch.nn.functional as F


class Normalize(nn.Module):
    """
    This layer normalizes embeddings to unit length
    """

    def __init__(self):
        """
        Initialize Normalize layer.

        Returns
        -------
        None
        """
        super(Normalize, self).__init__()

    def forward(self, features: Dict[str, Tensor]):
        """
        Forward pass through the Normalize layer.

        Parameters
        ----------
        features : Dict[str, Tensor]
            Dictionary containing input features, including "sentence_embedding".

        Returns
        -------
        Dict[str, Tensor]
            Updated features dictionary after normalizing embeddings.
        """
        features.update({"sentence_embedding": F.normalize(features["sentence_embedding"], p=2, dim=1)})
        return features

    def save(self, output_path):
        """
        Save the Normalize layer configuration to the specified output path.

        Parameters
        ----------
        output_path : str
            Path to save the configuration file.

        Returns
        -------
        None
        """
        pass

    @staticmethod
    def load(input_path):
        """
        Load a saved Normalize layer from the specified input path.

        Parameters
        ----------
        input_path : str
            Path from which to load the configuration file.

        Returns
        -------
        Normalize
            Loaded Normalize layer.
        """
        return Normalize()
