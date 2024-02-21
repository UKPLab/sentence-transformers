from torch import Tensor
from torch import nn
from typing import Dict
import os
import json


class Dropout(nn.Module):
    """Dropout layer.

    :param dropout: Sets a dropout value for dense layer.
    """

    def __init__(self, dropout: float = 0.2):
        """
        Initialize Dropout layer.

        Parameters
        ----------
        dropout : float, optional
            The dropout probability. Default is 0.2.

        Returns
        -------
        None
        """
        super(Dropout, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, features: Dict[str, Tensor]):
        """
        Forward pass through the Dropout layer.

        Parameters
        ----------
        features : Dict[str, Tensor]
            Dictionary containing input features, including "sentence_embedding".

        Returns
        -------
        Dict[str, Tensor]
            Updated features dictionary after applying dropout.
        """
        features.update({"sentence_embedding": self.dropout_layer(features["sentence_embedding"])})

        return features

    def save(self, output_path) -> None:
        """
        Save the Dropout layer configuration to the specified output path.

        Parameters
        ----------
        output_path : str
            Path to save the configuration file.

        Returns
        -------
        None
        """
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump({"dropout": self.dropout}, fOut)

    @staticmethod
    def load(input_path):
        """
        Load a saved Dropout layer from the specified input path.

        Parameters
        ----------
        input_path : str
            Path from which to load the configuration file.

        Returns
        -------
        Dropout
            Loaded Dropout layer.
        """
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        model = Dropout(**config)
        return model
