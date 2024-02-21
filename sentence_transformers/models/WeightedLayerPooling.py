import torch
from torch import Tensor
from torch import nn
from typing import Dict
import os
import json


class WeightedLayerPooling(nn.Module):
    """
    Token embeddings are weighted mean of their different hidden layer representations.

    Parameters
    ----------
    word_embedding_dimension : int
        The dimensionality of the word embeddings.

    num_hidden_layers : int, optional
        The total number of hidden layers in the model. Default is 12.

    layer_start : int, optional
        The starting index of the hidden layers to consider for pooling. Default is 4.

    layer_weights : Tensor, optional
        The weights assigned to each hidden layer for computing the weighted average. If None, weights are initialized
        as equal values. Default is None.

    Methods
    -------
    forward(features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        Forward pass of the weighted layer pooling module.

    get_word_embedding_dimension() -> int:
        Get the dimensionality of the word embeddings.

    get_config_dict() -> Dict[str, int]:
        Get the configuration dictionary of the weighted layer pooling module.

    save(output_path: str):
        Save the weighted layer pooling module to the specified output path.

    @staticmethod
    load(input_path: str) -> 'WeightedLayerPooling':
        Load the weighted layer pooling module from the specified input path.
    """
    def __init__(
        self, word_embedding_dimension, num_hidden_layers: int = 12, layer_start: int = 4, layer_weights=None
    ):
        """
        Initialize the WeightedLayerPooling module.

        Parameters
        ----------
        word_embedding_dimension : int
            The dimensionality of the word embeddings.

        num_hidden_layers : int, optional
            The total number of hidden layers in the model. Default is 12.

        layer_start : int, optional
            The starting index of the hidden layers to consider for pooling. Default is 4.

        layer_weights : Tensor, optional
            The weights assigned to each hidden layer for computing the weighted average. If None, weights are initialized
            as equal values. Default is None.
        """
        super(WeightedLayerPooling, self).__init__()
        self.config_keys = ["word_embedding_dimension", "layer_start", "num_hidden_layers"]
        self.word_embedding_dimension = word_embedding_dimension
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float))
        )

    def forward(self, features: Dict[str, Tensor]):
        """
        Forward pass of the weighted layer pooling module.

        Parameters
        ----------
        features : Dict[str, Tensor]
            Input features containing the embeddings of all layers.

        Returns
        -------
        Dict[str, Tensor]
            Output features containing the token embeddings after weighted layer pooling.
        """
        ft_all_layers = features["all_layer_embeddings"]

        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start :, :, :, :]  # Start from 4th layers output

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        features.update({"token_embeddings": weighted_average})
        return features

    def get_word_embedding_dimension(self):
        """
        Get the dimensionality of the word embeddings.

        Returns
        -------
        int
            Dimensionality of the word embeddings.
        """
        return self.word_embedding_dimension

    def get_config_dict(self):
        """
        Get the configuration dictionary of the weighted layer pooling module.

        Returns
        -------
        Dict[str, int]
            Configuration dictionary containing word_embedding_dimension, num_hidden_layers, and layer_start.
        """
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path:str) -> None:
        """
        Save the weighted layer pooling module to the specified output path.

        Parameters
        ----------
        output_path : str
            Path to save the module.
        
        Returns
        -------
        None
        """
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path:str):
        """
        Load the weighted layer pooling module from the specified input path.

        Parameters
        ----------
        input_path : str
            Path from which to load the module.

        Returns
        -------
        WeightedLayerPooling
            Loaded weighted layer pooling module.
        """
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        model = WeightedLayerPooling(**config)
        model.load_state_dict(
            torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
        )
        return model
