import torch
from torch import nn
from typing import List
import os
import json


class CNN(nn.Module):
    """CNN-layer with multiple kernel-sizes over the word embeddings"""

    def __init__(
        self,
        in_word_embedding_dimension: int,
        out_channels: int = 256,
        kernel_sizes: List[int] = [1, 3, 5],
        stride_sizes: List[int] = None,
    ):
        """
        Initialize CNN layer.

        Parameters
        ----------
        in_word_embedding_dimension : int
            The dimensionality of the input word embeddings.
            
        out_channels : int, optional
            The number of output channels (filters). Default is 256.
            
        kernel_sizes : List[int], optional
            A list of kernel sizes for the convolutional layers. Default is [1, 3, 5].
            
        stride_sizes : List[int], optional
            A list of stride sizes for the convolutional layers. Default is None.

        Returns
        -------
        None
        """
        nn.Module.__init__(self)
        self.config_keys = ["in_word_embedding_dimension", "out_channels", "kernel_sizes"]
        self.in_word_embedding_dimension = in_word_embedding_dimension
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.embeddings_dimension = out_channels * len(kernel_sizes)
        self.convs = nn.ModuleList()

        in_channels = in_word_embedding_dimension
        if stride_sizes is None:
            stride_sizes = [1] * len(kernel_sizes)

        for kernel_size, stride in zip(kernel_sizes, stride_sizes):
            padding_size = int((kernel_size - 1) / 2)
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding_size,
            )
            self.convs.append(conv)

    def forward(self, features):
        """
        Forward pass through the CNN layer.

        Parameters
        ----------
        features : dict
            Dictionary containing input features, including "token_embeddings".

        Returns
        -------
        dict
            Updated features dictionary after passing through the CNN layer.
        """
        token_embeddings = features["token_embeddings"]

        token_embeddings = token_embeddings.transpose(1, -1)
        vectors = [conv(token_embeddings) for conv in self.convs]
        out = torch.cat(vectors, 1).transpose(1, -1)

        features.update({"token_embeddings": out})
        return features

    def get_word_embedding_dimension(self) -> int:
        """
        Get the dimensionality of the word embeddings produced by the CNN layer.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The dimensionality of the word embeddings.
        """
        return self.embeddings_dimension

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize input text.

        Parameters
        ----------
        text : str
            Input text to tokenize.

        Returns
        -------
        List[int]
            List of tokenized integers.
        """
        raise NotImplementedError()

    def save(self, output_path: str) -> None:
        """
        Save the CNN model and configuration to the specified output path.

        Parameters
        ----------
        output_path : str
            Path to save the model and configuration files.

        Returns
        -------
        None
        """
        with open(os.path.join(output_path, "cnn_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    def get_config_dict(self) -> dict:
        """
        Get the configuration dictionary of the CNN layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        return {key: self.__dict__[key] for key in self.config_keys}

    @staticmethod
    def load(input_path: str):
        """
        Load a saved CNN model from the specified input path.

        Parameters
        ----------
        input_path : str
            Path from which to load the model.

        Returns
        -------
        CNN
            Loaded CNN model.
        """
        with open(os.path.join(input_path, "cnn_config.json"), "r") as fIn:
            config = json.load(fIn)

        weights = torch.load(os.path.join(input_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
        model = CNN(**config)
        model.load_state_dict(weights)
        return model
