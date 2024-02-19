import torch
from torch import nn
from typing import List
import os
import json


class LSTM(nn.Module):
    """
    Bidirectional LSTM running over word embeddings.
    """

    def __init__(
        self,
        word_embedding_dimension: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0,
        bidirectional: bool = True,
    ):
        """
        Initialize Bidirectional LSTM layer.

        Parameters
        ----------
        word_embedding_dimension : int
            The dimensionality of the input word embeddings.

        hidden_dim : int
            The number of features in the hidden state.

        num_layers : int, optional
            Number of recurrent layers. Default is 1.

        dropout : float, optional
            If non-zero, introduces a dropout layer on the outputs of each LSTM layer except the last layer. Default is 0.
        
        bidirectional : bool, optional
            If True, becomes a bidirectional LSTM. Default is True.

        Returns
        -------
        None
        """
        nn.Module.__init__(self)
        self.config_keys = ["word_embedding_dimension", "hidden_dim", "num_layers", "dropout", "bidirectional"]
        self.word_embedding_dimension = word_embedding_dimension
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embeddings_dimension = hidden_dim
        if self.bidirectional:
            self.embeddings_dimension *= 2

        self.encoder = nn.LSTM(
            word_embedding_dimension,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, features):
        """
        Forward pass through the Bidirectional LSTM layer.

        Parameters
        ----------
        features : Dict[str, Tensor]
            Dictionary containing input features, including "token_embeddings" and "sentence_lengths".

        Returns
        -------
        Dict[str, Tensor]
            Updated features dictionary after passing through the LSTM layer.
        """
        token_embeddings = features["token_embeddings"]
        sentence_lengths = torch.clamp(features["sentence_lengths"], min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            token_embeddings, sentence_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed = self.encoder(packed)
        unpack = nn.utils.rnn.pad_packed_sequence(packed[0], batch_first=True)[0]
        features.update({"token_embeddings": unpack})
        
        return features

    def get_word_embedding_dimension(self) -> int:
        """
        Get the dimensionality of the word embeddings produced by the Bidirectional LSTM layer.

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

    def save(self, output_path: str):
        """
        Save the Bidirectional LSTM model and configuration to the specified output path.

        Parameters
        ----------
        output_path : str
            Path to save the model and configuration files.

        Returns
        -------
        None
        """
        with open(os.path.join(output_path, "lstm_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    def get_config_dict(self):
        """
        Get the configuration dictionary of the Bidirectional LSTM layer.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        return {key: self.__dict__[key] for key in self.config_keys}

    @staticmethod
    def load(input_path: str):
        """
        Load a saved Bidirectional LSTM model from the specified input path.

        Parameters
        ----------
        input_path : str
            Path from which to load the model.

        Returns
        -------
        LSTM
            Loaded Bidirectional LSTM model.
        """
        with open(os.path.join(input_path, "lstm_config.json"), "r") as fIn:
            config = json.load(fIn)

        weights = torch.load(os.path.join(input_path, "pytorch_model.bin"))
        model = LSTM(**config)
        model.load_state_dict(weights)
        return model
