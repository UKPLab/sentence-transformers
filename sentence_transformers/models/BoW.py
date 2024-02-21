import torch
from torch import Tensor
from torch import nn
from typing import List, Dict
import os
import json
import logging
import numpy as np
from .tokenizer import WhitespaceTokenizer


logger = logging.getLogger(__name__)


class BoW(nn.Module):
    """
    Implements a Bag-of-Words (BoW) model to derive sentence embeddings.

    A weighting can be added to allow the generation of tf-idf vectors. The output vector has the size of the vocab.

    Parameters
    ----------
    vocab : List[str]
        Vocabulary list.

    word_weights : Dict[str, float], optional
        Dictionary mapping words to their weights. Default is an empty dictionary.

    unknown_word_weight : float, optional
        Weight assigned to unknown words. Default is 1.

    cumulative_term_frequency : bool, optional
        If True, cumulative term frequency is used. If False, only the last occurrence of a term is considered.
        Default is True.

    Methods
    -------
    forward(features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        Forward pass of the BoW model.

    tokenize(texts: List[str]) -> List[int]:
        Tokenizes a list of texts.

    get_sentence_embedding_dimension() -> int:
        Get the dimension of the sentence embeddings.

    get_sentence_features(tokenized_texts: List[List[int]], pad_seq_length: int = 0) -> Dict[str, Tensor]:
        Get sentence features from tokenized texts.

    get_config_dict() -> Dict[str, Union[List[str], Dict[str, float], float, bool]]:
        Get the configuration dictionary.

    save(output_path: str):
        Save the BoW model to the specified output path.

    @staticmethod
    load(input_path: str) -> 'BoW':
        Load the BoW model from the specified input path.
    """
    def __init__(
        self,
        vocab: List[str],
        word_weights: Dict[str, float] = {},
        unknown_word_weight: float = 1,
        cumulative_term_frequency: bool = True,
    ):
        """
        Initialize the BoW model.

        Parameters
        ----------
        vocab : List[str]
            Vocabulary list.

        word_weights : Dict[str, float], optional
            Dictionary mapping words to their weights. Default is an empty dictionary.

        unknown_word_weight : float, optional
            Weight assigned to unknown words. Default is 1.

        cumulative_term_frequency : bool, optional
            If True, cumulative term frequency is used. If False, only the last occurrence of a term is considered.
            Default is True.
        """
        super(BoW, self).__init__()
        vocab = list(set(vocab))  # Ensure vocab is unique
        self.config_keys = ["vocab", "word_weights", "unknown_word_weight", "cumulative_term_frequency"]
        self.vocab = vocab
        self.word_weights = word_weights
        self.unknown_word_weight = unknown_word_weight
        self.cumulative_term_frequency = cumulative_term_frequency

        # Maps wordIdx -> word weight
        self.weights = []
        num_unknown_words = 0
        for word in vocab:
            weight = unknown_word_weight
            if word in word_weights:
                weight = word_weights[word]
            elif word.lower() in word_weights:
                weight = word_weights[word.lower()]
            else:
                num_unknown_words += 1
            self.weights.append(weight)

        logger.info(
            "{} out of {} words without a weighting value. Set weight to {}".format(
                num_unknown_words, len(vocab), unknown_word_weight
            )
        )

        self.tokenizer = WhitespaceTokenizer(vocab, stop_words=set(), do_lower_case=False)
        self.sentence_embedding_dimension = len(vocab)

    def forward(self, features: Dict[str, Tensor]):
        """
        Forward pass of the BoW model.

        Parameters
        ----------
        features : Dict[str, Tensor]
            Input features with key-value pairs.

        Returns
        -------
        Dict[str, Tensor]
            Output features with key-value pairs.
        """
        # Nothing to do, everything is done in get_sentence_features
        return features

    def tokenize(self, texts: List[str]) -> List[int]:
        """
        Tokenizes a list of texts.

        Parameters
        ----------
        texts : List[str]
            List of texts to be tokenized.

        Returns
        -------
        List[int]
            List of token IDs.
        """
        tokenized = [self.tokenizer.tokenize(text) for text in texts]
        return self.get_sentence_features(tokenized)

    def get_sentence_embedding_dimension(self):
        """
        Get the dimension of the sentence embeddings.

        Returns
        -------
        int
            Dimension of the sentence embeddings.
        """
        return self.sentence_embedding_dimension

    def get_sentence_features(self, tokenized_texts: List[List[int]], pad_seq_length: int = 0):
        """
        Get sentence features from tokenized texts.

        Parameters
        ----------
        tokenized_texts : List[List[int]]
            List of tokenized texts.

        pad_seq_length : int, optional
            Padding sequence length. Default is 0.

        Returns
        -------
        Dict[str, Tensor]
            Dictionary containing sentence embeddings.
        """
        vectors = []

        for tokens in tokenized_texts:
            vector = np.zeros(self.get_sentence_embedding_dimension(), dtype=np.float32)
            for token in tokens:
                if self.cumulative_term_frequency:
                    vector[token] += self.weights[token]
                else:
                    vector[token] = self.weights[token]
            vectors.append(vector)

        return {"sentence_embedding": torch.tensor(vectors, dtype=torch.float)}

    def get_config_dict(self):
        """
        Get the configuration dictionary.

        Returns
        -------
        Dict[str, Union[List[str], Dict[str, float], float, bool]]
            Configuration dictionary.
        """
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path:str) -> None:
        """
        Save the BoW model to the specified output path.

        Parameters
        ----------
        output_path : str
            Path to save the model.

        Returns
        -------
        None
        """
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path:str):
        """
        Load the BoW model from the specified input path.

        Parameters
        ----------
        input_path : str
            Path from which to load the model.

        Returns
        -------
        BoW
            Loaded BoW model.
        """
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        return BoW(**config)
