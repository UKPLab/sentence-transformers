from typing import List, Iterable
import collections
import string
import os
import json
from .WordTokenizer import WordTokenizer, ENGLISH_STOP_WORDS


class WhitespaceTokenizer(WordTokenizer):
    """
    Simple and fast white-space tokenizer. Splits sentence based on white spaces.
    Punctuation are stripped from tokens.

    Parameters
    ----------
    vocab : Iterable[str], optional
        Vocabulary of the tokenizer. Default is an empty list.

    stop_words : Iterable[str], optional
        Set of stop words. Default is ENGLISH_STOP_WORDS.
        
    do_lower_case : bool, optional
        Whether to convert text to lowercase. Default is False.

    Methods
    -------
    get_vocab() -> Iterable[str]:
        Get the vocabulary used for tokenization.

    set_vocab(vocab: Iterable[str]):
        Set the vocabulary used for tokenization.

    tokenize(text: str) -> List[int]:
        Tokenize the input text.

    save(output_path: str):
        Save the WhitespaceTokenizer model to the specified output path.

    @staticmethod
    load(input_path: str) -> 'WhitespaceTokenizer':
        Load the WhitespaceTokenizer model from the specified input path.
    """
    def __init__(
        self, vocab: Iterable[str] = [], stop_words: Iterable[str] = ENGLISH_STOP_WORDS, do_lower_case: bool = False
    ):
        """
        Initialize the WhitespaceTokenizer.

        Parameters
        ----------
        vocab : Iterable[str], optional
            Vocabulary of the tokenizer. Default is an empty list.

        stop_words : Iterable[str], optional
            Set of stop words. Default is ENGLISH_STOP_WORDS.

        do_lower_case : bool, optional
            Whether to convert text to lowercase. Default is False.
        """
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.set_vocab(vocab)

    def get_vocab(self):
        """
        Get the vocabulary used for tokenization.

        Returns
        -------
        Iterable[str]
            Vocabulary used for tokenization.
        """
        return self.vocab

    def set_vocab(self, vocab: Iterable[str]) -> None:
        """
        Set the vocabulary used for tokenization.

        Parameters
        ----------
        vocab : Iterable[str]
            Vocabulary to set.

        Returns
        -------
        None
        """
        self.vocab = vocab
        self.word2idx = collections.OrderedDict([(word, idx) for idx, word in enumerate(vocab)])

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize the input text.

        Parameters
        ----------
        text : str
            Input text to tokenize.

        Returns
        -------
        List[int]
            List of token indices.
        """
        if self.do_lower_case:
            text = text.lower()

        tokens = text.split()

        tokens_filtered = []
        for token in tokens:
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.strip(string.punctuation)
            if token in self.stop_words:
                continue
            elif len(token) > 0 and token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.lower()
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

        return tokens_filtered

    def save(self, output_path: str) -> None:
        """
        Save the WhitespaceTokenizer model to the specified output path.

        Parameters
        ----------
        output_path : str
            Path to save the model.

        Returns
        -------
        None
        """
        with open(os.path.join(output_path, "whitespacetokenizer_config.json"), "w") as fOut:
            json.dump(
                {
                    "vocab": list(self.word2idx.keys()),
                    "stop_words": list(self.stop_words),
                    "do_lower_case": self.do_lower_case,
                },
                fOut,
            )

    @staticmethod
    def load(input_path: str):
        """
        Load the WhitespaceTokenizer model from the specified input path.

        Parameters
        ----------
        input_path : str
            Path from which to load the model.

        Returns
        -------
        WhitespaceTokenizer
            Loaded WhitespaceTokenizer model.
        """
        with open(os.path.join(input_path, "whitespacetokenizer_config.json"), "r") as fIn:
            config = json.load(fIn)

        return WhitespaceTokenizer(**config)
