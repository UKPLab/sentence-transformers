from typing import List, Iterable
import collections
import string
import os
import json
import logging
from .WordTokenizer import WordTokenizer, ENGLISH_STOP_WORDS
import nltk


logger = logging.getLogger(__name__)


class PhraseTokenizer(WordTokenizer):
    """Tokenizes the text with respect to existent phrases in the vocab.

    This tokenizers respects phrases that are in the vocab. Phrases are separated with 'ngram_separator', for example,
    in Google News word2vec file, ngrams are separated with a _ like New_York. These phrases are detected in text and merged as one special token. (New York is the ... => [New_York, is, the])
    """

    def __init__(
        self,
        vocab: Iterable[str] = [],
        stop_words: Iterable[str] = ENGLISH_STOP_WORDS,
        do_lower_case: bool = False,
        ngram_separator: str = "_",
        max_ngram_length: int = 5,
    ):
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.ngram_separator = ngram_separator
        self.max_ngram_length = max_ngram_length
        self.set_vocab(vocab)

    def get_vocab(self):
        return self.vocab

    def set_vocab(self, vocab: Iterable[str]):
        self.vocab = vocab
        self.word2idx = collections.OrderedDict([(word, idx) for idx, word in enumerate(vocab)])

        # Check for ngram in vocab
        self.ngram_lookup = set()
        self.ngram_lengths = set()
        for word in vocab:
            if self.ngram_separator is not None and self.ngram_separator in word:
                # Sum words might me malformed in e.g. google news word2vec, containing two or more _ after each other
                ngram_count = word.count(self.ngram_separator) + 1
                if self.ngram_separator + self.ngram_separator not in word and ngram_count <= self.max_ngram_length:
                    self.ngram_lookup.add(word)
                    self.ngram_lengths.add(ngram_count)

        if len(vocab) > 0:
            logger.info("PhraseTokenizer - Phrase ngram lengths: {}".format(self.ngram_lengths))
            logger.info("PhraseTokenizer - Num phrases: {}".format(len(self.ngram_lookup)))

    def tokenize(self, text: str) -> List[int]:
        tokens = nltk.word_tokenize(text, preserve_line=True)

        # phrase detection
        for ngram_len in sorted(self.ngram_lengths, reverse=True):
            idx = 0
            while idx <= len(tokens) - ngram_len:
                ngram = self.ngram_separator.join(tokens[idx : idx + ngram_len])
                if ngram in self.ngram_lookup:
                    tokens[idx : idx + ngram_len] = [ngram]
                elif ngram.lower() in self.ngram_lookup:
                    tokens[idx : idx + ngram_len] = [ngram.lower()]
                idx += 1

        # Map tokens to idx, filter stop words
        tokens_filtered = []
        for token in tokens:
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.lower()
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

        return tokens_filtered

    def save(self, output_path: str):
        with open(os.path.join(output_path, "phrasetokenizer_config.json"), "w") as fOut:
            json.dump(
                {
                    "vocab": list(self.word2idx.keys()),
                    "stop_words": list(self.stop_words),
                    "do_lower_case": self.do_lower_case,
                    "ngram_separator": self.ngram_separator,
                    "max_ngram_length": self.max_ngram_length,
                },
                fOut,
            )

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, "phrasetokenizer_config.json"), "r") as fIn:
            config = json.load(fIn)

        return PhraseTokenizer(**config)
