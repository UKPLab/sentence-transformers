import torch
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import collections
import string
import logging
import gzip
from tqdm import tqdm
import numpy as np
import nltk
import os
import json
from ..util import import_from_string, fullname

ENGLISH_STOP_WORDS = ['!', '"', "''", "``", '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',  '{', '|', '}', '~', 'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldn', 'couldnt', 'cry', 'd', 'de', 'describe', 'detail', 'did', 'didn', 'do', 'does', 'doesn', 'doing', 'don', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'hadn', 'has', 'hasn', 'hasnt', 'have', 'haven', 'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'isn', 'it', 'its', 'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'll', 'ltd', 'm', 'ma', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mightn', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'mustn', 'my', 'myself', 'name', 'namely', 'needn', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'shan', 'she', 'should', 'shouldn', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 't', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 've', 'very', 'via', 'was', 'wasn', 'we', 'well', 'were', 'weren', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'won', 'would', 'wouldn', 'y', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves']

class WhitespaceTokenizer:
    """
    Simple and fast white-space tokenizer. Splits sentence based on white spaces.
    Punctuation are stripped from tokens.
    """
    def __init__(self, vocab: Iterable[str] = [], stop_words: Iterable[str] = ENGLISH_STOP_WORDS, do_lower_case: bool = False):
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.set_vocab(vocab)

    def set_vocab(self, vocab: Iterable[str]):
        self.word2idx = collections.OrderedDict([(word, idx) for idx, word in enumerate(vocab)])

    def tokenize(self, text: str) -> List[int]:
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

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'whitespacetokenizer_config.json'), 'w') as fOut:
            json.dump({'vocab': list(self.word2idx.keys()), 'stop_words': list(self.stop_words), 'do_lower_case': self.do_lower_case}, fOut)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'whitespacetokenizer_config.json'), 'r') as fIn:
            config = json.load(fIn)

        return WhitespaceTokenizer(**config)


class NgramTokenizer:
    """
    This tokenizers respects ngrams that are in the vocab. Ngrams are separated with 'ngram_separator', for example,
    in Google News word2vec file, ngrams are separated with a _. These ngrams are detected in text and merged as one special token.
    """
    def __init__(self, vocab: Iterable[str] = [], stop_words: Iterable[str] = ENGLISH_STOP_WORDS, do_lower_case: bool = False, ngram_separator: str = "_", max_ngram_length: int = 5):
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.ngram_separator = ngram_separator
        self.max_ngram_length = max_ngram_length
        self.set_vocab(vocab)

    def set_vocab(self, vocab: Iterable[str]):
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
            logging.info("NgramTokenizer - Ngram lengths: {}".format(self.ngram_lengths))
            logging.info("NgramTokenizer - Num ngrams: {}".format(len(self.ngram_lookup)))

    def tokenize(self, text: str) -> List[int]:
        tokens = nltk.word_tokenize(text, preserve_line=True)

        #phrase detection
        for ngram_len in sorted(self.ngram_lengths, reverse=True):
            idx = 0
            while idx <= len(tokens) - ngram_len:
                ngram = self.ngram_separator.join(tokens[idx:idx + ngram_len])
                if ngram in self.ngram_lookup:
                    tokens[idx:idx + ngram_len] = [ngram]
                elif ngram.lower() in self.ngram_lookup:
                    tokens[idx:idx + ngram_len] = [ngram.lower()]
                idx += 1

        #Map tokens to idx, filter stop words
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
        with open(os.path.join(output_path, 'whitespacetokenizer_config.json'), 'w') as fOut:
            json.dump({'vocab': list(self.word2idx.keys()), 'stop_words': list(self.stop_words), 'do_lower_case': self.do_lower_case, 'ngram_separator': self.ngram_separator, 'max_ngram_length': self.max_ngram_length}, fOut)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'whitespacetokenizer_config.json'), 'r') as fIn:
            config = json.load(fIn)

        return WhitespaceTokenizer(**config)





class WordEmbeddings(nn.Module):
    def __init__(self, tokenizer, embedding_weights, update_embeddings: bool = False, max_seq_length: int = 1000000):
        nn.Module.__init__(self)
        if isinstance(embedding_weights, list):
            embedding_weights = np.asarray(embedding_weights)

        if isinstance(embedding_weights, np.ndarray):
            embedding_weights = torch.from_numpy(embedding_weights)

        num_embeddings, embeddings_dimension = embedding_weights.size()
        self.embeddings_dimension = embeddings_dimension
        self.emb_layer = nn.Embedding(num_embeddings, embeddings_dimension)
        self.emb_layer.load_state_dict({'weight': embedding_weights})
        self.emb_layer.weight.requires_grad = update_embeddings
        self.tokenizer = tokenizer
        self.update_embeddings = update_embeddings
        self.max_seq_length = max_seq_length

    def forward(self, features):
        token_embeddings = self.emb_layer(features['input_ids'])
        cls_tokens = None
        return {'token_embeddings': token_embeddings, 'cls_token_embeddings': cls_tokens, 'input_mask': features['input_mask']}

    def get_sentence_features(self, tokens: List[str], pad_seq_length: int) -> Tuple[List[int], List[int], List[int], int]:
        pad_seq_length = min(pad_seq_length, self.max_seq_length)

        tokens = tokens[0:pad_seq_length] #Truncate tokens if needed
        input_ids = tokens

        sentence_length = len(input_ids)
        input_mask = [1] * len(input_ids)
        padding = [0] * (pad_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding

        assert len(input_ids) == pad_seq_length
        assert len(input_mask) == pad_seq_length

        return {'input_ids': input_ids, 'input_mask': input_mask, 'sentence_lengths': sentence_length}

    def word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'wordembedding_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))
        self.tokenizer.save(output_path)

    def get_config_dict(self):
        return {'tokenizer_class': fullname(self.tokenizer), 'update_embeddings': self.update_embeddings, 'max_seq_length': self.max_seq_length}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'wordembedding_config.json'), 'r') as fIn:
            config = json.load(fIn)

        tokenizer_class = import_from_string(config['tokenizer_class'])
        tokenizer = tokenizer_class.load(input_path)
        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'))
        embedding_weights = weights['emb_layer.weight']
        model = WordEmbeddings(tokenizer=tokenizer, embedding_weights=embedding_weights, update_embeddings=config['update_embeddings'])
        return model

    @staticmethod
    def from_text_file(embeddings_file_path: str, update_embeddings: bool = False, item_separator: str = " ", tokenizer=WhitespaceTokenizer(), max_vocab_size: int = None):
        logging.info("Read in embeddings file")

        embeddings_dimension = None

        vocab = []
        embeddings = []

        with gzip.open(embeddings_file_path, "rt", encoding="utf8") if embeddings_file_path.endswith('.gz') else open(embeddings_file_path, encoding="utf8") as fIn:
            iterator = tqdm(fIn, desc="Load Word Embeddings", unit="Embeddings")
            for line in iterator:
                split = line.rstrip().split(item_separator)
                word = split[0]

                if embeddings_dimension == None:
                    embeddings_dimension = len(split) - 1
                    vocab.append("PADDING_TOKEN")
                    embeddings.append(np.zeros(embeddings_dimension))

                if (len(split) - 1) != embeddings_dimension:  # Assure that all lines in the embeddings file are of the same length
                    logging.error("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
                    continue

                vector = np.array([float(num) for num in split[1:]])
                embeddings.append(vector)
                vocab.append(word)

                if max_vocab_size is not None and max_vocab_size > 0 and len(vocab) > max_vocab_size:
                    break

            embeddings = np.asarray(embeddings)

            tokenizer.set_vocab(vocab)
            return WordEmbeddings(tokenizer=tokenizer, embedding_weights=embeddings, update_embeddings=update_embeddings)

