import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import logging
import gzip
from tqdm import tqdm
import numpy as np
import os
import json
from ..util import import_from_string, fullname, http_get
from .tokenizer import WordTokenizer, WhitespaceTokenizer


class WordEmbeddings(nn.Module):
    def __init__(self, tokenizer: WordTokenizer, embedding_weights, update_embeddings: bool = False, max_seq_length: int = 1000000):
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
        features.update({'token_embeddings': token_embeddings, 'cls_token_embeddings': cls_tokens, 'input_mask': features['input_mask']})
        return features

    def get_sentence_features(self, tokens: List[str], pad_seq_length: int):
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

        return {'input_ids': np.asarray(input_ids, dtype=np.int64),
                'input_mask': np.asarray(input_mask, dtype=np.int64),
                'sentence_lengths': np.asarray(sentence_length, dtype=np.int64)}

    def get_word_embedding_dimension(self) -> int:
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
        logging.info("Read in embeddings file {}".format(embeddings_file_path))

        if not os.path.exists(embeddings_file_path):
            logging.info("{} does not exist, try to download from server".format(embeddings_file_path))

            if '/' in embeddings_file_path or '\\' in embeddings_file_path:
                raise ValueError("Embeddings file not found: ".format(embeddings_file_path))

            url = "https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/"+embeddings_file_path
            http_get(url, embeddings_file_path)

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

