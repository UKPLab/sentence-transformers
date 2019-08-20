from torch import Tensor
from torch import nn
from pytorch_transformers import RobertaModel, RobertaTokenizer
import json
from typing import Union, Tuple, List, Dict
import os
import numpy as np
import logging

class RoBERTa(nn.Module):
    """RoBERTa model to generate token embeddings.

    Each token is mapped to an output vector from RoBERTa.
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, do_lower_case: bool = True):
        super(RoBERTa, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        if max_seq_length > 510:
            logging.warning("RoBERTa only allows a max_seq_length of 510 (512 with special tokens). Value will be set to 510")
            max_seq_length = 510
        self.max_seq_length = max_seq_length


        self.roberta = RobertaModel.from_pretrained(model_name_or_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])[0]
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])[0]

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        #RoBERTa does not use token_type_ids
        output_tokens = self.roberta(input_ids=features['input_ids'], token_type_ids=None, attention_mask=features['input_mask'])[0]
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'input_mask': features['input_mask']})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.roberta.config.hidden_size

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def get_sentence_features(self, tokens: List[str], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length)

        tokens = tokens[:pad_seq_length]
        input_ids = [self.cls_token_id] + tokens + [self.sep_token_id] + [self.sep_token_id]
        sentence_length = len(input_ids)

        pad_seq_length += 3  ##Add Space for CLS + SEP + SEP token

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length. BERT: Pad to the right
        padding = [0] * (pad_seq_length - len(input_ids))
        input_ids += padding

        input_mask += padding

        assert len(input_ids) == pad_seq_length
        assert len(input_mask) == pad_seq_length


        return {'input_ids': np.asarray(input_ids, dtype=np.int64), 'input_mask': np.asarray(input_mask, dtype=np.int64), 'sentence_lengths': np.asarray(sentence_length, dtype=np.int64)}

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.roberta.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_roberta_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'sentence_roberta_config.json')) as fIn:
            config = json.load(fIn)
        return RoBERTa(model_name_or_path=input_path, **config)






