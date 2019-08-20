from torch import Tensor
from torch import nn
from pytorch_transformers import XLNetModel, XLNetTokenizer
import json
from typing import Union, Tuple, List, Dict
import os
import numpy as np

class XLNet(nn.Module):
    """XLNet model to generate token embeddings.

    Each token is mapped to an output vector from XLNet.
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, do_lower_case: bool = False):
        super(XLNet, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case

        self.xlnet = XLNetModel.from_pretrained(model_name_or_path)
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])[0]
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])[0]

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        output_tokens = self.xlnet(input_ids=features['input_ids'], token_type_ids=features['token_type_ids'], attention_mask=features['input_mask'])[0]
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'input_mask': features['input_mask']})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.xlnet.config.d_model

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def get_sentence_features(self, tokens: List[str], pad_seq_length: int) -> Dict[str, Tensor]:
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length)

        sep_token = self.sep_token_id
        cls_token = self.cls_token_id
        sequence_a_segment_id = 0
        cls_token_segment_id = 2
        pad_token_segment_id = 4
        pad_token = 0

        tokens = tokens[:pad_seq_length] + [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # XLNet CLS token at the end
        tokens = tokens + [cls_token]
        token_type_ids = token_type_ids + [cls_token_segment_id]
        pad_seq_length += 2  ##+2 for CLS and SEP token

        input_ids = tokens
        input_mask = [1] * len(input_ids)
        sentence_length = len(input_ids)

        # Zero-pad up to the sequence length. XLNet: Pad to the left
        padding_length = pad_seq_length - len(input_ids)
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0] * padding_length) + input_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids

        assert len(input_ids) == pad_seq_length
        assert len(input_mask) == pad_seq_length
        assert len(token_type_ids) == pad_seq_length


        return {'input_ids': np.asarray(input_ids, dtype=np.int64),
                'token_type_ids': np.asarray(token_type_ids, dtype=np.int64),
                'input_mask': np.asarray(input_mask, dtype=np.int64),
                'sentence_lengths': np.asarray(sentence_length, dtype=np.int64)}

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.xlnet.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_xlnet_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'sentence_xlnet_config.json')) as fIn:
            config = json.load(fIn)
        return XLNet(model_name_or_path=input_path, **config)






