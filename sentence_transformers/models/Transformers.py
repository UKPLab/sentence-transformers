from torch import nn
from transformers import AutoModel, AutoTokenizer
import json
from typing import List
import os
import numpy as np
import logging


class Transformers(nn.Module):
    """BERT model to generate token embeddings.

    Each token is mapped to an output vector from BERT.
    """

    def __init__(self, model_name_or_path: str, model_type: str, max_seq_length: int = 128, do_lower_case: bool = True):
        super(Transformers, self).__init__()
        self.config_keys = ['model_type', 'max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        self.model = AutoModel.from_pretrained(model_name_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       do_lower_case=do_lower_case)
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])[0]
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])[0]

        if max_seq_length > self.tokenizer.max_len_single_sentence:
            logging.warning(
                "BERT only allows a max_seq_length of 510 (512 with special tokens). Value will be set to 510")
            max_seq_length = self.tokenizer.max_len_single_sentence
        self.max_seq_length = max_seq_length
        self.model_type = model_type

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        output_tokens = self.model(input_ids=features['input_ids'], token_type_ids=features['token_type_ids'],
                                   attention_mask=features['input_mask'])[0]
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens,
                         'input_mask': features['input_mask']})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.model.config.hidden_size

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def model_specific_token_ids(self):
        cls_token_at_end, sep_token_extra, pad_left = False, False, False
        cls_token_segment_id, pad_token_segment_id = 0, 0
        if self.model_type in ["xlnet"]:
            cls_token_at_end = True
            pad_left = True
            cls_token_segment_id = 2
            pad_token_segment_id = 4
        if self.model_type in ["roberta"]:
            sep_token_extra = True
        return cls_token_at_end, sep_token_extra, pad_left, \
               cls_token_segment_id, pad_token_segment_id

    def get_sentence_features(self, tokens: List[int], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        max_seq_length = min(pad_seq_length, self.max_seq_length)
        cls_token_at_end, sep_token_extra, pad_left, cls_token_segment_id, pad_token_segment_id = self.model_specific_token_ids()
        sep_token = self.sep_token_id
        cls_token = self.cls_token_id
        sequence_a_segment_id = 0
        pad_token = 0

        tokens = tokens[:max_seq_length]

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = self.tokenizer.num_added_tokens()
        max_seq_length += special_tokens_count  # TODO this can go over the maximum seq length, needs to be fixed

        tokens += [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            token_type_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens

        input_ids = tokens
        sentence_length = len(input_ids)

        token_type_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if pad_left:
            # Zero-pad up to the sequence length. XLNet: Pad to the left
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0] * padding_length) + input_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            # Zero-pad up to the sequence length: Pad to the Right
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        return {'input_ids': np.asarray(input_ids, dtype=np.int64),
                'token_type_ids': np.asarray(token_type_ids, dtype=np.int64),
                'input_mask': np.asarray(input_mask, dtype=np.int64),
                'sentence_lengths': np.asarray(sentence_length, dtype=np.int64)}

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_transformer_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        # Handle backwards compatibility for old pre-trained specific models
        if os.path.exists(os.path.join(input_path, 'sentence_bert_config.json')):
            with open(os.path.join(input_path, 'sentence_bert_config.json')) as fIn:
                config = json.load(fIn)
                config['model_type'] = 'bert'
        else:
            with open(os.path.join(input_path, 'sentence_transformer_config.json')) as fIn:
                config = json.load(fIn)
        return Transformers(model_name_or_path=input_path, **config)
