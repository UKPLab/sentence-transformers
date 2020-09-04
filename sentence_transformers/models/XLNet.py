from torch import Tensor
from torch import nn
from transformers import XLNetModel, XLNetTokenizer
import json
from typing import Union, Tuple, List, Dict, Optional
import os
import numpy as np

class XLNet(nn.Module):
    """DEPRECATED: Please use models.Transformer instead.

    XLNet model to generate token embeddings.

    Each token is mapped to an output vector from XLNet.
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, do_lower_case: Optional[bool] = None, model_args: Dict = {}, tokenizer_args: Dict = {}):
        super(XLNet, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case

        if self.do_lower_case is not None:
            tokenizer_args['do_lower_case'] = do_lower_case

        self.xlnet = XLNetModel.from_pretrained(model_name_or_path, **model_args)
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])[0]
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])[0]

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        output_states = self.xlnet(**features)
        output_tokens = output_states[0]
        cls_tokens = output_tokens[:, -1, :]  # CLS token is the last token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})

        if self.xlnet.config.output_hidden_states:
            hidden_states = output_states[2]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.xlnet.config.d_model

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def get_sentence_features(self, tokens: List[int], pad_seq_length: int) -> Dict[str, Tensor]:
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length) + 3 #Add space for special tokens
        return self.tokenizer.prepare_for_model(tokens, max_length=pad_seq_length, padding='max_length', return_tensors='pt', truncation=True, prepend_batch_axis=True)

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






