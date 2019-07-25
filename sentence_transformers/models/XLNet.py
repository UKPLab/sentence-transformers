import torch
from torch import Tensor
from pytorch_transformers import XLNetConfig, XLNetModel, XLNetTokenizer
from pytorch_transformers.modeling_xlnet import XLNetPreTrainedModel


from typing import Union, Tuple, List
from .. import LossFunction
from .. import SentenceTransformerConfig
from .TransformerModel import TransformerModel


from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class XLNet(XLNetPreTrainedModel, TransformerModel):
    def __init__(self, config: XLNetConfig, sentence_transformer_config: SentenceTransformerConfig = None):
        XLNetPreTrainedModel.__init__(self, config)

        self.model_config = config
        self.model_hidden_size = config.d_model

        TransformerModel.__init__(self, sentence_transformer_config)

        self.transformer = XLNetModel(config)
        self.apply(self.init_weights)

        ##Code from summary
        self.summary = nn.Identity()
        if hasattr(config, 'summary_use_proj') and config.summary_use_proj:
            self.summary = nn.Linear(config.d_model, config.d_model)

        self.activation = nn.Identity()
        if hasattr(config, 'summary_activation') and config.summary_activation=='tanh':
            self.activation = nn.Tanh()

        self.first_dropout = nn.Identity()
        if hasattr(config, 'summary_first_dropout') and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = nn.Identity()
        if hasattr(config, 'summary_last_dropout') and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)


    def set_tokenizer(self, tokenizer_model: str, do_lower_case: bool):
        """
        Sets the tokenizer for this model
        """
        self.tokenizer_model = XLNetTokenizer.from_pretrained(tokenizer_model, do_lower_case=do_lower_case)


    def forward(self, input_ids: List[Tensor], token_type_ids: List[Tensor], attention_mask: List[Tensor],
                labels: Tensor = None, alternative_loss: LossFunction = None) -> Union[
        Tensor, Tuple[List[Tensor], Tensor]]:
        """
        Forward pass of the model for a variable number of sentences, each represented as input ids,
        segment ids and masks.

        If labels are given, then the training loss is returned otherwise sentence embeddings
        and the LossFunction value is returned.
        The training loss is not the same as the LossFunction value.

        :param input_ids:
            list of Tensors of the token embedding ids
        :param token_type_ids:
            list of Tensors of the segment ids
        :param attention_mask:
            list of Tensors of the masks
        :param labels:
            Tensor of the training labels
        :param alternative_loss:
            an alternative loss different to self.sentence_transformer_config.loss_function for multitask learning.
            the loss still uses the configuration as given in self.sbert_config, so you cannot for example
            have two different LossFunction.SOFTMAX with different number of labels
        :return: the training loss or the two sentence embeddings and the LossFunction value (if available for the
            chosen loss)
        """
        reps = [self.get_sentence_representation(ids, seg, mask) for ids, seg, mask
                in zip(input_ids, token_type_ids, attention_mask)]

        return self.compute_loss(reps, labels, alternative_loss)


    def _get_transformer_output(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor):
        """
        Internal method that invokes the underlying model
        and returns the output vectors for all input tokens
        """
        transformer_outputs = self.transformer(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        output_tokens = transformer_outputs[0]
        cls_tokens = output_tokens[:, -1] # CLS token is the last token

        #cls_tokens = self.first_dropout(cls_tokens)
        cls_tokens = self.summary(cls_tokens)
        cls_tokens = self.activation(cls_tokens)
        #cls_tokens = self.last_dropout(cls_tokens)

        return output_tokens, cls_tokens

    def get_sentence_features(self, tokens: List[str], max_seq_length: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param max_seq_length:
            the maximal length of the sequence.
            If this is greater than self.max_seq_length, then self.max_seq_length is used.
            If this is 0, then self.max_seq_length is used.
        :return: embedding ids, segment ids and mask for the sentence
        """
        sep_token = self.tokenizer_model.sep_token
        cls_token = self.tokenizer_model.cls_token
        sequence_a_segment_id = 0
        cls_token_segment_id = 2
        pad_token_segment_id = 4
        pad_token = 0


        max_seq_length += 2 ##Add space for CLS + SEP token

        tokens = tokens[:(max_seq_length - 2)] + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        # XLNet CLS token at at
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]

        input_ids = self.tokenizer_model.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length. XLNet: Pad to the left
        padding_length = max_seq_length - len(input_ids)
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids

        assert len(input_ids)==max_seq_length
        assert len(input_mask)==max_seq_length
        assert len(segment_ids)==max_seq_length




        return input_ids, segment_ids, input_mask



