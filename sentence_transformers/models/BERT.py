import torch
from torch import Tensor
from pytorch_transformers import BertConfig, BertModel, BertTokenizer
from pytorch_transformers.modeling_bert import BertPreTrainedModel


from typing import Union, Tuple, List
from .. import LossFunction
from .. import SentenceTransformerConfig
from .TransformerModel import TransformerModel



class BERT(BertPreTrainedModel, TransformerModel):
    """
    PyTorch model of Sentence BERT
    """
    def __init__(self, model_config: BertConfig, sentence_transformer_config: SentenceTransformerConfig = None):
        """
        Creates a new Sentence BERT model with the given config for BertPreTrainedModel and the given config
        for Sentence BERT

        :param model_config:
            config for BertPreTrainedModel
        :param sentence_transformer_config:
            config for the Sentence Transformer
        """
        BertPreTrainedModel.__init__(self, model_config)
        self.model_config = model_config
        self.model_hidden_size = model_config.hidden_size

        TransformerModel.__init__(self, sentence_transformer_config)

        self.bert = BertModel(model_config)
        self.apply(self.init_weights)


    def set_tokenizer(self, tokenizer_model: str, do_lower_case: bool):
        """
        Sets the tokenizer for this model
        """
        self.tokenizer_model = BertTokenizer.from_pretrained(tokenizer_model, do_lower_case=do_lower_case)


    def forward(self, input_ids: List[Tensor], token_type_ids: List[Tensor], attention_mask: List[Tensor],
                labels: Tensor = None, alternative_loss: LossFunction = None) -> Union[Tensor, Tuple[List[Tensor], Tensor]]:
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
        output_tokens = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        return output_tokens, cls_tokens

    def get_sentence_features(self, tokens: List[str], max_seq_length: int) \
            -> Tuple[List[int], List[int], List[int]]:
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
        max_seq_length += 2 ##Add Space for CLS + SEP token

        tokens = tokens[:(max_seq_length - 2)]
        tokens = [self.tokenizer_model.cls_token] + tokens + [self.tokenizer_model.sep_token]
        input_ids = self.tokenizer_model.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length. BERT: Pad to the right
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        segment_ids += padding
        input_mask += padding

        assert len(input_ids)==max_seq_length
        assert len(input_mask)==max_seq_length
        assert len(segment_ids)==max_seq_length

        return input_ids, segment_ids, input_mask


