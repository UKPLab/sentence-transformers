from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
import logging

logger = logging.getLogger(__name__)


class DenoisingAutoEncoderLoss(nn.Module):
    """
    This loss expects as input a batch consisting of damaged sentences and the corresponding original ones.
    The data generation process has already been implemented in readers/DenoisingAutoEncoderReader.py
    During training, the decoder reconstructs the original sentences from the encoded sentence embeddings.
    Here the argument 'decoder_name_or_path' indicates the pretrained model (supported by Huggingface) to be used as the decoder.
    Since decoding process is included, here the decoder should have a class called XXXLMHead (in the context of Huggingface's Transformers).
    Flag 'tie_encoder_decoder' indicates whether to tie the trainable parameters of encoder and decoder,
    which is shown beneficial to model performance while limiting the amount of required memory.
    Only when the encoder and decoder are from the same architecture, can the flag 'tie_encoder_decoder' works.
    For more information, please refer to the TSDAE paper.
    """

    def __init__(self, model: SentenceTransformer, decoder_name_or_path: str = None, tie_encoder_decoder: bool = True):
        """
        :param model: SentenceTransformer model
        :param decoder_name_or_path: Model name or path for initializing a decoder (compatible with Huggingface's Transformers)
        :param tie_encoder_decoder: whether to tie the trainable parameters of encoder and decoder
        """
        super(DenoisingAutoEncoderLoss, self).__init__()
        self.encoder = model  # This will be the final model used during the inference time.
        self.tokenizer_encoder = model.tokenizer

        encoder_name_or_path = model[0].auto_model.config._name_or_path
        if decoder_name_or_path is None:
            assert (
                tie_encoder_decoder
            ), "Must indicate the decoder_name_or_path argument when tie_encoder_decoder=False!"
        if tie_encoder_decoder:
            if decoder_name_or_path:
                logger.warning("When tie_encoder_decoder=True, the decoder_name_or_path will be invalid.")
            decoder_name_or_path = encoder_name_or_path

        self.tokenizer_decoder = AutoTokenizer.from_pretrained(decoder_name_or_path)
        self.need_retokenization = not isinstance(self.tokenizer_encoder, type(self.tokenizer_decoder))

        decoder_config = AutoConfig.from_pretrained(decoder_name_or_path)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        kwargs_decoder = {"config": decoder_config}
        try:
            self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name_or_path, **kwargs_decoder)
        except ValueError as e:
            logger.error(
                f'Model name or path "{decoder_name_or_path}" does not support being as a decoder. Please make sure the decoder model has an "XXXLMHead" class.'
            )
            raise e
        assert model[0].auto_model.config.hidden_size == decoder_config.hidden_size, "Hidden sizes do not match!"
        if self.tokenizer_decoder.pad_token is None:
            # Needed by GPT-2, etc.
            self.tokenizer_decoder.pad_token = self.tokenizer_decoder.eos_token
            self.decoder.config.pad_token_id = self.decoder.config.eos_token_id

        if len(AutoTokenizer.from_pretrained(encoder_name_or_path)) != len(self.tokenizer_encoder):
            logger.warning(
                "WARNING: The vocabulary of the encoder has been changed. One might need to change the decoder vocabulary, too."
            )

        if tie_encoder_decoder:
            assert not self.need_retokenization, "The tokenizers should be the same when tie_encoder_decoder=True."
            if len(self.tokenizer_encoder) != len(self.tokenizer_decoder):  # The vocabulary has been changed.
                self.tokenizer_decoder = self.tokenizer_encoder
                self.decoder.resize_token_embeddings(len(self.tokenizer_decoder))
                logger.warning(
                    "Since the encoder vocabulary has been changed and --tie_encoder_decoder=True, now the new vocabulary has also been used for the decoder."
                )
            decoder_base_model_prefix = self.decoder.base_model_prefix
            PreTrainedModel._tie_encoder_decoder_weights(
                model[0].auto_model, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def retokenize(self, sentence_features):
        input_ids = sentence_features["input_ids"]
        device = input_ids.device
        sentences_decoded = self.tokenizer_encoder.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        retokenized = self.tokenizer_decoder(
            sentences_decoded, padding=True, truncation="longest_first", return_tensors="pt", max_length=None
        ).to(device)
        return retokenized

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        source_features, target_features = tuple(sentence_features)
        if self.need_retokenization:
            # since the sentence_features here are all tokenized by encoder's tokenizer,
            # retokenization by the decoder's one is needed if different tokenizers used
            target_features = self.retokenize(target_features)
        reps = self.encoder(source_features)["sentence_embedding"]  # (bsz, hdim)

        # Prepare input and output
        target_length = target_features["input_ids"].shape[1]
        decoder_input_ids = target_features["input_ids"].clone()[:, : target_length - 1]
        label_ids = target_features["input_ids"][:, 1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=None,
            attention_mask=None,
            encoder_hidden_states=reps[:, None],  # (bsz, hdim) -> (bsz, 1, hdim)
            encoder_attention_mask=source_features["attention_mask"][:, 0:1],
            labels=None,
            return_dict=None,
            use_cache=False,
        )

        # Calculate loss
        lm_logits = decoder_outputs[0]
        ce_loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer_decoder.pad_token_id)
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label_ids.reshape(-1))
        return loss
