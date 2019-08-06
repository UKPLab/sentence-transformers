import torch
from torch import Tensor
from typing import List, Tuple
from numpy import ndarray
import numpy as np
from .config import SentenceTransformerConfig
from .models import TransformerModel
from tqdm import tqdm


class SentenceEncoder:
    """
    Wrapper for the encoding and embedding of sentences with Sentence Transformers
    """
    def __init__(self, transformer_model: TransformerModel, transformer_model_config: SentenceTransformerConfig):
        """
        Creates a new encoder with the given model and config

        :param transformer_model:
            the model that encodes and embeds sentences
        :param transformer_model_config:
            the config for the embedding and encoding
        """
        self.do_lower_case = transformer_model_config.do_lower_case
        self.max_seq_length = transformer_model_config.max_seq_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = transformer_model
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = transformer_model.tokenizer_model

    def get_sentence_embeddings(self, sentences: List[str], batch_size: int = 8, show_progress_bar: bool = None) -> List[ndarray]:
        """
        Computes the Sentence BERT embeddings for the sentences

        :param sentences:
            the sentences to embed
        :param batch_size:
            the batch size used for the computation
        :return:
            a list with ndarrays of the embeddings for each sentence
        """
        all_embeddings = []

        length_sorted_idx = np.argsort([len(sen) for sen in sentences])

        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Batches")

        for batch_idx in iterator:
            batch_tokens = []
            batch_input_ids = []
            batch_segment_ids = []
            batch_input_masks = []

            batch_start = batch_idx
            batch_end = min(batch_start+batch_size, len(sentences))

            longest_seq = 0

            for idx in length_sorted_idx[batch_start: batch_end]:
                sentence = sentences[idx]
                if self.do_lower_case:
                    sentence = sentence.lower()

                tokens = self.tokenizer.tokenize(sentence)
                longest_seq = max(longest_seq, len(tokens))
                batch_tokens.append(tokens)

            for tokens in batch_tokens:
                input_ids, segment_ids, input_mask = self.model.get_sentence_features(tokens, longest_seq)

                batch_input_ids.append(input_ids)
                batch_segment_ids.append(segment_ids)
                batch_input_masks.append(input_mask)

            batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(self.device)
            batch_segment_ids = torch.tensor(batch_segment_ids, dtype=torch.long).to(self.device)
            batch_input_masks = torch.tensor(batch_input_masks, dtype=torch.long).to(self.device)

            with torch.no_grad():
                embeddings = self.model.get_sentence_representation(batch_input_ids, batch_segment_ids,
                                                                    batch_input_masks)
                embeddings = embeddings.to('cpu').numpy()
                all_embeddings.extend(embeddings)

        reverting_order = np.argsort(length_sorted_idx)
        all_embeddings = [all_embeddings[idx] for idx in reverting_order]

        return all_embeddings



    def smart_batching_collate(self, batch: List[Tuple[List[List[str]], Tensor]]) \
            -> Tuple[List[Tensor], List[Tensor], List[Tensor], Tensor]:
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0][0])

        labels = []
        paired_texts = [[] for _ in range(num_texts)]
        max_seq_len = [0] * num_texts
        for tokens, label in batch:
            labels.append(label)
            for i in range(num_texts):
                paired_texts[i].append(tokens[i])
                max_seq_len[i] = max(max_seq_len[i], len(tokens[i]))

        inputs = [[] for _ in range(num_texts)]
        segments = [[] for _ in range(num_texts)]
        masks = [[] for _ in range(num_texts)]

        for texts in zip(*paired_texts):
            features = [self.model.get_sentence_features(text, max_len) for text, max_len in zip(texts, max_seq_len)]
            for i in range(num_texts):
                inputs[i].append(features[i][0])
                segments[i].append(features[i][1])
                masks[i].append(features[i][2])

        tensor_labels = torch.stack(labels)
        tensor_inputs = [torch.tensor(input_ids, dtype=torch.long) for input_ids in inputs]
        tensor_masks = [torch.tensor(mask_ids, dtype=torch.long) for mask_ids in masks]
        tensor_segments = [torch.tensor(segment_ids, dtype=torch.long) for segment_ids in segments]


        return tensor_inputs, tensor_segments, tensor_masks, tensor_labels
