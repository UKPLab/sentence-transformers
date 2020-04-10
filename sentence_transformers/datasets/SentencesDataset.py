from torch.utils.data import Dataset
from typing import List
import torch
import logging
from tqdm import tqdm
from .. import SentenceTransformer
from ..readers.InputExample import InputExample


class SentencesDataset(Dataset):
    """
    Dataset for smart batching, that is each batch is only padded to its longest sequence instead of padding all
    sequences to the max length.
    The SentenceBertEncoder.smart_batching_collate is required for this to work.
    SmartBatchingDataset does *not* work without it.
    """
    def __init__(self, examples: List[InputExample], model: SentenceTransformer, show_progress_bar: bool = None):
        """
        Create a new SentencesDataset with the tokenized texts and the labels as Tensor
        """
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.convert_input_examples(examples, model)

    def convert_input_examples(self, examples: List[InputExample], model: SentenceTransformer):
        """
        Converts input examples to a SmartBatchingDataset usable to train the model with
        SentenceTransformer.smart_batching_collate as the collate_fn for the DataLoader

        smart_batching_collate as collate_fn is required because it transforms the tokenized texts to the tensors.

        :param examples:
            the input examples for the training
        :param model
            the Sentence BERT model for the conversion
        :return: a SmartBatchingDataset usable to train the model with SentenceTransformer.smart_batching_collate as the collate_fn
            for the DataLoader
        """
        num_texts = len(examples[0].texts)
        inputs = [[] for _ in range(num_texts)]
        labels = []
        too_long = [0] * num_texts
        label_type = None
        iterator = examples
        max_seq_length = model.get_max_seq_length()

        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Convert dataset")

        for ex_index, example in enumerate(iterator):
            if label_type is None:
                if isinstance(example.label, int):
                    label_type = torch.long
                elif isinstance(example.label, float):
                    label_type = torch.float
            tokenized_texts = [model.tokenize(text) for text in example.texts]

            for i, token in enumerate(tokenized_texts):
                if max_seq_length != None and max_seq_length > 0 and len(token) >= max_seq_length:
                    too_long[i] += 1

            labels.append(example.label)
            for i in range(num_texts):
                inputs[i].append(tokenized_texts[i])

        tensor_labels = torch.tensor(labels, dtype=label_type)

        logging.info("Num sentences: %d" % (len(examples)))
        for i in range(num_texts):
            logging.info("Sentences {} longer than max_seqence_length: {}".format(i, too_long[i]))

        self.tokens = inputs
        self.labels = tensor_labels

    def __getitem__(self, item):
        return [self.tokens[i][item] for i in range(len(self.tokens))], self.labels[item]

    def __len__(self):
        return len(self.tokens[0])