from torch.utils.data import Dataset
from typing import List
import torch
import logging
import itertools
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
        batch_size = len(examples)
        iterator = self.chunks(examples, batch_size)
        max_seq_length = model.get_max_seq_length()

        if self.show_progress_bar:
            batch_size = round(len(examples)/10)
            iterator = tqdm(self.chunks(examples, batch_size), desc = f"Tokenizing inputs in batches of size {batch_size}", total=round(len(examples)/batch_size))

        for chunk in iterator:
            flat_chunk = list(itertools.chain(*[example.texts for example in chunk]))
            flat_tokenized_examples = model.batch_tokenize(flat_chunk)
            tokenized_examples = list(self.chunks(flat_tokenized_examples, 2))
            for i in range(num_texts):
                inputs[i].extend([tokenized_example[i] for tokenized_example in tokenized_examples])

            for example in chunk:
                if label_type is None:
                    if isinstance(example.label, int):
                        label_type = torch.long
                    elif isinstance(example.label, float):
                        label_type = torch.float
                labels.append(example.label)

        for i, tokenized_texts in enumerate(inputs): 
            for tokenized_text in tokenized_texts:
                if max_seq_length != None and max_seq_length > 0 and len(tokenized_text) >= max_seq_length:
                    too_long[i] += 1

        tensor_labels = torch.tensor(labels, dtype=label_type)

        logging.info("Num sentences: %d" % (len(examples)))
        for i in range(num_texts):
            logging.info("Sentences {} longer than max_sequence_length: {}".format(i, too_long[i]))

        self.tokens = inputs
        self.labels = tensor_labels

    def __getitem__(self, item):
        return [self.tokens[i][item] for i in range(len(self.tokens))], self.labels[item]

    def __len__(self):
        return len(self.tokens[0])

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst.
        >>> list(chunks([1,2,3,4,5,6,7],3))
        [[1,2,3],[4,5,6],[7]]"""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]