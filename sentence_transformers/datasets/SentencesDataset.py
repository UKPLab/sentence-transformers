from torch.utils.data import Dataset
from typing import List
import torch
import logging
from tqdm import tqdm
from .. import SentenceTransformer
from ..readers.InputExample import InputExample
from multiprocessing import Pool, cpu_count
import multiprocessing
import time

class SentencesDataset(Dataset):
    """
    Dataset for smart batching, that is each batch is only padded to its longest sequence instead of padding all
    sequences to the max length.
    The SentenceBertEncoder.smart_batching_collate is required for this to work.
    SmartBatchingDataset does *not* work without it.
    """
    def __init__(self,
                 examples: List[InputExample],
                 model: SentenceTransformer,
                 parallel_tokenization: bool = True,
                 max_processes: int = 4,
                 chunk_size: int = 5000
                 ):
        """
        Create a new SentencesDataset with the tokenized texts and the labels as Tensor

        :param examples
            A list of sentence.transformers.readers.InputExample
        :param model:
            SentenceTransformerModel
        :param parallel_tokenization
            If true, multiple processes will be started for the tokenization
        :param max_processes
            Maximum number of processes started for tokenization. Cannot be larger can cpu_count()
        :param chunk_size
            #chunk_size number of examples are send to each process. Larger values increase overall tokenization speed
        """
        self.model = model
        self.max_processes = min(max_processes, cpu_count())
        self.chunk_size = chunk_size
        self.tokens = None
        self.labels = None
        self.parallel_tokenization = parallel_tokenization

        if self.parallel_tokenization:
            if multiprocessing.get_start_method() != 'fork':
                logging.info("Parallel tokenization is only available on Unix systems which allow to fork processes. Fall back to sequential tokenization")
                self.parallel_tokenization = False

        self.examples = examples
        self.convert_input_examples()

    def convert_input_examples(self):
        """
        Converts input examples to a SmartBatchingDataset usable to train the model with
        SentenceTransformer.smart_batching_collate as the collate_fn for the DataLoader

        smart_batching_collate as collate_fn is required because it transforms the tokenized texts to the tensors.

        :param examples:
            the input examples for the training
        :param model
            the Sentence BERT model for the conversion

        """
        num_texts = len(self.examples[0].texts)
        inputs = [[] for _ in range(num_texts)]
        labels = []
        too_long = [0] * num_texts
        label_type = None
        max_seq_length = self.model.get_max_seq_length()

        logging.info("Start tokenization")
        if not self.parallel_tokenization or self.max_processes == 1 or len(self.examples) <= self.chunk_size:
            tokenized_texts = [self.tokenize_example(example) for example in self.examples]
        else:
            logging.info("Use multi-process tokenization with {} processes".format(self.max_processes))
            self.model.to('cpu')
            with Pool(self.max_processes) as p:
                tokenized_texts = list(p.imap(self.tokenize_example, self.examples, chunksize=self.chunk_size))

        for ex_index, example in enumerate(self.examples):
            if label_type is None:
                if isinstance(example.label, int):
                    label_type = torch.long
                elif isinstance(example.label, float):
                    label_type = torch.float

            example.texts_tokenized = tokenized_texts[ex_index]

            for i, token in enumerate(example.texts_tokenized):
                if max_seq_length is not None and max_seq_length > 0 and len(token) >= max_seq_length:
                    too_long[i] += 1

            labels.append(example.label)
            for i in range(num_texts):
                inputs[i].append(example.texts_tokenized[i])

        tensor_labels = torch.tensor(labels, dtype=label_type)

        logging.info("Num sentences: %d" % (len(self.examples)))
        for i in range(num_texts):
            logging.info("Sentences {} longer than max_seqence_length: {}".format(i, too_long[i]))

        self.tokens = inputs
        self.labels = tensor_labels


    def tokenize_example(self, example):
        if example.texts_tokenized is not None:
            return example.texts_tokenized

        return [self.model.tokenize(text) for text in example.texts]


    def __getitem__(self, item):
        return [self.tokens[i][item] for i in range(len(self.tokens))], self.labels[item]


    def __len__(self):
        return len(self.tokens[0])
