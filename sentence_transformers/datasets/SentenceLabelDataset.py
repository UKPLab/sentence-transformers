from torch.utils.data import Dataset
from typing import List
import bisect
import torch
import logging
import numpy as np
from tqdm import tqdm
from .. import SentenceTransformer
from ..readers.InputExample import InputExample
from multiprocessing import Pool, cpu_count
import multiprocessing

class SentenceLabelDataset(Dataset):
    """
    Dataset for training with triplet loss.
    This dataset takes a list of sentences grouped by their label and uses this grouping to dynamically select a
    positive example from the same group and a negative example from the other sentences for a selected anchor sentence.

    This dataset should be used in combination with dataset_reader.LabelSentenceReader

    One iteration over this dataset selects every sentence as anchor once.

    This also uses smart batching like SentenceDataset.
    """

    def __init__(self, examples: List[InputExample], model: SentenceTransformer, provide_positive: bool = True,
                 provide_negative: bool = True,
                 parallel_tokenization: bool = True,
                 max_processes: int = 4,
                 chunk_size: int = 5000):
        """
        Converts input examples to a SentenceLabelDataset usable to train the model with
        SentenceTransformer.smart_batching_collate as the collate_fn for the DataLoader

        Assumes only one sentence per InputExample and labels as integers from 0 to max_num_labels
        and should be used in combination with dataset_reader.LabelSentenceReader.

        Labels with only one example are ignored.

        smart_batching_collate as collate_fn is required because it transforms the tokenized texts to the tensors.

        :param examples:
            the input examples for the training
        :param model
            the Sentence BERT model for the conversion
        :param provide_positive:
            set this to False, if you don't need a positive example (e.g. for BATCH_HARD_TRIPLET_LOSS).
        :param provide_negative:
            set this to False, if you don't need a negative example (e.g. for BATCH_HARD_TRIPLET_LOSS
            or MULTIPLE_NEGATIVES_RANKING_LOSS).
        :param parallel_tokenization
            If true, multiple processes will be started for the tokenization
        :param max_processes
            Maximum number of processes started for tokenization. Cannot be larger can cpu_count()
        :param chunk_size
            #chunk_size number of examples are send to each process. Larger values increase overall tokenization speed
        """
        self.model = model
        self.groups_right_border = []
        self.grouped_inputs = []
        self.grouped_labels = []
        self.num_labels = 0
        self.max_processes = min(max_processes, cpu_count())
        self.chunk_size = chunk_size
        self.parallel_tokenization = parallel_tokenization

        if self.parallel_tokenization:
            if multiprocessing.get_start_method() != 'fork':
                logging.info("Parallel tokenization is only available on Unix systems which allow to fork processes. Fall back to sequential tokenization")
                self.parallel_tokenization = False

        self.convert_input_examples(examples, model)

        self.idxs = np.arange(len(self.grouped_inputs))

        self.provide_positive = provide_positive
        self.provide_negative = provide_negative


    def convert_input_examples(self, examples: List[InputExample], model: SentenceTransformer):
        """
        Converts input examples to a SentenceLabelDataset.

        Assumes only one sentence per InputExample and labels as integers from 0 to max_num_labels
        and should be used in combination with dataset_reader.LabelSentenceReader.

        Labels with only one example are ignored.

        :param examples:
            the input examples for the training
        :param model
            the Sentence Transformer model for the conversion
        :param is_pretokenized
            If set to true, no tokenization will be applied. It is expected that the input is tokenized via model.tokenize
        """

        inputs = []
        labels = []

        label_sent_mapping = {}
        too_long = 0
        label_type = None

        logging.info("Start tokenization")
        if not self.parallel_tokenization or self.max_processes == 1 or len(examples) <= self.chunk_size:
            tokenized_texts = [self.tokenize_example(example) for example in examples]
        else:
            logging.info("Use multi-process tokenization with {} processes".format(self.max_processes))
            self.model.to('cpu')
            with Pool(self.max_processes) as p:
                tokenized_texts = list(p.imap(self.tokenize_example, examples, chunksize=self.chunk_size))

        # Group examples and labels
        # Add examples with the same label to the same dict
        for ex_index, example in enumerate(tqdm(examples, desc="Convert dataset")):
            if label_type is None:
                if isinstance(example.label, int):
                    label_type = torch.long
                elif isinstance(example.label, float):
                    label_type = torch.float
            tokenized_text = tokenized_texts[ex_index][0]

            if hasattr(model, 'max_seq_length') and model.max_seq_length is not None and model.max_seq_length > 0 and len(tokenized_text) > model.max_seq_length:
                too_long += 1

            if example.label in label_sent_mapping:
                label_sent_mapping[example.label].append(ex_index)
            else:
                label_sent_mapping[example.label] = [ex_index]

            inputs.append(tokenized_text)
            labels.append(example.label)

        # Group sentences, such that sentences with the same label
        # are besides each other. Only take labels with at least 2 examples
        distinct_labels = list(label_sent_mapping.keys())
        for i in range(len(distinct_labels)):
            label = distinct_labels[i]
            if len(label_sent_mapping[label]) >= 2:
                self.grouped_inputs.extend([inputs[j] for j in label_sent_mapping[label]])
                self.grouped_labels.extend([labels[j] for j in label_sent_mapping[label]])
                self.groups_right_border.append(len(self.grouped_inputs)) #At which position does this label group / bucket end?
                self.num_labels += 1

        self.grouped_labels = torch.tensor(self.grouped_labels, dtype=label_type)
        logging.info("Num sentences: %d" % (len(self.grouped_inputs)))
        logging.info("Sentences longer than max_seqence_length: {}".format(too_long))
        logging.info("Number of labels with >1 examples: {}".format(len(distinct_labels)))


    def tokenize_example(self, example):
        if example.texts_tokenized is not None:
            return example.texts_tokenized

        return [self.model.tokenize(text) for text in example.texts]

    def __getitem__(self, item):
        if not self.provide_positive and not self.provide_negative:
            return [self.grouped_inputs[item]], self.grouped_labels[item]

        # Anchor element
        anchor = self.grouped_inputs[item]

        # Check start and end position for this label in our list of grouped sentences
        group_idx = bisect.bisect_right(self.groups_right_border, item)
        left_border = 0 if group_idx == 0 else self.groups_right_border[group_idx - 1]
        right_border = self.groups_right_border[group_idx]

        if self.provide_positive:
            positive_item_idx = np.random.choice(np.concatenate([self.idxs[left_border:item], self.idxs[item + 1:right_border]]))
            positive = self.grouped_inputs[positive_item_idx]
        else:
            positive = []

        if self.provide_negative:
            negative_item_idx = np.random.choice(np.concatenate([self.idxs[0:left_border], self.idxs[right_border:]]))
            negative = self.grouped_inputs[negative_item_idx]
        else:
            negative = []

        return [anchor, positive, negative], self.grouped_labels[item]


    def __len__(self):
        return len(self.grouped_inputs)