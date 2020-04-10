from torch.utils.data import Dataset
import torch
import logging
import gzip
import os
import random
from .. import SentenceTransformer


class ParallelSentencesDataset(Dataset):
    """
    This dataset reader can be used to read-in parallel sentences, i.e., it reads in a file with tab-seperated sentences with the same
    sentence in different languages. For example, the file can look like this (EN\tDE\tES):
    hello world     hallo welt  hola mundo
    second sentence zweiter satz    segunda oración

    The sentence in the first column will be mapped to a sentence embedding using the given the embedder. For example,
    embedder is a mono-lingual sentence embedding method for English. The sentences in the other languages will also be
    mapped to this English sentence embedding.

    When getting a sample from the dataset, we get one sentence with the according sentence embedding for this sentence.
    """

    def __init__(self, student_model: SentenceTransformer, teacher_model):
        """
        Parallel sentences dataset reader to train student model given a teacher model
        :param student_model: Student sentence embedding model that should be trained
        :param teacher_model: Teacher model, that provides the sentence embeddings for the first column in the dataset file
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.datasets = []
        self.dataset_indices = []
        self.copy_dataset_indices = []

    def load_data(self, filepath: str, weight: int = 100, max_sentences: int = None, max_sentence_length: int = 128):
        """
        Reads in a tab-seperated .txt/.csv/.tsv or .gz file. The different columns contain the different translations of the sentence in the first column

        :param filepath: Filepath to the file
        :param weight: If more that one dataset is loaded with load_data: With which frequency should data be sampled from this dataset?
        :param max_sentences: Max number of lines to be read from filepath
        :param max_sentence_length: Skip the example if one of the sentences is has more characters than max_sentence_length
        :return:
        """
        sentences_map = {}
        with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath, encoding='utf8') as fIn:
            count = 0
            for line in fIn:
                sentences = line.strip().split("\t")
                sentence_lengths = [len(sent) for sent in sentences]
                if max(sentence_lengths) > max_sentence_length:
                    continue

                eng_sentence = sentences[0]
                if eng_sentence not in sentences_map:
                    sentences_map[eng_sentence] = set()

                for sent in sentences:
                    sentences_map[eng_sentence].add(sent)

                count += 1
                if max_sentences is not None and count >= max_sentences:
                    break

        eng_sentences = list(sentences_map.keys())
        logging.info("Create sentence embeddings for " + os.path.basename(filepath))
        labels = torch.tensor(self.teacher_model.encode(eng_sentences, batch_size=32, show_progress_bar=True),
                              dtype=torch.float)

        data = []
        for idx in range(len(eng_sentences)):
            eng_key = eng_sentences[idx]
            label = labels[idx]
            for sent in sentences_map[eng_key]:
                data.append([[self.student_model.tokenize(sent)], label])

        dataset_id = len(self.datasets)
        self.datasets.append(data)
        self.dataset_indices.extend([dataset_id] * weight)

    def __len__(self):
        return max([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx):
        if len(self.copy_dataset_indices) == 0:
            self.copy_dataset_indices = self.dataset_indices.copy()
            random.shuffle(self.copy_dataset_indices)

        dataset_idx = self.copy_dataset_indices.pop()
        return self.datasets[dataset_idx][idx % len(self.datasets[dataset_idx])]
