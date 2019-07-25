"""
This file contains various data readers for common NLP datasets
"""
from . import InputExample
import csv
import gzip
import os


class NLIDataReader(object):
    """Reads in the Stanford NLI dataset and the MultiGenre NLI dataset"""
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_examples(self, filename, max_examples=0):
        """
        data_splits specified which data split to use (train, dev, test).
        Expects that self.dataset_folder contains the files s1.$data_split.gz,  s2.$data_split.gz,
        labels.$data_split.gz, e.g., for the train split, s1.train.gz, s2.train.gz, labels.train.gz
        """
        s1 = gzip.open(os.path.join(self.dataset_folder, 's1.' + filename),
                       mode="rt", encoding="utf-8").readlines()
        s2 = gzip.open(os.path.join(self.dataset_folder, 's2.' + filename),
                       mode="rt", encoding="utf-8").readlines()
        labels = gzip.open(os.path.join(self.dataset_folder, 'labels.' + filename),
                           mode="rt", encoding="utf-8").readlines()

        examples = []
        id = 0
        for sentence_a, sentence_b, label in zip(s1, s2, labels):
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=self.map_label(label)))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]


class TripletReader(object):
    """
    Reads in the a Triplet Dataset: Each line contains (at least) 3 columns, one anchor column (s1),
    one positive example (s2) and one negative example (s3)
    """
    def __init__(self, dataset_folder, s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, has_header=False, delimiter="\t",
                 quoting=csv.QUOTE_NONE):
        self.dataset_folder = dataset_folder
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.s3_col_idx = s3_col_idx
        self.has_header = has_header
        self.delimiter = delimiter
        self.quoting = quoting

    def get_examples(self, filename, max_examples=0):
        """

        """
        data = csv.reader(open(os.path.join(self.dataset_folder, filename), encoding="utf-8"), delimiter=self.delimiter,
                          quoting=self.quoting)
        examples = []
        if self.has_header:
            next(data)

        for id, row in enumerate(data):
            s1 = row[self.s1_col_idx]
            s2 = row[self.s2_col_idx]
            s3 = row[self.s3_col_idx]

            examples.append(InputExample(guid=filename+str(id), texts=[s1, s2, s3], label=1))
            if max_examples > 0 and len(examples) >= max_examples:
                break

        return examples


class STSDataReader:
    """
    Reads in the STS dataset. Each line contains two sentences (s1_col_idx, s2_col_idx) and one label (score_col_idx)
    """
    def __init__(self, dataset_folder, s1_col_idx=5, s2_col_idx=6, score_col_idx=4, delimiter="\t",
                 quoting=csv.QUOTE_NONE, normalize_scores=True, min_score=0, max_score=5):
        self.dataset_folder = dataset_folder
        self.score_col_idx = score_col_idx
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.delimiter = delimiter
        self.quoting = quoting
        self.normalize_scores = normalize_scores
        self.min_score = min_score
        self.max_score = max_score

    def get_examples(self, filename, max_examples=0):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        data = csv.reader(open(os.path.join(self.dataset_folder, filename), encoding="utf-8"),
                          delimiter=self.delimiter, quoting=self.quoting)
        examples = []
        for id, row in enumerate(data):
            score = float(row[self.score_col_idx])
            if self.normalize_scores:  # Normalize to a 0...1 value
                score = (score - self.min_score) / (self.max_score - self.min_score)

            s1 = row[self.s1_col_idx]
            s2 = row[self.s2_col_idx]
            examples.append(InputExample(guid=filename+str(id), texts=[s1, s2], label=score))

            if max_examples > 0 and len(examples) >= max_examples:
                break

        return examples


class LabelSentenceReader:
    """Reads in a file that has at least two columns: a label and a sentence.
    This reader can for example be used with the BatchHardTripletLoss.
    Maps labels automatically to integers"""
    def __init__(self, folder, label_col_idx=0, sentence_col_idx=1):
        self.folder = folder
        self.label_map = {}
        self.label_col_idx = label_col_idx
        self.sentence_col_idx = sentence_col_idx

    def get_examples(self, filename, max_examples=0):
        examples = []

        id = 0
        for line in open(os.path.join(self.folder, filename), encoding="utf-8"):
            splits = line.strip().split('\t')
            label = splits[self.label_col_idx]
            sentence = splits[self.sentence_col_idx]

            if label not in self.label_map:
                self.label_map[label] = len(self.label_map)

            label_id = self.label_map[label]
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence], label=label_id))

            if 0 < max_examples <= id:
                break

        return examples
