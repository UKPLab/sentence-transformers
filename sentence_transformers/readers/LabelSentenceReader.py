from . import InputExample
import csv
import gzip
import os

class LabelSentenceReader:
    """Reads in a file that has at least two columns: a label and a sentence.
    This reader can for example be used with the BatchHardTripletLoss.
    Maps labels automatically to integers"""
    def __init__(self, folder, label_col_idx=0, sentence_col_idx=1, separator='\t'):
        self.folder = folder
        self.label_map = {}
        self.label_col_idx = label_col_idx
        self.sentence_col_idx = sentence_col_idx
        self.separator = separator

    def get_examples(self, filename, max_examples=0):
        examples = []

        id = 0
        for line in open(os.path.join(self.folder, filename), encoding="utf-8"):
            splits = line.strip().split(self.separator)
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
