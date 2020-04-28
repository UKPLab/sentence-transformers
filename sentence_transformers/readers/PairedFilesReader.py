from . import InputExample
import csv
import gzip
import os
import gzip

class PairedFilesReader(object):
    """
    Reads in the a Pair Dataset, split in two files
    """
    def __init__(self, filepaths):
        self.filepaths = filepaths


    def get_examples(self, max_examples=0):
        """
        """
        fIns = []
        for filepath in self.filepaths:
            fIn = gzip.open(filepath, 'rt', encoding='utf-8') if filepath.endswith('.gz') else open(filepath, encoding='utf-8')
            fIns.append(fIn)

        examples = []

        eof = False
        while not eof:
            texts = []
            for fIn in fIns:
                text = fIn.readline()

                if text == '':
                    eof = True
                    break

                texts.append(text)

            if eof:
                break;

            examples.append(InputExample(guid=str(len(examples)), texts=texts, label=1))
            if max_examples > 0 and len(examples) >= max_examples:
                break

        return examples