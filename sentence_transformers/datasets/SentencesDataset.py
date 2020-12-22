from torch.utils.data import Dataset
from typing import List
import torch
from .. import SentenceTransformer
from ..readers.InputExample import InputExample

class SentencesDataset(Dataset):
    """
    Dataset for smart batching, that is each batch is only padded to its longest sequence instead of padding all
    sequences to the max length.
    The SentenceBertEncoder.smart_batching_collate is required for this to work.
    SmartBatchingDataset does *not* work without it.
    """
    def __init__(self,
                 examples: List[InputExample],
                 model: SentenceTransformer
                 ):
        """
        Create a new SentencesDataset with the tokenized texts and the labels as Tensor

        :param examples
            A list of sentence.transformers.readers.InputExample
        :param model:
            SentenceTransformerModel
        """
        self.examples = examples


    def __getitem__(self, item):
        return self.examples[item]


    def __len__(self):
        return len(self.examples)
