from torch.utils.data import Dataset
from typing import List
from ..readers.InputExample import InputExample
import numpy as np
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer

class DenoisingAutoEncoderDataset(Dataset):
    """
    The DenoisingAutoEncoderDataset returns InputExamples in the format: texts=[noise_fn(sentence), sentence]
    It is used in combination with the DenoisingAutoEncoderLoss: Here, a decoder tries to re-construct the
    sentence without noise.

    :param sentences: A list of sentences
    :param noise_fn: A noise function: Given a string, it returns a string with noise, e.g. deleted words
    """
    def __init__(self, sentences: List[str], noise_fn=lambda s: DenoisingAutoEncoderDataset.delete(s)):
        self.sentences = sentences
        self.noise_fn = noise_fn


    def __getitem__(self, item):
        sent = self.sentences[item]
        return InputExample(texts=[self.noise_fn(sent), sent])


    def __len__(self):
        return len(self.sentences)

    # Deletion noise.
    @staticmethod
    def delete(text, del_ratio=0.6):
        words = nltk.word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True # guarantee that at least one word remains
        words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
        return words_processed