from typing import List

import numpy as np
from torch.utils.data import Dataset
from transformers.utils.import_utils import NLTK_IMPORT_ERROR, is_nltk_available

from sentence_transformers.readers.InputExample import InputExample


class DenoisingAutoEncoderDataset(Dataset):
    """
    The DenoisingAutoEncoderDataset returns InputExamples in the format: texts=[noise_fn(sentence), sentence]
    It is used in combination with the DenoisingAutoEncoderLoss: Here, a decoder tries to re-construct the
    sentence without noise.

    Args:
        sentences: A list of sentences
        noise_fn: A noise function: Given a string, it returns a string
            with noise, e.g. deleted words
    """

    def __init__(self, sentences: List[str], noise_fn=lambda s: DenoisingAutoEncoderDataset.delete(s)):
        if not is_nltk_available():
            raise ImportError(NLTK_IMPORT_ERROR.format(self.__class__.__name__))

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
        from nltk import TreebankWordDetokenizer, word_tokenize

        words = word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True  # guarantee that at least one word remains
        words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
        return words_processed
