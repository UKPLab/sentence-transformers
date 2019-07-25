"""
This file contains sampler functions, that can be used to sample mini-batches with specific properties.
"""
from torch.utils.data import Sampler
import numpy as np
from .datasets import SentenceLabelDataset


class LabelSampler(Sampler):
    """
    This sampler is used for some specific Triplet Losses like BATCH_HARD_TRIPLET_LOSS
    or MULTIPLE_NEGATIVES_RANKING_LOSS which require multiple or only one sample from one label per batch.

    It draws n consecutive, random and unique samples from one label at a time. This is repeated for each label.

    Labels with fewer than n unique samples are ignored.
    This also applied to drawing without replacement, once less than n samples remain for a label, it is skipped.

    This *DOES NOT* check if there are more labels than the batch is large or if the batch size is divisible
    by the samples drawn per label.


    """
    def __init__(self, data_source: SentenceLabelDataset, samples_per_label: int = 5,
                 with_replacement: bool = False):
        """
        Creates a LabelSampler for a SentenceLabelDataset.

        :param data_source:
            the dataset from which samples are drawn
        :param samples_per_label:
            the number of consecutive, random and unique samples drawn per label
        :param with_replacement:
            if this is True, then each sample is drawn at most once (depending on the total number of samples per label).
            if this is False, then one sample can be drawn in multiple draws, but still not multiple times in the same
            drawing.
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.samples_per_label = samples_per_label
        self.label_range = np.arange(data_source.num_labels)
        self.borders = data_source.labels_right_border
        self.with_replacement = with_replacement
        np.random.shuffle(self.label_range)

    def __iter__(self):
        label_idx = 0
        count = 0
        already_seen = {}
        while count < len(self.data_source):
            label = self.label_range[label_idx]
            if label not in already_seen:
                already_seen[label] = []

            left_border = 0 if label == 0 else self.borders[label-1]
            right_border = self.borders[label]

            if self.with_replacement:
                selection = np.arange(left_border, right_border)
            else:
                selection = [i for i in np.arange(left_border, right_border) if i not in already_seen[label]]

            if len(selection) >= self.samples_per_label:
                for element_idx in np.random.choice(selection, self.samples_per_label, replace=False):
                    count += 1
                    already_seen[label].append(element_idx)
                    yield element_idx

            label_idx += 1
            if label_idx >= len(self.label_range):
                label_idx = 0
                np.random.shuffle(self.label_range)

    def __len__(self):
        return len(self.data_source)