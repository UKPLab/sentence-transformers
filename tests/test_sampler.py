import pytest
from datasets import Dataset
from sentence_transformers.sampler import GroupByLabelBatchSampler
import numpy as np


@pytest.fixture
def dummy_dataset():
    """

    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "data": [0, 1, 2, ..., 99],
        "label_a": [0, 1, 0, 1, ..., 0, 1],
        "label_b": [0, 1, 2, 3, 0, 1, ..., 3]
    }
    """
    data = {"data": list(range(100)), "label_a": [i % 2 for i in range(100)], "label_b": [i % 4 for i in range(100)]}
    return Dataset.from_dict(data)


def test_group_by_label_batch_sampler_label_a(dummy_dataset):
    batch_size = 10

    sampler = GroupByLabelBatchSampler(
        dataset=dummy_dataset, batch_size=batch_size, drop_last=False, valid_label_columns=["label_a", "label_b"]
    )

    batches = list(iter(sampler))
    assert all(len(batch) == batch_size for batch in batches)

    # Check if all labels within each batch are identical
    # In this case, label_a has 50 0's and 50 1's, so with a batch size of 10 we expect each batch to
    # have only 0's or only 1's.
    for batch in batches:
        labels = [dummy_dataset[int(idx)]["label_a"] for idx in batch]
        assert len(set(labels)) == 1, f"Batch {batch} does not have identical labels: {labels}"


def test_group_by_label_batch_sampler_label_b(dummy_dataset):
    batch_size = 8

    sampler = GroupByLabelBatchSampler(
        dataset=dummy_dataset, batch_size=batch_size, drop_last=True, valid_label_columns=["label_b"]
    )

    # drop_last=True, so each batch should be the same length and the last batch is dropped.
    batches = list(iter(sampler))
    assert all(
        len(batch) == batch_size for batch in batches
    ), "Not all batches are the same size, while drop_last was True."
    assert sum(len(batch) for batch in batches) == 100 - (100 % batch_size)

    # Check if all labels within each batch are identical.
    # Since we have 25 occurrences each of label_b values 0, 1, 2, and 3 and a batch_size of 8, we expect:
    # - For each label, there will be multiple batches where all elements have the same label.
    # - There will be three batches which contain a mix of two labels (where the sampler transitions from one label to the next)
    number_of_unique_labels_per_batch = []
    for batch in batches:
        labels = [dummy_dataset[int(idx)]["label_b"] for idx in batch]
        number_of_unique_labels_per_batch.append(len(np.unique(labels)))
    assert number_of_unique_labels_per_batch.count(2) == 3
