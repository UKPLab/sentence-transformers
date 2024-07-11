import random

import pytest
from datasets import Dataset

from sentence_transformers.sampler import NoDuplicatesBatchSampler


@pytest.fixture
def dummy_dataset():
    """
    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "data": [0, 47, 3, 30, 3, ... 2],
        "label": [0, 1, 0, 1, ..., 0, 1],
    }
    """

    # Create a list of two 0's, two 1's, two 2's, ... two 49's. Then shuffle.
    values = [j for i in range(50) for j in (i, i)]
    random.shuffle(values)
    data = {"data": values, "label": [i % 2 for i in range(100)]}
    return Dataset.from_dict(data)


def test_group_by_label_batch_sampler_label_a(dummy_dataset):
    batch_size = 10

    sampler = NoDuplicatesBatchSampler(
        dataset=dummy_dataset, batch_size=batch_size, drop_last=True, valid_label_columns=["label"]
    )

    batches = list(iter(sampler))

    # Assert all batch sizes are correct
    assert all(len(batch) == batch_size for batch in batches)

    # Assert batches contain no duplicate values
    for batch in batches:
        batch_values = [dummy_dataset[i]["data"] for i in batch]
        assert len(batch_values) == len(set(batch_values)), f"Batch {batch} contains duplicate values: {batch_values}"
