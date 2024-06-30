import pytest
from datasets import Dataset
from sentence_transformers.sampler import RoundRobinBatchSampler

from torch.utils.data import BatchSampler, SequentialSampler, ConcatDataset


@pytest.fixture
def dummy_dataset() -> ConcatDataset:
    """
    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "data": [0, 1, 2, ... , 23, 24, 100, 101, ..., 123, 124],
        "label": [0, 1, 0, 1, ..., 0, 1],
    }
    """

    values_1 = list(range(25))
    labels = [x % 2 for x in values_1]
    dataset_1 = Dataset.from_dict({"data": values_1, "label": labels})

    values_2 = [x + 100 for x in values_1]
    dataset_2 = Dataset.from_dict({"data": values_2, "label": labels})

    return ConcatDataset([dataset_1, dataset_2])


def test_round_robin_batch_sampler(dummy_dataset):
    batch_sampler_1 = BatchSampler(SequentialSampler(range(len(dummy_dataset))), batch_size=4, drop_last=True)
    batch_sampler_2 = BatchSampler(SequentialSampler(range(len(dummy_dataset))), batch_size=4, drop_last=True)
    sampler = RoundRobinBatchSampler(dataset=dummy_dataset, batch_samplers=[batch_sampler_1, batch_sampler_2])

    batches = list(iter(sampler))

    for batch in batches:
        for idx in batch:
            print(dummy_dataset[idx])
