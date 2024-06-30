import pytest
from datasets import Dataset
from sentence_transformers.sampler import RoundRobinBatchSampler

from torch.utils.data import BatchSampler, SequentialSampler, ConcatDataset

DATASET_LENGTH = 25


@pytest.fixture
def dummy_dataset() -> ConcatDataset:
    """
    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "data": [0, 1, 2, ... , 23, 24, 100, 101, ..., 123, 124],
        "label": [0, 1, 0, 1, ..., 0, 1],
    }
    """
    values_1 = list(range(DATASET_LENGTH))
    labels = [x % 2 for x in values_1]
    dataset_1 = Dataset.from_dict({"data": values_1, "label": labels})

    values_2 = [x + 100 for x in values_1]
    dataset_2 = Dataset.from_dict({"data": values_2, "label": labels})

    return ConcatDataset([dataset_1, dataset_2])


def test_round_robin_batch_sampler(dummy_dataset):
    batch_size = 4
    batch_sampler_1 = BatchSampler(SequentialSampler(range(DATASET_LENGTH)), batch_size=batch_size, drop_last=True)
    batch_sampler_2 = BatchSampler(SequentialSampler(range(DATASET_LENGTH)), batch_size=batch_size, drop_last=True)

    sampler = RoundRobinBatchSampler(dataset=dummy_dataset, batch_samplers=[batch_sampler_1, batch_sampler_2])
    batches = list(iter(sampler))

    assert len(batches) == 2 * DATASET_LENGTH // batch_size

    # Assert that batches are produced in a round-robin fashion
    for i in range(0, len(batches), 2):
        # Batch from the first part of the dataset
        batch_1 = batches[i]
        assert all(
            dummy_dataset[idx]["data"] < 100 for idx in batch_1
        ), f"Batch {i} contains data from the second part of the dataset: {[dummy_dataset[idx]['data'] for idx in batch_1]}"

        # Batch from the second part of the dataset
        batch_2 = batches[i + 1]
        assert all(
            dummy_dataset[idx]["data"] >= 100 for idx in batch_2
        ), f"Batch {i+1} contains data from the first part of the dataset: {[dummy_dataset[idx]['data'] for idx in batch_2]}"
