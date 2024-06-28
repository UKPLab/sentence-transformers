import logging
from collections import defaultdict
from itertools import accumulate, cycle
from typing import Iterator, List, Any

import torch
from torch.utils.data import BatchSampler, ConcatDataset, SubsetRandomSampler

from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset

logger = logging.getLogger(__name__)


class SetEpochMixin:
    """
    Required for a BatchSampler as the Trainer will call set_epoch on the BatchSampler at the beginning of each epoch.
    The BatchSampler can then set the generator seed accordingly.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DefaultBatchSampler(SetEpochMixin, BatchSampler):
    pass


class GroupByLabelBatchSampler(SetEpochMixin, BatchSampler):
    """
    This sampler groups samples by their labels and aims to create batches such that
    each batch contains samples where the labels are as homogeneous as possible.
    This sampler is meant to be used alongside the `Batch...TripletLoss` classes, which
    require that each batch contains at least 2 examples per label class.

    Args:
        dataset (Dataset): The dataset to sample from.
        batch_size (int): Number of samples per batch. Must be divisible by 2.
        drop_last (bool): If True, drop the last incomplete batch, if the dataset size
            is not divisible by the batch size.
        valid_label_columns (List[str]): List of column names to check for labels.
            The first column name found in the dataset will
            be used as the label column.
        generator (torch.Generator, optional): Optional random number generator for shuffling
            the indices.
        seed (int, optional): Seed for the random number generator to ensure reproducibility.
    """

    def __init__(
        self,
        dataset: "Dataset",
        batch_size: int,
        drop_last: bool,
        valid_label_columns: List[str] = None,
        generator: torch.Generator = None,
        seed: int = 0,
    ) -> None:
        super().__init__(dataset, batch_size, drop_last)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator
        self.seed = seed

        if self.batch_size % 2 == 1:
            raise ValueError("The batch size for `GroupByLabelBatchSampler` must be divisible by 2.")

        labels = self._determine_labels_to_use(dataset, valid_label_columns)
        groups = defaultdict(list)
        for sample_idx, label in enumerate(labels):
            groups[label].append(sample_idx)

        self.groups = {
            label: sample_indices[: len(sample_indices) // 2 * 2] for label, sample_indices in groups.items()
        }

    @staticmethod
    def _determine_labels_to_use(dataset: "Dataset", valid_label_columns: List[str]) -> List[Any]:
        for column_name in valid_label_columns or []:
            if column_name in dataset.column_names:
                return dataset[column_name]
        else:
            raise ValueError(f"None of the valid_label_columns {valid_label_columns} are in the dataset.")

    def __iter__(self) -> Iterator[List[int]]:
        if self.generator and self.seed:
            self.generator.manual_seed(self.seed + self.epoch)

        partial_batch = []
        unique_labels = list(self.groups.keys())
        for label_idx in torch.randperm(len(self.groups), generator=self.generator):
            label = unique_labels[label_idx]
            samples = self.groups[label]
            partial_batch.extend(samples)
            while len(partial_batch) >= self.batch_size:
                yield partial_batch[: self.batch_size]
                partial_batch = partial_batch[self.batch_size :]

        if not self.drop_last and partial_batch:
            yield partial_batch


class NoDuplicatesBatchSampler(SetEpochMixin, BatchSampler):
    def __init__(
        self,
        dataset: "Dataset",
        batch_size: int,
        drop_last: bool,
        valid_label_columns: List[str] = [],
        generator: torch.Generator = None,
        seed: int = 0,
    ) -> None:
        super().__init__(dataset, batch_size, drop_last)
        if label_columns := set(dataset.column_names) & (set(valid_label_columns) | {"dataset_name"}):
            dataset = dataset.remove_columns(label_columns)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator
        self.seed = seed

    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over the remaining non-yielded indices. For each index, check if the sample values are already in the
        batch. If not, add the sample values to the batch keep going until the batch is full. If the batch is full, yield
        the batch indices and continue with the next batch.
        """
        if self.generator and self.seed:
            self.generator.manual_seed(self.seed + self.epoch)

        remaining_indices = set(torch.randperm(len(self.dataset), generator=self.generator).tolist())
        while remaining_indices:
            batch_values = set()
            batch_indices = []
            for index in remaining_indices:
                sample_values = set(self.dataset[index].values())
                if sample_values & batch_values:
                    continue

                batch_indices.append(index)
                if len(batch_indices) == self.batch_size:
                    yield batch_indices
                    break

                batch_values.update(sample_values)

            else:
                # NOTE: some indices might still have been ignored here
                if not self.drop_last:
                    yield batch_indices

            remaining_indices -= set(batch_indices)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class RoundRobinBatchSampler(SetEpochMixin, BatchSampler):
    def __init__(
        self,
        dataset: ConcatDataset,
        batch_samplers: List[BatchSampler],
        generator: torch.Generator,
        seed: int,
    ) -> None:
        super().__init__(dataset, batch_samplers[0].batch_size, batch_samplers[0].drop_last)
        self.dataset = dataset
        self.batch_samplers = batch_samplers
        self.generator = generator
        self.seed = seed

    def __iter__(self) -> Iterator[List[int]]:
        self.generator.manual_seed(self.seed + self.epoch)

        num_samples = [len(dataset) for dataset in self.dataset.datasets]
        sample_offsets = [0] + list(accumulate(num_samples))

        batch_samplers = [iter(sampler) for sampler in self.batch_samplers]
        for dataset_idx in cycle(range(len(batch_samplers))):
            sample_offset = sample_offsets[dataset_idx]
            try:
                yield [idx + sample_offset for idx in next(batch_samplers[dataset_idx])]

            except StopIteration:
                # current iterator is apparently exhausted
                break

    def __len__(self) -> int:
        return min([len(sampler) for sampler in self.batch_samplers]) * len(self.batch_samplers)


class ProportionalBatchSampler(SetEpochMixin, BatchSampler):
    def __init__(
        self,
        dataset: ConcatDataset,
        batch_samplers: List[BatchSampler],
        generator: torch.Generator,
        seed: int,
    ) -> None:
        super().__init__(dataset, batch_samplers[0].batch_size, batch_samplers[0].drop_last)
        self.dataset = dataset
        self.batch_samplers = batch_samplers
        self.generator = generator
        self.seed = seed

    def __iter__(self) -> Iterator[List[int]]:
        self.generator.manual_seed(self.seed + self.epoch)

        num_samples = [len(dataset) for dataset in self.dataset.datasets]
        sample_offsets = [0] + list(accumulate(num_samples))

        num_batches = [len(sampler) for sampler in self.batch_samplers]
        dataset_indices = [idx for idx, length in enumerate(num_batches) for _ in range(length)]
        dataset_idx_sampler = SubsetRandomSampler(dataset_indices, generator=self.generator)

        batch_samplers = [iter(sampler) for sampler in self.batch_samplers]
        for dataset_idx in dataset_idx_sampler:
            sample_offset = sample_offsets[dataset_idx]
            yield [idx + sample_offset for idx in next(batch_samplers[dataset_idx])]

    def __len__(self) -> int:
        return sum([len(sampler) for sampler in self.batch_samplers])
