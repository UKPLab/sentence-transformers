from itertools import accumulate, cycle
from typing import Iterable, List, Sequence, TYPE_CHECKING
import logging

from torch.utils.data import BatchSampler, SubsetRandomSampler
import torch

if TYPE_CHECKING:
    from sentence_transformers.trainer import SentenceTransformerTrainer

logger = logging.getLogger(__name__)


# TODO: Round Robin sampling is inherited from `SentenceTransformer.fit`, but we can just use
# a smarter sampler instead, i.e. one that samples from each dataset in proportion to the
# number of samples in each dataset.
class RoundRobinSampler:
    def __init__(self, samplers: Sequence[Iterable], trainer: "SentenceTransformerTrainer" = None):
        """
        a sampler that will cycle through the list of given samplers and
        'forward' the next() call each sampler in turn ("round robin").

        Args:
            samplers (Sequence[Iterable]): the list of samplers that will be cycled
        """
        self.samplers = samplers
        self.dataset_idx = 0
        self.trainer = trainer

    def __iter__(self):
        iterators = [iter(sampler) for sampler in self.samplers]

        for dataset_idx in cycle(range(len(iterators))):
            try:
                yield next(iterators[dataset_idx])
                if self.trainer and self.trainer.training_with_dataset_dict:
                    self.trainer.dataset_idx = dataset_idx
                    self.trainer.dataset_name = self.trainer.dataset_names[dataset_idx]

            except StopIteration:
                # current iterator is apparently exhausted
                break


class RoundRobinBatchSampler:
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        drop_last: bool = False,
        seed: int = 0,
        shuffle: bool = True,
        trainer: "SentenceTransformerTrainer" = None,
    ):
        self.lengths = lengths
        accumulated = list(accumulate(self.lengths))
        self.ranges = [(start, end) for start, end in zip([0] + accumulated, accumulated)]
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.batch_size = batch_size
        self.trainer = trainer

    def __iter__(self):
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        batch_samplers = [
            BatchSampler(
                SubsetRandomSampler(range(start, end), generator=g) if self.shuffle else range(start, end),
                self.batch_size,
                self.drop_last,
            )
            for (start, end) in self.ranges
        ]

        self.sampler = RoundRobinSampler(batch_samplers, self.trainer)
        return iter(self.sampler)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        # return min([length // self.batch_size for length in self.lengths])

        if self.drop_last:
            return min([length // self.batch_size for length in self.lengths]) * len(self.lengths)
        else:
            return min([(length + self.batch_size - 1) // self.batch_size for length in self.lengths]) * len(
                self.lengths
            )

    @property
    def dataset_idx(self) -> int:
        return self.sampler.dataset_idx
