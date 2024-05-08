from dataclasses import dataclass, field
from typing import Union
from transformers import TrainingArguments as TransformersTrainingArguments
from transformers.utils import ExplicitEnum


class BatchSamplers(ExplicitEnum):
    """
    Stores the acceptable string identifiers for batch samplers.
    """

    BATCH_SAMPLER = "batch_sampler"  # Just the default PyTorch batch sampler [default]
    NO_DUPLICATES = "no_duplicates"  # Ensures no duplicate samples in a batch
    GROUP_BY_LABEL = "group_by_label"  # Ensure each batch has 2+ samples from the same label


class MultiDatasetBatchSamplers(ExplicitEnum):
    """
    Stores the acceptable string identifiers for multi-dataset batch samplers.
    """

    ROUND_ROBIN = "round_robin"  # Round-robin sampling from each dataset
    PROPORTIONAL = "proportional"  # Sample from each dataset in proportion to its size [default]


@dataclass
class SentenceTransformerTrainingArguments(TransformersTrainingArguments):
    batch_sampler: Union[BatchSamplers, str] = field(
        default=BatchSamplers.BATCH_SAMPLER, metadata={"help": "The batch sampler to use."}
    )
    multi_dataset_batch_sampler: Union[MultiDatasetBatchSamplers, str] = field(
        default=MultiDatasetBatchSamplers.PROPORTIONAL, metadata={"help": "The multi-dataset batch sampler to use."}
    )

    def __post_init__(self):
        super().__post_init__()

        self.batch_sampler = BatchSamplers(self.batch_sampler)
        self.multi_dataset_batch_sampler = MultiDatasetBatchSamplers(self.multi_dataset_batch_sampler)

        # The `compute_loss` method in `SentenceTransformerTrainer` is overridden to only compute the prediction loss,
        # so we set `prediction_loss_only` to `True` here to avoid
        self.prediction_loss_only = True
