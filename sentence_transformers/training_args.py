import logging
from dataclasses import dataclass, field
from typing import Union

from transformers import TrainingArguments as TransformersTrainingArguments
from transformers.training_args import ParallelMode
from transformers.utils import ExplicitEnum

logger = logging.getLogger(__name__)


class BatchSamplers(ExplicitEnum):
    """
    Stores the acceptable string identifiers for batch samplers.

    The batch sampler is responsible for determining how samples are grouped into batches during training.
    Valid options are:

    - ``BatchSamplers.BATCH_SAMPLER``: The default PyTorch batch sampler.
    - ``BatchSamplers.NO_DUPLICATES``: Ensures no duplicate samples in a batch.
    - ``BatchSamplers.GROUP_BY_LABEL``: Ensures each batch has 2+ samples from the same label.
    """

    BATCH_SAMPLER = "batch_sampler"
    NO_DUPLICATES = "no_duplicates"
    GROUP_BY_LABEL = "group_by_label"


class MultiDatasetBatchSamplers(ExplicitEnum):
    """
    Stores the acceptable string identifiers for multi-dataset batch samplers.

    The multi-dataset batch sampler is responsible for determining in what order batches are sampled from multiple
    datasets during training. Valid options are:

    - ``MultiDatasetBatchSamplers.ROUND_ROBIN``: Round-robin sampling from each dataset until one is exhausted.
      With this strategy, it's likely that not all samples from each dataset are used, but each dataset is sampled
      from equally.
    - ``MultiDatasetBatchSamplers.PROPORTIONAL``: Sample from each dataset in proportion to its size [default].
      With this strategy, all samples from each dataset are used and larger datasets are sampled from more frequently.
    """

    ROUND_ROBIN = "round_robin"  # Round-robin sampling from each dataset
    PROPORTIONAL = "proportional"  # Sample from each dataset in proportion to its size [default]


@dataclass
class SentenceTransformerTrainingArguments(TransformersTrainingArguments):
    """
    SentenceTransformerTrainingArguments extends :class:`~transformers.TrainingArguments` with additional arguments
    specific to Sentence Transformers. See :class:`~transformers.TrainingArguments` for the complete list of
    available arguments.

    Args:
        output_dir (`str`):
            The output directory where the model checkpoints will be written.
        batch_sampler (Union[:class:`~sentence_transformers.training_args.BatchSamplers`, `str`], *optional*):
            The batch sampler to use. See :class:`~sentence_transformers.training_args.BatchSamplers` for valid options.
            Defaults to ``BatchSamplers.BATCH_SAMPLER``.
        multi_dataset_batch_sampler (Union[:class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers`, `str`], *optional*):
            The multi-dataset batch sampler to use. See :class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers`
            for valid options. Defaults to ``MultiDatasetBatchSamplers.PROPORTIONAL``.
    """

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

        # Disable broadcasting of buffers to avoid `RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation.` when training with DDP & a BertModel-based model.
        self.ddp_broadcast_buffers = False

        if self.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
            # If output_dir is "unused", then this instance is created to compare training arguments vs the defaults,
            # so we don't have to warn.
            if self.output_dir != "unused":
                logger.warning(
                    "Currently using DataParallel (DP) for multi-gpu training, while DistributedDataParallel (DDP) is recommended for faster training. "
                    "See https://sbert.net/docs/sentence_transformer/training/distributed.html for more information."
                )

        elif self.parallel_mode == ParallelMode.DISTRIBUTED and not self.dataloader_drop_last:
            # If output_dir is "unused", then this instance is created to compare training arguments vs the defaults,
            # so we don't have to warn.
            if self.output_dir != "unused":
                logger.warning(
                    "When using DistributedDataParallel (DDP), it is recommended to set `dataloader_drop_last=True` to avoid hanging issues with an uneven last batch. "
                    "Setting `dataloader_drop_last=True`."
                )
            self.dataloader_drop_last = True
