from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import TrainingArguments as TransformersTrainingArguments
from transformers.training_args import ParallelMode
from transformers.utils import ExplicitEnum

logger = logging.getLogger(__name__)


class BatchSamplers(ExplicitEnum):
    """
    Stores the acceptable string identifiers for batch samplers.

    The batch sampler is responsible for determining how samples are grouped into batches during training.
    Valid options are:

    - ``BatchSamplers.BATCH_SAMPLER``: **[default]** Uses :class:`~sentence_transformers.sampler.DefaultBatchSampler`, the default
      PyTorch batch sampler.
    - ``BatchSamplers.NO_DUPLICATES``: Uses :class:`~sentence_transformers.sampler.NoDuplicatesBatchSampler`,
      ensuring no duplicate samples in a batch. Recommended for losses that use in-batch negatives, such as:

        - :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`
        - :class:`~sentence_transformers.losses.CachedMultipleNegativesRankingLoss`
        - :class:`~sentence_transformers.losses.MultipleNegativesSymmetricRankingLoss`
        - :class:`~sentence_transformers.losses.CachedMultipleNegativesSymmetricRankingLoss`
        - :class:`~sentence_transformers.losses.MegaBatchMarginLoss`
        - :class:`~sentence_transformers.losses.GISTEmbedLoss`
        - :class:`~sentence_transformers.losses.CachedGISTEmbedLoss`
    - ``BatchSamplers.GROUP_BY_LABEL``: Uses :class:`~sentence_transformers.sampler.GroupByLabelBatchSampler`,
      ensuring that each batch has 2+ samples from the same label. Recommended for losses that require multiple
      samples from the same label, such as:

        - :class:`~sentence_transformers.losses.BatchAllTripletLoss`
        - :class:`~sentence_transformers.losses.BatchHardSoftMarginTripletLoss`
        - :class:`~sentence_transformers.losses.BatchHardTripletLoss`
        - :class:`~sentence_transformers.losses.BatchSemiHardTripletLoss`

    If you want to use a custom batch sampler, you can create a new Trainer class that inherits from
    :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` and overrides the
    :meth:`~sentence_transformers.trainer.SentenceTransformerTrainer.get_batch_sampler` method. The
    method must return a class instance that supports ``__iter__`` and ``__len__`` methods. The former
    should yield a list of indices for each batch, and the latter should return the number of batches.

    Usage:
        ::

            from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
            from sentence_transformers.training_args import BatchSamplers
            from sentence_transformers.losses import MultipleNegativesRankingLoss
            from datasets import Dataset

            model = SentenceTransformer("microsoft/mpnet-base")
            train_dataset = Dataset.from_dict({
                "anchor": ["It's nice weather outside today.", "He drove to work."],
                "positive": ["It's so sunny.", "He took the car to the office."],
            })
            loss = MultipleNegativesRankingLoss(model)
            args = SentenceTransformerTrainingArguments(
                output_dir="checkpoints",
                batch_sampler=BatchSamplers.NO_DUPLICATES,
            )
            trainer = SentenceTransformerTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                loss=loss,
            )
            trainer.train()
    """

    BATCH_SAMPLER = "batch_sampler"
    NO_DUPLICATES = "no_duplicates"
    GROUP_BY_LABEL = "group_by_label"


class MultiDatasetBatchSamplers(ExplicitEnum):
    """
    Stores the acceptable string identifiers for multi-dataset batch samplers.

    The multi-dataset batch sampler is responsible for determining in what order batches are sampled from multiple
    datasets during training. Valid options are:

    - ``MultiDatasetBatchSamplers.ROUND_ROBIN``: Uses :class:`~sentence_transformers.sampler.RoundRobinBatchSampler`,
      which uses round-robin sampling from each dataset until one is exhausted.
      With this strategy, it's likely that not all samples from each dataset are used, but each dataset is sampled
      from equally.
    - ``MultiDatasetBatchSamplers.PROPORTIONAL``: **[default]** Uses :class:`~sentence_transformers.sampler.ProportionalBatchSampler`,
      which samples from each dataset in proportion to its size.
      With this strategy, all samples from each dataset are used and larger datasets are sampled from more frequently.

    Usage:
        ::

            from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
            from sentence_transformers.training_args import MultiDatasetBatchSamplers
            from sentence_transformers.losses import CoSENTLoss
            from datasets import Dataset, DatasetDict

            model = SentenceTransformer("microsoft/mpnet-base")
            train_general = Dataset.from_dict({
                "sentence_A": ["It's nice weather outside today.", "He drove to work."],
                "sentence_B": ["It's so sunny.", "He took the car to the bank."],
                "score": [0.9, 0.4],
            })
            train_medical = Dataset.from_dict({
                "sentence_A": ["The patient has a fever.", "The doctor prescribed medication.", "The patient is sweating."],
                "sentence_B": ["The patient feels hot.", "The medication was given to the patient.", "The patient is perspiring."],
                "score": [0.8, 0.6, 0.7],
            })
            train_legal = Dataset.from_dict({
                "sentence_A": ["This contract is legally binding.", "The parties agree to the terms and conditions."],
                "sentence_B": ["Both parties acknowledge their obligations.", "By signing this agreement, the parties enter into a legal relationship."],
                "score": [0.7, 0.8],
            })
            train_dataset = DatasetDict({
                "general": train_general,
                "medical": train_medical,
                "legal": train_legal,
            })

            loss = CoSENTLoss(model)
            args = SentenceTransformerTrainingArguments(
                output_dir="checkpoints",
                multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
            )
            trainer = SentenceTransformerTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                loss=loss,
            )
            trainer.train()
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
        prompts (`Union[Dict[str, Dict[str, str]], Dict[str, str], str]`, *optional*):
            The prompts to use for each column in the training, evaluation and test datasets. Four formats are accepted:

            1. `str`: A single prompt to use for all columns in the datasets, regardless of whether the training/evaluation/test
               datasets are :class:`datasets.Dataset` or a :class:`datasets.DatasetDict`.
            2. `Dict[str, str]`: A dictionary mapping column names to prompts, regardless of whether the training/evaluation/test
               datasets are :class:`datasets.Dataset` or a :class:`datasets.DatasetDict`.
            3. `Dict[str, str]`: A dictionary mapping dataset names to prompts. This should only be used if your training/evaluation/test
               datasets are a :class:`datasets.DatasetDict` or a dictionary of :class:`datasets.Dataset`.
            4. `Dict[str, Dict[str, str]]`: A dictionary mapping dataset names to dictionaries mapping column names to
               prompts. This should only be used if your training/evaluation/test datasets are a
               :class:`datasets.DatasetDict` or a dictionary of :class:`datasets.Dataset`.

        batch_sampler (Union[:class:`~sentence_transformers.training_args.BatchSamplers`, `str`], *optional*):
            The batch sampler to use. See :class:`~sentence_transformers.training_args.BatchSamplers` for valid options.
            Defaults to ``BatchSamplers.BATCH_SAMPLER``.
        multi_dataset_batch_sampler (Union[:class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers`, `str`], *optional*):
            The multi-dataset batch sampler to use. See :class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers`
            for valid options. Defaults to ``MultiDatasetBatchSamplers.PROPORTIONAL``.
    """

    prompts: Optional[str] = field(  # noqa: UP007
        default=None,
        metadata={
            "help": "The prompts to use for each column in the datasets. "
            "Either 1) a single string prompt, 2) a mapping of column names to prompts, 3) a mapping of dataset names "
            "to prompts, or 4) a mapping of dataset names to a mapping of column names to prompts."
        },
    )
    batch_sampler: Union[BatchSamplers, str] = field(  # noqa: UP007
        default=BatchSamplers.BATCH_SAMPLER, metadata={"help": "The batch sampler to use."}
    )
    multi_dataset_batch_sampler: Union[MultiDatasetBatchSamplers, str] = field(  # noqa: UP007
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
