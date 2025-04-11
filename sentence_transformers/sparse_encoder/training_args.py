from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers.training_args import SentenceTransformerTrainingArguments


@dataclass
class SparseEncoderTrainingArguments(SentenceTransformerTrainingArguments):
    """
    SparseEncoderTrainingArguments extends :class:`~transformers.TrainingArguments` with additional arguments
    specific to Sentence Transformers. See :class:`~transformers.TrainingArguments` for the complete list of
    available arguments.

    Args:

    """
