from __future__ import annotations

from sentence_transformers.sparse_encoder.data_collator import SparseEncoderDataCollator
from sentence_transformers.sparse_encoder.evaluation import (
    SparseEmbeddingSimilarityEvaluator,
    SparseInformationRetrievalEvaluator,
)
from sentence_transformers.sparse_encoder.losses import (
    CSRLoss,
    ReconstructionLoss,
    SparseMultipleNegativesRankingLoss,
)
from sentence_transformers.sparse_encoder.models import CSRSparsity, TopKActivation
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder
from sentence_transformers.sparse_encoder.trainer import SparseEncoderTrainer
from sentence_transformers.sparse_encoder.training_args import (
    SparseEncoderTrainingArguments,
)

__all__ = [
    "SparseEncoder",
    "SparseEncoderDataCollator",
    "SparseEncoderTrainer",
    "SparseEncoderTrainingArguments",
    "CSRSparsity",
    "TopKActivation",
    "CSRLoss",
    "ReconstructionLoss",
    "SparseMultipleNegativesRankingLoss",
    "SparseInformationRetrievalEvaluator",
    "SparseEmbeddingSimilarityEvaluator",
]
# TODO : Complete the SparseEncoder class by finishing the overide of the functions
# TODO : Add tests for the SparseEncoder class
# TODO : Add in models the possibility to have the MLM head(for splade)
# TODO : Check for every loss if compatible with the SparseEncoder but should have some minor modifications for the ones not using the utils similarity function
# TODO : Same for the evaluator
# TODO : Add the equivalent of the quantization file for the sparse encoder
