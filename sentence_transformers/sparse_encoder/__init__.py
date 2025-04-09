from __future__ import annotations

from sentence_transformers.sparse_encoder.data_collator import SparseEncoderDataCollator
from sentence_transformers.sparse_encoder.evaluation import (
    SparseBinaryClassificationEvaluator,
    SparseEmbeddingSimilarityEvaluator,
    SparseInformationRetrievalEvaluator,
    SparseMSEEvaluator,
    SparseNanoBEIREvaluator,
    SparseTripletEvaluator,
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
    # Core components
    "SparseEncoder",
    "SparseEncoderDataCollator",
    "SparseEncoderTrainer",
    "SparseEncoderTrainingArguments",
    # Models
    "CSRSparsity",
    "TopKActivation",
    # Losses
    "CSRLoss",
    "ReconstructionLoss",
    "SparseMultipleNegativesRankingLoss",
    # Evaluators
    "SparseBinaryClassificationEvaluator",
    "SparseEmbeddingSimilarityEvaluator",
    "SparseInformationRetrievalEvaluator",
    "SparseMSEEvaluator",
    "SparseNanoBEIREvaluator",
    "SparseTripletEvaluator",
]
# TODO : Complete the SparseEncoder class
# TODO : Add tests for all the components
# TODO : Add in models the possibility to have the MLM head(for splade)
# TODO : Check for every loss if compatible with the SparseEncoder but should have some minor modifications for the ones not using the utils similarity function
# TODO : Same for the evaluator
# TODO : Add the equivalent of the quantization file for the sparse encoder
