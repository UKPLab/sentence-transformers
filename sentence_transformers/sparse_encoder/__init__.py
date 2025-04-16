from __future__ import annotations

from sentence_transformers.sparse_encoder.data_collator import SparseEncoderDataCollator
from sentence_transformers.sparse_encoder.evaluation import (
    SparseBinaryClassificationEvaluator,
    SparseEmbeddingSimilarityEvaluator,
    SparseInformationRetrievalEvaluator,
    SparseMSEEvaluator,
    SparseNanoBEIREvaluator,
    SparseRerankingEvaluator,
    SparseTranslationEvaluator,
    SparseTripletEvaluator,
)
from sentence_transformers.sparse_encoder.losses import (
    CSRLoss,
    CSRReconstructionLoss,
    SparseMultipleNegativesRankingLoss,
)
from sentence_transformers.sparse_encoder.models import CSRSparsity, MLMTransformer, SpladePooling
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
    "MLMTransformer",
    "SpladePooling",
    # Losses
    "CSRLoss",
    "CSRReconstructionLoss",
    "SparseMultipleNegativesRankingLoss",
    # Evaluators
    "SparseBinaryClassificationEvaluator",
    "SparseEmbeddingSimilarityEvaluator",
    "SparseInformationRetrievalEvaluator",
    "SparseMSEEvaluator",
    "SparseNanoBEIREvaluator",
    "SparseTranslationEvaluator",
    "SparseRerankingEvaluator",
    "SparseTripletEvaluator",
]
# TODO : Complete the SparseEncoder class
# TODO : Add tests for all the components
# TODO : Ask Update to TOM on loss to implement
# TODO : Add the equivalent of the quantization file for the sparse encoder
