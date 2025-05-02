from __future__ import annotations

from sentence_transformers.sparse_encoder.callbacks.splade_callbacks import (
    SchedulerType,
    SpladeLambdaSchedulerCallback,
)
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
    FlopsLoss,
    SparseAnglELoss,
    SparseCachedGISTEmbedLoss,
    SparseCachedMultipleNegativesRankingLoss,
    SparseCoSENTLoss,
    SparseCosineSimilarityLoss,
    SparseDistillKLDivLoss,
    SparseGISTEmbedLoss,
    SparseMarginMSELoss,
    SparseMSELoss,
    SparseMultipleNegativesRankingLoss,
    SparseTripletLoss,
    SpladeLoss,
)
from sentence_transformers.sparse_encoder.model_card import SparseEncoderModelCardData
from sentence_transformers.sparse_encoder.models import IDF, CSRSparsity, MLMTransformer, SpladePooling
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
    "IDF",
    # Losses
    "CSRLoss",
    "CSRReconstructionLoss",
    "SparseMultipleNegativesRankingLoss",
    "SparseCoSENTLoss",
    "SparseTripletLoss",
    "SparseCachedMultipleNegativesRankingLoss",
    "SparseMarginMSELoss",
    "SparseGISTEmbedLoss",
    "SparseCachedGISTEmbedLoss",
    "SparseCosineSimilarityLoss",
    "SparseMSELoss",
    "SparseAnglELoss",
    "SparseDistillKLDivLoss",
    "FlopsLoss",
    "SpladeLoss",
    # Callbacks
    "SpladeLambdaSchedulerCallback",
    "SchedulerType",
    # Evaluators
    "SparseBinaryClassificationEvaluator",
    "SparseEmbeddingSimilarityEvaluator",
    "SparseInformationRetrievalEvaluator",
    "SparseMSEEvaluator",
    "SparseNanoBEIREvaluator",
    "SparseTranslationEvaluator",
    "SparseRerankingEvaluator",
    "SparseTripletEvaluator",
    # Model card
    "SparseEncoderModelCardData",
]
# TODO : Add tests for all the components
