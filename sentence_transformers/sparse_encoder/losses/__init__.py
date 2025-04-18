from __future__ import annotations

from sentence_transformers.sparse_encoder.losses.CSRLoss import CSRLoss
from sentence_transformers.sparse_encoder.losses.CSRReconstructionLoss import (
    CSRReconstructionLoss,
)
from sentence_transformers.sparse_encoder.losses.SparseAnglELoss import SparseAnglELoss
from sentence_transformers.sparse_encoder.losses.SparseCachedGISTEmbedLoss import (
    SparseCachedGISTEmbedLoss,
)
from sentence_transformers.sparse_encoder.losses.SparseCachedMultipleNegativesRankingLoss import (
    SparseCachedMultipleNegativesRankingLoss,
)
from sentence_transformers.sparse_encoder.losses.SparseCoSENTLoss import (
    SparseCoSENTLoss,
)
from sentence_transformers.sparse_encoder.losses.SparseCosineSimilarityLoss import (
    SparseCosineSimilarityLoss,
)
from sentence_transformers.sparse_encoder.losses.SparseGISTEmbedLoss import (
    SparseGISTEmbedLoss,
)
from sentence_transformers.sparse_encoder.losses.SparseMarginMSELoss import (
    SparseMarginMSELoss,
)
from sentence_transformers.sparse_encoder.losses.SparseMSELoss import SparseMSELoss
from sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss import (
    SparseMultipleNegativesRankingLoss,
)
from sentence_transformers.sparse_encoder.losses.SparseTripletLoss import (
    SparseTripletLoss,
)

__all__ = [
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
]
