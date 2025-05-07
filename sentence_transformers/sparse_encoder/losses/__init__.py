from __future__ import annotations

from .CSRLoss import CSRLoss, CSRReconstructionLoss
from .RegularizerLoss import FlopsLoss, L0FlopsLoss
from .SparseAnglELoss import SparseAnglELoss
from .SparseCachedGISTEmbedLoss import SparseCachedGISTEmbedLoss
from .SparseCachedMultipleNegativesRankingLoss import SparseCachedMultipleNegativesRankingLoss
from .SparseCoSENTLoss import SparseCoSENTLoss
from .SparseCosineSimilarityLoss import SparseCosineSimilarityLoss
from .SparseDistillKLDivLoss import SparseDistillKLDivLoss
from .SparseGISTEmbedLoss import SparseGISTEmbedLoss
from .SparseMarginMSELoss import SparseMarginMSELoss
from .SparseMSELoss import SparseMSELoss
from .SparseMultipleNegativesRankingLoss import SparseMultipleNegativesRankingLoss
from .SparseTripletLoss import SparseTripletLoss
from .SpladeLoss import SpladeLoss

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
    "SparseDistillKLDivLoss",
    "FlopsLoss",
    "L0FlopsLoss",
    "SpladeLoss",
]
# TODO: Test cached losses
