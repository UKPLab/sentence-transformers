from .CosineSimilarityLoss import CosineSimilarityLoss
from .SoftmaxLoss import SoftmaxLoss
from .MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
from .MultipleNegativesSymmetricRankingLoss import MultipleNegativesSymmetricRankingLoss
from .TripletLoss import TripletDistanceMetric, TripletLoss
from .MarginMSELoss import MarginMSELoss
from .MSELoss import MSELoss
from .CachedMultipleNegativesRankingLoss import CachedMultipleNegativesRankingLoss
from .ContrastiveLoss import SiameseDistanceMetric, ContrastiveLoss
from .ContrastiveTensionLoss import (
    ContrastiveTensionLoss,
    ContrastiveTensionLossInBatchNegatives,
    ContrastiveTensionDataLoader,
)
from .OnlineContrastiveLoss import OnlineContrastiveLoss
from .MegaBatchMarginLoss import MegaBatchMarginLoss
from .DenoisingAutoEncoderLoss import DenoisingAutoEncoderLoss

# Triplet losses
from .BatchHardTripletLoss import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction
from .BatchHardSoftMarginTripletLoss import BatchHardSoftMarginTripletLoss
from .BatchSemiHardTripletLoss import BatchSemiHardTripletLoss
from .BatchAllTripletLoss import BatchAllTripletLoss

__all__ = [
    "CosineSimilarityLoss",
    "SoftmaxLoss",
    "MultipleNegativesRankingLoss",
    "MultipleNegativesSymmetricRankingLoss",
    "TripletLoss",
    "TripletDistanceMetric",
    "MarginMSELoss",
    "MSELoss",
    "ContrastiveLoss",
    "SiameseDistanceMetric",
    "CachedMultipleNegativesRankingLoss",
    "ContrastiveTensionLoss",
    "ContrastiveTensionLossInBatchNegatives",
    "ContrastiveTensionDataLoader",
    "OnlineContrastiveLoss",
    "MegaBatchMarginLoss",
    "DenoisingAutoEncoderLoss",
    "BatchHardTripletLoss",
    "BatchHardTripletLossDistanceFunction",
    "BatchHardSoftMarginTripletLoss",
    "BatchSemiHardTripletLoss",
    "BatchAllTripletLoss",
]
