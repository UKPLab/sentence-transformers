from __future__ import annotations

from sentence_transformers.losses.TripletLoss import TripletDistanceMetric, TripletLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseTripletLoss(TripletLoss):
    def __init__(
        self, model: SparseEncoder, distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin: float = 5
    ) -> None:
        super().__init__(model, distance_metric=distance_metric, triplet_margin=triplet_margin)
