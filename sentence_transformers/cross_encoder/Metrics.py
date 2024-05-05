from enum import Enum


class Metric(Enum):
    AVERAGE_PRECISION = "average_precision"
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    MRR = "mrr"
    NDCG = "ndcg"
