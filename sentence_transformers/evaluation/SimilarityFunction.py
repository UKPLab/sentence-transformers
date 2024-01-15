from enum import Enum


class SimilarityFunction(Enum):
    COSINE = 0
    EUCLIDEAN = 1
    MANHATTAN = 2
    DOT_PRODUCT = 3
