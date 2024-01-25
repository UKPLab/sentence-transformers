from enum import Enum

class SimilarityFunction(Enum):
    COSINE = "cos_sim"
    EUCLIDEAN = "euclidean_sim"
    MANHATTAN = "manhattan_sim"
    DOT_PRODUCT = "dot_prod"
