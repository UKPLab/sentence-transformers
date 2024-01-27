from enum import Enum

class SimilarityFunction(Enum):
    COSINE_SIMILARITY = "cos_sim"
    EUCLIDEAN_SIMILARITY = "euclidean_sim"
    MANHATTAN_SIMILARITY = "manhattan_sim"
    DOT_SCORE = "dot_score"