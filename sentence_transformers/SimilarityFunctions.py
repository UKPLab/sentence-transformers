from enum import Enum
from .util import (
    cos_sim,
    manhattan_sim,
    euclidean_sim,
    dot_score
)

class SimilarityFunction(Enum):
    COSINE = "cos_sim"
    EUCLIDEAN = "euclidean_sim"
    MANHATTAN = "manhattan_sim"
    DOT_SCORE = "dot_score"

    @staticmethod
    def map_to_function(score_function):
        if isinstance(score_function, Enum):
            score_function = score_function.value

        if score_function == SimilarityFunction.COSINE.value:
            return cos_sim
        elif score_function == SimilarityFunction.MANHATTAN.value:
            return manhattan_sim
        elif score_function == SimilarityFunction.EUCLIDEAN.value:
            return euclidean_sim
        elif score_function == SimilarityFunction.DOT_SCORE.value:
            return dot_score
        else:
            raise ValueError("The provided function is not supported. Use one of the supported values: {}.".format([m.value for m in SimilarityFunction]))
        
    @staticmethod
    def possible_values():
        return [m.value for m in SimilarityFunction]