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
    def map_to_function(score_function_name):
        if score_function_name == SimilarityFunction.COSINE:
            return cos_sim
        elif score_function_name == SimilarityFunction.MANHATTAN:
            return manhattan_sim
        elif score_function_name == SimilarityFunction.EUCLIDEAN:
            return euclidean_sim
        elif score_function_name == SimilarityFunction.DOT_SCORE:
            return dot_score
        else:
            raise ValueError("""
                The provided function name is not supported. 
                Use of the supported values: {}.""".format([m.value for m in SimilarityFunction])
            )
        
    @staticmethod
    def possible_values():
        return [m.value for m in SimilarityFunction]