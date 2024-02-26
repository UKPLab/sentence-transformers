from enum import Enum
from .util import (
    cos_sim_fn,
    manhattan_sim_fn,
    euclidean_sim_fn,
    dot_score_fn,
    pairwise_cos_sim_fn,
    pairwise_dot_score_fn,
    pairwise_manhattan_sim_fn,
    pairwise_euclidean_sim_fn
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
            return cos_sim_fn
        elif score_function == SimilarityFunction.MANHATTAN.value:
            return manhattan_sim_fn
        elif score_function == SimilarityFunction.EUCLIDEAN.value:
            return euclidean_sim_fn
        elif score_function == SimilarityFunction.DOT_SCORE.value:
            return dot_score_fn
        else:
            raise ValueError("The provided function {} is not supported. Use one of the supported values: {}.".format(score_function, SimilarityFunction.possible_values()))

    @staticmethod
    def map_to_pairwise_function(score_function):
        if isinstance(score_function, Enum):
            score_function = score_function.value

        if score_function == SimilarityFunction.COSINE.value:
            return pairwise_cos_sim_fn
        elif score_function == SimilarityFunction.MANHATTAN.value:
            return pairwise_manhattan_sim_fn
        elif score_function == SimilarityFunction.EUCLIDEAN.value:
            return pairwise_euclidean_sim_fn
        elif score_function == SimilarityFunction.DOT_SCORE.value:
            return pairwise_dot_score_fn
        else:
            raise ValueError("The provided function {} is not supported. Use one of the supported values: {}.".format(score_function, SimilarityFunction.possible_values()))
        
    @staticmethod
    def possible_values():
        return [m.value for m in SimilarityFunction]