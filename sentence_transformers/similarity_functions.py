from enum import Enum
from typing import Callable, Union

from numpy import ndarray
from torch import Tensor
from .util import (
    cos_sim,
    manhattan_sim,
    euclidean_sim,
    dot_score,
    pairwise_cos_sim,
    pairwise_manhattan_sim,
    pairwise_euclidean_sim,
    pairwise_dot_score,
)


class SimilarityFunction(Enum):
    COSINE = "cosine"
    DOT_PRODUCT = "dot"
    DOT = "dot"  # Alias for DOT_PRODUCT
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

    @staticmethod
    def to_similarity_fn(
        similarity_function: Union[str, "SimilarityFunction"],
    ) -> Callable[[Union[Tensor, ndarray], Union[Tensor, ndarray]], Tensor]:
        similarity_function = SimilarityFunction(similarity_function)

        if similarity_function == SimilarityFunction.COSINE:
            return cos_sim
        if similarity_function == SimilarityFunction.DOT_PRODUCT:
            return dot_score
        if similarity_function == SimilarityFunction.MANHATTAN:
            return manhattan_sim
        if similarity_function == SimilarityFunction.EUCLIDEAN:
            return euclidean_sim

        raise ValueError(
            "The provided function {} is not supported. Use one of the supported values: {}.".format(
                similarity_function, SimilarityFunction.possible_values()
            )
        )

    @staticmethod
    def to_similarity_pairwise_fn(
        similarity_function: Union[str, "SimilarityFunction"],
    ) -> Callable[[Union[Tensor, ndarray], Union[Tensor, ndarray]], Tensor]:
        similarity_function = SimilarityFunction(similarity_function)

        if similarity_function == SimilarityFunction.COSINE:
            return pairwise_cos_sim
        if similarity_function == SimilarityFunction.DOT_PRODUCT:
            return pairwise_dot_score
        if similarity_function == SimilarityFunction.MANHATTAN:
            return pairwise_manhattan_sim
        if similarity_function == SimilarityFunction.EUCLIDEAN:
            return pairwise_euclidean_sim

        raise ValueError(
            "The provided function {} is not supported. Use one of the supported values: {}.".format(
                similarity_function, SimilarityFunction.possible_values()
            )
        )

    @staticmethod
    def possible_values():
        return [m.value for m in SimilarityFunction]
