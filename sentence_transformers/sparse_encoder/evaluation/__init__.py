from __future__ import annotations

from sentence_transformers.sparse_encoder.evaluation.SparseBinaryClassificationEvaluator import (
    SparseBinaryClassificationEvaluator,
)
from sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator import (
    SparseEmbeddingSimilarityEvaluator,
)
from sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator import (
    SparseInformationRetrievalEvaluator,
)
from sentence_transformers.sparse_encoder.evaluation.SparseMSEEvaluator import (
    SparseMSEEvaluator,
)
from sentence_transformers.sparse_encoder.evaluation.SparseMSEEvaluatorDataFrame import (
    SparseMSEEvaluatorDataFrame,
)
from sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator import (
    SparseNanoBEIREvaluator,
)
from sentence_transformers.sparse_encoder.evaluation.SparseRerankingEvaluator import (
    SparseRerankingEvaluator,
)
from sentence_transformers.sparse_encoder.evaluation.SparseTranslationEvaluator import (
    SparseTranslationEvaluator,
)
from sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator import (
    SparseTripletEvaluator,
)

__all__ = [
    "SparseEmbeddingSimilarityEvaluator",
    "SparseInformationRetrievalEvaluator",
    "SparseBinaryClassificationEvaluator",
    "SparseMSEEvaluator",
    "SparseNanoBEIREvaluator",
    "SparseTripletEvaluator",
    "SparseTranslationEvaluator",
    "SparseRerankingEvaluator",
    "SparseMSEEvaluatorDataFrame",
]


# TODO: SparseMSEEvaluatorDataFrame : As for now handle sparse embed with numpy because of : trg_embeddings = np.asarray(self.embed_inputs(model, trg_sentences)) in MSEEvaluatorFromDataFrame check if needed and adapt to it
# TODO: Adapt ParaphraseMiningEvaluator for handling Sparse override (but lot of fct to check esp in utils)
# TODO: Check label accuracy (not understand how to adapt yet) if possible to have Sparse version
