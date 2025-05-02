import logging

from sentence_transformers.sparse_encoder import (
    SparseEncoder,
    SparseNanoBEIREvaluator,
)

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")


evaluator = SparseNanoBEIREvaluator(
    dataset_names=None,  # None means evaluate on all datasets
    show_progress_bar=True,
    batch_size=16,
)

# Run evaluation
results = evaluator(model)
"""
Average Querie: num_rows: 49.92307692307692, num_cols: 30522.0, row_non_zero_mean: 74.91560451801007, row_sparsity_mean: 0.9975455219929035
Average Corpus: num_rows: 4334.7692307692305, num_cols: 30522.0, row_non_zero_mean: 174.81000049297626, row_sparsity_mean: 0.9942726905529315
Aggregated for Score Function: dot
Accuracy@1: 58.72%
Accuracy@3: 75.37%
Accuracy@5: 80.76%
Accuracy@10: 87.07%
Precision@1: 58.72%
Recall@1: 35.61%
Precision@3: 36.31%
Recall@3: 50.84%
Precision@5: 27.75%
Recall@5: 56.55%
Precision@10: 19.18%
Recall@10: 64.24%
MRR@10: 0.6821
NDCG@10: 0.6204
"""
# Print the results
print(f"Primary metric: {evaluator.primary_metric}")
# => Primary metric: NanoBEIR_mean_dot_ndcg@10
print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.6204
