import logging

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

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
Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 58.72%
Accuracy@3: 75.37%
Accuracy@5: 80.76%
Accuracy@10: 87.07%
Precision@1: 58.72%
Recall@1: 35.61%
Precision@3: 36.31%
Recall@3: 50.84%
Precision@5: 27.72%
Recall@5: 56.55%
Precision@10: 19.18%
Recall@10: 64.21%
MRR@10: 0.6822
NDCG@10: 0.6204
Model Sparsity Stats  Query : Row Non-Zero Mean: 74.93406589214618, Row Sparsity Mean: 0.9975449305314285
Model Sparsity Stats  Corpus : Row Non-Zero Mean: 174.8070262028621, Row Sparsity Mean: 0.9942727547425491
"""
# Print the results
print(f"Primary metric: {evaluator.primary_metric}")
# => Primary metric: NanoBEIR_mean_dot_ndcg@10
print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.6204
