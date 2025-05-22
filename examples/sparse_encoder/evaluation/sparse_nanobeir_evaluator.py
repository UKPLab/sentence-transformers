import logging

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

datasets = ["QuoraRetrieval", "MSMARCO"]

evaluator = SparseNanoBEIREvaluator(
    dataset_names=datasets,
    show_progress_bar=True,
    batch_size=32,
)

# Run evaluation
results = evaluator(model)
"""
Evaluating NanoQuoraRetrieval
Information Retrieval Evaluation of the model on the NanoQuoraRetrieval dataset:
Queries: 50
Corpus: 5046

Score-Function: dot
Accuracy@1: 92.00%
Accuracy@3: 96.00%
Accuracy@5: 98.00%
Accuracy@10: 100.00%
Precision@1: 92.00%
Precision@3: 40.00%
Precision@5: 24.80%
Precision@10: 13.20%
Recall@1: 79.73%
Recall@3: 92.53%
Recall@5: 94.93%
Recall@10: 98.27%
MRR@10: 0.9439
NDCG@10: 0.9339
MAP@100: 0.9072
Model Query Sparsity: Active Dimensions: 63.0, Sparsity Ratio: 0.9979
Model Corpus Sparsity: Active Dimensions: 63.4, Sparsity Ratio: 0.9979

Information Retrieval Evaluation of the model on the NanoMSMARCO dataset:
Queries: 50
Corpus: 5043

Score-Function: dot
Accuracy@1: 48.00%
Accuracy@3: 74.00%
Accuracy@5: 76.00%
Accuracy@10: 88.00%
Precision@1: 48.00%
Precision@3: 24.67%
Precision@5: 15.20%
Precision@10: 8.80%
Recall@1: 48.00%
Recall@3: 74.00%
Recall@5: 76.00%
Recall@10: 88.00%
MRR@10: 0.6211
NDCG@10: 0.6838
MAP@100: 0.6277
Model Query Sparsity: Active Dimensions: 48.1, Sparsity Ratio: 0.9984
Model Corpus Sparsity: Active Dimensions: 125.4, Sparsity Ratio: 0.9959

Average Queries: 50.0
Average Corpus: 5044.5
Aggregated for Score Function: dot
Accuracy@1: 70.00%
Accuracy@3: 85.00%
Accuracy@5: 87.00%
Accuracy@10: 94.00%
Precision@1: 70.00%
Recall@1: 63.87%
Precision@3: 32.33%
Recall@3: 83.27%
Precision@5: 20.00%
Recall@5: 85.47%
Precision@10: 11.00%
Recall@10: 93.13%
MRR@10: 0.7825
NDCG@10: 0.8089
Model Query Sparsity: Active Dimensions: 55.5, Sparsity Ratio: 0.9982
Model Corpus Sparsity: Active Dimensions: 94.4, Sparsity Ratio: 0.9969
"""
# Print the results
print(f"Primary metric: {evaluator.primary_metric}")
# => Primary metric: NanoBEIR_mean_dot_ndcg@10
print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.8089
