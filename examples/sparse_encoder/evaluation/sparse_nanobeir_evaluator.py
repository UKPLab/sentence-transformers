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
Query info: num_rows: 50, num_cols: 30522, row_non_zero_mean: 62.97999954223633, row_sparsity_mean: 0.9979365468025208 1/1 [00:04<00:00,  4.12s/it]
Corpus info: num_rows: 5046, num_cols: 30522, row_non_zero_mean: 63.394371032714844, row_sparsity_mean: 0.9979230165481567
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

Evaluating NanoMSMARCO
Information Retrieval Evaluation of the model on the NanoMSMARCO dataset:
Query info: num_rows: 50, num_cols: 30522, row_non_zero_mean: 48.099998474121094, row_sparsity_mean: 0.99842399358749391/1 [00:19<00:00, 19.40s/it]
Corpus info: num_rows: 5043, num_cols: 30522, row_non_zero_mean: 125.38131713867188, row_sparsity_mean: 0.9958921670913696
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

Average Querie: num_rows: 50.0, num_cols: 30522.0, row_non_zero_mean: 55.53999900817871, row_sparsity_mean: 0.9981802701950073
Average Corpus: num_rows: 5044.5, num_cols: 30522.0, row_non_zero_mean: 94.38784408569336, row_sparsity_mean: 0.9969075918197632
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
"""
# Print the results
print(f"Primary metric: {evaluator.primary_metric}")
# => Primary metric: NanoBEIR_mean_dot_ndcg@10
print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.8089
