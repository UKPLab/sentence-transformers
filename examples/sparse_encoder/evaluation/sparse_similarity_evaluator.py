import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseEmbeddingSimilarityEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
model.similarity_fn_name = "cosine"  # even though the model is trained with dot, we need to set it to cosine for evaluation as the score in the dataset is cosine similarity

# Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

# Initialize the evaluator
dev_evaluator = SparseEmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    name="sts_dev",
)
results = dev_evaluator(model)
"""
Model Sparsity Stats: num_rows: 1500, num_cols: 30522, row_non_zero_mean: 80.34333038330078, row_sparsity_mean: 0.9973676204681396
Model Sparsity Stats: num_rows: 1500, num_cols: 30522, row_non_zero_mean: 81.78266906738281, row_sparsity_mean: 0.9973204731941223
EmbeddingSimilarityEvaluator: Evaluating the model on the sts_dev dataset:
Cosine-Similarity :     Pearson: 0.8430 Spearman: 0.8368
"""
# Print the results
print(f"Primary metric: {dev_evaluator.primary_metric}")
# => Primary metric: sts_dev_spearman_cosine
print(f"Primary metric value: {results[dev_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.8368
