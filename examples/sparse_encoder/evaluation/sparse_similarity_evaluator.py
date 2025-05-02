import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseEmbeddingSimilarityEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

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
EmbeddingSimilarityEvaluator: Evaluating the model on the sts_dev dataset:
Dot-Similarity :	Pearson: 0.7513	Spearman: 0.8010
"""
# Print the results
print(f"Primary metric: {dev_evaluator.primary_metric}")
# => Primary metric: sts_dev_spearman_dot
print(f"Primary metric value: {results[dev_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.8010
