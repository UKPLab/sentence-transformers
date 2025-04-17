"""
This script demonstrates how to use the SparseTripletEvaluator to evaluate a sparse model on triplet data.

The example uses the AllNLI dataset which contains triplets of (anchor, positive, negative) sentences.
"""

import logging

from datasets import load_dataset

from sentence_transformers import LoggingHandler
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseTripletEvaluator
from sentence_transformers.sparse_encoder.models import CSRSparsity

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# Initialize model components
model_name = "sentence-transformers/all-mpnet-base-v2"
transformer = Transformer(model_name)
pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
csr_sparsity = CSRSparsity(
    input_dim=transformer.get_word_embedding_dimension(),
    hidden_dim=4 * transformer.get_word_embedding_dimension(),
    k=32,  # Number of top values to keep
    k_aux=512,  # Number of top values for auxiliary loss
)

# Create the SparseEncoder model
model = SparseEncoder(modules=[transformer, pooling, csr_sparsity])

# Load triplets from the AllNLI dataset
# The dataset contains triplets of (anchor, positive, negative) sentences
dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev[:1000]")

# Initialize the SparseTripletEvaluator
evaluator = SparseTripletEvaluator(
    anchors=dataset["anchor"],
    positives=dataset["positive"],
    negatives=dataset["negative"],
    name="all_nli_dev",
    batch_size=32,
    show_progress_bar=True,
)

# Run the evaluation
results = evaluator(model)

# Print the results
print(f"Primary metric: {evaluator.primary_metric}")
print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")

# Print all metrics
for metric_name, value in results.items():
    print(f"{metric_name}: {value:.4f}")
