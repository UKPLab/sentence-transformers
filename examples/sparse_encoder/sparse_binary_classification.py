from datasets import load_dataset

from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import (
    SparseBinaryClassificationEvaluator,
)
from sentence_transformers.sparse_encoder.models import CSRSparsity

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

# Load a dataset with two text columns and a class label column
# Using the Quora Duplicates dataset as an example
eval_dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train[-1000:]")

# Initialize the evaluator
binary_acc_evaluator = SparseBinaryClassificationEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    labels=eval_dataset["label"],
    name="quora_duplicates_dev",
    show_progress_bar=True,
    similarity_fn_names=["cosine", "dot", "euclidean", "manhattan"],
)
results = binary_acc_evaluator(model)

# Print the results
print(f"Primary metric: {binary_acc_evaluator.primary_metric}")
print(f"Primary metric value: {results[binary_acc_evaluator.primary_metric]:.4f}")

# Example of using multiple similarity functions
multi_sim_evaluator = SparseBinaryClassificationEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    labels=eval_dataset["label"],
    name="quora_duplicates_multi_sim",
    similarity_fn_names=["cosine", "dot", "euclidean", "manhattan"],
    show_progress_bar=True,
)
multi_sim_results = multi_sim_evaluator(model)

# Print the results with multiple similarity functions
print(f"Primary metric with multiple similarity functions: {multi_sim_evaluator.primary_metric}")
print(f"Primary metric value: {multi_sim_results[multi_sim_evaluator.primary_metric]:.4f}")

# Print all metrics for comparison
print("\nComparison of similarity functions:")
for sim_fn in ["cosine", "dot", "euclidean", "manhattan"]:
    print(f"\n{sim_fn.upper()} SIMILARITY:")
    print(f"  Accuracy: {multi_sim_results[f'quora_duplicates_multi_sim_{sim_fn}_accuracy']:.4f}")
    print(f"  F1: {multi_sim_results[f'quora_duplicates_multi_sim_{sim_fn}_f1']:.4f}")
    print(f"  Precision: {multi_sim_results[f'quora_duplicates_multi_sim_{sim_fn}_precision']:.4f}")
    print(f"  Recall: {multi_sim_results[f'quora_duplicates_multi_sim_{sim_fn}_recall']:.4f}")
    print(f"  AP: {multi_sim_results[f'quora_duplicates_multi_sim_{sim_fn}_ap']:.4f}")
    print(f"  MCC: {multi_sim_results[f'quora_duplicates_multi_sim_{sim_fn}_mcc']:.4f}")
