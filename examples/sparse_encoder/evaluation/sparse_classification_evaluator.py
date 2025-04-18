from datasets import load_dataset

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseBinaryClassificationEvaluator,
    SparseEncoder,
    SpladePooling,
)

# Initialize the SPLADE model
model_name = "naver/splade-cocondenser-ensembledistil"
model = SparseEncoder(
    modules=[
        MLMTransformer(model_name),
        SpladePooling(pooling_strategy="max"),  # You can also use 'sum'
    ],
    device="cuda:0",
)

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
