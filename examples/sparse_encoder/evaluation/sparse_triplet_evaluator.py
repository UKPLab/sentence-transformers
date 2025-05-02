import logging

from datasets import load_dataset

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseEncoder,
    SparseTripletEvaluator,
    SpladePooling,
)

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# Initialize the SPLADE model
model_name = "naver/splade-cocondenser-ensembledistil"
model = SparseEncoder(
    modules=[
        MLMTransformer(model_name),
        SpladePooling(pooling_strategy="max"),  # You can also use 'sum'
    ],
    device="cuda:0",
)
# Load triplets from the AllNLI dataset
# The dataset contains triplets of (anchor, positive, negative) sentences
dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev[:1000]")

# Initialize the SparseTripletEvaluator
evaluator = SparseTripletEvaluator(
    anchors=dataset[:1000]["anchor"],
    positives=dataset[:1000]["positive"],
    negatives=dataset[:1000]["negative"],
    name="all_nli_dev",
    batch_size=32,
    show_progress_bar=True,
)

# Run the evaluation
results = evaluator(model)

# Print the results
print(f"Primary metric: {evaluator.primary_metric}")
print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
