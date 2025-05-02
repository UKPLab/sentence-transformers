import logging

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseEncoder,
    SparseNanoBEIREvaluator,
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
# Create evaluator for some NanoBEIR datasets
evaluator = SparseNanoBEIREvaluator(
    dataset_names=["QuoraRetrieval", "MSMARCO"],
    show_progress_bar=True,
    batch_size=32,
)

# Run evaluation
results = evaluator(model)

# Print the results
print(f"Primary metric: {evaluator.primary_metric}")
print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
