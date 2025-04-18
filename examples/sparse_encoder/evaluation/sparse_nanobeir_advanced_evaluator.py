from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseEncoder,
    SparseNanoBEIREvaluator,
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
# Create evaluator for all NanoBEIR datasets
evaluator = SparseNanoBEIREvaluator(
    dataset_names=None,  # None means evaluate on all datasets
    show_progress_bar=True,
    batch_size=32,
)

# Run evaluation
print("Starting evaluation on all NanoBEIR datasets")
results = evaluator(model)

print(f"Primary metric: {evaluator.primary_metric}")
print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")

# Print results for each dataset
for key, value in results.items():
    if key.startswith("Nano"):
        print(f"{key}: {value:.4f}")
