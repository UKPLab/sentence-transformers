from datasets import load_dataset

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseEmbeddingSimilarityEvaluator,
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

print(dev_evaluator.primary_metric)
print(results[dev_evaluator.primary_metric])
