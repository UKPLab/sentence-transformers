from datasets import load_dataset

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseEncoder,
    SparseRerankingEvaluator,
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

# Load a dataset with queries, positives, and negatives
eval_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

samples = [
    {
        "query": sample["query"],
        "positive": [
            text
            for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"])
            if is_selected
        ],
        "negative": [
            text
            for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"])
            if not is_selected
        ],
    }
    for sample in eval_dataset
]


# Now evaluate using only the documents from the 1000 samples
reranking_evaluator = SparseRerankingEvaluator(
    samples=samples,
    name="ms-marco-dev-small",
    show_progress_bar=True,
    batch_size=32,
)

results = reranking_evaluator(model)
print(results)

print(reranking_evaluator.primary_metric)
print(results[reranking_evaluator.primary_metric])
