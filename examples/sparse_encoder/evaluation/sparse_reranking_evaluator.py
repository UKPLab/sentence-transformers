import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseRerankingEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load a dataset with queries, positives, and negatives
eval_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation").select(range(1000))

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
"""
RerankingEvaluator: Evaluating the model on the ms-marco-dev-small dataset:
Queries: 967 	 Positives: Min 1.0, Mean 1.1, Max 3.0 	 Negatives: Min 1.0, Mean 7.1, Max 9.0
MAP: 53.46
MRR@10: 54.18
NDCG@10: 65.10
Model Sparsity Stats  Query : Row Non-Zero Mean: 43.89658737182617, Row Sparsity Mean: 0.9985617995262146
Model Sparsity Stats  Corpus : Row Non-Zero Mean: 128.37216186523438, Row Sparsity Mean: 0.9957940578460693
"""
# Print the results
print(f"Primary metric: {reranking_evaluator.primary_metric}")
# => Primary metric: ms-marco-dev-small_ndcg@10
print(f"Primary metric value: {results[reranking_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.6510
