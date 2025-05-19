import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseTranslationEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Load a model, not mutilingual but hope to see some on the hub soon
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load a parallel sentences dataset
dataset = load_dataset("sentence-transformers/parallel-sentences-news-commentary", "en-nl", split="train[:1000]")

# Initialize the TranslationEvaluator using the same texts from two languages
translation_evaluator = SparseTranslationEvaluator(
    source_sentences=dataset["english"],
    target_sentences=dataset["non_english"],
    name="news-commentary-en-nl",
)
results = translation_evaluator(model)
"""
Evaluating translation matching Accuracy of the model on the news-commentary-en-nl dataset:
Accuracy src2trg: 41.40
Accuracy trg2src: 47.70
Model Sparsity Stats: Row Non-Zero Mean: 113.6150016784668, Row Sparsity Mean: 0.9962776005268097
"""
# Print the results
print(f"Primary metric: {translation_evaluator.primary_metric}")
# => Primary metric: news-commentary-en-nl_mean_accuracy
print(f"Primary metric value: {results[translation_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.4455
