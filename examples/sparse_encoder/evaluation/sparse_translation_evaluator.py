from datasets import load_dataset

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseEncoder,
    SparseTranslationEvaluator,
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
# Load a parallel sentences dataset
dataset = load_dataset("sentence-transformers/parallel-sentences-news-commentary", "en-nl", split="train[:1000]")

# Initialize the TranslationEvaluator using the same texts from two languages
translation_evaluator = SparseTranslationEvaluator(
    source_sentences=dataset["english"],
    target_sentences=dataset["non_english"],
    name="news-commentary-en-nl",
)
results = translation_evaluator(model)
print(results)

print(translation_evaluator.primary_metric)
print(results[translation_evaluator.primary_metric])
