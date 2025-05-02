import logging

from datasets import load_dataset

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseEncoder,
    SparseMSEEvaluator,
    SpladePooling,
)

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# Initialize the SPLADE model
student_model_name = "prithivida/Splade_PP_en_v1"
student_model = SparseEncoder(
    modules=[
        MLMTransformer(student_model_name),
        SpladePooling(pooling_strategy="max"),  # You can also use 'sum'
    ],
    device="cuda:0",
)

# Initialize the SPLADE model
teacher_model_name = "naver/splade-cocondenser-ensembledistil"
teacher_model = SparseEncoder(
    modules=[
        MLMTransformer(teacher_model_name),
        SpladePooling(pooling_strategy="max"),  # You can also use 'sum'
    ],
    device="cuda:0",
)

# Load any dataset with some texts
dataset = load_dataset("sentence-transformers/stsb", split="validation")
sentences = dataset["sentence1"] + dataset["sentence2"]

# Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR metrics.
mse_evaluator = SparseMSEEvaluator(
    source_sentences=sentences,
    target_sentences=sentences,
    teacher_model=teacher_model,
    name="stsb-dev",
)
results = mse_evaluator(student_model)

# Print the results
print(f"Primary metric: {mse_evaluator.primary_metric}")
print(f"Primary metric value: {results[mse_evaluator.primary_metric]:.4f}")
