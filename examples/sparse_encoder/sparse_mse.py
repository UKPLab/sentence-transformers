from datasets import load_dataset

from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseMSEEvaluator
from sentence_transformers.sparse_encoder.models import CSRSparsity

# Initialize student model components
student_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
student_transformer = Transformer(student_model_name)
student_pooling = Pooling(student_transformer.get_word_embedding_dimension(), pooling_mode="mean")
student_csr_sparsity = CSRSparsity(
    input_dim=student_transformer.get_word_embedding_dimension(),
    hidden_dim=4 * student_transformer.get_word_embedding_dimension(),
    k=32,  # Number of top values to keep
    k_aux=512,  # Number of top values for auxiliary loss
)
# Create the student SparseEncoder model
student_model = SparseEncoder(modules=[student_transformer, student_pooling, student_csr_sparsity])

# Initialize teacher model components
teacher_model_name = "sentence-transformers/all-mpnet-base-v2"
teacher_transformer = Transformer(teacher_model_name)
teacher_pooling = Pooling(teacher_transformer.get_word_embedding_dimension(), pooling_mode="mean")
teacher_csr_sparsity = CSRSparsity(
    input_dim=teacher_transformer.get_word_embedding_dimension(),
    hidden_dim=4 * teacher_transformer.get_word_embedding_dimension(),
    k=32,  # Number of top values to keep
    k_aux=512,  # Number of top values for auxiliary loss
)
# Create the teacher SparseEncoder model
teacher_model = SparseEncoder(modules=[teacher_transformer, teacher_pooling, teacher_csr_sparsity])
# teacher_model = student_model to check if mse well at O when student_model == teacher_model
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
"""
MSE evaluation (lower = better) on the stsb-dev dataset:
"""
print(mse_evaluator.primary_metric)
print(results[mse_evaluator.primary_metric])

import torch

# Add these before calling the evaluator
test_sentence = sentences[0:1]  # Take just one sentence
student_emb = student_model.encode(test_sentence, convert_to_sparse_tensor=False)
teacher_emb = teacher_model.encode(test_sentence, convert_to_sparse_tensor=False)

# Check if they're identical
print("Identical embeddings:", torch.allclose(student_emb, teacher_emb))
print("Diff:", torch.mean((student_emb - teacher_emb) ** 2).item())

# Check what's happening in the evaluator itself
print("Student embeddings shape:", student_emb.shape)
print("Student embeddings sparsity:", (student_emb == 0).float().mean().item())
