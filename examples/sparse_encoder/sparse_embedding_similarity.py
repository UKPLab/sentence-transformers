from datasets import load_dataset

from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseEmbeddingSimilarityEvaluator
from sentence_transformers.sparse_encoder.models import CSRSparsity

# Initialize model components
model_name = "sentence-transformers/all-mpnet-base-v2"
transformer = Transformer(model_name)
pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
csr_sparsity = CSRSparsity(
    input_dim=transformer.get_word_embedding_dimension(),
    hidden_dim=4 * transformer.get_word_embedding_dimension(),
    k=32,  # Number of top values to keep
    k_aux=512,  # Number of top values for auxiliary loss
)
# Create the SparseEncoder model
model = SparseEncoder(modules=[transformer, pooling, csr_sparsity])

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
"""

"""
print(dev_evaluator.primary_metric)
# => "sts_dev_pearson_cosine"
print(results[dev_evaluator.primary_metric])
