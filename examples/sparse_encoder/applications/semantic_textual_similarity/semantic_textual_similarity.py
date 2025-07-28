"""
This is a simple application for sparse encoder: Semantic Textual Similarity

We have multiple sentences and we want to compute the similarity between them.
Here we use the SPLADE model to compute the similarity between two lists of sentences.
The default similarity metric is dot product.
"""

from sentence_transformers import SparseEncoder

# Initialize the SPLADE model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Two lists of sentences
sentences1 = [
    "The new movie is awesome",
    "The cat sits outside",
    "A man is playing guitar",
]

sentences2 = [
    "The dog plays in the garden",
    "The new movie is so great",
    "A woman watches TV",
]

# Compute embeddings for both lists
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

# Compute cosine similarities
similarities = model.similarity(embeddings1, embeddings2)

# Output the pairs with their score
for idx_i, sentence1 in enumerate(sentences1):
    print(sentence1)
    for idx_j, sentence2 in enumerate(sentences2):
        print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")
"""
The new movie is awesome
 - The dog plays in the garden   : 1.1750
 - The new movie is so great     : 24.0100
 - A woman watches TV            : 0.1358
The cat sits outside
 - The dog plays in the garden   : 2.7264
 - The new movie is so great     : 0.6256
 - A woman watches TV            : 0.2129
A man is playing guitar
 - The dog plays in the garden   : 7.5841
 - The new movie is so great     : 0.0316
 - A woman watches TV            : 1.5672
"""
