## Semantic Textual Similarity

Demonstrates how to compute similarity between sentences using sparse encoders [`semantic_textual_similarity.py`](semantic_textual_similarity/semantic_textual_similarity.py). This example shows how to:
  - Generate sparse embeddings for sentences
  - Compute a similarity between them
  - Visualize the pairs with their associated scores

```python

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
```
The similarity metric that is used is stored in the SparseEncoder instance under :attr:`SparseEncoder.similarity_fn_name <sentence_transformers.sparse_encoder.SparseEncoder.similarity_fn_name>`. Valid options are:

- ``SimilarityFunction.DOT_PRODUCT`` (a.k.a `"dot"`): Dot Product (**default**)
- ``SimilarityFunction.COSINE`` (a.k.a `"cosine"`): Cosine Similarity 
- ``SimilarityFunction.EUCLIDEAN`` (a.k.a `"euclidean"`): Negative Euclidean Distance
- ``SimilarityFunction.MANHATTAN`` (a.k.a. `"manhattan"`): Negative Manhattan Distance

This value can be changed in a handful of ways:

1. By initializing the SentenceTransformer instance with the desired similarity function::
``` python
from sentence_transformers import SparseEncoder, SimilarityFunction

model = SparseEncoder("naver/splade-cocondenser-ensembledistil", similarity_fn_name=SimilarityFunction.COSINE)
```
2. By setting the value directly on the SparseEncoder instance:
``` python
from sentence_transformers import SparseEncoder, SimilarityFunction

model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
model.similarity_fn_name = SimilarityFunction.COSINE

```
3. By setting the value under the ``"similarity_fn_name"`` key in the ``config_sentence_transformers.json`` file of a saved model. When you save a SparseEncoder model, this value will be automatically saved as well.

Sentence Transformers implements two methods to calculate the similarity between embeddings that we didn't override for SparseEncoder:

- :meth:`SentenceTransformer.similarity <sentence_transformers.SentenceTransformer.similarity>`: Calculates the similarity between all pairs of embeddings.
- :meth:`SentenceTransformer.similarity_pairwise <sentence_transformers.SentenceTransformer.similarity_pairwise>`: Calculates the similarity between embeddings in a pairwise fashion.


``` python
from sentence_transformers import SparseEncoder, SimilarityFunction

# Load a pretrained Sentence Transformer model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Embed some sentences
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(model.similarity_fn_name)
# => "dot_product"
print(similarities)
# tensor([[3.5629e+01, 9.1541e+00, 1.1269e-01],
#         [9.1541e+00, 2.7478e+01, 1.9061e-02],
#         [1.1269e-01, 1.9061e-02, 2.9612e+01]], device='cuda:0')

# Change the similarity function to Manhattan distance
model.similarity_fn_name = SimilarityFunction.COSINE
print(model.similarity_fn_name)
# => "cosine"

similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000e+00, 2.9256e-01, 3.4694e-03],
#         [2.9256e-01, 1.0000e+00, 6.6823e-04],
#         [3.4694e-03, 6.6823e-04, 1.0000e+00]], device='cuda:0')

```