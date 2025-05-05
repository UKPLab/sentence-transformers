# Semantic Search

Here we can find examples demonstrating how to do a manual implementation of Semantic Search but also integrate sparse encoder models with popular vector databases for efficient semantic search. Sparse encoders produce sparse vector representations that are particularly well-suited for search applications.

If you aren't familiar with Semantic Search, see [SemanticSearch](../../../sentence_transformer/applications/semantic-search/README.md) for a more details explication.

## Manual implementation 

Here we show how to implement a simple semantic search system using sparse encoders [semantic_search_manual_implem.py](semantic_search/semantic_search_manual_implem.py). This example demonstrates:
  - Encoding a corpus of documents with sparse representations
  - Finding the most similar documents for given queries
  - Analyzing which tokens contribute most to the similarity scores

```python
"""
This is a simple application for sparse encoder: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""

import torch

from sentence_transformers import SparseEncoder

# Initialize the SPLADE model
embedder = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Corpus with example sentences
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]
# Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field.",
]

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
    scores, indices = torch.topk(similarity_scores, k=top_k)

    print("\nQuery:", query)
    print(
        "Top 5 most similar sentences in corpus with associated top 10 tokens which contribute to the similarity score:"
    )
    for score, idx in zip(scores, indices):
        # Compute the pointwise product between the query embedding and the current corpus embedding to get token wise scores for interpretability
        doc_embedding = corpus_embeddings[idx]
        prod = (query_embedding * doc_embedding).to_dense()

        # Retrieve the top 10 tokens contributing most to the similarity score
        top_values, token_ids = torch.topk(prod, k=10)
        # Convert token IDs to tokens using the embedder’s tokenizer
        top_tokens = embedder.tokenizer.convert_ids_to_tokens(token_ids.cpu().tolist())

        token_scores = ", ".join([f'("{token.strip()}", {value:.2f})' for token, value in zip(top_tokens, top_values)])
        print(f"Score: {score:.4f} - Sentence: {corpus[idx]}, Top influential tokens: {token_scores}")
```
<details>
<summary>Toggle To See Results</summary>

```python 
"""
Query: A man is eating pasta.
Top 5 most similar sentences in corpus with associated top 10 tokens which contribute to the similarity score:
Score: 21.0578 - Sentence: A man is eating food., Top influential tokens: ("man", 5.48), ("eating", 3.83), ("eat", 3.15), ("men", 3.12), ("food", 1.78), ("male", 0.87), ("person", 0.62), ("a", 0.39), ("hunger", 0.28), ("meat", 0.27)
Score: 18.3213 - Sentence: A man is eating a piece of bread., Top influential tokens: ("man", 4.85), ("eating", 3.49), ("eat", 3.02), ("men", 2.74), ("male", 0.68), ("food", 0.66), ("person", 0.58), ("a", 0.51), ("meat", 0.36), ("culture", 0.27)
Score: 10.2319 - Sentence: A man is riding a horse., Top influential tokens: ("man", 4.85), ("men", 3.11), ("male", 0.68), ("a", 0.60), ("person", 0.59), ("animal", 0.21), ("adam", 0.07), ("god", 0.06), ("sex", 0.03), ("who", 0.01)
Score: 6.5993 - Sentence: A man is riding a white horse on an enclosed ground., Top influential tokens: ("man", 3.31), ("men", 1.58), ("a", 0.51), ("male", 0.41), ("person", 0.34), ("on", 0.17), ("animal", 0.16), ("wearing", 0.04), ("god", 0.04), ("culture", 0.02)
Score: 5.2490 - Sentence: Two men pushed carts through the woods., Top influential tokens: ("men", 2.60), ("man", 2.51), ("a", 0.12), ("murder", 0.01), ("said", 0.00), ("[unused2]", 0.00), ("[unused3]", 0.00), ("[unused1]", 0.00), ("[PAD]", 0.00), ("[unused0]", 0.00)

Query: Someone in a gorilla costume is playing a set of drums.
Top 5 most similar sentences in corpus with associated top 10 tokens which contribute to the similarity score:
Score: 16.7709 - Sentence: A monkey is playing drums., Top influential tokens: ("drums", 4.38), ("drum", 2.27), ("play", 2.16), ("playing", 1.77), ("drummer", 0.80), ("dance", 0.67), ("monkey", 0.55), ("music", 0.50), ("a", 0.40), ("sound", 0.39)
Score: 8.7609 - Sentence: A woman is playing violin., Top influential tokens: ("play", 2.12), ("playing", 1.79), ("dance", 0.68), ("person", 0.67), ("music", 0.55), ("instrument", 0.52), ("guitar", 0.39), ("a", 0.35), ("wearing", 0.32), ("player", 0.21)
Score: 2.8393 - Sentence: A man is riding a horse., Top influential tokens: ("person", 0.91), ("a", 0.49), ("man", 0.45), ("animal", 0.37), ("sport", 0.32), ("savage", 0.10), ("dance", 0.08), ("billy", 0.06), ("god", 0.04), ("hunting", 0.01)
Score: 2.4528 - Sentence: A man is eating a piece of bread., Top influential tokens: ("person", 0.90), ("man", 0.45), ("a", 0.42), ("someone", 0.29), ("animal", 0.08), ("god", 0.07), ("ritual", 0.07), ("culture", 0.07), ("something", 0.05), ("who", 0.03)
Score: 2.3295 - Sentence: A man is riding a white horse on an enclosed ground., Top influential tokens: ("person", 0.53), ("a", 0.42), ("man", 0.31), ("sport", 0.27), ("animal", 0.27), ("savage", 0.09), ("character", 0.09), ("wearing", 0.07), ("symbol", 0.07), ("hunting", 0.05)

Query: A cheetah chases prey on across a field.
Top 5 most similar sentences in corpus with associated top 10 tokens which contribute to the similarity score:
Score: 16.3632 - Sentence: A cheetah is running behind its prey., Top influential tokens: ("che", 3.80), ("##eta", 3.72), ("prey", 2.77), ("hunting", 0.75), ("behavior", 0.70), ("##h", 0.62), ("movement", 0.45), ("animal", 0.33), ("predator", 0.30), ("chasing", 0.29)
Score: 2.2318 - Sentence: A monkey is playing drums., Top influential tokens: ("animal", 0.43), ("a", 0.41), ("behavior", 0.28), ("hunting", 0.22), ("movement", 0.19), ("bird", 0.17), ("dance", 0.17), ("species", 0.07), ("dog", 0.07), ("game", 0.06)
Score: 1.4788 - Sentence: A man is riding a horse., Top influential tokens: ("a", 0.51), ("animal", 0.48), ("movement", 0.33), ("sport", 0.10), ("hunting", 0.04), ("dance", 0.02), ("[unused1]", 0.00), ("[unused2]", 0.00), ("[unused0]", 0.00), ("[PAD]", 0.00)
Score: 1.4335 - Sentence: A man is riding a white horse on an enclosed ground., Top influential tokens: ("a", 0.43), ("animal", 0.35), ("hunting", 0.21), ("movement", 0.17), ("breed", 0.12), ("sport", 0.08), ("bird", 0.04), ("dog", 0.02), ("[PAD]", 0.00), ("[unused0]", 0.00)
Score: 1.4279 - Sentence: Two men pushed carts through the woods., Top influential tokens: ("hunting", 0.49), ("cross", 0.41), ("move", 0.22), ("a", 0.10), ("escape", 0.08), ("they", 0.06), ("across", 0.05), ("obstacle", 0.01), ("deer", 0.01), ("[PAD]", 0.00)
"""
```
</details>

## How It Works for vector databases

1. **Document Encoding**: Both examples load a dataset (Natural Questions) and encode the documents using a pretrained sparse encoder.
2. **Indexing**: The encoded documents are indexed in the vector database.
3. **Query Processing**: User queries are encoded with the same sparse encoder.
4. **Retrieval**: The vector database performs a similarity search to find the most relevant documents.
5. **Results**: Search results are returned with their similarity scores and document content.

The Advantages of Sparse Vectors for Search are:

- **Efficiency**: Sparse vectors (where most values are zero) can be stored and searched more efficiently than dense vectors
- **Interpretability**: Non-zero dimensions in sparse embeddings often correspond to specific tokens, making them more interpretable
- **Exact Matching**: Sparse vectors can preserve exact term matching signals that might be lost in dense embeddings


## Qdrant Integration

### Prerequisites:
- Qdrant running locally (or accessible), see more [Qdrant Quickstart](https://qdrant.tech/documentation/quickstart/)
- Python Qdrant client installed:
  ```bash
  pip install qdrant-client
  ```
This example demonstrates how to set up Qdrant for sparse vector search by showing how to efficiently encode and index documents with sparse encoders, formulating search queries with sparse vectors, and providing an interactive query interface. See [`semantic_search_qdrant.py`](semantic_search/semantic_search_qdrant.py) or below: 


```python
"""
This script contains an example how to perform semantic search with Qdrant.

You need Qdrant up and running locally:
https://qdrant.tech/documentation/quickstart/

Further, you need the Python Qdrant Client installed: https://python-client.qdrant.tech/, e.g.:

pip install qdrant-client

This script was created for `qdrant-client` v1.0+.
"""

import time

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.search_engines import semantic_search_qdrant

# 1. Load the natural-questions dataset with 100K answers
dataset = load_dataset("sentence-transformers/natural-questions", split="train", trust_remote_code=True)
corpus = dataset["answer"]

# 2. Come up with some queries
queries = dataset["query"][:2]

# 3. Load the model
sparse_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# 5. Encode the corpus
corpus_embeddings = sparse_model.encode(corpus, convert_to_sparse_tensor=True, batch_size=16, show_progress_bar=True)

# Initially, we don't have a qdrant index yet
corpus_index = None
while True:
    # 6. Encode the queries using the full precision
    start_time = time.time()
    query_embeddings = sparse_model.encode(queries, convert_to_sparse_tensor=True)
    print(f"Encoding time: {time.time() - start_time:.6f} seconds")

    # 7. Perform semantic search using qdrant
    results, search_time, corpus_index = semantic_search_qdrant(
        query_embeddings,
        corpus_index=corpus_index,
        corpus_embeddings=corpus_embeddings if corpus_index is None else None,
        top_k=5,
        output_index=True,
    )

    # 8. Output the results
    print(f"Search time: {search_time:.6f} seconds")
    for query, result in zip(queries, results):
        print(f"Query: {query}")
        for entry in result:
            print(f"(Score: {entry['score']:.4f}) {corpus[entry['corpus_id']]}, corpus_id: {entry['corpus_id']}")
        print("")

    # 10. Prompt for more queries
    queries = [input("Please enter a question: ")]
```

## Elasticsearch Integration

### Prerequisites:
- Elasticsearch running locally (or accessible), see more [Elasticsearch locally](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html)
- Python Elasticsearch client installed:
  ```bash
  pip install elasticsearch
  ```
This example demonstrates how to set up Elasticsearch for sparse vector search by showing how to efficiently encode and index documents with sparse encoders, formulating search queries with sparse vectors, and providing an interactive query interface. See [`semantic_search_elasticsearch.py`](semantic_search/semantic_search_elasticsearch.py) or below :

```python
"""
This script contains an example how to perform semantic search with Elasticsearch.

You need Elasticsearch up and running locally:
https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html

Further, you need the Python Elasticsearch Client installed: https://elasticsearch-py.readthedocs.io/, e.g.:

pip install elasticsearch

This script was created for `elasticsearch` v8.0+.
"""

import time

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.search_engines import semantic_search_elasticsearch

# 1. Load the natural-questions dataset with 100K answers
dataset = load_dataset("sentence-transformers/natural-questions", split="train", trust_remote_code=True)
corpus = dataset["answer"]

# 2. Come up with some queries
queries = dataset["query"][:2]

# 3. Load the model
sparse_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# 5. Encode the corpus
corpus_embeddings = sparse_model.encode(corpus, convert_to_sparse_tensor=True, batch_size=32, show_progress_bar=True)

corpus_index = None
while True:
    # 6. Encode the queries using the full precision
    start_time = time.time()
    query_embeddings = sparse_model.encode(queries, convert_to_sparse_tensor=True)
    print(f"Encoding time: {time.time() - start_time:.6f} seconds")

    # 7. Perform semantic search using Elasticsearch
    results, search_time, corpus_index = semantic_search_elasticsearch(
        query_embeddings,
        corpus_index=corpus_index,
        corpus_embeddings=corpus_embeddings if corpus_index is None else None,
        top_k=5,
        output_index=True,
    )

    # 8. Output the results
    print(f"Search time: {search_time:.6f} seconds")
    for query, result in zip(queries, results):
        print(f"Query: {query}")
        for entry in result:
            print(f"(Score: {entry['score']:.4f}) {corpus[entry['corpus_id']]}, corpus_id: {entry['corpus_id']}")
        print("")

    # 10. Prompt for more queries
    queries = [input("Please enter a question: ")]
```

