# Semantic Search

Semantic search refers to search techniques that go beyond traditional keyword-based search. Instead of relying solely on exact matches of keywords, semantic search aims to understand the meaning and context of the query and the documents being searched. This allows for more relevant and accurate search results, even when the exact keywords may not match.

Sparse embeddings are a type of representation where most of the values are zero, and only a small number of dimensions contain non-zero values. This is in contrast to dense embeddings, where all dimensions typically have non-zero values. Traditional sparse embedding solutions are often lexically based, meaning they rely on exact matches of terms or phrases. However, modern sparse encoders like SPLADE and other sparse encoder models can generate embeddings that capture semantic meaning while still being sparse.

These embeddings can allow for extremely efficient semantic search, as long as the search solution takes good advantage of the fact that the large majority of sparse embedding dimensions are 0. This page shows an example demonstrating how to perform semantic search manually, but also how to integrate a SparseEncoder model with popular vector databases/search systems.

If you aren't familiar with Semantic Search, see the [Sentence Transformers > Semantic Search](../../../sentence_transformer/applications/semantic-search/README.md) for a broader explanation using dense embedding models.

## Manual Search

Manually performing semantic search with sparse encoders is straightforward, and only consists of a few steps:

1. **Load a SparseEncoder model**: Load a pretrained sparse encoder model from the Hugging Face Hub or your local directory.
2. **Encode the corpus**: Use the model to encode a set of documents (the corpus) into sparse embeddings.
3. **Encode the queries**: Encode the user queries into sparse embeddings using the same model.
4. **Compute similarity**: Calculate the similarity between the query embeddings and the corpus embeddings using a suitable similarity function (e.g., cosine similarity, dot product).
5. **Retrieve results**: Sort the results based on similarity scores and return the most relevant documents.
6. **Analyze results**: Optionally, analyze the results to understand which tokens contributed most to the similarity scores.

```{eval-rst}

.. sidebar:: Documentation

   1. :class:`SparseEncoder <sentence_transformers.sparse_encoder.SparseEncoder>`
   2. :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.SparseEncoder.encode>`
   3. :meth:`SparseEncoder.similarity <sentence_transformers.sparse_encoder.SparseEncoder.similarity>`
   4. `naver/splade-cocondenser-ensembledistil <https://huggingface.co/naver/splade-cocondenser-ensembledistil>`_

::

    import torch
    from sentence_transformers import SparseEncoder

    # 1. Load a pretrained SparseEncoder model
    model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

    # 2. Encode a corpus of texts using the SparseEncoder model
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
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    # 3. Encode the user queries using the same SparseEncoder model
    queries = [
        "A man is eating pasta.",
        "Someone in a gorilla costume is playing a set of drums.",
        "A cheetah chases prey on across a field.",
    ]
    query_embeddings = model.encode(queries, convert_to_tensor=True)

    # 4. Use the similarity function between the query and corpus embeddings
    top_k = min(5, len(corpus)) # Find at most 5 sentences of the corpus for each query sentence
    similarity_scores = model.similarity(query_embeddings, corpus_embeddings)
    scores, indices = torch.topk(similarity_scores, k=top_k)

    # 5. Sort the results and print the top 5 most similar sentences for each query
    for query_id, query in enumerate(queries):
        print(f"Query: {query}")
        for corpus_id, score in zip(indices[query_id].tolist(), scores[query_id]):
            sentence = corpus[corpus_id]

            # Retrieve the top 10 tokens contributing most to the similarity score
            product = (query_embeddings[query_id] * corpus_embeddings[corpus_id]).to_dense()
            top_values, token_ids = torch.topk(product, k=10)

            # 6. Convert token IDs to tokens using the embedder’s tokenizer
            top_tokens = model.tokenizer.batch_decode(token_ids)
            top_tokens = [(token, round(value, 2)) for token, value in zip(top_tokens, top_values)]
            print(f"Score: {score:.4f} - Sentence: {sentence} - Top influential tokens: {top_tokens}")
        print("")
```

<details><summary>Toggle To See Results</summary>

```python 
"""
Query: A man is eating pasta.
Score: 21.3464 - Sentence: A man is eating food. - Top influential tokens: ("man", 5.48), ("eating", 3.83), ("eat", 3.15), ("men", 3.12), ("food", 1.78), ("male", 0.87), ("person", 0.62), ("a", 0.39), ("hunger", 0.28), ("meat", 0.27)
Score: 18.4783 - Sentence: A man is eating a piece of bread. - Top influential tokens: ("man", 4.85), ("eating", 3.49), ("eat", 3.02), ("men", 2.74), ("male", 0.68), ("food", 0.66), ("person", 0.58), ("a", 0.51), ("meat", 0.36), ("culture", 0.27)
Score: 10.2556 - Sentence: A man is riding a horse. - Top influential tokens: ("man", 4.85), ("men", 3.11), ("male", 0.68), ("a", 0.60), ("person", 0.59), ("animal", 0.21), ("god", 0.08), ("adam", 0.08), ("sex", 0.03), ("who", 0.01)
Score: 6.6108 - Sentence: A man is riding a white horse on an enclosed ground. - Top influential tokens: ("man", 3.31), ("men", 1.58), ("a", 0.51), ("male", 0.41), ("person", 0.34), ("on", 0.17), ("animal", 0.16), ("god", 0.05), ("wearing", 0.04), ("culture", 0.02)
Score: 5.2575 - Sentence: Two men pushed carts through the woods. - Top influential tokens: ("men", 2.60), ("man", 2.51), ("a", 0.12), ("murder", 0.01), (".", 0.01), ("said", 0.00), ("had", 0.00), ("were", 0.00), ("was", 0.00), ("the", 0.00)

Query: Someone in a gorilla costume is playing a set of drums.
Score: 16.7709 - Sentence: A monkey is playing drums. - Top influential tokens: ("drums", 4.38), ("drum", 2.27), ("play", 2.16), ("playing", 1.77), ("drummer", 0.80), ("dance", 0.67), ("monkey", 0.55), ("music", 0.50), ("a", 0.40), ("sound", 0.39)
Score: 8.7609 - Sentence: A woman is playing violin. - Top influential tokens: ("play", 2.12), ("playing", 1.79), ("dance", 0.68), ("person", 0.67), ("music", 0.55), ("instrument", 0.52), ("guitar", 0.39), ("a", 0.35), ("wearing", 0.32), ("player", 0.21)
Score: 2.8393 - Sentence: A man is riding a horse. - Top influential tokens: ("person", 0.91), ("a", 0.49), ("man", 0.45), ("animal", 0.37), ("sport", 0.32), ("savage", 0.10), ("dance", 0.08), ("billy", 0.06), ("god", 0.04), ("hunting", 0.01)
Score: 2.4528 - Sentence: A man is eating a piece of bread. - Top influential tokens: ("person", 0.90), ("man", 0.45), ("a", 0.42), ("someone", 0.29), ("animal", 0.08), ("god", 0.07), ("ritual", 0.07), ("culture", 0.07), ("something", 0.05), ("who", 0.03)
Score: 2.3295 - Sentence: A man is riding a white horse on an enclosed ground. - Top influential tokens: ("person", 0.53), ("a", 0.42), ("man", 0.31), ("sport", 0.27), ("animal", 0.27), ("savage", 0.09), ("character", 0.09), ("wearing", 0.07), ("symbol", 0.07), ("hunting", 0.05)

Query: A cheetah chases prey on across a field.
Score: 16.4271 - Sentence: A cheetah is running behind its prey. - Top influential tokens: ("che", 3.80), ("##eta", 3.72), ("prey", 2.77), ("hunting", 0.75), ("behavior", 0.70), ("##h", 0.62), ("movement", 0.45), ("animal", 0.33), ("predator", 0.30), ("chasing", 0.29)
Score: 2.2981 - Sentence: A monkey is playing drums. - Top influential tokens: ("animal", 0.43), ("a", 0.41), ("behavior", 0.28), ("hunting", 0.22), ("movement", 0.19), ("bird", 0.17), ("dance", 0.17), ("species", 0.07), ("dog", 0.07), ("bug", 0.07)
Score: 1.5377 - Sentence: A man is riding a horse. - Top influential tokens: ("a", 0.51), ("animal", 0.48), ("movement", 0.33), ("sport", 0.16), ("hunting", 0.04), ("dance", 0.02), ("##h", 0.00), ("across", 0.00), ("game", 0.00), ("they", 0.00)
Score: 1.4831 - Sentence: A man is riding a white horse on an enclosed ground. - Top influential tokens: ("a", 0.43), ("animal", 0.35), ("hunting", 0.21), ("movement", 0.17), ("sport", 0.13), ("breed", 0.12), ("bird", 0.04), ("dog", 0.02), ("they", 0.00), ("game", 0.00)
Score: 1.4279 - Sentence: Two men pushed carts through the woods. - Top influential tokens: ("hunting", 0.49), ("cross", 0.41), ("move", 0.22), ("a", 0.10), ("escape", 0.08), ("they", 0.06), ("across", 0.05), ("obstacle", 0.01), ("deer", 0.01), ("game", 0.00)
"""
```

</details>
<br>

## Vector Database Search

Alternatively, some vector databases and search engines can be used to perform semantic search with sparse encoders. These systems are designed to efficiently handle large-scale vector data and provide fast retrieval of relevant documents. They can leverage the sparsity of the embeddings to optimize storage and search operations. 

The overall structure is similar to the manual search, but the vector database handles the indexing and retrieval of documents. The steps are approximately as follows:

1. **Encode the corpus**: Load your data and encode the documents using a pretrained sparse encoder.
2. **Indexing**: The documents and their sparse embeddings are indexed in the vector database.
3. **Encode the query**: User queries are encoded with the same sparse encoder.
4. **Retrieval**: The vector database performs a similarity search to find the most relevant documents.
5. **Results**: Search results are returned with their similarity scores and document content.

The advantages of Sparse Vectors for search are:

- **Efficiency**: Sparse vectors (where most values are zero) can be stored and searched more efficiently than dense vectors.
- **Interpretability**: Non-zero dimensions in sparse embeddings often correspond to specific tokens, allowing you to understand which tokens contributed to the similarity score.
- **Exact Matching**: Sparse vectors can preserve exact term matching signals that might be lost in dense embeddings.

## Qdrant Integration

This example demonstrates how to set up Qdrant for sparse vector search by showing how to efficiently encode and index documents with sparse encoders, formulating search queries with sparse vectors, and providing an interactive query interface. See [semantic_search_qdrant.py](semantic_search_qdrant.py) or below: 

### Prerequisites:
- Qdrant running locally (or accessible), see the [Qdrant Quickstart](https://qdrant.tech/documentation/quickstart/) for more details.
- Python Qdrant Client installed:
  ```bash
  pip install qdrant-client
  ```

```{eval-rst}

.. sidebar:: Documentation

   1. :class:`SparseEncoder <sentence_transformers.sparse_encoder.SparseEncoder>`
   2. :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.SparseEncoder.encode>`
   3. :meth:`semantic_search_qdrant <sentence_transformers.sparse_encoder.search_engines.semantic_search_qdrant>`
   4. `naver/splade-cocondenser-ensembledistil <https://huggingface.co/naver/splade-cocondenser-ensembledistil>`_
   5. `sentence-transformers/natural-questions <https://huggingface.co/datasets/sentence-transformers/natural-questions>`_

:: 

    import time

    from datasets import load_dataset
    from sentence_transformers import SparseEncoder
    from sentence_transformers.sparse_encoder.search_engines import semantic_search_qdrant

    # 1. Load the natural-questions dataset with 100K answers
    dataset = load_dataset("sentence-transformers/natural-questions", split="train")
    num_docs = 10_000
    corpus = dataset["answer"][:num_docs]

    # 2. Come up with some queries
    queries = dataset["query"][:2]

    # 3. Load the model
    sparse_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

    # 5. Encode the corpus
    corpus_embeddings = sparse_model.encode(corpus, convert_to_sparse_tensor=True, show_progress_bar=True)

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
                score = entry["score"]
                corpus_id = entry["corpus_id"]
                print(f"(Score: {score:.4f}) {corpus[corpus_id]}, corpus_id: {corpus_id}")
            print("")

        # 10. Prompt for more queries
        queries = [input("Please enter a question: ")]
```

## Elasticsearch Integration

This example demonstrates how to set up Elasticsearch for sparse vector search by showing how to efficiently encode and index documents with sparse encoders, formulating search queries with sparse vectors, and providing an interactive query interface. See [semantic_search_elasticsearch.py](semantic_search_elasticsearch.py) or below:

### Prerequisites:
- Elasticsearch running locally (or accessible), see [Elasticsearch locally](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html) for more details.
- Python Elasticsearch client installed:
  ```bash
  pip install elasticsearch
  ```

```{eval-rst}

.. sidebar:: Documentation

   1. :class:`SparseEncoder <sentence_transformers.sparse_encoder.SparseEncoder>`
   2. :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.SparseEncoder.encode>`
   3. :meth:`semantic_search_elasticsearch <sentence_transformers.sparse_encoder.search_engines.semantic_search_elasticsearch>`
   4. `naver/splade-cocondenser-ensembledistil <https://huggingface.co/naver/splade-cocondenser-ensembledistil>`_
   5. `sentence-transformers/natural-questions <https://huggingface.co/datasets/sentence-transformers/natural-questions>`_

::

    import time

    from datasets import load_dataset

    from sentence_transformers import SparseEncoder
    from sentence_transformers.sparse_encoder.search_engines import semantic_search_elasticsearch

    # 1. Load the natural-questions dataset with 100K answers
    dataset = load_dataset("sentence-transformers/natural-questions", split="train")
    num_docs = 10_000
    corpus = dataset["answer"][:num_docs]

    # 2. Come up with some queries
    queries = dataset["query"][:2]

    # 3. Load the model
    sparse_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

    # 5. Encode the corpus
    corpus_embeddings = sparse_model.encode(corpus, convert_to_sparse_tensor=True, show_progress_bar=True)

    corpus_index = None
    while True:
        # 6. Encode the queries using the full precision
        start_time = time.time()
        query_embeddings = sparse_model.encode(queries, convert_to_sparse_tensor=True)
        print(f"Encoding time: {time.time() - start_time:.6f} seconds")

        # 7. Perform semantic search using qdrant
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
                score = entry["score"]
                corpus_id = entry["corpus_id"]
                print(f"(Score: {score:.4f}) {corpus[corpus_id]}, corpus_id: {corpus_id}")
            print("")

        # 10. Prompt for more queries
        queries = [input("Please enter a question: ")]
```
