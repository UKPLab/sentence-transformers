"""
This script contains an example how to perform semantic search with Elasticsearch.

You need Elasticsearch up and running locally:
https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html

Further, you need the Python Elasticsearch Client installed: https://elasticsearch-py.readthedocs.io/, e.g.:
```
pip install elasticsearch
```
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
            print(f"(Score: {entry['score']:.4f}) {corpus[entry['corpus_id']]}, corpus_id: {entry['corpus_id']}")
        print("")

    # 10. Prompt for more queries
    queries = [input("Please enter a question: ")]
