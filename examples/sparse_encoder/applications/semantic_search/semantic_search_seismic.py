"""
This script contains an example how to perform semantic search with Seismic.
For more information, please refer to the documentation:
https://github.com/TusKANNy/seismic/blob/main/docs/Guidelines.md

All you need is installing the `pyseismic-lsr` package:
```
pip install pyseismic-lsr
```
"""

import time

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.search_engines import semantic_search_seismic

# 1. Load the natural-questions dataset with 100K answers
dataset = load_dataset("sentence-transformers/natural-questions", split="train")
num_docs = 10_000
corpus = dataset["answer"][:num_docs]

# 2. Come up with some queries
queries = dataset["query"][:2]

# 3. Load the model
sparse_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# 4. Encode the corpus
print("Start encoding corpus...")
start_time = time.time()
corpus_embeddings = sparse_model.encode(corpus, convert_to_sparse_tensor=True, batch_size=16, show_progress_bar=True)
corpus_embeddings_decoded = sparse_model.decode(corpus_embeddings)
print(f"Corpus encoding time: {time.time() - start_time:.6f} seconds")

corpus_index = None
while True:
    # 5. Encode the queries using the full precision
    start_time = time.time()
    query_embeddings = sparse_model.encode(queries, convert_to_sparse_tensor=True)
    query_embeddings_decoded = sparse_model.decode(query_embeddings)
    print(f"Encoding time: {time.time() - start_time:.6f} seconds")

    # 6. Perform semantic search using Seismic
    results, search_time, corpus_index = semantic_search_seismic(
        query_embeddings_decoded,
        corpus_embeddings_decoded=corpus_embeddings_decoded if corpus_index is None else None,
        corpus_index=corpus_index,
        top_k=5,
        output_index=True,
    )

    # 7. Output the results
    print(f"Search time: {search_time:.6f} seconds")
    for query, result in zip(queries, results):
        print(f"Query: {query}")
        for entry in result:
            print(f"(Score: {entry['score']:.4f}) {corpus[entry['corpus_id']]}, corpus_id: {entry['corpus_id']}")
        print("")

    # 8. Prompt for more queries
    queries = [input("Please enter a question: ")]
