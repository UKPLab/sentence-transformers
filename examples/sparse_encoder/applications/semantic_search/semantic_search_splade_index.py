"""
This script contains an example how to perform semantic search with splade_index.
For more information, please refer to the documentation:
https://github.com/rasyosef/splade-index

All you need is installing the `splade-index` package:
```
pip install splade-index
```
"""

import time

from datasets import load_dataset
from splade_index import SPLADE

from sentence_transformers import SparseEncoder

# 1. Load the natural-questions dataset with 100K answers
dataset = load_dataset("sentence-transformers/natural-questions", split="train")
num_docs = 10_000
corpus = dataset["answer"][:num_docs]

# 2. Come up with some queries
queries = dataset["query"][:2]

# 3. Load the model
sparse_model = SparseEncoder("rasyosef/splade-tiny")

# 4. Encode the corpus & create the index
print("Start encoding corpus and creating index...")
start_time = time.time()
corpus_index = SPLADE()
corpus_index.index(model=sparse_model, documents=corpus, batch_size=16, show_progress=True)
print(f"Encoded corpus and created index in {time.time() - start_time:.6f} seconds")

while True:
    # 5. Encode the queries using the full precision
    start_time = time.time()
    all_doc_ids, all_documents, all_scores = corpus_index.retrieve(queries, k=5)
    print(f"Encoding & Search time: {time.time() - start_time:.6f} seconds")

    # 7. Output the results
    for query, doc_ids, documents, scores in zip(queries, all_doc_ids, all_documents, all_scores):
        print(f"Query: {query}")
        for doc_id, document, score in zip(doc_ids, documents, scores):
            print(f"(Score: {score:.4f}) {document}, corpus_id: {doc_id}")
        print("")

    # 8. Prompt for more queries
    queries = [input("Please enter a question: ")]
