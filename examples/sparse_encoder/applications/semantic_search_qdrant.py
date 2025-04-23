import time

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.search_engines import semantic_search_qdrant

# 1. Load the quora corpus with questions
dataset = load_dataset("quora", split="train", trust_remote_code=True).map(
    lambda batch: {"text": [text for sample in batch["questions"] for text in sample["text"]]},
    batched=True,
    remove_columns=["questions", "is_duplicate"],
)
max_corpus_size = 100_000
corpus = dataset["text"][:max_corpus_size]

# 2. Come up with some queries
queries = [
    "How do I become a good programmer?",
    "How do I become a good data scientist?",
]

# 3. Load the model
sparse_model = SparseEncoder("sparse-embedding/splade_example")


# 5. Encode the corpus
corpus_embeddings = sparse_model.encode(corpus, convert_to_sparse_tensor=True, batch_size=32, show_progress_bar=True)

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
        top_k=10,
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
