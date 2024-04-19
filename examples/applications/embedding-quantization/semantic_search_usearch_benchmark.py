from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings, semantic_search_usearch
from datasets import load_dataset

# 1. Load the quora corpus with questions
dataset = load_dataset("quora", split="train").map(
    lambda batch: {"text": [text for sample in batch["questions"] for text in sample["text"]]},
    batched=True,
    remove_columns=["questions", "is_duplicate"],
)
max_corpus_size = 100_000
corpus = dataset["text"][:max_corpus_size]
num_queries = 1_000
queries = corpus[:num_queries]

# 2. Load the model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# 3. Encode the corpus
full_corpus_embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=True)

# 4. Encode the queries using the full precision
query_embeddings = model.encode(queries, normalize_embeddings=True)

for exact in (True, False):
    for corpus_precision in ("float32", "int8", "binary"):
        corpus_embeddings = quantize_embeddings(full_corpus_embeddings, precision=corpus_precision)
        # NOTE: We can also pass "precision=..." to the encode method to quantize the embeddings directly,
        # but we want to keep the full precision embeddings to act as a calibration dataset for quantizing
        # the query embeddings. This is important only if you are using uint8 or int8 precision

        # 5. Perform semantic search using usearch
        rescore_multiplier = 4
        results, search_time = semantic_search_usearch(
            query_embeddings,
            corpus_embeddings=corpus_embeddings,
            corpus_precision=corpus_precision,
            top_k=10,
            calibration_embeddings=full_corpus_embeddings,
            rescore=corpus_precision != "float32",
            rescore_multiplier=rescore_multiplier,
            exact=exact,
        )

        print(
            f"{'Exact' if exact else 'Approximate'} search time using {corpus_precision} corpus: {search_time:.6f} seconds"
            + (f" (rescore_multiplier: {rescore_multiplier})" if corpus_precision != "float32" else "")
        )
