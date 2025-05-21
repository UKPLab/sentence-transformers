"""
This is a simple application for sparse encoder: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""

from sentence_transformers import SparseEncoder, util

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

# 4. Use the similarity function to compute the similarity scores between the query and corpus embeddings
top_k = min(5, len(corpus))  # Find at most 5 sentences of the corpus for each query sentence
results = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k, score_function=model.similarity)

# 5. Sort the results and print the top 5 most similar sentences for each query
for query_id, query in enumerate(queries):
    pointwise_scores = model.intersection(query_embeddings[query_id], corpus_embeddings)

    print(f"Query: {query}")
    for res in results[query_id]:
        corpus_id, score = res.values()
        sentence = corpus[corpus_id]

        pointwise_score = model.decode(pointwise_scores[corpus_id], top_k=10)

        token_scores = ", ".join([f'("{token.strip()}", {value:.2f})' for token, value in pointwise_score])

        print(f"Score: {score:.4f} - Sentence: {sentence} - Top influential tokens: {token_scores}")
    print("")

"""
Query: A man is eating pasta.
Score: 21.3464 - Sentence: A man is eating food. - Top influential tokens: ("man", 5.48), ("eating", 3.83), ("eat", 3.15), ("men", 3.12), ("food", 1.78), ("male", 0.87), ("person", 0.62), ("a", 0.39), ("hunger", 0.28), ("meat", 0.27)
Score: 18.4783 - Sentence: A man is eating a piece of bread. - Top influential tokens: ("man", 4.85), ("eating", 3.49), ("eat", 3.02), ("men", 2.74), ("male", 0.68), ("food", 0.66), ("person", 0.58), ("a", 0.51), ("meat", 0.36), ("culture", 0.27)
Score: 10.2556 - Sentence: A man is riding a horse. - Top influential tokens: ("man", 4.85), ("men", 3.11), ("male", 0.68), ("a", 0.60), ("person", 0.59), ("animal", 0.21), ("god", 0.08), ("adam", 0.08), ("sex", 0.03), ("who", 0.01)
Score: 6.6108 - Sentence: A man is riding a white horse on an enclosed ground. - Top influential tokens: ("man", 3.31), ("men", 1.58), ("a", 0.51), ("male", 0.41), ("person", 0.34), ("on", 0.17), ("animal", 0.16), ("god", 0.05), ("wearing", 0.04), ("culture", 0.02)
Score: 5.2575 - Sentence: Two men pushed carts through the woods. - Top influential tokens: ("men", 2.60), ("man", 2.51), ("a", 0.12), ("murder", 0.01), (".", 0.01), ("said", 0.00)

Query: Someone in a gorilla costume is playing a set of drums.
Score: 16.7709 - Sentence: A monkey is playing drums. - Top influential tokens: ("drums", 4.38), ("drum", 2.27), ("play", 2.16), ("playing", 1.77), ("drummer", 0.80), ("dance", 0.67), ("monkey", 0.55), ("music", 0.50), ("a", 0.40), ("sound", 0.39)
Score: 8.7609 - Sentence: A woman is playing violin. - Top influential tokens: ("play", 2.12), ("playing", 1.79), ("dance", 0.68), ("person", 0.67), ("music", 0.55), ("instrument", 0.52), ("guitar", 0.39), ("a", 0.35), ("wearing", 0.32), ("player", 0.21)
Score: 2.8393 - Sentence: A man is riding a horse. - Top influential tokens: ("person", 0.91), ("a", 0.49), ("man", 0.45), ("animal", 0.37), ("sport", 0.32), ("savage", 0.10), ("dance", 0.08), ("billy", 0.06), ("god", 0.04), ("hunting", 0.01)
Score: 2.4528 - Sentence: A man is eating a piece of bread. - Top influential tokens: ("person", 0.90), ("man", 0.45), ("a", 0.42), ("someone", 0.29), ("animal", 0.08), ("god", 0.07), ("ritual", 0.07), ("culture", 0.07), ("something", 0.05), ("who", 0.03)
Score: 2.3295 - Sentence: A man is riding a white horse on an enclosed ground. - Top influential tokens: ("person", 0.53), ("a", 0.42), ("man", 0.31), ("sport", 0.27), ("animal", 0.27), ("savage", 0.09), ("character", 0.09), ("wearing", 0.07), ("symbol", 0.07), ("hunting", 0.05)

Query: A cheetah chases prey on across a field.
Score: 16.4271 - Sentence: A cheetah is running behind its prey. - Top influential tokens: ("che", 3.80), ("##eta", 3.72), ("prey", 2.77), ("hunting", 0.75), ("behavior", 0.70), ("##h", 0.62), ("movement", 0.45), ("animal", 0.33), ("predator", 0.30), ("chasing", 0.29)
Score: 2.2981 - Sentence: A monkey is playing drums. - Top influential tokens: ("animal", 0.43), ("a", 0.41), ("behavior", 0.28), ("hunting", 0.22), ("movement", 0.19), ("bird", 0.17), ("dance", 0.17), ("species", 0.07), ("dog", 0.07), ("bug", 0.07)
Score: 1.5377 - Sentence: A man is riding a horse. - Top influential tokens: ("a", 0.51), ("animal", 0.48), ("movement", 0.33), ("sport", 0.16), ("hunting", 0.04), ("dance", 0.02)
Score: 1.4831 - Sentence: A man is riding a white horse on an enclosed ground. - Top influential tokens: ("a", 0.43), ("animal", 0.35), ("hunting", 0.21), ("movement", 0.17), ("sport", 0.13), ("breed", 0.12), ("bird", 0.04), ("dog", 0.02)
Score: 1.4279 - Sentence: Two men pushed carts through the woods. - Top influential tokens: ("hunting", 0.49), ("cross", 0.41), ("move", 0.22), ("a", 0.10), ("escape", 0.08), ("they", 0.06), ("across", 0.05), ("obstacle", 0.01), ("deer", 0.01)
"""
