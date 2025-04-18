import random

from datasets import load_dataset

from sentence_transformers.sparse_encoder import (
    MLMTransformer,
    SparseEncoder,
    SparseInformationRetrievalEvaluator,
    SpladePooling,
)

# Initialize the SPLADE model
model_name = "naver/splade-cocondenser-ensembledistil"
model = SparseEncoder(
    modules=[
        MLMTransformer(model_name),
        SpladePooling(pooling_strategy="max"),  # You can also use 'sum'
    ],
    device="cuda:0",
)

# Load the Touche-2020 IR dataset (https://huggingface.co/datasets/BeIR/webis-touche2020, https://huggingface.co/datasets/BeIR/webis-touche2020-qrels)
corpus = load_dataset("BeIR/webis-touche2020", "corpus", split="corpus")
queries = load_dataset("BeIR/webis-touche2020", "queries", split="queries")
relevant_docs_data = load_dataset("BeIR/webis-touche2020-qrels", split="test")

# For this dataset, we want to concatenate the title and texts for the corpus
corpus = corpus.map(lambda x: {"text": x["title"] + " " + x["text"]}, remove_columns=["title"])

# Shrink the corpus size heavily to only the relevant documents + 30,000 random documents
required_corpus_ids = set(map(str, relevant_docs_data["corpus-id"]))
required_corpus_ids |= set(random.sample(corpus["_id"], k=30_000))
corpus = corpus.filter(lambda x: x["_id"] in required_corpus_ids)

# Convert the datasets to dictionaries
corpus = dict(zip(corpus["_id"], corpus["text"]))  # Our corpus (cid => document)
queries = dict(zip(queries["_id"], queries["text"]))  # Our queries (qid => question)
relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for qid, corpus_ids in zip(relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]):
    qid = str(qid)
    corpus_ids = str(corpus_ids)
    if qid not in relevant_docs:
        relevant_docs[qid] = set()
    relevant_docs[qid].add(corpus_ids)

# Given queries, a corpus and a mapping with relevant documents, the SparseInformationRetrievalEvaluator computes different IR metrics.
ir_evaluator = SparseInformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="BeIR-touche2020-subset-test",
    show_progress_bar=True,
    batch_size=32,
)

# Run evaluation
print("Starting evaluation ")
results = ir_evaluator(model)

print(f"Primary metric: {ir_evaluator.primary_metric}")
print(f"Primary metric value: {results[ir_evaluator.primary_metric]:.4f}")

# Print results for each dataset
for key, value in results.items():
    if key.startswith("Nano"):
        print(f"{key}: {value:.4f}")
