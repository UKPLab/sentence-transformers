# Evaluating SentenceTransformer Models with MTEB

The [Massive Text Embedding Benchmark (MTEB)](https://github.com/embeddings-benchmark/mteb) is a comprehensive benchmark suite for evaluating embedding models across diverse NLP tasks like classification, retrieval, clustering, reranking, and semantic similarity.

This guide walks you through using MTEB **with SentenceTransformer models for post-training evaluation**. This is *not* designed for use during training loops.

---

## Installation

Install MTEB and its dependencies:

```bash
pip install mteb
```

> This also installs `sentence-transformers`, `datasets`, and task-specific dependencies.

---

##  Quick Start: One-Line Evaluation

```python
from sentence_transformers import SentenceTransformer
from mteb import MTEB

model = SentenceTransformer("all-MiniLM-L6-v2")
evaluation = MTEB(tasks=["STSBenchmark"])
evaluation.run(model, output_folder="results/")
```

This evaluates your model on the **STS Benchmark**, a Semantic Textual Similarity dataset with human-annotated sentence pairs. Output is saved in `results/`.

---

##  Supported Task Types and Examples

MTEB supports the following **task families**:

### 1. **Semantic Textual Similarity (STS)**

* **STSBenchmark**: Predict similarity score (0-5) for English sentence pairs.
* **BIOSSES**: Biomedical sentence similarity.
* **SICK-R**: Sentence Involving Compositional Knowledge with human ratings.

### 2. **Classification**

* **AmazonCounterfactualClassification**: Predict product sentiment in counterfactual reviews.
* **TwitterSentiment**: Predict sentiment on tweets (positive/negative).
* **MassiveIntentClassification**: Classify user intent in multi-lingual commands.

### 3. **Retrieval (Information Retrieval)**

* **TREC-COVID**: Retrieve COVID-19 papers for search queries.
* **SciFact**: Retrieve abstracts relevant to scientific claims.
* **NFCorpus**: Retrieve passages for health and nutrition questions.

### 4. **Reranking**

* **MSMARCO**: Rerank candidate passages by relevance to a query.
* **StackExchangeReranking**: Rerank answers on StackExchange questions.

### 5. **Bitext Mining**

* **BUCC**: Identify aligned sentences in multilingual corpora.
* **Tatoeba**: Match translations across 100+ languages.

You can evaluate specific tasks or full categories:

```python
MTEB(tasks=["SICK-R", "AmazonCounterfactualClassification"])
MTEB(task_types=["Classification", "Retrieval"])
```

---

##  Customize Output and Results Handling

To avoid writing results to disk:

```python
evaluation.run(model, output_folder=None)
```

To extract scores programmatically:

```python
results = evaluation.run(model, output_folder=None)

from mteb import MTEBResults
summary = MTEBResults(results).main_scores()
print(summary)
```

To export all results as a Markdown table:

```python
df = MTEBResults(results).to_markdown()
print(df)
```

---


**Important**: MTEB is for *post-training* benchmarking only.

* Using it during training risks **overfitting** to public benchmarks.
* It writes to disk and caches aggressively.
* Official guidance recommends using SentenceTransformer's built-in evaluators like:

  * `EmbeddingSimilarityEvaluator`
  * `BinaryClassificationEvaluator`

---

## Submitting to the Leaderboard

You can compare your results on the [official leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

Export your scores:

```python
from mteb import MTEBResults
df = MTEBResults(results).to_markdown()
print(df)
```

Follow submission instructions in the [MTEB repo](https://github.com/embeddings-benchmark/mteb).


## 📚 References

* [MTEB GitHub](https://github.com/embeddings-benchmark/mteb)
* [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
* [SentenceTransformers Docs](https://www.sbert.net/)
