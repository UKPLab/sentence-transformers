# Evaluating SentenceTransformer Models with MTEB

The [Massive Text Embedding Benchmark (MTEB)](https://github.com/embeddings-benchmark/mteb) is a comprehensive benchmark suite for evaluating embedding models across diverse NLP tasks like classification, retrieval, clustering, reranking, and semantic similarity.

This guide walks you through using MTEB **with SentenceTransformer models for post-training evaluation**. This is *not* designed for use during training loops. To fully integrate your model to `MTEB` you can follow this [guide](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md)

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
from mteb import MTEB, get_tasks

model = SentenceTransformer("all-MiniLM-L6-v2")

# Example 1: Run a specific single task (STS22)
tasks = get_tasks(["STS22"])
evaluation = MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results/")


You can filter available MTEB tasks based on task type, domain, and language.  
For example, the following snippet evaluates on **English retrieval tasks in the medical domain**:


# Example 2: Filtered tasks by type, domain, and language
# This fetches Retrieval tasks from the Medical domain in English
filtered_tasks = get_tasks(
    task_types=["Retrieval"],
    domains=["Medical"],
    languages=["en"]
)

# Evaluate on filtered tasks
evaluation = MTEB(tasks=filtered_tasks)
results = evaluation.run(model, output_folder="results/medical_retrieval/")
```

This evaluates your model on **STS22**, a multilingual semantic similarity dataset from the SemEval 2022 challenge. Output is saved in `results/`.

---
> Note: The following tasks are only examples.

> For the full list of supported benchmarks, visit the [MTEB GitHub repo](https://github.com/embeddings-benchmark/mteb#tasks).

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
mteb.get_tasks(tasks=["SICK-R", "AmazonCounterfactualClassification"])
mteb.get_tasks(task_types=["Classification", "Retrieval"])
```

---

##  Customize Output and Results Handling

To avoid writing results to disk:

```python
evaluation.run(model, output_folder=None)
```

To extract scores programmatically:

```python

from mteb import MTEBResults

results = evaluation.run(model, output_folder="results/")
for task, scores in results.items():
    print(f"{task}: {scores['main_score']}")
```

To export all results as a Markdown table:

```python
df = MTEBResults(results).to_dataframe()
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

To add your results to the MTEB Leaderboard, follow the submission instructions in the [MTEB repository](https://github.com/embeddings-benchmark/mteb).


## 📚 References

* [MTEB GitHub](https://github.com/embeddings-benchmark/mteb)
* [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
* [SentenceTransformers Docs](https://www.sbert.net/)
