# Evaluation with MTEB

The [Massive Text Embedding Benchmark (MTEB)](https://github.com/embeddings-benchmark/mteb) is a comprehensive benchmark suite for evaluating embedding models across diverse NLP tasks like retrieval, classification, clustering, reranking, and semantic similarity.

This guide walks you through using MTEB with SentenceTransformer models for post-training evaluation. This is *not* designed for use during training, as this risks overfitting on public benchmarks. For evaluation during training, please see the [Evaluator section in the Training Overview](../training_overview.md#evaluator). To fully integrate your model to MTEB, you can follow the [Adding a model to the Leaderboard](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md) guide from the MTEB Documentation.

## Installation

Install MTEB and its dependencies:

```bash
pip install mteb
```

## Evaluation

You can evaluate your SentenceTransformer model on individual tasks from the MTEB suite like so:

```python
import mteb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Example 1: Run a specific single task
tasks = mteb.get_tasks(tasks=["STS22.v2"], languages=["eng"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results/")
```

For the full list of available tasks, you can check the [MTEB Tasks documentation](https://github.com/embeddings-benchmark/mteb/blob/main/docs/tasks.md).

You can also filter available MTEB tasks based on task type, domain, language, and more.
For example, the following snippet evaluates on English retrieval tasks in the medical domain:

```python
import mteb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Example 2: Run all English retrieval tasks in the medical domain
tasks = mteb.get_tasks(
    task_types=["Retrieval"],
    domains=["Medical"],
    languages=["eng"]
)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results/")
```

Lastly, it's often valuable to evaluate on predefined benchmarks. For example, to run all retrieval tasks in the `MTEB(eng, v2)` benchmark:

```python
import mteb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Example 3: Run the MTEB benchmark for English tasks
benchmark = mteb.get_benchmark("MTEB(eng, v2)")
evaluation = mteb.MTEB(tasks=benchmark)
results = evaluation.run(model, output_folder="results/")
```

For the full list of supported benchmarks, visit the [MTEB Benchmarks documentation](https://github.com/embeddings-benchmark/mteb/blob/main/docs/benchmarks.md).

## Additional Arguments

When running evaluations, you can pass arguments down to `model.encode()` using the `encode_kwargs` parameter on `evaluation.run()`. This allows you to customize how embeddings are generated, such as setting `batch_size`, `truncate_dim`, or `normalize_embeddings`. For example:

```python
...

results = evaluation.run(
    model,
    verbosity=2,
    output_folder="results/",
    encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
)
```

Additionally, your SentenceTransformer model may have been configured to use `prompts`. MTEB will automatically detect and use these prompts if they are defined in your model's configuration. For task-specific or document/query-specific prompts, you should read the MTEB Documentation on [Running SentenceTransformer models with prompts](https://github.com/embeddings-benchmark/mteb/blob/main/docs/usage/usage.md#running-sentencetransformer-model-with-prompts).

## Results Handling

MTEB caches all results to disk, so you can rerun `evaluation.run()` without needing to redownload datasets or recomputing scores. 

```python
import mteb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

tasks = mteb.get_tasks(tasks=["STS17", "STS22.v2"], languages=["eng"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results/")

for task_results in results:
    # Print the aggregated main scores for each task
    print(f"{task_results.task_name}: {task_results.get_score():.4f} mean {task_results.task.metadata.main_score}")
    """
    STS17: 0.2881 mean cosine_spearman
    STS22.v2: 0.4925 mean cosine_spearman
    """

    # Or e.g. print the individual scores for each split or subset
    print(task_results.only_main_score().to_dict())
```

## Leaderboard Submission

To add your model to the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard), you will need to follow the [Adding a Model](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md) MTEB Documentation.

For the process, you'll need to follow these steps:
1. Add your model metadata (name, languages, number of parameters, framework, training datasets, etc.) to the [MTEB Repository](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/models).
2. Evaluate your model using MTEB on your desired tasks and save the results.
2. Submit your results to the [MTEB Results Repository](https://github.com/embeddings-benchmark/results).

Once both are merged, after a day you'll be able to find your model on the [official leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
