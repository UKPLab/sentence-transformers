"""
This examples evaluates different similarity metrics on a set of datasets.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from datasets import load_dataset, Dataset
import logging
import sys

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout


def evaluate(
    model: SentenceTransformer,
    dataset: Dataset,
    max_samples: int,
    batch_size: int,
    use_train_as_ensemble: bool,
    seed: int = 42,
):
    dataset = dataset.shuffle(seed=seed)
    samples = {
        split: [
            InputExample(
                texts=[row["sentence1"], row["sentence2"]],
                label=row["similarity_score"],
            )
            for row in dataset[split].select(range(max_samples))
        ]
        for split in ("train", "test")
    }
    if use_train_as_ensemble:
        ensemble_texts = list(set(text for inp_example in samples["train"] for text in inp_example.texts))
    else:
        ensemble_texts = None

    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        samples["test"],
        name=f"{dataset_name}-test",
        ensemble_texts=ensemble_texts,
        show_progress_bar=True,
        batch_size=batch_size,
    )
    test_evaluator(model)


if __name__ == "__main__":
    max_samples = 1000
    batch_size = 64
    use_train_as_ensemble = False
    model_name = sys.argv[1] if len(sys.argv) > 1 else "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    dataset_name = "Ukhushn/home-depot"
    logging.info(f"Read {dataset_name} dataset")
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.train_test_split(test_size=0.5)
    dataset = dataset.map(
        lambda row: {
            "sentence1": row["search_term"],
            "sentence2": row["product_title"] + row["product_description"],
            "similarity_score": row["relevance"] / 3,
        }
    )
    evaluate(
        model,
        dataset,
        min(max_samples, dataset["test"].num_rows),
        batch_size,
        use_train_as_ensemble,
    )

    dataset_name = "pranavkotz/anatomy_dataset"
    logging.info(f"Read {dataset_name} dataset")
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.train_test_split(test_size=0.5)
    evaluate(
        model,
        dataset,
        min(max_samples, dataset["test"].num_rows),
        batch_size,
        use_train_as_ensemble,
    )

"""
Results:

2024-02-08 23:08:25 - EmbeddingSimilarityEvaluator: Evaluating the model on Ukhushn/home-depot-test dataset:
2024-02-08 23:09:48 - Cosine-Similarity :       Pearson: 0.4157 Spearman: 0.4074
2024-02-08 23:09:48 - Manhattan-Distance:       Pearson: 0.4186 Spearman: 0.4106
2024-02-08 23:09:48 - Euclidean-Distance:       Pearson: 0.4149 Spearman: 0.4074
2024-02-08 23:09:48 - Dot-Product-Similarity:   Pearson: 0.4157 Spearman: 0.4074
2024-02-08 23:09:48 - Surprise-Similarity:      Pearson: 0.2783 Spearman: 0.3593
2024-02-08 23:09:48 - Surprise-Similarity-Dev:  Pearson: 0.3697 Spearman: 0.3592


2024-02-08 23:09:50 - EmbeddingSimilarityEvaluator: Evaluating the model on pranavkotz/anatomy_dataset-test dataset:
2024-02-08 23:09:58 - Cosine-Similarity :       Pearson: 0.9484 Spearman: 0.9297
2024-02-08 23:09:58 - Manhattan-Distance:       Pearson: 0.9430 Spearman: 0.9304
2024-02-08 23:09:58 - Euclidean-Distance:       Pearson: 0.9430 Spearman: 0.9297
2024-02-08 23:09:58 - Dot-Product-Similarity:   Pearson: 0.9484 Spearman: 0.9297
2024-02-08 23:09:58 - Surprise-Similarity:      Pearson: 0.4786 Spearman: 0.8827
2024-02-08 23:09:58 - Surprise-Similarity-Dev:  Pearson: 0.8946 Spearman: 0.8902
"""
