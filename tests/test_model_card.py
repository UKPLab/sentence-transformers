from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.model_card import generate_model_card


@pytest.fixture(scope="session")
def dummy_dataset():
    """
    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "anchor": ["anchor 1", "anchor 2", ..., "anchor 10"],
        "positive": ["positive 1", "positive 2", ..., "positive 10"],
        "negative": ["negative 1", "negative 2", ..., "negative 10"],
    }
    """
    return Dataset.from_dict(
        {
            "anchor": [f"anchor {i}" for i in range(1, 11)],
            "positive": [f"positive {i}" for i in range(1, 11)],
            "negative": [f"negative {i}" for i in range(1, 11)],
        }
    )


@pytest.mark.parametrize(
    ("num_datasets", "expected_substrings"),
    [
        # 0 actually refers to just a single dataset
        (
            0,
            [
                "This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors).",
                "**Maximum Sequence Length:** 512 tokens",
                "**Output Dimensionality:** 128 dimensions",
                "**Similarity Function:** Cosine Similarity",
                "#### Unnamed Dataset",
                " | <code>anchor 1</code> | <code>positive 1</code> | <code>negative 1</code> |",
                "* Loss: [<code>CoSENTLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosentloss) with these parameters:",
            ],
        ),
        (
            1,
            [
                "This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on the train_0 dataset.",
                "#### train_0",
            ],
        ),
        (
            2,
            [
                "This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on the train_0 and train_1 datasets.",
                "#### train_0",
                "#### train_1",
            ],
        ),
        (
            10,
            [
                "This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on the train_0, train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8 and train_9 datasets.",
                "<details><summary>train_0</summary>",  # We start using <details><summary> if we have more than 3 datasets
                "#### train_0",
                "</details>\n<details><summary>train_9</summary>",
                "#### train_9",
            ],
        ),
        # We start using "50 datasets" when the ", "-joined dataset name exceed 200 characters
        (
            50,
            [
                "This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on 50 datasets.",
                "<details><summary>train_0</summary>",
                "#### train_0",
                "</details>\n<details><summary>train_49</summary>",
                "#### train_49",
            ],
        ),
    ],
)
def test_model_card_base(
    stsb_bert_tiny_model: SentenceTransformer,
    dummy_dataset: Dataset,
    num_datasets: int,
    expected_substrings: list[str],
) -> None:
    model = stsb_bert_tiny_model

    train_dataset = dummy_dataset
    if num_datasets:
        train_dataset = DatasetDict({f"train_{i}": train_dataset for i in range(num_datasets)})

    # This adds data to model.model_card_data
    SentenceTransformerTrainer(
        model,
        train_dataset=train_dataset,
    )

    model_card = generate_model_card(model)

    # For debugging purposes, we save the model card to a file
    # with open(f"test_model_card_{num_datasets}.md", "w", encoding="utf8") as f:
    #     f.write(model_card)

    for substring in expected_substrings:
        assert substring in model_card

    # We don't want to have two consecutive empty lines anywhere
    assert "\n\n\n" not in model_card
