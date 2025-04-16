from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import ModelCard

from sentence_transformers.model_card import (
    SentenceTransformerModelCardCallback,
    SentenceTransformerModelCardData,
)
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import (
        Dataset,
        DatasetDict,
        IterableDataset,
        IterableDatasetDict,
        Sequence,
        Value,
    )

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseEncoderModelCardCallback(SentenceTransformerModelCardCallback):
    pass


@dataclass
class SparseEncoderModelCardData(SentenceTransformerModelCardData):
    """A dataclass storing data used in the model card.

    Args:
        language (`Optional[Union[str, List[str]]]`): The model language, either a string or a list,
            e.g. "en" or ["en", "de", "nl"]
        license (`Optional[str]`): The license of the model, e.g. "apache-2.0", "mit",
            or "cc-by-nc-sa-4.0"
        model_name (`Optional[str]`): The pretty name of the model, e.g. "SparseEncoder based on answerdotai/ModernBERT-base".
        model_id (`Optional[str]`): The model ID when pushing the model to the Hub,
            e.g. "tomaarsen/se-mpnet-base-ms-marco".
        train_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the training datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}, {"name": "STSB"}]
        eval_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the evaluation datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"id": "mteb/stsbenchmark-sts"}]
        task_name (`str`): The human-readable task the model is trained on,
            e.g. "semantic search and sparse retrieval".
        tags (`Optional[List[str]]`): A list of tags for the model,
            e.g. ["sentence-transformers", "sparse-encoder"].

    .. tip::

        Install `codecarbon <https://github.com/mlco2/codecarbon>`_ to automatically track carbon emission usage and
        include it in your model cards.

    Example::

        >>> model = SparseEncoder(
        ...     "microsoft/mpnet-base",
        ...     model_card_data=SparseEncoderModelCardData(
        ...         model_id="tomaarsen/se-mpnet-base-allnli",
        ...         train_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],
        ...         eval_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],
        ...         license="apache-2.0",
        ...         language="en",
        ...     ),
        ... )
    """

    # Potentially provided by the user
    task_name: str = field(default=None)
    tags: list[str] | None = field(
        default_factory=lambda: [
            "sentence-transformers",
            "sparse-encoder",
        ]
    )

    # Automatically filled by `SparseEncoderModelCardCallback` and the Trainer directly
    predict_example: list[list[str]] | None = field(default=None, init=False)

    # Computed once, always unchanged
    pipeline_tag: str = field(default=None, init=False)
    template_path: Path = field(default=Path(__file__).parent / "model_card_template.md", init=False)

    # Passed via `register_model` only
    model: SparseEncoder | None = field(default=None, init=False, repr=False)

    def set_widget_examples(self, dataset: Dataset | DatasetDict) -> None:
        """
        We don't set widget examples, but only load the prediction example.
        This is because the Hugging Face Hub doesn't currently have a widget that accepts
        text input for sparse encoding.
        """
        if isinstance(dataset, DatasetDict):
            dataset = dataset[list(dataset.keys())[0]]

        if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
            # We can't set widget examples from an IterableDataset without losing data
            return

        if len(dataset) == 0:
            return

        columns = [
            column
            for column, feature in dataset.features.items()
            if (isinstance(feature, Value) and feature.dtype in {"string", "large_string"})
            or (
                isinstance(feature, Sequence)
                and isinstance(feature.feature, Value)
                and feature.feature.dtype in {"string", "large_string"}
            )
        ]
        if len(columns) < 1:
            return

        text_column = columns[0]
        texts = dataset[:5][text_column]

        if isinstance(texts[0], str):
            self.predict_example = [[text] for text in texts]

    def register_model(self, model) -> None:
        super().register_model(model)

        if self.task_name is None:
            self.task_name = "semantic search and sparse retrieval"
        if self.pipeline_tag is None:
            self.pipeline_tag = "feature-extraction"

    def tokenize(self, text: str | list[str]) -> dict[str, Any]:
        return self.model.tokenizer(text)

    def get_model_specific_metadata(self) -> dict[str, Any]:
        return {
            "model_max_length": self.model.max_length,
            "sparsity_type": self.model.sparsity_type,
            "vocab_size": self.model.vocab_size,
        }


def generate_model_card(model: SparseEncoder) -> str:
    template_path = Path(__file__).parent / "model_card_template.md"
    model_card = ModelCard.from_template(card_data=model.model_card_data, template_path=template_path, hf_emoji="🤗")
    return model_card.content
