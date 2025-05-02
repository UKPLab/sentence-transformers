from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sentence_transformers.model_card import SentenceTransformerModelCardCallback, SentenceTransformerModelCardData
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    pass

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

    def register_model(self, model: SparseEncoder) -> None:
        self.model = model

        if self.task_name is None:
            self.task_name = "semantic search and sparse retrieval"
        if self.pipeline_tag is None:
            self.pipeline_tag = "feature-extraction"

    def tokenize(self, text: str | list[str]) -> dict[str, Any]:
        return self.model.tokenizer(text)

    def get_model_specific_metadata(self) -> dict[str, Any]:
        return {
            "model_max_length": self.model.get_max_seq_length(),
            "output_dimensionality": self.model.get_sentence_embedding_dimension(),
        }
