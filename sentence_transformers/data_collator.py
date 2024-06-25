from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import torch


@dataclass
class SentenceTransformerDataCollator:
    """Collator for a SentenceTransformers model.
    This encodes the text columns to {column}_input_ids and {column}_attention_mask columns.
    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/training/overview.html
    """

    tokenize_fn: Callable
    valid_label_columns: List[str] = field(default_factory=lambda: ["label", "score"])

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        columns = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in columns:
            columns.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in columns:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                columns.remove(label_column)
                break

        # Extract the feature columns
        for column in columns:
            tokenized = self.tokenize_fn([row[column] for row in features])
            for key, value in tokenized.items():
                batch[f"{column}_{key}"] = value
        return batch
