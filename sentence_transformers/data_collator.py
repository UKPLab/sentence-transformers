from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import torch


@dataclass
class SentenceTransformerDataCollator:
    """Collator for a SentenceTransformers model.
    This encodes the text columns to {column}_input_ids and {column}_attention_mask columns.
    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/training/overview.html"""

    tokenize_fn: Callable

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {"return_loss": True}
        columns = list(features[0].keys())
        if "label" in features[0]:
            batch["label"] = torch.tensor([row["label"] for row in features])
            columns.remove("label")
        for column in columns:
            tokenized = self.tokenize_fn([row[column] for row in features])
            for key, value in tokenized.items():
                batch[f"{column}_{key}"] = value
        return batch
