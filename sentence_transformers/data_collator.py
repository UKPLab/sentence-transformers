from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils import BatchEncoding, TruncationStrategy
from transformers.utils.generic import PaddingStrategy


@dataclass
class SentenceTransformerDataCollator:
    """Collator for a SentenceTransformers model.
    This encodes the text columns to {column}_input_ids and {column}_attention_mask columns.
    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/training/overview.html"""

    tokenizer: PreTrainedTokenizerBase

    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str, TruncationStrategy] = True
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {"return_loss": True}
        columns = list(features[0].keys())
        if "label" in features[0]:
            batch["label"] = torch.tensor([row["label"] for row in features])
            columns.remove("label")
        for column in columns:
            padded = self._encode([row[column] for row in features])
            batch[f"{column}_input_ids"] = padded.input_ids
            batch[f"{column}_attention_mask"] = padded.attention_mask
        return batch

    def _encode(self, texts: List[str]) -> BatchEncoding:
        return self.tokenizer(
            texts, padding=self.padding, truncation=self.truncation, return_tensors=self.return_tensors
        )
