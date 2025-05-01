from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
from tokenizers import Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from sentence_transformers.models.Module import Module


# TODO: BaseModule? ModuleBase?
# TODO: "WithTokenizer" might be problematic if we're going multi-modal
class ModuleWithTokenizer(Module):
    save_in_root: bool = True
    tokenizer: PreTrainedTokenizerBase | Tokenizer

    @abstractmethod
    def tokenize(self, texts: list[str], **kwargs) -> dict[str, torch.Tensor | Any]: ...

    def save_tokenizer(self, output_path: str, **kwargs) -> None:
        if not hasattr(self, "tokenizer"):
            return

        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            self.tokenizer.save_pretrained(output_path, **kwargs)
        elif isinstance(self.tokenizer, Tokenizer):
            self.tokenizer.save(output_path, **kwargs)
        return
