from typing import Literal, overload

import numpy as np
from torch import Tensor, nn

from .fit_mixin import FitMixin
from .peft_mixin import PeftAdapterMixin

class SentenceTransformer(nn.Sequential, FitMixin, PeftAdapterMixin):
    # Return a single tensor because we're passing a single sentence.
    @overload
    def encode(
        self,
        sentences: str,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding", "token_embeddings"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: Literal[False] = ...,
        convert_to_tensor: bool = ...,
        device: str | None = ...,
        normalize_embeddings: bool = ...,
        **kwargs,
    ) -> Tensor: ...

    # Return a single array, because convert_to_numpy is True
    # and "sentence_embeddings" is passed
    @overload
    def encode(
        self,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: Literal[True] = ...,
        convert_to_tensor: Literal[False] = ...,
        device: str | None = ...,
        normalize_embeddings: bool = ...,
        **kwargs,
    ) -> np.ndarray: ...

    # Return a single tensor, because convert_to_tensor is True
    # and "sentence_embeddings" is passed
    @overload
    def encode(
        self,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: Literal[True] = ...,
        device: str | None = ...,
        normalize_embeddings: bool = ...,
        **kwargs,
    ) -> Tensor: ...

    # Return a list of tensors. Value of convert_ doesn't matter.
    @overload
    def encode(
        self,
        sentences: list[str] | np.ndarray,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding", "token_embeddings"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | None = ...,
        normalize_embeddings: bool = ...,
        **kwargs,
    ) -> list[Tensor]: ...

    # Return a list of dict of features, ignore the conversion args.
    @overload
    def encode(
        self,
        sentences: list[str] | np.ndarray,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: None = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | None = ...,
        normalize_embeddings: bool = ...,
        **kwargs,
    ) -> list[dict[str, Tensor]]: ...

    # Return a dict of features, ignore the conversion args.
    @overload
    def encode(
        self,
        sentences: str,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: None = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | None = ...,
        normalize_embeddings: bool = ...,
        **kwargs,
    ) -> dict[str, Tensor]: ...

    # If "token_embeddings" is True, then the output is a single tensor.
    @overload
    def encode(
        self,
        sentences: str,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["token_embeddings"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | None = ...,
        normalize_embeddings: bool = ...,
        **kwargs,
    ) -> Tensor: ...
