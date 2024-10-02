from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


@dataclass
class SentenceTransformerDataCollator:
    """Collator for a SentenceTransformers model.
    This encodes the text columns to {column}_input_ids and {column}_attention_mask columns.
    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/sentence_transformer/training_overview.html

    It is important that the columns are in the expected order. For example, if your dataset has columns
    "answer", "question" in that order, then the MultipleNegativesRankingLoss will consider
    "answer" as the anchor and "question" as the positive, and it will (unexpectedly) optimize for
    "given the answer, what is the question?".
    """

    tokenize_fn: Callable
    valid_label_columns: list[str] = field(default_factory=lambda: ["label", "score"])
    _warned_columns: set[tuple[str]] = field(default_factory=set, init=False, repr=False)
    _warned_about_prompts: bool = field(default=False, repr=False)
    _columns_without_prompts: set[str] = field(default_factory=set, init=False, repr=False)
    prompts: dict[str, dict[str, str]] | dict[str, str] | str | None = None
    _prompt_lengths: dict[str, int] | dict[str, dict[str, int]] | int = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        # Extract the feature columns
        for column_name in column_names:
            values = [row[column_name] for row in features]
            sentences, prompt_len = self._maybe_add_prompts_and_lengths(
                values, column_name, batch.get("dataset_name", None)
            )
            tokenized = self.tokenize_fn(sentences)
            n_samples = len(values)
            for key, value in tokenized.items():
                batch[f"{column_name}_{key}"] = value
            if prompt_len is not None:
                batch[f"{column_name}_prompt_length"] = torch.Tensor(
                    [prompt_len] * n_samples, device=batch[f"{column_name}_input_ids"].device
                )
        self.maybe_warn_about_missing_prompt()
        return batch

    def _maybe_add_prompts_and_lengths(
        self, column_values: list[str], column_name: str, dataset_name: str | None = None
    ) -> tuple[list[str], int | None]:
        if self.prompts is None:
            return column_values, None
        # We expect to have a dictionary mapping column_name to prompt in the end.
        # if self.prompt is a nested dictionary, assume that the first key is the dataset name.
        if isinstance(self.prompts, str):
            prompt_dict = {column_name: self.prompts}
            prompt_length_dict = {column_name: self._prompt_lenghts}
        else:
            k = list(self.prompts.keys())[0]
            prompts_is_flat_dict = isinstance(self.prompts[k], str)
            if prompts_is_flat_dict:
                # if the dataset name is in the prompts, use it for all columns
                # Otherwise, Asume that the dictionary is mapping columns.
                # We deal with missing column_names later
                prompt_dict = self.prompts.get(dataset_name, self.prompts)
                prompt_length_dict = self._prompt_lenghts.get(dataset_name, self._prompt_lenghts)
            elif dataset_name in self.prompts:
                prompt_dict = self.prompts[dataset_name]
                prompt_length_dict = self._prompt_lenghts[dataset_name]
            else:
                raise ValueError(
                    f"A nested prompts dictionary was provided, but the dataset {dataset_name!r} was not found. The provided datasets are {self.prompts.keys}"
                )
        # prompt_dict should have a key with the column name.
        prompt = prompt_dict.get(column_name, None)
        prompt_len = prompt_length_dict.get(column_name, None)
        if prompt is None:
            self._columns_without_prompts.append(column_name)
            return column_values, None
        return column_values, prompt_len

    def set_prompts(self, prompts: dict[str, dict[str, str]] | dict[str, str] | str):
        self.prompts = prompts
        if isinstance(prompts, str):
            self.prompts = prompts
            tokenized_prompt = self.tokenize_fn([prompts])
            self._prompt_lenghts = len(tokenized_prompt["input_ids"]) - 1
        self._prompt_lengths = {}
        for key, value in prompts.items():
            if isinstance(value, str):
                tokenized_prompt = self.tokenize_fn(value)
                self._prompt_lengths[key] = len(tokenized_prompt["input_ids"]) - 1
            elif isinstance(value, dict):
                self._prompt_lengths[key] = {}
                for k, v in value.items():
                    tokenized_prompt = self.tokenize_fn(v)
                    self._prompt_lengths[key][k] = len(tokenized_prompt["input_ids"]) - 1
            else:
                raise ValueError(f"Invalid prompts type: {type(value)}")

    def maybe_warn_about_missing_prompt(self) -> None:
        if self._warned_about_prompts:
            return
        if len(self._columns_without_prompts) > 0:
            logger.warning(
                "You provided a dictionary of prompts per column to the data collator, but no "
                f"prompts to the columns {self._columns_without_prompts!r} wer provided. No prompt will be added to "
                "these columns. If it is an expected behavior, you can safely ignore this warning."
            )
            self._warned_about_prompts = True

    def maybe_warn_about_column_order(self, column_names: list[str]) -> None:
        """Warn the user if the columns are likely not in the expected order."""
        # A mapping from common column names to the expected index in the dataset
        column_name_to_expected_idx = {
            "anchor": 0,
            "positive": 1,
            "negative": 2,
            "question": 0,
            "answer": 1,
            "query": 0,
            "response": 1,
            "hypothesis": 0,
            "entailment": 1,
            "contradiction": 2,
        }
        for column_name, expected_idx in column_name_to_expected_idx.items():
            if column_name in column_names and column_names.index(column_name) != expected_idx:
                if column_name in ("anchor", "positive", "negative"):
                    proposed_fix_columns = ["anchor", "positive", "negative"]
                elif column_name in ("question", "answer"):
                    proposed_fix_columns = ["question", "answer"]
                elif column_name in ("query", "response"):
                    proposed_fix_columns = ["query", "response"]
                elif column_name in ("hypothesis", "entailment", "contradiction"):
                    proposed_fix_columns = ["hypothesis", "entailment", "contradiction"]

                logger.warning(
                    f"Column {column_name!r} is at index {column_names.index(column_name)}, whereas "
                    f"a column with this name is usually expected at index {expected_idx}. Note that the column "
                    "order can be important for some losses, e.g. MultipleNegativesRankingLoss will always "
                    "consider the first column as the anchor and the second as the positive, regardless of "
                    "the dataset column names. Consider renaming the columns to match the expected order, e.g.:\n"
                    f"dataset = dataset.select_columns({proposed_fix_columns})"
                )
                # We only need to warn once per list of column names to prevent spamming the user
                break

        self._warned_columns.add(tuple(column_names))
