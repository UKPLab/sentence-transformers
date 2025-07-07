from __future__ import annotations

import json

from transformers import HfArgumentParser

from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers


def test_hf_argument_parser():
    # See https://github.com/UKPLab/sentence-transformers/issues/3090;
    # Ensure that the HfArgumentParser can be used to parse SentenceTransformerTrainingArguments.
    parser = HfArgumentParser(SentenceTransformerTrainingArguments)
    args = parser.parse_args(
        args=[
            "--output_dir",
            "test_output_dir",
            "--prompts",
            '{"query_column": "query_prompt", "positive_column": "positive_prompt", "negative_column": "negative_prompt"}',
            "--batch_sampler",
            "no_duplicates",
            "--multi_dataset_batch_sampler",
            "proportional",
            "--router_mapping",
            '{"dataset1": {"column_A": "query", "column_B": "document"}}',
            "--learning_rate_mapping",
            '{"dataset1": 0.001, "dataset2": 0.002}',
            "--learning_rate",
            "0.0005",
        ]
    )
    assert args.output_dir == "test_output_dir"
    assert json.loads(args.prompts) == {
        "query_column": "query_prompt",
        "positive_column": "positive_prompt",
        "negative_column": "negative_prompt",
    }
    assert args.batch_sampler == BatchSamplers.NO_DUPLICATES
    assert args.multi_dataset_batch_sampler == MultiDatasetBatchSamplers.PROPORTIONAL
    assert json.loads(args.router_mapping) == {"dataset1": {"column_A": "query", "column_B": "document"}}
    assert json.loads(args.learning_rate_mapping) == {
        "dataset1": 0.001,
        "dataset2": 0.002,
    }
    assert args.learning_rate == 0.0005
