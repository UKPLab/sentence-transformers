from __future__ import annotations

import json

from transformers import HfArgumentParser

from sentence_transformers import SentenceTransformerTrainingArguments


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
        ]
    )
    assert args.output_dir == "test_output_dir"
    assert json.loads(args.prompts) == {
        "query_column": "query_prompt",
        "positive_column": "positive_prompt",
        "negative_column": "negative_prompt",
    }
