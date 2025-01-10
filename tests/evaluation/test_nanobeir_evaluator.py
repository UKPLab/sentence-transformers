from __future__ import annotations

import re

import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator


def test_nanobeir_evaluator():
    """Tests that the NanoBERTEvaluator can be loaded and produces expected metrics"""
    datasets = ["QuoraRetrieval", "MSMARCO"]
    query_prompts = {
        "QuoraRetrieval": "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\\nQuery: ",
        "MSMARCO": "Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery: ",
    }

    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")

    evaluator = NanoBEIREvaluator(
        dataset_names=datasets,
        query_prompts=query_prompts,
    )

    results = evaluator(model)

    assert len(results) > 0
    assert all(isinstance(results[metric], float) for metric in results)


def test_nanobeir_evaluator_with_invalid_dataset():
    """Test that NanoBEIREvaluator raises an error for invalid dataset names."""
    invalid_datasets = ["invalidDataset"]

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"Dataset(s) ['invalidDataset'] not found in the NanoBEIR collection. "
            r"Valid dataset names are: ['climatefever', 'dbpedia', 'fever', 'fiqa2018', 'hotpotqa', 'msmarco', 'nfcorpus', 'nq', 'quoraretrieval', 'scidocs', 'arguana', 'scifact', 'touche2020']"
        ),
    ):
        NanoBEIREvaluator(dataset_names=invalid_datasets)


def test_nanobeir_evaluator_empty_inputs():
    """Test that NanoBEIREvaluator behaves correctly with empty datasets."""
    with pytest.raises(ValueError, match="dataset_names cannot be empty. Use None to evaluate on all datasets."):
        NanoBEIREvaluator(dataset_names=[])
