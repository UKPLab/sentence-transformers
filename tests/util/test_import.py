from __future__ import annotations

import importlib

import pytest

from sentence_transformers import util


@pytest.mark.parametrize(
    "function_name",
    [
        "community_detection",
        "cos_sim",
        "dot_score",
        "pytorch_cos_sim",
        "normalize_embeddings",
        "paraphrase_mining",
        "semantic_search",
        "http_get",
        "batch_to_device",
    ],
)
def test_direct_import(function_name):
    """Test that functions can be imported directly from sentence_transformers.util"""
    # Import the function dynamically
    module = importlib.import_module("sentence_transformers.util")
    imported_function = getattr(module, function_name)

    # Verify the function exists and is callable
    assert imported_function is not None
    assert callable(imported_function)


@pytest.mark.parametrize(
    "function_name",
    [
        "community_detection",
        "cos_sim",
        "dot_score",
        "pytorch_cos_sim",
        "normalize_embeddings",
        "paraphrase_mining",
        "semantic_search",
        "http_get",
        "batch_to_device",
    ],
)
def test_module_import(function_name):
    """Test that functions can be accessed via the util module"""
    # Check that function exists in the module
    assert hasattr(util, function_name)

    # Check that it's callable
    assert callable(getattr(util, function_name))


def test_import_reload():
    """Test that util module can be reloaded without errors"""
    importlib.reload(util)

    # Verify module still works after reload
    assert hasattr(util, "cos_sim")
    assert callable(util.cos_sim)
