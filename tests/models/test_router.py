from __future__ import annotations

import importlib
import os
import re
import tempfile
from copy import deepcopy

import pytest
import torch
from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.models import Asym, Dense, Normalize, Router
from sentence_transformers.models.InputModule import InputModule


class MockModule(InputModule):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        return features

    def tokenize(self, texts, **kwargs):
        return {}

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        pass


class MockModuleWithMaxLength(MockModule):
    def __init__(self, max_seq_length=32):
        super().__init__()
        self.max_seq_length = max_seq_length


# Create a custom dict subclass to track access
class TaskTypesTrackingDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_types = []

    def get(self, key, default=None):
        self.task_types.append(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.task_types.append(key)
        return super().__getitem__(key)


@pytest.mark.parametrize("routes", [{}, None])
def test_router_empty_routes_raises_value_error(routes):
    """Test that Router raises ValueError when initialized with empty routes dictionary or None."""
    with pytest.raises(ValueError, match="The routes dictionary cannot be empty."):
        Router(routes)


def test_router_max_seq_length_edges():
    # Fabricate a module without max_seq_length to test the default behavior
    module = MockModule()
    router = Router({"route_1": [module]})
    model = SentenceTransformer(modules=[router])
    assert model.max_seq_length is None

    # Use a single module with a max_seq_length
    module_with_max_length = MockModuleWithMaxLength(128)
    router = Router({"route_1": [module], "route_2": [module_with_max_length]})
    model = SentenceTransformer(modules=[router])
    assert model.max_seq_length == 128

    # With multiple routes, the max_seq_length should be the maximum of the individual modules
    module_one = MockModuleWithMaxLength(256)
    module_two = MockModuleWithMaxLength(512)
    module_three = MockModuleWithMaxLength(128)
    router = Router(
        {
            "route_1": [module_one],
            "route_2": [module_two],
            "route_3": [module_three],
        }
    )
    model = SentenceTransformer(modules=[router])
    assert model.max_seq_length == 512

    model.max_seq_length = 1024
    assert module_one.max_seq_length == 1024
    assert module_two.max_seq_length == 1024
    assert module_three.max_seq_length == 1024


def test_router_init_basic():
    """Test basic initialization of Router."""
    query_module = MockModuleWithMaxLength(256)
    doc_module = MockModuleWithMaxLength(512)

    router = Router({"query": [query_module], "document": [doc_module]})

    assert router.sub_modules == {"query": [query_module], "document": [doc_module]}
    assert router.default_route == "query"  # First key with allow_empty_key=True

    router = Router(
        {
            "document": [doc_module],
            "query": [query_module],
        }
    )

    assert router.sub_modules == {"query": [query_module], "document": [doc_module]}
    assert router.default_route == "document"  # First key with allow_empty_key=True


def test_router_init_with_default_route():
    """Test initialization with explicit default route."""
    query_module = MockModuleWithMaxLength()
    doc_module = MockModuleWithMaxLength()

    router = Router({"query": [query_module], "document": [doc_module]}, default_route="document")

    assert router.default_route == "document"


def test_router_init_without_default_route():
    """Test initialization without default route and allow_empty_key=False."""
    query_module = MockModuleWithMaxLength()
    doc_module = MockModuleWithMaxLength()

    router = Router({"query": [query_module], "document": [doc_module]}, allow_empty_key=False)

    assert router.default_route is None


def test_router_init_invalid_default_route():
    """Test initialization with invalid default route raises ValueError."""
    module = MockModuleWithMaxLength()

    with pytest.raises(ValueError, match="Default route 'invalid' not found in route keys"):
        Router({"query": [module]}, default_route="invalid")


def test_router_init_multiple_modules_per_route():
    """Test initialization with multiple modules per route."""
    module1 = MockModuleWithMaxLength()
    module2 = MockModuleWithMaxLength()  # Technically, this should be a Module subclass, not an InputModule subclass
    module3 = MockModuleWithMaxLength()

    router = Router({"query": [module1, module2], "document": [module3]})

    assert router.sub_modules["query"] == [module1, module2]
    assert router.sub_modules["document"] == [module3]


def test_router_encode(static_embedding_model):
    """Test encoding with Router."""
    # Create a Router with StaticEmbedding modules
    router = Router({"query": [static_embedding_model], "document": [static_embedding_model]})

    # Replace the dictionary with our tracking version
    tracking_dict = TaskTypesTrackingDict(router.sub_modules)
    router.sub_modules = tracking_dict

    model = SentenceTransformer(modules=[router])

    # Test encoding
    query_texts = ["What is the capital of France?"]
    doc_texts = ["The capital of France is Paris."]

    model.encode_query(query_texts)
    assert "query" in tracking_dict.task_types
    tracking_dict.task_types = []

    model.encode_document(doc_texts)
    assert "document" in tracking_dict.task_types
    tracking_dict.task_types = []

    # The default route should be used if no task type is specified
    model.encode(query_texts)
    assert router.default_route == "query"
    assert "query" in tracking_dict.task_types
    tracking_dict.task_types = []

    # Test with a different default route
    router.default_route = "document"
    model.encode(doc_texts)
    assert "document" in tracking_dict.task_types

    # Test with an incorrect route
    with pytest.raises(
        ValueError, match=re.escape("No route found for task type 'invalid'. Available routes: ['query', 'document']")
    ):
        model.encode("This should fail", task_type="invalid")

    router.default_route = None  # Reset default route to None
    with pytest.raises(
        ValueError,
        match=re.escape("``task_type`` must be specified, or the ``Router`` must have a ``default_route`` set."),
    ):
        model.encode(doc_texts)


def test_router_is_alias_for_asym():
    """Test that Router is an alias for Asym."""

    assert Router is Asym


def test_router_backwards_compatibility(static_embedding_model):
    """Test that Router can load models saved with Asym."""

    # Create a mock Asym model
    asym_model = Asym({"query": [static_embedding_model], "document": [static_embedding_model]})

    # Replace the dictionary with our tracking version
    tracking_dict = TaskTypesTrackingDict(asym_model.sub_modules)
    asym_model.sub_modules = tracking_dict

    model = SentenceTransformer(modules=[asym_model])
    model.encode([{"query": "What is the capital of France?"}, {"query": "The capital of France is Paris."}])
    assert tracking_dict.task_types == ["query", "query"]
    tracking_dict.task_types = []

    model.encode([{"document": "What is the capital of France?"}, {"document": "The capital of France is Paris."}])
    assert tracking_dict.task_types == ["document", "document"]
    tracking_dict.task_types = []


@pytest.mark.parametrize(
    ("module_names", "module_attributes"),
    [
        (
            [
                "sentence_transformers.models.Asym",
                "sentence_transformers.models.Router",
                "sentence_transformers.models",
            ],
            [Asym, Router],
        ),
    ],
)
def test_asym_import(module_names: list[str], module_attributes: list[object]) -> None:
    for module_name in module_names:
        module = importlib.import_module(module_name)
        for module_attribute in module_attributes:
            obj = getattr(module, module_attribute.__name__, None)
            assert obj is module_attribute


def test_router_save_load(static_embedding_model):
    """Test saving and loading a SentenceTransformer model with Router."""
    # Create a Router with StaticEmbedding modules
    router = Router({"query": [static_embedding_model], "document": [static_embedding_model]})
    model = SentenceTransformer(modules=[router])

    # Test data for encoding
    query_texts = ["What is the capital of France?"]
    doc_texts = ["The capital of France is Paris."]

    # Get original embeddings
    query_embeddings_original = model.encode_query(query_texts)
    doc_embeddings_original = model.encode_document(doc_texts)

    # Save the model to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model")
        model.save(model_path)

        # Load the model
        loaded_model = SentenceTransformer(model_path)

        # Verify loaded model structure
        assert len(list(loaded_model.children())) == 1
        assert isinstance(loaded_model[0], Router)
        loaded_router = loaded_model[0]
        assert set(loaded_router.sub_modules.keys()) == {"query", "document"}
        assert loaded_router.default_route == "query"

        # Get embeddings from loaded model
        query_embeddings_loaded = loaded_model.encode_query(query_texts)
        doc_embeddings_loaded = loaded_model.encode_document(doc_texts)

        # Verify embeddings are the same
        assert (query_embeddings_original == query_embeddings_loaded).all()
        assert (doc_embeddings_original == doc_embeddings_loaded).all()


def test_router_save_load_with_custom_default_route(static_embedding_model):
    """Test saving and loading a model with custom default route."""
    router = Router(
        {"query": [static_embedding_model], "document": [static_embedding_model]}, default_route="document"
    )
    model = SentenceTransformer(modules=[router])

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model")
        model.save(model_path)

        loaded_model = SentenceTransformer(model_path)
        loaded_router = loaded_model[0]

        # Verify default route was preserved
        assert loaded_router.default_route == "document"

        # Test that default encoding uses the document route
        texts = ["Test text"]
        default_embeddings = loaded_model.encode(texts)
        doc_embeddings = loaded_model.encode_document(texts)
        assert (default_embeddings == doc_embeddings).all()


def test_router_save_load_without_default_route(static_embedding_model):
    """Test saving and loading a model without a default route."""
    router = Router({"query": [static_embedding_model], "document": [static_embedding_model]}, allow_empty_key=False)
    model = SentenceTransformer(modules=[router])

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model")
        model.save(model_path)

        loaded_model = SentenceTransformer(model_path)
        loaded_router = loaded_model[0]
        # Verify default route is None
        assert loaded_router.default_route is None

        # Test that encoding without task_type raises error
        with pytest.raises(
            ValueError,
            match=re.escape("``task_type`` must be specified, or the ``Router`` must have a ``default_route`` set."),
        ):
            loaded_model.encode(["Test text"])


def test_router_save_load_with_multiple_modules_per_route(static_embedding_model):
    """Test saving and loading a model with multiple modules per route."""
    # Create two different mock modules for testing
    static_embedding_model_one = deepcopy(static_embedding_model)
    static_embedding_model_two = deepcopy(static_embedding_model)
    dense = Dense(in_features=static_embedding_model.get_sentence_embedding_dimension(), out_features=128)
    normalize_one = Normalize()
    normalize_two = Normalize()
    router = Router(
        {
            "query": [static_embedding_model_one, dense, normalize_one],
            "document": [static_embedding_model_two, normalize_two],
        }
    )
    model = SentenceTransformer(modules=[router])

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model")
        model.save(model_path)

        loaded_model = SentenceTransformer(model_path)
        loaded_router = loaded_model[0]

        # Verify structure
        assert len(loaded_router.sub_modules["query"]) == 3
        assert len(loaded_router.sub_modules["document"]) == 2

        # The first route has priority here, but usually all routes have the same embedding dimension
        # as they can't be compared otherwise
        assert loaded_model.get_sentence_embedding_dimension() == 128

        # If we swap the order of the routes, the new first route should be used
        loaded_router.sub_modules = {
            "document": loaded_router.sub_modules["document"],
            "query": loaded_router.sub_modules["query"],
        }
        assert loaded_model.get_sentence_embedding_dimension() == 768


def test_router_with_trainer(static_embedding_model):
    """Test Router works correctly with a training setup using router_mapping."""

    # Create a Router with StaticEmbedding modules
    router = Router({"query": [static_embedding_model], "document": [static_embedding_model]}, allow_empty_key=False)
    model = SentenceTransformer(modules=[router])

    tracking_dict = TaskTypesTrackingDict(router.sub_modules)
    router.sub_modules = tracking_dict

    train_dataset = Dataset.from_dict(
        {
            "question": ["What is the capital of France?", "What is the largest ocean?"],
            "answer": ["The capital of France is Paris.", "The largest ocean is the Pacific Ocean."],
        }
    )

    # Setup router mapping for training
    router_mapping = {"question": "query", "answer": "document"}

    # Create a loss function that works with router
    loss = losses.MultipleNegativesRankingLoss(model=model)

    args = SentenceTransformerTrainingArguments(
        output_dir=tempfile.mkdtemp(),
        router_mapping=router_mapping,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        loss=loss,
        args=args,
    )
    tracking_dict.task_types.clear()  # Clear tracking before training
    trainer.train()

    # Once for tokenizing, once for forward
    assert tracking_dict.task_types == ["query", "document"] * 6


def test_router_module_forward_kwargs():
    """Test that Router's forward method passes kwargs correctly to sub-modules."""

    class ExampleModuleWithForwardKwargsOne(InputModule):
        forward_kwargs = {"one"}

        def __init__(self):
            super().__init__()
            self.kwargs_tracker = set()

        def forward(self, features, **kwargs):
            # Just return the features for testing
            for key in kwargs.keys():
                self.kwargs_tracker.add(key)
            features["sentence_embedding"] = features.get("sentence_embedding", torch.rand(1, 768))
            return features

        def tokenize(self, texts, **kwargs):
            return {}

        def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
            pass

    class ExampleModuleWithForwardKwargsTwo(ExampleModuleWithForwardKwargsOne):
        forward_kwargs = {"two", "task_type"}

    class ExampleModuleWithForwardKwargsThree(ExampleModuleWithForwardKwargsOne):
        forward_kwargs = {"three_a", "three_b"}

    module_one = ExampleModuleWithForwardKwargsOne()
    module_two = ExampleModuleWithForwardKwargsTwo()
    module_three = ExampleModuleWithForwardKwargsThree()

    router = Router({"query": [module_one], "document": [module_two, module_three]}, allow_empty_key=False)
    model = SentenceTransformer(modules=[router])

    model.encode(
        "Test input",
        task_type="query",
        one="value_one",
        two="value_two",
        three_a="value_three_a",
        three_b="value_three_b",
    )

    assert module_one.kwargs_tracker == {"one"}
    assert module_two.kwargs_tracker == set()
    assert module_three.kwargs_tracker == set()
    module_one.kwargs_tracker.clear()
    module_two.kwargs_tracker.clear()
    module_three.kwargs_tracker.clear()

    model.encode(
        "Test input",
        task_type="document",
        one="value_one",
        two="value_two",
        three_a="value_three_a",
        three_b="value_three_b",
    )

    assert module_one.kwargs_tracker == set()
    assert module_two.kwargs_tracker == {"two", "task_type"}
    assert module_three.kwargs_tracker == {"three_a", "three_b"}
    module_one.kwargs_tracker.clear()
    module_two.kwargs_tracker.clear()
    module_three.kwargs_tracker.clear()

    model.encode("Test input", task_type="query", three_a="value_three_a")
    assert module_one.kwargs_tracker == set()
    assert module_two.kwargs_tracker == set()
    assert module_three.kwargs_tracker == set()
    module_one.kwargs_tracker.clear()
    module_two.kwargs_tracker.clear()
    module_three.kwargs_tracker.clear()

    model.encode("Test input", task_type="document")
    assert module_one.kwargs_tracker == set()
    assert module_two.kwargs_tracker == {"task_type"}
    assert module_three.kwargs_tracker == set()
    module_one.kwargs_tracker.clear()
    module_two.kwargs_tracker.clear()
    module_three.kwargs_tracker.clear()
