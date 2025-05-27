from __future__ import annotations

import re

import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Asym, Router
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

    task_types = []

    # Create a custom dict subclass to track access
    class TrackingDict(dict):
        def get(self, key, default=None):
            task_types.append(key)
            return super().get(key, default)

        def __getitem__(self, key):
            task_types.append(key)
            return super().__getitem__(key)

    # Replace the dictionary with our tracking version
    router.sub_modules = TrackingDict(router.sub_modules)

    model = SentenceTransformer(modules=[router])

    # Test encoding
    query_texts = ["What is the capital of France?"]
    doc_texts = ["The capital of France is Paris."]

    model.encode_query(query_texts)
    assert "query" in task_types
    task_types = []

    model.encode_document(doc_texts)
    assert "document" in task_types
    task_types = []

    # The default route should be used if no task type is specified
    model.encode(query_texts)
    assert router.default_route == "query"
    assert "query" in task_types
    task_types = []

    # Test with a different default route
    router.default_route = "document"
    model.encode(doc_texts)
    assert "document" in task_types

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

    task_types = []

    # Create a custom dict subclass to track access
    class TrackingDict(dict):
        def get(self, key, default=None):
            task_types.append(key)
            return super().get(key, default)

        def __getitem__(self, key):
            task_types.append(key)
            return super().__getitem__(key)

    # Replace the dictionary with our tracking version
    asym_model.sub_modules = TrackingDict(asym_model.sub_modules)

    model = SentenceTransformer(modules=[asym_model])
    model.encode([{"query": "What is the capital of France?"}, {"query": "The capital of France is Paris."}])
    assert task_types == ["query", "query"]
    task_types = []

    model.encode([{"document": "What is the capital of France?"}, {"document": "The capital of France is Paris."}])
    assert task_types == ["document", "document"]
    task_types = []
