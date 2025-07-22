from __future__ import annotations

import importlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.models import Asym, Dense, Normalize, Router
from sentence_transformers.models.InputModule import InputModule
from sentence_transformers.models.StaticEmbedding import StaticEmbedding
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset
else:
    pytest.skip("The datasets library is not available.", allow_module_level=True)


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


class InvertMockModule(MockModule):
    def forward(self, features):
        features["sentence_embedding"] = -features["sentence_embedding"]
        return features


# Create a custom ModuleDict subclass to track access
class TaskTypesTrackingModuleDict(nn.ModuleDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = []

    def get(self, key, default=None):
        self.tasks.append(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.tasks.append(key)
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

    assert isinstance(router.sub_modules, nn.ModuleDict)
    assert list(router.sub_modules.keys()) == ["query", "document"]
    assert isinstance(router.sub_modules["query"], nn.Sequential)
    assert router.sub_modules["query"][0] == query_module
    assert isinstance(router.sub_modules["document"], nn.Sequential)
    assert router.sub_modules["document"][0] == doc_module
    assert router.default_route == "query"  # First key with allow_empty_key=True

    router = Router(
        {
            "document": [doc_module],
            "query": [query_module],
        }
    )

    assert isinstance(router.sub_modules, nn.ModuleDict)
    assert list(router.sub_modules.keys()) == ["document", "query"]
    assert isinstance(router.sub_modules["document"], nn.Sequential)
    assert router.sub_modules["document"][0] == doc_module
    assert isinstance(router.sub_modules["query"], nn.Sequential)
    assert router.sub_modules["query"][0] == query_module
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

    assert list(router.sub_modules["query"].children()) == [module1, module2]
    assert list(router.sub_modules["document"].children()) == [module3]


def test_router_encode(static_embedding_model):
    """Test encoding with Router."""
    # Create a Router with StaticEmbedding modules
    router = Router({"query": [static_embedding_model], "document": [static_embedding_model]})

    # Replace the dictionary with our tracking version
    tracking_dict = TaskTypesTrackingModuleDict(router.sub_modules)
    router.sub_modules = tracking_dict

    model = SentenceTransformer(modules=[router])

    # Test encoding
    query_texts = ["What is the capital of France?"]
    doc_texts = ["The capital of France is Paris."]

    model.encode_query(query_texts)
    assert "query" in tracking_dict.tasks
    tracking_dict.tasks = []

    model.encode_document(doc_texts)
    assert "document" in tracking_dict.tasks
    tracking_dict.tasks = []

    # The default route should be used if no task type is specified
    model.encode(query_texts)
    assert router.default_route == "query"
    assert "query" in tracking_dict.tasks
    tracking_dict.tasks = []

    # Test with a different default route
    router.default_route = "document"
    model.encode(doc_texts)
    assert "document" in tracking_dict.tasks

    # Test with an incorrect route
    with pytest.raises(
        ValueError, match=re.escape("No route found for task type 'invalid'. Available routes: ['query', 'document']")
    ):
        model.encode("This should fail", task="invalid")

    router.default_route = None  # Reset default route to None
    with pytest.raises(
        ValueError,
        match=re.escape(
            "You must provide a `task` argument when calling this method, "
            "or set a default route in the `Router` module."
        ),
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
    tracking_dict = TaskTypesTrackingModuleDict(asym_model.sub_modules)
    asym_model.sub_modules = tracking_dict

    model = SentenceTransformer(modules=[asym_model])
    model.encode([{"query": "What is the capital of France?"}, {"query": "The capital of France is Paris."}])
    assert tracking_dict.tasks == ["query", "query"]
    tracking_dict.tasks = []

    model.encode([{"document": "What is the capital of France?"}, {"document": "The capital of France is Paris."}])
    assert tracking_dict.tasks == ["document", "document"]
    tracking_dict.tasks = []

    with pytest.raises(ValueError, match=r"You cannot pass a list of dictionaries with different task types\. .*"):
        model.encode(
            [
                {"document": "What is the capital of France?"},
                {"document": "The capital of France is Paris."},
                {"query": "This is a question?"},
            ]
        )


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


def test_router_save_load(static_embedding_model: StaticEmbedding, tmp_path: Path):
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
    model_path = os.path.join(tmp_path, "test_model")
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


def test_router_save_load_with_custom_default_route(static_embedding_model: StaticEmbedding, tmp_path: Path):
    """Test saving and loading a model with custom default route."""
    router = Router(
        {"query": [static_embedding_model], "document": [static_embedding_model]}, default_route="document"
    )
    model = SentenceTransformer(modules=[router])

    model_path = os.path.join(tmp_path, "test_model")
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


def test_router_save_load_without_default_route(static_embedding_model: StaticEmbedding, tmp_path: Path):
    """Test saving and loading a model without a default route."""
    router = Router({"query": [static_embedding_model], "document": [static_embedding_model]}, allow_empty_key=False)
    model = SentenceTransformer(modules=[router])

    model_path = os.path.join(tmp_path, "test_model")
    model.save(model_path)

    loaded_model = SentenceTransformer(model_path)
    loaded_router = loaded_model[0]
    # Verify default route is None
    assert loaded_router.default_route is None

    # Test that encoding without task raises error
    with pytest.raises(
        ValueError,
        match=re.escape(
            "You must provide a `task` argument when calling this method, "
            "or set a default route in the `Router` module."
        ),
    ):
        loaded_model.encode(["Test text"])


def test_router_save_load_with_multiple_modules_per_route(static_embedding_model: StaticEmbedding, tmp_path: Path):
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

    model_path = os.path.join(tmp_path, "test_model")
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
    loaded_router.sub_modules = nn.ModuleDict(
        {
            "document": loaded_router.sub_modules["document"],
            "query": loaded_router.sub_modules["query"],
        }
    )
    assert loaded_model.get_sentence_embedding_dimension() == 768


def test_router_with_trainer(static_embedding_model: StaticEmbedding, tmp_path: Path):
    """Test Router works correctly with a training setup using router_mapping."""

    # Create a Router with StaticEmbedding modules
    router = Router({"query": [static_embedding_model], "document": [static_embedding_model]}, allow_empty_key=False)
    model = SentenceTransformer(modules=[router])
    model.model_card_data.generate_widget_examples = False  # Disable widget examples generation for testing

    tracking_dict = TaskTypesTrackingModuleDict(router.sub_modules)
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
        output_dir=tmp_path,
        router_mapping=router_mapping,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        loss=loss,
        args=args,
    )
    tracking_dict.tasks.clear()  # Clear tracking before training
    trainer.train()

    # Once for tokenizing, once for forward
    assert tracking_dict.tasks == ["query", "document"] * 6


def test_router_with_trainer_without_router_mapping(static_embedding_model: StaticEmbedding, tmp_path: Path):
    """Test Router crashes with a useful ValueError when training without router_mapping."""

    # Create a Router with StaticEmbedding modules
    router = Router.for_query_document([static_embedding_model], [static_embedding_model], allow_empty_key=False)
    router.default_route = None  # Ensure no default route is set
    model = SentenceTransformer(modules=[router])

    train_dataset = Dataset.from_dict(
        {
            "question": ["What is the capital of France?", "What is the largest ocean?"],
            "answer": ["The capital of France is Paris.", "The largest ocean is the Pacific Ocean."],
        }
    )

    # Create a loss function that works with router
    loss = losses.MultipleNegativesRankingLoss(model=model)

    args = SentenceTransformerTrainingArguments(output_dir=tmp_path)

    with pytest.raises(
        ValueError,
        match="You are using a Router module in your model, but you did not provide a `router_mapping` in the training arguments. .*",
    ):
        SentenceTransformerTrainer(
            model=model,
            train_dataset=train_dataset,
            loss=loss,
            args=args,
        )


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
        forward_kwargs = {"two", "task"}

    class ExampleModuleWithForwardKwargsThree(ExampleModuleWithForwardKwargsOne):
        forward_kwargs = {"three_a", "three_b"}

    module_one = ExampleModuleWithForwardKwargsOne()
    module_two = ExampleModuleWithForwardKwargsTwo()
    module_three = ExampleModuleWithForwardKwargsThree()

    router = Router({"query": [module_one], "document": [module_two, module_three]}, allow_empty_key=False)
    model = SentenceTransformer(modules=[router])

    model.encode(
        "Test input",
        task="query",
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
        task="document",
        one="value_one",
        two="value_two",
        three_a="value_three_a",
        three_b="value_three_b",
    )

    assert module_one.kwargs_tracker == set()
    assert module_two.kwargs_tracker == {"two", "task"}
    assert module_three.kwargs_tracker == {"three_a", "three_b"}
    module_one.kwargs_tracker.clear()
    module_two.kwargs_tracker.clear()
    module_three.kwargs_tracker.clear()

    model.encode("Test input", task="query", three_a="value_three_a")
    assert module_one.kwargs_tracker == set()
    assert module_two.kwargs_tracker == set()
    assert module_three.kwargs_tracker == set()
    module_one.kwargs_tracker.clear()
    module_two.kwargs_tracker.clear()
    module_three.kwargs_tracker.clear()

    model.encode("Test input", task="document")
    assert module_one.kwargs_tracker == set()
    assert module_two.kwargs_tracker == {"task"}
    assert module_three.kwargs_tracker == set()
    module_one.kwargs_tracker.clear()
    module_two.kwargs_tracker.clear()
    module_three.kwargs_tracker.clear()


@pytest.mark.parametrize("legacy_config", [True, False])
@pytest.mark.parametrize("module_in_root", [True, False])
def test_router_load_with_config(
    legacy_config: bool, module_in_root: bool, static_embedding_model: StaticEmbedding, tmp_path: Path
):
    """Test that Router can be loaded from a saved directory with config file."""
    if module_in_root and legacy_config:
        pytest.skip("Cannot have both module in root and legacy config at the same time.")

    # Create and save a Router
    query_module = static_embedding_model
    doc_module = static_embedding_model

    router = Router({"query": [query_module], "document": [doc_module]}, default_route="query")
    model = SentenceTransformer(modules=[router])

    model.save_pretrained(tmp_path)
    assert router.config_file_name == "router_config.json"
    assert os.path.exists(os.path.join(tmp_path, "router_config.json"))

    if legacy_config:
        # Rename the config file to legacy name
        os.rename(os.path.join(tmp_path, "router_config.json"), os.path.join(tmp_path, "config.json"))

    if module_in_root:
        # Move the module to the root directory
        for file in os.listdir(os.path.join(tmp_path, "document_0_StaticEmbedding")):
            source_path = os.path.join(tmp_path, "document_0_StaticEmbedding", file)
            dest_path = os.path.join(tmp_path, file)
            if os.path.isfile(source_path):
                os.rename(source_path, dest_path)

        with open(os.path.join(tmp_path, "router_config.json")) as f:
            config = json.load(f)
        config["structure"]["document"] = [""]
        config["types"][""] = config["types"].pop("document_0_StaticEmbedding", "")
        with open(os.path.join(tmp_path, "router_config.json"), "w") as f:
            json.dump(config, f, indent=4)

    # Load the Router back
    loaded_model = SentenceTransformer(str(tmp_path))
    loaded_router = loaded_model[0]

    # Check that the loaded router has the same structure
    assert set(loaded_router.sub_modules.keys()) == set(router.sub_modules.keys())
    assert loaded_router.default_route == router.default_route


def test_router_as_middle_module(static_embedding_model: StaticEmbedding, tmp_path: Path):
    """Test SentenceTransformer with multiple modules including a Router."""

    # Create a Router with different module configurations for each route
    router = Router(
        {
            "query": [InvertMockModule()],  # Simple route with single module
            "document": [InvertMockModule(), InvertMockModule()],  # Route with two modules
        }
    )

    normalize = Normalize()

    # Create a SentenceTransformer with static_embedding followed by router
    model = SentenceTransformer(modules=[static_embedding_model, router, normalize])

    # Create tracking dicts to monitor module usage
    tracking_dict = TaskTypesTrackingModuleDict(router.sub_modules)
    router.sub_modules = tracking_dict

    # Test texts
    query_texts = ["What is the meaning of life?"]
    doc_texts = ["The meaning of life is 42."]

    # Test encode_query
    model.encode_query(query_texts)
    assert "query" in tracking_dict.tasks
    assert tracking_dict.tasks.count("query") == 1
    assert "document" not in tracking_dict.tasks
    tracking_dict.tasks.clear()

    # Test encode_document
    model.encode_document(doc_texts)
    assert "document" in tracking_dict.tasks
    assert tracking_dict.tasks.count("document") == 1
    assert "query" not in tracking_dict.tasks
    tracking_dict.tasks.clear()

    # Test that the model processes through all modules (static_embedding + router)
    # by checking the embedding dimensions match what we expect
    query_embedding = model.encode_query(query_texts)
    assert query_embedding.shape[1] == static_embedding_model.get_sentence_embedding_dimension()

    doc_embedding = model.encode_document(doc_texts)
    assert doc_embedding.shape[1] == static_embedding_model.get_sentence_embedding_dimension()

    # Test that default encode uses the default route (query)
    default_embedding = model.encode(query_texts)
    query_embedding_direct = model.encode_query(query_texts)
    assert (default_embedding == query_embedding_direct).all()

    # Test that using the same text for both query and document gives exactly opposite embeddings
    # because of the InvertMockModule applied once or twice
    query_embedding = model.encode_query(query_texts, convert_to_tensor=True)
    doc_embedding = model.encode_document(query_texts, convert_to_tensor=True)
    assert torch.equal(query_embedding, -doc_embedding)

    # Also test that we can save and load the model with the Router as a middle module
    test_texts = ["This is a test text for both query and document.", "Another test text for validation."]

    # Get original embeddings
    original_query_embedding = model.encode_query(test_texts)
    original_doc_embedding = model.encode_document(test_texts)

    # Save the model to a temporary directory
    model_path = os.path.join(tmp_path, "test_model")
    model.save(model_path)

    # Load the model
    loaded_model = SentenceTransformer(model_path)

    # Verify loaded model structure
    assert len(list(loaded_model.children())) == 3
    assert isinstance(loaded_model[1], Router)
    loaded_router = loaded_model[1]
    assert set(loaded_router.sub_modules.keys()) == {"query", "document"}

    # Get embeddings from loaded model
    loaded_query_embedding = loaded_model.encode_query(test_texts)
    loaded_doc_embedding = loaded_model.encode_document(test_texts)

    # Verify embeddings are the same
    assert (original_query_embedding == loaded_query_embedding).all()
    assert (original_doc_embedding == loaded_doc_embedding).all()

    # Verify that using the same text for both query and document still gives exactly opposite embeddings
    loaded_query_embedding = loaded_model.encode_query(test_texts, convert_to_tensor=True)
    loaded_doc_embedding = loaded_model.encode_document(test_texts, convert_to_tensor=True)
    assert torch.equal(loaded_query_embedding, -loaded_doc_embedding)
