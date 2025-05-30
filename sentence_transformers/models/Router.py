from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from torch import Tensor, nn
from transformers.utils.logging import warning_once

from sentence_transformers.models.InputModule import InputModule
from sentence_transformers.models.Module import Module
from sentence_transformers.util import import_from_string, load_dir_path

logger = logging.getLogger(__name__)


class Router(InputModule, nn.Sequential):
    forward_kwargs = {"task_type"}
    config_keys: list[str] = ["default_route", "allow_empty_key"]
    config_file_name = "router_config.json"

    def __init__(
        self, sub_modules: dict[str, list[Module]], default_route: str | None = None, allow_empty_key: bool = True
    ):
        """
        This model allows to create asymmetric SentenceTransformer models that apply different modules depending on the specified route,
        such as "query" or "document". Especially useful for models that have different encoders for queries and documents.

        In the below example, ...

        Notably, the ``task_type`` argument of ``model.encode`` can be used to specify which route to use, and
        ``model.encode_query`` and ``model.encode_document`` are shorthands for using ``task_type="query"`` and
        ``task_type="document"``, respectively. These methods also optionally apply ``prompts`` specific to queries
        or documents.

        Example:
            ::

                from sentence_transformers import SentenceTransformer
                from sentence_transformers.models import Router, Normalize

                # Use a regular SentenceTransformer for the document embeddings, and a static embedding model for the query embeddings
                document_embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
                query_embedder = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
                router = Router.for_query_document(
                    query_modules=list(query_embedder.children()),
                    document_modules=list(document_embedder.children()),
                )
                normalize = Normalize()

                # Create an asymmetric model with different encoders for queries and documents
                model = SentenceTransformer(
                    modules=[router, normalize],
                )

                # ... requires more training to align the vector spaces

                # Use the query & document routes
                query_embedding = model.encode_query("What is the capital of France?")
                document_embedding = model.encode_document("Paris is the capital of France.")

            ::

                from sentence_transformers.models import Router
                from sentence_transformers.sparse_encoder import SparseEncoder
                from sentence_transformers.sparse_encoder.models import IDF, MLMTransformer, SpladePooling

                # Load an asymmetric model with different encoders for queries and documents
                doc_encoder = MLMTransformer("opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill")
                router = Router.for_query_document(
                    query_modules=[
                        IDF.from_json(
                            "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
                            tokenizer=doc_encoder.tokenizer,
                            frozen=True,
                        ),
                    ],
                    document_modules=[
                        doc_encoder,
                        SpladePooling(pooling_strategy="max", activation_function="log1p_relu"),
                    ],
                )

                model = SparseEncoder(modules=[router], similarity_fn_name="dot")

                query = "What's the weather in ny now?"
                document = "Currently New York is rainy."

                query_embed = model.encode_query(query)
                document_embed = model.encode_document(document)

                sim = model.similarity(query_embed, document_embed)
                print(f"Similarity: {sim}")

                # Visualize top tokens for each text
                top_k = 10
                print(f"Top tokens {top_k} for each text:")

                decoded_query = model.decode(query_embed, top_k=top_k)
                decoded_document = model.decode(document_embed)

                for i in range(min(top_k, len(decoded_query))):
                    query_token, query_score = decoded_query[i]
                    doc_score = next((score for token, score in decoded_document if token == query_token), 0)
                    if doc_score != 0:
                        print(f"Token: {query_token}, Query score: {query_score:.4f}, Document score: {doc_score:.4f}")

                '''
                Similarity: tensor([[11.1105]], device='cuda:0')
                Top tokens 10 for each text:
                Token: ny, Query score: 5.7729, Document score: 0.8049
                Token: weather, Query score: 4.5684, Document score: 0.9710
                Token: now, Query score: 3.5895, Document score: 0.4720
                Token: ?, Query score: 3.3313, Document score: 0.0286
                Token: what, Query score: 2.7699, Document score: 0.0787
                Token: in, Query score: 0.4989, Document score: 0.0417
                '''

        Note:
            These models are not necessarily stronger than non-asymmetric models. Rudimentary experiments indicate
            that non-Router models perform better in many cases.

        Args:
            sub_modules: Mapping of route keys to lists of modules. Each key corresponds to a specific task type,
                often "query" or "document", and the list contains the modules to be applied for that task type.
            default_route: The default route to use if no task type is specified. If None, an exception will be thrown
                if no task type is specified. If ``allow_empty_key`` is True, the first key in sub_modules will be used as
                the default route. Defaults to None.
            allow_empty_key: If True, allows the default route to be set to the first key in `sub_modules` if
                ``default_route`` is None. Defaults to True.
        """
        self.sub_modules = sub_modules
        if self.sub_modules is None or len(self.sub_modules) == 0:
            raise ValueError("The routes dictionary cannot be empty.")

        if default_route is not None and default_route not in sub_modules:
            raise ValueError(f"Default route '{default_route}' not found in route keys: {list(sub_modules.keys())}")

        # If allow_empty_key is True, we can set a default route to the first key in sub_modules.
        if allow_empty_key and default_route is None:
            default_route = next(iter(sub_modules.keys()))
        self.default_route = default_route
        self.allow_empty_key = allow_empty_key

        ordered_dict = OrderedDict()
        for name, models in sub_modules.items():
            if not isinstance(models, list):
                models = [models]

            for idx, model in enumerate(models):
                ordered_dict[f"{name}_{idx}_{type(model).__name__}"] = model

        super().__init__(ordered_dict)

    @classmethod
    def for_query_document(
        cls,
        query_modules: list[Module],
        document_modules: list[Module],
        default_route: str | None = None,
        allow_empty_key: bool = True,
    ) -> Self:
        """
        Creates a Router model specifically for query and document modules, allowing convenient usage via `model.encode_query`
        and `model.encode_document`.

        Args:
            query_modules: List of modules to be applied for the "query" task type.
            document_modules: List of modules to be applied for the "document" task type.
            default_route: The default route to use if no task type is specified. If None, an exception will be thrown
                if no task type is specified. If ``allow_empty_key`` is True, the first key in sub_modules will be used as
                the default route. Defaults to None.
            allow_empty_key: If True, allows the default route to be set to the first key in `sub_modules` if
                ``default_route`` is None. Defaults to True.

        Returns:
            Router: An instance of the Router model with the specified query and document modules.
        """
        return cls(
            sub_modules={"query": query_modules, "document": document_modules},
            default_route=default_route or "document",
            allow_empty_key=allow_empty_key,
        )

    def forward(self, features: dict[str, Tensor], task_type: str | None = None, **kwargs) -> dict[str, Tensor]:
        if task_type is None:
            task_type = features.get("task_type", self.default_route)
        if task_type is None:
            # TODO: Write a more useful error, e.g. specific to training/inference
            raise ValueError("``task_type`` must be specified, or the ``Router`` must have a ``default_route`` set.")

        if task_type not in self.sub_modules:
            raise ValueError(
                f"No route found for task type '{task_type}'. Available routes: {list(self.sub_modules.keys())}"
            )

        kwargs["task_type"] = task_type
        for module in self.sub_modules[task_type]:
            module_kwargs = {
                key: value
                for key, value in kwargs.items()
                if hasattr(module, "forward_kwargs") and key in module.forward_kwargs
            }
            features = module(features, **module_kwargs)
        return features

    def get_sentence_embedding_dimension(self) -> int:
        for sub_modules in self.sub_modules.values():
            for module in reversed(sub_modules):
                if hasattr(module, "get_sentence_embedding_dimension"):
                    return module.get_sentence_embedding_dimension()
        return None

    def save(self, output_path):
        model_lookup = {}
        model_types = {}
        model_structure = {}

        for name, models in self.sub_modules.items():
            model_structure[name] = []
            for module_idx, model in enumerate(models):
                model_id = f"{name}_{module_idx}_{type(model).__name__}"
                model_lookup[model_id] = model
                model_types[model_id] = f"{type(model).__module__}.{type(model).__name__}"
                model_structure[name].append(model_id)

        for model_id, model in model_lookup.items():
            model_path = os.path.join(output_path, str(model_id))
            os.makedirs(model_path, exist_ok=True)
            model.save(model_path)

        with open(os.path.join(output_path, self.config_file_name), "w", encoding="utf8") as fOut:
            json.dump(
                {
                    "types": model_types,
                    "structure": model_structure,
                    "parameters": self.get_config_dict(),
                },
                fOut,
                indent=4,
            )

    def tokenize(self, texts: list[str] | list[tuple[str, str]], task_type: str | None = None, **kwargs):
        """Tokenizes a text and maps tokens to token-ids"""
        if isinstance(texts[0], dict):
            # Extract the first key from the first dictionary, and remove dictionary structure
            task_type = next(iter(texts[0].keys())) if task_type is None else task_type
            texts = [text[task_type] for text in texts]

        if task_type is None:
            task_type = self.default_route
        if task_type is None:
            # TODO: Write a more useful error, e.g. specific to training/inference
            raise ValueError("``task_type`` must be specified, or the ``Router`` must have a ``default_route`` set.")
        if task_type not in self.sub_modules:
            raise ValueError(
                f"No route found for task type '{task_type}'. Available routes: {list(self.sub_modules.keys())}"
            )

        input_module = self.sub_modules[task_type][0]
        tokenized = input_module.tokenize(texts, **kwargs)
        tokenized["task_type"] = task_type
        return tokenized

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        hub_kwargs = {
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        # Try the official config file first, then fall back to the legacy config file
        config = cls.load_config(model_name_or_path=model_name_or_path, subfolder=subfolder, **hub_kwargs)
        if not config:
            config = cls.load_config(
                model_name_or_path=model_name_or_path, config_filename="config.json", subfolder=subfolder, **hub_kwargs
            )
        modules = {}
        for model_id, model_type in config["types"].items():
            module_class: Module = import_from_string(model_type)
            try:
                module = module_class.load(
                    model_name_or_path, subfolder=os.path.join(subfolder, model_id), **hub_kwargs, **kwargs
                )
            except TypeError:
                local_path = load_dir_path(
                    model_name_or_path=model_name_or_path, subfolder=os.path.join(subfolder, model_id), **hub_kwargs
                )
                module = module_class.load(local_path)
            modules[model_id] = module

        model_structure = {}
        for key_name, models_list in config["structure"].items():
            model_structure[key_name] = []
            for model_id in models_list:
                model_structure[key_name].append(modules[model_id])

        model = cls(model_structure, **config["parameters"])
        return model

    @property
    def tokenizer(self):
        # We might have multiple tokenizers, one for each route, but we can only return one here.
        for sub_modules in self.sub_modules.values():
            input_module: InputModule = sub_modules[0]
            if hasattr(input_module, "tokenizer") and input_module.tokenizer is not None:
                return input_module.tokenizer
        return None

    @property
    def max_seq_length(self) -> int:
        # Collect all unique max_seq_length values
        max_seq_lengths = set()
        for modules in self.sub_modules.values():
            input_module: InputModule = modules[0]
            if modules and hasattr(input_module, "max_seq_length"):
                max_seq_lengths.add(input_module.max_seq_length)

        if not max_seq_lengths:
            return None
        elif len(max_seq_lengths) == 1:
            # Only one unique max_seq_length
            return max_seq_lengths.pop()
        else:
            warning_once(f"Different max_seq_lengths detected: {max_seq_lengths}. Using the maximum value.")
            return max(max_seq_lengths)

    @max_seq_length.setter
    def max_seq_length(self, value) -> None:
        # Check which modules have max_seq_length
        has_max_seq_length_keys = []
        for key, models in self.sub_modules.items():
            if models and hasattr(models[0], "max_seq_length"):
                has_max_seq_length_keys.append(key)

        if len(has_max_seq_length_keys) == 0:
            logger.warning("No modules have a max_seq_length attribute to set.")
            return

        for key in has_max_seq_length_keys:
            input_module: InputModule = self.sub_modules[key][0]
            input_module.max_seq_length = value


Asym = Router

# TODO: Remove this before release/merging
if __name__ == "__main__":
    from sentence_transformers import models
    from sentence_transformers.sparse_encoder import SparseEncoder
    from sentence_transformers.sparse_encoder.models import IDF, MLMTransformer, SpladePooling

    doc_encoder = MLMTransformer("opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill")
    asym = models.Router(
        {
            "query": [
                IDF.from_json(
                    "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
                    tokenizer=doc_encoder.tokenizer,
                    frozen=True,
                )
            ],
            "doc": [
                doc_encoder,
                SpladePooling("max"),
            ],
        }
    )

    model = SparseEncoder(modules=[asym], similarity_fn_name="dot")
    # For inference, you can pass dictionaries with the Asym keys:
    res = model.encode(
        [
            {"doc": "Currently New York is rainy."},
            {"query": "What's the weather in ny now?"},
            {"doc": 'The definition of <3 is "Love".'},
        ]
    )
    model.encode(
        [
            "how long do you have to wait to apply for cerb?",
            "<3 what does this symbol mean?",
            'The definition of <3 is "Love".',
        ]
    )
    sim = model.similarity(res[0], res[1])
    print(f"Similarity: {sim}")
    query = "What's the weather in ny now?"
    document = "Currently New York is rainy."

    query_embed = model.encode([{"query": query}])
    document_embed = model.encode([{"doc": document}])

    sim = model.similarity(query_embed, document_embed)
    print(f"Similarity: {sim}")

    model.push_to_hub(
        "arthurbresnu/SparseEncodder_format_opensearch-neural-sparse-encoding-doc-v2-distill", private=True
    )
    model = SparseEncoder("arthurbresnu/SparseEncodder_format_opensearch-neural-sparse-encoding-doc-v2-distill")

    # For inference, you can pass dictionaries with the Asym keys:
    res = model.encode(
        [
            {"doc": "Currently New York is rainy."},
            {"query": "What's the weather in ny now?"},
            {"doc": 'The definition of <3 is "Love".'},
        ]
    )
    model.encode(
        [
            "how long do you have to wait to apply for cerb?",
            "<3 what does this symbol mean?",
            'The definition of <3 is "Love".',
        ]
    )
    sim = model.similarity(res[0], res[1])
    print(f"Similarity: {sim}")
    query = "What's the weather in ny now?"
    document = "Currently New York is rainy."

    query_embed = model.encode([{"query": query}])
    document_embed = model.encode([{"doc": document}])

    sim = model.similarity(query_embed, document_embed)
    print(f"Similarity: {sim}")

    doc_encoder = MLMTransformer("naver/efficient-splade-VI-BT-large-doc")
    query_encoder = MLMTransformer("naver/efficient-splade-VI-BT-large-query")

    asym = models.Router(
        {
            "query": [
                query_encoder,
                SpladePooling("max"),
            ],
            "doc": [
                doc_encoder,
                SpladePooling("max"),
            ],
        }
    )

    model = SparseEncoder(modules=[asym], similarity_fn_name="dot")

    query = "What's the weather in ny now?"
    document = "Currently New York is rainy."

    query_embed = model.encode([{"query": query}])
    document_embed = model.encode([{"doc": document}])

    sim = model.similarity(query_embed, document_embed)
    print(f"Similarity: {sim}")

    model.push_to_hub(
        "arthurbresnu/SparseEncodder_format_efficient-splade-VI-BT-large-doc-and-query",
        private=True,
    )

    model = SparseEncoder(
        "arthurbresnu/SparseEncodder_format_efficient-splade-VI-BT-large-doc-and-query",
    )

    query = "What's the weather in ny now?"
    document = "Currently New York is rainy."

    query_embed = model.encode([{"query": query}])
    document_embed = model.encode([{"doc": document}])

    sim = model.similarity(query_embed, document_embed)
    print(f"Similarity: {sim}")
