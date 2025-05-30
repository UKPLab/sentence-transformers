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

                TODO

        Note:
            These models are not necessarily stronger than non-asymmetric models. Rudimentary experiments indicate
            that non-Asym models perform better in many cases.

        Args:
            sub_modules: Mapping of route keys to lists of modules. Each key corresponds to a specific task type,
                often "query" or "document", and the list contains the modules to be applied for that task type.
            default_route: The default route to use if no task type is specified. If None, an exception will be thrown
                if no task type is specified. If ``allow_empty_key`` is True, the first key in sub_modules will be used as
                the default route. Defaults to None.
            allow_empty_key: If True, allows the default route to be set to the first key in `sub_modules` if
                ``default_route`` is None. Defaults to True.
        """
        # TODO: What's a good default route? How about if we have query-document models?
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
                module = module_class.load(model_name_or_path, subfolder=model_id, **hub_kwargs, **kwargs)
            except TypeError:
                local_path = load_dir_path(model_name_or_path=model_name_or_path, subfolder=model_id, **hub_kwargs)
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
            # Multiple different max_seq_lengths, log warning and return max value
            logger.warning(f"Different max_seq_lengths detected: {max_seq_lengths}. Using the maximum value.")
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
