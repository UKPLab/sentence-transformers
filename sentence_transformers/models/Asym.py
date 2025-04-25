from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict

import torch
from torch import Tensor, nn

from sentence_transformers.util import import_from_string

logger = logging.getLogger(__name__)


class Asym(nn.Sequential):
    # save_in_root: bool = True # as for now we can't handle this, save for later as should be done

    def __init__(self, sub_modules: dict[str, list[nn.Module]], allow_empty_key: bool = True):
        """
        This model allows to create asymmetric SentenceTransformer models, that apply different models depending on the specified input key.

        In the below example, we create two different Dense models for 'query' and 'doc'. Text that is passed as {'query': 'My query'} will
        be passed along along the first Dense model, and text that will be passed as {'doc': 'My document'} will use the other Dense model.

        Note, that when you call encode(), that only inputs of the same type can be encoded. Mixed-Types cannot be encoded.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                # Load a SentenceTransformer model (pretrained or not), and add an Asym module
                model = SentenceTransformer("microsoft/mpnet-base")
                dim = model.get_sentence_embedding_dimension()
                asym_model = models.Asym({
                    'query': [models.Dense(dim, dim)],
                    'doc': [models.Dense(dim, dim)]
                })
                model.add_module("asym", asym_model)

                train_dataset = Dataset.from_dict({
                    "query": ["is toprol xl the same as metoprolol?", "are eyes always the same size?"],
                    "answer": ["Metoprolol succinate is also known by the brand name Toprol XL.", "The eyes are always the same size from birth to death."],
                })

                # This mapper turns normal texts into a dictionary mapping Asym keys to the text
                def mapper(sample):
                    return {
                        "question": {"query": sample["question"]},
                        "answer": {"doc": sample["answer"]},
                    }

                train_dataset = train_dataset.map(mapper)
                loss = losses.MultipleNegativesRankingLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()

                # For inference, you can pass dictionaries with the Asym keys:
                model.encode([
                    {'query': 'how long do you have to wait to apply for cerb?'},
                    {'query': '<3 what does this symbol mean?'},
                    {'doc': 'The definition of <3 is "Love".'}]
                )

        Note:
            These models are not necessarily stronger than non-asymmetric models. Rudimentary experiments indicate
            that non-Asym models perform better in most cases.

        Args:
            sub_modules: Dict in the format str -> List[models]. The
                models in the specified list will be applied for input
                marked with the respective key.
            allow_empty_key: If true, inputs without a key can be
                processed. If false, an exception will be thrown if no
                key is specified.
        """
        self.sub_modules = sub_modules
        self.allow_empty_key = allow_empty_key

        # Check if modules have tokenizers and validate them
        self._validate_tokenizers()

        # Check for device consistency
        self._validate_devices()

        ordered_dict = OrderedDict()
        for name, models in sub_modules.items():
            if not isinstance(models, list):
                models = [models]

            for idx, model in enumerate(models):
                # Use a more descriptive naming convention
                ordered_dict[f"{name}-{idx}_{type(model).__name__}"] = model
        super().__init__(ordered_dict)

    def _validate_tokenizers(self):
        """Validate that all modules with tokenizers have compatible tokenizers"""
        self.has_tokenizers = {}
        self.tokenizers = {}

        for key, models in self.sub_modules.items():
            if models and hasattr(models[0], "tokenizer") and models[0].tokenizer is not None:
                self.has_tokenizers[key] = True
                self.tokenizers[key] = models[0].tokenizer
            else:
                self.has_tokenizers[key] = False
                self.tokenizers[key] = None

        # Check if all modules with tokenizers have the same tokenizer
        if sum(self.has_tokenizers.values()) > 1:
            tokenizer_types = {
                key: type(tokenizer).__name__ for key, tokenizer in self.tokenizers.items() if tokenizer is not None
            }
            if len(set(tokenizer_types.values())) > 1:
                logger.warning(
                    f"Different tokenizer types detected across modules: {tokenizer_types}. "
                    "This may cause issues when processing mixed batches."
                )
                self.have_common_tokenizer = False
            else:
                self.have_common_tokenizer = True

    def _validate_devices(self):
        """Check if all modules are on the same device"""
        devices = {}
        for key, models in self.sub_modules.items():
            if models and hasattr(models[0], "device"):
                devices[key] = models[0].device

        if len(set(devices.values())) > 1:
            logger.warning(f"Modules are on different devices: {devices}. This may cause issues during processing.")

    def forward(self, features: dict[str, Tensor]):
        if "text_keys" in features and len(features["text_keys"]) > 0:
            # Group indices by text_key
            key_to_indices = {}
            for idx, key in enumerate(features["text_keys"]):
                if key not in key_to_indices:
                    key_to_indices[key] = []
                key_to_indices[key].append(idx)

            # Process each key group separately
            all_outputs = {}
            for text_key, indices in key_to_indices.items():
                # Extract batch for this text key
                batch_features = {}
                for k, v in features.items():
                    if isinstance(v, Tensor):
                        batch_features[k] = v[indices]
                    elif isinstance(v, list) and len(v) == len(features["text_keys"]):
                        batch_features[k] = [v[i] for i in indices]
                    else:
                        batch_features[k] = v

                batch_features["text_keys"] = [text_key] * len(indices)  # Ensure all items have same key

                # Apply models for this text key
                for model in self.sub_modules[text_key]:
                    batch_features = model(batch_features)

                # Store results
                for k, v in batch_features.items():
                    if k not in all_outputs:
                        all_outputs[k] = [None] * len(features["text_keys"])
                    for i, idx in enumerate(indices):
                        if isinstance(v, Tensor) and v.dim() > 0:
                            all_outputs[k][idx] = v[i] if i < v.size(0) else v[0]
                        elif isinstance(v, list) and len(v) == len(indices):
                            all_outputs[k][idx] = v[i]
                        else:
                            # Handle scalar tensors or other types
                            all_outputs[k][idx] = v
            # Reconstruct features with processed outputs
            for k, v in all_outputs.items():
                if all(x is not None for x in v):
                    if all(isinstance(x, Tensor) for x in v):
                        try:
                            features[k] = torch.stack(v)
                        except Exception:
                            # If tensors can't be stacked (different shapes), keep as list
                            features[k] = v
                    else:
                        features[k] = v

        elif not self.allow_empty_key:
            raise ValueError("Input did not specify any keys and allow_empty_key is False")

        return features

    def get_sentence_embedding_dimension(self) -> int:
        for name in self.sub_modules:
            if hasattr(self.sub_modules[name][0], "get_sentence_embedding_dimension"):
                return self.sub_modules[name][0].get_sentence_embedding_dimension()
        return None

    def save(self, output_path):
        model_lookup = {}
        model_types = {}
        model_structure = {}

        block_counts = {key: 0 for key in self.sub_modules.keys()}
        for name, models in self.sub_modules.items():
            model_structure[name] = []
            for model in models:
                # Use block count instead of random id
                model_id = f"{name}-{block_counts[name]}_{type(model).__name__}"
                model_lookup[model_id] = model
                model_types[model_id] = type(model).__module__
                model_structure[name].append(model_id)
                block_counts[name] += 1

        for model_id, model in model_lookup.items():
            model_path = os.path.join(output_path, str(model_id))
            os.makedirs(model_path, exist_ok=True)
            model.save(model_path)

        with open(os.path.join(output_path, "config.json"), "w", encoding="utf8") as fOut:
            json.dump(
                {
                    "types": model_types,
                    "structure": model_structure,
                    "parameters": {"allow_empty_key": self.allow_empty_key},
                },
                fOut,
                indent=2,
            )

    def tokenize(self, texts: list[str] | list[tuple[str, str]], **kwargs):
        """Tokenizes a text and maps tokens to token-ids"""
        if not isinstance(texts[0], dict):
            logger.warning(
                "Texts are not in the expected format. Expected a list of dictionaries with keys. "
                "Using 'doc' as the default key."
            )
            texts = [{"doc": text} for text in texts]

        module_key = None

        for lookup in texts:
            text_key, text = next(iter(lookup.items()))
            if module_key is None:
                module_key = text_key
            if text_key != module_key and not self.have_common_tokenizer:
                raise AssertionError(
                    f"Mixed batches are not allowed. Found different keys: {text_key} and {module_key}. "
                    "Please ensure all inputs have the same key."
                )
        return self.sub_modules[module_key][0].tokenize(texts, **kwargs)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        modules = {}
        for model_id, model_type in config["types"].items():
            module_class = import_from_string(model_type)
            module = module_class.load(os.path.join(input_path, model_id))
            modules[model_id] = module

        model_structure = {}
        for key_name, models_list in config["structure"].items():
            model_structure[key_name] = []
            for model_id in models_list:
                model_structure[key_name].append(modules[model_id])

        model = Asym(model_structure, **config["parameters"])
        return model

    @property
    def tokenizer(self):
        # Check if both modules have tokenizers
        has_tokenizer_keys = [key for key, has_tokenizer in self.has_tokenizers.items() if has_tokenizer]

        if len(has_tokenizer_keys) == 0:
            return None
        elif len(has_tokenizer_keys) == 1:
            # Only one module has a tokenizer, return it
            return self.tokenizers[has_tokenizer_keys[0]]
        else:
            # Both modules have tokenizers
            tokenizer_types = {
                key: type(tokenizer).__name__ for key, tokenizer in self.tokenizers.items() if tokenizer is not None
            }

            if len(set(tokenizer_types.values())) > 1:
                # Different tokenizer types, warn and return the first one
                logger.warning(
                    f"Different tokenizer types detected: {tokenizer_types}. Using the one of the first key by default."
                )

            # Return the first tokenizer
            return self.tokenizers[has_tokenizer_keys[0]]

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        # Set the tokenizer for all modules that have a tokenizer
        has_tokenizer_keys = [key for key, has_tokenizer in self.has_tokenizers.items() if has_tokenizer]

        if len(has_tokenizer_keys) == 0:
            logger.warning("No modules have a tokenizer attribute to set.")
            return

        for key in has_tokenizer_keys:
            if self.sub_modules[key]:
                self.sub_modules[key][0].tokenizer = value
                self.tokenizers[key] = value

    def get_max_seq_length(self) -> int | None:
        # Check which modules have max_seq_length
        has_max_seq_length_keys = []
        for key, models in self.sub_modules.items():
            if models and hasattr(models[0], "get_max_seq_length"):
                has_max_seq_length_keys.append(key)

        if len(has_max_seq_length_keys) == 0:
            return None
        elif len(has_max_seq_length_keys) == 1:
            # Only one module has max_seq_length, return it
            return self.sub_modules[has_max_seq_length_keys[0]][0].get_max_seq_length()
        else:
            # Both modules have max_seq_length
            max_seq_lengths = {key: self.sub_modules[key][0].get_max_seq_length() for key in has_max_seq_length_keys}

            if len(set(max_seq_lengths.values())) > 1:
                # Different max_seq_lengths, warn and return the first one
                logger.warning(
                    f"Different max_seq_lengths detected: {max_seq_lengths}. Using the one of the first key by default."
                )

            # Return the first max_seq_length
            return max_seq_lengths[has_max_seq_length_keys[0]]

    @property
    def max_seq_length(self) -> int:
        return self.get_max_seq_length()

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
            self.sub_modules[key][0].max_seq_length = value


if __name__ == "__main__":
    from datasets import Dataset

    from sentence_transformers.sparse_encoder import (
        SparseEncoder,
        SparseEncoderTrainer,
        SparseMultipleNegativesRankingLoss,
    )

    # doc_encoder = MLMTransformer("opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill")
    # asym = models.Asym(
    #     {
    #         "query": [
    #             IDF.from_json(
    #                 "runs/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill/idf.json",
    #                 tokenizer=doc_encoder.tokenizer,
    #                 frozen=True,
    #             )
    #         ],
    #         "doc": [
    #             doc_encoder,
    #             SpladePooling("max"),
    #         ],
    #     }
    # )

    # model = SparseEncoder(modules=[asym], similarity_fn_name="dot")
    # model.push_to_hub(
    #     "sparse-embedding/SparseEncodder_format_opensearch-neural-sparse-encoding-doc-v2-distill", private=True
    # )
    model = SparseEncoder("sparse-embedding/SparseEncodder_format_opensearch-neural-sparse-encoding-doc-v2-distill")

    # doc_encoder = MLMTransformer("naver/efficient-splade-VI-BT-large-doc")
    # query_encoder = MLMTransformer("naver/efficient-splade-VI-BT-large-query")

    # asym = models.Asym(
    #     {
    #         "query": [
    #             query_encoder,
    #             SpladePooling("max"),
    #         ],
    #         "doc": [
    #             doc_encoder,
    #             SpladePooling("max"),
    #         ],
    #     }
    # )

    # model = SparseEncoder(modules=[asym], similarity_fn_name="dot")

    # model.push_to_hub(
    #     "arthurbresnu/SparseEncodder_format_efficient-splade-VI-BT-large-doc-and-query",
    #     private=True,
    # )

    train_dataset = Dataset.from_dict(
        {
            "query": [
                "is toprol xl the same as metoprolol?",
                "are eyes always the same size?",
            ],
            "answer": [
                "Metoprolol succinate is also known by the brand name Toprol XL.",
                "The eyes are always the same size from birth to death.",
            ],
        }
    )

    # This mapper turns normal texts into a dictionary mapping Asym keys to the text
    def mapper(sample):
        return {
            "query": {"query": sample["query"]},
            "answer": {"doc": sample["answer"]},
        }

    train_dataset = train_dataset.map(mapper)
    loss = SparseMultipleNegativesRankingLoss(model)

    trainer = SparseEncoderTrainer(
        model=model,
        train_dataset=train_dataset,
        loss=loss,
    )
    # trainer.train()

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
    # -----------------------------------------------------------------------------------------------------

    # from datasets import Dataset

    # from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, models

    # # # Load a SentenceTransformer model (pretrained or not), and add an Asym module
    # model = SentenceTransformer("microsoft/mpnet-base")
    # dim = model.get_sentence_embedding_dimension()
    # asym_model = models.Asym({"query": [models.Dense(dim, dim)], "doc": [models.Dense(dim, dim)]})
    # model.add_module("asym", asym_model)

    # # asym = models.Asym(
    # #     {
    # #         "query": [
    # #             models.Transformer("microsoft/mpnet-base"),
    # #             models.Pooling("cls"),
    # #             models.Dense(dim, dim),
    # #         ],
    # #         "doc": [
    # #             models.Transformer("microsoft/mpnet-base"),
    # #             models.Pooling("cls"),
    # #             models.Dense(dim, dim),
    # #         ],
    # #     }
    # # )
    # # model = SentenceTransformer(modules=[asym])
    # # model.push_to_hub("sparse-embedding/ST_model_with_asym_first_in_root", private=True)
    # # model = SentenceTransformer("sparse-embedding/ST_model_with_asym_first_in_root")

    # train_dataset = Dataset.from_dict(
    #     {
    #         "query": ["is toprol xl the same as metoprolol?", "are eyes always the same size?"],
    #         "answer": [
    #             "Metoprolol succinate is also known by the brand name Toprol XL.",
    #             "The eyes are always the same size from birth to death.",
    #         ],
    #     }
    # )

    # # This mapper turns normal texts into a dictionary mapping Asym keys to the text
    # def mapper(sample):
    #     return {
    #         "query": {"query": sample["query"]},
    #         "answer": {"doc": sample["answer"]},
    #     }

    # train_dataset = train_dataset.map(mapper)
    # loss = losses.MultipleNegativesRankingLoss(model)

    # trainer = SentenceTransformerTrainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     loss=loss,
    # )
    # # trainer.train()

    # # For inference, you can pass dictionaries with the Asym keys:
    # model.encode(
    #     [
    #         {"query": "how long do you have to wait to apply for cerb?"},
    #         {"query": "<3 what does this symbol mean?"},
    #         {"doc": 'The definition of <3 is "Love".'},
    #     ]
    # )
