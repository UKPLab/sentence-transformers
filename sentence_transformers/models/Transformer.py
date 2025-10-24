from __future__ import annotations

import inspect
import logging
import os
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Union

import torch

from sentence_transformers.backend import load_onnx_model, load_openvino_model
from sentence_transformers.models.modality_utils import parse_inputs

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from tokenizers.normalizers import Lowercase, Sequence
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    BaseVideoProcessor,
    FeatureExtractionMixin,
    ImageProcessingMixin,
    MT5Config,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    T5Config,
    TimmWrapperConfig,
)
from transformers.utils import ModelOutput
from transformers.utils.import_utils import is_peft_available
from transformers.utils.peft_utils import find_adapter_config_file

from sentence_transformers.models.InputModule import InputModule

logger = logging.getLogger(__name__)

if TYPE_CHECKING and is_peft_available():
    from peft import PeftConfig


def _save_pretrained_wrapper(_save_pretrained_fn: Callable, subfolder: str) -> Callable[..., None]:
    def wrapper(save_directory: str | Path, **kwargs) -> None:
        os.makedirs(Path(save_directory) / subfolder, exist_ok=True)
        return _save_pretrained_fn(Path(save_directory) / subfolder, **kwargs)

    return wrapper


TRANSFORMER_TASK_TO_AUTO_MODEL = {
    "feature-extraction": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "text-generation": AutoModelForCausalLM,
    "fill-mask": AutoModelForMaskedLM,
}

# Maps transformer tasks -> modalities -> methods -> model output fields -> module feature names
# Structure: {task: {modality: {method_name: {model_output_field: module_feature_name}}}}
TASK_MODALITY_METHOD_CONFIG = {
    "feature-extraction": {
        "text": {
            "get_text_features": {None: "sentence_embedding"},
            "forward": {"last_hidden_state": "token_embeddings", "text_embeds": "sentence_embedding"},
        },
        "image": {
            "get_image_features": {None: "sentence_embedding"},
            "forward": {"last_hidden_state": "token_embeddings", "image_embeds": "sentence_embedding"},
        },
        "audio": {
            "get_audio_features": {None: "sentence_embedding"},
            "forward": {"last_hidden_state": "token_embeddings", "audio_embeds": "sentence_embedding"},
        },
        "video": {
            "get_video_features": {None: "sentence_embedding"},
            "forward": {"last_hidden_state": "token_embeddings", "video_embeds": "sentence_embedding"},
        },
        "multimodal": {"forward": {"last_hidden_state": "token_embeddings"}},
    },
    "sequence-classification": {
        "text": {"forward": {"logits": "scores"}},
        "image": {"forward": {"logits": "scores"}},
        "audio": {"forward": {"logits": "scores"}},
        "video": {"forward": {"logits": "scores"}},
        "multimodal": {"forward": {"logits": "scores"}},
    },
    "text-generation": {
        "text": {"forward": {"logits": "causal_logits"}},
        "image": {"forward": {"logits": "causal_logits"}},
        "audio": {"forward": {"logits": "causal_logits"}},
        "video": {"forward": {"logits": "causal_logits"}},
        "multimodal": {"forward": {"logits": "causal_logits"}},
    },
    "fill-mask": {
        "text": {"forward": {"logits": "token_embeddings"}},
        "image": {"forward": {"logits": "token_embeddings"}},
        "audio": {"forward": {"logits": "token_embeddings"}},
        "video": {"forward": {"logits": "token_embeddings"}},
        "multimodal": {"forward": {"logits": "token_embeddings"}},
    },
}


class Transformer(InputModule):
    """Hugging Face AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    Args:
        model_name_or_path: Hugging Face models name
            (https://huggingface.co/models)
        max_seq_length: Truncate any inputs longer than max_seq_length
        model_args: Keyword arguments passed to the Hugging Face
            Transformers model
        tokenizer_args: Keyword arguments passed to the Hugging Face
            Transformers tokenizer
        config_args: Keyword arguments passed to the Hugging Face
            Transformers config
        cache_dir: Cache dir for Hugging Face Transformers to store/load
            models
        do_lower_case: If true, lowercases the input (independent if the
            model is cased or not)
        tokenizer_name_or_path: Name or path of the tokenizer. When
            None, then model_name_or_path is used
        backend: Backend used for model inference. Can be `torch`, `onnx`,
            or `openvino`. Default is `torch`.
    """

    config_file_name: str = "sentence_bert_config.json"
    # TODO: Could we get rid of "max_seq_length" and "do_lower_case" here? Or are they not saved?
    config_keys: list[str] = [
        "transformer_task",
        "modality_config",
        "module_output_name",
    ]  # , "max_seq_length", "do_lower_case"]
    save_in_root: bool = True

    # TODO: Replace model_args with model_kwargs, perhaps replace tokenizer_args with processing_kwargs/processor_kwargs, config_args with config_kwargs?
    # TODO: Perhaps remove do_lower_case and put that in tokenizer_args?
    # TODO: Idem for max_seq_length?
    def __init__(
        self,
        model_name_or_path: str,
        transformer_task: Literal[
            "feature-extraction", "sequence-classification", "text-generation", "fill-mask"
        ] = "feature-extraction",
        max_seq_length: int | None = None,
        model_args: dict[str, Any] | None = None,
        tokenizer_args: dict[str, Any] | None = None,
        config_args: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        do_lower_case: bool = False,
        tokenizer_name_or_path: str | None = None,
        backend: str = "torch",
        modality_config: dict[str, dict[str, bool]] | None = None,
        module_output_name: str | None = None,
    ) -> None:
        """
        modalities:
            {
                "text": {
                    "method": "get_text_features",
                    "method_output_name": None,
                },
                "image": {
                    "method": "forward",
                    "method_output_name": "image_embeds",  # Can also be a tuple for nested dictionary keys like ("vision_outputs", "last_hidden_state"), e.g. for BLIP-2
                },
                ("text", "image"): {
                    "method": "forward",
                    "method_output_name": "last_hidden_state",
                },
            }
        """
        super().__init__()
        self.transformer_task = transformer_task
        if transformer_task not in TRANSFORMER_TASK_TO_AUTO_MODEL:
            raise ValueError(
                f"Unsupported transformer_task '{transformer_task}'. Supported tasks are: {list(TRANSFORMER_TASK_TO_AUTO_MODEL.keys())}"
            )
        # TODO: Reorder the args in __init__ body?
        self.do_lower_case = do_lower_case
        self.backend = backend
        if model_args is None:
            model_args = {}
        if tokenizer_args is None:
            tokenizer_args = {}
        if config_args is None:
            config_args = {}

        config, is_peft_model = self._load_config(model_name_or_path, cache_dir, backend, config_args)
        self.model = self._load_model(
            model_name_or_path, transformer_task, config, cache_dir, backend, is_peft_model, **model_args
        )

        # Get the signature of the auto_model's forward method to pass only the expected arguments from `features`,
        # plus some common values like "input_ids", "attention_mask", etc.
        # TODO: Cache (or only run) all signature calls like this
        model_forward_params = list(inspect.signature(self.model.forward).parameters)
        self.model_forward_params = set(model_forward_params) | {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "inputs_embeds",
        }

        if max_seq_length is not None and "model_max_length" not in tokenizer_args:
            tokenizer_args["model_max_length"] = max_seq_length
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path,
            cache_dir=cache_dir,
            **tokenizer_args,
        )

        # TODO: self.processor.is_fast might not work
        if do_lower_case:
            if self.processor.is_fast:

                def has_lowercase(normalizer):
                    if normalizer is None:
                        return False
                    if isinstance(normalizer, Lowercase):
                        return True
                    if isinstance(normalizer, Sequence):
                        return any(isinstance(n, Lowercase) for n in normalizer)
                    return False

                normalizer = self.processor.backend_tokenizer.normalizer
                if not has_lowercase(normalizer):
                    new_normalizers = [Lowercase()]
                    if isinstance(normalizer, Sequence):
                        new_normalizers += list(normalizer)
                    elif normalizer is not None:
                        new_normalizers.append(normalizer)
                    self.processor.backend_tokenizer.normalizer = Sequence(new_normalizers)
            else:
                self.processor.do_lower_case = do_lower_case

        # print("Has a chat template?", hasattr(self.processor, "chat_template"))
        print(f"Chat template length: {len(getattr(self.processor, 'chat_template', '') or '')}")

        # No max_seq_length set. Try to infer from model
        # TODO: self.processor.model_max_length might not work
        if max_seq_length is None:
            if (
                hasattr(self.model, "config")
                and hasattr(self.model.config, "max_position_embeddings")
                and hasattr(self.processor, "model_max_length")
            ):
                max_seq_length = min(self.model.config.max_position_embeddings, self.processor.model_max_length)

        self.max_seq_length = max_seq_length

        if modality_config is not None:
            self.modality_config = modality_config
            if module_output_name is None:
                raise ValueError(
                    "Loading the Transformer module with a custom modality_config requires also providing "
                    "module_output_name with the name of the output feature that this module should create, "
                    'for example "token_embeddings" or "sentence_embedding".'
                )
            self.module_output_name = module_output_name
            # TODO: Check if modality_config has the correct format
        else:
            self.modality_config, self.module_output_name = self.infer_modalities(self.model, self.processor)
        logger.info(f"Inferred modalities: {self.modality_config}")

        # TODO: Do we need this? Perhaps even remove tokenizer_name_or_path?
        if tokenizer_name_or_path is not None:
            self.model.config.tokenizer_class = self.processor.__class__.__name__

    @property
    def auto_model(self) -> PreTrainedModel:
        return self.model

    @property
    def modalities(self) -> list[str]:
        return list(self.modality_config.keys())

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if isinstance(self.processor, PreTrainedTokenizerBase):
            return self.processor
        return self.processor.tokenizer

    def _load_config(
        self, model_name_or_path: str, cache_dir: str | None, backend: str, config_args: dict[str, Any]
    ) -> tuple[PeftConfig | PretrainedConfig, bool]:
        """Loads the transformers or PEFT configuration

        Args:
            model_name_or_path (str): The model name on Hugging Face (e.g. 'sentence-transformers/all-MiniLM-L6-v2')
                or the path to a local model directory.
            cache_dir (str | None): The cache directory to store the model configuration.
            backend (str): The backend used for model inference. Can be `torch`, `onnx`, or `openvino`.
            config_args (dict[str, Any]): Keyword arguments passed to the Hugging Face Transformers config.

        Returns:
            tuple[PretrainedConfig, bool]: The model configuration and a boolean indicating whether the model is a PEFT model.
        """
        if (
            find_adapter_config_file(
                model_name_or_path,
                cache_dir=cache_dir,
                token=config_args.get("token"),
                revision=config_args.get("revision"),
                local_files_only=config_args.get("local_files_only", False),
            )
            is not None
        ):
            if not is_peft_available():
                raise Exception(
                    "Loading a PEFT model requires installing the `peft` package. You can install it via `pip install peft`."
                )
            if backend != "torch":
                # TODO: Consider following these steps automatically so we can load PEFT models with other backends
                raise ValueError(
                    "PEFT models can currently only be loaded with the `torch` backend. "
                    'To use other backends, load the model with `backend="torch"`, call `model.transformers_model.merge_and_unload()`, '
                    "save that model with `model.save_pretrained()` and then load the model with the desired backend."
                )
            from peft import PeftConfig

            return PeftConfig.from_pretrained(model_name_or_path, **config_args, cache_dir=cache_dir), True

        return AutoConfig.from_pretrained(model_name_or_path, **config_args, cache_dir=cache_dir), False

    def _load_model(
        self,
        model_name_or_path: str,
        transformer_task: Literal["feature-extraction", "sequence-classification", "text-generation", "fill-mask"],
        config: PeftConfig | PretrainedConfig,
        cache_dir: str,
        backend: str,
        is_peft_model: bool,
        **model_args,
    ) -> None:
        """Loads the transformers or PEFT model into the `auto_model` attribute

        Args:
            model_name_or_path (str): The model name on Hugging Face (e.g. 'sentence-transformers/all-MiniLM-L6-v2')
                or the path to a local model directory.
            config ("PeftConfig" | PretrainedConfig): The model configuration.
            cache_dir (str | None): The cache directory to store the model configuration.
            backend (str): The backend used for model inference. Can be `torch`, `onnx`, or `openvino`.
            is_peft_model (bool): Whether the model is a PEFT model.
            model_args (dict[str, Any]): Keyword arguments passed to the Hugging Face Transformers model.
        """
        if backend == "torch":
            # When loading a PEFT model, we need to load the base model first,
            # but some model_args are only for the adapter
            if is_peft_model:
                for adapter_only_kwarg in ["revision"]:
                    model_args.pop(adapter_only_kwarg, None)

            if transformer_task == "feature-extraction":
                if isinstance(config, T5Config):
                    return self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
                elif isinstance(config, MT5Config):
                    return self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)

            # TODO: What if transformer_task is something else?
            model_cls = TRANSFORMER_TASK_TO_AUTO_MODEL.get(transformer_task, AutoModel)
            return model_cls.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, **model_args)
        elif backend == "onnx":
            return load_onnx_model(
                model_name_or_path=model_name_or_path,
                config=config,
                task_name=transformer_task,
                **model_args,
            )
        elif backend == "openvino":
            return load_openvino_model(
                model_name_or_path=model_name_or_path,
                config=config,
                task_name=transformer_task,
                **model_args,
            )
        else:
            raise ValueError(f"Unsupported backend '{backend}'. `backend` should be `torch`, `onnx`, or `openvino`.")

    def _load_t5_model(self, model_name_or_path: str, config: PretrainedConfig, cache_dir: str, **model_args) -> None:
        """Loads the encoder model from T5"""
        from transformers import T5EncoderModel

        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        return T5EncoderModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, **model_args)

    def _load_mt5_model(self, model_name_or_path: str, config: PretrainedConfig, cache_dir: str, **model_args) -> None:
        """Loads the encoder model from T5"""
        from transformers import MT5EncoderModel

        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        return MT5EncoderModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, **model_args)

    def _find_valid_method_and_output(
        self,
        model: PreTrainedModel,
        method_to_output_mapping: dict[str, dict[str | None, str]],
        output_field_extractor: Callable,
        modality_name: str | tuple[str, ...],
        exclude_methods: set[str] | None = None,
    ) -> tuple[dict[str, dict[str, str]], str] | None:
        """
        Find a valid method and output configuration for a modality.

        Iterates through the provided methods and their expected outputs to find a valid
        combination that exists on the model, and constructs the modality configuration.

        Args:
            model (PreTrainedModel): The Hugging Face transformers model to check.
            method_to_output_mapping (dict): Dictionary mapping method names to their expected
                output fields and corresponding module output names.
                Format: {method_name: {method_output_name: module_output_name}}
            output_field_extractor (Callable): Function to extract the output field names from a method.
            modality_name (str | tuple[str, ...]): The modality key(s) to use in the returned
                modality configuration.
            exclude_methods (set[str] | None): Set of method names to skip during iteration.

        Returns:
            tuple[dict[str, dict[str, str]], str] | None: A tuple of (modality_config,
                module_output_name) if a valid configuration is found, otherwise None.
                The modality_config maps the modality_name to a dict with 'method' and
                'method_output_name' keys.
        """
        exclude_methods = exclude_methods or set()

        for method_name, output_mapping in method_to_output_mapping.items():
            if method_name in exclude_methods:
                continue

            if not hasattr(model, method_name):
                continue

            try:
                available_output_fields = output_field_extractor(getattr(model, method_name))
            except Exception:
                continue

            for method_output_name, module_output_name in output_mapping.items():
                if method_output_name is None or method_output_name in available_output_fields:
                    modality_config = {
                        modality_name: {
                            "method": method_name,
                            "method_output_name": method_output_name,
                        }
                    }
                    return modality_config, module_output_name
                else:
                    logger.debug(
                        f"Method '{method_name}' output '{method_output_name}' not found in fields {available_output_fields} for modality {modality_name}"
                    )

        return None

    def _handle_special_model_cases(self, model: PreTrainedModel) -> tuple[dict[str, dict[str, str]], str] | None:
        """Handle special cases for specific model architectures.

        Args:
            model (PreTrainedModel): The Hugging Face transformers model.

        Returns:
            tuple[dict[str, dict[str, str]], str] | None: Modality config and module output name if
                this is a special case model, otherwise None.
        """
        if not (hasattr(model, "config") and hasattr(model.config, "model_type")):
            return None

        model_type = model.config.model_type.lower()

        # Registry of special model types and their configurations
        special_cases = {
            "deepseek_vl": (
                {
                    "message": {
                        "method": "forward",
                        "method_output_name": "last_hidden_state",
                    }
                },
                "token_embeddings",
            ),
        }

        if model_type in special_cases:
            return special_cases[model_type]

        return None

    def _get_method_output_fields(self, method: Callable) -> list[str] | None:
        """Extract the output field names from a method's return type annotation.

        Args:
            method (Callable): The method to inspect.

        Returns:
            list[str] | None: List of output field names, or None if they cannot be determined.
        """

        def find_model_output_class(type_annotation):
            if hasattr(type_annotation, "__origin__") and type_annotation.__origin__ is Union:
                for arg in type_annotation.__args__:
                    result = find_model_output_class(arg)
                    if result is not None:
                        return result
            elif isinstance(type_annotation, type) and issubclass(type_annotation, ModelOutput):
                return type_annotation
            return None

        return_annotation = inspect.signature(method).return_annotation
        output_class = find_model_output_class(return_annotation)
        if output_class is None:
            return None
        return [field.name for field in fields(output_class)]

    def _infer_single_modality(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin
        | PreTrainedTokenizerBase
        | FeatureExtractionMixin
        | BaseVideoProcessor
        | ImageProcessingMixin,
    ) -> tuple[dict[str, dict[str, str]], str] | None:
        """Infer modality configuration for single-modality processors.

        Args:
            model (PreTrainedModel): The Hugging Face transformers model.
            processor: The processor to check.

        Returns:
            tuple[dict[str, dict[str, str]], str] | None: Modality config and module output name if
                a single modality is detected, otherwise None.
        """
        task_modality_config = TASK_MODALITY_METHOD_CONFIG[self.transformer_task]

        # Check modalities in order, with video before image since BaseVideoProcessor subclasses ImageProcessingMixin
        modality_checks = [
            ("text", PreTrainedTokenizerBase),
            ("audio", FeatureExtractionMixin),
            ("video", BaseVideoProcessor),
            ("image", ImageProcessingMixin),
        ]

        for modality_name, processor_class in modality_checks:
            if not isinstance(processor, processor_class):
                continue

            method_to_output_mapping = task_modality_config.get(modality_name, {})
            result = self._find_valid_method_and_output(
                model, method_to_output_mapping, self._get_method_output_fields, modality_name
            )
            if result is not None:
                return result

        return None

    def _infer_multimodal(
        self, model: PreTrainedModel, processor: ProcessorMixin
    ) -> tuple[dict[str, dict[str, str]], str] | None:
        """Infer modality configuration for multi-modal processors.

        Args:
            model (PreTrainedModel): The Hugging Face transformers model.
            processor (ProcessorMixin): The multi-modal processor.

        Returns:
            tuple[dict[str, dict[str, str]], str] | None: Modality config and module output name if
                modalities are detected, otherwise None.
        """
        if not isinstance(processor, ProcessorMixin):
            return None

        task_modality_config = TASK_MODALITY_METHOD_CONFIG[self.transformer_task]

        modality_config = {}
        module_output_name = None
        detected_modalities = []

        # Check which modality processors are available
        processor_attribute_mapping = [
            ("tokenizer", "text"),
            ("image_processor", "image"),
            ("feature_extractor", "audio"),
            ("video_processor", "video"),
        ]

        for processor_attribute, modality_name in processor_attribute_mapping:
            if processor_attribute not in processor.attributes:
                continue

            detected_modalities.append(modality_name)

            # Try to find single-modality methods (excluding 'forward' which likely needs all modalities)
            method_to_output_mapping = task_modality_config.get(modality_name, {})
            result = self._find_valid_method_and_output(
                model,
                method_to_output_mapping,
                self._get_method_output_fields,
                modality_name,
                exclude_methods={"forward"},
            )
            if result is not None:
                single_modality_config, module_output_name = result
                modality_config.update(single_modality_config)

        if not detected_modalities:
            return None

        # Check if there's a method that handles all modalities together
        method_to_output_mapping = task_modality_config.get("multimodal", {})
        result = self._find_valid_method_and_output(
            model, method_to_output_mapping, self._get_method_output_fields, tuple(detected_modalities)
        )
        if result is not None:
            # Override single-modality configs with the multimodal method
            # This is because multimodal methods often use different output names (e.g., pooled vs non-pooled)
            modality_config, module_output_name = result

            # If the processor has a chat template, add message modality with same configuration
            if processor.chat_template:
                modality_config["message"] = modality_config[tuple(detected_modalities)]

        if modality_config and module_output_name:
            return modality_config, module_output_name

        return None

    def infer_modalities(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin
        | PreTrainedTokenizerBase
        | FeatureExtractionMixin
        | BaseVideoProcessor
        | ImageProcessingMixin,
    ) -> tuple[dict[str, dict[str, str]], str]:
        """Infers the modalities supported by the model based on its architecture and processor.

        This method attempts to automatically detect what input modalities (text, image, audio, video)
        the model supports and how to invoke the model for each modality.

        Args:
            model (PreTrainedModel): The Hugging Face transformers model.
            processor: The processor (tokenizer, image processor, etc.) associated with the model.

        Returns:
            tuple[dict[str, dict[str, str]], str]: A tuple of (modality_config, module_output_name).
                The modality_config maps modality keys to dicts with 'method' and 'method_output_name'.
                The module_output_name is the name of the output feature this module creates.

        Raises:
            ValueError: If modalities cannot be inferred from the processor or model.
        """

        # Check for special model cases
        result = self._handle_special_model_cases(model)
        if result is not None:
            return result

        # Check for a single-modality model/processor
        result = self._infer_single_modality(model, processor)
        if result is not None:
            modality_config, module_output_name = result
            return modality_config, module_output_name

        # Check for a multi-modal model/processor
        result = self._infer_multimodal(model, processor)
        if result is not None:
            modality_config, module_output_name = result
            return modality_config, module_output_name

        error_msg = (
            f"Could not infer modalities from the processor (type: {type(processor).__name__}) or model. "
            f"Processor attributes: {getattr(processor, 'attributes', 'N/A')}. "
            "Please provide a custom modality_config and module_output_name when initializing the Transformer."
        )
        raise ValueError(error_msg)

    def __repr__(self) -> str:
        return f"Transformer({dict(self.get_config_dict(), architecture=self.model.__class__.__name__)})"

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """
        Forward pass through the transformer model.

        This method processes the input features through the underlying transformers model
        and returns the token embeddings along with any other relevant outputs.

        Notes:
            - Only passes arguments that are expected by the underlying transformer model

        Args:
            features (dict[str, torch.Tensor]): Input features dictionary containing at least
                'input_ids' and 'attention_mask'. May also contain other tensors required by
                the underlying transformer model.
            **kwargs: Additional keyword arguments to pass to the underlying transformer model.

        Returns:
            dict[str, torch.Tensor]: Updated features dictionary containing the input features, plus:
                - 'token_embeddings': Token-level embeddings from the transformer model
                - 'attention_mask': Possibly modified attention mask if using PeftModel with prompt learning
                - 'all_layer_embeddings': If the model outputs hidden states, contains embeddings from all layers
        """

        # TODO: Should we pass along the modality in 'features'?
        modality_name = features["modality"]
        modality_params = self.modality_config[modality_name]
        # TODO: Allow 'method' to be a tuple of methods to execute sequentially? A bit messy with the kwargs though
        method_name = modality_params["method"]
        method_output_name = modality_params["method_output_name"]
        if isinstance(method_output_name, str):
            method_output_name = (method_output_name,)

        # TODO: Does this prioritize features or kwargs?
        all_kwargs = {**features, **kwargs, "return_dict": True}
        model_method = getattr(self.model, method_name, None)
        if model_method is None:
            raise ValueError(f"Model does not have the requested '{method_name}' method")

        if method_name == "forward":
            filtered_kwargs = {key: value for key, value in all_kwargs.items() if key in self.model_forward_params}
        else:
            signature = inspect.signature(model_method)
            filtered_kwargs = {key: value for key, value in all_kwargs.items() if key in signature.parameters}

        # TODO: I (re)moved return_dict=True, and I changed up **kwargs
        model_output = model_method(**filtered_kwargs)

        if method_output_name is None:
            embedding = model_output
        else:
            embedding = model_output
            for output_key in method_output_name:
                embedding = embedding[output_key]

        if embedding.ndim == 4:
            # Some image models return (batch_size, num_channels, height, width) instead of (batch_size, seq_len, hidden_size)
            # We flatten the height and width dimensions and transpose to get (batch_size, height*width, num_channels)
            # which a subsequent Pooling layer can handle to remove the height*width dimension
            embedding = embedding.flatten(2).transpose(1, 2)

        features[self.module_output_name] = embedding

        # If the AutoModel is wrapped with a PeftModel(ForFeatureExtraction), then it may have added virtual tokens
        # We need to extend the attention mask to include these virtual tokens, or the pooling will fail
        if "input_ids" in features and "attention_mask" in features and is_peft_available():
            from peft import PeftModel

            if isinstance(self.model, PeftModel) and self.model.active_peft_config.is_prompt_learning:
                batch_size = features["input_ids"].shape[0]
                attention_mask = features["attention_mask"]
                prefix_attention_mask = torch.ones(
                    batch_size, self.model.active_peft_config.num_virtual_tokens, device=attention_mask.device
                )
                features["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        # TODO: Check if this is still viable
        if (
            hasattr(self.model.config, "output_hidden_states")
            and self.model.config.output_hidden_states
            and "hidden_states" in model_output
        ):
            features["all_layer_embeddings"] = model_output["hidden_states"]

        return features

    def get_word_embedding_dimension(self) -> int:
        """Get the output embedding dimension from the transformer model.

        Returns:
            int: The hidden dimension size of the model's embeddings.

        Raises:
            ValueError: If the embedding dimension cannot be determined from the model config.
        """
        # Edge case for timm models
        if isinstance(self.model.config, TimmWrapperConfig):
            return self.model.config.num_features

        # Get text config, e.g. for multi-modal models
        try:
            text_config = self.model.config.get_text_config()
        except AttributeError:
            text_config = self.model.config

        if hasattr(text_config, "hidden_size"):
            return text_config.hidden_size

        # Try hidden_sizes list (e.g., ResNet, some vision models)
        if hasattr(text_config, "hidden_sizes"):
            if isinstance(text_config.hidden_sizes, list):
                return text_config.hidden_sizes[-1]  # Use final layer dimension
            return text_config.hidden_sizes

        # Unable to determine dimension
        raise ValueError(
            f"Could not determine embedding dimension from model config. "
            f"Config type: {type(text_config).__name__}. "
            f"Available attributes: {[attr for attr in dir(text_config) if 'hidden' in attr.lower() or 'size' in attr.lower() or 'dim' in attr.lower()]}. "
            f"Please report this issue with your model name: {self.model.config.model_type if hasattr(self.model.config, 'model_type') else 'unknown'}"
        )

    # TODO: Perhaps rename to 'preprocess' or 'process'?
    # TODO: Consider moving modality checking to SentenceTransformer, so you can make multiple towers for different modalities
    def tokenize(
        self,
        texts: list[str] | list[dict] | list[tuple[str, str]],
        padding: str | bool = True,
        modality: str | tuple[str, ...] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Tokenizes inputs and maps tokens to token-ids.

        Args:
            texts: List of inputs which can be:
                - str: Text inputs
                - dict: Dictionary with modality keys (text, image, audio, video) or chat messages
                - PIL.Image.Image: Image inputs
                - np.ndarray/torch.Tensor: Audio (1-2D) or video (3-5D) inputs
            padding: Padding strategy for tokenization
            modality: Optional modality to use. If not provided, will be inferred from inputs.

        Returns:
            Dictionary containing tokenized inputs with 'modality' key indicating the input type
        """
        # Configuration for different modality types
        common_kwargs = {"return_tensors": "pt"}
        modality_kwargs = {
            "text": {"padding": padding, "truncation": "longest_first"},
            "audio": {
                "padding": padding
            },  # Note: padding can be counterproductive for some audio models (e.g., Whisper)
            "image": {},
            "video": {},
        }

        # Parse inputs, throw error if multiple modalities are detected, and process the single modality
        modality, processor_inputs = parse_inputs(texts)

        if modality not in self.modality_config:
            raise ValueError(
                f"Modality '{modality}' is not supported by this model. "
                f"Supported modalities: {list(self.modality_config.keys())}"
            )

        # Tackle an edge case: audio sampling_rate must be passed via the modality_kwargs
        if "sampling_rate" in processor_inputs:
            modality_kwargs["audio"]["sampling_rate"] = processor_inputs.pop("sampling_rate")

        processor_output = self._call_processor(modality, processor_inputs, modality_kwargs, common_kwargs)
        processor_output["modality"] = modality

        return processor_output

    def _is_chat_format(self, inputs: list) -> bool:
        """Check if inputs are in chat message format (list of dicts with 'role' and 'content')."""
        if not inputs:
            return False

        # Check the first item (could recursively check for nested lists)
        first_item = inputs[0]
        if isinstance(first_item, (list, tuple)) and len(first_item) > 0:
            return self._is_chat_format(first_item)

        return isinstance(first_item, dict) and "role" in first_item and "content" in first_item

    def _process_chat_messages(
        self, messages: list, text_kwargs: dict, common_kwargs: dict
    ) -> dict[str, torch.Tensor]:
        """Process chat messages using the processor's chat template."""
        processor_output = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            **text_kwargs,
            **common_kwargs,
        )

        if "message" not in self.modality_config:
            raise ValueError(
                f"The model does not support 'message' modality, but the input looks like a chat message. "
                f"Supported modalities: {list(self.modality_config.keys())}"
            )

        processor_output["modality"] = "message"
        return processor_output

    def _call_processor(
        self,
        modality: str | tuple[str, ...],
        processor_inputs: dict[str, list],
        modality_kwargs: dict[str, dict],
        common_kwargs: dict,
    ) -> dict[str, torch.Tensor]:
        """Call the appropriate processor with the correct arguments.

        Args:
            modality: The modality or tuple of modalities being processed
            processor_inputs: Dictionary of processor argument names to lists of values
            modality_kwargs: Configuration kwargs for each modality type
            common_kwargs: Common kwargs to pass to all processor calls

        Returns:
            Processor output dictionary
        """
        # Handle chat/message format
        if modality == "message":
            return self._process_chat_messages(processor_inputs["message"], modality_kwargs["text"], common_kwargs)

        if isinstance(self.processor, ProcessorMixin):
            # Multi-modal processor: pass modality-specific kwargs
            return self.processor(
                **processor_inputs,
                text_kwargs=modality_kwargs["text"],
                audio_kwargs=modality_kwargs["audio"],
                common_kwargs=common_kwargs,
            )

        # Single-modality processor: determine type and call appropriately
        # Check in order: text, audio, video, image (video before image due to inheritance)
        processor_type_checks = [
            ("text", PreTrainedTokenizerBase, modality_kwargs["text"]),
            ("audio", FeatureExtractionMixin, modality_kwargs["audio"]),
            ("video", BaseVideoProcessor, modality_kwargs["video"]),
            ("image", ImageProcessingMixin, modality_kwargs["image"]),
        ]

        for modality_type, processor_class, type_kwargs in processor_type_checks:
            if not isinstance(self.processor, processor_class):
                continue

            # Combine type-specific and common kwargs
            call_kwargs = {**type_kwargs, **common_kwargs}

            # If the modality type is in the inputs, extract it as primary argument
            if modality_type in processor_inputs:
                primary_input = processor_inputs.pop(modality_type)
                return self.processor(primary_input, **processor_inputs, **call_kwargs)
            else:
                return self.processor(**processor_inputs, **call_kwargs)

        raise RuntimeError(
            f"Could not determine how to call processor of type {type(self.processor).__name__} "
            f"for modality '{modality}'"
        )

    def save(self, output_path: str, safe_serialization: bool = True, **kwargs) -> None:
        self.model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.processor.save_pretrained(output_path)
        self.save_config(output_path)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        # Loading arguments
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        # Module-specific arguments
        trust_remote_code: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: str = "torch",
        **kwargs,
    ) -> Self:
        init_kwargs = cls._load_init_kwargs(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )
        return cls(model_name_or_path=model_name_or_path, **init_kwargs)

    @classmethod
    def _load_init_kwargs(
        cls,
        model_name_or_path: str,
        # Loading arguments
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        # Module-specific arguments
        trust_remote_code: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: str = "torch",
        **kwargs,
    ) -> dict[str, Any]:
        config = cls.load_config(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "revision": revision,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }

        # 3rd priority: config file
        if "model_args" not in config:
            config["model_args"] = {}
        if "tokenizer_args" not in config:
            config["tokenizer_args"] = {}
        if "config_args" not in config:
            config["config_args"] = {}

        # 2nd priority: hub_kwargs
        config["model_args"].update(hub_kwargs)
        config["tokenizer_args"].update(hub_kwargs)
        config["config_args"].update(hub_kwargs)

        # 1st priority: kwargs passed to SentenceTransformer
        if model_kwargs:
            config["model_args"].update(model_kwargs)
        if tokenizer_kwargs:
            config["tokenizer_args"].update(tokenizer_kwargs)
        if config_kwargs:
            config["config_args"].update(config_kwargs)

        return {**config, "cache_dir": cache_folder, "backend": backend}

    @classmethod
    def load_config(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        config_filename: str | None = None,
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> dict[str, Any]:
        config_filenames = (
            [config_filename]
            if config_filename
            else [
                "sentence_bert_config.json",
                "sentence_roberta_config.json",
                "sentence_distilbert_config.json",
                "sentence_camembert_config.json",
                "sentence_albert_config.json",
                "sentence_xlm-roberta_config.json",
                "sentence_xlnet_config.json",
            ]
        )
        for config_filename in config_filenames:
            config = super().load_config(
                model_name_or_path=model_name_or_path,
                subfolder=subfolder,
                config_filename=config_filename,
                token=token,
                cache_folder=cache_folder,
                revision=revision,
                local_files_only=local_files_only,
            )
            if config:
                break

        # Don't allow configs to set trust_remote_code
        if "model_args" in config and "trust_remote_code" in config["model_args"]:
            config["model_args"].pop("trust_remote_code")
        if "tokenizer_args" in config and "trust_remote_code" in config["tokenizer_args"]:
            config["tokenizer_args"].pop("trust_remote_code")
        if "config_args" in config and "trust_remote_code" in config["config_args"]:
            config["config_args"].pop("trust_remote_code")
        return config
