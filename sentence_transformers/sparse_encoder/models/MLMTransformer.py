from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, PretrainedConfig
from transformers.utils.import_utils import is_peft_available
from transformers.utils.peft_utils import find_adapter_config_file

from sentence_transformers.models.InputModule import InputModule

if TYPE_CHECKING and is_peft_available():
    from peft import PeftConfig

logger = logging.getLogger(__name__)


class MLMTransformer(InputModule):
    """
    MLMTransformer adapts a Masked Language Model (MLM) for sparse encoding applications.

    This class extends the Transformer class to work specifically with models that have a
    MLM head (like BERT, RoBERTa, etc.) and is designed to be used with SpladePooling
    for creating SPLADE sparse representations.

    MLMTransformer accesses the MLM prediction head to get vocabulary logits for each token,
    which are later used by SpladePooling to create sparse lexical representations.

    Args:
        model_name_or_path: Hugging Face models name
            (https://huggingface.co/models)
        max_seq_length: Truncate any inputs longer than max_seq_length
        model_args: Keyword arguments passed to the Hugging Face
            MLMTransformers model
        tokenizer_args: Keyword arguments passed to the Hugging Face
            MLMTransformers tokenizer
        config_args: Keyword arguments passed to the Hugging Face
            MLMTransformers config
        cache_dir: Cache dir for Hugging Face MLMTransformers to store/load
            models
        do_lower_case: If true, lowercases the input (independent if the
            model is cased or not)
        tokenizer_name_or_path: Name or path of the tokenizer. When
            None, then model_name_or_path is used
        backend: Backend used for model inference. Can be only `torch` for now for this class.
    """

    config_file_name: str = "sentence_bert_config.json"
    config_keys: list[str] = ["max_seq_length", "do_lower_case"]
    save_in_root: bool = True

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int | None = None,
        model_args: dict[str, Any] | None = None,
        tokenizer_args: dict[str, Any] | None = None,
        config_args: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        do_lower_case: bool = False,
        tokenizer_name_or_path: str | None = None,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.do_lower_case = do_lower_case
        self.backend = backend
        if model_args is None:
            model_args = {}
        if tokenizer_args is None:
            tokenizer_args = {}
        if config_args is None:
            config_args = {}

        self.config, is_peft_model = self._load_config(model_name_or_path, cache_dir, backend, config_args)
        self._load_model(model_name_or_path, self.config, cache_dir, backend, is_peft_model, **model_args)

        if max_seq_length is not None and "model_max_length" not in tokenizer_args:
            tokenizer_args["model_max_length"] = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            (tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path),
            cache_dir=cache_dir,
            **tokenizer_args,
        )

        # Set max_seq_length
        self.max_seq_length = max_seq_length
        if max_seq_length is None:
            if hasattr(self.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                self.max_seq_length = min(self.config.max_position_embeddings, self.tokenizer.model_max_length)

    def _load_config(
        self, model_name_or_path: str, cache_dir: str | None, backend: str, config_args: dict[str, Any]
    ) -> tuple[PeftConfig | PretrainedConfig, bool]:
        """Loads the transformers or PEFT configuration

        Args:
            model_name_or_path (str): The model name on Hugging Face (e.g. 'naver/splade-cocondenser-ensembledistil')
                or the path to a local model directory.
            cache_dir (str | None): The cache directory to store the model configuration.
            backend (str): Backend used for model inference. Can be only `torch` for now for this class.
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
        config: PeftConfig | PretrainedConfig,
        cache_dir: str,
        backend: str,
        is_peft_model: bool,
        **model_args,
    ) -> None:
        """Loads the transformers or PEFT model into the `auto_model` attribute

        Args:
            model_name_or_path (str): The model name on Hugging Face (e.g. 'naver/splade-cocondenser-ensembledistil')
                or the path to a local model directory.
            config ("PeftConfig" | PretrainedConfig): The model configuration.
            cache_dir (str | None): The cache directory to store the model configuration.
            backend (str): Backend used for model inference. Can be only `torch` for now for this class.
            is_peft_model (bool): Whether the model is a PEFT model.
            model_args (dict[str, Any]): Keyword arguments passed to the Hugging Face Transformers model.
        """
        if backend == "torch":
            # When loading a PEFT model, we need to load the base model first,
            # but some model_args are only for the adapter
            if is_peft_model:
                for adapter_only_kwarg in ["revision"]:
                    model_args.pop(adapter_only_kwarg, None)

            self.auto_model = AutoModelForMaskedLM.from_pretrained(
                model_name_or_path, config=config, cache_dir=cache_dir, **model_args
            )
        else:
            raise ValueError(
                f"Backend '{backend}' is not yet supported. MLMTransformer currently only works with the 'torch' backend."
            )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Returns the MLM head logits for the input features as token embeddings."""
        trans_features = {
            key: value
            for key, value in features.items()
            if key in ["input_ids", "attention_mask", "token_type_ids", "inputs_embeds"]
        }
        try:
            features["token_embeddings"] = self.auto_model(**trans_features).logits
        except AttributeError:
            features["token_embeddings"] = self.auto_model(**trans_features)[0]

        return features

    def tokenize(
        self, texts: list[str] | list[dict] | list[tuple[str, str]], padding: str | bool = True
    ) -> dict[str, torch.Tensor]:
        """Tokenizes a text and maps tokens to token-ids"""
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=padding,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length,
            )
        )
        return output

    def get_sentence_embedding_dimension(self) -> int:
        return self.auto_model.config.vocab_size

    def __repr__(self) -> str:
        return f"MLMTransformer({dict(self.get_config_dict(), architecture=self.auto_model.__class__.__name__)})"

    def save(self, output_path: str, safe_serialization: bool = True, **kwargs) -> None:
        self.auto_model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.save_tokenizer(output_path)
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
