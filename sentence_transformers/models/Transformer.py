from __future__ import annotations

import json
import logging
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import huggingface_hub
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, MT5Config, PretrainedConfig, T5Config
from transformers.utils.import_utils import is_peft_available
from transformers.utils.peft_utils import find_adapter_config_file

logger = logging.getLogger(__name__)

if TYPE_CHECKING and is_peft_available():
    from peft import PeftConfig


def _save_pretrained_wrapper(_save_pretrained_fn: Callable, subfolder: str) -> Callable[..., None]:
    def wrapper(save_directory: str | Path, **kwargs) -> None:
        os.makedirs(Path(save_directory) / subfolder, exist_ok=True)
        return _save_pretrained_fn(Path(save_directory) / subfolder, **kwargs)

    return wrapper


class Transformer(nn.Module):
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
        tokenizer_name_or_path: str = None,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.config_keys = ["max_seq_length", "do_lower_case"]
        self.do_lower_case = do_lower_case
        self.backend = backend
        if model_args is None:
            model_args = {}
        if tokenizer_args is None:
            tokenizer_args = {}
        if config_args is None:
            config_args = {}

        config, is_peft_model = self._load_config(model_name_or_path, cache_dir, backend, config_args)
        self._load_model(model_name_or_path, config, cache_dir, backend, is_peft_model, **model_args)

        if max_seq_length is not None and "model_max_length" not in tokenizer_args:
            tokenizer_args["model_max_length"] = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path,
            cache_dir=cache_dir,
            **tokenizer_args,
        )

        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if (
                hasattr(self.auto_model, "config")
                and hasattr(self.auto_model.config, "max_position_embeddings")
                and hasattr(self.tokenizer, "model_max_length")
            ):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

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
                    'To use other backends, load the model with `backend="torch"`, call `model[0].auto_model.merge_and_unload()`, '
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
            adapter_only_kwargs = {}
            if is_peft_model:
                for adapter_only_kwarg in ["revision"]:
                    if adapter_only_kwarg in model_args:
                        adapter_only_kwargs[adapter_only_kwarg] = model_args.pop(adapter_only_kwarg)

            if isinstance(config, T5Config):
                self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
            elif isinstance(config, MT5Config):
                self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
            else:
                self.auto_model = AutoModel.from_pretrained(
                    model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                )

            if is_peft_model:
                self._load_peft_model(model_name_or_path, config, cache_dir, **model_args, **adapter_only_kwargs)
        elif backend == "onnx":
            self._load_onnx_model(model_name_or_path, config, cache_dir, **model_args)
        elif backend == "openvino":
            self._load_openvino_model(model_name_or_path, config, cache_dir, **model_args)
        else:
            raise ValueError(f"Unsupported backend '{backend}'. `backend` should be `torch`, `onnx`, or `openvino`.")

    def _load_peft_model(self, model_name_or_path: str, config: PeftConfig, cache_dir: str, **model_args) -> None:
        from peft import PeftModel

        self.auto_model = PeftModel.from_pretrained(
            self.auto_model, model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )

    def _load_openvino_model(
        self, model_name_or_path: str, config: PretrainedConfig, cache_dir: str, **model_args
    ) -> None:
        if isinstance(config, T5Config) or isinstance(config, MT5Config):
            raise ValueError("T5 models are not yet supported by the OpenVINO backend.")

        try:
            from optimum.intel import OVModelForFeatureExtraction
            from optimum.intel.openvino import OV_XML_FILE_NAME
        except ModuleNotFoundError:
            raise Exception(
                "Using the OpenVINO backend requires installing Optimum and OpenVINO. "
                "You can install them with pip: `pip install optimum[openvino]`."
            )

        load_path = Path(model_name_or_path)
        is_local = load_path.exists()
        backend_name = "OpenVINO"
        target_file_glob = "openvino*.xml"

        # Determine whether the model should be exported or whether we can load it directly
        export, model_args = self._backend_should_export(
            load_path, is_local, model_args, OV_XML_FILE_NAME, target_file_glob, backend_name
        )

        # If we're exporting, then there's no need for a file_name to load the model from
        if export:
            model_args.pop("file_name", None)

        # ov_config can be either a dictionary, or point to a json file with an OpenVINO config
        if "ov_config" in model_args:
            ov_config = model_args["ov_config"]
            if not isinstance(ov_config, dict):
                if not Path(ov_config).exists():
                    raise ValueError(
                        "ov_config should be a dictionary or a path to a .json file containing an OpenVINO config"
                    )
                with open(ov_config, encoding="utf-8") as f:
                    model_args["ov_config"] = json.load(f)
        else:
            model_args["ov_config"] = {}

        # Either load an exported model, or export the model to OpenVINO
        self.auto_model: OVModelForFeatureExtraction = OVModelForFeatureExtraction.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=cache_dir,
            export=export,
            **model_args,
        )
        # Wrap the save_pretrained method to save the model in the correct subfolder
        self.auto_model._save_pretrained = _save_pretrained_wrapper(self.auto_model._save_pretrained, self.backend)

        # Warn the user to save the model if they haven't already
        if export:
            self._backend_warn_to_save(model_name_or_path, is_local, backend_name)

    def _load_onnx_model(
        self, model_name_or_path: str, config: PretrainedConfig, cache_dir: str, **model_args
    ) -> None:
        try:
            import onnxruntime as ort
            from optimum.onnxruntime import ONNX_WEIGHTS_NAME, ORTModelForFeatureExtraction
        except ModuleNotFoundError:
            raise Exception(
                "Using the ONNX backend requires installing Optimum and ONNX Runtime. "
                "You can install them with pip: `pip install optimum[onnxruntime]` "
                "or `pip install optimum[onnxruntime-gpu]`"
            )

        # Default to the highest priority available provider if not specified
        # E.g. Tensorrt > CUDA > CPU
        model_args["provider"] = model_args.pop("provider", ort.get_available_providers()[0])

        load_path = Path(model_name_or_path)
        is_local = load_path.exists()
        backend_name = "ONNX"
        target_file_glob = "*.onnx"

        # Determine whether the model should be exported or whether we can load it directly
        export, model_args = self._backend_should_export(
            load_path, is_local, model_args, ONNX_WEIGHTS_NAME, target_file_glob, backend_name
        )

        # If we're exporting, then there's no need for a file_name to load the model from
        if export:
            model_args.pop("file_name", None)

        # Either load an exported model, or export the model to ONNX
        self.auto_model: ORTModelForFeatureExtraction = ORTModelForFeatureExtraction.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=cache_dir,
            export=export,
            **model_args,
        )
        # Wrap the save_pretrained method to save the model in the correct subfolder
        self.auto_model._save_pretrained = _save_pretrained_wrapper(self.auto_model._save_pretrained, self.backend)

        # Warn the user to save the model if they haven't already
        if export:
            self._backend_warn_to_save(model_name_or_path, is_local, backend_name)

    def _backend_should_export(
        self,
        load_path: Path,
        is_local: bool,
        model_args: dict[str, Any],
        target_file_name: str,
        target_file_glob: str,
        backend_name: str,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Determines whether the model should be exported to the backend, or if it can be loaded directly.
        Also update the `file_name` and `subfolder` model_args if necessary.

        These are the cases:

        1. If export is set in model_args, just return export
        2. If `<subfolder>/<file_name>` exists; set export to False
        3. If `<backend>/<file_name>` exists; set export to False and set subfolder to the backend (e.g. "onnx")
        4. If `<file_name>` contains a folder, add those folders to the subfolder and set the file_name to the last part

        We will warn if:

        1. The expected file does not exist in the model directory given the optional file_name and subfolder.
           If there are valid files for this backend, but they're don't align with file_name, then we give a useful warning.
        2. Multiple files are found in the model directory that match the target file name and the user did not
           specify the desired file name via `model_kwargs={"file_name": "<file_name>"}`

        Args:
            load_path: The model repository or directory, as a Path instance
            is_local: Whether the model is local or remote, i.e. whether load_path is a local directory
            model_args: The model_args dictionary. Notable keys are "export", "file_name", and "subfolder"
            target_file_name: The expected file name in the model directory, e.g. "model.onnx" or "openvino_model.xml"
            target_file_glob: The glob pattern to match the target file name, e.g. "*.onnx" or "openvino*.xml"
            backend_name: The human-readable name of the backend for use in warnings, e.g. "ONNX" or "OpenVINO"

        Returns:
            Tuple[bool, dict[str, Any]]: A tuple of the export boolean and the updated model_args dictionary.
        """

        export = model_args.pop("export", None)
        if export:
            return export, model_args

        file_name = model_args.get("file_name", target_file_name)
        subfolder = model_args.get("subfolder", None)
        primary_full_path = Path(subfolder, file_name).as_posix() if subfolder else Path(file_name).as_posix()
        secondary_full_path = (
            Path(subfolder, self.backend, file_name).as_posix()
            if subfolder
            else Path(self.backend, file_name).as_posix()
        )
        glob_pattern = f"{subfolder}/**/{target_file_glob}" if subfolder else f"**/{target_file_glob}"

        # Get the list of files in the model directory that match the target file name
        if is_local:
            model_file_names = [path.relative_to(load_path).as_posix() for path in load_path.glob(glob_pattern)]
        else:
            all_files = huggingface_hub.list_repo_files(
                load_path.as_posix(),
                repo_type="model",
                revision=model_args.get("revision", None),
                token=model_args.get("token", None),
            )
            model_file_names = [fname for fname in all_files if fnmatch(fname, glob_pattern)]

        # First check if the expected file exists in the root of the model directory
        # If it doesn't, check if it exists in the backend subfolder.
        # If it does, set the subfolder to include the backend
        model_found = primary_full_path in model_file_names
        if not model_found and "subfolder" not in model_args:
            model_found = secondary_full_path in model_file_names
            if model_found:
                if len(model_file_names) > 1 and "file_name" not in model_args:
                    logger.warning(
                        f"Multiple {backend_name} files found in {load_path.as_posix()!r}: {model_file_names}, defaulting to {secondary_full_path!r}. "
                        f'Please specify the desired file name via `model_kwargs={{"file_name": "<file_name>"}}`.'
                    )
                model_args["subfolder"] = self.backend
                model_args["file_name"] = file_name
        if export is None:
            export = not model_found

        # If the file_name contains subfolders, set it as the subfolder instead
        file_name_parts = Path(file_name).parts
        if len(file_name_parts) > 1:
            model_args["file_name"] = file_name_parts[-1]
            model_args["subfolder"] = Path(model_args.get("subfolder", ""), *file_name_parts[:-1]).as_posix()

        if export:
            logger.warning(
                f"No {file_name!r} found in {load_path.as_posix()!r}. Exporting the model to {backend_name}."
            )

            if model_file_names:
                logger.warning(
                    f"If you intended to load one of the {model_file_names} {backend_name} files, "
                    f'please specify the desired file name via `model_kwargs={{"file_name": "{model_file_names[0]}"}}`.'
                )

        return export, model_args

    def _backend_warn_to_save(self, model_name_or_path: str, is_local: str, backend_name: str) -> None:
        to_log = f"Saving the exported {backend_name} model is heavily recommended to avoid having to export it again."
        if is_local:
            to_log += f" Do so with `model.save_pretrained({model_name_or_path!r})`."
        else:
            to_log += f" Do so with `model.push_to_hub({model_name_or_path!r}, create_pr=True)`."
        logger.warning(to_log)

    def _load_t5_model(self, model_name_or_path: str, config: PretrainedConfig, cache_dir: str, **model_args) -> None:
        """Loads the encoder model from T5"""
        from transformers import T5EncoderModel

        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = T5EncoderModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )

    def _load_mt5_model(self, model_name_or_path: str, config: PretrainedConfig, cache_dir: str, **model_args) -> None:
        """Loads the encoder model from T5"""
        from transformers import MT5EncoderModel

        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = MT5EncoderModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )

    def __repr__(self) -> str:
        return f"Transformer({self.get_config_dict()}) with Transformer model: {self.auto_model.__class__.__name__} "

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Returns token_embeddings, cls_token"""
        trans_features = {
            key: value
            for key, value in features.items()
            if key in ["input_ids", "attention_mask", "token_type_ids", "inputs_embeds"]
        }

        output_states = self.auto_model(**trans_features, **kwargs, return_dict=False)
        output_tokens = output_states[0]

        # If the AutoModel is wrapped with a PeftModelForFeatureExtraction, then it may have added virtual tokens
        # We need to extend the attention mask to include these virtual tokens, or the pooling will fail
        if is_peft_available():
            from peft import PeftModelForFeatureExtraction

            if (
                isinstance(self.auto_model, PeftModelForFeatureExtraction)
                and self.auto_model.active_peft_config.is_prompt_learning
            ):
                batch_size = output_tokens.size(0)
                attention_mask = features["attention_mask"]
                prefix_attention_mask = torch.ones(
                    batch_size, self.auto_model.active_peft_config.num_virtual_tokens, device=attention_mask.device
                )
                features["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        features["token_embeddings"] = output_tokens

        if self.auto_model.config.output_hidden_states and len(output_states) > 2:
            all_layer_idx = 2  # I.e. after last_hidden_states and pooler_output
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features["all_layer_embeddings"] = hidden_states

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

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

    def get_config_dict(self) -> dict[str, Any]:
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        self.auto_model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, "sentence_bert_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @classmethod
    def load(cls, input_path: str) -> Transformer:
        # Old classes used other config names than 'sentence_bert_config.json'
        for config_name in [
            "sentence_bert_config.json",
            "sentence_roberta_config.json",
            "sentence_distilbert_config.json",
            "sentence_camembert_config.json",
            "sentence_albert_config.json",
            "sentence_xlm-roberta_config.json",
            "sentence_xlnet_config.json",
        ]:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        # Don't allow configs to set trust_remote_code
        if "model_args" in config and "trust_remote_code" in config["model_args"]:
            config["model_args"].pop("trust_remote_code")
        if "tokenizer_args" in config and "trust_remote_code" in config["tokenizer_args"]:
            config["tokenizer_args"].pop("trust_remote_code")
        if "config_args" in config and "trust_remote_code" in config["config_args"]:
            config["config_args"].pop("trust_remote_code")
        return cls(model_name_or_path=input_path, **config)
