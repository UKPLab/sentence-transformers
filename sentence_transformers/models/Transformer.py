from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import huggingface_hub
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, MT5Config, T5Config

logger = logging.getLogger(__name__)


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

        config = AutoConfig.from_pretrained(model_name_or_path, **config_args, cache_dir=cache_dir)
        self._load_model(model_name_or_path, config, cache_dir, backend, **model_args)

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

    def _load_model(self, model_name_or_path, config, cache_dir, backend, **model_args) -> None:
        """Loads the transformer model"""
        if backend == "torch":
            if isinstance(config, T5Config):
                self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
            elif isinstance(config, MT5Config):
                self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
            else:
                self.auto_model = AutoModel.from_pretrained(
                    model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                )
        elif backend == "onnx":
            self._load_onnx_model(model_name_or_path, config, cache_dir, **model_args)
        elif backend == "openvino":
            self._load_openvino_model(model_name_or_path, config, cache_dir, **model_args)
        else:
            raise ValueError(f"Unsupported backend '{backend}'. `backend` should be `torch`, `onnx`, or `openvino`.")

    def _load_openvino_model(self, model_name_or_path, config, cache_dir, **model_args) -> None:
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

        # Determine whether the model should be exported or whether we can load it directly
        export = self._backend_should_export(load_path, is_local, model_args, OV_XML_FILE_NAME)

        # If we're exporting, then there's no need for a file_name to load the model from
        if export:
            model_args.pop("file_name", None)

        # ov_config can be either a dictionary, or point to a json file with an OpenVINO config
        if "ov_config" in model_args:
            ov_config = model_args["ov_config"]
            if not isinstance(ov_config, dict):
                if not Path(ov_config).exists():
                    raise ValueError(
                        "ov_config should be a dictionary or point to a .json file containing an OpenVINO config"
                    )
                with open(ov_config, encoding="utf-8") as f:
                    model_args["ov_config"] = json.load(f)
        else:
            model_args["ov_config"] = {}

        # Either load an exported model, or export the model to ONNX
        self.auto_model: OVModelForFeatureExtraction = OVModelForFeatureExtraction.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=cache_dir,
            export=export,
            **model_args,
        )

        # Warn the user to save the model if they haven't already
        if export:
            self._backend_warn_to_save(model_name_or_path, is_local, "ONNX")

    def _load_onnx_model(self, model_name_or_path, config, cache_dir, **model_args) -> None:
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

        # Determine whether the model should be exported or whether we can load it directly
        export = self._backend_should_export(load_path, is_local, model_args, ONNX_WEIGHTS_NAME)

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

        # Warn the user to save the model if they haven't already
        if export:
            self._backend_warn_to_save(model_name_or_path, is_local, "ONNX")

    def _backend_should_export(
        self, load_path: Path, is_local: bool, model_args: dict[str, Any], target_file_name: str
    ) -> None:
        export = model_args.pop("export", None)
        if export is not None:
            return export

        file_name = model_args.get("file_name", target_file_name)
        subfolder = model_args.get("subfolder", None)
        full_path = os.path.join(subfolder, file_name) if subfolder else file_name

        if is_local:
            return not (load_path / full_path).is_file()

        all_files = huggingface_hub.list_repo_files(
            load_path.as_posix(),
            repo_type="model",
            revision=model_args.get("revision", None),
            token=model_args.get("token", None),
        )
        return full_path not in all_files

    def _backend_warn_to_save(self, model_name_or_path: str, is_local: str, backend_name: str) -> None:
        to_log = f"Saving the exported {backend_name} model is heavily recommended to avoid having to export it again."
        if is_local:
            to_log += f" Do so with `model.save_pretrained({model_name_or_path!r})`."
        else:
            to_log += f" Do so with `model.push_to_hub({model_name_or_path!r}, create_pr=True)`."
        logger.warning(to_log)

    def _load_t5_model(self, model_name_or_path, config, cache_dir, **model_args) -> None:
        """Loads the encoder model from T5"""
        from transformers import T5EncoderModel

        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = T5EncoderModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )

    def _load_mt5_model(self, model_name_or_path, config, cache_dir, **model_args) -> None:
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
        trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = self.auto_model(**trans_features, **kwargs, return_dict=False)
        output_tokens = output_states[0]

        features.update({"token_embeddings": output_tokens, "attention_mask": features["attention_mask"]})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})

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
