from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import trange

from sentence_transformers import SentenceTransformer
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.sparse_encoder.model_card import SparseEncoderModelCardData
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
from sentence_transformers.util import batch_to_device, truncate_embeddings_for_sparse

logger = logging.getLogger(__name__)


class SparseEncoder(SentenceTransformer):
    # TODO: Add the proper description with associate example + docstring to all the functions
    # TODO: Check the last three functions not done

    # NOTE: Function available in SparseEvaluator:
    # ---------------------------------------------Not done---------------------------------------------
    # - start_multi_process_pool
    # - encode_multi_process
    # - _encode_multi_process_worker

    # --------------------------------------Done-----------------------------------------------------------
    # - __init__ (make sure nothing else need to be overrided)
    # - encode (clean the implementation)

    # - save (just for the docstring)
    # - save_pretrained (just for the docstring)
    # - push_to_hub (just for the docstring)
    # - _load_auto_model (to have a default splade model loaded)
    # - _load_sbert_model (just for the docstring)
    # - _update_default_model_id
    # - load

    # -----------------------------------In my opinion shouldn't be done ------------------------------------
    # - get_backend
    # - forward
    # - similarity_fn_name
    # - similarity
    # - similarity_pairwise
    # - stop_multi_process_pool
    # - set_pooling_include_prompt
    # - get_max_seq_length
    # - tokenize
    # - get_sentence_features
    # - get_sentence_embedding_dimension
    # - truncate_sentence_embeddings
    # - _first_module
    # - _last_module
    # - _create_model_card
    # - save_to_hub
    # - _text_length
    # - evaluate
    # - _load_module_class_from_ref
    # - device
    # - tokenizer
    # - max_seq_length
    # - _target_device
    # - _no_split_modules
    # - _keys_to_ignore_on_save
    # - gradient_checkpointing_enable

    # -----------------------------------Added------------------------------------
    # - get_sparsity_stats

    def __init__(
        self,
        model_name_or_path: str | None = None,
        modules: Iterable[nn.Module] | None = None,
        device: str | None = None,
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        similarity_fn_name: str | SimilarityFunction | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        use_auth_token: bool | str | None = None,
        truncate_dim: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        model_card_data: SparseEncoderModelCardData | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            modules=modules,
            device=device,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
            similarity_fn_name=similarity_fn_name,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            use_auth_token=use_auth_token,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            model_card_data=model_card_data,
            backend=backend,
        )
        self.model_card_data = model_card_data or SparseEncoderModelCardData()
        self.model_card_data.register_model(self)

    def encode(
        self,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
        convert_to_tensor: bool = True,
        convert_to_sparse_tensor: bool = True,
        device: str | None = None,
        **kwargs: Any,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]:
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)

        if output_value != "sentence_embedding":
            convert_to_tensor = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if prompt is None:
            if prompt_name is not None:
                try:
                    prompt = self.prompts[prompt_name]
                except KeyError:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(self.prompts.keys())!r}."
                    )
            elif self.default_prompt_name is not None:
                prompt = self.prompts.get(self.default_prompt_name, None)
        else:
            if prompt_name is not None:
                logger.warning(
                    "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                    "Ignoring the `prompt_name` in favor of `prompt`."
                )

        extra_features = {}
        if prompt is not None:
            sentences = [prompt + sentence for sentence in sentences]

            # Some models (e.g. INSTRUCTOR, GRIT) require removing the prompt before pooling
            # Tracking the prompt length allow us to remove the prompt during pooling
            tokenized_prompt = self.tokenize([prompt])
            if "input_ids" in tokenized_prompt:
                extra_features["prompt_length"] = tokenized_prompt["input_ids"].shape[-1] - 1

        if device is None:
            device = self.device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, self.device)
            features.update(extra_features)

            with torch.no_grad():
                out_features = self.forward(features, **kwargs)
                out_features["sentence_embedding"] = truncate_embeddings_for_sparse(
                    out_features["sentence_embedding"], truncate_dim=self.truncate_dim
                )

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb in out_features[output_value]:
                        if convert_to_sparse_tensor:
                            token_emb = token_emb.to_sparse()
                        embeddings.append(token_emb)
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if convert_to_sparse_tensor:
                        embeddings = embeddings.to_sparse()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if not convert_to_tensor:
            return all_embeddings

        all_embeddings = torch.stack(all_embeddings)

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def save(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ) -> None:
        """
        Saves a model and its configuration files to a directory, so that it can be loaded
        with ``SparseEncoder(path)`` again.

        Args:
            path (str): Path on disc where the model will be saved.
            model_name (str, optional): Optional model name.
            create_model_card (bool, optional): If True, create a README.md with basic information about this model.
            train_datasets (List[str], optional): Optional list with the names of the datasets used to train the model.
            safe_serialization (bool, optional): If True, save the model using safetensors. If False, save the model
                the traditional (but unsafe) PyTorch way.
        """
        return super().save(
            path=path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )

    def save_pretrained(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ) -> None:
        """
        Saves a model and its configuration files to a directory, so that it can be loaded
        with ``SparseEncoder(path)`` again.

        Args:
            path (str): Path on disc where the model will be saved.
            model_name (str, optional): Optional model name.
            create_model_card (bool, optional): If True, create a README.md with basic information about this model.
            train_datasets (List[str], optional): Optional list with the names of the datasets used to train the model.
            safe_serialization (bool, optional): If True, save the model using safetensors. If False, save the model
                the traditional (but unsafe) PyTorch way.
        """
        return super().save_pretrained(
            path=path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )

    def _update_default_model_id(self, model_card):
        if self.model_card_data.model_id:
            model_card = model_card.replace(
                'model = SparseEncoder("sparse_encoder_model_id"',
                f'model = SparseEncoder("{self.model_card_data.model_id}"',
            )
        return model_card

    def push_to_hub(
        self,
        repo_id: str,
        token: str | None = None,
        private: bool | None = None,
        safe_serialization: bool = True,
        commit_message: str | None = None,
        local_model_path: str | None = None,
        exist_ok: bool = False,
        replace_model_card: bool = False,
        train_datasets: list[str] | None = None,
        revision: str | None = None,
        create_pr: bool = False,
    ) -> str:
        """
        Uploads all elements of this Sparse Encoder to a new HuggingFace Hub repository.

        Args:
            repo_id (str): Repository name for your model in the Hub, including the user or organization.
            token (str, optional): An authentication token (See https://huggingface.co/settings/token)
            private (bool, optional): Set to true, for hosting a private model
            safe_serialization (bool, optional): If true, save the model using safetensors. If false, save the model the traditional PyTorch way
            commit_message (str, optional): Message to commit while pushing.
            local_model_path (str, optional): Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded
            exist_ok (bool, optional): If true, saving to an existing repository is OK. If false, saving only to a new repository is possible
            replace_model_card (bool, optional): If true, replace an existing model card in the hub with the automatically created model card
            train_datasets (List[str], optional): Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.
            revision (str, optional): Branch to push the uploaded files to
            create_pr (bool, optional): If True, create a pull request instead of pushing directly to the main branch

        Returns:
            str: The url of the commit of your model in the repository on the Hugging Face Hub.
        """
        return super().push_to_hub(
            repo_id=repo_id,
            token=token,
            private=private,
            safe_serialization=safe_serialization,
            commit_message=commit_message,
            local_model_path=local_model_path,
            exist_ok=exist_ok,
            replace_model_card=replace_model_card,
            train_datasets=train_datasets,
            revision=revision,
            create_pr=create_pr,
        )

    def _load_auto_model(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
    ) -> list[nn.Module]:
        """
        Creates a simple MLMTransformer + SPladePooling model and returns the modules

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            token (Optional[Union[bool, str]]): The token to use for the model.
            cache_folder (Optional[str]): The folder to cache the model.
            revision (Optional[str], optional): The revision of the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            local_files_only (bool, optional): Whether to use only local files. Defaults to False.
            model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the model. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the tokenizer. Defaults to None.
            config_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the config. Defaults to None.

        Returns:
            List[nn.Module]: A list containing the MLMTransformer model and the spladepooling model.
        """
        logger.warning(
            f"No sparse-encoder model found with name {model_name_or_path}. Creating a new one with spladepooling."
        )

        shared_kwargs = {
            "token": token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        model_kwargs = shared_kwargs if model_kwargs is None else {**shared_kwargs, **model_kwargs}
        tokenizer_kwargs = shared_kwargs if tokenizer_kwargs is None else {**shared_kwargs, **tokenizer_kwargs}
        config_kwargs = shared_kwargs if config_kwargs is None else {**shared_kwargs, **config_kwargs}

        mlmtransformer_model = MLMTransformer(
            model_name_or_path,
            cache_dir=cache_folder,
            model_args=model_kwargs,
            tokenizer_args=tokenizer_kwargs,
            config_args=config_kwargs,
            backend=self.backend,
        )
        splade_pooling_model = SpladePooling(pooling_strategy="max")
        if not local_files_only:
            self.model_card_data.set_base_model(model_name_or_path, revision=revision)
        return [mlmtransformer_model, splade_pooling_model]

    def _load_sbert_model(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, nn.Module]:
        """
        Loads a full SparseEncoder model using the modules.json file.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            token (Optional[Union[bool, str]]): The token to use for the model.
            cache_folder (Optional[str]): The folder to cache the model.
            revision (Optional[str], optional): The revision of the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            local_files_only (bool, optional): Whether to use only local files. Defaults to False.
            model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the model. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the tokenizer. Defaults to None.
            config_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the config. Defaults to None.

        Returns:
            OrderedDict[str, nn.Module]: An ordered dictionary containing the modules of the model.
        """
        return super()._load_sbert_model(
            model_name_or_path=model_name_or_path,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
        )

    @staticmethod
    def load(input_path) -> SparseEncoder:
        return SparseEncoder(input_path)

    def get_sparsity_stats(self, embeddings: torch.Tensor) -> dict[str, float]:
        """
        Calculate sparsity statistics for the given embeddings.

        Args:
            embeddings (torch.Tensor): The embeddings to analyze

        Returns:
            Dict[str, float]: Dictionary with sparsity statistics
        """
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("Embeddings must be a torch.Tensor")

        if embeddings.is_sparse or embeddings.is_sparse_csr:
            # For sparse tensors, calculate directly
            total_elements = np.prod(embeddings.shape)
            non_zero = embeddings._nnz()
        else:
            # For dense tensors
            non_zero = torch.count_nonzero(embeddings).item()
            total_elements = embeddings.numel()

        sparsity = 1.0 - (non_zero / total_elements)
        density = 1.0 - sparsity

        return {"sparsity": sparsity, "density": density, "non_zero_count": non_zero, "total_elements": total_elements}
