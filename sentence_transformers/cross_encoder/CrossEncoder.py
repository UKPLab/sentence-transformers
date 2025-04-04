from __future__ import annotations

import logging
import os
import tempfile
import traceback
from typing import Callable, Literal, overload

import numpy as np
import torch
from huggingface_hub import HfApi
from packaging import version
from torch import nn
from tqdm.autonotebook import trange
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.utils import PushToHubMixin
from typing_extensions import deprecated

from sentence_transformers import __version__
from sentence_transformers.cross_encoder.fit_mixin import FitMixin
from sentence_transformers.cross_encoder.model_card import CrossEncoderModelCardData, generate_model_card
from sentence_transformers.cross_encoder.util import (
    cross_encoder_init_args_decorator,
    cross_encoder_predict_rank_args_decorator,
)
from sentence_transformers.util import fullname, get_device_name, import_from_string, load_file_path

logger = logging.getLogger(__name__)


class CrossEncoder(nn.Module, PushToHubMixin, FitMixin):
    """
    A CrossEncoder takes exactly two sentences / texts as input and either predicts
    a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
    on a scale of 0 ... 1.

    It does not yield a sentence embedding and does not work for individual sentences.

    Args:
        model_name_or_path (str): A model name from Hugging Face Hub that can be loaded with AutoModel, or a path to a local
            model. We provide several pre-trained CrossEncoder models that can be used for common tasks.
        num_labels (int, optional): Number of labels of the classifier. If 1, the CrossEncoder is a regression model that
            outputs a continuous score 0...1. If > 1, it output several scores that can be soft-maxed to get
            probability scores for the different classes. Defaults to None.
        max_length (int, optional): Max length for input sequences. Longer sequences will be truncated. If None, max
            length of the model will be used. Defaults to None.
        activation_fn (Callable, optional): Callable (like nn.Sigmoid) about the default activation function that
            should be used on-top of model.predict(). If None. nn.Sigmoid() will be used if num_labels=1,
            else nn.Identity(). Defaults to None.
        device (str, optional): Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU
            can be used.
        cache_folder (`str`, `Path`, optional): Path to the folder where cached files are stored.
        trust_remote_code (bool, optional): Whether or not to allow for custom models defined on the Hub in their own modeling files.
            This option should only be set to True for repositories you trust and in which you have read the code, as it
            will execute code present on the Hub on your local machine. Defaults to False.
        revision (str, optional): The specific model version to use. It can be a branch name, a tag name, or a commit id,
            for a stored model on Hugging Face. Defaults to None.
        local_files_only (bool, optional): Whether or not to only look at local files (i.e., do not try to download the model).
        token (bool or str, optional): Hugging Face authentication token to download private models.
        model_kwargs (Dict[str, Any], optional): Additional model configuration parameters to be passed to the Hugging Face Transformers model.
            Particularly useful options are:

            - ``torch_dtype``: Override the default `torch.dtype` and load the model under a specific `dtype`.
              The different options are:

                    1. ``torch.float16``, ``torch.bfloat16`` or ``torch.float``: load in a specified
                    ``dtype``, ignoring the model's ``config.torch_dtype`` if one exists. If not specified - the model will
                    get loaded in ``torch.float`` (fp32).

                    2. ``"auto"`` - A ``torch_dtype`` entry in the ``config.json`` file of the model will be
                    attempted to be used. If this entry isn't found then next check the ``dtype`` of the first weight in
                    the checkpoint that's of a floating point type and use that as ``dtype``. This will load the model
                    using the ``dtype`` it was saved in at the end of the training. It can't be used as an indicator of how
                    the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.
            - ``attn_implementation``: The attention implementation to use in the model (if relevant). Can be any of
              `"eager"` (manual implementation of the attention), `"sdpa"` (using `F.scaled_dot_product_attention
              <https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html>`_),
              or `"flash_attention_2"` (using `Dao-AILab/flash-attention <https://github.com/Dao-AILab/flash-attention>`_).
              By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"`
              implementation.

            See the `AutoModelForSequenceClassification.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForSequenceClassification.from_pretrained>`_
            documentation for more details.
        tokenizer_kwargs (Dict[str, Any], optional): Additional tokenizer configuration parameters to be passed to the Hugging Face Transformers tokenizer.
            See the `AutoTokenizer.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained>`_
            documentation for more details.
        config_kwargs (Dict[str, Any], optional): Additional model configuration parameters to be passed to the Hugging Face Transformers config.
            See the `AutoConfig.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained>`_
            documentation for more details. For example, you can set ``classifier_dropout`` via this parameter.
        model_card_data (:class:`~sentence_transformers.model_card.SentenceTransformerModelCardData`, optional): A model
            card data object that contains information about the model. This is used to generate a model card when saving
            the model. If not set, a default model card data object is created.
    """

    @cross_encoder_init_args_decorator
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int = None,
        max_length: int = None,
        activation_fn: Callable | None = None,
        device: str | None = None,
        cache_folder: str = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        model_kwargs: dict = None,
        tokenizer_kwargs: dict = None,
        config_kwargs: dict = None,
        model_card_data: CrossEncoderModelCardData | None = None,
    ) -> None:
        super().__init__()
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}
        if config_kwargs is None:
            config_kwargs = {}
        self.model_card_data = model_card_data or CrossEncoderModelCardData()
        self.trust_remote_code = trust_remote_code
        self._model_card_text = None

        config: PretrainedConfig = AutoConfig.from_pretrained(
            model_name_or_path,
            cache_dir=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            **config_kwargs,
        )
        if hasattr(config, "sentence_transformers") and "version" in config.sentence_transformers:
            model_version = config.sentence_transformers["version"]
            if version.parse(model_version) > version.parse(__version__):
                logger.warning(
                    f"You are trying to use a model that was created with Sentence Transformers version {model_version}, "
                    f"but you're currently using version {__version__}. This might cause unexpected behavior or errors. "
                    "In that case, try to update to the latest version."
                )

        classifier_trained = False
        if config.architectures is not None:
            classifier_trained = any([arch.endswith("ForSequenceClassification") for arch in config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            config.num_labels = num_labels
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            **model_kwargs,
        )
        if "model_max_length" not in tokenizer_kwargs and max_length is not None:
            tokenizer_kwargs["model_max_length"] = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            **tokenizer_kwargs,
        )
        if "model_max_length" not in tokenizer_kwargs and hasattr(self.config, "max_position_embeddings"):
            self.tokenizer.model_max_length = min(self.tokenizer.model_max_length, self.config.max_position_embeddings)

        # Check if a readme exists
        model_card_path = load_file_path(
            model_name_or_path,
            "README.md",
            token=model_kwargs.get("token", None),
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if model_card_path is not None:
            try:
                with open(model_card_path, encoding="utf8") as fIn:
                    self._model_card_text = fIn.read()
            except Exception:
                pass

        self.set_activation_fn(activation_fn)

        if device is None:
            device = get_device_name()
            logger.info(f"Use pytorch device: {device}")
        self.model.to(device)

        # Pass the model to the model card data for later use in generating a model card upon saving this model
        self.model_card_data.register_model(self)
        self.model_card_data.set_base_model(model_name_or_path, revision=revision)

    def set_activation_fn(self, activation_fn: Callable | None, set_default: bool = True) -> None:
        if activation_fn is not None:
            self.activation_fn = activation_fn
        else:
            self.activation_fn = self.get_default_activation_fn()

        if set_default:
            self.set_config_value("activation_fn", fullname(self.activation_fn))

    def get_default_activation_fn(self) -> Callable:
        activation_fn_path = None
        if hasattr(self.config, "sentence_transformers") and "activation_fn" in self.config.sentence_transformers:
            activation_fn_path = self.config.sentence_transformers["activation_fn"]

        # Backwards compatibility with <v4.0: we stored the activation_fn under 'sbert_ce_default_activation_function'
        elif (
            hasattr(self.config, "sbert_ce_default_activation_function")
            and self.config.sbert_ce_default_activation_function is not None
        ):
            activation_fn_path = self.config.sbert_ce_default_activation_function
            del self.config.sbert_ce_default_activation_function

        if activation_fn_path is not None:
            if self.trust_remote_code or activation_fn_path.startswith("torch."):
                return import_from_string(activation_fn_path)()
            logger.warning(
                f"Activation function path '{activation_fn_path}' is not trusted, using default activation function instead. "
                "Please load the CrossEncoder with `trust_remote_code=True` to allow loading custom activation "
                "functions via the configuration."
            )

        if self.config.num_labels == 1:
            return nn.Sigmoid()
        return nn.Identity()

    def set_config_value(self, key: str, value) -> None:
        """
        Set a value in the underlying model's config.

        Args:
            key (str): The key to set.
            value: The value to set.
        """
        try:
            if not hasattr(self.config, "sentence_transformers"):
                self.config.sentence_transformers = {}
            self.config.sentence_transformers[key] = value
        except Exception as e:
            logger.warning(f"Was not able to add '{key}' to the config: {str(e)}")

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def num_labels(self) -> int:
        return self.config.num_labels

    @property
    def max_length(self) -> int:
        return self.tokenizer.model_max_length

    @max_length.setter
    def max_length(self, value: int) -> None:
        self.tokenizer.model_max_length = value

    @property
    @deprecated(
        "The `default_activation_function` property was renamed and is now deprecated. "
        "Please use `activation_fn` instead."
    )
    def default_activation_function(self) -> Callable:
        return self.activation_fn

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @overload
    def predict(
        self,
        sentences: tuple[str, str] | list[str],
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        activation_fn: Callable | None = ...,
        apply_softmax: bool | None = ...,
        convert_to_numpy: Literal[False] = ...,
        convert_to_tensor: Literal[False] = ...,
    ) -> torch.Tensor: ...

    @overload
    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]] | tuple[str, str] | list[str],
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        activation_fn: Callable | None = ...,
        apply_softmax: bool | None = ...,
        convert_to_numpy: Literal[True] = True,
        convert_to_tensor: Literal[False] = False,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]] | tuple[str, str] | list[str],
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        activation_fn: Callable | None = ...,
        apply_softmax: bool | None = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: Literal[True] = ...,
    ) -> torch.Tensor: ...

    @overload
    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]],
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        activation_fn: Callable | None = ...,
        apply_softmax: bool | None = ...,
        convert_to_numpy: Literal[False] = ...,
        convert_to_tensor: Literal[False] = ...,
    ) -> list[torch.Tensor]: ...

    @torch.inference_mode()
    @cross_encoder_predict_rank_args_decorator
    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]] | tuple[str, str] | list[str],
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        activation_fn: Callable | None = None,
        apply_softmax: bool | None = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> list[torch.Tensor] | np.ndarray | torch.Tensor:
        """
        Performs predictions with the CrossEncoder on the given sentence pairs.

        Args:
            sentences (Union[List[Tuple[str, str]], Tuple[str, str]]): A list of sentence pairs [(Sent1, Sent2), (Sent3, Sent4)]
                or one sentence pair (Sent1, Sent2).
            batch_size (int, optional): Batch size for encoding. Defaults to 32.
            show_progress_bar (bool, optional): Output progress bar. Defaults to None.
            activation_fn (callable, optional): Activation function applied on the logits output of the CrossEncoder.
                If None, the ``model.activation_fn`` will be used, which defaults to :class:`torch.nn.Sigmoid` if num_labels=1, else
                :class:`torch.nn.Identity`. Defaults to None.
            convert_to_numpy (bool, optional): Convert the output to a numpy matrix. Defaults to True.
            apply_softmax (bool, optional): If set to True and `model.num_labels > 1`, applies softmax on the logits
                output such that for each sample, the scores of each class sum to 1. Defaults to False.
            convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, output
                a list of PyTorch tensors. Defaults to True.
            convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
                Defaults to False.

        Returns:
            Union[List[torch.Tensor], np.ndarray, torch.Tensor]: Predictions for the passed sentence pairs.
            The return type depends on the ``convert_to_numpy`` and ``convert_to_tensor`` parameters.
            If ``convert_to_tensor`` is True, the output will be a :class:`torch.Tensor`.
            If ``convert_to_numpy`` is True, the output will be a :class:`numpy.ndarray`.
            Otherwise, the output will be a list of :class:`torch.Tensor` values.

        Examples:
            ::

                from sentence_transformers import CrossEncoder

                model = CrossEncoder("cross-encoder/stsb-roberta-base")
                sentences = [["I love cats", "Cats are amazing"], ["I prefer dogs", "Dogs are loyal"]]
                model.predict(sentences)
                # => array([0.6912767, 0.4303499], dtype=float32)
        """
        input_was_singular = False
        if isinstance(sentences[0], str):  # Cast an individual pair to a list with length 1
            sentences = [sentences]
            input_was_singular = True

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )

        if activation_fn is not None:
            self.set_activation_fn(activation_fn, set_default=False)

        pred_scores = []
        self.model.eval()
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            batch = sentences[start_index : start_index + batch_size]
            features = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            features.to(self.model.device)
            model_predictions = self.model(**features, return_dict=True)
            logits = self.activation_fn(model_predictions.logits)

            if apply_softmax and logits.ndim > 1:
                logits = torch.nn.functional.softmax(logits, dim=1)
            pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().float().numpy() for score in pred_scores])

        if input_was_singular:
            pred_scores = pred_scores[0]

        return pred_scores

    @cross_encoder_predict_rank_args_decorator
    def rank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        return_documents: bool = False,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        activation_fn: Callable | None = None,
        apply_softmax=False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> list[dict[Literal["corpus_id", "score", "text"], int | float | str]]:
        """
        Performs ranking with the CrossEncoder on the given query and documents. Returns a sorted list with the document indices and scores.

        Args:
            query (str): A single query.
            documents (List[str]): A list of documents.
            top_k (Optional[int], optional): Return the top-k documents. If None, all documents are returned. Defaults to None.
            return_documents (bool, optional): If True, also returns the documents. If False, only returns the indices and scores. Defaults to False.
            batch_size (int, optional): Batch size for encoding. Defaults to 32.
            show_progress_bar (bool, optional): Output progress bar. Defaults to None.
            activation_fn ([type], optional): Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity. Defaults to None.
            convert_to_numpy (bool, optional): Convert the output to a numpy matrix. Defaults to True.
            apply_softmax (bool, optional): If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output. Defaults to False.
            convert_to_tensor (bool, optional): Convert the output to a tensor. Defaults to False.

        Returns:
            List[Dict[Literal["corpus_id", "score", "text"], Union[int, float, str]]]: A sorted list with the "corpus_id", "score", and optionally "text" of the documents.

        Example:
            ::

                from sentence_transformers import CrossEncoder
                model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

                query = "Who wrote 'To Kill a Mockingbird'?"
                documents = [
                    "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
                    "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
                    "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961.",
                    "Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.",
                    "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era.",
                    "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."
                ]

                model.rank(query, documents, return_documents=True)

            ::

                [{'corpus_id': 0,
                'score': 10.67858,
                'text': "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature."},
                {'corpus_id': 2,
                'score': 9.761677,
                'text': "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961."},
                {'corpus_id': 1,
                'score': -3.3099542,
                'text': "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil."},
                {'corpus_id': 5,
                'score': -4.8989105,
                'text': "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."},
                {'corpus_id': 4,
                'score': -5.082967,
                'text': "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era."}]
        """
        if self.num_labels != 1:
            raise ValueError(
                "CrossEncoder.rank() only works for models with num_labels=1. "
                "Consider using CrossEncoder.predict() with input pairs instead."
            )
        query_doc_pairs = [[query, doc] for doc in documents]
        scores = self.predict(
            sentences=query_doc_pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            activation_fn=activation_fn,
            apply_softmax=apply_softmax,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
        )

        results = []
        for i, score in enumerate(scores):
            results.append({"corpus_id": i, "score": score})
            if return_documents:
                results[-1].update({"text": documents[i]})

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, path: str, *, safe_serialization: bool = True, **kwargs) -> None:
        """
        Saves the model and tokenizer to path; identical to `save_pretrained`
        """
        if path is None:
            return

        logger.info(f"Save model to {path}")
        self.set_config_value("version", __version__)
        self.model.save_pretrained(path, safe_serialization=safe_serialization, **kwargs)
        self.tokenizer.save_pretrained(path, **kwargs)
        self._create_model_card(path)

    def save_pretrained(self, path: str, *, safe_serialization: bool = True, **kwargs) -> None:
        """
        Saves the model and tokenizer to path; identical to `save`
        """
        return self.save(path, safe_serialization=safe_serialization, **kwargs)

    def _create_model_card(self, path: str) -> None:
        """
        Create an automatic model and stores it in the specified path. If no training was done and the loaded model
        was a CrossEncoder model already, then its model card is reused.

        Args:
            path (str): The path where the model card will be stored.

        Returns:
            None
        """
        # If we loaded a model from the Hub, and no training was done, then
        # we don't generate a new model card, but reuse the old one instead.
        if self._model_card_text and "generated_from_trainer" not in self.model_card_data.tags:
            model_card = self._model_card_text
            if self.model_card_data.model_id:
                # If the original model card was saved without a model_id, we replace the model_id with the new model_id
                model_card = model_card.replace(
                    'model = CrossEncoder("cross_encoder_model_id"',
                    f'model = CrossEncoder("{self.model_card_data.model_id}"',
                )
        else:
            try:
                model_card = generate_model_card(self)
            except Exception:
                logger.error(
                    f"Error while generating model card:\n{traceback.format_exc()}"
                    "Consider opening an issue on https://github.com/UKPLab/sentence-transformers/issues with this traceback.\n"
                    "Skipping model card creation."
                )
                return

        with open(os.path.join(path, "README.md"), "w", encoding="utf8") as fOut:
            fOut.write(model_card)

    def push_to_hub(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        private: bool | None = None,
        safe_serialization: bool = True,
        commit_message: str | None = None,
        exist_ok: bool = False,
        revision: str | None = None,
        create_pr: bool = False,
        tags: list[str] | None = None,
    ) -> str:
        """
        Upload the CrossEncoder model to the Hugging Face Hub.

        Example:
            ::

                from sentence_transformers import CrossEncoder

                model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
                model.push_to_hub("username/my-crossencoder-model")
                # => "https://huggingface.co/username/my-crossencoder-model"

        Args:
            repo_id (str): The name of the repository on the Hugging Face Hub, e.g. "username/repo_name",
                "organization/repo_name" or just "repo_name".
            token (str, optional): The authentication token to use for the Hugging Face Hub API.
                If not provided, will use the token stored via the Hugging Face CLI.
            private (bool, optional): Whether to create a private repository. If not specified,
                the repository will be public.
            safe_serialization (bool, optional): Whether or not to convert the model weights in safetensors
                format for safer serialization. Defaults to True.
            commit_message (str, optional): The commit message to use for the push. Defaults to "Add new CrossEncoder model".
            exist_ok (bool, optional): If True, do not raise an error if the repository already exists.
                Ignored if ``create_pr=True``. Defaults to False.
            revision (str, optional): The git branch to commit to. Defaults to the head of the 'main' branch.
            create_pr (bool, optional): Whether to create a Pull Request with the upload or directly commit. Defaults to False.
            tags (list[str], optional): A list of tags to add to the model card. Defaults to None.

        Returns:
            str: URL of the commit or pull request (if create_pr=True)
        """
        api = HfApi(token=token)
        repo_url = api.create_repo(
            repo_id=repo_id,
            private=private,
            repo_type=None,
            exist_ok=exist_ok or create_pr,
        )
        repo_id = repo_url.repo_id  # Update the repo_id in case the old repo_id didn't contain a user or organization
        self.model_card_data.set_model_id(repo_id)
        if tags is not None:
            self.model_card_data.add_tags(tags)

        if revision is not None:
            api.create_branch(repo_id=repo_id, branch=revision, exist_ok=True)

        if commit_message is None:
            commit_message = "Add new CrossEncoder model"
        commit_description = ""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(
                tmp_dir,
                safe_serialization=safe_serialization,
            )
            folder_url = api.upload_folder(
                repo_id=repo_id,
                folder_path=tmp_dir,
                commit_message=commit_message,
                commit_description=commit_description,
                revision=revision,
                create_pr=create_pr,
            )

        if create_pr:
            return folder_url.pr_url
        return folder_url.commit_url

    def to(self, device: int | str | torch.device | None = None) -> None:
        return self.model.to(device)

    @property
    def _target_device(self) -> torch.device:
        logger.warning(
            "`CrossEncoder._target_device` has been removed, please use `CrossEncoder.device` instead.",
        )
        return self.device

    @_target_device.setter
    def _target_device(self, device: int | str | torch.device | None = None) -> None:
        self.to(device)

    @property
    def device(self) -> torch.device:
        return self.model.device
