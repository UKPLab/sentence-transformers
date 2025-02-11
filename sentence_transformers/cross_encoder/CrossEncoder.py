from __future__ import annotations

import logging
import os
import tempfile
import traceback
from typing import Callable, Literal, overload

import numpy as np
import torch
from huggingface_hub import HfApi
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.utils import PushToHubMixin

from sentence_transformers.cross_encoder.fit_mixin import FitMixin
from sentence_transformers.cross_encoder.model_card import CrossEncoderModelCardData, generate_model_card
from sentence_transformers.util import fullname, get_device_name, import_from_string

logger = logging.getLogger(__name__)


class CrossEncoder(nn.Module, PushToHubMixin, FitMixin):
    """
    A CrossEncoder takes exactly two sentences / texts as input and either predicts
    a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
    on a scale of 0 ... 1.

    It does not yield a sentence embedding and does not work for individual sentences.

    Args:
        model_name (str): A model name from Hugging Face Hub that can be loaded with AutoModel, or a path to a local
            model. We provide several pre-trained CrossEncoder models that can be used for common tasks.
        num_labels (int, optional): Number of labels of the classifier. If 1, the CrossEncoder is a regression model that
            outputs a continuous score 0...1. If > 1, it output several scores that can be soft-maxed to get
            probability scores for the different classes. Defaults to None.
        max_length (int, optional): Max length for input sequences. Longer sequences will be truncated. If None, max
            length of the model will be used. Defaults to None.
        device (str, optional): Device that should be used for the model. If None, it will use CUDA if available.
            Defaults to None.
        automodel_args (Dict, optional): Arguments passed to AutoModelForSequenceClassification. Defaults to None.
        tokenizer_args (Dict, optional): Arguments passed to AutoTokenizer. Defaults to None.
        config_args (Dict, optional): Arguments passed to AutoConfig. Defaults to None.
        cache_dir (`str`, `Path`, optional): Path to the folder where cached files are stored.
        trust_remote_code (bool, optional): Whether or not to allow for custom models defined on the Hub in their own modeling files.
            This option should only be set to True for repositories you trust and in which you have read the code, as it
            will execute code present on the Hub on your local machine. Defaults to False.
        revision (Optional[str], optional): The specific model version to use. It can be a branch name, a tag name, or a commit id,
            for a stored model on Hugging Face. Defaults to None.
        local_files_only (bool, optional): If `True`, avoid downloading the model. Defaults to False.
        default_activation_function (Callable, optional): Callable (like nn.Sigmoid) about the default activation function that
            should be used on-top of model.predict(). If None. nn.Sigmoid() will be used if num_labels=1,
            else nn.Identity(). Defaults to None.
        classifier_dropout (float, optional): The dropout ratio for the classification head. Defaults to None.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = None,
        max_length: int = None,
        device: str | None = None,
        automodel_args: dict = None,
        tokenizer_args: dict = None,
        config_args: dict = None,
        cache_dir: str = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        default_activation_function=None,
        classifier_dropout: float = None,
        model_card_data: CrossEncoderModelCardData | None = None,
    ) -> None:
        super().__init__()
        if tokenizer_args is None:
            tokenizer_args = {}
        if automodel_args is None:
            automodel_args = {}
        if config_args is None:
            config_args = {}
        self.model_card_data = model_card_data or CrossEncoderModelCardData()
        self.trust_remote_code = trust_remote_code
        self.config: PretrainedConfig = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            **config_args,
        )
        classifier_trained = False  # TODO: This is 'breaking'
        if self.config.architectures is not None:
            classifier_trained = any(
                [arch.endswith("ForSequenceClassification") for arch in self.config.architectures]
            )

        if classifier_dropout is not None:
            self.config.classifier_dropout = classifier_dropout

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            **automodel_args,
        )
        if max_length is not None and "model_max_length" not in tokenizer_args:
            tokenizer_args["model_max_length"] = max_length
        if hasattr(self.config, "max_position_embeddings") and "model_max_length" not in tokenizer_args:
            tokenizer_args["model_max_length"] = self.config.max_position_embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            **tokenizer_args,
        )

        if device is None:
            device = get_device_name()
            logger.info(f"Use pytorch device: {device}")
        self.model.to(device)

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = fullname(self.default_activation_function)
            except Exception as e:
                logger.warning(f"Was not able to update config about the default_activation_function: {str(e)}")
        elif (
            hasattr(self.config, "sbert_ce_default_activation_function")
            and self.config.sbert_ce_default_activation_function is not None
        ):
            self.default_activation_function = import_from_string(self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

        # Pass the model to the model card data for later use in generating a model card upon saving this model
        self.model_card_data.register_model(self)
        self.model_card_data.set_base_model(model_name, revision=revision)

    @property
    def num_labels(self) -> int:
        return self.config.num_labels

    @property
    def max_length(self) -> int:
        return self.tokenizer.model_max_length

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @overload
    def predict(
        self,
        sentences: tuple[str, str] | list[str],
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        num_workers: int = ...,
        activation_fct: Callable | None = ...,
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
        num_workers: int = ...,
        activation_fct: Callable | None = ...,
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
        num_workers: int = ...,
        activation_fct: Callable | None = ...,
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
        num_workers: int = ...,
        activation_fct: Callable | None = ...,
        apply_softmax: bool | None = ...,
        convert_to_numpy: Literal[False] = ...,
        convert_to_tensor: Literal[False] = ...,
    ) -> list[torch.Tensor]: ...

    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]] | tuple[str, str] | list[str],
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        num_workers: int = 0,
        activation_fct: Callable | None = None,
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
            num_workers (int, optional): Number of workers for tokenization. Defaults to 0.
            activation_fct (callable, optional): Activation function applied on the logits output of the CrossEncoder.
                If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity. Defaults to None.
            convert_to_numpy (bool, optional): Convert the output to a numpy matrix. Defaults to True.
            apply_softmax (bool, optional): If there are more than 2 dimensions and apply_softmax=True,
                applies softmax on the logits output. Defaults to False.
            convert_to_tensor (bool, optional): Convert the output to a tensor. Defaults to False.

        Returns:
            Union[List[float], np.ndarray, torch.Tensor]: Predictions for the passed sentence pairs.
            The return type depends on the `convert_to_numpy` and `convert_to_tensor` parameters.
            If `convert_to_tensor` is True, the output will be a torch.Tensor.
            If `convert_to_numpy` is True, the output will be a numpy.ndarray.
            Otherwise, the output will be a list of float values.

        Examples:
            ::

                from sentence_transformers import CrossEncoder

                model = CrossEncoder("cross-encoder/stsb-roberta-base")
                sentences = [["I love cats", "Cats are amazing"], ["I prefer dogs", "Dogs are loyal"]]
                model.predict(sentences)
                # => array([0.6912767, 0.4303499], dtype=float32)
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(
            sentences,
            batch_size=batch_size,
            collate_fn=self.smart_batching_collate_text_only,
            num_workers=num_workers,
            shuffle=False,
        )

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().float().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores

    def rank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        return_documents: bool = False,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        num_workers: int = 0,
        activation_fct=None,
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
            num_workers (int, optional): Number of workers for tokenization. Defaults to 0.
            activation_fct ([type], optional): Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity. Defaults to None.
            convert_to_numpy (bool, optional): Convert the output to a numpy matrix. Defaults to True.
            apply_softmax (bool, optional): If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output. Defaults to False.
            convert_to_tensor (bool, optional): Convert the output to a tensor. Defaults to False.

        Returns:
            List[Dict[Literal["corpus_id", "score", "text"], Union[int, float, str]]]: A sorted list with the "corpus_id", "score", and optionally "text" of the documents.

        Example:
            ::

                from sentence_transformers import CrossEncoder
                model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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
        if self.config.num_labels != 1:
            raise ValueError(
                "CrossEncoder.rank() only works for models with num_labels=1. "
                "Consider using CrossEncoder.predict() with input pairs instead."
            )
        query_doc_pairs = [[query, doc] for doc in documents]
        scores = self.predict(
            sentences=query_doc_pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            num_workers=num_workers,
            activation_fct=activation_fct,
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
        # TODO: Introduce this:
        """
        # If we loaded a model from the Hub, and no training was done, then
        # we don't generate a new model card, but reuse the old one instead.
        if self._model_card_text and "generated_from_trainer" not in self.model_card_data.tags:
            model_card = self._model_card_text
            if self.model_card_data.model_id:
                # If the original model card was saved without a model_id, we replace the model_id with the new model_id
                model_card = model_card.replace(
                    'model = SentenceTransformer("sentence_transformers_model_id"',
                    f'model = SentenceTransformer("{self.model_card_data.model_id}"',
                )
        else:
        """
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
