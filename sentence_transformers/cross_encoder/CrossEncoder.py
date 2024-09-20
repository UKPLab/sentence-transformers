from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Callable, Literal, overload

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, is_torch_npu_available
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import PushToHubMixin

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import fullname, get_device_name, import_from_string

logger = logging.getLogger(__name__)


class CrossEncoder(PushToHubMixin):
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
    ) -> None:
        if tokenizer_args is None:
            tokenizer_args = {}
        if automodel_args is None:
            automodel_args = {}
        if config_args is None:
            config_args = {}
        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            **config_args,
        )
        classifier_trained = True
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
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            **automodel_args,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            **tokenizer_args,
        )
        self.max_length = max_length

        if device is None:
            device = get_device_name()
            logger.info(f"Use pytorch device: {device}")

        self._target_device = torch.device(device)

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

    def smart_batching_collate(self, batch: list[InputExample]) -> tuple[BatchEncoding, Tensor]:
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

        tokenized = self.tokenizer(
            *texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.max_length
        )
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(
            self._target_device
        )

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def smart_batching_collate_text_only(self, batch: list[InputExample]) -> BatchEncoding:
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(
            *texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.max_length
        )

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized

    def fit(
        self,
        train_dataloader: DataLoader,
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        loss_fct=None,
        activation_fct=nn.Identity(),
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: type[Optimizer] = torch.optim.AdamW,
        optimizer_params: dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
    ) -> None:
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        Args:
            train_dataloader (DataLoader): DataLoader with training InputExamples
            evaluator (SentenceEvaluator, optional): An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc. Defaults to None.
            epochs (int, optional): Number of epochs for training. Defaults to 1.
            loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss(). Defaults to None.
            activation_fct: Activation function applied on top of logits output of model.
            scheduler (str, optional): Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts. Defaults to "WarmupLinear".
            warmup_steps (int, optional): Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero. Defaults to 10000.
            optimizer_class (Type[Optimizer], optional): Optimizer. Defaults to torch.optim.AdamW.
            optimizer_params (Dict[str, object], optional): Optimizer parameters. Defaults to {"lr": 2e-5}.
            weight_decay (float, optional): Weight decay for model parameters. Defaults to 0.01.
            evaluation_steps (int, optional): If > 0, evaluate the model using evaluator after each number of training steps. Defaults to 0.
            output_path (str, optional): Storage path for the model and evaluation files. Defaults to None.
            save_best_model (bool, optional): If true, the best model (according to evaluator) is stored at output_path. Defaults to True.
            max_grad_norm (float, optional): Used for gradient normalization. Defaults to 1.
            use_amp (bool, optional): Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0. Defaults to False.
            callback (Callable[[float, int, int], None], optional): Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`. Defaults to None.
            show_progress_bar (bool, optional): If True, output a tqdm progress bar. Defaults to True.
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            if is_torch_npu_available():
                scaler = torch.npu.amp.GradScaler()
            else:
                scaler = torch.cuda.amp.GradScaler()
        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
            )

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(
                train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar
            ):
                if use_amp:
                    with torch.autocast(device_type=self._target_device.type):
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(
                        evaluator, output_path, save_best_model, epoch, training_steps, callback
                    )

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

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
        self.model.to(self._target_device)
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
        query_doc_pairs = [[query, doc] for doc in documents]
        scores = self.predict(
            query_doc_pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            num_workers=num_workers,
            activation_fct=activation_fct,
            apply_softmax=apply_softmax,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
        )

        results = []
        for i in range(len(scores)):
            if return_documents:
                results.append({"corpus_id": i, "score": scores[i], "text": documents[i]})
            else:
                results.append({"corpus_id": i, "score": scores[i]})

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback) -> None:
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def save(self, path: str, *, safe_serialization: bool = True, **kwargs) -> None:
        """
        Saves the model and tokenizer to path; identical to `save_pretrained`
        """
        if path is None:
            return

        logger.info(f"Save model to {path}")
        self.model.save_pretrained(path, safe_serialization=safe_serialization, **kwargs)
        self.tokenizer.save_pretrained(path, **kwargs)

    def save_pretrained(self, path: str, *, safe_serialization: bool = True, **kwargs) -> None:
        """
        Saves the model and tokenizer to path; identical to `save`
        """
        return self.save(path, safe_serialization=safe_serialization, **kwargs)

    @wraps(PushToHubMixin.push_to_hub)
    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: str | None = None,
        private: bool | None = None,
        safe_serialization: bool = True,
        tags: list[str] | None = None,
        **kwargs,
    ) -> str:
        if isinstance(tags, str):
            tags = [tags]
        elif tags is None:
            tags = []
        if "cross-encoder" not in tags:
            tags.insert(0, "cross-encoder")
        return super().push_to_hub(
            repo_id=repo_id,
            safe_serialization=safe_serialization,
            commit_message=commit_message,
            private=private,
            tags=tags,
            **kwargs,
        )
