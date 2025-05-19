from __future__ import annotations

import logging
import math
import queue
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from multiprocessing import Queue
from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt
import torch
import torch.multiprocessing as mp
from torch import Tensor, nn
from tqdm import trange
from transformers import AutoConfig, is_torch_npu_available
from typing_extensions import deprecated

from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.sparse_encoder.model_card import SparseEncoderModelCardData
from sentence_transformers.sparse_encoder.models import CSRSparsity, MLMTransformer, SpladePooling
from sentence_transformers.util import batch_to_device, select_max_active_dims

logger = logging.getLogger(__name__)


class SparseEncoder(SentenceTransformer):
    """
    Loads or creates a SparseEncoder model that can be used to map sentences / text to sparse embeddings.

    Args:
        model_name_or_path (str, optional): If it is a filepath on disc, it loads the model from that path. If it is not a path,
            it first tries to download a pre-trained SparseEncoder model. If that fails, tries to construct a model
            from the Hugging Face Hub with that name.
        modules (Iterable[nn.Module], optional): A list of torch Modules that should be called sequentially, can be used to create custom
            SparseEncoder models from scratch.
        device (str, optional): Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU
            can be used.
        prompts (Dict[str, str], optional): A dictionary with prompts for the model. The key is the prompt name, the value is the prompt text.
            The prompt text will be prepended before any text to encode. For example:
            `{"query": "query: ", "passage": "passage: "}` or `{"clustering": "Identify the main category based on the
            titles in "}`.
        default_prompt_name (str, optional): The name of the prompt that should be used by default. If not set,
            no prompt will be applied.
        similarity_fn_name (str or SimilarityFunction, optional): The name of the similarity function to use. Valid options are "cosine", "dot",
            "euclidean", and "manhattan". If not set, it is automatically set to "cosine" if `similarity` or
            `similarity_pairwise` are called while `model.similarity_fn_name` is still `None`.
        cache_folder (str, optional): Path to store models. Can also be set by the SENTENCE_TRANSFORMERS_HOME environment variable.
        trust_remote_code (bool, optional): Whether or not to allow for custom models defined on the Hub in their own modeling files.
            This option should only be set to True for repositories you trust and in which you have read the code, as it
            will execute code present on the Hub on your local machine.
        revision (str, optional): The specific model version to use. It can be a branch name, a tag name, or a commit id,
            for a stored model on Hugging Face.
        local_files_only (bool, optional): Whether or not to only look at local files (i.e., do not try to download the model).
        token (bool or str, optional): Hugging Face authentication token to download private models.
        use_auth_token (bool or str, optional): Deprecated argument. Please use `token` instead.
        max_active_dims (int, optional): The maximum number of active (non-zero) dimensions in the output of the model. Defaults to None. This means there will be no
            limit on the number of active dimensions and can be slow or memory-intensive if your model wasn't (yet) finetuned to high sparsity.
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
            - ``provider``: If backend is "onnx", this is the provider to use for inference, for example "CPUExecutionProvider",
              "CUDAExecutionProvider", etc. See https://onnxruntime.ai/docs/execution-providers/ for all ONNX execution providers.
            - ``file_name``: If backend is "onnx" or "openvino", this is the file name to load, useful for loading optimized
              or quantized ONNX or OpenVINO models.
            - ``export``: If backend is "onnx" or "openvino", then this is a boolean flag specifying whether this model should
              be exported to the backend. If not specified, the model will be exported only if the model repository or directory
              does not already contain an exported model.

            See the `PreTrainedModel.from_pretrained
            <https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_
            documentation for more details.
        tokenizer_kwargs (Dict[str, Any], optional): Additional tokenizer configuration parameters to be passed to the Hugging Face Transformers tokenizer.
            See the `AutoTokenizer.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained>`_
            documentation for more details.
        config_kwargs (Dict[str, Any], optional): Additional model configuration parameters to be passed to the Hugging Face Transformers config.
            See the `AutoConfig.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained>`_
            documentation for more details.
        model_card_data (:class:`~sentence_transformers.sparse_encoder.model_card.SparseEncoderModelCardData`, optional): A model
            card data object that contains information about the model. This is used to generate a model card when saving
            the model. If not set, a default model card data object is created.
        backend (str): The backend to use for inference. Can be one of "torch" (default), "onnx", or "openvino".
            See https://sbert.net/docs/sentence_transformer/usage/efficiency.html for benchmarking information
            on the different backends.

    Example:
        ::

            from sentence_transformers import SparseEncoder

            # Load a pre-trained SparseEncoder model
            model = SparseEncoder('naver/splade-cocondenser-ensembledistil')

            # Encode some texts
            sentences = [
                "The weather is lovely today.",
                "It's so sunny outside!",
                "He drove to the stadium.",
            ]
            embeddings = model.encode(sentences)
            print(embeddings.shape)
            # (3, 30522)

            # Get the similarity scores between all sentences
            similarities = model.similarity(embeddings, embeddings)
            print(similarities)
            # tensor([[35.629, 9.1541, 0.11269],
            #         [9.1541, 27.478, 0.019061],
            #         [0.11269, 0.019061, 29.612]])
    """

    model_card_data_class = SparseEncoderModelCardData

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
        max_active_dims: int | None = None,
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
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            model_card_data=model_card_data,
            backend=backend,
        )
        self.max_active_dims = max_active_dims

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
        save_to_cpu: bool = False,
        device: str | None = None,
        max_active_dims: int | None = None,
        **kwargs: Any,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]:
        """
        Computes sparse sentence embeddings.

        Args:
            sentences (Union[str, List[str]]): The sentences to embed.
            prompt_name (Optional[str], optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
                which is either set in the constructor or loaded from the model configuration. For example if
                ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
            prompt (Optional[str], optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
            batch_size (int, optional): The batch size used for the computation. Defaults to 32.
            show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to None.
            output_value (Optional[Literal["sentence_embedding", "token_embeddings"]], optional): The type of embeddings to return:
                "sentence_embedding" to get sentence embeddings, "token_embeddings" to get wordpiece token embeddings, and `None`,
                to get all output values. Defaults to "sentence_embedding".
            convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
                Defaults to False.
            convert_to_sparse_tensor (bool, optional): Whether the output should be in the format of a sparse tensor.
                Defaults to True.
            save_to_cpu (bool, optional):  Whether the output should be moved to cpu or stay on the device it has been computed on.
                Defaults to False
            device (str, optional): Which :class:`torch.device` to use for the computation, if None current device will be use. Defaults to None.
            max_active_dims (int, optional): The maximum number of active (non-zero) dimensions in the output of the model. `None` means we will
                used the value of the model's config. Defaults to None. If None in model's config it means there will be no limit on the number
                of active dimensions and can be slow or memory-intensive if your model wasn't (yet) finetuned to high sparsity.

        Returns:
            Union[List[Tensor], ndarray, Tensor]: By default, a 2d torch sparse tensor with shape [num_inputs, output_dimension] is returned.
            If only one string input is provided, then the output is a 1d array with shape [output_dimension]. If save_to_cpu is True,
            the embeddings are moved to the CPU.

        Example:
            ::

                from sentence_transformers import SparseEncoder

                # Load a pre-trained SparseEncoder model
                model = SparseEncoder('naver/splade-cocondenser-ensembledistil')

                # Encode some texts
                sentences = [
                    "The weather is lovely today.",
                    "It's so sunny outside!",
                    "He drove to the stadium.",
                ]
                embeddings = model.encode(sentences)
                print(embeddings.shape)
                # (3, 30522)
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (
                logging.INFO,
                logging.DEBUG,
            )

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

        max_active_dims = max_active_dims if max_active_dims is not None else self.max_active_dims
        if max_active_dims is not None:
            kwargs["max_active_dims"] = max_active_dims

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

                if max_active_dims:
                    out_features["sentence_embedding"] = select_max_active_dims(
                        out_features["sentence_embedding"], max_active_dims=max_active_dims
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
                    if save_to_cpu:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if not convert_to_tensor:
            return all_embeddings

        all_embeddings = torch.stack(all_embeddings)

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    @property
    def similarity_fn_name(self) -> Literal["cosine", "dot", "euclidean", "manhattan"]:
        """Return the name of the similarity function used by :meth:`SparseEncoder.similarity` and :meth:`SparseEncoder.similarity_pairwise`.

        Returns:
            Optional[str]: The name of the similarity function. Can be None if not set, in which case it will
                default to "cosine" when first called.

        Example:
            >>> model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
            >>> model.similarity_fn_name
            'dot'
        """
        if self._similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.DOT
        return self._similarity_fn_name

    @similarity_fn_name.setter
    def similarity_fn_name(
        self,
        value: Literal["cosine", "dot", "euclidean", "manhattan"] | SimilarityFunction,
    ) -> None:
        if isinstance(value, SimilarityFunction):
            value = value.value
        self._similarity_fn_name = value

        if value is not None:
            self._similarity = SimilarityFunction.to_similarity_fn(value)
            self._similarity_pairwise = SimilarityFunction.to_similarity_pairwise_fn(value)

    @property
    def similarity(self) -> Callable[[Tensor | npt.NDArray[np.float32], Tensor | npt.NDArray[np.float32]], Tensor]:
        """
        Compute the similarity between two collections of embeddings. The output will be a matrix with the similarity
        scores between all embeddings from the first parameter and all embeddings from the second parameter. This
        differs from `similarity_pairwise` which computes the similarity between each pair of embeddings.
        This method supports only embeddings with fp32 precision and does not accommodate quantized embeddings.

        Args:
            embeddings1 (Union[Tensor, ndarray]): [num_embeddings_1, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.
            embeddings2 (Union[Tensor, ndarray]): [num_embeddings_2, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.

        Returns:
            Tensor: A [num_embeddings_1, num_embeddings_2]-shaped torch tensor with similarity scores.

        Example:
            ::

                >>> model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                >>> sentences = [
                ...     "The weather is so nice!",
                ...     "It's so sunny outside.",
                ...     "He's driving to the movie theater.",
                ...     "She's going to the cinema.",
                ... ]
                >>> embeddings = model.encode(sentences, normalize_embeddings=True)
                >>> model.similarity(embeddings, embeddings)
                tensor([[   30.9529,    12.9143,     0.0000,     0.0106],
                        [   12.9143,    27.5256,     0.5829,     0.5958],
                        [    0.0000,     0.5829,    36.0683,    15.3007],
                        [    0.0106,     0.5958,    15.3007,    39.4664]], device='cuda:0')
                >>> model.similarity_fn_name
                "dot"
                >>> model.similarity_fn_name = "cosine"
                >>> model.similarity(embeddings, embeddings)
                tensor([[    1.0000,     0.4424,     0.0000,     0.0003],
                        [    0.4424,     1.0000,     0.0185,     0.0181],
                        [    0.0000,     0.0185,     1.0000,     0.4055],
                        [    0.0003,     0.0181,     0.4055,     1.0000]], device='cuda:0')
        """
        if self.similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.DOT
        return self._similarity

    @property
    def similarity_pairwise(
        self,
    ) -> Callable[[Tensor | npt.NDArray[np.float32], Tensor | npt.NDArray[np.float32]], Tensor]:
        """
        Compute the similarity between two collections of embeddings. The output will be a vector with the similarity
        scores between each pair of embeddings.
        This method supports only embeddings with fp32 precision and does not accommodate quantized embeddings.

        Args:
            embeddings1 (Union[Tensor, ndarray]): [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.
            embeddings2 (Union[Tensor, ndarray]): [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.

        Returns:
            Tensor: A [num_embeddings]-shaped torch tensor with pairwise similarity scores.

        Example:
            ::

                >>> model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                >>> sentences = [
                ...     "The weather is so nice!",
                ...     "It's so sunny outside.",
                ...     "He's driving to the movie theater.",
                ...     "She's going to the cinema.",
                ... ]
                >>> embeddings = model.encode(sentences, convert_to_sparse_tensor=False)
                >>> model.similarity_pairwise(embeddings[::2], embeddings[1::2])
                tensor([12.9143, 15.3007], device='cuda:0')
                >>> model.similarity_fn_name
                "dot"
                >>> model.similarity_fn_name = "cosine"
                >>> model.similarity_pairwise(embeddings[::2], embeddings[1::2])
                tensor([0.4424, 0.4055], device='cuda:0')
        """
        if self.similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.DOT
        return self._similarity_pairwise

    def start_multi_process_pool(
        self, target_devices: list[str] = None
    ) -> dict[Literal["input", "output", "processes"], Any]:
        """
        Starts a multi-process pool to process the encoding with several independent processes
        via :meth:`SparseEncoder.encode_multi_process <sentence_transformers.sparse_encoder.SparseEncoder.encode_multi_process>`.

        This method is recommended if you want to encode on multiple GPUs or CPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        and stop_multi_process_pool.

        Args:
            target_devices (List[str], optional): PyTorch target devices, e.g. ["cuda:0", "cuda:1", ...],
                ["npu:0", "npu:1", ...], or ["cpu", "cpu", "cpu", "cpu"]. If target_devices is None and CUDA/NPU
                is available, then all available CUDA/NPU devices will be used. If target_devices is None and
                CUDA/NPU is not available, then 4 CPU devices will be used.

        Returns:
            Dict[str, Any]: A dictionary with the target processes, an input queue, and an output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            elif is_torch_npu_available():
                target_devices = [f"npu:{i}" for i in range(torch.npu.device_count())]
            else:
                logger.info("CUDA/NPU is not available. Starting 4 CPU workers")
                target_devices = ["cpu"] * 4

        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices))))

        self.to("cpu")
        self.share_memory()
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device_id in target_devices:
            p = ctx.Process(
                target=SparseEncoder._encode_multi_process_worker,
                args=(device_id, self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    def encode_multi_process(
        self,
        sentences: list[str],
        pool: dict[Literal["input", "output", "processes"], Any],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        chunk_size: int = None,
        show_progress_bar: bool | None = None,
        max_active_dims: int | None = None,
    ) -> Tensor:
        """
        Encodes a list of sentences using multiple processes and GPUs via
        :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.SparseEncoder.encode>`.
        The sentences are chunked into smaller packages and sent to individual processes, which encode them on different
        GPUs or CPUs. This method is only suitable for encoding large sets of sentences.

        Args:
            sentences (List[str]): List of sentences to encode.
            pool (Dict[Literal["input", "output", "processes"], Any]): A pool of workers started with
                :meth:`SparseEncoder.start_multi_process_pool <sentence_transformers.sparse_encoder.SparseEncoder.start_multi_process_pool>`.
            prompt_name (Optional[str], optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
                which is either set in the constructor or loaded from the model configuration. For example if
                ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
            prompt (Optional[str], optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
            batch_size (int): Encode sentences with batch size. (default: 32)
            chunk_size (int): Sentences are chunked and sent to the individual processes. If None, it determines a
                sensible size. Defaults to None.
            show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to None.
            max_active_dims (int, optional): The maximum number of active (non-zero) dimensions in the output of the model. `None` means we will
                used the value of the model's config. Defaults to None. If None in model's config it means there will be no limit on the number
                of active dimensions and can be slow or memory-intensive if your model wasn't (yet) finetuned to high sparsity.

        Returns:
            Tensor: A 2D tensor with shape [num_inputs, output_dimension].

        Example:
            ::

                from sentence_transformers import SparseEncoder

                def main():
                    model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                    sentences = [
                        "The weather is so nice!",
                        "It's so sunny outside.",
                        "He's driving to the movie theater.",
                        "She's going to the cinema.",
                    ] * 1000

                    pool = model.start_multi_process_pool()
                    embeddings = model.encode_multi_process(sentences, pool)
                    model.stop_multi_process_pool(pool)

                    print(embeddings.shape)
                    # => (4000, 30522)


                if __name__ == "__main__":
                    main()
        """

        if chunk_size is None:
            chunk_size = min(math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000)

        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)

        logger.debug(f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}")

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, batch_size, chunk, prompt_name, prompt, max_active_dims])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk, prompt_name, prompt, max_active_dims])
            last_chunk_id += 1

        output_queue = pool["output"]
        results_list = sorted(
            [output_queue.get() for _ in trange(last_chunk_id, desc="Chunks", disable=not show_progress_bar)],
            key=lambda x: x[0],
        )
        embeddings = torch.concatenate([result[1] for result in results_list])
        return embeddings

    @staticmethod
    def _encode_multi_process_worker(
        target_device: str, model: SparseEncoder, input_queue: Queue, results_queue: Queue
    ) -> None:
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                chunk_id, batch_size, sentences, prompt_name, prompt, max_active_dims = input_queue.get()
                embeddings = model.encode(
                    sentences,
                    prompt_name=prompt_name,
                    prompt=prompt,
                    device=target_device,
                    show_progress_bar=False,
                    batch_size=batch_size,
                    convert_to_sparse_tensor=True,
                    save_to_cpu=True,
                    max_active_dims=max_active_dims,
                )

                results_queue.put([chunk_id, embeddings])
            except queue.Empty:
                break

    def get_sentence_embedding_dimension(self) -> int | None:
        """
        Returns the number of dimensions in the output of :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.SparseEncoder.encode>`.
        We override the function without updating regarding the truncate dim as for sparse model the dimension of the output
        is the same, only the active dimensions number changes.

        Returns:
            Optional[int]: The number of dimensions in the output of `encode`. If it's not known, it's `None`.
        """
        output_dim = None
        for mod in reversed(self._modules.values()):
            sent_embedding_dim_method = getattr(mod, "get_sentence_embedding_dimension", None)
            if callable(sent_embedding_dim_method):
                output_dim = sent_embedding_dim_method()
                break
        return output_dim

    @contextmanager
    def truncate_sentence_embeddings(self, truncate_dim: int | None) -> Iterator[None]:
        raise NotImplementedError(
            "SparseEncoder does not support truncating sentence embeddings. "
            "Use the `max_active_dims` parameter in the encode method instead if you want to limit the embedding memory usage."
        )

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
        has_modules: bool = False,
    ) -> list[nn.Module]:
        """
        Creates a simple transformer-based model and returns the modules.
        For MLMTransformer (models ending with ForMaskedLM), uses SpladePooling with 'max' strategy.
        For regular Transformer, uses CSR implementation by default.

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
            has_modules (bool, optional): Whether the model has modules.json. Defaults to False.

        Returns:
            List[nn.Module]: A list containing the transformer model and the pooling model.
        """
        logger.warning(
            f"No sparse-encoder model found with name {model_name_or_path}. Creating a new one with defaults settings compatible to the base model."
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

        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_folder, **config_kwargs)

        # Check if the architecture ends with "ForMaskedLM"
        is_mlm_model = False
        if hasattr(config, "architectures") and config.architectures:
            for architecture in config.architectures:
                if architecture.endswith("ForMaskedLM"):
                    is_mlm_model = True
                    break
        if has_modules:
            logger.info(
                "A SentenceTransformer model found, using Sentence Transformer modules with CSR sparsity modules on Top"
            )
            modules, self.module_kwargs = self._load_sbert_model(
                model_name_or_path,
                token=token,
                cache_folder=cache_folder,
                revision=revision,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
                config_kwargs=config_kwargs,
            )
            modules = [modules[str(i)] for i in range(len(modules.keys()))]
            csr_sparsity = CSRSparsity(
                input_dim=modules[0].get_word_embedding_dimension(),
                hidden_dim=4 * modules[0].get_word_embedding_dimension(),
                k=256,  # Number of top values to keep
                k_aux=512,  # Number of top values for auxiliary loss
            )
            modules.append(csr_sparsity)

        elif is_mlm_model:
            # For MLM models like BERT, RoBERTa, etc., use MLMTransformer with SpladePooling
            logger.info(f"Detected MLM architecture: {config.architectures}, using SpladePooling")
            transformer_model = MLMTransformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args=model_kwargs,
                tokenizer_args=tokenizer_kwargs,
                config_args=config_kwargs,
                backend=self.backend,
            )
            pooling_model = SpladePooling(pooling_strategy="max")

            modules = [transformer_model, pooling_model]

        else:
            logger.info(
                "No MLM model found and no SentenceTransformer model found, using default transformer modules and mean pooling with CSR sparsity modules on Top"
            )
            transformer_model = Transformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args=model_kwargs,
                tokenizer_args=tokenizer_kwargs,
                config_args=config_kwargs,
                backend=self.backend,
            )
            pooling = Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode="mean")
            csr_sparsity = CSRSparsity(
                input_dim=pooling.get_sentence_embedding_dimension(),
                hidden_dim=4 * pooling.get_sentence_embedding_dimension(),
                k=256,  # Number of top values to keep
                k_aux=512,  # Number of top values for auxiliary loss
            )
            modules = [transformer_model, pooling, csr_sparsity]

        if not local_files_only:
            self.model_card_data.set_base_model(model_name_or_path, revision=revision)
        return modules

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
    @deprecated("SparseEncoder.load(...) is deprecated, use SparseEncoder(...) instead.")
    def load(input_path) -> SparseEncoder:
        return SparseEncoder(input_path)

    @staticmethod
    def get_sparsity_stats(embeddings: torch.Tensor) -> dict[str, float]:
        """
        Calculate row-wise sparsity statistics for the given embeddings.

        Args:
            embeddings (torch.Tensor): The embeddings to analyze (2D tensor expected).

        Returns:
            dict[str, float]: Dictionary with row-wise sparsity statistics (mean and std).
                            Includes 'num_rows', 'num_cols', 'row_non_zero_mean', 'row_sparsity_mean',.
        """
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("Embeddings must be a torch.Tensor")
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D tensor, but got {embeddings.ndim} dimensions")

        num_rows, num_cols = embeddings.shape

        if num_rows == 0:
            # Handle empty tensor case
            return {
                "num_rows": 0,
                "num_cols": num_cols,
                "row_non_zero_mean": float("nan"),
                "row_sparsity_mean": float("nan"),
            }

        if embeddings.is_sparse or embeddings.is_sparse_csr:
            if embeddings.layout == torch.sparse_coo:
                embeddings = embeddings.to_sparse_csr()  # Convert to CSR for easier row-wise ops

            # is_sparse_csr path
            indptr = embeddings.crow_indices()
            non_zero_per_row = indptr[1:] - indptr[:-1]

        else:  # Dense tensor
            non_zero_per_row = torch.count_nonzero(embeddings, dim=1)

        if num_cols == 0:
            # Handle case with zero columns (all rows are empty)
            density_per_row = torch.zeros(num_rows, device=embeddings.device, dtype=torch.float32)
        else:
            density_per_row = non_zero_per_row.float() / num_cols
        sparsity_per_row = 1.0 - density_per_row

        # Use torch.nanmean and torch.nanstd if NaN values are possible and should be ignored,
        # but standard mean/std should be fine if inputs are handled (e.g. num_cols > 0).
        # Calculate std only if num_rows > 1 to avoid NaN/errors.
        results = {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "row_non_zero_mean": torch.mean(non_zero_per_row.float()).item(),
            "row_sparsity_mean": torch.mean(sparsity_per_row).item(),
        }
        return results

    @property
    def max_seq_length(self) -> int:
        """
        Returns the maximal input sequence length for the model. Longer inputs will be truncated.

        Returns:
            int: The maximal input sequence length.

        Example:
            ::

                from sentence_transformers import SparseEncoder

                model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                print(model.max_seq_length)
                # => 512
        """
        return super().max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value) -> None:
        """
        Property to set the maximal input sequence length for the model. Longer inputs will be truncated.
        """
        self._first_module().max_seq_length = value

    def intersection(
        self,
        embeddings_1: torch.Tensor,
        embeddings_2: torch.Tensor,
    ):
        """
        Compute the intersection of two sparse embeddings.

        Args:
            embeddings_1 (torch.Tensor): First embedding tensor (vocab).
            embeddings_2 (torch.Tensor): Second embedding tensor (batch_2, vocab).

        Returns:
            torch.Tensor: Intersection of the two embeddings.
        """
        if not embeddings_1.is_sparse:
            embeddings_1 = embeddings_1.to_sparse()
        if not embeddings_2.is_sparse:
            embeddings_2 = embeddings_2.to_sparse()

        if embeddings_1.ndim != 1:
            raise ValueError(f"Expected 1D tensor for embeddings_1, but got {embeddings_1.shape} shape.")

        if embeddings_2.ndim == 1:
            intersection = embeddings_1 * embeddings_2
        elif embeddings_2.ndim == 2:
            intersection = torch.stack([embeddings_1 * embedding for embedding in embeddings_2])
        else:
            raise ValueError(f"Expected 1D tensor or 2D tensor for embeddings_2, but got {embeddings_2.shape} shape.")

        # Cheaply remove zero values
        intersection = intersection.coalesce()
        active_dims = intersection.values() > 0
        intersection = torch.sparse_coo_tensor(
            intersection.indices()[:, active_dims],
            intersection.values()[active_dims],
            size=intersection.size(),
            device=intersection.device,
        )

        return intersection

    def decode(self, embeddings: torch.Tensor, top_k: int = 10) -> dict:
        """
        Decode top K tokens and weights from a sparse embedding.

        Args:
            embeddings (torch.Tensor): Sparse embedding tensor (batch, vocab).
            top_k (int): Number of top tokens to return.

        Returns:
            dict: Dictionary with decoded tokens and weights.
        """
        if not embeddings.is_sparse:
            embeddings = embeddings.to_sparse()

        tokenizer = self.tokenizer

        if embeddings.dim() == 2:
            return [self.decode(embeddings[i], top_k=top_k) for i in range(embeddings.size(0))]
        elif embeddings.dim() == 1:
            if embeddings.is_sparse or getattr(embeddings, "is_sparse_csr", False):
                embeddings = embeddings.coalesce() if embeddings.is_sparse else embeddings
                values = embeddings.values()
                indices = embeddings.indices().squeeze()
                if values.numel() == 0:
                    return []
                topk = min(top_k, indices.numel())
                top_values, top_idx = torch.topk(values, topk)
                # Convert indices to tokens
                top_tokens = tokenizer.convert_ids_to_tokens(indices[top_idx].tolist())
                return list(zip(top_tokens, top_values.tolist()))
            else:
                top_values, top_indices = torch.topk(embeddings, top_k)
                top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
                return list(zip(top_tokens, top_values.tolist()))
        else:
            raise ValueError("Input tensor must be 1D or 2D.")
