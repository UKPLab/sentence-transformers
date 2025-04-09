from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor
from tqdm import trange

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.sparse_encoder.models import CSRSparsity
from sentence_transformers.util import batch_to_device

logger = logging.getLogger(__name__)


class SparseEncoder(SentenceTransformer):
    """
    A specialized SentenceTransformer model that produces sparse embeddings.
    This class extends SentenceTransformer to create sparse representation of text.
    Sparse embeddings are a type of representation where most of the values are zero,
    and only a few values are non-zero. They are useful for efficient similarity search, reduced memory usage,
    and can improve performance in certain information retrieval tasks.

    Args:
        model_name_or_path (str, optional): If it is a filepath on disc, it loads the model from that path. If it is not a path,
            it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model
            from the Hugging Face Hub with that name.
        modules (Iterable[nn.Module], optional): A list of torch Modules that should be called sequentially, can be used to create custom
            SentenceTransformer models from scratch.
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
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to. `None` does no truncation. Truncation is
            only applicable during inference when :meth:`SentenceTransformer.encode` is called.
        sparsity_threshold (float, optional): Default threshold for considering values as non-zero. Defaults to 0.0.
        topk (int, optional): Default number of top-k elements to keep in each embedding. Defaults to 0.
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
        model_card_data (:class:`~sentence_transformers.model_card.SentenceTransformerModelCardData`, optional): A model
            card data object that contains information about the model. This is used to generate a model card when saving
            the model. If not set, a default model card data object is created.
        backend (str): The backend to use for inference. Can be one of "torch" (default), "onnx", or "openvino".
            See https://sbert.net/docs/sentence_transformer/usage/efficiency.html for benchmarking information
            on the different backends.


    Example:
        ::

            from sentence_transformers import SentenceTransformer

            # Load a pre-trained SentenceTransformer model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Encode some texts
            sentences = [
                "The weather is lovely today.",
                "It's so sunny outside!",
                "He drove to the stadium.",
            ]
            embeddings = model.encode(sentences)
            print(embeddings.shape)
            # (3, 768)

            # Get the similarity scores between all sentences
            similarities = model.similarity(embeddings, embeddings)
            print(similarities)
            # tensor([[1.0000, 0.6817, 0.0492],
            #         [0.6817, 1.0000, 0.0421],
            #         [0.0492, 0.0421, 1.0000]])
    """

    # TODO: Check if there is no other things we need to overwrite espacially for models specificty
    # def __init__(
    #     self,
    #     model_name_or_path: str | None,
    #     *args: Any,
    #     sparsity_threshold: float | None = 0.0,
    #     topk: int | None = 0,
    #     **kwargs: Any,
    # ) -> None:
    #     """Initialize the SparseEncoder with sparsity parameters."""
    #     super().__init__(model_name_or_path, *args, **kwargs)
    #     self.sparsity_threshold = sparsity_threshold
    #     self.topk = topk

    #     logger.info(f"Initialized SparseEncoder with threshold={sparsity_threshold}, topk={topk}")

    # TODO: Look at the overload ? Add handling for dict and list[dict]
    def encode(
        self,
        sentences: str | list[str] | np.ndarray,
        # prompt_name: str | None = None,
        # prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        # output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
        # precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        # convert_to_numpy: bool = True,
        # convert_to_tensor: bool = False,
        # device: str | None = None,
        # normalize_embeddings: bool = False,
        convert_to_sparse_tensor: bool = True,
        # sparsity_threshold: float | None = None,
        # topk: int = None,
        **kwargs: Any,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]:
        """
        Computes sentence embeddings with possible sparsity in the output.

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
            precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.
                Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions are quantized embeddings.
                Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy. They are useful for
                reducing the size of the embeddings of a corpus for semantic search, among other tasks. Defaults to "float32".
            convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
                Defaults to False.
            convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
                Defaults to True.
            convert_to_sparse_tensor (bool, optional): Whether the output should be a sparse tensor. Overwrites `convert_to_numpy` and `convert_to_tensor`.
                Defaults to True.
            device (str, optional): Which :class:`torch.device` to use for the computation. Defaults to None.
            normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.
            sparsity_threshold (float, optional): The threshold for considering values as non-zero. Values below this threshold will be set to zero.
            topk (int, optional): The number of top-k elements to keep in each embedding. If set, the rest will be set to zero.
        Returns:
            Union[List[torch.Tensor], np.ndarray, torch.Tensor, Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
                The sentence embeddings. If `convert_to_numpy` is True, it is a list of numpy vectors.
                If `convert_to_tensor` is True, it is a single large tensor.
                If `convert_to_sparse_tensor` is True, it is a sparse tensor.

        Example:
            # TODO: Add example usage
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (
                logging.INFO,
                logging.DEBUG,
            )

        all_embeddings = []
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, self.device)

            with torch.no_grad():
                out_features = self.forward(features, **kwargs)
                embeddings = out_features["sparse_embedding"]
                if convert_to_sparse_tensor:
                    embeddings = embeddings.to_sparse()
                all_embeddings.extend(embeddings)

        all_embeddings = torch.stack(all_embeddings)

        return all_embeddings

    # TODO: Check but forward, similarity_fn_name, similarity, similarity_pairwise shouldn't be overwritten

    def set_sparsity_threshold(self, threshold: float) -> None:
        """
        Set the sparsity threshold for the encoder.

        Args:
            threshold (float): The threshold value. Values below this threshold will be set to zero.
        """
        # Will be useful later for splade models
        # if threshold < 0:
        #     raise ValueError("Sparsity threshold must be non-negative")
        # # self.sparsity_threshold = threshold
        # # logger.info(f"Set sparsity threshold to {threshold}")
        # for module in self:
        #     if isinstance(module, CSRSparsity):
        #         module.sparsity_threshold = threshold
        #         logger.info(f"Set sparsity threshold to {threshold} in {module.__class__.__name__}")
        #         break

    def set_topk(self, topk: int) -> None:
        """
        Set the number of top-k elements to keep in the sparse representation.

        Args:
            topk (int): The number of top-k elements to keep. Set to None to disable top-k filtering.
        """
        if topk < 0:
            raise ValueError("Top-k must be an integer")
        for module in self:
            if isinstance(module, CSRSparsity):
                module.topk = topk
                logger.info(f"Set topk to {topk} in {module.__class__.__name__}")
                break

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

        return {
            "sparsity": sparsity,
            "density": density,
            "non_zero_count": non_zero,
            "total_elements": total_elements,
        }

    # TODO : Probably gonna need to overwrite the multiprocess functions, such as save, load, evaluate, etc.


if __name__ == "__main__":
    # Small test on a ST model to check if the class works
    # Load a pre-trained SentenceTransformer model
    transformer = Transformer("sentence-transformers/all-mpnet-base-v2")
    pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    csr_sparsity = CSRSparsity(
        input_dim=transformer.get_word_embedding_dimension(),
        hidden_dim=4 * transformer.get_word_embedding_dimension(),
        k=16,  # Number of top values to keep
        k_aux=512,  # Number of top values for auxiliary loss
    )

    # Create the SparseEncoder model
    model = SparseEncoder(modules=[transformer, pooling, csr_sparsity])
    # NOTE: We can (somehow, not sure yet) update `model.module_kwargs` with `CSRSparsity` automatically.
    # See https://sbert.net/docs/sentence_transformer/usage/custom_models.html#advanced-keyword-argument-passthrough-in-custom-modules
    # In short, this means that we can use model.encode(..., topk=16) and it will be passed to the CSRSparsity module.

    # Encode some texts
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    embeddings = model.encode(sentences, convert_to_sparse_tensor=True)
    print(embeddings.shape)
    # (3, 4*768))

    # Get sparsity statistics
    stats = model.get_sparsity_stats(embeddings)
    print("Sparsity statistics:", stats)

    # Test similarity computation
    similarities = model.similarity(embeddings, embeddings)
    print("Similarity matrix:\n", similarities)

    breakpoint()
    """
        # For later:
    from datasets import Dataset

    from sentence_transformers.sparse_encoder.losses import CSRLoss
    from sentence_transformers.sparse_encoder.trainer import SparseEncoderTrainer
    from sentence_transformers.sparse_encoder.training_args import SparseEncoderTrainingArguments

    student_model = SparseEncoder("microsoft/mpnet-base")
    teacher_model = SparseEncoder("all-mpnet-base-v2")
    train_dataset = Dataset.from_dict(
        {
            "query": ["It's nice weather outside today.", "He drove to work."],
            "passage1": ["It's so sunny.", "He took the car to work."],
            "passage2": ["It's very sunny.", "She walked to the store."],
        })

    # Initialize training arguments
    training_args = SparseEncoderTrainingArguments(
        load_best_model_at_end=True,
        topk=16
    )
    def compute_labels(batch):
        emb_queries = teacher_model.encode(batch["query"], convert_to_sparse_tensor=True, topk=0)
        emb_passages1 = teacher_model.encode(batch["passage1"], convert_to_sparse_tensor=True, topk=0)
        emb_passages2 = teacher_model.encode(batch["passage2"], convert_to_sparse_tensor=True, topk=0)
        return {
            "label": teacher_model.similarity_pairwise(emb_queries, emb_passages1)
            - teacher_model.similarity_pairwise(emb_queries, emb_passages2)
        }

    train_dataset = train_dataset.map(compute_labels, batched=True)
    # In this example, the labels become -0.036 and 0.68, respectively
    loss = losses.MarginMSELoss(student_model)

    trainer = SentenceTransformerTrainer(
        model=student_model,
        train_dataset=train_dataset,
        args=training_args,
        loss=loss,

    trainer.train()

    """
