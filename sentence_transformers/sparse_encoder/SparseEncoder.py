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
from sentence_transformers.util import batch_to_device, truncate_embeddings_for_sparse

logger = logging.getLogger(__name__)


class SparseEncoder(SentenceTransformer):
    # TODO: Check if there is no other things we need to overwrite espacially for models specificty in the init
    # TODO: Look at the overload of encode ? Add handling for dict and list[dict]
    # TODO: Add the proper description with associate example
    # TODO: Clean encode implementation

    # NOTE: Function available in SparseEvaluator and if we override them (D for DONE) or no (ND for NOT DONE) or no and shouldn't be done (N&SBD):
    # - __init__ ND
    # - get_backend ND
    # - encode D (look at all the overloads possibility tho)
    # - forward ND
    # - similarity_fn_name ND
    # - similarity ND
    # - similarity_pairwise ND
    # - start_multi_process_pool ND
    # - stop_multi_process_pool ND
    # - encode_multi_process ND
    # - _encode_multi_process_worker ND
    # - set_pooling_include_prompt ND
    # - get_max_seq_length ND
    # - tokenize ND
    # - get_sentence_features ND
    # - get_sentence_embedding_dimension ND
    # - truncate_sentence_embeddings ND
    # - _first_module ND
    # - _last_module ND
    # - save ND
    # - save_pretrained ND
    # - _create_model_card ND
    # - save_to_hub ND
    # - push_to_hub ND
    # - _text_length ND
    # - evaluate ND
    # - _load_auto_model ND
    # - _load_module_class_from_ref ND
    # - _load_sbert_model ND
    # - load ND
    # - device ND
    # - tokenizer ND
    # - max_seq_length ND
    # - _target_device ND
    # - _no_split_modules ND
    # - _keys_to_ignore_on_save ND
    # - gradient_checkpointing_enable ND

    # -----------------------------------Added------------------------------------
    # - set_topk D
    # - get_sparsity_stats D

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
                embeddings = out_features["sentence_embedding"]
                embeddings = truncate_embeddings_for_sparse(embeddings, truncate_dim=self.truncate_dim)

                if convert_to_sparse_tensor:
                    embeddings = embeddings.to_sparse()
                all_embeddings.extend(embeddings)

        all_embeddings = torch.stack(all_embeddings)

        return all_embeddings

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
