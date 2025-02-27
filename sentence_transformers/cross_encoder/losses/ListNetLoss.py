from __future__ import annotations

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import fullname


class ListNetLoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        activation_fct: nn.Module | None = nn.Identity(),
        pad_value: int = -1,
    ) -> None:
        """
        ListNet loss for learning to rank. This loss function implements the ListNet ranking algorithm
        which uses a list-wise approach to learn ranking models. It minimizes the cross entropy
        between the predicted ranking distribution and the ground truth ranking distribution.
        The implementation is optimized to handle padded documents efficiently by only processing
        valid documents during model inference.

        Args:
            model (CrossEncoder): CrossEncoder model to be trained
            activation_fct (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the loss. Defaults to :class:`~torch.nn.Identity`.
            pad_value (int): Value used for padding in variable-length document lists.
                Documents with this value will be excluded from model inference for efficiency.

        References:
            - Learning to Rank: From Pairwise Approach to Listwise Approach: https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/
            - `Training Examples > Learning to Rank <../../../examples/training/cross-encoder/training_ms_marco_ListNetLoss_v4.py>`_

        Requirements:
            1. Query with multiple documents (listwise approach)
            2. Documents must have relevance scores/labels. Both binary and continuous labels are supported.
            3. Variable-length document lists are handled through padding (pad_value)
            4. Padded documents are automatically excluded from model inference for efficiency

        Inputs:
            +----------------------------------------+--------------------------------+-------------------------------+
            | Texts                                  | Labels                         | Number of Model Output Labels |
            +========================================+================================+===============================+
            | (query, [doc1, doc2, ..., docN])       | [score1, score2, ..., scoreN]  | 1                             |
            +----------------------------------------+--------------------------------+-------------------------------+

        .. note::

            Documents with ``label=pad_value`` are efficiently skipped during processing.

        Example:
            ::

                from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
                from datasets import Dataset

                model = CrossEncoder("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "query": ["What are pandas?", "What is the capital of France?"],
                    "docs": [
                        ["Pandas are a kind of bear.", "Pandas are kind of like fish."],
                        ["The capital of France is Paris.", "Paris is the capital of France."],
                    ],
                    "labels": [[1, 0], [1, 1]],
                })
                loss = losses.ListNetLoss(model)

                trainer = CrossEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.pad_value = pad_value
        self.activation_fct = activation_fct or nn.Identity()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} supports a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str], list[list[str]]], labels: Tensor) -> Tensor:
        """
        Compute ListNet loss for a batch of queries and their documents.

        Args:
            inputs: List of (queries, documents_list)
            labels: Ground truth relevance scores, shape (batch_size, num_documents)

        Returns:
            Tensor: Mean ListNet loss over the batch
        """
        queries, docs_list = inputs
        labels = labels.float()
        batch_size, max_docs = labels.size()

        mask = labels != self.pad_value  # shape: (batch_size, max_docs)

        batch_indices, doc_indices = torch.where(mask)

        pairs = [
            (queries[batch_index], docs_list[batch_index][doc_index])
            for batch_index, doc_index in zip(batch_indices.tolist(), doc_indices.tolist())
        ]

        if not pairs:
            # Handle edge case where all documents are padded
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = tokens.to(self.model.device)

        logits = self.model(**tokens)[0].view(-1)
        logits = self.activation_fct(logits)

        # Create output tensor filled with 0 (padded logits will be ignored via labels)
        full_logits = torch.full((batch_size, max_docs), -1e16, device=self.model.device)

        # Place logits back in their original positions
        full_logits[batch_indices, doc_indices] = logits

        # Set padded positions in labels to -inf for consistent softmax
        labels = labels.to(self.model.device)
        labels[~mask] = float("-inf")

        # Compute cross entropy loss between distributions
        loss = self.cross_entropy_loss(full_logits, labels.softmax(dim=1))

        return loss

    def get_config_dict(self) -> dict[str, float]:
        """
        Get configuration parameters for this loss function.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {
            "pad_value": self.pad_value,
            "activation_fct": fullname(self.activation_fct),
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{cao2007learning,
    title={Learning to rank: from pairwise approach to listwise approach},
    author={Cao, Zhe and Qin, Tao and Liu, Tie-Yan and Tsai, Ming-Feng and Li, Hang},
    booktitle={Proceedings of the 24th international conference on Machine learning},
    pages={129--136},
    year={2007}
}
"""
