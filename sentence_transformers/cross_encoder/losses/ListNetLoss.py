from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import fullname


class ListNetLoss(nn.Module):
    """
    ListNet loss for learning to rank. This loss function implements the ListNet ranking algorithm
    which uses a list-wise approach to learn ranking models. It minimizes the cross entropy
    between the predicted ranking distribution and the ground truth ranking distribution.
    The implementation is optimized to handle padded documents efficiently by only processing
    valid documents during model inference.

    Args:
        model (CrossEncoder): CrossEncoder model to be trained
        eps (float): Small constant to prevent numerical instability in log.
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
        +----------------------------------------+--------------------------------+
        | Texts                                   | Labels                         |
        +========================================+================================+
        | (query, [doc1, doc2, ..., docN])       | [score1, score2, ..., scoreN]  |
        +----------------------------------------+--------------------------------+
        Note: Documents with label=pad_value are efficiently skipped during processing

    Relations:
        - /
    """

    def __init__(
        self,
        model: CrossEncoder,
        eps: float = 1e-10,
        pad_value: int = -1,
        activation_fct: nn.Module | None = nn.Sigmoid(),
    ) -> None:
        super().__init__()
        self.model = model
        self.eps = eps
        self.pad_value = pad_value
        self.activation_fct = activation_fct or nn.Identity()

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

        # Create output tensor filled with -inf (for softmax)
        full_logits = torch.full((batch_size, max_docs), float("-inf"), device=self.model.device)

        # Place logits back in their original positions
        full_logits[batch_indices, doc_indices] = logits

        full_logits = self.activation_fct(full_logits)

        # Set padded positions in labels to -inf for consistent softmax
        labels = labels.to(self.model.device)
        labels[~mask] = float("-inf")

        # Compute probability distributions through softmax
        P = F.softmax(labels, dim=1)
        Q = F.softmax(full_logits, dim=1)

        # Calculate cross entropy between distributions
        loss = -torch.sum(P * torch.log(Q + self.eps), dim=1)

        return loss.mean()

    def get_config_dict(self) -> dict[str, float]:
        """
        Get configuration parameters for this loss function.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {
            "eps": self.eps,
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
