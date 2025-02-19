from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder


class BaseWeighingScheme(nn.Module):
    """Base class for implementing weighing schemes in LambdaLoss."""

    def forward(self, G: Tensor, D: Tensor, true_sorted: Tensor) -> Tensor:
        """
        Calculate weights for the loss function.

        Args:
            G: Normalized gains tensor
            D: Discount tensor
            true_sorted: Sorted ground truth labels

        Returns:
            Tensor: Calculated weights for the loss
        """
        raise NotImplementedError


class NoWeighingScheme(BaseWeighingScheme):
    """Implementation of no weighing scheme (weights = 1.0)."""

    def forward(self, G: Tensor, D: Tensor, true_sorted: Tensor) -> Tensor:
        return torch.tensor(1.0, device=G.device)


class NDCGLoss1Scheme(BaseWeighingScheme):
    """Implementation of NDCG Loss1 weighing scheme."""

    def forward(self, G: Tensor, D: Tensor, true_sorted: Tensor) -> Tensor:
        return (G / D)[:, :, None]


class NDCGLoss2Scheme(BaseWeighingScheme):
    """Implementation of NDCG Loss2 weighing scheme."""

    def forward(self, G: Tensor, D: Tensor, true_sorted: Tensor) -> Tensor:
        pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
        delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
        deltas = torch.abs(
            torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.0) - torch.pow(torch.abs(D[0, delta_idxs]), -1.0)
        )
        deltas.diagonal().zero_()
        return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


class LambdaRankScheme(BaseWeighingScheme):
    """Implementation of LambdaRank weighing scheme."""

    def forward(self, G: Tensor, D: Tensor, true_sorted: Tensor) -> Tensor:
        return torch.abs(torch.pow(D[:, :, None], -1.0) - torch.pow(D[:, None, :], -1.0)) * torch.abs(
            G[:, :, None] - G[:, None, :]
        )


class NDCGLoss2PPScheme(BaseWeighingScheme):
    """Implementation of NDCG Loss2++ weighing scheme."""

    def __init__(self, mu: float = 10.0):
        super().__init__()
        self.mu = mu
        self.ndcg_loss2 = NDCGLoss2Scheme()
        self.lambda_rank = LambdaRankScheme()

    def forward(self, G: Tensor, D: Tensor, true_sorted: Tensor) -> Tensor:
        ndcg_weights = self.ndcg_loss2(G, D, true_sorted)
        lambda_weights = self.lambda_rank(G, D, true_sorted)
        return self.mu * ndcg_weights + lambda_weights


class LambdaLoss(nn.Module):
    """
    LambdaLoss implementation for learning to rank in sentence-transformers.

    This loss function implements the LambdaLoss framework for Learning to Rank,
    which provides various weighing schemes including LambdaRank and NDCG variations.
    The implementation is optimized to handle padded documents efficiently by only
    processing valid documents during model inference.

    Args:
        model (CrossEncoder): CrossEncoder model to be trained
        weighing_scheme (Optional[BaseWeighingScheme]): Weighing scheme to use
        k (Optional[int]): Rank at which the loss is truncated
        sigma (float): Score difference weight used in sigmoid
        eps (float): Small constant for numerical stability
        pad_value (int): Value used for padding in variable-length document lists
        reduction (Literal["sum", "mean"]): Method to reduce the loss
        reduction_log (Literal["natural", "binary"]): Type of logarithm to use
        activation_fct (Optional[nn.Module]): Activation function to apply to model outputs

    References:
        - The LambdaLoss Framework for Ranking Metric Optimization
        - Learning to Rank: From Pairwise Approach to Listwise Approach

    Requirements:
        1. Query with multiple documents (listwise approach)
        2. Documents must have relevance scores/labels
        3. Variable-length document lists are handled through padding
        4. Padded documents are automatically excluded from model inference

    Inputs:
        +----------------------------------------+--------------------------------+
        | Texts                                   | Labels                         |
        +========================================+================================+
        | (query, [doc1, doc2, ..., docN])       | [score1, score2, ..., scoreN] |
        +----------------------------------------+--------------------------------+
        Note: Documents with label=pad_value are efficiently skipped during processing
    """

    def __init__(
        self,
        model: CrossEncoder,
        weighing_scheme: BaseWeighingScheme | None = None,
        k: int | None = None,
        sigma: float = 1.0,
        eps: float = 1e-10,
        pad_value: int = -1,
        reduction: Literal["sum", "mean"] = "sum",
        reduction_log: Literal["natural", "binary"] = "binary",
        activation_fct: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.weighing_scheme = weighing_scheme or NoWeighingScheme()
        self.k = k
        self.sigma = sigma
        self.eps = eps
        self.pad_value = pad_value
        self.reduction = reduction
        self.reduction_log = reduction_log
        self.activation_fct = activation_fct

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} supports a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str], list[list[str]]], labels: Tensor) -> Tensor:
        """
        Compute LambdaLoss for a batch of queries and their documents.

        Args:
            inputs: List of (queries, documents_list)
            labels: Ground truth relevance scores, shape (batch_size, num_documents)

        Returns:
            Tensor: Computed loss value
        """
        queries, docs_list = inputs
        labels = labels.float()
        batch_size, max_docs = labels.size()

        # Create mask for valid (non-padded) documents
        mask = labels != self.pad_value  # shape: (batch_size, max_docs)
        batch_indices, doc_indices = torch.where(mask)

        # Create input pairs for the model
        pairs = [
            (queries[batch_index], docs_list[batch_index][doc_index])
            for batch_index, doc_index in zip(batch_indices.tolist(), doc_indices.tolist())
        ]

        if not pairs:
            # Handle edge case where all documents are padded
            return torch.tensor(0.0, device=self.model.device)

        # Tokenize inputs and get model predictions
        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = tokens.to(self.model.device)
        logits = self.model(**tokens)[0].view(-1)

        # Create full logits tensor with -inf for padded positions
        y_pred = torch.full((batch_size, max_docs), float("-inf"), device=self.model.device)
        y_pred[batch_indices, doc_indices] = logits

        if self.activation_fct is not None:
            y_pred = self.activation_fct(y_pred)

        # Move labels to correct device and handle padding
        y_true = labels.to(self.model.device)

        # Calculate LambdaLoss components
        device = y_pred.device
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        # Create masks for valid pairs
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        if not isinstance(self.weighing_scheme, NDCGLoss1Scheme):
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        # Create truncation mask if k is specified
        k = self.k or max_docs
        ndcg_at_k_mask = torch.zeros((max_docs, max_docs), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        # Calculate gains and discounts
        true_sorted_by_preds.clamp_(min=0.0)
        y_true_sorted.clamp_(min=0.0)

        pos_idxs = torch.arange(1, max_docs + 1).to(device)
        D = torch.log2(1.0 + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=self.eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # Apply weighing scheme
        weights = self.weighing_scheme(G, D, true_sorted_by_preds)

        # Calculate scores differences and probabilities
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs.masked_fill_(torch.isnan(scores_diffs), 0.0)
        weighted_probas = (torch.sigmoid(self.sigma * scores_diffs).clamp(min=self.eps) ** weights).clamp(min=self.eps)

        # Calculate losses based on specified logarithm base
        if self.reduction_log == "natural":
            losses = torch.log(weighted_probas)
        else:  # binary
            losses = torch.log2(weighted_probas)

        # Apply final reduction
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        if self.reduction == "sum":
            loss = -torch.sum(masked_losses)
        else:  # mean
            loss = -torch.mean(masked_losses)

        return loss

    def get_config_dict(self) -> dict[str, float | int | str | None]:
        """
        Get configuration parameters for this loss function.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {
            "weighing_scheme": self.weighing_scheme.__class__.__name__,
            "k": self.k,
            "sigma": self.sigma,
            "eps": self.eps,
            "pad_value": self.pad_value,
            "reduction": self.reduction,
            "reduction_log": self.reduction_log,
            "activation_fct": (self.activation_fct.__class__.__name__ if self.activation_fct else None),
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{wang2018lambdaloss,
  title={The lambdaloss framework for ranking metric optimization},
  author={Wang, Xuanhui and Li, Cheng and Golbandi, Nadav and Bendersky, Michael and Najork, Marc},
  booktitle={Proceedings of the 27th ACM international conference on information and knowledge management},
  pages={1313--1322},
  year={2018}
}
"""
