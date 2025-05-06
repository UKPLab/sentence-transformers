from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import torch
import torch.nn as nn

from sentence_transformers.sparse_encoder.losses import (
    FlopsLoss,
    IDFFlopsLoss,
    SparseDistillKLDivLoss,
    SparseMarginMSELoss,
    SparseMultipleNegativesRankingLoss,
)
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class PrincipalLoss(Enum):
    """The principal loss types for the model"""

    MMSE = SparseMarginMSELoss
    KL = SparseDistillKLDivLoss
    MRL = SparseMultipleNegativesRankingLoss


class RegularizerLoss(Enum):
    """The regularizer loss types for the model"""

    FLOPS = FlopsLoss
    IDFFLOPS = IDFFlopsLoss


class SpladeLoss(nn.Module):
    def __init__(
        self,
        model: SparseEncoder,
        loss: nn.Module,
        lambda_corpus: float = 0.1,
        lambda_query: float = 0.1,
        corpus_regularizer: nn.Module = None,
        query_regularizer: nn.Module = None,
    ):
        """
        SpladeLoss implements the loss function for the SPLADE (Sparse Lexical and Expansion) model,
        which combines a main loss function with regularization terms to control efficiency.

        This loss function balances effectiveness (via the main loss) with efficiency by regularizing
        both the query and document representations to be sparse, reducing computational requirements
        at inference time.

        Args:
            model: SparseEncoder model
            loss: The principal loss function to use (can be :class:`~sentence_transformers.sparse_encoder.losses.SparseMarginMSELoss`, :class:`~sentence_transformers.sparse_encoder.losses.SparseDistillKLDivLoss`,
                       or :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss`)
            lambda_corpus: Regularization weight for corpus (document) embeddings
            lambda_query: Regularization weight for query embeddings

        References:
            - For more details, see the paper "From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective"
              https://arxiv.org/abs/2205.04733

        Requirements:
            1. Input requirements depend on the chosen loss
            2. Usually used with a teacher model in a knowledge distillation setup and an associated loss

        Example:
            ::

                from datasets import Dataset
                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                student_model = SparseEncoder("prithivida/Splade_PP_en_v1")
                teacher_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                train_dataset = Dataset.from_dict(
                    {
                        "query": ["It's nice weather outside today.", "He drove to work."],
                        "passage1": ["It's so sunny.", "He took the car to work."],
                        "passage2": ["It's very sunny.", "She walked to the store."],
                    }
                )

                def compute_labels(batch):
                    emb_queries = teacher_model.encode(batch["query"])
                    emb_passages1 = teacher_model.encode(batch["passage1"])
                    emb_passages2 = teacher_model.encode(batch["passage2"])
                    return {
                        "label": teacher_model.similarity_pairwise(emb_queries, emb_passages1)
                        - teacher_model.similarity_pairwise(emb_queries, emb_passages2)
                    }

                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.SpladeLoss(
                    student_model,
                    loss=losses.SparseMarginMSELoss(student_model),
                    lambda_corpus=5e-3,
                    lambda_query=0.1,
                )

                trainer = SparseEncoderTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.lambda_corpus = lambda_corpus
        self.lambda_query = lambda_query
        self.loss = loss
        if self.loss.__class__.__name__ not in PrincipalLoss.__members__:
            raise ValueError(
                f"Principal loss must be one of {list(PrincipalLoss.__members__.keys())}, but got {self.loss.__class__.__name__}"
            )

        self.corpus_regularizer = corpus_regularizer if corpus_regularizer is not None else FlopsLoss(model)
        if self.corpus_regularizer.__class__.__name__ not in RegularizerLoss.__members__:
            raise ValueError(
                f"Corpus regularizer must be one of {list(RegularizerLoss.__members__.keys())}, but got {self.corpus_regularizer.__class__.__name__}"
            )
        if self.corpus_regularizer.__class__.__name__ == "IDFFlopsLoss" or lambda_query == 0:
            self.query_regularizer = None
        elif query_regularizer is None:
            self.query_regularizer = FlopsLoss(model)
        else:
            self.query_regularizer = query_regularizer
            if self.query_regularizer.__class__.__name__ not in RegularizerLoss.__members__:
                raise ValueError(
                    f"Query regularizer must be one of {list(RegularizerLoss.__members__.keys())}, but got {self.query_regularizer.__class__.__name__}"
                )

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        # Compute embeddings using the model
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        loss_value = self.loss.compute_loss_from_embeddings(embeddings, labels)

        corpus_loss = self.corpus_regularizer(embeddings, "corpus")

        # Compute total loss
        total_loss = loss_value + self.lambda_corpus * corpus_loss

        # Add query regularization if enabled
        if self.query_regularizer is not None:
            query_loss = self.query_regularizer(embeddings, "query")
            total_loss = total_loss + self.lambda_query * query_loss

        return total_loss

    def get_config_dict(self):
        """
        Get the configuration dictionary.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {
            "loss": self.loss,
            "lambda_corpus": self.lambda_corpus,
            "lambda_query": self.lambda_query,
            "corpus_regularizer": self.corpus_regularizer,
            "query_regularizer": self.query_regularizer,
        }

    @property
    def citation(self) -> str:
        return """
@misc{formal2022distillationhardnegativesampling,
      title={From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective},
      author={Thibault Formal and Carlos Lassance and Benjamin Piwowarski and Stéphane Clinchant},
      year={2022},
      eprint={2205.04733},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2205.04733},
}
"""
