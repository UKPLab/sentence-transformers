from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn as nn

from sentence_transformers.sparse_encoder.losses import FlopsLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder

logger = logging.getLogger(__name__)


class SpladeLoss(nn.Module):
    def __init__(
        self,
        model: SparseEncoder,
        loss: nn.Module,
        lambda_corpus: float,
        lambda_query: float | None = None,
        all_docs: bool = False,
        corpus_threshold: int | None = None,
        query_threshold: int | None = None,
        corpus_regularizer: nn.Module | None = None,
        query_regularizer: nn.Module | None = None,
    ):
        """
        SpladeLoss implements the loss function for the SPLADE (Sparse Lexical and Expansion) model,
        which combines a main loss function with regularization terms to control efficiency.

        This loss function balances effectiveness (via the main loss) with efficiency by regularizing
        both the query and document representations to be sparse, reducing computational requirements
        at inference time.

        Args:
            model: SparseEncoder model
            loss: The principal loss function to use can be any of the SparseEncoder losses except CSR related losses and flops loss.
            lambda_corpus: Weight for the corpus regularization term. This term encourages sparsity in the document embeddings.
                Will be applied to positive documents and all negatives one if some are provided.
            lambda_query: Weight for the query regularization term. This term encourages sparsity in the query embeddings.
                If None, no query regularization will be applied, it's not a problem if you are in an inference-free setup or
                if you are having all_docs=True. Else you should have a lambda_query > 0.
            all_docs: If True, all input embeddings are treated as documents and regularized together with lambda_corpus.
                Especially useful when training with symmetric texts (e.g. pairs of documents) or more.
            corpus_threshold: Optional threshold for the number of non-zero (active) elements in the corpus embeddings to be considered in the FlopsLoss.
                If specified, only corpus embeddings with more than this number of non-zero (active) elements will be considered.
                Only used when corpus_regularizer is None (for the default FlopsLoss).
            query_threshold: Optional threshold for the number of non-zero (active) elements in the query embeddings to be considered in the FlopsLoss.
                If specified, only query embeddings with more than this number of non-zero (active) elements will be considered.
                Only used when query_regularizer is None (for the default FlopsLoss).
            corpus_regularizer: Optional regularizer to use specifically for corpus regularization instead of the default FlopsLoss.
                This allows for different regularization strategies for documents vs queries.
            query_regularizer: Optional regularizer to use specifically for query regularization instead of the default FlopsLoss.
                This allows for different regularization strategies for queries vs documents.

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

                student_model = SparseEncoder("distilbert/distilbert-base-uncased")
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
                    student_model, loss=losses.SparseMarginMSELoss(student_model), lambda_corpus=3e-5, lambda_query=5e-5
                )

                trainer = SparseEncoderTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.lambda_corpus = lambda_corpus
        self.lambda_query = lambda_query
        self.all_docs = all_docs

        # Set up regularizers with defaults to FlopsLoss using specific thresholds
        self.corpus_regularizer = (
            corpus_regularizer if corpus_regularizer is not None else FlopsLoss(model, threshold=corpus_threshold)
        )
        if query_regularizer is not None:
            self.query_regularizer = query_regularizer
        elif not all_docs:
            self.query_regularizer = FlopsLoss(model, threshold=query_threshold)

        if self.lambda_query is None and not all_docs:
            logging.warning(
                "lambda_query is None. This means that the query regularization will not be applied. If you are in an inference free set up it's fine else you should have a lambda_query > 0."
            )
        if self.all_docs and self.lambda_query is not None:
            logging.warning(
                "lambda_query should be None when all_docs is True. all_docs mean we consider all the input to be of the same type and so under the same regularization. lambda_query will be ignored."
            )
            self.lambda_query = None
        if not hasattr(loss, "compute_loss_from_embeddings"):
            raise ValueError(
                "The provided loss does not have a 'compute_loss_from_embeddings' method, which is required for SpladeLoss. "
                "This method must have the signature `compute_loss_from_embeddings(embeddings: List[Tensor], labels: Tensor | None = None)`."
            )

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        # Compute embeddings using the model
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        losses = {}
        base_loss = self.loss.compute_loss_from_embeddings(embeddings, labels)
        if isinstance(base_loss, dict):
            losses.update(base_loss)
        else:
            losses["base_loss"] = base_loss

        if self.all_docs:
            # If all_docs is True, we consider all the input to be of the same type and so under the same regularization
            corpus_loss = self.corpus_regularizer.compute_loss_from_embeddings(torch.cat(embeddings))
        else:
            corpus_loss = self.corpus_regularizer.compute_loss_from_embeddings(torch.cat(embeddings[1:]))
        losses["corpus_regularizer_loss"] = corpus_loss * self.lambda_corpus

        # Add query regularization if enabled
        if self.lambda_query is not None:
            query_loss = self.query_regularizer.compute_loss_from_embeddings(embeddings[0])
            losses["query_regularizer_loss"] = query_loss * self.lambda_query

        return losses

    def get_config_dict(self):
        """
        Get the configuration dictionary.

        Returns:
            Dictionary containing the configuration parameters
        """
        config_dict = {
            "loss": self.loss,
            "lambda_corpus": self.lambda_corpus,
        }
        if self.lambda_query is not None:
            config_dict["lambda_query"] = self.lambda_query
        # Include regularizer names (if not flops) and threshold information (if not None)

        if not isinstance(self.corpus_regularizer, FlopsLoss):
            config_dict["corpus_regularizer"] = self.corpus_regularizer.__class__.__name__
        if hasattr(self.corpus_regularizer, "threshold") and self.corpus_regularizer.threshold is not None:
            config_dict["corpus_threshold"] = self.corpus_regularizer.threshold

        if hasattr(self, "query_regularizer") and self.query_regularizer is not None:
            if not isinstance(self.query_regularizer, FlopsLoss):
                config_dict["query_regularizer"] = self.query_regularizer.__class__.__name__
            if hasattr(self.query_regularizer, "threshold") and self.query_regularizer.threshold is not None:
                config_dict["query_threshold"] = self.query_regularizer.threshold
        return config_dict

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
