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
        lambda_query: float = None,
        all_docs: bool = False,
        threshold: int = None,
        regularizer: nn.modules = None,
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
            threshold: Optional threshold for the number of non-zero elements in the embeddings to be considered in the FlopsLoss.
                If specified, only embeddings with more than this number of non-zero elements will be considered.
                This can help to ignore embeddings that are too sparse and may not contribute meaningfully to the loss.
            regularizer: Optional regularizer to use instead of the default FlopsLoss. This can be useful for custom regularization strategies.

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
        self.regularizer = regularizer if regularizer is not None else FlopsLoss(model, threshold=threshold)
        self.lambda_corpus = lambda_corpus
        self.lambda_query = lambda_query

        if self.lambda_query is None and not all_docs:
            logging.warning(
                "lambda_query is None. This means that the query regularization will not be applied. If you are in an inference free set up it's fine else you should have a lambda_query > 0."
            )
        self.all_docs = all_docs
        if self.all_docs and self.lambda_query is not None:
            logging.warning(
                "lambda_query should be None when all_docs is True. all_docs mean we consider all the input to be of the same type and so under the same regularization. lambda_query will be ignored."
            )
            self.lambda_query = None

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        # Compute embeddings using the model
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        loss_value = self.loss.compute_loss_from_embeddings(embeddings, labels)
        if self.all_docs:
            # If all_docs is True, we consider all the input to be of the same type and so under the same regularization
            corpus_loss = self.regularizer.compute_loss_from_embeddings(torch.cat(embeddings))
        else:
            corpus_loss = self.regularizer.compute_loss_from_embeddings(torch.cat(embeddings[1:]))

        # Compute total loss
        total_loss = loss_value + self.lambda_corpus * corpus_loss

        # Add query regularization if enabled
        if self.lambda_query is not None:
            query_loss = self.regularizer.compute_loss_from_embeddings(embeddings[0])
            total_loss = total_loss + self.lambda_query * query_loss

        return total_loss

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
        if self.regularizer.threshold is not None:
            config_dict["threshold"] = self.regularizer.threshold
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
