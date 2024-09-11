from __future__ import annotations

__version__ = "3.1.0"
__MODEL_HUB_ORGANIZATION__ = "sentence-transformers"

import importlib
import os

from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
from sentence_transformers.datasets import ParallelSentencesDataset, SentencesDataset
from sentence_transformers.LoggingHandler import LoggingHandler
from sentence_transformers.model_card import SentenceTransformerModelCardData
from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers.readers import InputExample
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# If codecarbon is installed and the log level is not defined,
# automatically overwrite the default to "error"
if importlib.util.find_spec("codecarbon") and "CODECARBON_LOG_LEVEL" not in os.environ:
    os.environ["CODECARBON_LOG_LEVEL"] = "error"

__all__ = [
    "LoggingHandler",
    "SentencesDataset",
    "ParallelSentencesDataset",
    "SentenceTransformer",
    "SimilarityFunction",
    "InputExample",
    "CrossEncoder",
    "SentenceTransformerTrainer",
    "SentenceTransformerTrainingArguments",
    "SentenceTransformerModelCardData",
    "quantize_embeddings",
]
