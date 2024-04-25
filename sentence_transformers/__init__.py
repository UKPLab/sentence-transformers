__version__ = "3.0.0.dev0"
__MODEL_HUB_ORGANIZATION__ = "sentence-transformers"

import importlib
import os

from .datasets import SentencesDataset, ParallelSentencesDataset
from .LoggingHandler import LoggingHandler
from .SentenceTransformer import SentenceTransformer
from .similarity_functions import SimilarityFunction
from .readers import InputExample
from .cross_encoder.CrossEncoder import CrossEncoder
from .trainer import SentenceTransformerTrainer
from .training_args import SentenceTransformerTrainingArguments
from .model_card import SentenceTransformerModelCardData
from .quantization import quantize_embeddings


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
