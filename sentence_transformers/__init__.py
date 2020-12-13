__version__ = "0.3.9"

__DOWNLOAD_SERVER__ = 'https://sbert.net/models/'

import logging

from .datasets import SentencesDataset, SentenceLabelDataset, ParallelSentencesDataset
from .LoggingHandler import LoggingHandler
from .SentenceTransformer import SentenceTransformer
from .readers import InputExample
from .cross_encoder.CrossEncoder import CrossEncoder


logger = logging.getLogger(__name__)


