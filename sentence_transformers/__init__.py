__version__ = "0.3.9"

__DOWNLOAD_SERVER__ = 'https://sbert.net/models/'

import os
import logging

from .datasets import SentencesDataset, SentenceLabelDataset, ParallelSentencesDataset
from .LoggingHandler import LoggingHandler
from .LoggingHandler import install_logger
from .SentenceTransformer import SentenceTransformer
from .readers import InputExample
from .cross_encoder.CrossEncoder import CrossEncoder

logger = logging.getLogger(__name__)


# configure the library logger from which all other loggers will inherit
install_logger(logger, level=os.environ.get("ST_LOG_LEVEL", logging.WARNING))



