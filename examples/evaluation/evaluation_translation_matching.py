"""
Given a tab seperated file (.tsv) with parallel sentences, where the second column is the translation of the sentence in the first column, for example, in the format:
src1    trg1
src2    trg2
...

where trg_i is the translation of src_i.

Given src_i, the TranslationEvaluator checks which trg_j has the highest similarity using cosine similarity. If i == j, we assume
a match, i.e., the correct translation has been found for src_i out of all possible target sentences.

It then computes an accuracy over all possible source sentences src_i. Equivalently, it computes also the accuracy for the other direction.

A high accuracy score indicates that the model is able to find the correct translation out of a large pool with sentences.

Usage:
python [model_name_or_path] [parallel-file1] [parallel-file2] ...

For example:
python distiluse-base-multilingual-cased  TED2020-en-de.tsv.gz

See the training_multilingual/get_parallel_data_...py scripts for getting parallel sentence data from different sources
"""

from sentence_transformers import SentenceTransformer, evaluation, LoggingHandler
import sys
import gzip
import os
import logging


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

model_name = sys.argv[1]
filepaths = sys.argv[2:]
inference_batch_size = 32

model = SentenceTransformer(model_name)


for filepath in filepaths:
    src_sentences = []
    trg_sentences = []
    with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            splits = line.strip().split('\t')
            if len(splits) >= 2:
                src_sentences.append(splits[0])
                trg_sentences.append(splits[1])

    logger.info(os.path.basename(filepath)+": "+str(len(src_sentences))+" sentence pairs")
    dev_trans_acc = evaluation.TranslationEvaluator(src_sentences, trg_sentences, name=os.path.basename(filepath), batch_size=inference_batch_size)
    dev_trans_acc(model)


