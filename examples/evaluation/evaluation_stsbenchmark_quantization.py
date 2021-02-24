"""
This examples loads a pre-trained model, and creates two instances of it.
One model's weights are untouched and the others are quantized. Both are then evaluated on the STSBenchmark dataset

For more on quantization see https://pytorch.org/docs/stable/quantization.html
Usage:
python evaluation_stsbenchmark.py
OR
python evaluation_stsbenchmark.py model_name
"""
import logging
import os
import sys

import torch
# pyinfer not in base environment - pip install pyinfer
from pyinfer import MultiInferenceReport
from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader
from torch.utils.data import DataLoader


script_folder_path = os.path.dirname(os.path.realpath(__file__))

#Limit torch to 4 threads
torch.set_num_threads(4)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
### /print debug information to stdout

model_name = sys.argv[1] if len(sys.argv) > 1 else 'paraphrase-distilroberta-base-v1'

# Load a named sentence model (based on BERT). This will download the model from our server.
# Alternatively, you can also pass a filepath to SentenceTransformer()
model = SentenceTransformer(model_name)
q_model = SentenceTransformer(model_name, quantize_model=True)

sts_reader = STSBenchmarkDataReader(os.path.join(script_folder_path, '../datasets/stsbenchmark'))
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(sts_reader.get_examples("sts-test.csv"), name='sts-test')

logging.info("Starting regular model evaluation")
model.evaluate(evaluator)


logging.info("Starting quantized model evaluation")
q_model.evaluate(evaluator)


logging.info("Evaluating inference speed statistics")


sentences = ['This framework generates embeddings for each input sentence',
            'Sentences are passed as a list of string.', 
            'The quick brown fox jumps over the lazy dog.']

logging.getLogger().setLevel(5)
report = MultiInferenceReport([model.encode, q_model.encode], sentences, n_iterations=500, model_names=["distilroberta","q_distilroberta"])
report.run()
