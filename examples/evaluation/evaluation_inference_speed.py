"""
This examples measures the inference speed of a certain model

Usage:
python evaluation_inference_speed.py
OR
python evaluation_inference_speed.py model_name
"""
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader
import logging
import sys
import os
import time
import torch

#Limit torch to 4 threads
torch.set_num_threads(4)

script_folder_path = os.path.dirname(os.path.realpath(__file__))


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-nli-mean-tokens'

# Load a named sentence model (based on BERT). This will download the model from our server.
# Alternatively, you can also pass a filepath to SentenceTransformer()
model = SentenceTransformer(model_name)

sts_reader = STSBenchmarkDataReader(os.path.join(script_folder_path, '../datasets/stsbenchmark'))
examples = sts_reader.get_examples("sts-train.csv")
sentences = [text for ex in examples for text in ex.texts]
print("Number of sentences:", len(sentences))

start_time = time.time()
emb = model.encode(sentences, batch_size=32)
end_time = time.time()
diff_time = end_time - start_time
print("Done after {:.2f} sec".format(diff_time))
print("Speed: {:.2f}".format(len(sentences) / diff_time))