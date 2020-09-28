"""
This examples measures the inference speed of a certain model

Usage:
python evaluation_inference_speed.py
OR
python evaluation_inference_speed.py model_name
"""
from sentence_transformers import SentenceTransformer, util
import sys
import os
import time
import torch
import gzip
import csv

#Limit torch to 4 threads
torch.set_num_threads(4)


model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-nli-mean-tokens'

# Load a named sentence model (based on BERT). This will download the model from our server.
# Alternatively, you can also pass a filepath to SentenceTransformer()
model = SentenceTransformer(model_name)


nli_dataset_path = 'datasets/AllNLI.tsv.gz'
sentences = set()
max_sentences = 100000


#Download datasets if needed
if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        sentences.add(row['sentence1'])
        if len(sentences) >= max_sentences:
            break

sentences = list(sentences)
print("Model Name:", model_name)
print("Number of sentences:", len(sentences))

for i in range(3):
    print("Run", i)
    start_time = time.time()
    emb = model.encode(sentences, num_workers=2, batch_size=32)
    end_time = time.time()
    diff_time = end_time - start_time
    print("Done after {:.2f} seconds".format(diff_time))
    print("Speed: {:.2f} sentences / second".format(len(sentences) / diff_time))
    print("=====")