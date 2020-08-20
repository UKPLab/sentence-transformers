"""
This examples measures the inference speed of a certain model

Usage:
python evaluation_inference_speed.py
OR
python evaluation_inference_speed.py model_name
"""
from sentence_transformers import SentenceTransformer
import sys
import os
import time
import torch
import gzip

#Limit torch to 4 threads
torch.set_num_threads(4)

script_folder_path = os.path.dirname(os.path.realpath(__file__))



model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-nli-mean-tokens'

# Load a named sentence model (based on BERT). This will download the model from our server.
# Alternatively, you can also pass a filepath to SentenceTransformer()
model = SentenceTransformer(model_name)

#Perform computation with FP16
#model.half()

#Fork multiple processes to tokenize the input data
#model.parallel_tokenization = True

sentences = []
max_sentences = 100000

with gzip.open(os.path.join(script_folder_path, '../datasets/AllNLI/s1.train.gz'), 'rt', encoding='utf8') as fIn:
    for line in fIn:
        sentences.append(line.strip())
        if len(sentences) >= max_sentences:
            break

print("Model Name:", model_name)
print("Number of sentences:", len(sentences))

for i in range(3):
    print("Run", i)
    start_time = time.time()
    emb = model.encode(sentences, batch_size=32)
    end_time = time.time()
    diff_time = end_time - start_time
    print("Done after {:.2f} seconds".format(diff_time))
    print("Speed: {:.2f} sentences / second".format(len(sentences) / diff_time))
    print("=====")