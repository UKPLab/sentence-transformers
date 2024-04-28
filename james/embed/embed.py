
import os
import time
import argparse
import sys

import pandas as pd
import torch
import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--tsv", required=True, type=str)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--output_file", type=str, help="should end with .npy")
parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")

args = parser.parse_args()
print(vars(args), flush=True)

# query_dev = pd.read_csv("./data/queries/queries.dev.tsv",
#                         header=None, sep="\t", index_col=0, names=["query"])
# collection = pd.read_csv("./data/collection/collection.tsv",
#                          header=None, sep="\t", index_col=0, names=["passage"])
dataframe = pd.read_csv(args.tsv, header=None, sep="\t",
                        index_col=0, names=["text"])
print(f"{len(dataframe)=}", flush=True)
model = SentenceTransformer(args.model_name).cuda()
print(f"Model ({args.model_name}) Initialized", flush=True)


def do_embedding(model, series, batch_size=128):
  embeddings = []
  n = len(series)
  start = time.time()
  for i in tqdm.tqdm(range((n + (batch_size - 1)) // batch_size)):
    a, b = batch_size * i, min(n, batch_size * (i + 1))
    embedding_i = model.encode(series[a:b].tolist())
    embeddings.append(embedding_i)
  end = time.time()
  print(f"Time Taken: {(end-start):0.4f} s")
  return np.vstack(embeddings), end - start


text_embeddings, time_taken = do_embedding(
  model, dataframe["text"], batch_size=args.batch_size)
print(f"{time_taken=}", flush=True)
print(f"{text_embeddings.shape=}", flush=True)

print(f"Saving to: {args.output_file}", flush=True)
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
np.save(args.output_file, text_embeddings)
