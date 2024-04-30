import os
import numpy as np
import faiss
import time
import argparse
import sys
from bao.post_quant.embedding_quantization import post_quantize_embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--residing_folder", type=str, required=True)
parser.add_argument("--collection_file", type=str, required=True)
parser.add_argument("--query_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--output_file_scores", type=str, required=True)
parser.add_argument("--nlist", type=int, default=10000)
parser.add_argument("--nprobe", type=int, default=200)
parser.add_argument("--d", type=int, default=384) #TODO: remove this? 
parser.add_argument("--k", type=int, default=1000)
parser.add_argument("--truncation_d", type=int, default=-
                    1, help="dimension used to truncate")
parser.add_argument("--post_quant", type=str, default = None)
args = parser.parse_args()
print(vars(args), flush=True)

# residing_folder = "./data/results/baseline"
# collection_file = "collection_embeddings.npy"
# query_file = "query_embeddings.npy"
# output_file = "knn.npy"
# k = 1000 # knn
# nlist = 10000 # typically 4 * sqrt(n)
# nprobe = 200 # typically arbitrary
# d = 384

residing_folder, collection_file, query_file = args.residing_folder, args.collection_file, args.query_file
output_file, nlist, nprobe, d = args.output_file, args.nlist, args.nprobe, args.d
output_file_scores = args.output_file_scores
truncation_d = args.truncation_d
post_quant_precision = args.post_quant
k = args.k

corpus_embeddings = np.load(os.path.join(residing_folder, collection_file))
query_embeddings = np.load(os.path.join(residing_folder, query_file))

if truncation_d != -1:
  print(f"Applying Truncation (d={truncation_d})", flush=True)
  corpus_embeddings = corpus_embeddings[:, :truncation_d]
  query_embeddings = query_embeddings[:, :truncation_d]
  print(f"after truncation {corpus_embeddings.shape=}", flush=True)
  print(f"after truncation {query_embeddings.shape=}", flush=True)

if post_quant_precision is not None:
  #TODO: extend for mixed precision for queries | corpus
  corpus_embeddings = post_quantize_embeddings(corpus_embeddings, post_quant_precision)
  query_embeddings = post_quantize_embeddings(query_embeddings, post_quant_precision)
  print(f"after quantization {corpus_embeddings.shape=}", flush=True)
  print(f"after quantization {query_embeddings.shape=}", flush=True)

# just for debugging/test purposes on a subset
# corpus_embeddings = corpus_embeddings[:40000, :]
# query_embeddings = query_embeddings[:5, :]

# modify this if you do dimensionality reduction
assert corpus_embeddings.shape[-1] == query_embeddings.shape[-1]
faiss_d = corpus_embeddings.shape[-1]
quantizer = faiss.IndexFlatIP(faiss_d)
cpu_index = faiss.IndexIVFFlat(quantizer, faiss_d, nlist)
a = time.time()
cpu_index.train(corpus_embeddings)
cpu_index.add(corpus_embeddings)
b = time.time()
print(f"Time to train: {(b-a):0.2f} s", flush=True)

cpu_index.nprobe = nprobe
a = time.time()
D, I = cpu_index.search(query_embeddings, k)
b = time.time()
print(f"Time to search: {(b - a):0.2f} s", flush=True)
print(f"Saving to: {output_file}, {output_file_scores}", flush=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
os.makedirs(os.path.dirname(output_file_scores), exist_ok=True)
np.save(output_file, I)
np.save(output_file_scores, D)
