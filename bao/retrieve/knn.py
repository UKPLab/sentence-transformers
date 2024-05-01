import os
import numpy as np
import faiss
import time
import argparse
import sys
from bao.post_quant.embedding_quantization import post_quantize_embeddings
from bao.indexing.instantiate_indexes import get_index

parser = argparse.ArgumentParser()
# PATH
parser.add_argument("--residing_folder", type=str, required=True)
parser.add_argument("--collection_file", type=str, required=True)
parser.add_argument("--query_file", type=str, required=True)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--output_file_scores", type=str, required=True)

# POST PROCESSING 1: DIMENSION REDUCTION
parser.add_argument("--truncation_d", type=int, default=-1, help="dimension used to truncate")
#TODO: add PCA

# POST PROCESSING 2: QUANTIZATION
parser.add_argument("--post_quant", type=str, default = None)

# CUSTOM INDEX
parser.add_argument("--use_usearch", action='store_true')
parser.add_argument("--use_flat", action='store_true')
parser.add_argument("--use_ivf", action='store_true')

# SEARCH PARAMETER
parser.add_argument("--k", type=int, default=1000)

args = parser.parse_args()
print('------------------------- NEW PIPELINE RUN ----------------------------')
print(vars(args), flush=True)

residing_folder, collection_file, query_file = args.residing_folder, args.collection_file, args.query_file
output_folder, output_file, output_file_scores = args.output_folder, args.output_file, args.output_file_scores
truncation_d = args.truncation_d
quantization_method = args.post_quant
k = args.k

corpus_embeddings = np.load(os.path.join(residing_folder, collection_file))
query_embeddings = np.load(os.path.join(residing_folder, query_file))


if truncation_d != -1:
  print(f"Applying Truncation (d={truncation_d})", flush=True)
  corpus_embeddings = corpus_embeddings[:, :truncation_d]
  query_embeddings = query_embeddings[:, :truncation_d]
  print(f"after truncation {corpus_embeddings.shape=}", flush=True)
  print(f"after truncation {query_embeddings.shape=}", flush=True)

if quantization_method is not None:
  a = time.time()
  #TODO: extend for mixed precision for queries | corpus
  corpus_embeddings = post_quantize_embeddings(corpus_embeddings, quantization_method,**vars(args))
  query_embeddings = post_quantize_embeddings(query_embeddings, quantization_method, **vars(args))
  print(f"after quantization {corpus_embeddings.shape=}", flush=True)
  print(f"after quantization {query_embeddings.shape=}", flush=True)
  print(f"Time to post-quantize: {(time.time() - a):0.2f} s", flush=True)

# just for debugging/test purposes on a subset
# corpus_embeddings = corpus_embeddings[:40000, :]
# query_embeddings = query_embeddings[:5, :]

dim_index = corpus_embeddings.shape[-1]
index = get_index(dim=dim_index, precision=quantization_method, use_usearch=args.use_usearch, use_flat = args.use_flat, use_ivf = args.use_ivf)
index.train_and_add(corpus_embeddings)
D, I = index.search(query_embeddings, k)

output_file_path = os.path.join(output_folder, output_file) 
output_file_scores_path = os.path.join(output_folder, output_file_scores)
print(f"Saving to: {output_file_path}, {output_file_scores_path}", flush=True)
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
os.makedirs(os.path.dirname(output_file_scores_path), exist_ok=True)
np.save(output_file_path, I)
np.save(output_file_scores_path, D)

if "IVF" in index.default_path: # We only save where there is retraining
    index.save_index(index.default_path)
    print(f"Index has been saved to the path {index.default_path}")
print('------------------------- END OF RUN ----------------------------')