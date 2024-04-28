import os, numpy as np, faiss
import time
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--residing_folder", type=str, required=True)
parser.add_argument("--collection_file", type=str, required=True)
parser.add_argument("--query_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--nlist", type=int, default=10000)
parser.add_argument("--nprobe", type=int, default=200)
parser.add_argument("--d", type=int, default=384)
args = parser.parse_args()
print(vars(args))
sys.stdout.flush()

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
k = 1000

corpus_embeddings = np.load(os.path.join(residing_folder, collection_file))
query_embeddings = np.load(os.path.join(residing_folder, query_file))
print(f"{corpus_embeddings.shape=}")
print(f"{query_embeddings.shape=}")

quantizer = faiss.IndexFlatIP(d)
cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist)
a = time.time()
cpu_index.train(corpus_embeddings)
cpu_index.add(corpus_embeddings)
b = time.time()
print(f"Time to train: {(b-a):0.2f} s")
sys.stdout.flush()

cpu_index.nprobe = nprobe
a = time.time()
D, I = cpu_index.search(query_embeddings, k)
b = time.time()
print(f"Time to search: {(b - a):0.2f} s")
sys.stdout.flush()

np.save(os.path.join(residing_folder, output_file), I)