"""
This file contains some utilities functions used to find parallel sentences
in two monolingual corpora.

Code in this file has been adapted from the LASER repository:
https://github.com/facebookresearch/LASER
"""

import faiss
import numpy as np
import time
import gzip
import lzma

########  Functions to find and score candidates
def score(x, y, fwd_mean, bwd_mean, margin):
    return margin(x.dot(y), (fwd_mean + bwd_mean) / 2)


def score_candidates(x, y, candidate_inds, fwd_mean, bwd_mean, margin):
    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = candidate_inds[i, j]
            scores[i, j] = score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin)
    return scores


def kNN(x, y, k, use_ann_search=False, ann_num_clusters=32768, ann_num_cluster_probe=3):
    start_time = time.time()
    if use_ann_search:
        print("Perform approx. kNN search")
        n_cluster = min(ann_num_clusters, int(y.shape[0]/1000))
        quantizer = faiss.IndexFlatIP(y.shape[1])
        index = faiss.IndexIVFFlat(quantizer, y.shape[1], n_cluster, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = ann_num_cluster_probe
        index.train(y)
        index.add(y)
        sim, ind = index.search(x, k)
    else:
        print("Perform exact search")
        idx = faiss.IndexFlatIP(y.shape[1])
        idx.add(y)
        sim, ind = idx.search(x, k)

    print("Done: {:.2f} sec".format(time.time()-start_time))
    return sim, ind


def file_open(filepath):
    #Function to allowing opening files based on file extension
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rt', encoding='utf8')
    elif filepath.endswith('xz'):
        return lzma.open(filepath, 'rt', encoding='utf8')
    else:
        return open(filepath, 'r', encoding='utf8')
