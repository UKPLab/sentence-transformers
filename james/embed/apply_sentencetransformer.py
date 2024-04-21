#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import time

import pandas as pd
import torch
import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer


# In[6]:


query_dev = pd.read_csv("./data/queries/queries.dev.tsv", header=None, sep="\t", index_col=0, names=["query"])
collection = pd.read_csv("./data/collection/collection.tsv", header=None, sep="\t", index_col=0, names=["passage"])


# In[7]:


model = SentenceTransformer("all-MiniLM-L6-v2").cuda()


# In[8]:


def do_embedding(model, series, batch_size=128):
  embeddings = []
  n = len(series)
  start = time.time()
  for i in tqdm.tqdm(range((n + (batch_size-1)) // batch_size)):
    a, b = batch_size*i, min(n, batch_size*(i+1))
    embedding_i = model.encode(series[a:b].tolist())
    embeddings.append(embedding_i)
  end = time.time()
  print(f"Time Taken: {(end-start):0.4f} s")
  return np.vstack(embeddings), end-start


# In[9]:


query_embeddings, query_time = do_embedding(model, query_dev["query"], batch_size=128)
collection_embeddings, collection_time = do_embedding(model, collection["passage"], batch_size=128)
print(f"{query_time=} {collection_time=}")
print(f"{query_embeddings.shape=} {collection_embeddings.shape=}")
print(f"{query_embeddings.dtype=} {collection_embeddings.dtype=}")


# In[10]:


output = "data/results/baseline"
os.makedirs(output, exist_ok=True)
np.save(os.path.join(output, "query_embeddings.npy"), query_embeddings)
np.save(os.path.join(output, "collection_embeddings.npy"), collection_embeddings)

