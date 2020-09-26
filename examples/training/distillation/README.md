# Model Distillation - Faster, cheaper & lighter Model
This folder contains example to make SentenceTransformer models faster, cheaper and lighter.

## Dimension Reduction
By default, the pretrained models output embeddings with size 768 (base-models) or with size 1024 (large-models). However, when you store Millions of embeddings, this can require quite a lot of memory / storage.

[dimensionality_reduction.py](dimensionality_reduction.py) contains a simple example how to reduce the embedding dimension to any size by using Principle Component Analysis (PCA). In that example, we reduce 768 dimension to 128 dimension, reducing the storage requirement by factor 6. The performance only slightly drops from 85.44 to 84.96 on the STS benchmark dataset.

This dimensionality reduction technique can easily be applied to existent models. We could even reduce the embeddings size to 32, reducing the storage requirment by factor 24 (performance decreases to 81.82). 

Note: This technique neither improves the runtime, nor the memory requirement for running the model. It only reduces the needed space to store embeddings, for example, for [semantic search](https://www.sbert.net/docs/usage/semantic_search.html).

