# Semantic Search
Semantic search seeks to improve search accuracy by understanding the content of the search query. In contrast to traditional search engines, that only finds documents based on lexical matches, semantic search can also find synonyms.


## Background
The idea behind semantic search is to embedd all entries in your corpus, which can be sentences, paragraphs, or documents, into a vector space. 

At search time, the query is embedded into the same vector space and the closest embedding from your corpus are found. These entries should have a high semantic overlap with the query.

![SemanticSearch](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/SemanticSearch.png) 


## Python

For small corpora (up to about 1 million entries) we can compute the cosine-similarity between the query and all entries in the corpus.

In the following example, we define a small corpus with few example sentences and compute the embeddings for the corpus as well as for our query.

We then use the [util.pytorch_cos_sim()](../../../docs/usage/semantic_textual_similarity.md) function to compute the cosine similarity between the query and all corpus entries.

For large corpora, sorting all scores would take too much time. Hence, we use [torch.topk](https://pytorch.org/docs/stable/generated/torch.topk.html) to only get the top k entries.

For a simple example, see [semantic_search.py](semantic_search.py):

```eval_rst
.. literalinclude:: semantic_search.py
```


## util.semantic_search

Instead of implementing semantic search by your self, you can use the *util.semantic_search* function.

The function accepts the following parameters:

```eval_rst
.. autofunction:: sentence_transformers.util.semantic_search
```

By default, up to 100 queries are processes in parallel. Further, the corpus is chunked into set of up to 500k entries. You can increase *query_chunk_size* and *corpus_chunk_size*, which leads to increased speed for large corpora, but also increases the memory requirement.

## Speed Optimization
To get the optimal speed for the `util.semantic_search` method, it is advisable to have the `query_embeddings` as well as the `corpus_embeddings` on the same GPU-device. This significantly boost the performance.

Further, we can normalize the corpus embeddings so that each corpus embeddings is of length 1. In that case, we must pass `corpus_normalized=True` to the `semantic_search` method.
```python
corpus_embeddings = corpus_embeddings.to('cuda')
corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

query_embeddings = query_embeddings.to('cuda')
hits = util.semantic_search(query_embeddings, corpus_embeddings, corpus_normalized=True)
```


## Similar Questions Retrieval
[semantic_search_quora_pytorch.py](semantic_search_quora_pytorch.py) [ [Colab version](https://colab.research.google.com/drive/12cn5Oo0v3HfQQ8Tv6-ukgxXSmT3zl35A?usp=sharing) ] shows an example based on the [Quora duplicate questions](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) dataset. The user can enter a question, and the code retrieves the most similar questions from the dataset using the *util.semantic_search* method. As model, we use *distilbert-multilingual-nli-stsb-quora-ranking*, which was trained to identify similar questions and supports 50+ languages.


## ElasticSearch
Starting with version 7.3, [ElasticSearch](https://www.elastic.co/elasticsearch/) introduced the possibility to index dense vectors and to use to for document scoring. Hence, we can use ElasticSearch to index embeddings along the documents and we can use the query embeddings to retrieve relevant entries.

An advantage of ElasticSearch is that it is easy to add new documents to an index and that we can store also other data along with our vectors. A disadvantage is the slow performance, as it compares the query embeddings with all stored embeddings. This has a linear run-time and might be too slow for large (>100k) corpora.

For further details, see [semantic_search_quora_elasticsearch.py](semantic_search_quora_elasticsearch.py).


## Approximate Nearest Neighbor
Searching a large corpus with millions of embeddings can be time-consuming if exact nearest neighbor search is used (like it is used by *util.semantic_search*).

In that case, Approximate Nearest Neighor (ANN) can be helpful. Here, the data is partitioned into smaller fractions of similar embeddings. This index can be search efficiently and the embeddings with the highest similarity (the nearest neighbors) can be retrieved within milliseconds, even if you have Millions of vectors.

However, the results are not necessarily exact: It can happen, that some vectors with high similarity are missed. That's the reason why it is called approximate nearest neighbor.

For all ANN methods, there is usually one or more parameters to tune that determine the recall - speed trade-off. If you want the highest speed, you have a high chance of missing hits. If you want high recall, the search speed decreases.

Three popular libraries for approximate nearest neighbor are [Annoy](https://github.com/spotify/annoy), [FAISS](https://github.com/facebookresearch/faiss), and [hnswlib](https://github.com/nmslib/hnswlib/). Personally I find hnswlib the most suitable library: It is easy to use, offers a great performance and has nice features included that are important for real applications.

For an example how to use SentenceTransformers with HNSWLib, see: [semantic_search_quora_hnswlib.py](semantic_search_quora_hnswlib.py)

For an example how to use SentenceTransformers with Annoy, see: [semantic_search_quora_annoy.py](semantic_search_quora_annoy.py)

For an example how to use SentenceTransformers with FAISS, see: [semantic_search_quora_faiss.py](semantic_search_quora_faiss.py)
