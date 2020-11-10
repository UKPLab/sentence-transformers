# Semantic Search
Semantic search seeks to improve search accuracy by understanding the content of the search query. In contrast to traditional search engines, that only finds documents based on lexical matches, semantic search can also find synonyms.


## Python

For small corpora (up to about 100k entries) we can compute the cosine-similarity between the query and all entries in the corpus.

In the following example, we define a small corpus with few example question and compute the embeddings for the corpus as well as for our query.

We then use the [util.pytorch_cos_sim()](semantic_textual_similarity.md) function to compute the cosine similarity between the query and all corpus entries.

For large corpora, sorting all scores would take too much time. Hence, we use [torch.topk](https://pytorch.org/docs/stable/generated/torch.topk.html) to only get the top k entries.

For a simple example, see [examples/semantic_search.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search.py):

```eval_rst
.. literalinclude:: ../../examples/applications/semantic_search.py
```


## util.semantic_search

Instead of implementing semantic search by your self, you can use the *util.semantic_search* function.

The function accepts the following parameters:

```eval_rst
.. autofunction:: sentence_transformers.util.semantic_search
```

By default, up to 100 queries are processes in parallel. Further, the corpus is chunked into set of up to 100k entries. You can increase *query_chunk_size* and *corpus_chunk_size*, which leads to and increased speed for large corpora, but also increases the memory requirement.

Depending on your real-time requirements, you can use this function for corpora up to 1 Million entries given you have enough memory.

## Full-Scale Example
For a full-scale example, see [semantic_search_quora_pytorch.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search_quora_elasticsearch.py)

There, we embed the [Quora duplicate questions](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) dataset, which has around 500k questions asked on Quora. We use *util.semantic_search* to search this corpus.


## ElasticSearch
Starting with version 7.3, [ElasticSearch](https://www.elastic.co/elasticsearch/) introduced the possibility to index dense vectors and to use to for document scoring. Hence, we can use ElasticSearch to index embeddings along the documents and we can use the query embeddings to retrieve relevant entries.

An advantage of ElasticSearch is that it is easy to add new documents to an index and that we can store also other data along with our vectors. A disadvantage is the slow performance, as it compares the query embeddings with all stored embeddings. This has a linear run-time and might be too slow for large (>100k) corpora.

For further details, see [semantic_search_quora_elasticsearch.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search_quora_elasticsearch.py).


## Approximate Nearest Neighbor
Searching a large corpus with millions of embeddings can be time-consuming if exact nearest neighbor search is used (like it is used by *util.semantic_search*).

In that case, Approximate Nearest Neighor (ANN) can be helpful. Here, the data is partitioned into smaller fractions of similar embeddings. This index can be search efficiently and the embeddings with the highest similarity (the nearest neighbors) can be retrieved within milliseconds, even if you have Millions of vectors.

However, the results are not necessarily exact: It can happen, that some vectors with high similarity are missed. That's the reason why it is called approximate nearest neighbor.

For all ANN methods, there is usually one or more parameters to tune that determine the recall - speed trade-off. If you want the highest speed, you have a high chance of missing hits. If you want high recall, the search speed decreases.

Three popular libraries for approximate nearest neighbor are [Annoy](https://github.com/spotify/annoy), [FAISS](https://github.com/facebookresearch/faiss), and [hnswlib](https://github.com/nmslib/hnswlib/). Personally I find hnswlib the most suitable library: It is easy to use, offers a great performance and has nice features included that are important for real applications.

For an example how to use SentenceTransformers with HNSWLib, see: [semantic_search_quora_hnswlib.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search_quora_hnswlib.py)

For an example how to use SentenceTransformers with Annoy, see: [semantic_search_quora_annoy.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search_quora_annoy.py)

For an example how to use SentenceTransformers with FAISS, see: [semantic_search_quora_faiss.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic_search_quora_faiss.py)
