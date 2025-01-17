# Semantic Search
Semantic search seeks to improve search accuracy by understanding the semantic meaning of the search query and the corpus to search over. Semantic search can also perform well given synonyms, abbreviations, and misspellings, unlike keyword search engines that can only find documents based on lexical matches.

## Background
The idea behind semantic search is to embed all entries in your corpus, whether they be sentences, paragraphs, or documents, into a vector space. At search time, the query is embedded into the same vector space and the closest embeddings from your corpus are found. These entries should have a high semantic similarity with the query.

![SemanticSearch](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/SemanticSearch.png) 

## Symmetric vs. Asymmetric Semantic Search

A **critical distinction** for your setup is *symmetric* vs. *asymmetric semantic search*:
- For **symmetric semantic search** your query and the entries in your corpus are of about the same length and have the same amount of content. An example would be searching for similar questions: Your query could for example be *"How to learn Python online?"* and you want to find an entry like *"How to learn Python on the web?"*. For symmetric tasks, you could potentially flip the query and the entries in your corpus. 
    - Related training example: [Quora Duplicate Questions](../../training/quora_duplicate_questions/README.md).
    - Suitable models: [Pre-Trained Sentence Embedding Models](../../../docs/sentence_transformer/pretrained_models.md)
- For **asymmetric semantic search**, you usually have a **short query** (like a question or some keywords) and you want to find a longer paragraph answering the query. An example would be a query like *"What is Python"* and you want to find the paragraph *"Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy ..."*. For asymmetric tasks, flipping the query and the entries in your corpus usually does not make sense.
    - Related training example: [MS MARCO](../../training/ms_marco/README.md)
    - Suitable models: [Pre-Trained MS MARCO Models](../../../docs/pretrained-models/msmarco-v5.md)

It is critical **that you choose the right model** for your type of task.

## Manual Implementation

```{eval-rst}
For small corpora (up to about 1 million entries), we can perform semantic search with a manual implementation by computing the embeddings for the corpus as well as for our query, and then calculating the `semantic textual similarity <../../../docs/sentence_transformer/usage/semantic_textual_similarity.html>`_ using :func:`SentenceTransformer.similarity <sentence_transformers.SentenceTransformer.similarity>`.
```
For a simple example, see [semantic_search.py](semantic_search.py):

```{eval-rst}

.. sidebar:: Output

   .. code-block:: text

        Query: A man is eating pasta.
        Top 5 most similar sentences in corpus:
        A man is eating food. (Score: 0.7035)
        A man is eating a piece of bread. (Score: 0.5272)
        A man is riding a horse. (Score: 0.1889)
        A man is riding a white horse on an enclosed ground. (Score: 0.1047)
        A cheetah is running behind its prey. (Score: 0.0980)

        Query: Someone in a gorilla costume is playing a set of drums.
        Top 5 most similar sentences in corpus:
        A monkey is playing drums. (Score: 0.6433)
        A woman is playing violin. (Score: 0.2564)
        A man is riding a horse. (Score: 0.1389)
        A man is riding a white horse on an enclosed ground. (Score: 0.1191)
        A cheetah is running behind its prey. (Score: 0.1080)

        Query: A cheetah chases prey on across a field.
        Top 5 most similar sentences in corpus:
        A cheetah is running behind its prey. (Score: 0.8253)
        A man is eating food. (Score: 0.1399)
        A monkey is playing drums. (Score: 0.1292)
        A man is riding a white horse on an enclosed ground. (Score: 0.1097)
        A man is riding a horse. (Score: 0.0650)

.. literalinclude:: semantic_search.py
```


## Optimized Implementation

```{eval-rst}
Instead of implementing semantic search by yourself, you can use the :func:`util.semantic_search <sentence_transformers.util.semantic_search>` function.
```

The function accepts the following parameters:

```{eval-rst}
.. autofunction:: sentence_transformers.util.semantic_search
```

By default, up to 100 queries are processed in parallel. Further, the corpus is chunked into set of up to 500k entries. You can increase ``query_chunk_size`` and ``corpus_chunk_size``, which leads to increased speed for large corpora, but also increases the memory requirement.

## Speed Optimization
```{eval-rst}
To get the optimal speed for the :func:`util.semantic_search <sentence_transformers.util.semantic_search>` method, it is advisable to have the ``query_embeddings`` as well as the ``corpus_embeddings`` on the same GPU-device. This significantly boost the performance. Further, we can normalize the corpus embeddings so that each corpus embeddings is of length 1. In that case, we can use dot-product for computing scores.

.. code-block:: python

    corpus_embeddings = corpus_embeddings.to("cuda")
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    query_embeddings = query_embeddings.to("cuda")
    query_embeddings = util.normalize_embeddings(query_embeddings)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score)
```

## Elasticsearch
[Elasticsearch](https://www.elastic.co/elasticsearch/) has the possibility to [index dense vectors](https://www.elastic.co/what-is/vector-search) and to use them for document scoring. We can easily index embedding vectors, store other data alongside our vectors and, most importantly, efficiently retrieve relevant entries using [approximate nearest neighbor search](https://www.elastic.co/blog/introducing-approximate-nearest-neighbor-search-in-elasticsearch-8-0) (HNSW, see also below) on the embeddings.

For further details, see [semantic_search_quora_elasticsearch.py](semantic_search_quora_elasticsearch.py).


## Approximate Nearest Neighbor
```{eval-rst}
Searching a large corpus with millions of embeddings can be time-consuming if exact nearest neighbor search is used (like it is used by :func:`util.semantic_search <sentence_transformers.util.semantic_search>`).
```

In that case, Approximate Nearest Neighbor (ANN) can be helpful. Here, the data is partitioned into smaller fractions of similar embeddings. This index can be searched efficiently and the embeddings with the highest similarity (the nearest neighbors) can be retrieved within milliseconds, even if you have millions of vectors. However, the results are not necessarily exact. It is possible that some vectors with high similarity will be missed.

For all ANN methods, there are usually one or more parameters to tune that determine the recall-speed trade-off. If you want the highest speed, you have a high chance of missing hits. If you want high recall, the search speed decreases.

Three popular libraries for approximate nearest neighbor are [Annoy](https://github.com/spotify/annoy), [FAISS](https://github.com/facebookresearch/faiss), and [hnswlib](https://github.com/nmslib/hnswlib/).

Examples:

- [semantic_search_quora_hnswlib.py](semantic_search_quora_hnswlib.py)
- [semantic_search_quora_annoy.py](semantic_search_quora_annoy.py)
- [semantic_search_quora_faiss.py](semantic_search_quora_faiss.py)

## Retrieve & Re-Rank
For complex semantic search scenarios, a two-stage retrieve & re-rank pipeline is advisable:
![InformationRetrieval](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/InformationRetrieval.png)

For further details, see [Retrieve & Re-rank](../retrieve_rerank/README.md).

## Examples

We list a handful of common use cases:

### Similar Questions Retrieval
[semantic_search_quora_pytorch.py](semantic_search_quora_pytorch.py) [ [Colab version](https://colab.research.google.com/drive/12cn5Oo0v3HfQQ8Tv6-ukgxXSmT3zl35A?usp=sharing) ] shows an example based on the [Quora duplicate questions](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) dataset. The user can enter a question, and the code retrieves the most similar questions from the dataset using `util.semantic_search`. As model, we use [distilbert-multilingual-nli-stsb-quora-ranking](https://huggingface.co/sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking), which was trained to identify similar questions and supports 50+ languages. Hence, the user can input the question in any of the 50+ languages. This is a **symmetric search task**, as the search queries have the same length and content as the questions in the corpus.

### Similar Publication Retrieval
[semantic_search_publications.py](semantic_search_publications.py) [ [Colab version](https://colab.research.google.com/drive/12hfBveGHRsxhPIUMmJYrll2lFU4fOX06?usp=sharing) ] shows an example how to find similar scientific publications. As corpus, we use all publications that have been presented at the EMNLP 2016 - 2018 conferences. As search query, we input the title and abstract of more recent publications and find related publications from our corpus. We use the [SPECTER](https://huggingface.co/sentence-transformers/allenai-specter) model. This is a **symmetric search task**, as the paper in the corpus consists of title & abstract and we search for title & abstract.

### Question & Answer Retrieval
[semantic_search_wikipedia_qa.py](semantic_search_wikipedia_qa.py) [ [Colab Version](https://colab.research.google.com/drive/11GunvCqJuebfeTlgbJWkIMT0xJH6PWF1?usp=sharing) ]: This example uses a model that was trained on the [Natural Questions dataset](https://huggingface.co/datasets/sentence-transformers/natural-questions). It consists of about 100k real Google search queries, together with an annotated passage from Wikipedia that provides the answer. It is an example of an **asymmetric search task**. As corpus, we use the smaller [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page) so that it fits easily into memory.

[retrieve_rerank_simple_wikipedia.ipynb](../retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb) [ [Colab Version](https://colab.research.google.com/github/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb) ]: This script uses the [Retrieve & Re-rank](../retrieve_rerank/README.md) strategy and is an example for an **asymmetric search task**. We split all Wikipedia articles into paragraphs and encode them with a bi-encoder. If a new query / question is entered, it is encoded by the same bi-encoder and the paragraphs with the highest cosine-similarity are retrieved. Next, the retrieved candidates are scored by a Cross-Encoder re-ranker and the 5 passages with the highest score from the Cross-Encoder are presented to the user. We use models that were trained on the [MS Marco Passage Reranking](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset, a dataset with about 500k real queries from Bing search.
