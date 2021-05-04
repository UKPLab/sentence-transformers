# Semantic Search
Semantic search seeks to improve search accuracy by understanding the content of the search query. In contrast to traditional search engines, that only finds documents based on lexical matches, semantic search can also find synonyms.


## Background
The idea behind semantic search is to embedd all entries in your corpus, which can be sentences, paragraphs, or documents, into a vector space. 

At search time, the query is embedded into the same vector space and the closest embedding from your corpus are found. These entries should have a high semantic overlap with the query.

![SemanticSearch](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/SemanticSearch.png) 


## Symmetric vs. Asymmetric Semantic Search

A **critical distinction** for your setup is *symmetric* vs. *asymmetric semantic search*:
- For **symmetric semantic search** your query and the entries in your corpus are of about the same length and have the same amount of content. An example would be searching for similar questions: Your query could for example be *"How to learn Python online?"* and you want to find an entry like *"How to learn Python on the web?"*. For symmetric tasks, you could potentially flip the query and the entries in your corpus.
- For **asymmetric semantic search**, you usually have a **short query** (like a question or some keywords) and you want to find a longer paragraph answering the query. An example would be a query like *"What is Python"* and you wand to find the paragraph *"Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy ..."*. For asymmetric tasks, flipping the query and the entries in your corpus usually does not make sense.

It is critical **that you choose the right model** for your type of task.

Suitable models for **symmetric semantic search**:
- paraphrase-distilroberta-base-v1 / paraphrase-xlm-r-multilingual-v1
- quora-distilbert-base / quora-distilbert-multilingual 
- distiluse-base-multilingual-cased-v2     


Suitable models for **asymmetric semantic search**:
- msmarco-distilbert-base-v2

See [Pretrained Models](../../../docs/pretrained_models.md) for further information.

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

Further, we can normalize the corpus embeddings so that each corpus embeddings is of length 1. In that case, we can use dot-product for computing scores.
```python
corpus_embeddings = corpus_embeddings.to('cuda')
corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

query_embeddings = query_embeddings.to('cuda')
query_embeddings = util.normalize_embeddings(query_embeddings)
hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score)
```




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

Examples:
- [semantic_search_quora_hnswlib.py](semantic_search_quora_hnswlib.py)
- [semantic_search_quora_annoy.py](semantic_search_quora_annoy.py)
- [semantic_search_quora_faiss.py](semantic_search_quora_faiss.py)

## Retrieve & Re-Rank
For complex semantic search scenarios, a retrieve & re-rank pipeline is advisable:
![InformationRetrieval](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/InformationRetrieval.png)

For further details, see [Retrieve & Re-rank](../retrieve_rerank/README.md).

## Examples

In the following we list examples for different use-cases.

### Similar Questions Retrieval
[semantic_search_quora_pytorch.py](semantic_search_quora_pytorch.py) [ [Colab version](https://colab.research.google.com/drive/12cn5Oo0v3HfQQ8Tv6-ukgxXSmT3zl35A?usp=sharing) ] shows an example based on the [Quora duplicate questions](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) dataset. The user can enter a question, and the code retrieves the most similar questions from the dataset using the *util.semantic_search* method. As model, we use *distilbert-multilingual-nli-stsb-quora-ranking*, which was trained to identify similar questions and supports 50+ languages. Hence, the user can input the question in any of the 50+ languages. This is a **symmetric search task**, as the search queries have the same length and content as the questions in the corpus.

### Similar Publication Retrieval
[semantic_search_publications.py](semantic_search_publications.py) [ [Colab version](https://colab.research.google.com/drive/12hfBveGHRsxhPIUMmJYrll2lFU4fOX06?usp=sharing) ] shows an example how to find similar scientific publications. As corpus, we use all publications that have been presented at the EMNLP 2016 - 2018 conferences. As search query, we input the title and abstract of more recent publications and find related publications from our copurs. We use the [SPECTER](https://arxiv.org/abs/2004.07180) model. This is a **symmetric search task**, as the paper in the corpus consists of title & abstract and we search for title & abstract.

### Question & Answer Retrieval
[semantic_search_wikipedia_qa.py](semantic_search_wikipedia_qa.py) [ [Colab Version](https://colab.research.google.com/drive/11GunvCqJuebfeTlgbJWkIMT0xJH6PWF1?usp=sharing) ]: This example uses a model that was trained on the [Natural Questions dataset](https://ai.google.com/research/NaturalQuestions/). It consists of about 100k real Google search queries, together with an annotated passage from Wikipedia that provides the answer. It is an example of an **asymmetric search task**. As corpus, we use the smaller [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page) so that it fits easily into memory.

[retrieve_rerank_simple_wikipedia.py](../retrieve_rerank/retrieve_rerank_simple_wikipedia.py) [ [Colab Version](https://colab.research.google.com/drive/1l6stpYdRMmeDBK_vw0L5NitdiAuhdsAr?usp=sharing) ]: This script uses the [Retrieve & Re-rank](../retrieve_rerank/README.md) strategy and is an example for an **asymmetric search task**. We split all Wikipedia articles into paragraphs and encode them with a bi-encoder. If a new query / question is entered, it is encoded by the same bi-encoder and the paragraphs with the highest cosine-similarity are retrieved (see [semantic search](../semantic-search/README.md)). Next, the retrieved candidates are scored by a Cross-Encoder re-ranker and the 5 passages with the highest score from the Cross-Encoder are presented to the user. We use models that were trained on the [MS Marco Passage Reranking](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset, a dataset with about 500k real queries from Bing search.
