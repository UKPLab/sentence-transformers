# MS MARCO
[MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) is a large dataset to train models for information retrieval. It consists of about 500k real search queries from Bing search engine with the relevant text passage that answers the query.

This pages shows how to **train** models (Cross-Encoder and Sentence Embedding Models) on this dataset so that it can be used for searching text passages given queries (key words, phrases or questions).

If you are interested in how to use these models, see [Application - Retrieve & Re-Rank](../../applications/retrieve_rerank/README.md).

There are **pre-trained models** available, which you can directly use without the need of training your own models. For more information, see: [Pretrained Models](https://www.sbert.net/docs/pretrained_models.html) | [Pretrained Cross-Encoders](https://www.sbert.net/docs/pretrained_cross-encoders.html)



## Bi-Encoder

Cross-Encoder are only suitable for reranking a small set of passages. For retrieval of suitable documents from a large collection, we have to use a bi-encoder. The documents are independently encoded into fixed-sized embeddings. A query is embedded into the same vector space. Relevant documents can then be found by using cosine-similarity.

![BiEncoder](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/BiEncoder.png)

To **train an bi-encoder** on the MS MARCO dataset, see: **[train_bi-encoder.py](train_bi-encoder.py)**.


## Cross-Encoder
A [Cross-Encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) accepts both inputs, the query and the possible relevant passage and returns a score between 0 and 1 how relevant the passage is for the given query.

![CrossEncoder](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/CrossEncoder.png)

Cross-Encoders are often used for **re-ranking:** Given a list with possible relevant passages for a query, for example retrieved from BM25 / ElasticSearch, the cross-encoder re-ranks this list so that the most relevant passages are the top of the result list. 

To **train an cross-encoder** on the MS MARCO dataset, see: 
- **[train_cross-encoder.py](train_cross-encoder.py)** trains a cross-encoder from scratch using the provided data from the MS MARCO dataset.
  
## Cross-Encoder Knowledge Distillation
![](https://github.com/UKPLab/sentence-transformers/raw/master/docs/img/msmarco-training-ce-distillation.png)
- **[train_cross-encoder-v2.py](train_cross-encoder-v2.py)** uses a knowledge distillation setup: [Hostätter et al.](https://arxiv.org/abs/2010.02666) trained an ensemble of 3 (large) models for the MS MARCO dataset and predicted the scores for various (query, passage)-pairs (50% positive, 50% negative). In this example, we use knowledge distillation with a small & fast model and learn the logits scores from the teacher ensemble. This yields performances comparable to  large models, while being 18 times faster.