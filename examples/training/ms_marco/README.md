# MS Marco
[MS Marco Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) is a large dataset to train models for information retrieval. It consists of about 500k real search queries from Bing search engine with the relevant text passage that answers the query.

This pages shows how to **train** models (Cross-Encoder and Sentence Embedding Models) on this dataset so that it can be used for searching text passages given queries (key words, phrases or questions).

There are **pre-trained models** available, which you can directly use without the need of training your own models. For more information, see: [Pretrained Models](https://www.sbert.net/docs/pretrained_models.html) | [Pretrained Cross-Encoders](https://www.sbert.net/docs/pretrained_cross-encoders.html)

## Cross-Encoder
A [Cross-Encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) accepts both inputs, the query and the possible relevant passage and returns a score between 0 and 1 how relevant the passage is for the given query.

![CrossEncoder](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/CrossEncoder.png)

Cross-Encoders are often used for **re-ranking:** Given a list with possible relevant passages for a query, for example retrieved from BM25 / ElasticSearch, the cross-encoder re-ranks this list so that the most relevant passages are the top of the result list. See [nboost](https://github.com/koursaros-ai/nboost/) for a proxy for ElasticSearch that applies this strategy.

To **train an cross-encoder** on the MS Marco dataset, see: **[train_cross-encoder.py](train_cross-encoder.py)**.

## Bi-Encoder

The training for bi-encoder, that produces independent embeddings for queries and documents will shortly be added here. Stay tuned.

![BiEncoder](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/BiEncoder.png)
