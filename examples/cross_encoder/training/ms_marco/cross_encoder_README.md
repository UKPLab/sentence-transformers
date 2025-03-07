# MS MARCO
[MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) is a large dataset to train models for information retrieval. It consists of about 500k real search queries from Bing search engine with the relevant text passage that answers the query.

This page shows how to **train** Cross Encoder models on this dataset so that it can be used for searching text passages given queries (key words, phrases or questions).

If you are interested in how to use these models, see [Application - Retrieve & Re-Rank](../../../sentence_transformer/applications/retrieve_rerank/README.md).

There are **pre-trained models** available, which you can directly use without the need of training your own models. For more information, see [Pretrained Cross-Encoders](../../../../docs/cross_encoder/pretrained_models.md#ms-marco).

## Cross-Encoder
A [Cross-Encoder](../../applications/README.md) accepts both inputs, the query and the possible relevant passage and returns a score between 0 and 1 how relevant the passage is for the given query.

![CrossEncoder](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/CrossEncoder.png)

Cross-Encoders are often used for **re-ranking:** Given a list with possible relevant passages for a query, for example retrieved from BM25 / Elasticsearch, the cross-encoder re-ranks this list so that the most relevant passages are the top of the result list. 

To **train an cross-encoder** on the MS MARCO dataset, see: 
- **[train_cross-encoder_scratch.py](train_cross-encoder_scratch.py)** trains a cross-encoder from scratch using the provided data from the MS MARCO dataset.
  
## Cross-Encoder Knowledge Distillation
![](https://github.com/UKPLab/sentence-transformers/raw/master/docs/img/msmarco-training-ce-distillation.png)
- **[train_cross-encoder_kd.py](train_cross-encoder_kd.py)** uses a knowledge distillation setup: [Hostätter et al.](https://arxiv.org/abs/2010.02666) trained an ensemble of 3 (large) models for the MS MARCO dataset and predicted the scores for various (query, passage)-pairs (50% positive, 50% negative). In this example, we use knowledge distillation with a small & fast model and learn the logits scores from the teacher ensemble. This yields performances comparable to  large models, while being 18 times faster.