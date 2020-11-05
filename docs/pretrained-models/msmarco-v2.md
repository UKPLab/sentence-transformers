# MSMARCO Models (Version 2)
[MS MARCO](https://microsoft.github.io/msmarco/) is a large scale information retrieval corpus that was created based on real user search queries using Bing search engine. The provided models can be used for semantic search, i.e., given keywords / a search phrase / a question, the model will find passages that are relevant for the search query.

The training data constist of over 500k examples, while the complete  corpus consist of over 8.8 Million passages.
 
## Usage
```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilroberta-base-msmarco-v2')

query_embedding = model.encode('How big is London')
passage_embedding = model.encode('London has 9,787,426 inhabitants at the 2011 census')

print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))
```

## Performance
Performance is evaluated on [TREC-DL 2019](https://microsoft.github.io/TREC-2019-Deep-Learning/), which is a query-passage retrieval task where multiple queries have been annotated as with their relevance with respect to the given query.

As baseline, we use ElasticSearch. We evaluate re-ranking performance. We retrieve the top 100/1000 passages from BM25 and re-rank results with respecte to cosine similarity. We also compare against [Cross-Encoder](https://www.sbert.net/docs/usage/cross-encoder.html), that compute a similarity score for a given query and passage. However, Cross-Encoders scores cannot be precomputed and are hence rather slow.

| Approach       | TREC-DL 2019 (NDCG@10) |   
| ------------- |:-------------: | 
| BM25 (ElasticSearch)   | 45.46 |  |
| BM25 + rerank top 100 with distilroberta-base-msmarco-v2      | 63.51      |   
| BM25 + rerank top 1000 with distilroberta-base-msmarco-v2  | 64.53      | 
| **Cross-Encoders** | |
| BM25 + rerank top 100 with nboost/pt-tinybert-msmarco  | 62.34      |
| BM25 + rerank top 100 with nboost/pt-bert-base-uncased-msmarco | 67.81     |



## Version Histroy 
As we work on the topic, we will publish updated (and improved) models.

- [Version 1](msmarco-v1.md)