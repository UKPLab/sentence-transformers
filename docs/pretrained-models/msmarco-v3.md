# MSMARCO Models (Version 3)
[MS MARCO](https://microsoft.github.io/msmarco/) is a large scale information retrieval corpus that was created based on real user search queries using Bing search engine. The provided models can be used for semantic search, i.e., given keywords / a search phrase / a question, the model will find passages that are relevant for the search query.

The training data constist of over 500k examples, while the complete  corpus consist of over 8.8 Million passages.
 
## Usage
```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('msmarco-distilroberta-base-v3')

query_embedding = model.encode('How big is London')
passage_embedding = model.encode('London has 9,787,426 inhabitants at the 2011 census')

print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))
```


For more details on the usage, see [Applications - Information Retrieval](../../examples/applications/retrieve_rerank/README.md)


## Performance
Performance is evaluated on [TREC-DL 2019](https://microsoft.github.io/TREC-2019-Deep-Learning/), which is a query-passage retrieval task where multiple queries have been annotated as with their relevance with respect to the given query.  Further, we evaluate on the [MS Marco Passage Retrieval](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset. 

As baseline we show the results for lexical search with BM25 using ElasticSearch.

| Approach       | NDCG@10 (TREC DL 19 Reranking) | MRR@10 (MS Marco Dev) |  
| ------------- |:-------------: | :---: |
| msmarco-distilbert-base-v3| 69.02 | 33.13 |
| msmarco-roberta-base-v3 | 69.08 | 33.01
| msmarco-roberta-base-ance-fristp | 67.35 | 31.27
| **Previous approaches** |  |  |
| BM25 (ElasticSearch)   | 45.46 | 17.29  |
| msmarco-distilroberta-base-v2   | 65.65 |  28.55    |  
| msmarco-roberta-base-v2 | 67.18 | 29.17 | 
| msmarco-distilbert-base-v2 | 68.35 | 30.77 |

**Notes:**
- **msmarco-roberta-base-ance-fristp** is the MSMARCO Dev Passage Retrieval ANCE(FirstP) 600K model from [ANCE](https://github.com/microsoft/ANCE). This model should be used with dot-product instead of cosine similarity.


## Changes in v3
The models from v2 have been used for find for all training queries similar passages. An [MS MARCO Cross-Encoder](ce-msmarco.md) based on the electra-base-model has been then used to classify if these retrieved passages answer the question.

If they received a low score by the cross-encoder, we saved them as hard negatives: They got a high score from the bi-encoder, but a low-score from the (better) cross-encoder.

We then trained the v2 models with these new hard negatives.

## Version Histroy 
As we work on the topic, we will publish updated (and improved) models.

- [Version 2](msmarco-v2.md)
- [Version 1](msmarco-v1.md)
