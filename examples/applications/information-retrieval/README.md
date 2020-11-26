# Information Retrieval





## Cross-Encoder

The query and a possible document is passed simultaneously to a Cross-Encoder, which then outputs a single score between 0 and 1 indicating how relevant the document is for the given query. 

![CrossEncoder](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/CrossEncoder.png)

The advantage of Cross-Encoders is the higher performance, as they perform attention across the query and the document. However, for information retrieval, Crooss-Encoder must be coupled with a retrieval system.

A common setup is to use ElasticSearch and to retrieve e.g. the top 100 or top 1000 hits for a given query. Then, a Cross-Encoder is applied to re-rank the top 100/1000 results.

## Pre-trained Cross-Encoders

Pre-trained models can be used like this:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder('model_name', max_length=512)
scores = model.predict([('Query', 'Paragraph1'), ('Query', 'Paragraph2') , ('Query', 'Paragraph3')])
```

In the following table, we provide various pre-trained Cross-Encoders together with their performance on the [TREC Deep Learning 2019](https://microsoft.github.io/TREC-2019-Deep-Learning/) and the [MS Marco Passage Reranking](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset. 


| Model-Name        | NDCG@10 (TREC DL 19) | MRR@10 (MS Marco Dev)  | Docs / Sec (BertTokenizerFast) | Docs / Sec |
| ------------- |:-------------| -----| --- | --- |
| sentence-transformers/ce-ms-marco-TinyBERT-L-2  | 67.43 | 30.15  | 9000 | 780
| sentence-transformers/ce-ms-marco-TinyBERT-L-4  | 68.09 | 34.50  | 2900 | 760
| sentence-transformers/ce-ms-marco-TinyBERT-L-6 |  69.57 | 36.13  | 680 | 660
| sentence-transformers/ce-ms-marco-electra-base | 71.99 | 36.41 | 340 | 340
| *Other models* | | | |
| nboost/pt-tinybert-msmarco | 63.63 | 28.80 | 2900 | 760
| nboost/pt-bert-base-uncased-msmarco | 70.94 | 34.75 | 340 | 340|
| nboost/pt-bert-large-msmarco | 73.36 | 36.48 | 100 | 100 |
| Capreolus/electra-base-msmarco | 71.23 | 36.89 | 340 | 340 |
| amberoad/bert-multilingual-passage-reranking-msmarco | 68.40 | 35.54 | 330 | 330 
 
 Note: Runtime was computed on a V100 GPU. A bottleneck for smaller models is the standard Python tokenizer from Huggingface. Replacing it with the fast tokenizer based on Rust, the throughput is significantly improved:
 
 ```python
from sentence_transformers import CrossEncoder
import transformers
model = CrossEncoder('model_name', max_length=512)
model.tokenizer = transformers.BertTokenizerFast.from_pretrained('model_name')
``` 