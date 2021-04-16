# MS MARCO Cross-Encoders

[MS MARCO](https://microsoft.github.io/msmarco/) is a large scale information retrieval corpus that was created based on real user search queries using Bing search engine. The provided models can be used for semantic search, i.e., given keywords / a search phrase / a question, the model will find passages that are relevant for the search query.

The training data constist of over 500k examples, while the complete  corpus consist of over 8.8 Million passages.

## Usage with SentenceTransformers
Pre-trained models can be used like this:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder('model_name', max_length=512)
scores = model.predict([('Query', 'Paragraph1'), ('Query', 'Paragraph2') , ('Query', 'Paragraph3')])
```

## Usage with Transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('model_name')
tokenizer = AutoTokenizer.from_pretrained('model_name')

features = tokenizer(['Query', 'Query'], ['Paragraph1', 'Paragraph2'],  padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    print(scores)
```


## Models & Performance

In the following table, we provide various pre-trained Cross-Encoders together with their performance on the [TREC Deep Learning 2019](https://microsoft.github.io/TREC-2019-Deep-Learning/) and the [MS Marco Passage Reranking](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset. 


| Model-Name        | NDCG@10 (TREC DL 19) | MRR@10 (MS Marco Dev)  | Docs / Sec |
| ------------- | :-------------: | :-----: | ---: | 
| **Version 2 models** | | | 
| cross-encoder/ms-marco-TinyBERT-L-2-v2 | 69.84 | 32.56 | 9000
| cross-encoder/ms-marco-MiniLM-L-2-v2 | 71.01 | 34.85 | 4100
| cross-encoder/ms-marco-MiniLM-L-4-v2 | 73.04 | 37.70 | 2500
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 74.30 | 39.01 | 1800
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 74.31 | 39.02 | 960
| **Version 1 models** | | | 
| cross-encoder/ms-marco-TinyBERT-L-2  | 67.43 | 30.15  | 9000 | 
| cross-encoder/ms-marco-TinyBERT-L-4  | 68.09 | 34.50  | 2900 | 
| cross-encoder/ms-marco-TinyBERT-L-6 |  69.57 | 36.13  | 680 | 
| cross-encoder/ms-marco-electra-base | 71.99 | 36.41 | 340 | 
| **Other models** | | | |
| nboost/pt-tinybert-msmarco | 63.63 | 28.80 | 2900 | 
| nboost/pt-bert-base-uncased-msmarco | 70.94 | 34.75 | 340 | 
| nboost/pt-bert-large-msmarco | 73.36 | 36.48 | 100 |  
| Capreolus/electra-base-msmarco | 71.23 | 36.89 | 340 | 
| amberoad/bert-multilingual-passage-reranking-msmarco | 68.40 | 35.54 | 330 |  
| sebastian-hofstaetter/distilbert-cat-margin_mse-T2-msmarco | 72.82 | 37.88 | 720
 
 Note: Runtime was computed on a V100 GPU with Huggingface Transformers v4. 
