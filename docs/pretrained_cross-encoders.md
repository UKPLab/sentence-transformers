# Pretrained Cross-Encoders

This page lists available **pretrained Cross-Encoders**. Cross-Encoders require the input of a text pair and output a score 0...1. They do not work for individual sentences and they don't compute embeddings for individual texts.

![BiEncoder](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/Bi_vs_Cross-Encoder.png)


## STSbenchmark
The following models can be used like this:
```
from sentence_transformers import CrossEncoder
model = CrossEncoder('model_name')
scores = model.predict([('Sent A1', 'Sent B1'), ('Sent A2', 'Sent B2')])
```

They return a score  0...1 indicating the semantic similarity of the given sentence pair.
- **sentence-transformers/ce-distilroberta-base-stsb** - STSbenchmark test performance: 88.27
- **sentence-transformers/ce-roberta-base-stsb** - STSbenchmark test performance: 89.85
- **sentence-transformers/ce-roberta-large-stsb** - STSbenchmark test performance: 91.23 

## Quora Duplicate Questions
These models have been trained on the [Quora duplicate questions dataset](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs). They can used like the STSb models and give a score 0...1 indicating the probability that two questions are duplicate questions.

- **sentence-transformers/ce-distilroberta-base-quora** - Average Precision dev set: 87.48
- **sentence-transformers/ce-roberta-base-quora** - Average Precision dev set: 87.80
- **sentence-transformers/ce-roberta-large-quora** - Average Precision dev set: 87.91


## Information Retrieval

The following models are trained for Information Retrieval: Given a query (like key-words or a question), and a paragraph, can the query be answered by the paragraph? The models have beend trained on MS Marco, a large dataset with real-user queries from Bing search engine.

The models can be used like this:
```
from sentence_transformers import CrossEncoder
model = CrossEncoder('model_name', max_length=512)
scores = model.predict([('Query', 'Paragraph1'), ('Query', 'Paragraph2')])
```

This returns a score 0...1 indicating if the paragraph is relevant for a given query.

- **sentence-transformers/ce-ms-marco-TinyBERT-L-2** - MRR@10 on MS Marco Dev Set: 30.15
- **sentence-transformers/ce-ms-marco-TinyBERT-L-4** -  MRR@10 on MS Marco Dev Set: 34.50
- **sentence-transformers/ce-ms-marco-TinyBERT-L-6** - MRR@10 on MS Marco Dev Set: 36.13
- **sentence-transformers/ce-ms-marco-electra-base** - MRR@10 on MS Marco Dev Set: 36.41
