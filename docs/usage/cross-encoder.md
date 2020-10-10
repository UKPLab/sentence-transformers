#Cross-Encoder
SentenceTransformers also supports to load Cross-Encoders for sentence pair scoring and sentence pair classification tasks.


## Bi-Encoder vs. Cross-Encoder

First, it is important to understand the difference between Bi- and Cross-Encoder.

SentenceTransformer is centered around Bi-Encoders, which produces for a given sentence a sentence embedding. We pass to a BERT independelty the sentences A and B, which result in the sentence embeddings u and v. These sentence embedding can then be compared using cosine similarity:
![BiEncoder](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/BiEncoder.png)


In contrast, for a Cross-Encoder,  we pass both sentences simultanously to the Transformer network. It produces than an output value between 0 and 1 indicating the similarity of the input sentence pair. 
![CrossEncoder](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/CrossEncoder.png)

A Cross-Encoder does not produce a sentence embedding. Also, we are not able to pass individual sentences to a Cross-Encoder.

As detailed in our [paper](https://arxiv.org/abs/1908.10084), Cross-Encoder achieve better performances than Bi-Encoders. However, for many application they are not pratical as they do not produce embeddings we could e.g. index or efficiently compare using cosine similarity.


### When to use Cross- / Bi-Encoders?

Cross-Encoders can be used whenever you have a pre-defined set of sentence pairs you want to score. For example, you have 100 sentence pairs and you want to get similarity scores for these 100 pairs.


Bi-Encoders (see [Computing Sentence Embeddings](https://www.sbert.net/docs/usage/computing_sentence_embeddings.html)) are used whenever you need a sentence embedding in a vector space for efficient comparison. Applications are for example Information Retrieval / Semantic Search or Clustering. Cross-Encoders would be the wrong choice for these application: Clustering 10,000 sentence with CrossEncoders would require computing similarity scores for about 50 Million sentence combinations, which takes about 65 hours. With a Bi-Encoder, you compute the embedding for each sentence, which takes only 5 seconds. You can then perform the clustering.


## Cross-Encoders Usage
We provide various pre-trained cross-encoders, that are easy to use [cross-encoder_usage.py](examples/applications/cross-encoder_usage.py):
```eval_rst
.. literalinclude:: ../../examples/applications/cross-encoder_usage.py
```

## Pretrained Cross-Encoders
We provide the following pre-trained Cross-Encoders. They can by loaded via:
```
model = CrossEncoder('model_name')
```

## Combining Cross- and Bi-Encoders
Combining Cross- and Bi-Encoders can make sense in Information Retrieval / Semantic Search scenarios: First, you use an efficient Bi-Encoder to retrieve e.g. the top-100 most similar sentences for a query. Then, you use a Cross-Encoder to re-rank these 100 hits by computing the score for every (query, hit) combination.

## Training Cross-Encoders 
See [Cross-Encoder Training](../../examples/training/cross-encoder/README.md) how to train your own Cross-Encoder models.