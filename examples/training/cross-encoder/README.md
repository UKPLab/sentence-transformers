# CrossEncoder
SentenceTransformers also supports the option to train CrossEncoder for sentence pair score and sentence pair classification tasks.


## BiEncoder vs. CrossEncoder

First, it is important to understand the difference between BiEncoders and CrossEncoder.

SentenceTransformer is centered around BiEncoders, which produces for a given sentence a sentence embedding. We pass to a BERT independelty the sentences A and B, which result in the sentence embeddings u and v. These sentence embedding can then be compared using cosine similarity:
![BiEncoder](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/BiEncoder.png)


In contrast, for a CrossEncoder  we pass both sentences simultanously to BERT. It produces than an output value between 0 and 1 indicating the similarity of sentence pair. 
![CrossEncoder](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/CrossEncoder.png)

A CrossEncoder does not produce a sentence embedding. Also, we are not able to pass individual sentences to a CrossEncoder.

As detailed in our [paper](https://arxiv.org/abs/1908.10084), CrossEncoder achieve better performances than BiEncoders. However, for many application they are not pratical as they do not produce embeddings we could e.g. index or efficiently compare using cosine similarity.


### When to use Cross- / BiEncoders?

CrossEncoders can be used whenever you have a pre-defined set of sentence pairs you want to score. For example, you have 100 sentence pairs and you want to get similarity scores for these 100 pairs.


BiEncoders (see [Computing Sentence Embeddings](https://www.sbert.net/docs/usage/computing_sentence_embeddings.html)) are used whenever you need a sentence embedding in a vector space for efficient comparison. Applications are for example Information Retrieval / Semantic Search or Clustering. CrossEncoders would be the wrong choice for these application: Clustering 10,000 sentence with CrossEncoders would require computing similarity scores for about 50 Million sentence combinations, which takes about 65 hours. With a BiEncoder, you compute the embedding for each sentence, which takes only 5 seconds. You can then perform the clustering.


## Pretrained Cross-Encoders
We provide various pre-trained cross-encoders, that are easy to use:




## Training CrossEncoders

The `CrossEncoder` class is a wrapper around Huggingface `AutoModelForSequenceClassification`, but with some methods to make training and predicting scores a little bit easier. The saved models are 100% compatible with Huggingface and can also be loaded with their classes.

First, you need some sentence pair data. You can either have a continious score, like:
```python
from sentence_transformers import InputExample
train_samples = [
  InputExample(texts=['sentence1', 'sentence2'], label=0.3),
  InputExample(texts=['Another', 'pair'], label=0.8),
]
```

Or you have distinct classes as in the [training_nli.py](training_nli.py) example:
```python
from sentence_transformers import InputExample
label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = [
  InputExample(texts=['sentence1', 'sentence2'], label=label2int['neutral']),
  InputExample(texts=['Another', 'pair'], label=label2int['entailment']),
]
```

Then, you define the base model and the number of labels. You can take any [Huggingface pre-trained model](https://huggingface.co/transformers/pretrained_models.html) that is compatible with AutoModel:
```
model = CrossEncoder('distilroberta-base', num_labels=1)
```

For binary tasks and tasks with continious scores (like STS), we set num_labels=1. For classification tasks, we set it to the number of labels we have.

We start the training by calling `model.fit()`:
```python
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
```


## Combining Cross- and BiEncoders
Combining Cross- and BiEncoders can make sense in Information Retrieval / Semantic Search scenarios: First, you use an efficient BiEncoder to retrieve e.g. the top-100 most similar sentences for a query. Then, you use a CrossEncoder to re-rank these 100 hits by computing the score for every (query, hit) combination.

