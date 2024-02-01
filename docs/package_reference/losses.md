# Losses
`sentence_transformers.losses` define different loss functions, that can be used to fine-tune the network on training data. The loss function plays a critical role when fine-tuning the model. It determines how well our embedding model will work for the specific downstream task.

Sadly, there is no "one size fits all" loss function. Which loss function is suitable depends on the available training data and on the target task.
Feel free to consider the following tables to help narrow down your choice of loss function(s).

| Texts                                         | Labels                                                        | Appropriate Loss Functions                                                                                                                                                                                                                                                                                                       |
|-----------------------------------------------|---------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `single sentences`                            | `class`                                                       | <a href="#batchalltripletloss">`BatchAllTripletLoss`</a><br><a href="#batchhardsoftmargintripletloss">`BatchHardSoftMarginTripletLoss`</a><br><a href="#batchhardtripletloss">`BatchHardTripletLoss`</a><br><a href="#batchsemihardtripletloss">`BatchSemiHardTripletLoss`</a>                                                   |
| `single sentences`                            | `none`                                                        | <a href="#contrastivetensionloss">`ContrastiveTensionLoss`</a><br><a href="#denoisingautoencoderloss">`DenoisingAutoEncoderLoss`</a>                                                                                                                                                                                             |
| `(anchor, anchor) pairs`                      | `none`                                                        | <a href="#contrastivetensionlossinbatchnegatives">`ContrastiveTensionLossInBatchNegatives`</a>                                                                                                                                                                                                                                   |
| `(damaged_sentence, original_sentence) pairs` | `none`                                                        | <a href="#denoisingautoencoderloss">`DenoisingAutoEncoderLoss`</a>                                                                                                                                                                                                                                                               |
| `(sentence_A, sentence_B) pairs`              | `class`                                                       | <a href="#softmaxloss">`SoftmaxLoss`</a>                                                                                                                                                                                                                                                                                         |
| `(anchor, positive) pairs`                    | `none`                                                        | <a href="#cachedmultiplenegativesrankingloss">`CachedMultipleNegativesRankingLoss`</a><br><a href="#multiplenegativesrankingloss">`MultipleNegativesRankingLoss`</a><br><a href="#multiplenegativessymmetricrankingloss">`MultipleNegativesSymmetricRankingLoss`</a><br><a href="#megabatchmarginloss">`MegaBatchMarginLoss`</a> |
| `(anchor, positive/negative) pairs`           | `1 if positive, 0 if negative`                                | <a href="#contrastiveloss">`ContrastiveLoss`</a><br><a href="#onlinecontrastiveloss">`OnlineContrastiveLoss`</a>                                                                                                                                                                                                                 |
| `(sentence_A, sentence_B) pairs`              | `float similarity score`                                      | <a href="#cosentloss">`CoSENTLoss`</a><br><a href="#cosinesimilarityloss">`CosineSimilarityLoss`</a>                                                                                                                                                                                                                             |
| `(anchor, positive, negative) triplets`       | `none`                                                        | <a href="#cachedmultiplenegativesrankingloss">`CachedMultipleNegativesRankingLoss`</a><br><a href="#multiplenegativesrankingloss">`MultipleNegativesRankingLoss`</a><br><a href="#tripletloss">`TripletLoss`</a>                                                                                                                 |

Note that you can often convert one training data format into another, allowing more loss functions to be viable for your case. For example, (sentence_A, sentence_B) pairs with classes can be converted into (anchor, positive, negative) by sampling sentences with the same or different classes.

<b>Distillation</b><br>
These loss functions are specifically designed to be used when distilling the knowledge from one model into another.
For example, when finetuning a small model to behave more like a larger & stronger one, or when finetuning a model to become multi-lingual.

| Texts                                         | Labels                                                        | Appropriate Loss Functions                                                                                                                                                                                                                                                                                                       |
|-----------------------------------------------|---------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `single sentences`                            | `model sentence embeddings`                                   | <a href="#mseloss">`MSELoss`</a>                                                                                                                                                                                                                                                                                                 |
| `(query, passage_one, passage_two) triplets`  | `gold_sim(query, passage_one) - gold_sim(query, passage_two)` | <a href="#marginmseloss">`MarginMSELoss`</a>                                                                                                                                                                                                                                                                                     |

## BatchAllTripletLoss
```eval_rst
.. autoclass:: sentence_transformers.losses.BatchAllTripletLoss
```

## BatchHardSoftMarginTripletLoss
```eval_rst
.. autoclass:: sentence_transformers.losses.BatchHardSoftMarginTripletLoss
```

## BatchHardTripletLoss
```eval_rst
.. autoclass:: sentence_transformers.losses.BatchHardTripletLoss
```

## BatchSemiHardTripletLoss
```eval_rst
.. autoclass:: sentence_transformers.losses.BatchSemiHardTripletLoss
```

## ContrastiveLoss
```eval_rst
.. autoclass:: sentence_transformers.losses.ContrastiveLoss
```

## OnlineContrastiveLoss
```eval_rst
.. autoclass:: sentence_transformers.losses.OnlineContrastiveLoss
```

## ContrastiveTensionLoss
```eval_rst
.. autoclass:: sentence_transformers.losses.ContrastiveTensionLoss
```

## ContrastiveTensionLossInBatchNegatives
```eval_rst
.. autoclass:: sentence_transformers.losses.ContrastiveTensionLossInBatchNegatives
```

## CoSENTLoss

```eval_rst
.. autoclass:: sentence_transformers.losses.CoSENTLoss
```

## CosineSimilarityLoss

![SBERT Siamese Network Architecture](../img/SBERT_Siamese_Network.png "SBERT Siamese Architecture")


For each sentence pair, we pass sentence A and sentence B through our network which yields the embeddings *u* und *v*. The similarity of these embeddings is computed using cosine similarity and the result is compared to the gold similarity score. 

This allows our network to be fine-tuned to recognize the similarity of sentences.


```eval_rst
.. autoclass:: sentence_transformers.losses.CosineSimilarityLoss
```

## DenoisingAutoEncoderLoss

```eval_rst
.. autoclass:: sentence_transformers.losses.DenoisingAutoEncoderLoss
```

## MSELoss
```eval_rst
.. autoclass:: sentence_transformers.losses.MSELoss
```

## MarginMSELoss
```eval_rst
.. autoclass:: sentence_transformers.losses.MarginMSELoss
```

## MegaBatchMarginLoss

```eval_rst
.. autoclass:: sentence_transformers.losses.MegaBatchMarginLoss
```

## MultipleNegativesRankingLoss

*MultipleNegativesRankingLoss* is a great loss function if you only have positive pairs, for example, only pairs of similar texts like pairs of paraphrases, pairs of duplicate questions, pairs of (query, response), or pairs of (source_language, target_language).

```eval_rst
.. autoclass:: sentence_transformers.losses.MultipleNegativesRankingLoss
```

## CachedMultipleNegativesRankingLoss

```eval_rst
.. autoclass:: sentence_transformers.losses.CachedMultipleNegativesRankingLoss
```

## MultipleNegativesSymmetricRankingLoss

```eval_rst
.. autoclass:: sentence_transformers.losses.MultipleNegativesSymmetricRankingLoss
```

## SoftmaxLoss
```eval_rst
.. autoclass:: sentence_transformers.losses.SoftmaxLoss
```

## TripletLoss
```eval_rst
.. autoclass:: sentence_transformers.losses.TripletLoss
```
