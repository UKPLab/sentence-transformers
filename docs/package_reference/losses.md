# Losses
`sentence_transformers.losses` defines different loss functions that can be used to fine-tune embedding models on training data. The choice of loss function plays a critical role when fine-tuning the model. It determines how well our embedding model will work for the specific downstream task.

Sadly, there is no "one size fits all" loss function. Which loss function is suitable depends on the available training data and on the target task. Consider checking out the [Loss Overview](../training/loss_overview.html) to help narrow down your choice of loss function(s).

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

## AnglELoss

```eval_rst
.. autoclass:: sentence_transformers.losses.AnglELoss
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

## GISTEmbedLoss
```eval_rst
.. autoclass:: sentence_transformers.losses.GISTEmbedLoss
```

## MSELoss
```eval_rst
.. autoclass:: sentence_transformers.losses.MSELoss
```

## MarginMSELoss
```eval_rst
.. autoclass:: sentence_transformers.losses.MarginMSELoss
```

## MatryoshkaLoss
```eval_rst
.. autoclass:: sentence_transformers.losses.MatryoshkaLoss
```

## Matryoshka2dLoss
```eval_rst
.. autoclass:: sentence_transformers.losses.Matryoshka2dLoss
```

## AdaptiveLayerLoss
```eval_rst
.. autoclass:: sentence_transformers.losses.AdaptiveLayerLoss
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
